//! Container format for MiniCPM-V 4.7 1B (`MCV47\0\0\x01`), its Vision Tower (`MCVIS\0\0\x01`),
//! and its `DSpark` Drafter (`MCDSP\0\0\x01`).
//!
//! Layout:
//! - `0..8`: 8-byte magic
//! - `8..16`: `u64` little-endian JSON header length `H`
//! - `16..16+H`: JSON header (`format`, `version`, `config`, `tensors`, `payload_length`, `payload_sha256`,
//!   optional `compression` + zstd frame table)
//! - From the next 16-KiB boundary: the payload, either raw (mapped zero-copy into Metal) or
//!   `zstd_col_delta` frames that `MappedArtifact::open` decodes on every core into anonymous memory.
//!
//! Supported tensor dtypes:
//! - `int4_g64`: packed 4-bit nibbles `[rows, cols / 2]` (`codes`) + `[rows, cols / 64]` `half2` `(scale, bias)` (`groups`)
//! - `ptq1`: ternary g128 `PTQ1_0` 28-byte blocks `[rows, (cols / 128) * 28]`
//! - `int8_row`: per-row int8 `[rows, cols]` + `[rows]` `f32` row scales
//! - `bf16`: raw BF16 `[rows, cols]`
//! - `f32`: 32-bit float vectors/weights

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read as _, Write as _};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use memmap2::{Mmap, MmapMut};
use micro_metal::buffer::MetalBuffer;
use micro_metal::context::MetalContext;
use micro_metal::projection::{AnyMatrix, Bf16Matrix, Int4Matrix, Int8Matrix, Ptq1Matrix};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::Error;

pub const METAL_PAGE: usize = 16 * 1024;
pub const LM_MAGIC: [u8; 8] = *b"MCV47\0\0\x01";
pub const VIS_MAGIC: [u8; 8] = *b"MCVIS\0\0\x01";
pub const DSP_MAGIC: [u8; 8] = *b"MCDSP\0\0\x01";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDtype {
    Int4G64,
    Ptq1,
    Int8Row,
    Bf16,
    F32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub dtype: TensorDtype,
    pub shape: [usize; 2],
    /// Byte offset relative to payload start.
    pub offset: usize,
    /// Primary byte length (`codes` for `Int4G64`, `weights` for `Int8Row`, or whole tensor).
    pub length: usize,
    /// Secondary byte offset relative to payload start (`groups` for `Int4G64`, `scales` for `Int8Row`).
    #[serde(default)]
    pub aux_offset: usize,
    #[serde(default)]
    pub aux_length: usize,
    /// Whether this matrix lives in the 512-block signed-Hadamard rotated basis.
    #[serde(default)]
    pub rotated: bool,
    /// Optional base tensor name for lossless byte-delta encoding (`tensor - base mod 256`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_from: Option<String>,
}

/// A matrix tensor resolved by name once (see `MappedArtifact::matrix_handle`).
#[derive(Clone, Copy, Debug)]
pub struct MatrixHandle {
    format: MatrixFormat,
    offset: usize,
    aux_offset: usize,
    rows: u32,
    columns: u32,
    rotated: bool,
}

#[derive(Clone, Copy, Debug)]
enum MatrixFormat {
    Int4,
    Ptq1,
    Int8,
    Bf16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactHeader {
    pub format: String,
    pub version: u32,
    pub config: serde_json::Value,
    pub tensors: BTreeMap<String, TensorDescriptor>,
    pub payload_length: u64,
    pub payload_sha256: String,
    /// Payload compression: `"zstd"`, `"zstd_shuffle4"` or `"zstd_col_delta"` (`None` = raw).
    #[serde(default)]
    pub compression: Option<String>,
    /// Byte length of the optional zstd dictionary stored at `payload_start` (legacy artifacts).
    #[serde(default)]
    pub zstd_dict_length: usize,
    /// Uncompressed bytes per zstd frame (the last may be shorter); `0` = one frame (legacy).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub zstd_frame_bytes: u64,
    /// Compressed byte length of each zstd frame, in payload order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub zstd_frames: Vec<u64>,
}

#[expect(clippy::trivially_copy_pass_by_ref, reason = "serde `skip_serializing_if` signature")]
const fn is_zero(value: &u64) -> bool {
    *value == 0
}

/// Uncompressed bytes per independently compressed zstd frame (decoded in parallel at load).
pub const ZSTD_FRAME_BYTES: usize = 8 << 20;

/// Run `job(i)` for every `i in 0..count` on all cores.
fn par_for(count: usize, job: impl Fn(usize) + Sync) {
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map_or(8, usize::from)
        .min(count);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= count {
                        break;
                    }
                    job(i);
                }
            });
        }
    });
}

/// Run `work` once per job on all cores (each job goes to exactly one worker).
fn par_each<T: Send>(jobs: Vec<T>, work: impl Fn(T) + Sync) {
    let slots: Vec<Mutex<Option<T>>> = jobs.into_iter().map(|job| Mutex::new(Some(job))).collect();
    par_for(slots.len(), |i| {
        let job = slots[i].lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(job) = job {
            work(job);
        }
    });
}

#[derive(Clone)]
pub struct MappedArtifact {
    map: Arc<Mmap>,
    pub header: ArtifactHeader,
    pub payload_buffer: MetalBuffer,
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(64), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

/// Transpose 4-byte streams `(b0, b1, b2, b3)` -> `[all b0, all b1, all b2, all b3]` for Zstd.
#[must_use]
#[cfg_attr(target_arch = "aarch64", expect(unsafe_code))]
pub fn byte_shuffle4_encode(input: &[u8]) -> Vec<u8> {
    let mut out = vec![0_u8; input.len()];
    let n4 = input.len() / 4;
    let rem = input.len() % 4;
    let (p0, rest) = out.split_at_mut(n4);
    let (p1, rest) = rest.split_at_mut(n4);
    let (p2, rest) = rest.split_at_mut(n4);
    let (p3, tail) = rest.split_at_mut(n4);
    let mut idx = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{vld4q_u8, vst1q_u8};
        while idx + 16 <= n4 {
            unsafe {
                let quad = vld4q_u8(input.as_ptr().add(idx * 4));
                vst1q_u8(p0.as_mut_ptr().add(idx), quad.0);
                vst1q_u8(p1.as_mut_ptr().add(idx), quad.1);
                vst1q_u8(p2.as_mut_ptr().add(idx), quad.2);
                vst1q_u8(p3.as_mut_ptr().add(idx), quad.3);
            }
            idx += 16;
        }
    }
    for j in idx..n4 {
        p0[j] = input[j * 4];
        p1[j] = input[j * 4 + 1];
        p2[j] = input[j * 4 + 2];
        p3[j] = input[j * 4 + 3];
    }
    if rem > 0 {
        tail.copy_from_slice(&input[n4 * 4..]);
    }
    out
}

/// Inverse of `byte_shuffle4_encode`: re-interleave 4 byte planes into `output` using NEON `ST4` / `u8x16` SIMD.
fn byte_shuffle4_decode_chunk(
    p0: &[u8],
    p1: &[u8],
    p2: &[u8],
    p3: &[u8],
    output: &mut [u8],
) {
    let n4 = p0.len();
    let mut idx = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{uint8x16x4_t, vld1q_u8, vst4q_u8};
        while idx + 16 <= n4 {
            #[allow(unsafe_code)]
            unsafe {
                let v0 = vld1q_u8(p0.as_ptr().add(idx));
                let v1 = vld1q_u8(p1.as_ptr().add(idx));
                let v2 = vld1q_u8(p2.as_ptr().add(idx));
                let v3 = vld1q_u8(p3.as_ptr().add(idx));
                vst4q_u8(
                    output.as_mut_ptr().add(idx * 4),
                    uint8x16x4_t(v0, v1, v2, v3),
                );
            }
            idx += 16;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        use std::simd::u8x16;
        while idx + 16 <= n4 {
            let v0 = u8x16::from_slice(&p0[idx..idx + 16]);
            let v1 = u8x16::from_slice(&p1[idx..idx + 16]);
            let v2 = u8x16::from_slice(&p2[idx..idx + 16]);
            let v3 = u8x16::from_slice(&p3[idx..idx + 16]);
            let (ev_lo, ev_hi) = v0.interleave(v2);
            let (od_lo, od_hi) = v1.interleave(v3);
            let (q0, q1) = ev_lo.interleave(od_lo);
            let (q2, q3) = ev_hi.interleave(od_hi);
            let dst = &mut output[idx * 4..(idx + 16) * 4];
            q0.copy_to_slice(&mut dst[0..16]);
            q1.copy_to_slice(&mut dst[16..32]);
            q2.copy_to_slice(&mut dst[32..48]);
            q3.copy_to_slice(&mut dst[48..64]);
            idx += 16;
        }
    }
    for j in idx..n4 {
        output[j * 4] = p0[j];
        output[j * 4 + 1] = p1[j];
        output[j * 4 + 2] = p2[j];
        output[j * 4 + 3] = p3[j];
    }
}

/// Inverse of `byte_shuffle4_encode` on one thread (for callers that already parallelize).
fn byte_shuffle4_decode_serial(planes: &[u8], output: &mut [u8]) {
    let n4 = output.len() / 4;
    let (p0, rest) = planes.split_at(n4);
    let (p1, rest) = rest.split_at(n4);
    let (p2, rest) = rest.split_at(n4);
    let (p3, tail) = rest.split_at(n4);
    byte_shuffle4_decode_chunk(p0, p1, p2, p3, &mut output[..n4 * 4]);
    output[n4 * 4..].copy_from_slice(tail);
}

/// Inverse of `byte_shuffle4_encode`: re-interleave 4 byte planes into `output` using multi-core NEON `ST4` / `u8x16` SIMD.
pub fn byte_shuffle4_decode(planes: &[u8], output: &mut [u8]) {
    let n4 = output.len() / 4;
    let threads = std::thread::available_parallelism()
        .map_or(8, usize::from)
        .min(16);
    if n4 < 262_144 || threads == 1 {
        byte_shuffle4_decode_serial(planes, output);
        return;
    }
    let (p0, rest) = planes.split_at(n4);
    let (p1, rest) = rest.split_at(n4);
    let (p2, rest) = rest.split_at(n4);
    let (p3, tail) = rest.split_at(n4);
    let chunk_n4 = (n4 / threads).div_ceil(16) * 16;
    let (body, rem_out) = output.split_at_mut(n4 * 4);
    std::thread::scope(|s| {
        for (i, out_chunk) in body.chunks_mut(chunk_n4 * 4).enumerate() {
            let start = i * chunk_n4;
            let len = out_chunk.len() / 4;
            let (s0, s1, s2, s3) = (
                &p0[start..start + len],
                &p1[start..start + len],
                &p2[start..start + len],
                &p3[start..start + len],
            );
            s.spawn(move || {
                byte_shuffle4_decode_chunk(s0, s1, s2, s3, out_chunk);
            });
        }
    });
    rem_out.copy_from_slice(tail);
}

/// Transpose a 2D byte matrix `[rows, cols]` -> `[cols, rows]` in parallel across CPU cores when large.
#[must_use]
pub fn transpose_2d_bytes(input: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    if rows <= 1 || cols <= 1 || input.len() != rows * cols {
        return input.to_vec();
    }
    let mut out = vec![0_u8; input.len()];
    let threads = std::thread::available_parallelism()
        .map_or(8, usize::from)
        .min(16);
    if input.len() >= 1_048_576 && cols >= threads && threads > 1 {
        let cols_per_chunk = cols.div_ceil(threads);
        std::thread::scope(|s| {
            for (chunk_idx, out_chunk) in out.chunks_mut(cols_per_chunk * rows).enumerate() {
                let c_start = chunk_idx * cols_per_chunk;
                let c_count = out_chunk.len() / rows;
                s.spawn(move || {
                    for r in 0..rows {
                        let src_row = &input[r * cols + c_start..r * cols + c_start + c_count];
                        for (dc, &b) in src_row.iter().enumerate() {
                            out_chunk[dc * rows + r] = b;
                        }
                    }
                });
            }
        });
        return out;
    }
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        for (c, &b) in row.iter().enumerate() {
            out[c * rows + r] = b;
        }
    }
    out
}

/// Apply in-place lossless `col_delta` pre-transform before global `byte_shuffle4_encode`:
/// 1. Subtract `base` (`wrapping_sub`) for any tensor with `delta_from: Some(base)`.
/// 2. For `delta_from` tensors, pre-apply `byte_shuffle4_decode` to `codes` so that after global
///    `byte_shuffle4_encode` its 32-byte horizontal zero-runs remain contiguous.
/// 3. For large vocabulary `Int4G64` tensors (`rows >= 65_536` and `delta_from.is_none()`),
///    transpose `codes` (`[rows, cols/2] -> [cols/2, rows]`) and pre-apply `byte_shuffle4_decode`
///    so that after global `byte_shuffle4_encode` each column across the 248,094 vocabulary rows
///    remains contiguous.
pub fn col_delta_pre_encode_in_place(
    buf: &mut [u8],
    tensors: &BTreeMap<String, TensorDescriptor>,
) {
    let deltas: Vec<(usize, usize, usize, usize, usize, usize)> = tensors
        .values()
        .filter_map(|td| {
            let base_name = td.delta_from.as_ref()?;
            let btd = tensors.get(base_name)?;
            Some((
                td.offset,
                td.length,
                btd.offset,
                td.aux_offset,
                td.aux_length,
                btd.aux_offset,
            ))
        })
        .collect();
    for (off, len, boff, aoff, alen, baoff) in deltas {
        if off != boff {
            for i in 0..len {
                buf[off + i] = buf[off + i].wrapping_sub(buf[boff + i]);
            }
        }
        if alen > 0 && aoff != baoff {
            for i in 0..alen {
                buf[aoff + i] = buf[aoff + i].wrapping_sub(buf[baoff + i]);
            }
        }
    }
    let mut seen_offsets = std::collections::BTreeSet::new();
    for td in tensors.values() {
        if !seen_offsets.insert(td.offset) {
            continue;
        }
        let [rows, cols] = td.shape;
        if td.dtype == TensorDtype::Int4G64 {
            if td.delta_from.is_none() && td.aux_length > 0 && td.length == td.aux_length * 8 {
                let num_g = td.aux_length / 4;
                for g in 0..num_g {
                    let goff = td.aux_offset + g * 4;
                    let coff = td.offset + g * 32;
                    let sc = half::f16::from_le_bytes([buf[goff], buf[goff + 1]]).to_f32();
                    let bs = half::f16::from_le_bytes([buf[goff + 2], buf[goff + 3]]).to_f32();
                    let qz = if sc > 0.0 {
                        ((-bs / sc).round() as i32).clamp(0, 15)
                    } else {
                        8_i32
                    };
                    let mut words = [0_u8; 32];
                    for w_idx in 0..8 {
                        let w_off = coff + w_idx * 4;
                        let mut nibs = [0_u8; 8];
                        for k in 0..4 {
                            let b = buf[w_off + k];
                            let d0 = (((i32::from(b & 0x0F) - qz + 8) & 0x0F) - 8) as i8;
                            let d1 = (((i32::from(b >> 4) - qz + 8) & 0x0F) - 8) as i8;
                            nibs[2 * k] = (((d0 << 1) ^ (d0 >> 3)) as u8) & 0x0F;
                            nibs[2 * k + 1] = (((d1 << 1) ^ (d1 >> 3)) as u8) & 0x0F;
                        }
                        for (bit, out_b) in words[w_idx * 4..w_idx * 4 + 4].iter_mut().enumerate() {
                            let mut bp = 0_u8;
                            for (j, &n) in nibs.iter().enumerate() {
                                bp |= ((n >> bit) & 1) << j;
                            }
                            *out_b = bp;
                        }
                    }
                    buf[coff..coff + 32].copy_from_slice(&words);
                }
            }
            let rb = cols / 2;
            if td.delta_from.is_some() {
                let copy = buf[td.offset..td.offset + td.length].to_vec();
                byte_shuffle4_decode(&copy, &mut buf[td.offset..td.offset + td.length]);
            } else if rows >= 65_536 && rb * rows == td.length {
                let transposed =
                    transpose_2d_bytes(&buf[td.offset..td.offset + td.length], rows, rb);
                byte_shuffle4_decode(&transposed, &mut buf[td.offset..td.offset + td.length]);
            }
            if td.delta_from.is_none() && td.aux_length > 0 {
                for chunk in buf[td.aux_offset..td.aux_offset + td.aux_length]
                    .as_chunks_mut::<4>()
                    .0
                {
                    let sc = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let pred = if sc == 0 {
                        0_u16
                    } else {
                        sc.wrapping_add(0x0C00) | 0x8000
                    };
                    let bs = u16::from_le_bytes([chunk[2], chunk[3]]).wrapping_sub(pred);
                    chunk[2..4].copy_from_slice(&bs.to_le_bytes());
                }
            }
        } else if td.dtype == TensorDtype::Ptq1 && rows >= 65_536 && cols.is_multiple_of(128) {
            let rb = (cols / 128) * 28;
            if rb * rows == td.length {
                let transposed =
                    transpose_2d_bytes(&buf[td.offset..td.offset + td.length], rows, rb);
                byte_shuffle4_decode(&transposed, &mut buf[td.offset..td.offset + td.length]);
            }
        }
    }
}

/// Split `buf` into mutable slices for `ranges` (`(start, len)`, sorted, non-overlapping).
fn split_disjoint<'a>(mut buf: &'a mut [u8], ranges: &[(usize, usize)]) -> crate::Result<Vec<&'a mut [u8]>> {
    let mut out = Vec::with_capacity(ranges.len());
    let mut base = 0;
    for &(start, len) in ranges {
        let skip = start
            .checked_sub(base)
            .ok_or_else(|| Error::InvalidFormat("overlapping tensor ranges".into()))?;
        if skip.checked_add(len).is_none_or(|end| end > buf.len()) {
            return Err(Error::InvalidFormat("tensor range out of payload bounds".into()));
        }
        let (slice, rest) = std::mem::take(&mut buf)[skip..].split_at_mut(len);
        out.push(slice);
        buf = rest;
        base = start + len;
    }
    Ok(out)
}

/// Groups per parallel work item when restoring INT4 codes (16 KiB of codes).
const QZ_GROUPS_PER_JOB: usize = 512;

/// Inverse of `col_delta_pre_encode_in_place` (called after `byte_shuffle4_decode`), on all cores.
pub fn col_delta_post_decode_in_place(
    buf: &mut [u8],
    tensors: &BTreeMap<String, TensorDescriptor>,
) -> crate::Result<()> {
    let mut bp_lut = [0_u32; 256];
    for (b, slot) in bp_lut.iter_mut().enumerate() {
        let mut acc = 0_u32;
        for j in 0..8 {
            acc |= (((b as u32) >> j) & 1) << (j * 4);
        }
        *slot = acc;
    }
    let mut qz_byte_lut = [[0_u8; 256]; 16];
    for (qz, row) in qz_byte_lut.iter_mut().enumerate() {
        for (byte_zz, slot) in row.iter_mut().enumerate() {
            let z0 = (byte_zz as u8) & 0x0F;
            let z1 = (byte_zz as u8) >> 4;
            let d0 = i32::from(z0 >> 1) ^ -i32::from(z0 & 1);
            let d1 = i32::from(z1 >> 1) ^ -i32::from(z1 & 1);
            let qz_i32 = i32::from(qz as u8);
            let q0 = ((qz_i32 + d0) & 0x0F) as u8;
            let q1 = ((qz_i32 + d1) & 0x0F) as u8;
            *slot = q0 | (q1 << 4);
        }
    }
    let mut seen_offsets = std::collections::BTreeSet::new();
    let unique: Vec<&TensorDescriptor> = tensors
        .values()
        .filter(|td| seen_offsets.insert(td.offset))
        .collect();
    let int4: Vec<&TensorDescriptor> = unique
        .iter()
        .copied()
        .filter(|td| td.dtype == TensorDtype::Int4G64 && td.delta_from.is_none() && td.aux_length > 0)
        .collect();

    // 1. Restore biases predicted from scales.
    let mut aux_ranges: Vec<(usize, usize)> = int4.iter().map(|td| (td.aux_offset, td.aux_length)).collect();
    aux_ranges.sort_unstable();
    aux_ranges.dedup();
    let aux_jobs: Vec<&mut [u8]> = split_disjoint(buf, &aux_ranges)?
        .into_iter()
        .flat_map(|aux| aux.chunks_mut(QZ_GROUPS_PER_JOB * 4))
        .collect();
    par_each(aux_jobs, |aux| {
        for chunk in aux.as_chunks_mut::<4>().0 {
            let sc = u16::from_le_bytes([chunk[0], chunk[1]]);
            let pred = if sc == 0 {
                0_u16
            } else {
                sc.wrapping_add(0x0C00) | 0x8000
            };
            let bs = u16::from_le_bytes([chunk[2], chunk[3]]).wrapping_add(pred);
            chunk[2..4].copy_from_slice(&bs.to_le_bytes());
        }
    });

    // 2. Un-transpose vocabulary-sized tensors and un-shuffle `delta_from` codes.
    for td in &unique {
        let [rows, cols] = td.shape;
        let region = buf
            .get_mut(td.offset..td.offset.saturating_add(td.length))
            .ok_or_else(|| Error::InvalidFormat("tensor out of payload bounds".into()))?;
        let rb = match td.dtype {
            TensorDtype::Int4G64 => cols / 2,
            TensorDtype::Ptq1 if cols.is_multiple_of(128) => (cols / 128) * 28,
            _ => continue,
        };
        if td.dtype == TensorDtype::Int4G64 && td.delta_from.is_some() {
            let shuf = byte_shuffle4_encode(region);
            region.copy_from_slice(&shuf);
        } else if rows >= 65_536 && rb * rows == td.length {
            let restored = transpose_2d_bytes(&byte_shuffle4_encode(region), rb, rows);
            region.copy_from_slice(&restored);
        }
    }

    // 3. Undo zig-zag(q - zero point) + bit-plane transposition per 64-weight group.
    let qz_tensors: Vec<&TensorDescriptor> = int4
        .into_iter()
        .filter(|td| td.length == td.aux_length * 8)
        .collect();
    let mut ranges: Vec<(usize, usize)> = qz_tensors
        .iter()
        .flat_map(|td| [(td.offset, td.length), (td.aux_offset, td.aux_length)])
        .collect();
    ranges.sort_unstable();
    ranges.dedup();
    let mut slices: BTreeMap<usize, &mut [u8]> = ranges
        .iter()
        .map(|&(start, _)| start)
        .zip(split_disjoint(buf, &ranges)?)
        .collect();
    let mut jobs = Vec::new();
    for td in &qz_tensors {
        let (Some(codes), Some(aux)) = (slices.remove(&td.offset), slices.remove(&td.aux_offset)) else {
            continue;
        };
        let aux: &[u8] = aux;
        for (codes, aux) in codes
            .chunks_mut(QZ_GROUPS_PER_JOB * 32)
            .zip(aux.chunks(QZ_GROUPS_PER_JOB * 4))
        {
            jobs.push((codes, aux));
        }
    }
    par_each(jobs, |(codes, aux)| {
        for (group, sb) in codes.as_chunks_mut::<32>().0.iter_mut().zip(aux.as_chunks::<4>().0) {
            let sc = half::f16::from_le_bytes([sb[0], sb[1]]).to_f32();
            let bs = half::f16::from_le_bytes([sb[2], sb[3]]).to_f32();
            let qz = if sc > 0.0 {
                ((-bs / sc).round() as i32).clamp(0, 15) as usize
            } else {
                8_usize
            };
            let q_lut = &qz_byte_lut[qz];
            for word in group.as_chunks_mut::<4>().0 {
                let w_zz = bp_lut[word[0] as usize]
                    | (bp_lut[word[1] as usize] << 1)
                    | (bp_lut[word[2] as usize] << 2)
                    | (bp_lut[word[3] as usize] << 3);
                for (dst, zb) in word.iter_mut().zip(w_zz.to_le_bytes()) {
                    *dst = q_lut[zb as usize];
                }
            }
        }
    });

    // 4. Re-add `delta_from` bases.
    for td in tensors.values() {
        if let Some(base_name) = &td.delta_from
            && let Some(btd) = tensors.get(base_name)
        {
            if td.length == btd.length && td.offset != btd.offset {
                for i in 0..td.length {
                    buf[td.offset + i] = buf[td.offset + i].wrapping_add(buf[btd.offset + i]);
                }
            }
            if td.aux_length > 0
                && td.aux_length == btd.aux_length
                && td.aux_offset != btd.aux_offset
            {
                for i in 0..td.aux_length {
                    buf[td.aux_offset + i] =
                        buf[td.aux_offset + i].wrapping_add(buf[btd.aux_offset + i]);
                }
            }
        }
    }
    Ok(())
}

/// Decompress one zstd frame of `out.len()` bytes into `out`, un-shuffling byte planes when
/// `shuffled` (`parallel` lets a lone payload-sized legacy frame use every core).
fn decode_frame(src: &[u8], out: &mut [u8], dict: &[u8], shuffled: bool, parallel: bool) -> crate::Result<()> {
    let mut decoder = zstd::bulk::Decompressor::with_dictionary(dict)?;
    // Legacy single-frame artifacts use a 2^28 window (above zstd's default limit).
    decoder.window_log_max(31)?;
    let written = if shuffled {
        let mut planes = Vec::with_capacity(out.len());
        let written = decoder.decompress_to_buffer(src, &mut planes)?;
        if written == out.len() {
            if parallel {
                byte_shuffle4_decode(&planes, out);
            } else {
                byte_shuffle4_decode_serial(&planes, out);
            }
        }
        written
    } else {
        decoder.decompress_to_buffer(src, out)?
    };
    if written != out.len() {
        return Err(Error::InvalidFormat(format!(
            "zstd frame decoded {written} bytes, expected {}",
            out.len()
        )));
    }
    Ok(())
}

/// Decode a compressed payload (`body` = optional dictionary + zstd frames) into `dst`
/// (`header.payload_length` bytes): frames in parallel, then the inverse `col_delta` transforms.
/// Frame content checksums (XXH64) catch corruption.
fn decode_payload(header: &ArtifactHeader, body: &[u8], dst: &mut [u8]) -> crate::Result<()> {
    let mode = header.compression.as_deref().unwrap_or("none");
    if !matches!(mode, "zstd" | "zstd_shuffle4" | "zstd_col_delta") {
        return Err(Error::InvalidFormat(format!("unsupported compression {mode}")));
    }
    let shuffled = mode != "zstd";
    let (dict, stream) = body
        .split_at_checked(header.zstd_dict_length)
        .ok_or_else(|| Error::InvalidFormat("zstd dictionary exceeds file".into()))?;
    // Legacy artifacts hold one frame spanning the whole payload.
    let (frame_bytes, sizes) = if header.zstd_frames.is_empty() {
        (dst.len(), vec![stream.len() as u64])
    } else {
        (header.zstd_frame_bytes as usize, header.zstd_frames.clone())
    };
    if frame_bytes == 0
        || !frame_bytes.is_multiple_of(4)
        || sizes.len() != dst.len().div_ceil(frame_bytes)
    {
        return Err(Error::InvalidFormat("zstd frame table does not match payload".into()));
    }
    let mut frames = Vec::with_capacity(sizes.len());
    let mut rest = stream;
    for size in sizes {
        let (frame, tail) = rest
            .split_at_checked(size as usize)
            .ok_or_else(|| Error::InvalidFormat("zstd frame exceeds file".into()))?;
        frames.push(frame);
        rest = tail;
    }
    let parallel = frames.len() == 1;
    let jobs: Vec<(&[u8], &mut [u8])> = frames.into_iter().zip(dst.chunks_mut(frame_bytes)).collect();
    let first_error = Mutex::new(None);
    par_each(jobs, |(src, out)| {
        if let Err(e) = decode_frame(src, out, dict, shuffled, parallel) {
            first_error
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(e);
        }
    });
    if let Some(e) = first_error.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(e);
    }
    if mode == "zstd_col_delta" {
        col_delta_post_decode_in_place(dst, &header.tensors)?;
    }
    Ok(())
}

/// Apply the lossless `zstd_col_delta` transforms to `canonical` and compress it into independent
/// `ZSTD_FRAME_BYTES` frames (byte-shuffled per frame) on every core; records the frame table in
/// `header` and returns the concatenated frames.
fn encode_payload(header: &mut ArtifactHeader, mut canonical: Vec<u8>, level: i32) -> crate::Result<Vec<u8>> {
    col_delta_pre_encode_in_place(&mut canonical, &header.tensors);
    let chunks: Vec<&[u8]> = canonical.chunks(ZSTD_FRAME_BYTES).collect();
    let frames: Vec<Mutex<crate::Result<Vec<u8>>>> =
        chunks.iter().map(|_| Mutex::new(Ok(Vec::new()))).collect();
    par_for(chunks.len(), |i| {
        let frame = (|| {
            let mut encoder = zstd::bulk::Compressor::new(level)?;
            encoder.include_checksum(true)?;
            Ok(encoder.compress(&byte_shuffle4_encode(chunks[i]))?)
        })();
        *frames[i].lock().unwrap_or_else(PoisonError::into_inner) = frame;
    });
    let mut stream = Vec::new();
    let mut sizes = Vec::with_capacity(frames.len());
    for frame in frames {
        let frame = frame.into_inner().unwrap_or_else(PoisonError::into_inner)?;
        sizes.push(frame.len() as u64);
        stream.extend_from_slice(&frame);
    }
    header.compression = Some("zstd_col_delta".to_owned());
    header.zstd_dict_length = 0;
    header.zstd_frame_bytes = ZSTD_FRAME_BYTES as u64;
    header.zstd_frames = sizes;
    Ok(stream)
}

/// Atomically write `magic`, `header` and the stored payload `body` to `path`; returns the file size.
fn write_artifact(path: &Path, magic: [u8; 8], header: &ArtifactHeader, body: &[u8]) -> crate::Result<usize> {
    let header_bytes = serde_json::to_vec(header)?;
    let payload_start = (16 + header_bytes.len()).div_ceil(METAL_PAGE) * METAL_PAGE;
    let tmp_path = path.with_extension(format!("tmp.{}", std::process::id()));
    let written = (|| -> crate::Result<()> {
        let mut out = std::io::BufWriter::new(File::create(&tmp_path)?);
        out.write_all(&magic)?;
        out.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        out.write_all(&header_bytes)?;
        out.write_all(&vec![b' '; payload_start - 16 - header_bytes.len()])?;
        out.write_all(body)?;
        out.flush()?;
        Ok(())
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(payload_start + body.len())
}

/// Read magic, JSON header and payload start (16-KiB aligned) from an artifact file mapping.
fn parse_header(map: &[u8], path: &Path) -> crate::Result<([u8; 8], ArtifactHeader, usize)> {
    let (Some(magic), Some(len_bytes)) = (map.first_chunk::<8>(), map.get(8..16)) else {
        return Err(Error::InvalidFormat(format!("{}: truncated artifact", path.display())));
    };
    let header_len = u64::from_le_bytes(
        len_bytes
            .try_into()
            .map_err(|_| Error::InvalidFormat("header length slice".into()))?,
    ) as usize;
    let payload_start = (16 + header_len).div_ceil(METAL_PAGE) * METAL_PAGE;
    if payload_start > map.len() {
        return Err(Error::InvalidFormat("header exceeds file length".into()));
    }
    let header: ArtifactHeader = serde_json::from_slice(&map[16..16 + header_len])?;
    Ok((*magic, header, payload_start))
}

static HTTP_ZSTD_DICT_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let samples: &[&str] = &[
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" quick"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" brown"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" fox"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" jumps"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" over"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" lazy"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" dog."},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"\n"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"Here"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" concise"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":" answer"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":": "},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"42"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "data: [DONE]\n\n",
        r#"{"id":"chatcmpl-micro","object":"chat.completion","model":"minicpm-v-4.7-1b","choices":[{"index":0,"message":{"role":"assistant","content":"The answer is 42.","reasoning_content":"Let us compute step by step."},"finish_reason":"stop"}],"usage":{"prompt_tokens":32,"cached_prefix_tokens":24,"completion_tokens":16,"total_tokens":48},"telemetry":{"prefill_ms":4.2,"decode_ms":38.5,"prefill_tok_per_s":7619.0,"decode_tok_per_s":415.6,"speculative_rounds":4,"drafted_tokens":16,"accepted_tokens":12,"ngram_accepted_tokens":8}}"#,
        r#"{"id":"chatcmpl-micro","object":"chat.completion","model":"minicpm-v-4.7-1b","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I help you today?","reasoning_content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"cached_prefix_tokens":14,"completion_tokens":9,"total_tokens":27},"telemetry":{"prefill_ms":2.1,"decode_ms":19.4,"prefill_tok_per_s":8571.4,"decode_tok_per_s":463.9,"speculative_rounds":2,"drafted_tokens":8,"accepted_tokens":7,"ngram_accepted_tokens":3}}"#,
        r#"{"id":"chatcmpl-micro","object":"chat.completion","model":"minicpm-v-4.7-1b","choices":[{"index":0,"message":{"role":"assistant","content":"```rust\nfn main() {\n    println!(\"Hello, world!\");\n}\n```","reasoning_content":"Writing a minimal Rust main function."},"finish_reason":"stop"}],"usage":{"prompt_tokens":45,"cached_prefix_tokens":30,"completion_tokens":28,"total_tokens":73},"telemetry":{"prefill_ms":5.8,"decode_ms":64.1,"prefill_tok_per_s":7758.6,"decode_tok_per_s":436.8,"speculative_rounds":6,"drafted_tokens":24,"accepted_tokens":20,"ngram_accepted_tokens":10}}"#,
        r#"{"id":"chatcmpl-micro","object":"chat.completion","model":"minicpm-v-4.7-1b","choices":[{"index":0,"message":{"role":"assistant","content":"This image shows a high-resolution diagram with three interconnected modules.","reasoning_content":"Inspecting the visual layout and labels."},"finish_reason":"stop"}],"usage":{"prompt_tokens":128,"cached_prefix_tokens":0,"completion_tokens":22,"total_tokens":150},"telemetry":{"prefill_ms":18.5,"decode_ms":51.2,"prefill_tok_per_s":6918.9,"decode_tok_per_s":429.7,"speculative_rounds":5,"drafted_tokens":20,"accepted_tokens":16,"ngram_accepted_tokens":4}}"#,
        r#"{"object":"list","data":[{"id":"minicpm-v-4.7-1b","object":"model","owned_by":"openbmb"}]}"#,
        r#"{"status":"ok","model":"minicpm-v-4.7-1b","device":"Apple M4 Max","ane":{"enabled":true}}"#,
        r#"{"error":{"message":"prompt cannot be empty","type":"invalid_request_error"}}"#,
        r#"{"error":{"message":"unknown route; use GET /health, GET /v1/models, or POST /v1/chat/completions","type":"invalid_request_error"}}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"<think>\n"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"</think>\n"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"1. First step\n2. Second step\n"},"finish_reason":null}]}"#,
        r#"data: {"id":"chatcmpl-micro","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"{\"result\":true,\"score\":0.99}"},"finish_reason":null}]}"#,
    ];
    let mut repeated = Vec::with_capacity(samples.len() * 8);
    for _ in 0..8 {
        for &s in samples {
            repeated.push(s.as_bytes());
        }
    }
    zstd::dict::from_samples(&repeated, 8192).unwrap_or_default()
});

/// Return a pre-trained ~8 KiB Zstandard dictionary optimized for `OpenAI` `/v1/chat/completions` JSON & SSE responses.
#[must_use]
pub fn http_zstd_dict() -> &'static [u8] {
    &HTTP_ZSTD_DICT_BYTES
}

/// Compress `data` with Zstandard at `level`, optionally using a pre-trained `dict`.
pub fn zstd_compress_with_dict(data: &[u8], dict: &[u8], level: i32) -> crate::Result<Vec<u8>> {
    if dict.is_empty() {
        return Ok(zstd::stream::encode_all(data, level)?);
    }
    let mut enc = zstd::stream::write::Encoder::with_dictionary(Vec::new(), level, dict)?;
    enc.write_all(data)?;
    Ok(enc.finish()?)
}

/// Decompress Zstandard `data`, optionally using a pre-trained `dict`, refusing output over
/// `limit` bytes (stops decoding at `limit + 1`, so a decompression bomb never allocates more).
pub fn zstd_decompress_with_dict(data: &[u8], dict: &[u8], limit: usize) -> crate::Result<Vec<u8>> {
    let dec = zstd::stream::read::Decoder::with_dictionary(data, dict)?;
    let mut out = Vec::new();
    dec.take(limit as u64 + 1).read_to_end(&mut out)?;
    if out.len() > limit {
        return Err(Error::InvalidArgument(format!("zstd output exceeds {limit} bytes")));
    }
    Ok(out)
}

/// Check every tensor's primary and aux byte ranges against the payload length.
fn validate_tensor_bounds(header: &ArtifactHeader, payload_len: usize) -> crate::Result<()> {
    for (name, desc) in &header.tensors {
        let out_of_bounds = |offset: usize, length: usize| {
            offset.checked_add(length).is_none_or(|end| end > payload_len)
        };
        if out_of_bounds(desc.offset, desc.length)
            || (desc.aux_length > 0 && out_of_bounds(desc.aux_offset, desc.aux_length))
        {
            return Err(Error::InvalidFormat(format!(
                "tensor {name} out of payload bounds"
            )));
        }
    }
    Ok(())
}

impl MappedArtifact {
    /// Open an artifact and expose its 16-KiB-aligned payload to Metal without copies:
    /// raw payloads stay file-backed; compressed ones are decoded on every core into anonymous
    /// memory (no on-disk cache).
    #[expect(unsafe_code)]
    pub fn open(ctx: &MetalContext, path: &Path, expected_magic: [u8; 8]) -> crate::Result<Self> {
        let file = File::open(path)?;
        let file_map = unsafe { Mmap::map(&file)? };
        drop(file);
        let (magic, header, file_payload_start) = parse_header(&file_map, path)?;
        if magic != expected_magic {
            return Err(Error::InvalidFormat(format!("{}: bad magic", path.display())));
        }
        let expected_len = header.payload_length as usize;
        if !expected_len.is_multiple_of(METAL_PAGE) {
            return Err(Error::InvalidFormat(
                "payload length is not 16-KiB page aligned".into(),
            ));
        }
        validate_tensor_bounds(&header, expected_len)?;

        let (map, payload_start) = if header.compression.is_some() {
            let mut decoded = MmapMut::map_anon(expected_len)?;
            decode_payload(&header, &file_map[file_payload_start..], &mut decoded)?;
            drop(file_map);
            (Arc::new(decoded.make_read_only()?), 0_usize)
        } else {
            let payload_len = file_map.len() - file_payload_start;
            if payload_len != expected_len {
                return Err(Error::InvalidFormat(format!(
                    "payload length mismatch: file has {payload_len}, header says {expected_len}"
                )));
            }
            (Arc::new(file_map), file_payload_start)
        };

        let payload = &map[payload_start..];
        let ptr = NonNull::new(payload.as_ptr().cast_mut())
            .ok_or_else(|| Error::InvalidFormat("null mmap pointer".into()))?;
        let owner: Arc<dyn Send + Sync> = Arc::clone(&map) as Arc<dyn Send + Sync>;
        let payload_buffer =
            unsafe { MetalBuffer::from_bytes_no_copy(ctx.device(), ptr, payload.len(), owner)? };
        let artifact = Self {
            map,
            header,
            payload_buffer,
        };
        artifact.prefetch();
        Ok(artifact)
    }

    /// Ask the kernel to read file-backed weights in ahead of use. Pages the OS dropped (cold
    /// start, or reclaimed while the app sat in the background) otherwise fault back in one by
    /// one from the GPU; measured on an iPhone SE 3 that ran decode at 3.4 instead of ~50 tok/s.
    pub fn prefetch(&self) {
        let _ = self.map.advise(memmap2::Advice::WillNeed);
    }

    pub fn descriptor(&self, name: &str) -> crate::Result<&TensorDescriptor> {
        self.header
            .tensors
            .get(name)
            .ok_or_else(|| Error::InvalidFormat(format!("missing tensor {name:?}")))
    }

    pub fn matrix(&self, name: &str) -> crate::Result<(AnyMatrix<'_>, bool)> {
        Ok(self.matrix_at(self.matrix_handle(name)?))
    }

    /// Resolve a matrix tensor once, for hot paths that must not look names up.
    pub fn matrix_handle(&self, name: &str) -> crate::Result<MatrixHandle> {
        let d = self.descriptor(name)?;
        let format = match d.dtype {
            TensorDtype::Int4G64 => MatrixFormat::Int4,
            TensorDtype::Ptq1 => MatrixFormat::Ptq1,
            TensorDtype::Int8Row => MatrixFormat::Int8,
            TensorDtype::Bf16 => MatrixFormat::Bf16,
            TensorDtype::F32 => {
                return Err(Error::InvalidFormat(format!(
                    "tensor {name} is F32, not a quantized/BF16 matrix"
                )));
            }
        };
        Ok(MatrixHandle {
            format,
            offset: d.offset,
            aux_offset: d.aux_offset,
            rows: d.shape[0] as u32,
            columns: d.shape[1] as u32,
            rotated: d.rotated,
        })
    }

    /// Kernel operands of `handle` and whether it lives in the rotated basis.
    #[must_use]
    pub const fn matrix_at(&self, handle: MatrixHandle) -> (AnyMatrix<'_>, bool) {
        let MatrixHandle { format, offset, aux_offset, rows, columns, rotated } = handle;
        let buffer = &self.payload_buffer;
        let mat = match format {
            MatrixFormat::Int4 => AnyMatrix::Int4(Int4Matrix {
                codes: buffer,
                codes_offset: offset,
                groups: buffer,
                groups_offset: aux_offset,
                rows,
                columns,
            }),
            MatrixFormat::Ptq1 => AnyMatrix::Ptq1(Ptq1Matrix { buffer, offset, rows, columns }),
            MatrixFormat::Int8 => AnyMatrix::Int8(Int8Matrix {
                weights: buffer,
                weights_offset: offset,
                scales: buffer,
                scales_offset: aux_offset,
                rows,
                columns,
            }),
            MatrixFormat::Bf16 => AnyMatrix::Bf16(Bf16Matrix { buffer, offset, rows, columns }),
        };
        (mat, rotated)
    }

    pub fn f32_slice(&self, name: &str) -> crate::Result<(&MetalBuffer, usize, usize)> {
        let d = self.descriptor(name)?;
        if d.dtype != TensorDtype::F32 {
            return Err(Error::InvalidFormat(format!("tensor {name} is not F32")));
        }
        Ok((&self.payload_buffer, d.offset, d.shape[0] * d.shape[1]))
    }

}

/// Read an artifact's magic, header and decoded (canonical) payload; returns the file size first.
#[expect(unsafe_code)]
fn decode_artifact_canonical(path: &Path) -> crate::Result<(usize, [u8; 8], ArtifactHeader, Vec<u8>)> {
    let file = File::open(path)?;
    let file_map = unsafe { Mmap::map(&file)? };
    let (magic, header, payload_start) = parse_header(&file_map, path)?;
    let expected_len = header.payload_length as usize;
    validate_tensor_bounds(&header, expected_len)?;
    let body = &file_map[payload_start..];
    let canonical = if header.compression.is_some() {
        let mut out = vec![0_u8; expected_len];
        decode_payload(&header, body, &mut out)?;
        out
    } else {
        body.get(..expected_len)
            .ok_or_else(|| Error::InvalidFormat("payload truncated".into()))?
            .to_vec()
    };
    Ok((file_map.len(), magic, header, canonical))
}

/// Rewrite an artifact in place with FP14 re-snapping and `zstd_col_delta` compression.
pub fn repack_artifact(path: &Path, level: i32) -> crate::Result<(usize, usize)> {
    let (old_file_bytes, magic, mut header, mut canonical) = decode_artifact_canonical(path)?;
    let sha = hex(Sha256::digest(&canonical).as_slice());
    if !header.payload_sha256.is_empty() && header.payload_sha256 != sha {
        return Err(Error::InvalidFormat(format!(
            "payload SHA-256 mismatch: got {sha}, expected {}",
            header.payload_sha256
        )));
    }
    let mut seen_offsets = std::collections::BTreeSet::new();
    for td in header.tensors.values() {
        if td.dtype == TensorDtype::Int4G64
            && td.delta_from.is_none()
            && td.aux_length > 0
            && seen_offsets.insert(td.offset)
        {
            let mut codes = canonical[td.offset..td.offset + td.length].to_vec();
            let mut groups = canonical[td.aux_offset..td.aux_offset + td.aux_length].to_vec();
            round_int4_groups_and_resnap(&mut codes, &mut groups);
            canonical[td.offset..td.offset + td.length].copy_from_slice(&codes);
            canonical[td.aux_offset..td.aux_offset + td.aux_length].copy_from_slice(&groups);
        }
    }
    header.payload_sha256 = hex(Sha256::digest(&canonical).as_slice());
    let new_file_bytes = if level > 0 {
        let stream = encode_payload(&mut header, canonical, level)?;
        write_artifact(path, magic, &header, &stream)?
    } else {
        mark_raw(&mut header);
        write_artifact(path, magic, &header, &canonical)?
    };
    Ok((old_file_bytes, new_file_bytes))
}

/// Drop the compression fields: the payload is stored exactly as Metal reads it.
fn mark_raw(header: &mut ArtifactHeader) {
    header.compression = None;
    header.zstd_dict_length = 0;
    header.zstd_frame_bytes = 0;
    header.zstd_frames = Vec::new();
}

const fn round_fp16_mant4(bits: u16) -> u16 {
    if (bits & 0x7C00) == 0x7C00 {
        return bits;
    }
    let rounded = bits.wrapping_add(8) & 0xFFF0;
    if rounded.trailing_zeros() >= 15 && bits.trailing_zeros() < 15 {
        bits & 0xFFF0
    } else {
        rounded
    }
}


fn round_int4_groups_and_resnap(codes: &mut [u8], groups: &mut [u8]) {
    use half::f16;
    let num_total_groups = groups.len() / 4;
    for g in 0..num_total_groups {
        let goff = g * 4;
        let coff = g * 32;
        let sc_bits = u16::from_le_bytes([groups[goff], groups[goff + 1]]);
        let bs_bits = u16::from_le_bytes([groups[goff + 2], groups[goff + 3]]);
        let sc_old = f16::from_bits(sc_bits).to_f32();
        let bs_old = f16::from_bits(bs_bits).to_f32();
        let sc_new_bits = round_fp16_mant4(sc_bits);
        let bs_new_bits = round_fp16_mant4(bs_bits);
        groups[goff..goff + 2].copy_from_slice(&sc_new_bits.to_le_bytes());
        groups[goff + 2..goff + 4].copy_from_slice(&bs_new_bits.to_le_bytes());
        let sc_new = f16::from_bits(sc_new_bits).to_f32();
        let bs_new = f16::from_bits(bs_new_bits).to_f32();
        if sc_new > 0.0 {
            let inv = 1.0 / sc_new;
            let mut lut = [0_u8; 16];
            for (q, slot) in lut.iter_mut().enumerate() {
                let w = (q as f32).mul_add(sc_old, bs_old);
                *slot = ((w - bs_new) * inv).round().clamp(0.0, 15.0) as u8;
            }
            for b in &mut codes[coff..coff + 32] {
                let q0 = lut[(*b & 0x0F) as usize];
                let q1 = lut[(*b >> 4) as usize];
                *b = q0 | (q1 << 4);
            }
        }
    }
}
/// Fuse `minicpm-v-4.7-1b-tern4.bin`, `minicpm-v-4.7-1b-vision.bin`, and `minicpm-v-4.7-1b-dspark.bin`
/// into a single unified `MCV47` artifact:
/// 1. Ties `embed_tokens.weight` and `lm_head.weight` (one shared copy of the embedding codes).
/// 2. Rounds `INT4_G64` FP16 `(scale, bias)` mantissas to FP14 with `q` re-snapping (a no-op on
///    converter output, which is already FP14).
/// 3. Compresses the payload with `zstd_col_delta` at `level` (or stores raw when `level <= 0`).
/// 4. Replaces `tokenizer.json` with the compact, Zstd-compressed `tokenizer.bin`.
///
/// An already fused artifact (tied embedding) is rewritten losslessly, so re-running is safe.
pub fn fuse_and_requantize_suite(model_dir: &Path) -> crate::Result<serde_json::Value> {
    fuse_and_requantize_suite_with_level(model_dir, 19)
}

pub fn fuse_and_requantize_suite_with_level(
    model_dir: &Path,
    level: i32,
) -> crate::Result<serde_json::Value> {
    fn align256(buf: &mut Vec<u8>) {
        let rem = buf.len() % 256;
        if rem != 0 {
            buf.resize(buf.len() + (256 - rem), 0);
        }
    }

    let lm_path = model_dir.join("minicpm-v-4.7-1b-tern4.bin");
    let vis_path = model_dir.join("minicpm-v-4.7-1b-vision.bin");
    let dsp_path = model_dir.join("minicpm-v-4.7-1b-dspark.bin");

    let tokenizer_compacted = crate::tokenizer::compact_in_place(model_dir)?;
    let (old_lm_b, _, mut lm_hdr, mut lm_raw) = decode_artifact_canonical(&lm_path)?;
    if let (Some(emb), Some(head)) = (
        lm_hdr.tensors.get("model.language_model.embed_tokens.weight"),
        lm_hdr.tensors.get("lm_head.weight"),
    ) && emb.offset == head.offset
    {
        drop(lm_raw);
        let (old_file_bytes, new_file_bytes) = repack_artifact(&lm_path, level)?;
        return Ok(serde_json::json!({
            "fused_file": "minicpm-v-4.7-1b-tern4.bin",
            "repacked": true,
            "old_file_bytes": old_file_bytes,
            "new_file_bytes": new_file_bytes,
            "saved_mb": ((old_file_bytes.saturating_sub(new_file_bytes)) as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0,
            "compression": if level > 0 { "zstd_col_delta" } else { "none" },
            "tokenizer_compacted": tokenizer_compacted,
        }));
    }
    let mut old_total_b = old_lm_b;
    let mut source_parts = vec![(lm_hdr.tensors.clone(), std::mem::take(&mut lm_raw))];

    if vis_path.is_file() {
        let (vb, _, vhdr, vraw) = decode_artifact_canonical(&vis_path)?;
        old_total_b += vb;
        source_parts.push((vhdr.tensors, vraw));
    }
    if dsp_path.is_file() {
        let (db, _, dhdr, draw) = decode_artifact_canonical(&dsp_path)?;
        old_total_b += db;
        source_parts.push((dhdr.tensors, draw));
    }

    let mut codes_region = Vec::<u8>::new();
    let mut aux_region = Vec::<u8>::new();
    let mut fused_tensors = BTreeMap::<String, TensorDescriptor>::new();

    for (tensors, raw) in &source_parts {
        for (name, td) in tensors {
            // 1. Tie `embed_tokens.weight` to `lm_head.weight` when both exist with identical shape
            if name == "model.language_model.embed_tokens.weight"
                && let Some(head_td) = tensors.get("lm_head.weight")
                && td.shape == head_td.shape
            {
                continue;
            }

            match td.dtype {
                TensorDtype::Int4G64 => {
                    let src_td = if name == "lm_head.weight"
                        && let Some(emb_td) =
                            tensors.get("model.language_model.embed_tokens.weight")
                        && emb_td.dtype == TensorDtype::Int4G64
                        && emb_td.shape == td.shape
                    {
                        emb_td
                    } else {
                        td
                    };
                    let mut codes = raw[src_td.offset..src_td.offset + src_td.length].to_vec();
                    let mut groups =
                        raw[src_td.aux_offset..src_td.aux_offset + src_td.aux_length].to_vec();
                    round_int4_groups_and_resnap(&mut codes, &mut groups);
                    align256(&mut codes_region);
                    let c_off = codes_region.len();
                    codes_region.extend_from_slice(&codes);
                    align256(&mut aux_region);
                    let a_off = aux_region.len();
                    aux_region.extend_from_slice(&groups);
                    let desc = TensorDescriptor {
                        dtype: TensorDtype::Int4G64,
                        shape: td.shape,
                        offset: c_off,
                        length: codes.len(),
                        aux_offset: a_off,
                        aux_length: groups.len(),
                        rotated: td.rotated,
                        delta_from: None,
                    };
                    if name == "lm_head.weight"
                        && tensors.contains_key("model.language_model.embed_tokens.weight")
                    {
                        fused_tensors.insert(
                            "model.language_model.embed_tokens.weight".to_owned(),
                            desc.clone(),
                        );
                    }
                    fused_tensors.insert(name.clone(), desc);
                }
                TensorDtype::Ptq1 => {
                    align256(&mut codes_region);
                    let c_off = codes_region.len();
                    codes_region.extend_from_slice(&raw[td.offset..td.offset + td.length]);
                    fused_tensors.insert(
                        name.clone(),
                        TensorDescriptor {
                            dtype: TensorDtype::Ptq1,
                            shape: td.shape,
                            offset: c_off,
                            length: td.length,
                            aux_offset: 0,
                            aux_length: 0,
                            rotated: td.rotated,
                            delta_from: None,
                        },
                    );
                }
                TensorDtype::Bf16 => {
                    align256(&mut aux_region);
                    let off = aux_region.len();
                    let mut bf_bytes = raw[td.offset..td.offset + td.length].to_vec();
                    if name.contains("position_embedding") {
                        for chunk in bf_bytes.as_chunks_mut::<2>().0 {
                            let u = u16::from_le_bytes(*chunk);
                            let r = u.wrapping_add(2) & 0xFFFC;
                            *chunk = r.to_le_bytes();
                        }
                    }
                    aux_region.extend_from_slice(&bf_bytes);
                    fused_tensors.insert(
                        name.clone(),
                        TensorDescriptor {
                            dtype: TensorDtype::Bf16,
                            shape: td.shape,
                            offset: off,
                            length: td.length,
                            aux_offset: 0,
                            aux_length: 0,
                            rotated: td.rotated,
                            delta_from: None,
                        },
                    );
                }
                TensorDtype::F32 | TensorDtype::Int8Row => {
                    align256(&mut aux_region);
                    let off = aux_region.len();
                    aux_region.extend_from_slice(&raw[td.offset..td.offset + td.length]);
                    let (a_off, a_len) = if td.aux_length > 0 {
                        align256(&mut aux_region);
                        let ao = aux_region.len();
                        aux_region
                            .extend_from_slice(&raw[td.aux_offset..td.aux_offset + td.aux_length]);
                        (ao, td.aux_length)
                    } else {
                        (0, 0)
                    };
                    fused_tensors.insert(
                        name.clone(),
                        TensorDescriptor {
                            dtype: td.dtype,
                            shape: td.shape,
                            offset: off,
                            length: td.length,
                            aux_offset: a_off,
                            aux_length: a_len,
                            rotated: td.rotated,
                            delta_from: None,
                        },
                    );
                }
            }
        }
    }
    drop(source_parts);

    align256(&mut codes_region);
    let aux_base = codes_region.len();
    for td in fused_tensors.values_mut() {
        if matches!(td.dtype, TensorDtype::Bf16 | TensorDtype::F32 | TensorDtype::Int8Row) {
            td.offset += aux_base;
        }
        if td.aux_length > 0 {
            td.aux_offset += aux_base;
        }
    }

    let mut payload = codes_region;
    payload.extend_from_slice(&aux_region);
    drop(aux_region);
    let rem = payload.len() % METAL_PAGE;
    if rem != 0 {
        payload.resize(payload.len() + (METAL_PAGE - rem), 0);
    }
    let payload_len = payload.len();
    let sha_hex = hex(Sha256::digest(&payload).as_slice());
    lm_hdr.tensors = fused_tensors;
    lm_hdr.payload_length = payload_len as u64;
    lm_hdr.payload_sha256.clone_from(&sha_hex);
    let (new_file_bytes, comp_name) = if level > 0 {
        let stream = encode_payload(&mut lm_hdr, payload, level)?;
        (write_artifact(&lm_path, LM_MAGIC, &lm_hdr, &stream)?, "zstd_col_delta")
    } else {
        mark_raw(&mut lm_hdr);
        let n = write_artifact(&lm_path, LM_MAGIC, &lm_hdr, &payload)?;
        drop(payload);
        (n, "none")
    };
    let _ = std::fs::remove_file(&vis_path);
    let _ = std::fs::remove_file(&dsp_path);

    Ok(serde_json::json!({
        "fused_file": "minicpm-v-4.7-1b-tern4.bin",
        "old_total_bytes": old_total_b,
        "new_file_bytes": new_file_bytes,
        "new_file_mb": (new_file_bytes as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0,
        "saved_mb": ((old_total_b.saturating_sub(new_file_bytes)) as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0,
        "payload_bytes": payload_len,
        "payload_mb": (payload_len as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0,
        "tensors": lm_hdr.tensors.len(),
        "payload_sha256": sha_hex,
        "compression": comp_name,
        "tokenizer_compacted": tokenizer_compacted,
    }))
}

/// Write a synthetic 24-layer `minicpm-v-4.7-1b` (`MCV47`) artifact for benchmarks and tests,
/// returning the total file size in bytes.
pub fn write_synthetic_lm_artifact(path: &Path, compress: bool) -> crate::Result<usize> {
    use crate::model::{LAYERS, VOCAB, WIDTH};

    fn align256(buf: &mut Vec<u8>) {
        let rem = buf.len() % 256;
        if rem != 0 {
            buf.resize(buf.len() + (256 - rem), 0);
        }
    }

    let mut payload = Vec::<u8>::new();
    let mut tensors = BTreeMap::<String, TensorDescriptor>::new();

    let add_ptq1 = |payload: &mut Vec<u8>,
                    tensors: &mut BTreeMap<String, TensorDescriptor>,
                    name: &str,
                    rows: usize,
                    cols: usize,
                    rotated: bool| {
        align256(payload);
        let offset = payload.len();
        let bytes = rows * (cols / 128) * 28;
        payload.resize(offset + bytes, 0);
        for (idx, b) in payload[offset..offset + bytes.min(65_536)].iter_mut().enumerate() {
            if idx % 28 < 2 {
                *b = ((idx / 28) & 0x3F) as u8;
            } else {
                *b = ((idx * 17 + rows) & 0xFF) as u8;
            }
        }
        tensors.insert(
            name.to_owned(),
            TensorDescriptor {
                dtype: TensorDtype::Ptq1,
                shape: [rows, cols],
                offset,
                length: bytes,
                aux_offset: 0,
                aux_length: 0,
                rotated,
                delta_from: None,
            },
        );
    };
    let add_int4 = |payload: &mut Vec<u8>,
                    tensors: &mut BTreeMap<String, TensorDescriptor>,
                    name: &str,
                    rows: usize,
                    cols: usize| {
        align256(payload);
        let offset = payload.len();
        let codes_len = rows * (cols / 2);
        payload.resize(offset + codes_len, 0x88);
        for (idx, b) in payload[offset..offset + codes_len.min(16_384)].iter_mut().enumerate() {
            *b = 0x70 | (((idx * 13 + cols) & 0x03) as u8);
        }
        align256(payload);
        let aux_offset = payload.len();
        let aux_len = rows * (cols / 64) * 4;
        payload.resize(aux_offset + aux_len, 0);
        tensors.insert(
            name.to_owned(),
            TensorDescriptor {
                dtype: TensorDtype::Int4G64,
                shape: [rows, cols],
                offset,
                length: codes_len,
                aux_offset,
                aux_length: aux_len,
                rotated: false,
                delta_from: None,
            },
        );
    };
    let add_bf16 = |payload: &mut Vec<u8>,
                    tensors: &mut BTreeMap<String, TensorDescriptor>,
                    name: &str,
                    rows: usize,
                    cols: usize| {
        align256(payload);
        let offset = payload.len();
        let bytes = rows * cols * 2;
        payload.resize(offset + bytes, 0);
        tensors.insert(
            name.to_owned(),
            TensorDescriptor {
                dtype: TensorDtype::Bf16,
                shape: [rows, cols],
                offset,
                length: bytes,
                aux_offset: 0,
                aux_length: 0,
                rotated: false,
                delta_from: None,
            },
        );
    };
    let add_f32 = |payload: &mut Vec<u8>,
                   tensors: &mut BTreeMap<String, TensorDescriptor>,
                   name: &str,
                   elements: usize,
                   val: f32| {
        align256(payload);
        let offset = payload.len();
        let bytes = elements * 4;
        for _ in 0..elements {
            payload.extend_from_slice(&val.to_le_bytes());
        }
        tensors.insert(
            name.to_owned(),
            TensorDescriptor {
                dtype: TensorDtype::F32,
                shape: [1, elements],
                offset,
                length: bytes,
                aux_offset: 0,
                aux_length: 0,
                rotated: false,
                delta_from: None,
            },
        );
    };

    add_ptq1(
        &mut payload,
        &mut tensors,
        "model.language_model.embed_tokens.weight",
        VOCAB,
        WIDTH,
        false,
    );
    add_ptq1(&mut payload, &mut tensors, "lm_head.weight", VOCAB, WIDTH, true);
    add_f32(&mut payload, &mut tensors, "model.language_model.norm.weight", WIDTH, 1.0);

    for layer in 0..LAYERS {
        let p = format!("model.language_model.layers.{layer}");
        add_f32(&mut payload, &mut tensors, &format!("{p}.input_layernorm.weight"), WIDTH, 1.0);
        add_f32(
            &mut payload,
            &mut tensors,
            &format!("{p}.post_attention_layernorm.weight"),
            WIDTH,
            1.0,
        );
        add_int4(&mut payload, &mut tensors, &format!("{p}.mlp.gate_proj.weight"), 3584, WIDTH);
        add_int4(&mut payload, &mut tensors, &format!("{p}.mlp.up_proj.weight"), 3584, WIDTH);
        add_int4(&mut payload, &mut tensors, &format!("{p}.mlp.down_proj.weight"), WIDTH, 3584);
        if layer % 4 == 3 {
            add_int4(&mut payload, &mut tensors, &format!("{p}.self_attn.q_proj.weight"), 4096, WIDTH);
            add_int4(&mut payload, &mut tensors, &format!("{p}.self_attn.k_proj.weight"), 512, WIDTH);
            add_int4(&mut payload, &mut tensors, &format!("{p}.self_attn.v_proj.weight"), 512, WIDTH);
            add_int4(&mut payload, &mut tensors, &format!("{p}.self_attn.o_proj.weight"), WIDTH, 2048);
            add_f32(&mut payload, &mut tensors, &format!("{p}.self_attn.q_norm.weight"), 256, 1.0);
            add_f32(&mut payload, &mut tensors, &format!("{p}.self_attn.k_norm.weight"), 256, 1.0);
        } else {
            add_int4(&mut payload, &mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"), 6144, WIDTH);
            add_int4(&mut payload, &mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"), 2048, WIDTH);
            add_bf16(&mut payload, &mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), 16, WIDTH);
            add_bf16(&mut payload, &mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), 16, WIDTH);
            add_int4(&mut payload, &mut tensors, &format!("{p}.linear_attn.out_proj.weight"), WIDTH, 2048);
            add_f32(&mut payload, &mut tensors, &format!("{p}.linear_attn.conv1d.weight"), 6144 * 4, 0.01);
            add_f32(&mut payload, &mut tensors, &format!("{p}.linear_attn.A_log"), 16, -2.0);
            add_f32(&mut payload, &mut tensors, &format!("{p}.linear_attn.dt_bias"), 16, -1.0);
            add_f32(&mut payload, &mut tensors, &format!("{p}.linear_attn.norm.weight"), 128, 1.0);
        }
    }
    let rem = payload.len() % METAL_PAGE;
    if rem != 0 {
        payload.resize(payload.len() + (METAL_PAGE - rem), 0);
    }
    let payload_len = payload.len();
    let mut header = ArtifactHeader {
        format: "minicpm-v-4.7-1b".into(),
        version: 1,
        config: serde_json::json!({"hidden_size": WIDTH, "num_layers": LAYERS, "vocab_size": VOCAB}),
        tensors,
        payload_length: payload_len as u64,
        payload_sha256: String::new(),
        compression: None,
        zstd_dict_length: 0,
        zstd_frame_bytes: 0,
        zstd_frames: Vec::new(),
    };
    if compress {
        header.payload_sha256 = hex(Sha256::digest(&payload).as_slice());
        // Level 3 keeps test artifact creation fast.
        let stream = encode_payload(&mut header, payload, 3)?;
        return write_artifact(path, LM_MAGIC, &header, &stream);
    }
    // Benchmarks only time kernels: write a sparse all-zero payload instead of the bytes.
    drop(payload);
    let header_bytes = serde_json::to_vec(&header)?;
    let payload_start = (16 + header_bytes.len()).div_ceil(METAL_PAGE) * METAL_PAGE;
    let mut file = File::create(path)?;
    file.write_all(&LM_MAGIC)?;
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    file.set_len((payload_start + payload_len) as u64)?;
    file.flush()?;
    Ok(payload_start + payload_len)
}
