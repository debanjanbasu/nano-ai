//! Quantization and packing codec for `micro-ai`:
//! - `PTQ1_0` ternary g128 (L2-optimal or TWN in 512-block signed-Hadamard rotated basis)
//! - `INT4_G64` group-64 affine 4-bit (`codes` two per byte + `half2` `(scale, bias)`)
//! - `INT8_ROW` symmetric per-row int8
//! - Deterministic 512-block signed-Hadamard transform (FWHT)

use std::simd::f32x16;
use half::f16;
use micro_metal::projection::{HADAMARD_512, PTQ1_BLOCK_BYTES, PTQ1_BLOCK_ELEMENTS};

/// Deterministic ±1 signs for `columns` (a multiple of 512) using `SplitMix64` seeded with `columns`.
#[must_use]
pub fn deterministic_signs(columns: usize) -> Vec<f32> {
    let mut state = columns as u64 ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = Vec::with_capacity(columns);
    for _ in 0..columns {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.push(if (z & 1) == 0 { -1.0 } else { 1.0 });
    }
    out
}

/// In-place 512-point Fast Walsh-Hadamard Transform (unnormalized).
fn fwht_512_in_place(block: &mut [f32; HADAMARD_512]) {
    for chunk in block.as_chunks_mut::<16>().0 {
        let mut v = f32x16::from_slice(chunk).to_array();
        let mut len = 1;
        while len < 16 {
            for start in (0..16).step_by(2 * len) {
                for i in 0..len {
                    let a = v[start + i];
                    let b = v[start + len + i];
                    v[start + i] = a + b;
                    v[start + len + i] = a - b;
                }
            }
            len <<= 1;
        }
        *chunk = v;
    }
    let mut len = 16;
    while len < HADAMARD_512 {
        for start in (0..HADAMARD_512).step_by(2 * len) {
            for i in (0..len).step_by(16) {
                let a = f32x16::from_slice(&block[start + i..start + i + 16]);
                let b = f32x16::from_slice(&block[start + len + i..start + len + i + 16]);
                (a + b).copy_to_slice(&mut block[start + i..start + i + 16]);
                (a - b).copy_to_slice(&mut block[start + len + i..start + len + i + 16]);
            }
        }
        len <<= 1;
    }
}

/// Forward signed 512-block Hadamard transform of a row: `y_b = H_512 (s_b * x_b) / sqrt(512)`.
pub fn fwht512_forward_row(row: &mut [f32], signs: &[f32]) {
    let norm = f32x16::splat((HADAMARD_512 as f32).sqrt().recip());
    for (b, block) in row.as_chunks_mut::<HADAMARD_512>().0.iter_mut().enumerate() {
        let sb = &signs[b * HADAMARD_512..(b + 1) * HADAMARD_512];
        for i in (0..HADAMARD_512).step_by(16) {
            let x = f32x16::from_slice(&block[i..i + 16]);
            let s = f32x16::from_slice(&sb[i..i + 16]);
            (x * s * norm).copy_to_slice(&mut block[i..i + 16]);
        }
        fwht_512_in_place(block);
    }
}

/// Inverse signed 512-block Hadamard transform of a row: `x_b = s_b * (H_512 y_b) / sqrt(512)`.
pub fn fwht512_inverse_row(row: &mut [f32], signs: &[f32]) {
    let norm = f32x16::splat((HADAMARD_512 as f32).sqrt().recip());
    for (b, block) in row.as_chunks_mut::<HADAMARD_512>().0.iter_mut().enumerate() {
        let sb = &signs[b * HADAMARD_512..(b + 1) * HADAMARD_512];
        fwht_512_in_place(block);
        for i in (0..HADAMARD_512).step_by(16) {
            let x = f32x16::from_slice(&block[i..i + 16]);
            let s = f32x16::from_slice(&sb[i..i + 16]);
            (x * norm * s).copy_to_slice(&mut block[i..i + 16]);
        }
    }
}

/// Pack 128 ternary codes `{-1, 0, +1}` and an FP16 scale into one 28-byte `PTQ1_0` block.
#[must_use]
pub fn pack_ptq1_block(
    codes: &[i8; PTQ1_BLOCK_ELEMENTS],
    scale: f16,
) -> [u8; PTQ1_BLOCK_BYTES] {
    let trit = |index: usize| (i16::from(codes[index]) + 1) as u16;
    let byte = |q: u16| (q * 256).div_ceil(243) as u8;
    let mut block = [0_u8; PTQ1_BLOCK_BYTES];
    for (m, out) in block[..16].iter_mut().enumerate() {
        *out = byte((0..5).fold(0, |q, n| q * 3 + trit(m + n * 16)));
    }
    for (m, out) in block[16..24].iter_mut().enumerate() {
        *out = byte((0..5).fold(0, |q, n| q * 3 + trit(80 + m + n * 8)));
    }
    for (j, out) in block[24..26].iter_mut().enumerate() {
        *out = byte((0..4).fold(0, |q, m| q * 3 + trit(120 + j + m * 2)) * 3);
    }
    block[26..].copy_from_slice(&scale.to_bits().to_le_bytes());
    block
}

/// Quantize a 128-element group to L2-optimal ternary codes `{-1, 0, +1}` and FP16 scale.
#[must_use]
pub fn quantize_ptq1_group_opt(group: &[f32; PTQ1_BLOCK_ELEMENTS]) -> [u8; PTQ1_BLOCK_BYTES] {
    let mut indexed: [(f32, usize); PTQ1_BLOCK_ELEMENTS] =
        std::array::from_fn(|i| (group[i].abs(), i));
    indexed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let mut csum = 0.0_f64;
    let mut best_score = -1.0_f64;
    let mut best_k = 1_usize;
    for (idx, &(abs_val, _)) in indexed.iter().enumerate() {
        let k = idx + 1;
        csum += f64::from(abs_val);
        let score = (csum * csum) / (k as f64);
        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }
    let thresh = indexed[best_k - 1].0;
    let mut sum_kept = 0.0_f32;
    let mut count_kept = 0_usize;
    for &x in group {
        if x.abs() >= thresh && x != 0.0 {
            sum_kept += x.abs();
            count_kept += 1;
        }
    }
    let scale = if count_kept > 0 {
        f16::from_f32(sum_kept / (count_kept as f32))
    } else {
        f16::ZERO
    };
    let mut codes = [0_i8; PTQ1_BLOCK_ELEMENTS];
    if scale.to_f32() != 0.0 {
        for (i, &x) in group.iter().enumerate() {
            if x.abs() >= thresh {
                codes[i] = if x > 0.0 { 1 } else { -1 };
            }
        }
    }
    pack_ptq1_block(&codes, scale)
}


/// Quantize a matrix `[rows, columns]` into `PTQ1_0` blocks (optionally rotated by 512-block FWHT).
#[must_use]
pub fn quantize_matrix_ptq1(
    weights: &[f32],
    rows: usize,
    columns: usize,
    signs: Option<&[f32]>,
) -> Vec<u8> {
    let blocks_per_row = columns / PTQ1_BLOCK_ELEMENTS;
    let mut out = Vec::with_capacity(rows * blocks_per_row * PTQ1_BLOCK_BYTES);
    let mut row_buf = vec![0.0_f32; columns];
    for r in 0..rows {
        row_buf.copy_from_slice(&weights[r * columns..(r + 1) * columns]);
        if let Some(s) = signs {
            fwht512_forward_row(&mut row_buf, s);
        }
        for group in row_buf.as_chunks::<PTQ1_BLOCK_ELEMENTS>().0 {
            out.extend_from_slice(&quantize_ptq1_group_opt(group));
        }
    }
    out
}

/// Quantize a matrix `[rows, columns]` into group-64 4-bit `(codes, groups)` using L2-optimal clipping search.
#[must_use]
pub fn quantize_matrix_int4_g64(weights: &[f32], rows: usize, columns: usize) -> (Vec<u8>, Vec<u8>) {
    const CLIPS: [f32; 5] = [1.0, 0.98, 0.96, 0.94, 0.92];
    let num_groups = columns.div_ceil(64);
    let mut codes = Vec::with_capacity(rows * num_groups * 32);
    let mut groups = Vec::with_capacity(rows * num_groups * 4);
    for r in 0..rows {
        let row = &weights[r * columns..(r + 1) * columns];
        for g_idx in 0..num_groups {
            let start = g_idx * 64;
            let end = (start + 64).min(columns);
            let grp = &row[start..end];
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &val in grp {
                mn = mn.min(val);
                mx = mx.max(val);
            }
            if grp.len() < 64 {
                mn = mn.min(0.0);
                mx = mx.max(0.0);
            }
            let mid = f32::midpoint(mn, mx);
            let half_span = 0.5 * (mx - mn);
            let mut best_err = f32::INFINITY;
            let mut best_scale = f16::ZERO;
            let mut best_bias = f16::from_f32(mn);
            for &clip in &CLIPS {
                let c_mn = clip.mul_add(-half_span, mid);
                let c_mx = clip.mul_add(half_span, mid);
                let b_f16 = f16::from_f32(c_mn);
                let s_f16 = f16::from_f32((c_mx - b_f16.to_f32()) / 15.0);
                let sc = s_f16.to_f32();
                let bs = b_f16.to_f32();
                let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
                let mut err = 0.0_f32;
                for &val in grp {
                    let q_val = ((val - bs) * inv).round().clamp(0.0, 15.0);
                    let diff = val - q_val.mul_add(sc, bs);
                    err = diff.mul_add(diff, err);
                }
                if err < best_err {
                    best_err = err;
                    best_scale = s_f16;
                    best_bias = b_f16;
                }
            }
            let scale = best_scale.to_f32();
            let bias = best_bias.to_f32();
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            groups.extend_from_slice(&best_scale.to_bits().to_le_bytes());
            groups.extend_from_slice(&best_bias.to_bits().to_le_bytes());
            for p in 0..32 {
                let v0 = grp.get(2 * p).copied().unwrap_or(0.0);
                let v1 = grp.get(2 * p + 1).copied().unwrap_or(0.0);
                let c0 = ((v0 - bias) * inv).round().clamp(0.0, 15.0) as u8;
                let c1 = ((v1 - bias) * inv).round().clamp(0.0, 15.0) as u8;
                codes.push(c0 | (c1 << 4));
            }
        }
    }
    (codes, groups)
}
