//! Projection and rotation kernels (`shaders/projection.metal`):
//! - `Int4Matrix`: group-64 4-bit (`codes` + F16 `(scale, bias)`)
//! - `Int8Matrix`: per-row int8 (`weights` + F32 `scales`)
//! - `Ptq1Matrix`: ternary g128 (`PTQ1_0` 28-byte blocks)
//! - `Bf16Matrix`: raw BF16 (`ushort` rows)
//! - `SignedHadamard512`: 512-block signed Walsh-Hadamard transform
//!
//! INT4 GEMMs of `MPP_MIN_TOKENS`+ rows dequantize to a half scratch and run MPP `matmul2d`
//! (`shaders/mpp.metal`); smaller batches keep the int4 kernels.

use std::sync::OnceLock;

use half::f16;
use objc2_metal::{MTLResource as _, MTLSize};

use crate::Error;
use crate::batch::{CommandBatch, ComputeEncoder};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::ops::{MODEL_FFN, MODEL_WIDTH};
use crate::shaders::{Pipeline, ShaderLibrary};

pub const PTQ1_BLOCK_ELEMENTS: usize = 128;
pub const PTQ1_BLOCK_BYTES: usize = 28;
pub const INT4_GROUP: u32 = 64;
pub const HADAMARD_512: usize = 512;

#[derive(Clone, Copy)]
pub struct Bf16Matrix<'a> {
    pub buffer: &'a MetalBuffer,
    pub offset: usize,
    pub rows: u32,
    pub columns: u32,
}

#[derive(Clone, Copy)]
pub struct Int8Matrix<'a> {
    pub weights: &'a MetalBuffer,
    pub weights_offset: usize,
    pub scales: &'a MetalBuffer,
    pub scales_offset: usize,
    pub rows: u32,
    pub columns: u32,
}

#[derive(Clone, Copy)]
pub struct Int4Matrix<'a> {
    pub codes: &'a MetalBuffer,
    pub codes_offset: usize,
    pub groups: &'a MetalBuffer,
    pub groups_offset: usize,
    pub rows: u32,
    pub columns: u32,
}

#[derive(Clone, Copy)]
pub struct Ptq1Matrix<'a> {
    pub buffer: &'a MetalBuffer,
    pub offset: usize,
    pub rows: u32,
    pub columns: u32,
}

/// Any runtime projection matrix (4-bit, ternary PTQ1, per-row int8, or BF16).
#[derive(Clone, Copy)]
pub enum AnyMatrix<'a> {
    Int4(Int4Matrix<'a>),
    Ptq1(Ptq1Matrix<'a>),
    Int8(Int8Matrix<'a>),
    Bf16(Bf16Matrix<'a>),
}

impl AnyMatrix<'_> {
    #[must_use]
    pub const fn rows(self) -> u32 {
        match self {
            Self::Int4(m) => m.rows,
            Self::Ptq1(m) => m.rows,
            Self::Int8(m) => m.rows,
            Self::Bf16(m) => m.rows,
        }
    }

    #[must_use]
    pub const fn columns(self) -> u32 {
        match self {
            Self::Int4(m) => m.columns,
            Self::Ptq1(m) => m.columns,
            Self::Int8(m) => m.columns,
            Self::Bf16(m) => m.columns,
        }
    }

    /// Return a zero-copy view of `row_count` rows starting at `first_row`.
    #[must_use]
    pub const fn slice_rows(self, first_row: u32, row_count: u32) -> Self {
        let r = first_row as usize;
        match self {
            Self::Int4(m) => {
                let row_groups = (m.columns as usize).div_ceil(64);
                Self::Int4(Int4Matrix {
                    codes: m.codes,
                    codes_offset: m.codes_offset + r * row_groups * 32,
                    groups: m.groups,
                    groups_offset: m.groups_offset + r * row_groups * 4,
                    rows: row_count,
                    columns: m.columns,
                })
            }
            Self::Ptq1(m) => {
                let blocks = (m.columns as usize) / PTQ1_BLOCK_ELEMENTS;
                Self::Ptq1(Ptq1Matrix {
                    buffer: m.buffer,
                    offset: m.offset + r * blocks * PTQ1_BLOCK_BYTES,
                    rows: row_count,
                    columns: m.columns,
                })
            }
            Self::Int8(m) => Self::Int8(Int8Matrix {
                weights: m.weights,
                weights_offset: m.weights_offset + r * (m.columns as usize),
                scales: m.scales,
                scales_offset: m.scales_offset + r * 4,
                rows: row_count,
                columns: m.columns,
            }),
            Self::Bf16(m) => Self::Bf16(Bf16Matrix {
                buffer: m.buffer,
                offset: m.offset + r * (m.columns as usize) * 2,
                rows: row_count,
                columns: m.columns,
            }),
        }
    }
}

pub struct SignedHadamard512 {
    pub signs: MetalBuffer,
    pub columns: u32,
}

impl SignedHadamard512 {
    pub fn new(ctx: &MetalContext, signs: &[f32]) -> crate::Result<Self> {
        if signs.is_empty() || !signs.len().is_multiple_of(HADAMARD_512) {
            return Err(Error::InvalidArgument(
                "Hadamard512 signs must be a non-empty multiple of 512".into(),
            ));
        }
        Ok(Self {
            signs: MetalBuffer::from_slice(ctx.device(), signs)?,
            columns: signs.len() as u32,
        })
    }
}

pub fn decode_ptq1_row(packed: &[u8], output: &mut [f32]) -> crate::Result<()> {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    if output.is_empty()
        || !output.len().is_multiple_of(PTQ1_BLOCK_ELEMENTS)
        || packed.len() != (output.len() / PTQ1_BLOCK_ELEMENTS) * PTQ1_BLOCK_BYTES
    {
        return Err(Error::InvalidArgument("PTQ1_0 row size mismatch".into()));
    }
    for (block, values) in packed
        .as_chunks::<PTQ1_BLOCK_BYTES>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<PTQ1_BLOCK_ELEMENTS>().0)
    {
        let scale = f16::from_bits(u16::from_le_bytes([block[26], block[27]])).to_f32();
        for (element, value) in values.iter_mut().enumerate() {
            let (byte, power) = if element < 80 {
                (element % 16, element / 16)
            } else if element < 120 {
                (16 + (element - 80) % 8, (element - 80) / 8)
            } else {
                (24 + (element - 120) % 2, (element - 120) / 2)
            };
            let q = block[byte].wrapping_mul(POW3[power]);
            let trit = f32::from((u16::from(q) * 3) >> 8) - 1.0;
            *value = trit * scale;
        }
    }
    Ok(())
}

pub struct ProjectionKernels {
    int8_vec1: Pipeline,
    int8_vec2: Pipeline,
    int8_wide: Pipeline,
    int8_swiglu: Pipeline,
    int4_mv: [Pipeline; 2],
    int4_mv3: [Pipeline; 2],
    int4_mv_gdn: [Pipeline; 2],
    int4_mv_swiglu: [Pipeline; 2],
    int4_wide1: Pipeline,
    int4_wide2: Pipeline,
    int4_wide4: Pipeline,
    int4_wide_swiglu: Pipeline,
    int4_wide_concat3: Pipeline,
    int4_wide2_concat2: Pipeline,
    /// Threads per `int4_matmul_wide*` threadgroup: 32 per SIMD group sharing a tile's columns.
    wide_threads: usize,
    /// 8-row tiles per wide threadgroup for vocabulary-sized (`lm_head`) matrices.
    vocab_tiles: usize,
    ptq1_matvec: Pipeline,
    ptq1_swiglu: Pipeline,
    ptq1_vec2: Pipeline,
    ptq1_vec3: Pipeline,
    ptq1_vec4: Pipeline,
    ptq1_wide1: Pipeline,
    ptq1_wide2: Pipeline,
    fwht512_fwd: Pipeline,
    fwht512_inv: Pipeline,
    bf16_mv: Pipeline,
    bf16_pair: Pipeline,
    embed_bf16: Pipeline,
    embed_int8: Pipeline,
    embed_ptq1: Pipeline,
    embed_ptq1_fwht512: Pipeline,
    embed_int4: Pipeline,
    mpp_dequant: Pipeline,
    mpp_matmul: Pipeline,
    mpp_swiglu: Pipeline,
    /// Half weights for the MPP path, allocated on the first GEMM that takes it (decode never does).
    mpp_scratch: OnceLock<MetalBuffer>,
}

fn bind(encoder: &ComputeEncoder, buffer: &MetalBuffer, offset: usize, index: usize) {
    encoder.bind(buffer, offset, index);
}

fn set_val<T: Copy>(encoder: &ComputeEncoder, value: &T, index: usize) {
    encoder.set_val(value, index);
}

/// Threads per small-batch (`tokens <= 2`) threadgroup: 2 SIMD groups.
const MV_THREADS: usize = 64;
/// Rows per small-batch threadgroup: 4 rows per SIMD group.
const MV_TG_ROWS: u32 = 8;
/// Rows from which INT4 GEMMs dequantize to half and run MPP `matmul2d` (`shaders/mpp.metal`):
/// at 256 rows every LM shape ran 0.70-0.91x the wide int4 kernels' time on M4 Pro; at 128, k/v, o
/// and gate+up were slower.
const MPP_MIN_TOKENS: u32 = 256;
/// Output tile edge of the MPP kernels (`MPP_TILE` in `mpp.metal`), 4 SIMD groups each.
const MPP_TILE: usize = 64;
/// Half elements in the dequant scratch: the fused LM gate + up. Larger matrices (`lm_head`,
/// vision merger) keep the int4 kernels.
const MPP_SCRATCH_HALFS: usize = 2 * MODEL_FFN as usize * MODEL_WIDTH as usize;

fn dispatch_1d(encoder: &ComputeEncoder, groups: usize, threads: usize) {
    encoder.dispatch_threadgroups(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

/// `output[t, r] = input[t] . scratch[r]` (`mpp_matmul`) or its `SwiGLU` over gate rows `r` and up
/// rows `rows + r` (`mpp_matmul_swiglu`), for `[tokens, rows, columns]`.
fn mpp_dispatch(
    batch: &mut CommandBatch,
    pipeline: &Pipeline,
    scratch: &MetalBuffer,
    input: (&MetalBuffer, usize),
    output: (&MetalBuffer, usize),
    [tokens, rows, columns]: [u32; 3],
) {
    let enc = batch.encoder();
    enc.set_pipeline(pipeline);
    bind(enc, input.0, input.1, 0);
    bind(enc, scratch, 0, 1);
    bind(enc, output.0, output.1, 2);
    set_val(enc, &[tokens, rows, columns, 0], 3);
    enc.dispatch_threadgroups(
        MTLSize {
            width: (rows as usize).div_ceil(MPP_TILE),
            height: (tokens as usize).div_ceil(MPP_TILE),
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    batch.record_dispatch();
}

impl ProjectionKernels {
    pub fn new(shaders: &ShaderLibrary) -> crate::Result<Self> {
        let make = |name: &str| shaders.pipeline(name);
        // A-series (phone) GPUs have 4-6 cores. On an A15, 2 SIMD groups per wide threadgroup ran
        // verify blocks 1.3x faster than 8, whose cross-SIMD reduction dominated, and 4 tiles cut
        // `lm_head` 10% more. Other GPUs keep 8 and 2 (and their exact sums).
        let (wide, wide_split, vocab_tiles) = if shaders.device_name().starts_with("Apple A") {
            ("_s2", 2, 4)
        } else {
            ("", 8, 2)
        };
        let pair = |name: &str| -> crate::Result<[Pipeline; 2]> {
            Ok([make(&format!("{name}_1"))?, make(&format!("{name}_2"))?])
        };
        Ok(Self {
            int8_vec1: make("int8_matvec")?,
            int8_vec2: make("int8_matmul_2")?,
            int8_wide: make("int8_matmul_wide")?,
            int8_swiglu: make("int8_matvec_swiglu")?,
            int4_mv: pair("int4_mv")?,
            int4_mv3: pair("int4_mv3")?,
            int4_mv_gdn: pair("int4_mv_gdn")?,
            int4_mv_swiglu: pair("int4_mv_swiglu")?,
            int4_wide1: make(&format!("int4_matmul_wide{wide}"))?,
            int4_wide2: make(&format!("int4_matmul_wide_2{wide}"))?,
            int4_wide4: make(&format!("int4_matmul_wide_4{wide}"))?,
            int4_wide_swiglu: make(&format!("int4_matmul_wide_swiglu{wide}"))?,
            int4_wide_concat3: make(&format!("int4_matmul_wide_concat3{wide}"))?,
            int4_wide2_concat2: make(&format!("int4_matmul_wide_2_concat2{wide}"))?,
            wide_threads: 32 * wide_split,
            vocab_tiles,
            ptq1_matvec: make("ptq1_matvec")?,
            ptq1_swiglu: make("ptq1_matvec_swiglu")?,
            ptq1_vec2: make("ptq1_matmul_2")?,
            ptq1_vec3: make("ptq1_matmul_3")?,
            ptq1_vec4: make("ptq1_matmul_4")?,
            ptq1_wide1: make("ptq1_matmul_wide")?,
            ptq1_wide2: make("ptq1_matmul_wide_2")?,
            fwht512_fwd: make("fwht512_forward")?,
            fwht512_inv: make("fwht512_inverse")?,
            bf16_mv: make("bf16_mv")?,
            bf16_pair: make("bf16_mv_pair")?,
            embed_bf16: make("embed_gather_bf16")?,
            embed_int8: make("embed_gather_int8")?,
            embed_ptq1: make("embed_gather_ptq1")?,
            embed_ptq1_fwht512: make("embed_gather_ptq1_fwht512")?,
            embed_int4: make("embed_gather_int4")?,
            mpp_dequant: make("mpp_int4_dequant")?,
            mpp_matmul: make("mpp_matmul")?,
            mpp_swiglu: make("mpp_matmul_swiglu")?,
            mpp_scratch: OnceLock::new(),
        })
    }

    /// The dequant scratch when `tokens` rows against `columns`-wide INT4 weights of up to `halfs`
    /// elements per scratch fill take the MPP path; `None` keeps the int4 kernels.
    fn mpp_scratch(
        &self,
        batch: &CommandBatch,
        input: &MetalBuffer,
        columns: u32,
        halfs: usize,
        tokens: u32,
    ) -> Option<&MetalBuffer> {
        // One scratch serves every MPP GEMM: dequant -> matmul -> next dequant must not overlap.
        if tokens < MPP_MIN_TOKENS
            || !columns.is_multiple_of(INT4_GROUP)
            || halfs > MPP_SCRATCH_HALFS
            || batch.overlapping()
        {
            return None;
        }
        if let Some(scratch) = self.mpp_scratch.get() {
            return Some(scratch);
        }
        // Without the allocation the int4 kernels compute the same product.
        let scratch = MetalBuffer::empty(&input.raw().device(), MPP_SCRATCH_HALFS * 2).ok()?;
        Some(self.mpp_scratch.get_or_init(|| scratch))
    }

    /// Dequantize `matrix` to half rows at byte `offset` of `scratch`.
    fn mpp_dequant(
        &self,
        batch: &mut CommandBatch,
        matrix: Int4Matrix<'_>,
        scratch: &MetalBuffer,
        offset: usize,
    ) {
        let total = matrix.rows * (matrix.columns / 8);
        let enc = batch.encoder();
        enc.set_pipeline(&self.mpp_dequant);
        bind(enc, matrix.codes, matrix.codes_offset, 0);
        bind(enc, matrix.groups, matrix.groups_offset, 1);
        bind(enc, scratch, offset, 2);
        set_val(enc, &matrix.columns, 3);
        set_val(enc, &total, 4);
        dispatch_1d(enc, total.div_ceil(256) as usize, 256);
        batch.record_dispatch();
    }

    /// `output = input x matrix^T` for `tokens` rows through the MPP path.
    fn mpp_int4(
        &self,
        batch: &mut CommandBatch,
        matrix: Int4Matrix<'_>,
        scratch: &MetalBuffer,
        input: (&MetalBuffer, usize),
        output: (&MetalBuffer, usize),
        tokens: u32,
    ) {
        self.mpp_dequant(batch, matrix, scratch, 0);
        mpp_dispatch(
            batch,
            &self.mpp_matmul,
            scratch,
            input,
            output,
            [tokens, matrix.rows, matrix.columns],
        );
    }

    pub fn matmul(
        &self,
        batch: &mut CommandBatch,
        matrix: AnyMatrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        match matrix {
            AnyMatrix::Int4(m) => self.int4_matmul(batch, m, input, output, tokens),
            AnyMatrix::Ptq1(m) => self.ptq1_matmul(batch, m, input, output, tokens),
            AnyMatrix::Int8(m) => self.int8_matmul(batch, m, input, output, tokens),
            AnyMatrix::Bf16(m) => self.bf16_matmul(batch, m, input, output, tokens),
        }
    }

    pub fn matmul_at(
        &self,
        batch: &mut CommandBatch,
        matrix: AnyMatrix<'_>,
        input: &MetalBuffer,
        input_offset: usize,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        match matrix {
            AnyMatrix::Int4(m) => {
                self.int4_matmul_at(batch, m, input, input_offset, output, tokens)
            }
            AnyMatrix::Ptq1(m) => {
                self.ptq1_matmul_at(batch, m, input, input_offset, output, tokens)
            }
            AnyMatrix::Int8(_) | AnyMatrix::Bf16(_) => {
                self.matmul(batch, matrix, input, output, tokens)
            }
        }
    }

    /// Small-batch (`tokens <= 2`) projection of up to three INT4 matrices (rows multiple of 4)
    /// sharing `input`, each written to `(output, byte offset)`.
    fn int4_mv3(
        &self,
        batch: &mut CommandBatch,
        mats: &[(Int4Matrix<'_>, &MetalBuffer, usize)],
        input: &MetalBuffer,
        tokens: u32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.int4_mv3[tokens as usize - 1]);
        let mut dims = [0_u32, 0, 0, mats[0].0.columns];
        for slot in 0..3 {
            // Unused slots repeat matrix 0 so every argument is bound; `dims` keeps them empty.
            let (m, out, out_offset) = mats.get(slot).copied().unwrap_or(mats[0]);
            bind(enc, m.codes, m.codes_offset, 2 * slot);
            bind(enc, m.groups, m.groups_offset, 2 * slot + 1);
            bind(enc, out, out_offset, 7 + slot);
            if slot < mats.len() {
                dims[slot] = m.rows;
            }
        }
        bind(enc, input, 0, 6);
        set_val(enc, &dims, 10);
        set_val(enc, &tokens, 11);
        dispatch_1d(
            enc,
            (dims[0] + dims[1] + dims[2]).div_ceil(MV_TG_ROWS) as usize,
            MV_THREADS,
        );
        batch.record_dispatch();
    }

    pub fn int4_matmul_concat2(
        &self,
        batch: &mut CommandBatch,
        m0: Int4Matrix<'_>,
        m1: Int4Matrix<'_>,
        input: &MetalBuffer,
        out0: &MetalBuffer,
        out0_offset: usize,
        out1: &MetalBuffer,
        out1_offset: usize,
        tokens: u32,
    ) {
        if tokens <= 2 {
            self.int4_mv3(
                batch,
                &[(m0, out0, out0_offset), (m1, out1, out1_offset)],
                input,
                tokens,
            );
            return;
        }
        let halfs = m0.rows.max(m1.rows) as usize * m0.columns as usize;
        if let Some(scratch) = self.mpp_scratch(batch, input, m0.columns, halfs, tokens) {
            self.mpp_int4(batch, m0, scratch, (input, 0), (out0, out0_offset), tokens);
            self.mpp_int4(batch, m1, scratch, (input, 0), (out1, out1_offset), tokens);
            return;
        }
        let enc = batch.encoder();
        enc.set_pipeline(&self.int4_wide2_concat2);
        bind(enc, m0.codes, m0.codes_offset, 0);
        bind(enc, m0.groups, m0.groups_offset, 1);
        bind(enc, m1.codes, m1.codes_offset, 2);
        bind(enc, m1.groups, m1.groups_offset, 3);
        bind(enc, input, 0, 4);
        bind(enc, out0, out0_offset, 5);
        bind(enc, out1, out1_offset, 6);
        set_val(enc, &m0.rows, 7);
        set_val(enc, &m1.rows, 8);
        set_val(enc, &m0.columns, 9);
        set_val(enc, &tokens, 10);
        enc.dispatch_threadgroups(
            MTLSize {
                width: ((m0.rows + m1.rows) as usize) / 16,
                height: (tokens as usize).div_ceil(8),
                depth: 1,
            },
            MTLSize {
                width: self.wide_threads,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    pub fn int4_matmul_concat3(
        &self,
        batch: &mut CommandBatch,
        m0: Int4Matrix<'_>,
        m1: Int4Matrix<'_>,
        m2: Int4Matrix<'_>,
        input: &MetalBuffer,
        out0: &MetalBuffer,
        out1: &MetalBuffer,
        out2: &MetalBuffer,
        tokens: u32,
    ) {
        if tokens <= 2 {
            self.int4_mv3(
                batch,
                &[(m0, out0, 0), (m1, out1, 0), (m2, out2, 0)],
                input,
                tokens,
            );
            return;
        }
        let halfs = m0.rows.max(m1.rows).max(m2.rows) as usize * m0.columns as usize;
        if let Some(scratch) = self.mpp_scratch(batch, input, m0.columns, halfs, tokens) {
            for (matrix, output) in [(m0, out0), (m1, out1), (m2, out2)] {
                self.mpp_int4(batch, matrix, scratch, (input, 0), (output, 0), tokens);
            }
            return;
        }
        let enc = batch.encoder();
        enc.set_pipeline(&self.int4_wide_concat3);
        bind(enc, m0.codes, m0.codes_offset, 0);
        bind(enc, m0.groups, m0.groups_offset, 1);
        bind(enc, m1.codes, m1.codes_offset, 2);
        bind(enc, m1.groups, m1.groups_offset, 3);
        bind(enc, m2.codes, m2.codes_offset, 4);
        bind(enc, m2.groups, m2.groups_offset, 5);
        bind(enc, input, 0, 6);
        bind(enc, out0, 0, 7);
        bind(enc, out1, 0, 8);
        bind(enc, out2, 0, 9);
        set_val(enc, &m0.rows, 10);
        set_val(enc, &m1.rows, 11);
        set_val(enc, &m2.rows, 12);
        set_val(enc, &m0.columns, 13);
        set_val(enc, &tokens, 14);
        enc.dispatch_threadgroups(
            MTLSize {
                width: ((m0.rows + m1.rows + m2.rows) as usize) / 8,
                height: (tokens as usize).div_ceil(8),
                depth: 1,
            },
            MTLSize {
                width: self.wide_threads,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    pub fn int4_matmul_gdn(
        &self,
        batch: &mut CommandBatch,
        qkv: Int4Matrix<'_>,
        z: Int4Matrix<'_>,
        a: Bf16Matrix<'_>,
        b: Bf16Matrix<'_>,
        input: &MetalBuffer,
        out_qkv: &MetalBuffer,
        out_z: &MetalBuffer,
        out_a: &MetalBuffer,
        out_b: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens <= 2 {
            // Fused BF16 `a`/`b` (rows multiple of 4) + INT4 `qkv` + `z` in one dispatch.
            let enc = batch.encoder();
            enc.set_pipeline(&self.int4_mv_gdn[tokens as usize - 1]);
            bind(enc, qkv.codes, qkv.codes_offset, 0);
            bind(enc, qkv.groups, qkv.groups_offset, 1);
            bind(enc, z.codes, z.codes_offset, 2);
            bind(enc, z.groups, z.groups_offset, 3);
            bind(enc, a.buffer, a.offset, 4);
            bind(enc, b.buffer, b.offset, 5);
            bind(enc, input, 0, 6);
            bind(enc, out_qkv, 0, 7);
            bind(enc, out_z, 0, 8);
            bind(enc, out_a, 0, 9);
            bind(enc, out_b, 0, 10);
            set_val(enc, &[qkv.rows, z.rows, a.rows, qkv.columns], 11);
            set_val(enc, &tokens, 12);
            let rows = 2 * a.rows + qkv.rows + z.rows;
            dispatch_1d(enc, rows.div_ceil(MV_TG_ROWS) as usize, MV_THREADS);
            batch.record_dispatch();
            return Ok(());
        }
        let halfs = qkv.rows.max(z.rows) as usize * qkv.columns as usize;
        if let Some(scratch) = self.mpp_scratch(batch, input, qkv.columns, halfs, tokens) {
            self.mpp_int4(batch, qkv, scratch, (input, 0), (out_qkv, 0), tokens);
            self.mpp_int4(batch, z, scratch, (input, 0), (out_z, 0), tokens);
            self.bf16_matmul_pair(batch, a, b, input, out_a, out_b, tokens);
            return Ok(());
        }
        batch.independent(|batch| {
            let enc = batch.encoder();
            enc.set_pipeline(&self.int4_wide2_concat2);
            bind(enc, qkv.codes, qkv.codes_offset, 0);
            bind(enc, qkv.groups, qkv.groups_offset, 1);
            bind(enc, z.codes, z.codes_offset, 2);
            bind(enc, z.groups, z.groups_offset, 3);
            bind(enc, input, 0, 4);
            bind(enc, out_qkv, 0, 5);
            bind(enc, out_z, 0, 6);
            set_val(enc, &qkv.rows, 7);
            set_val(enc, &z.rows, 8);
            set_val(enc, &qkv.columns, 9);
            set_val(enc, &tokens, 10);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: ((qkv.rows + z.rows) as usize) / 16,
                    height: (tokens as usize).div_ceil(8),
                    depth: 1,
                },
                MTLSize {
                    width: self.wide_threads,
                    height: 1,
                    depth: 1,
                },
            );
            batch.record_dispatch();
            self.bf16_matmul_pair(batch, a, b, input, out_a, out_b, tokens);
            Ok(())
        })
    }

    pub fn bf16_matmul_pair(
        &self,
        batch: &mut CommandBatch,
        a: Bf16Matrix<'_>,
        b: Bf16Matrix<'_>,
        input: &MetalBuffer,
        out_a: &MetalBuffer,
        out_b: &MetalBuffer,
        tokens: u32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.bf16_pair);
        bind(enc, a.buffer, a.offset, 0);
        bind(enc, b.buffer, b.offset, 1);
        bind(enc, input, 0, 2);
        bind(enc, out_a, 0, 3);
        bind(enc, out_b, 0, 4);
        set_val(enc, &a.rows, 5);
        set_val(enc, &a.columns, 6);
        set_val(enc, &tokens, 7);
        enc.dispatch_threadgroups(
            MTLSize {
                width: (a.rows as usize) / 2,
                height: tokens as usize,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    pub fn matvec_swiglu(
        &self,
        batch: &mut CommandBatch,
        gate: AnyMatrix<'_>,
        up: AnyMatrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> crate::Result<()> {
        match (gate, up) {
            (AnyMatrix::Int4(g), AnyMatrix::Int4(u)) => {
                self.int4_mv_swiglu(batch, g, u, input, output, 1);
                Ok(())
            }
            (AnyMatrix::Ptq1(g), AnyMatrix::Ptq1(u)) => {
                let enc = batch.encoder();
                enc.set_pipeline(&self.ptq1_swiglu);
                bind(enc, g.buffer, g.offset, 0);
                bind(enc, input, 0, 1);
                bind(enc, output, 0, 2);
                set_val(enc, &g.rows, 3);
                set_val(enc, &g.columns, 4);
                bind(enc, u.buffer, u.offset, 5);
                dispatch_1d(enc, (g.rows as usize).div_ceil(4), 32);
                batch.record_dispatch();
                Ok(())
            }
            (AnyMatrix::Int8(g), AnyMatrix::Int8(u)) => {
                let enc = batch.encoder();
                enc.set_pipeline(&self.int8_swiglu);
                bind(enc, g.weights, g.weights_offset, 0);
                bind(enc, g.scales, g.scales_offset, 1);
                bind(enc, u.weights, u.weights_offset, 2);
                bind(enc, u.scales, u.scales_offset, 3);
                bind(enc, input, 0, 4);
                bind(enc, output, 0, 5);
                set_val(enc, &g.rows, 6);
                set_val(enc, &g.columns, 7);
                dispatch_1d(enc, (g.rows as usize).div_ceil(2), 32);
                batch.record_dispatch();
                Ok(())
            }
            _ => Err(Error::InvalidArgument(
                "SwiGLU requires matching gate/up formats".into(),
            )),
        }
    }

    /// Small-batch (`tokens <= 2`) fused `silu(gate) * up` over INT4 gate/up rows.
    fn int4_mv_swiglu(
        &self,
        batch: &mut CommandBatch,
        gate: Int4Matrix<'_>,
        up: Int4Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.int4_mv_swiglu[tokens as usize - 1]);
        bind(enc, gate.codes, gate.codes_offset, 0);
        bind(enc, gate.groups, gate.groups_offset, 1);
        bind(enc, up.codes, up.codes_offset, 2);
        bind(enc, up.groups, up.groups_offset, 3);
        bind(enc, input, 0, 4);
        bind(enc, output, 0, 5);
        set_val(enc, &gate.rows, 6);
        set_val(enc, &gate.columns, 7);
        set_val(enc, &tokens, 8);
        // 2 gate + 2 up rows per SIMD group.
        dispatch_1d(enc, gate.rows.div_ceil(MV_TG_ROWS / 2) as usize, MV_THREADS);
        batch.record_dispatch();
    }

    pub fn matmul_swiglu(
        &self,
        batch: &mut CommandBatch,
        gate: AnyMatrix<'_>,
        up: AnyMatrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Err(Error::InvalidArgument("matmul_swiglu tokens == 0".into()));
        }
        if tokens == 1 {
            return self.matvec_swiglu(batch, gate, up, input, output);
        }
        if let (AnyMatrix::Int4(g), AnyMatrix::Int4(u)) = (gate, up)
            && g.rows == u.rows
            && g.columns == u.columns
            && let Some(scratch) = self.mpp_scratch(
                batch,
                input,
                g.columns,
                2 * g.rows as usize * g.columns as usize,
                tokens,
            )
        {
            // Gate rows, then up rows, in one scratch; one fused MPP GEMM + SwiGLU.
            self.mpp_dequant(batch, g, scratch, 0);
            self.mpp_dequant(batch, u, scratch, g.rows as usize * g.columns as usize * 2);
            mpp_dispatch(
                batch,
                &self.mpp_swiglu,
                scratch,
                (input, 0),
                (output, 0),
                [tokens, g.rows, g.columns],
            );
            return Ok(());
        }
        match (gate, up) {
            (AnyMatrix::Int4(g), AnyMatrix::Int4(u)) if tokens == 2 => {
                self.int4_mv_swiglu(batch, g, u, input, output, tokens);
                Ok(())
            }
            (AnyMatrix::Int4(g), AnyMatrix::Int4(u)) if g.columns.is_multiple_of(128) => {
                let enc = batch.encoder();
                enc.set_pipeline(&self.int4_wide_swiglu);
                bind(enc, g.codes, g.codes_offset, 0);
                bind(enc, g.groups, g.groups_offset, 1);
                bind(enc, u.codes, u.codes_offset, 2);
                bind(enc, u.groups, u.groups_offset, 3);
                bind(enc, input, 0, 4);
                bind(enc, output, 0, 5);
                set_val(enc, &g.rows, 6);
                set_val(enc, &g.columns, 7);
                set_val(enc, &tokens, 8);
                enc.dispatch_threadgroups(
                    MTLSize {
                        width: (g.rows as usize).div_ceil(8),
                        height: (tokens as usize).div_ceil(8),
                        depth: 1,
                    },
                    MTLSize {
                        width: self.wide_threads,
                        height: 1,
                        depth: 1,
                    },
                );
                batch.record_dispatch();
                Ok(())
            }
            _ => Err(Error::InvalidArgument(
                "Multi-row SwiGLU requires INT4 gate/up with columns % 128 == 0".into(),
            )),
        }
    }

    pub fn int4_matmul(
        &self,
        batch: &mut CommandBatch,
        matrix: Int4Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.int4_matmul_at(batch, matrix, input, 0, output, tokens)
    }

    pub fn int4_matmul_at(
        &self,
        batch: &mut CommandBatch,
        matrix: Int4Matrix<'_>,
        input: &MetalBuffer,
        input_offset: usize,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Err(Error::InvalidArgument("int4_matmul tokens == 0".into()));
        }
        let halfs = matrix.rows as usize * matrix.columns as usize;
        if let Some(scratch) = self.mpp_scratch(batch, input, matrix.columns, halfs, tokens) {
            self.mpp_int4(
                batch,
                matrix,
                scratch,
                (input, input_offset),
                (output, 0),
                tokens,
            );
            return Ok(());
        }
        if tokens <= 2 {
            let enc = batch.encoder();
            enc.set_pipeline(&self.int4_mv[tokens as usize - 1]);
            bind(enc, matrix.codes, matrix.codes_offset, 0);
            bind(enc, matrix.groups, matrix.groups_offset, 1);
            bind(enc, input, input_offset, 2);
            bind(enc, output, 0, 3);
            set_val(enc, &matrix.rows, 4);
            set_val(enc, &matrix.columns, 5);
            set_val(enc, &tokens, 6);
            dispatch_1d(enc, matrix.rows.div_ceil(MV_TG_ROWS) as usize, MV_THREADS);
        } else {
            let tiles = match matrix.rows {
                65_536.. => self.vocab_tiles,
                4096.. => 2,
                _ => 1,
            };
            let enc = batch.encoder();
            enc.set_pipeline(match tiles {
                4 => &self.int4_wide4,
                2 => &self.int4_wide2,
                _ => &self.int4_wide1,
            });
            bind(enc, matrix.codes, matrix.codes_offset, 0);
            bind(enc, matrix.groups, matrix.groups_offset, 1);
            bind(enc, input, input_offset, 2);
            bind(enc, output, 0, 3);
            set_val(enc, &matrix.rows, 4);
            set_val(enc, &matrix.columns, 5);
            set_val(enc, &tokens, 6);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: (matrix.rows as usize).div_ceil(8 * tiles),
                    height: (tokens as usize).div_ceil(8),
                    depth: 1,
                },
                MTLSize {
                    width: self.wide_threads,
                    height: 1,
                    depth: 1,
                },
            );
        }
        batch.record_dispatch();
        Ok(())
    }

    pub fn int8_matmul(
        &self,
        batch: &mut CommandBatch,
        matrix: Int8Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Err(Error::InvalidArgument("int8_matmul tokens == 0".into()));
        }
        let chunks = tokens.div_ceil(8);
        let mut start = 0;
        for chunk in 0..chunks {
            let end = (tokens * (chunk + 1)).div_ceil(chunks);
            let count = end - start;
            let wide = count > 2;
            let enc = batch.encoder();
            enc.set_pipeline(if wide {
                &self.int8_wide
            } else if count == 2 {
                &self.int8_vec2
            } else {
                &self.int8_vec1
            });
            bind(enc, matrix.weights, matrix.weights_offset, 0);
            bind(enc, matrix.scales, matrix.scales_offset, 1);
            bind(enc, input, (start * matrix.columns) as usize * 4, 2);
            bind(enc, output, (start * matrix.rows) as usize * 4, 3);
            set_val(enc, &matrix.rows, 4);
            set_val(enc, &matrix.columns, 5);
            if wide {
                set_val(enc, &count, 6);
                dispatch_1d(enc, (matrix.rows as usize).div_ceil(8), 256);
            } else {
                dispatch_1d(enc, (matrix.rows as usize).div_ceil(4), 32);
            }
            batch.record_dispatch();
            start = end;
        }
        Ok(())
    }

    pub fn ptq1_matmul(
        &self,
        batch: &mut CommandBatch,
        matrix: Ptq1Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.ptq1_matmul_at(batch, matrix, input, 0, output, tokens)
    }

    pub fn ptq1_matmul_at(
        &self,
        batch: &mut CommandBatch,
        matrix: Ptq1Matrix<'_>,
        input: &MetalBuffer,
        input_offset: usize,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Err(Error::InvalidArgument("ptq1_matmul tokens == 0".into()));
        }
        if tokens == 1 {
            let enc = batch.encoder();
            enc.set_pipeline(&self.ptq1_matvec);
            bind(enc, matrix.buffer, matrix.offset, 0);
            bind(enc, input, input_offset, 1);
            bind(enc, output, 0, 2);
            set_val(enc, &matrix.rows, 3);
            set_val(enc, &matrix.columns, 4);
            dispatch_1d(enc, (matrix.rows as usize).div_ceil(8), 32);
        } else if (2..=4).contains(&tokens) {
            let enc = batch.encoder();
            enc.set_pipeline(match tokens {
                2 => &self.ptq1_vec2,
                3 => &self.ptq1_vec3,
                _ => &self.ptq1_vec4,
            });
            bind(enc, matrix.buffer, matrix.offset, 0);
            bind(enc, input, input_offset, 1);
            bind(enc, output, 0, 2);
            set_val(enc, &matrix.rows, 3);
            set_val(enc, &matrix.columns, 4);
            dispatch_1d(enc, (matrix.rows as usize).div_ceil(8), 32);
        } else {
            let tiles = if matrix.rows >= 2048 { 2 } else { 1 };
            let enc = batch.encoder();
            enc.set_pipeline(if tiles == 2 {
                &self.ptq1_wide2
            } else {
                &self.ptq1_wide1
            });
            bind(enc, matrix.buffer, matrix.offset, 0);
            bind(enc, input, input_offset, 1);
            bind(enc, output, 0, 2);
            set_val(enc, &matrix.rows, 3);
            set_val(enc, &matrix.columns, 4);
            set_val(enc, &tokens, 5);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: (matrix.rows as usize).div_ceil(8 * tiles),
                    height: (tokens as usize).div_ceil(8),
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        batch.record_dispatch();
        Ok(())
    }

    pub fn bf16_matmul(
        &self,
        batch: &mut CommandBatch,
        matrix: Bf16Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Err(Error::InvalidArgument("bf16_matmul tokens == 0".into()));
        }
        let enc = batch.encoder();
        enc.set_pipeline(&self.bf16_mv);
        bind(enc, matrix.buffer, matrix.offset, 0);
        bind(enc, input, 0, 1);
        bind(enc, output, 0, 2);
        set_val(enc, &matrix.rows, 3);
        set_val(enc, &matrix.columns, 4);
        dispatch_1d(enc, (matrix.rows * tokens) as usize, 32);
        batch.record_dispatch();
        Ok(())
    }

    pub fn rotate512(
        &self,
        batch: &mut CommandBatch,
        rotation: &SignedHadamard512,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
        inverse: bool,
    ) -> crate::Result<()> {
        let blocks_per_row = rotation.columns / HADAMARD_512 as u32;
        let total_blocks = (blocks_per_row * tokens) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(if inverse {
            &self.fwht512_inv
        } else {
            &self.fwht512_fwd
        });
        bind(enc, input, 0, 0);
        bind(enc, &rotation.signs, 0, 1);
        bind(enc, output, 0, 2);
        set_val(enc, &blocks_per_row, 3);
        dispatch_1d(enc, total_blocks, 64);
        batch.record_dispatch();
        Ok(())
    }

    pub fn embed_gather_bf16(
        &self,
        batch: &mut CommandBatch,
        table: Bf16Matrix<'_>,
        token_ids: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) {
        let total = (tokens * table.columns) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.embed_bf16);
        bind(enc, table.buffer, table.offset, 0);
        bind(enc, token_ids, 0, 1);
        bind(enc, output, 0, 2);
        set_val(enc, &table.columns, 3);
        set_val(enc, &table.rows, 4);
        set_val(enc, &tokens, 5);
        dispatch_1d(enc, total.div_ceil(256), 256);
        batch.record_dispatch();
    }

    pub fn embed_gather_int8(
        &self,
        batch: &mut CommandBatch,
        table: Int8Matrix<'_>,
        token_ids: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) {
        let total = (tokens * table.columns) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.embed_int8);
        bind(enc, table.weights, table.weights_offset, 0);
        bind(enc, table.scales, table.scales_offset, 1);
        bind(enc, token_ids, 0, 2);
        bind(enc, output, 0, 3);
        set_val(enc, &table.columns, 4);
        set_val(enc, &table.rows, 5);
        set_val(enc, &tokens, 6);
        dispatch_1d(enc, total.div_ceil(256), 256);
        batch.record_dispatch();
    }

    pub fn embed_gather_ptq1(
        &self,
        batch: &mut CommandBatch,
        table: Ptq1Matrix<'_>,
        token_ids: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) {
        let total = (tokens * table.columns) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.embed_ptq1);
        bind(enc, table.buffer, table.offset, 0);
        bind(enc, token_ids, 0, 1);
        bind(enc, output, 0, 2);
        set_val(enc, &table.columns, 3);
        set_val(enc, &table.rows, 4);
        set_val(enc, &tokens, 5);
        dispatch_1d(enc, total.div_ceil(256), 256);
        batch.record_dispatch();
    }

    pub fn embed_gather_ptq1_fwht512(
        &self,
        batch: &mut CommandBatch,
        table: Ptq1Matrix<'_>,
        token_ids: &MetalBuffer,
        rotation: &SignedHadamard512,
        output: &MetalBuffer,
        tokens: u32,
    ) {
        let blocks_per_row = table.columns / HADAMARD_512 as u32;
        let total_blocks = (tokens * blocks_per_row) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.embed_ptq1_fwht512);
        bind(enc, table.buffer, table.offset, 0);
        bind(enc, token_ids, 0, 1);
        bind(enc, &rotation.signs, 0, 2);
        bind(enc, output, 0, 3);
        set_val(enc, &blocks_per_row, 4);
        set_val(enc, &table.rows, 5);
        dispatch_1d(enc, total_blocks, 64);
        batch.record_dispatch();
    }

    pub fn embed_gather_int4(
        &self,
        batch: &mut CommandBatch,
        table: Int4Matrix<'_>,
        token_ids: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) {
        let total = (tokens * table.columns) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.embed_int4);
        bind(enc, table.codes, table.codes_offset, 0);
        bind(enc, table.groups, table.groups_offset, 1);
        bind(enc, token_ids, 0, 2);
        bind(enc, output, 0, 3);
        set_val(enc, &table.columns, 4);
        set_val(enc, &table.rows, 5);
        set_val(enc, &tokens, 6);
        dispatch_1d(enc, total.div_ceil(256), 256);
        batch.record_dispatch();
    }
}
