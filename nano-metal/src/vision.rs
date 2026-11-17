//! Vision tower kernels (`shaders/vision.metal`) for MiniCPM-V 4.7 1B:
//! `SigLIP2` 27-layer `ViT` + Intra-`ViT` 2x2 Window-Attention Merger + 2x2 `DownsampleMLP` Merger.

use objc2_metal::MTLSize;

use crate::batch::{CommandBatch, ComputeEncoder};
use crate::buffer::MetalBuffer;
use crate::projection::Int4Matrix;
use crate::shaders::{Pipeline, ShaderLibrary};

pub const VISION_WIDTH: u32 = 1152;
pub const VISION_HEADS: u32 = 16;
pub const VISION_HEAD_DIM: u32 = 72;

/// Dynamic threadgroup memory of `bv_attention_flash` (`BV_KS_ROWS` (35) x `BV_LD` (76) floats).
const ATTENTION_KS_BYTES: usize = 35 * 76 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    Store = 0,
    GeluTanh = 1,
    GeluErf = 2,
    Accumulate = 3,
}

#[derive(Clone, Copy)]
pub struct Linear<'a> {
    pub f32: bool,
    pub weights: &'a MetalBuffer,
    pub weights_offset: usize,
    pub bias: &'a MetalBuffer,
    pub bias_offset: usize,
    pub rows: u32,
    pub columns: u32,
}

pub struct VisionKernels {
    gemm: Pipeline,
    gemm_f32: Pipeline,
    int4_gemm_epilogue: Pipeline,
    int4_qkv_bias: Pipeline,
    layernorm: Pipeline,
    attention_flash: Pipeline,
    attention: Pipeline,
    window_gather: Pipeline,
    window_mean: Pipeline,
    bias_epilogue: Pipeline,
    add_positions_bf16: Pipeline,
}

fn bind(encoder: &ComputeEncoder, buffer: &MetalBuffer, offset: usize, index: usize) {
    encoder.bind(buffer, offset, index);
}

fn set_val<T: Copy>(encoder: &ComputeEncoder, value: &T, index: usize) {
    encoder.set_val(value, index);
}

impl VisionKernels {
    pub fn new(shaders: &ShaderLibrary) -> crate::Result<Self> {
        let make = |name: &str| shaders.pipeline(name);
        Ok(Self {
            gemm: make("bv_gemm")?,
            gemm_f32: make("bv_gemm_f32")?,
            int4_gemm_epilogue: make("bv_int4_matmul_epilogue")?,
            int4_qkv_bias: make("bv_int4_matmul_qkv_bias")?,
            layernorm: make("bv_layernorm")?,
            attention_flash: make("bv_attention_flash")?,
            attention: make("bv_attention")?,
            window_gather: make("bv_window_gather")?,
            window_mean: make("bv_window_mean")?,
            bias_epilogue: make("bv_bias_epilogue")?,
            add_positions_bf16: make("bv_add_positions_bf16")?,
        })
    }

    pub fn gemm(
        &self,
        batch: &mut CommandBatch,
        linear: Linear<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
        epilogue: Epilogue,
    ) {
        let mode = epilogue as u32;
        let enc = batch.encoder();
        enc.set_pipeline(if linear.f32 {
            &self.gemm_f32
        } else {
            &self.gemm
        });
        bind(enc, linear.weights, linear.weights_offset, 0);
        bind(enc, linear.bias, linear.bias_offset, 1);
        bind(enc, input, 0, 2);
        bind(enc, output, 0, 3);
        set_val(enc, &linear.rows, 4);
        set_val(enc, &linear.columns, 5);
        set_val(enc, &tokens, 6);
        set_val(enc, &mode, 7);
        if linear.f32 {
            enc.dispatch_threadgroups(
                MTLSize {
                    width: linear.rows as usize,
                    height: tokens as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            enc.dispatch_threadgroups(
                MTLSize {
                    width: (linear.rows as usize).div_ceil(32),
                    height: (tokens as usize).div_ceil(32),
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }
        batch.record_dispatch();
    }

    pub fn int4_gemm_epilogue(
        &self,
        batch: &mut CommandBatch,
        mat: Int4Matrix<'_>,
        bias: (&MetalBuffer, usize),
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
        epilogue: Epilogue,
    ) {
        let mode = epilogue as u32;
        let enc = batch.encoder();
        enc.set_pipeline(&self.int4_gemm_epilogue);
        bind(enc, mat.codes, mat.codes_offset, 0);
        bind(enc, mat.groups, mat.groups_offset, 1);
        bind(enc, input, 0, 2);
        bind(enc, bias.0, bias.1, 3);
        bind(enc, output, 0, 4);
        set_val(enc, &mat.rows, 5);
        set_val(enc, &mat.columns, 6);
        set_val(enc, &tokens, 7);
        set_val(enc, &mode, 8);
        enc.dispatch_threadgroups(
            MTLSize {
                width: (mat.rows as usize).div_ceil(16),
                height: (tokens as usize).div_ceil(8),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    pub fn int4_qkv_bias(
        &self,
        batch: &mut CommandBatch,
        q: Int4Matrix<'_>,
        bq: (&MetalBuffer, usize),
        k: Int4Matrix<'_>,
        bk: (&MetalBuffer, usize),
        v: Int4Matrix<'_>,
        bv: (&MetalBuffer, usize),
        input: &MetalBuffer,
        out_q: &MetalBuffer,
        out_k: &MetalBuffer,
        out_v: &MetalBuffer,
        tokens: u32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.int4_qkv_bias);
        bind(enc, q.codes, q.codes_offset, 0);
        bind(enc, q.groups, q.groups_offset, 1);
        bind(enc, bq.0, bq.1, 2);
        bind(enc, k.codes, k.codes_offset, 3);
        bind(enc, k.groups, k.groups_offset, 4);
        bind(enc, bk.0, bk.1, 5);
        bind(enc, v.codes, v.codes_offset, 6);
        bind(enc, v.groups, v.groups_offset, 7);
        bind(enc, bv.0, bv.1, 8);
        bind(enc, input, 0, 9);
        bind(enc, out_q, 0, 10);
        bind(enc, out_k, 0, 11);
        bind(enc, out_v, 0, 12);
        set_val(enc, &q.rows, 13);
        set_val(enc, &q.columns, 14);
        set_val(enc, &tokens, 15);
        let g_cnt = (q.rows as usize).div_ceil(16);
        enc.dispatch_threadgroups(
            MTLSize {
                width: 3 * g_cnt,
                height: (tokens as usize).div_ceil(8),
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }
    pub fn layernorm(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        weight: (&MetalBuffer, usize),
        bias: (&MetalBuffer, usize),
        output: &MetalBuffer,
        dimension: u32,
        tokens: u32,
        epsilon: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.layernorm);
        bind(enc, input, 0, 0);
        bind(enc, weight.0, weight.1, 1);
        bind(enc, bias.0, bias.1, 2);
        bind(enc, output, 0, 3);
        set_val(enc, &dimension, 4);
        set_val(enc, &epsilon, 5);
        enc.dispatch_threadgroups(
            MTLSize {
                width: tokens as usize,
                height: 1,
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

    pub fn attention(
        &self,
        batch: &mut CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        output: &MetalBuffer,
        count: u32,
        window: u32,
    ) {
        let enc = batch.encoder();
        if window == 0 {
            enc.set_pipeline(&self.attention_flash);
            bind(enc, q, 0, 0);
            bind(enc, k, 0, 1);
            bind(enc, v, 0, 2);
            bind(enc, output, 0, 3);
            set_val(enc, &count, 4);
            enc.set_threadgroup_memory(ATTENTION_KS_BYTES, 0);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: (count as usize).div_ceil(64),
                    height: VISION_HEADS as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            enc.set_pipeline(&self.attention);
            bind(enc, q, 0, 0);
            bind(enc, k, 0, 1);
            bind(enc, v, 0, 2);
            bind(enc, output, 0, 3);
            set_val(enc, &count, 4);
            set_val(enc, &window, 5);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: count as usize,
                    height: VISION_HEADS as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        batch.record_dispatch();
    }

    pub fn window_gather(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        window_out: &MetalBuffer,
        mean_out: &MetalBuffer,
        grid_h: u32,
        grid_w: u32,
    ) {
        let total = ((grid_h / 2) * (grid_w / 2) * VISION_WIDTH) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.window_gather);
        bind(enc, input, 0, 0);
        bind(enc, window_out, 0, 1);
        bind(enc, mean_out, 0, 2);
        set_val(enc, &grid_h, 3);
        set_val(enc, &grid_w, 4);
        enc.dispatch_threadgroups(
            MTLSize {
                width: total.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    pub fn window_mean(
        &self,
        batch: &mut CommandBatch,
        window_in: &MetalBuffer,
        mean_out: &MetalBuffer,
        total_windows: u32,
    ) {
        let total = (total_windows * VISION_WIDTH) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.window_mean);
        bind(enc, window_in, 0, 0);
        bind(enc, mean_out, 0, 1);
        set_val(enc, &total_windows, 2);
        enc.dispatch_threadgroups(
            MTLSize {
                width: total.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    pub fn bias_epilogue(
        &self,
        batch: &mut CommandBatch,
        data: &MetalBuffer,
        bias: (&MetalBuffer, usize),
        residual: &MetalBuffer,
        rows: u32,
        tokens: u32,
        epilogue: Epilogue,
    ) {
        let total = (tokens * rows) as usize;
        let mode = epilogue as u32;
        let enc = batch.encoder();
        enc.set_pipeline(&self.bias_epilogue);
        bind(enc, data, 0, 0);
        bind(enc, bias.0, bias.1, 1);
        bind(enc, residual, 0, 2);
        set_val(enc, &rows, 3);
        set_val(enc, &tokens, 4);
        set_val(enc, &mode, 5);
        enc.dispatch_threadgroups(
            MTLSize {
                width: total.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    pub fn add_positions_bf16(
        &self,
        batch: &mut CommandBatch,
        hidden: &MetalBuffer,
        pos_bf16: (&MetalBuffer, usize),
        cur_h: u32,
        cur_w: u32,
    ) {
        let total = (cur_h * cur_w * VISION_WIDTH) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.add_positions_bf16);
        bind(enc, hidden, 0, 0);
        bind(enc, pos_bf16.0, pos_bf16.1, 1);
        set_val(enc, &cur_h, 2);
        set_val(enc, &cur_w, 3);
        enc.dispatch_threadgroups(
            MTLSize {
                width: total.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }
}
