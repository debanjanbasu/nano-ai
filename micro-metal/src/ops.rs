//! Checked graph operations for the MiniCPM-V 4.7 1B language model (`shaders/ops.metal`).

use objc2_metal::MTLSize;

use crate::batch::{CommandBatch, ComputeEncoder};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::{Pipeline, ShaderLibrary};

pub const MODEL_WIDTH: u32 = 1024;
pub const MODEL_FFN: u32 = 3584;
pub const Q_HEADS: u32 = 8;
pub const KV_HEADS: u32 = 2;
pub const HEAD_DIM: u32 = 256;
pub const GDN_HEADS: u32 = 16;
pub const GDN_DIM: u32 = 128;
pub const CONV_CHANNELS: u32 = 6144;
pub const GDN_OUT_WIDTH: u32 = 2048;

/// KV cache format per token per full-attention layer (2 KV heads x 256 dims):
/// - `F16`: 1024 bytes/token
/// - `Q8`:  544 bytes/token (512 int8 + 16 half scales, Hadamard-256 rotated per head)
/// - `Q4`:  288 bytes/token (256 packed int4 nibbles + 16 half scales, Hadamard-256 rotated per head)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvFormat {
    F16 = 0,
    #[default]
    Q8 = 1,
    Q4 = 2,
}

impl KvFormat {
    #[must_use]
    pub const fn token_bytes(self) -> usize {
        match self {
            Self::F16 => 1024,
            Self::Q8 => 544,
            Self::Q4 => 288,
        }
    }
    #[must_use]
    pub const fn code(self) -> u32 {
        self as u32
    }
}

/// Rotary position specification for full-attention layers.
#[derive(Clone, Copy)]
pub enum Rope<'a> {
    /// Text rows: token at cache position `p` rotates by `p + delta` in all 3 M-RoPE sections.
    Text(i32),
    /// Explicit `(t, h, w, 0)` `u32` positions per token row (for Canvas 3D M-RoPE prefill).
    Canvas(&'a MetalBuffer, usize),
}

impl Rope<'_> {
    pub const TEXT: Self = Self::Text(0);
}

/// Keys per K/V tile staged by `mc_attn_flash` (`MC_FLASH_KEYS`).
const FLASH_KEYS: u32 = 16;
/// Fewest keys worth their own `mc_attn_flash` split: each split restages Q and writes a 1 KiB
/// partial per (row, head).
const MIN_SPLIT_KEYS: u32 = 256;
/// SIMD groups (2 query rows each) per `mc_attn_flash` threadgroup; 8 measured ~12% slower.
const FLASH_SIMDS: u32 = 4;
/// `mc_attn_flash` threadgroups to aim for (M4 Pro: ~13% faster long prefill than 128, same verify).
const FLASH_THREADGROUPS: u32 = 512;
/// `mc_attn_decode`: one 32-key chunk per split at least, at most 128 splits per KV head
/// (M4 Pro, 262K keys: 1.6 ms vs 2.2-2.4 ms at 256-512 splits; 16K: 0.14 ms).
const DECODE_SPLIT_KEYS: u32 = 32;
const DECODE_SPLITS: u32 = 128;

/// Split-K partials for `MicroOps::attention_causal_block`, with room for `rows`
/// (query row, key split) pairs: sized by the forward width, not the context length.
pub struct AttentionWorkspace {
    pub partials: MetalBuffer,
    pub rows: u32,
}

impl AttentionWorkspace {
    /// `rows` must cover the widest forward block; spare rows let narrow blocks split long prefixes.
    pub fn new(ctx: &MetalContext, rows: u32) -> crate::Result<Self> {
        let bytes = rows.max(1) as usize * Q_HEADS as usize * 258 * 4;
        Ok(Self {
            partials: MetalBuffer::empty(ctx.device(), bytes)?,
            rows,
        })
    }
}

pub struct MicroOps {
    rms: Pipeline,
    rms_fwht512: Pipeline,
    add: Pipeline,
    add_rms: Pipeline,
    add_rms_fwht512: Pipeline,
    add_rms_tap: Pipeline,
    swiglu: Pipeline,
    conv_l2_decay: Pipeline,
    gdn_rows_4_f16: Pipeline,
    conv_history_shift: Pipeline,
    gdn_state_only_f16: Pipeline,
    gdn_post: Pipeline,
    attn_prep: Pipeline,
    attn_flash: Pipeline,
    attn_decode: Pipeline,
    attn_reduce: Pipeline,
    copy_tap: Pipeline,
}

fn bind(encoder: &ComputeEncoder, buffer: &MetalBuffer, offset: usize, index: usize) {
    encoder.bind(buffer, offset, index);
}

fn set_val<T: Copy>(encoder: &ComputeEncoder, value: &T, index: usize) {
    encoder.set_val(value, index);
}

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

impl MicroOps {
    pub fn new(shaders: &ShaderLibrary) -> crate::Result<Self> {
        let make = |name: &str| shaders.pipeline(name);
        Ok(Self {
            rms: make("mc_rms")?,
            rms_fwht512: make("mc_rms_fwht512")?,
            add: make("mc_add")?,
            add_rms: make("mc_add_rms")?,
            add_rms_fwht512: make("mc_add_rms_fwht512")?,
            add_rms_tap: make("mc_add_rms_tap")?,
            swiglu: make("mc_swiglu")?,
            conv_l2_decay: make("mc_conv_l2_decay")?,
            gdn_rows_4_f16: make("mc_gdn_rows_4_f16")?,
            conv_history_shift: make("mc_conv_history_shift")?,
            gdn_state_only_f16: make("mc_gdn_state_only_f16")?,
            gdn_post: make("mc_gdn_post")?,
            attn_prep: make("mc_attn_prep")?,
            attn_flash: make("mc_attn_flash")?,
            attn_decode: make("mc_attn_decode")?,
            attn_reduce: make("mc_attn_reduce")?,
            copy_tap: make("mc_copy_tap")?,
        })
    }

    pub fn rms_norm(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        weight_offset: usize,
        output: &MetalBuffer,
        dimension: u32,
        rows: u32,
        epsilon: f32,
    ) {
        self.rms_norm_at(
            batch,
            input,
            0,
            weights,
            weight_offset,
            output,
            dimension,
            rows,
            epsilon,
        );
    }

    pub fn rms_norm_at(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        input_offset: usize,
        weights: &MetalBuffer,
        weight_offset: usize,
        output: &MetalBuffer,
        dimension: u32,
        rows: u32,
        epsilon: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.rms);
        bind(enc, input, input_offset, 0);
        bind(enc, weights, weight_offset, 1);
        bind(enc, output, 0, 2);
        set_val(enc, &dimension, 3);
        set_val(enc, &rows, 4);
        set_val(enc, &dimension, 5);
        set_val(enc, &epsilon, 6);
        dispatch_1d(enc, rows as usize, 128);
        batch.record_dispatch();
    }

    pub fn rms_norm_fwht512_at(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        input_offset: usize,
        weights: &MetalBuffer,
        weight_offset: usize,
        signs: &MetalBuffer,
        output: &MetalBuffer,
        dimension: u32,
        rows: u32,
        epsilon: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.rms_fwht512);
        bind(enc, input, input_offset, 0);
        bind(enc, weights, weight_offset, 1);
        bind(enc, signs, 0, 2);
        bind(enc, output, 0, 3);
        set_val(enc, &dimension, 4);
        set_val(enc, &rows, 5);
        set_val(enc, &epsilon, 6);
        dispatch_1d(enc, rows as usize, 128);
        batch.record_dispatch();
    }

    pub fn residual_add(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        residual: &MetalBuffer,
        output: &MetalBuffer,
        count: u32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.add);
        bind(enc, input, 0, 0);
        bind(enc, residual, 0, 1);
        bind(enc, output, 0, 2);
        set_val(enc, &count, 3);
        dispatch_1d(enc, (count as usize).div_ceil(256), 256);
        batch.record_dispatch();
    }

    pub fn residual_add_rms(
        &self,
        batch: &mut CommandBatch,
        hidden: &MetalBuffer,
        branch: &MetalBuffer,
        weights: &MetalBuffer,
        weight_offset: usize,
        sum_out: &MetalBuffer,
        normalized: &MetalBuffer,
        dimension: u32,
        rows: u32,
        epsilon: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.add_rms);
        bind(enc, hidden, 0, 0);
        bind(enc, branch, 0, 1);
        bind(enc, weights, weight_offset, 2);
        bind(enc, sum_out, 0, 3);
        bind(enc, normalized, 0, 4);
        set_val(enc, &dimension, 5);
        set_val(enc, &epsilon, 6);
        dispatch_1d(enc, rows as usize, 128);
        batch.record_dispatch();
    }

    pub fn residual_add_rms_fwht512(
        &self,
        batch: &mut CommandBatch,
        hidden: &MetalBuffer,
        branch: &MetalBuffer,
        weights: &MetalBuffer,
        weight_offset: usize,
        signs: &MetalBuffer,
        sum_out: &MetalBuffer,
        output: &MetalBuffer,
        dimension: u32,
        rows: u32,
        epsilon: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.add_rms_fwht512);
        bind(enc, hidden, 0, 0);
        bind(enc, branch, 0, 1);
        bind(enc, weights, weight_offset, 2);
        bind(enc, signs, 0, 3);
        bind(enc, sum_out, 0, 4);
        bind(enc, output, 0, 5);
        set_val(enc, &dimension, 6);
        set_val(enc, &rows, 7);
        set_val(enc, &epsilon, 8);
        dispatch_1d(enc, rows as usize, 128);
        batch.record_dispatch();
    }

    pub fn residual_add_rms_tap(
        &self,
        batch: &mut CommandBatch,
        hidden: &MetalBuffer,
        branch: &MetalBuffer,
        weights: &MetalBuffer,
        weight_offset: usize,
        sum_out: &MetalBuffer,
        normalized: &MetalBuffer,
        taps: &MetalBuffer,
        tap_slot: Option<u32>,
        dimension: u32,
        rows: u32,
        epsilon: f32,
    ) {
        let slot = tap_slot.unwrap_or(u32::MAX);
        let enc = batch.encoder();
        enc.set_pipeline(&self.add_rms_tap);
        bind(enc, hidden, 0, 0);
        bind(enc, branch, 0, 1);
        bind(enc, weights, weight_offset, 2);
        bind(enc, sum_out, 0, 3);
        bind(enc, normalized, 0, 4);
        bind(enc, taps, 0, 5);
        set_val(enc, &dimension, 6);
        set_val(enc, &epsilon, 7);
        set_val(enc, &slot, 8);
        dispatch_1d(enc, rows as usize, 128);
        batch.record_dispatch();
    }

    pub fn swiglu(
        &self,
        batch: &mut CommandBatch,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        output: &MetalBuffer,
        count: u32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.swiglu);
        bind(enc, gate, 0, 0);
        bind(enc, up, 0, 1);
        bind(enc, output, 0, 2);
        set_val(enc, &count, 3);
        dispatch_1d(enc, (count as usize).div_ceil(256), 256);
        batch.record_dispatch();
    }

    pub fn conv_l2_decay(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        conv_weights: (&MetalBuffer, usize),
        history: &MetalBuffer,
        final_history: &MetalBuffer,
        output: &MetalBuffer,
        a: (&MetalBuffer, usize),
        alpha: &MetalBuffer,
        dt: (&MetalBuffer, usize),
        raw_beta: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        tokens: u32,
        commit_tokens: u32,
        epsilon: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.conv_l2_decay);
        bind(enc, input, 0, 0);
        bind(enc, conv_weights.0, conv_weights.1, 1);
        bind(enc, history, 0, 2);
        bind(enc, output, 0, 3);
        bind(enc, a.0, a.1, 4);
        bind(enc, alpha, 0, 5);
        bind(enc, dt.0, dt.1, 6);
        bind(enc, raw_beta, 0, 7);
        bind(enc, decay, 0, 8);
        bind(enc, beta, 0, 9);
        bind(enc, final_history, 0, 10);
        set_val(enc, &tokens, 11);
        set_val(enc, &epsilon, 12);
        set_val(enc, &commit_tokens, 13);
        dispatch_1d(enc, 49, 128);
        batch.record_dispatch();
    }

    pub fn gdn_recurrence_f16(
        &self,
        batch: &mut CommandBatch,
        qkv: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        initial_state: &MetalBuffer,
        final_state: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
        commit_tokens: u32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.gdn_rows_4_f16);
        bind(enc, qkv, 0, 0);
        bind(enc, decay, 0, 1);
        bind(enc, beta, 0, 2);
        bind(enc, initial_state, 0, 3);
        bind(enc, output, 0, 4);
        bind(enc, final_state, 0, 5);
        set_val(enc, &tokens, 6);
        set_val(enc, &commit_tokens, 7);
        // 16 heads * 128 rows / 4 rows per SIMD group = 512 groups of 32 threads
        dispatch_1d(enc, 512, 32);
        batch.record_dispatch();
    }

    pub fn gdn_replay_state_only(
        &self,
        batch: &mut CommandBatch,
        tape_qkv: &MetalBuffer,
        tape_convolved: &MetalBuffer,
        tape_decay: &MetalBuffer,
        tape_beta: &MetalBuffer,
        initial_history: &MetalBuffer,
        final_history: &MetalBuffer,
        initial_state: &MetalBuffer,
        final_state: &MetalBuffer,
        kept_tokens: u32,
    ) {
        let _ = batch.independent(|batch| {
            let enc = batch.encoder();
            enc.set_pipeline(&self.conv_history_shift);
            bind(enc, tape_qkv, 0, 0);
            bind(enc, initial_history, 0, 1);
            bind(enc, final_history, 0, 2);
            set_val(enc, &kept_tokens, 3);
            dispatch_1d(enc, (CONV_CHANNELS as usize).div_ceil(256), 256);
            batch.record_dispatch();

            let enc = batch.encoder();
            enc.set_pipeline(&self.gdn_state_only_f16);
            bind(enc, tape_convolved, 0, 0);
            bind(enc, tape_decay, 0, 1);
            bind(enc, tape_beta, 0, 2);
            bind(enc, initial_state, 0, 3);
            bind(enc, final_state, 0, 4);
            set_val(enc, &kept_tokens, 5);
            dispatch_1d(enc, 512, 32);
            batch.record_dispatch();
            Ok::<(), ()>(())
        });
    }

    pub fn gdn_postprocess(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        gate: &MetalBuffer,
        norm_weights: (&MetalBuffer, usize),
        output: &MetalBuffer,
        tokens: u32,
        epsilon: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.gdn_post);
        bind(enc, input, 0, 0);
        bind(enc, gate, 0, 1);
        bind(enc, norm_weights.0, norm_weights.1, 2);
        bind(enc, output, 0, 3);
        set_val(enc, &epsilon, 4);
        dispatch_1d(enc, (GDN_HEADS * tokens) as usize, 128);
        batch.record_dispatch();
    }

    pub fn prepare_attention(
        &self,
        batch: &mut CommandBatch,
        qg: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_norm: (&MetalBuffer, usize),
        k_norm: (&MetalBuffer, usize),
        q_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        position: u32,
        kv_format: KvFormat,
        epsilon: f32,
        rope_base: f32,
        tokens: u32,
        rope: Rope<'_>,
    ) {
        let (rope_buf, rope_off, rope_mode, rope_delta) = match rope {
            Rope::Text(delta) => (q_norm.0, q_norm.1, 0_u32, delta.cast_unsigned()),
            Rope::Canvas(buf, off) => (buf, off, 1_u32, 0_u32),
        };
        let format_code = kv_format.code();
        let enc = batch.encoder();
        enc.set_pipeline(&self.attn_prep);
        bind(enc, qg, 0, 0);
        bind(enc, k, 0, 1);
        bind(enc, v, 0, 2);
        bind(enc, q_norm.0, q_norm.1, 3);
        bind(enc, k_norm.0, k_norm.1, 4);
        bind(enc, q_out, 0, 5);
        bind(enc, gate_out, 0, 6);
        bind(enc, k_cache, 0, 7);
        bind(enc, v_cache, 0, 8);
        bind(enc, rope_buf, rope_off, 9);
        set_val(enc, &position, 10);
        set_val(enc, &format_code, 11);
        set_val(enc, &rope_mode, 12);
        set_val(enc, &rope_delta, 13);
        set_val(enc, &epsilon, 14);
        set_val(enc, &rope_base, 15);
        dispatch_1d(enc, (Q_HEADS * tokens) as usize, 256);
        batch.record_dispatch();
    }

    /// Causal attention for `tokens` query rows at `position..position + tokens`, unrotated and
    /// gated into `output`. Single rows (plain decode) take the key-parallel `attn_decode`;
    /// wider blocks (speculative verify, prefill) take GQA-packed `attn_flash`. Both split the key
    /// prefix across threadgroups to fill the GPU, and `attn_reduce` merges the splits.
    pub fn attention_causal_block(
        &self,
        batch: &mut CommandBatch,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        gate: &MetalBuffer,
        output: &MetalBuffer,
        position: u32,
        tokens: u32,
        kv_format: KvFormat,
        workspace: &AttentionWorkspace,
    ) {
        debug_assert!(
            tokens <= workspace.rows,
            "attention block wider than its workspace"
        );
        let keys = position + tokens;
        let format_code = kv_format.code();
        let splits = if tokens == 1 {
            let splits = DECODE_SPLITS
                .min(workspace.rows)
                .min(keys.div_ceil(DECODE_SPLIT_KEYS));
            let split_keys = keys.div_ceil(splits).next_multiple_of(DECODE_SPLIT_KEYS);
            let enc = batch.encoder();
            enc.set_pipeline(&self.attn_decode);
            bind(enc, q, 0, 0);
            bind(enc, k_cache, 0, 1);
            bind(enc, v_cache, 0, 2);
            bind(enc, &workspace.partials, 0, 3);
            set_val(enc, &keys, 4);
            set_val(enc, &splits, 5);
            set_val(enc, &split_keys, 6);
            set_val(enc, &format_code, 7);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: splits as usize,
                    height: KV_HEADS as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            splits
        } else {
            let simds = tokens.div_ceil(2).min(FLASH_SIMDS);
            let groups = tokens.div_ceil(2 * simds) * KV_HEADS;
            let splits = (FLASH_THREADGROUPS / groups)
                .min(workspace.rows / tokens)
                .min(keys.div_ceil(MIN_SPLIT_KEYS))
                .max(1);
            let split_keys = keys.div_ceil(splits).next_multiple_of(FLASH_KEYS);
            let enc = batch.encoder();
            enc.set_pipeline(&self.attn_flash);
            bind(enc, q, 0, 0);
            bind(enc, k_cache, 0, 1);
            bind(enc, v_cache, 0, 2);
            bind(enc, &workspace.partials, 0, 3);
            set_val(enc, &position, 4);
            set_val(enc, &tokens, 5);
            set_val(enc, &splits, 6);
            set_val(enc, &split_keys, 7);
            set_val(enc, &format_code, 8);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: tokens.div_ceil(2 * simds) as usize,
                    height: KV_HEADS as usize,
                    depth: splits as usize,
                },
                MTLSize {
                    width: (32 * simds) as usize,
                    height: 1,
                    depth: 1,
                },
            );
            splits
        };
        batch.record_dispatch();

        let gated = 1_u32;
        let rotated = u32::from(kv_format != KvFormat::F16);
        let enc = batch.encoder();
        enc.set_pipeline(&self.attn_reduce);
        bind(enc, &workspace.partials, 0, 0);
        bind(enc, gate, 0, 1);
        bind(enc, output, 0, 2);
        set_val(enc, &splits, 3);
        set_val(enc, &gated, 4);
        set_val(enc, &rotated, 5);
        dispatch_1d(enc, (Q_HEADS * tokens) as usize, 256);
        batch.record_dispatch();
    }

    pub fn copy_tap(
        &self,
        batch: &mut CommandBatch,
        hidden: &MetalBuffer,
        taps: &MetalBuffer,
        tap_slot: u32,
        tokens: u32,
    ) {
        let total = (tokens * MODEL_WIDTH) as usize;
        let enc = batch.encoder();
        enc.set_pipeline(&self.copy_tap);
        bind(enc, hidden, 0, 0);
        bind(enc, taps, 0, 1);
        set_val(enc, &tap_slot, 2);
        set_val(enc, &tokens, 3);
        dispatch_1d(enc, total.div_ceil(256), 256);
        batch.record_dispatch();
    }
}
