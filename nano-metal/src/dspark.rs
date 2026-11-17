//! `DSpark` 5-Layer Semi-Autoregressive Speculative Drafter Kernels (`shaders/dspark.metal`).

use objc2_metal::MTLSize;

use crate::batch::{CommandBatch, ComputeEncoder};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::{Pipeline, ShaderLibrary};

pub struct DsparkKernels {
    qk_norm_rope_kv_store: Pipeline,
    k_norm_rope_kv_store: Pipeline,
    block_attention: Pipeline,
    block_attention_split: Pipeline,
    block_attention_reduce: Pipeline,
    cand_stage1: Pipeline,
    cand_stage2: Pipeline,
    walk_dots_int4: Pipeline,
    walk_pick_int4: Pipeline,
    attn_partials: MetalBuffer,
    chunk_cands: MetalBuffer,
    cand_ids: MetalBuffer,
    cand_logits: MetalBuffer,
    walk_dots: MetalBuffer,
    rep_ids: MetalBuffer,
}

fn bind(encoder: &ComputeEncoder, buffer: &MetalBuffer, offset: usize, index: usize) {
    encoder.bind(buffer, offset, index);
}

fn set_val<T: Copy>(encoder: &ComputeEncoder, value: &T, index: usize) {
    encoder.set_val(value, index);
}

impl DsparkKernels {
    pub fn new(ctx: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        let make = |name: &str| shaders.pipeline(name);
        Ok(Self {
            qk_norm_rope_kv_store: make("ds_qk_norm_rope_kv_store_f16")?,
            k_norm_rope_kv_store: make("ds_k_norm_rope_kv_store_f16")?,
            block_attention: make("ds_block_attention")?,
            block_attention_split: make("ds_block_attention_split")?,
            block_attention_reduce: make("ds_block_attention_reduce")?,
            cand_stage1: make("ds_cand_stage1")?,
            cand_stage2: make("ds_cand_stage2")?,
            walk_dots_int4: make("ds_markov_walk_dots_int4")?,
            walk_pick_int4: make("ds_markov_walk_pick_int4")?,
            attn_partials: MetalBuffer::empty(ctx.device(), 8 * 8 * 32 * 258 * 4)?,
            chunk_cands: MetalBuffer::empty(ctx.device(), 8 * 64 * 16 * 8)?,
            cand_ids: MetalBuffer::empty(ctx.device(), 8 * 16 * 4)?,
            cand_logits: MetalBuffer::empty(ctx.device(), 8 * 16 * 4)?,
            walk_dots: MetalBuffer::empty(ctx.device(), 8 * 16 * 16 * 4)?,
            rep_ids: MetalBuffer::empty(ctx.device(), 1024 * 8)?,
        })
    }

    pub fn qk_norm_rope_kv_store_f16(
        &self,
        batch: &mut CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_norm: (&MetalBuffer, usize),
        k_norm: (&MetalBuffer, usize),
        positions: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        start_pos: u32,
        tokens: u32,
        epsilon: f32,
        rope_base: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.qk_norm_rope_kv_store);
        bind(enc, q, 0, 0);
        bind(enc, k, 0, 1);
        bind(enc, q_norm.0, q_norm.1, 2);
        bind(enc, k_norm.0, k_norm.1, 3);
        bind(enc, positions, 0, 4);
        bind(enc, v, 0, 5);
        bind(enc, k_cache, 0, 6);
        bind(enc, v_cache, 0, 7);
        set_val(enc, &start_pos, 8);
        set_val(enc, &epsilon, 9);
        set_val(enc, &rope_base, 10);
        enc.dispatch_threadgroups(
            MTLSize {
                width: 10, // 8 Q heads + 2 KV heads
                height: tokens as usize,
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

    /// K-only norm + `RoPE` + Q8 K/V store for context rows `start_pos..start_pos + tokens`.
    pub fn k_norm_rope_kv_store_f16(
        &self,
        batch: &mut CommandBatch,
        k: (&MetalBuffer, usize),
        v: (&MetalBuffer, usize),
        k_norm: (&MetalBuffer, usize),
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        start_pos: u32,
        tokens: u32,
        epsilon: f32,
        rope_base: f32,
    ) {
        let enc = batch.encoder();
        enc.set_pipeline(&self.k_norm_rope_kv_store);
        bind(enc, k.0, k.1, 0);
        bind(enc, v.0, v.1, 1);
        bind(enc, k_norm.0, k_norm.1, 2);
        bind(enc, k_cache, 0, 3);
        bind(enc, v_cache, 0, 4);
        set_val(enc, &start_pos, 5);
        set_val(enc, &epsilon, 6);
        set_val(enc, &rope_base, 7);
        enc.dispatch_threadgroups(
            MTLSize {
                width: 2,
                height: tokens as usize,
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

    /// Non-causal attention of `block_tokens` query rows over keys `[first_key, total_keys)`.
    pub fn block_attention(
        &self,
        batch: &mut CommandBatch,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        output: &MetalBuffer,
        block_tokens: u32,
        total_keys: u32,
        first_key: u32,
    ) {
        if first_key == 0 && total_keys <= 128 {
            let enc = batch.encoder();
            enc.set_pipeline(&self.block_attention);
            bind(enc, q, 0, 0);
            bind(enc, k_cache, 0, 1);
            bind(enc, v_cache, 0, 2);
            bind(enc, output, 0, 3);
            set_val(enc, &total_keys, 4);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: block_tokens as usize,
                    height: 2,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            let splits = (total_keys - first_key).div_ceil(64).min(32);
            let qh_total = (block_tokens * 8) as usize;
            let enc = batch.encoder();
            enc.set_pipeline(&self.block_attention_split);
            bind(enc, q, 0, 0);
            bind(enc, k_cache, 0, 1);
            bind(enc, v_cache, 0, 2);
            bind(enc, &self.attn_partials, 0, 3);
            set_val(enc, &total_keys, 4);
            set_val(enc, &splits, 5);
            set_val(enc, &first_key, 6);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: splits as usize,
                    height: qh_total,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            batch.record_dispatch();

            let enc = batch.encoder();
            enc.set_pipeline(&self.block_attention_reduce);
            bind(enc, &self.attn_partials, 0, 0);
            bind(enc, output, 0, 1);
            set_val(enc, &splits, 2);
            enc.dispatch_threadgroups(
                MTLSize {
                    width: qh_total,
                    height: 1,
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
    }


    /// Kapitsa-style two-stage top-16 candidate selection + parallel Markov walk (`ds_cand_stage1` ->
    /// `ds_cand_stage2` -> `ds_markov_walk_dots_int4` -> `ds_markov_walk_pick_int4`).
    #[expect(unsafe_code)]
    pub fn markov_walk_int4(
        &mut self,
        batch: &mut CommandBatch,
        w1_codes: (&MetalBuffer, usize),
        w1_groups: (&MetalBuffer, usize),
        w2_codes: (&MetalBuffer, usize),
        w2_groups: (&MetalBuffer, usize),
        hidden: &MetalBuffer,
        base_logits: &MetalBuffer,
        special_logits: &MetalBuffer,
        conf_w: (&MetalBuffer, usize),
        conf_b: (&MetalBuffer, usize),
        draft_tokens: &MetalBuffer,
        conf_out: &MetalBuffer,
        steps: u32,
        vocab: u32,
        recent_counts: &[[u32; 2]],
        rep_penalty: f32,
        freq_penalty: f32,
    ) {
        let rep_len = if rep_penalty > 1.0001 && !recent_counts.is_empty() {
            let n = recent_counts.len().min(1024);
            unsafe {
                let dst = &mut self.rep_ids.as_mut_slice::<[u32; 2]>()[..n];
                dst.copy_from_slice(&recent_counts[..n]);
            }
            n as u32
        } else {
            0
        };
        let penalties = [rep_penalty, freq_penalty];
        let nchunks = (vocab as usize).div_ceil(4096) as u32;
        let enc = batch.encoder();
        enc.set_pipeline(&self.cand_stage1);
        bind(enc, base_logits, 0, 0);
        bind(enc, special_logits, 0, 1);
        bind(enc, &self.chunk_cands, 0, 2);
        set_val(enc, &vocab, 3);
        set_val(enc, &nchunks, 4);
        enc.dispatch_threadgroups(
            MTLSize {
                width: nchunks as usize,
                height: steps as usize,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();

        let enc = batch.encoder();
        enc.set_pipeline(&self.cand_stage2);
        bind(enc, &self.chunk_cands, 0, 0);
        bind(enc, &self.cand_ids, 0, 1);
        bind(enc, &self.cand_logits, 0, 2);
        set_val(enc, &nchunks, 3);
        enc.dispatch_threadgroups(
            MTLSize {
                width: steps as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();

        let enc = batch.encoder();
        enc.set_pipeline(&self.walk_dots_int4);
        bind(enc, &self.cand_ids, 0, 0);
        bind(enc, w1_codes.0, w1_codes.1, 1);
        bind(enc, w1_groups.0, w1_groups.1, 2);
        bind(enc, w2_codes.0, w2_codes.1, 3);
        bind(enc, w2_groups.0, w2_groups.1, 4);
        bind(enc, draft_tokens, 0, 5);
        bind(enc, &self.walk_dots, 0, 6);
        enc.dispatch_threadgroups(
            MTLSize {
                width: steps as usize,
                height: 16,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();

        let enc = batch.encoder();
        enc.set_pipeline(&self.walk_pick_int4);
        bind(enc, &self.cand_ids, 0, 0);
        bind(enc, &self.cand_logits, 0, 1);
        bind(enc, &self.walk_dots, 0, 2);
        bind(enc, w1_codes.0, w1_codes.1, 3);
        bind(enc, w1_groups.0, w1_groups.1, 4);
        bind(enc, hidden, 0, 5);
        bind(enc, conf_w.0, conf_w.1, 6);
        bind(enc, conf_b.0, conf_b.1, 7);
        bind(enc, draft_tokens, 0, 8);
        bind(enc, conf_out, 0, 9);
        set_val(enc, &steps, 10);
        bind(enc, &self.rep_ids, 0, 11);
        set_val(enc, &rep_len, 12);
        set_val(enc, &penalties, 13);
        enc.dispatch_threadgroups(
            MTLSize {
                width: 1,
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
