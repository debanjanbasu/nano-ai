//! GPU-resident top-k and greedy argmax selection (`shaders/sampling.metal`).

use bytemuck::{Pod, Zeroable};
use objc2_metal::MTLSize;

use crate::Error;
use crate::batch::{CommandBatch, ComputeEncoder};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::{Pipeline, ShaderLibrary};

pub const MAX_TOP_K: usize = 100;

#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct TopKCandidate {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct GreedyRowResult {
    pub best_id: u32,
    pub best_logit: f32,
    pub nonfinite: u32,
    pub second_logit: f32,
}

pub struct SamplingKernels {
    topk_partial: Pipeline,
    topk_final: Pipeline,
    greedy_partial: Pipeline,
    greedy_final: Pipeline,
    rep_penalty: Pipeline,
    topk_partials: MetalBuffer,
    topk_out: MetalBuffer,
    greedy_partials: MetalBuffer,
    greedy_out: MetalBuffer,
    rep_ids: MetalBuffer,
    max_vocab: usize,
    max_rows: usize,
}

fn bind(encoder: &ComputeEncoder, buffer: &MetalBuffer, offset: usize, index: usize) {
    encoder.bind(buffer, offset, index);
}

fn set_val<T: Copy>(encoder: &ComputeEncoder, value: &T, index: usize) {
    encoder.set_val(value, index);
}

impl SamplingKernels {
    pub fn new(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        max_vocab: usize,
        max_rows: usize,
    ) -> crate::Result<Self> {
        let make = |name: &str| shaders.pipeline(name);
        let topk_groups = max_vocab.div_ceil(4096);
        let greedy_groups = max_vocab.div_ceil(1024);
        Ok(Self {
            topk_partial: make("topk_rows_partial")?,
            topk_final: make("topk_rows_final")?,
            greedy_partial: make("greedy_rows_partial")?,
            greedy_final: make("greedy_rows_final")?,
            rep_penalty: make("apply_repetition_penalty")?,
            topk_partials: MetalBuffer::empty(
                ctx.device(),
                max_rows * topk_groups * MAX_TOP_K * size_of::<TopKCandidate>(),
            )?,
            topk_out: MetalBuffer::empty(
                ctx.device(),
                max_rows * MAX_TOP_K * size_of::<TopKCandidate>(),
            )?,
            greedy_partials: MetalBuffer::empty(
                ctx.device(),
                max_rows * greedy_groups * size_of::<GreedyRowResult>(),
            )?,
            greedy_out: MetalBuffer::empty(ctx.device(), max_rows * size_of::<GreedyRowResult>())?,
            rep_ids: MetalBuffer::empty(ctx.device(), 1024 * size_of::<[u32; 2]>())?,
            max_vocab,
            max_rows,
        })
    }

    #[expect(unsafe_code)]
    pub fn encode_repetition_penalty(
        &mut self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        history_counts: &[[u32; 2]],
        vocab: usize,
        rows: usize,
        penalty: f32,
        frequency_penalty: f32,
    ) {
        if history_counts.is_empty()
            || ((penalty - 1.0).abs() <= 1e-4 && frequency_penalty.abs() <= 1e-5)
        {
            return;
        }
        let len = history_counts.len().min(1024);
        unsafe {
            let dst = &mut self.rep_ids.as_mut_slice::<[u32; 2]>()[..len];
            dst.copy_from_slice(&history_counts[..len]);
            // One kernel thread applies each token's run; the all-rows (`0xFF`) entry goes first.
            dst.sort_unstable_by_key(|&[tagged, _]| {
                (tagged & 0x00FF_FFFF, tagged >> 24 != 0xFF, tagged)
            });
        }
        let len_u32 = len as u32;
        let vocab_u32 = vocab as u32;
        let rows_u32 = rows as u32;
        let penalties = [penalty, frequency_penalty];
        let enc = batch.encoder();
        enc.set_pipeline(&self.rep_penalty);
        bind(enc, logits, 0, 0);
        bind(enc, &self.rep_ids, 0, 1);
        set_val(enc, &len_u32, 2);
        set_val(enc, &vocab_u32, 3);
        set_val(enc, &rows_u32, 4);
        set_val(enc, &penalties, 5);
        enc.dispatch_threadgroups(
            MTLSize {
                width: len.div_ceil(256),
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

    pub fn encode_greedy_rows(
        &self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        vocab: usize,
        rows: usize,
    ) -> crate::Result<()> {
        if rows == 0 || rows > self.max_rows || vocab == 0 || vocab > self.max_vocab {
            return Err(Error::InvalidArgument("greedy_rows out of bounds".into()));
        }
        let groups = vocab.div_ceil(1024);
        let vocab_u32 = vocab as u32;
        let enc = batch.encoder();
        enc.set_pipeline(&self.greedy_partial);
        bind(enc, logits, 0, 0);
        bind(enc, &self.greedy_partials, 0, 1);
        set_val(enc, &vocab_u32, 2);
        enc.dispatch_threadgroups(
            MTLSize {
                width: groups,
                height: rows,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();

        let groups_u32 = groups as u32;
        let enc = batch.encoder();
        enc.set_pipeline(&self.greedy_final);
        bind(enc, &self.greedy_partials, 0, 0);
        bind(enc, &self.greedy_out, 0, 1);
        set_val(enc, &groups_u32, 2);
        enc.dispatch_threadgroups(
            MTLSize {
                width: rows,
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
        Ok(())
    }

    #[must_use]
    #[expect(unsafe_code)]
    pub fn read_greedy_rows(&self, rows: usize) -> Vec<GreedyRowResult> {
        let slice = unsafe { self.greedy_out.as_slice::<GreedyRowResult>() };
        slice[..rows].to_vec()
    }

    /// Row 0 of the greedy output without the Vec the multi-row read builds — the plain
    /// single-token decode path calls this once per generated token.
    #[must_use]
    #[expect(unsafe_code)]
    pub fn read_greedy_row0(&self) -> GreedyRowResult {
        let slice = unsafe { self.greedy_out.as_slice::<GreedyRowResult>() };
        slice[0]
    }


    pub fn encode_topk_rows(
        &self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        vocab: usize,
        rows: usize,
        k: usize,
    ) -> crate::Result<()> {
        if rows == 0 || rows > self.max_rows || vocab == 0 || vocab > self.max_vocab {
            return Err(Error::InvalidArgument("topk_rows out of bounds".into()));
        }
        let k = k.clamp(1, MAX_TOP_K);
        let groups = vocab.div_ceil(4096);
        let (vocab_u32, k_u32) = (vocab as u32, k as u32);
        let enc = batch.encoder();
        enc.set_pipeline(&self.topk_partial);
        bind(enc, logits, 0, 0);
        bind(enc, &self.topk_partials, 0, 1);
        set_val(enc, &vocab_u32, 2);
        set_val(enc, &k_u32, 3);
        set_val(enc, &vocab_u32, 4);
        enc.dispatch_threadgroups(
            MTLSize {
                width: groups,
                height: rows,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();

        let groups_u32 = groups as u32;
        let enc = batch.encoder();
        enc.set_pipeline(&self.topk_final);
        bind(enc, &self.topk_partials, 0, 0);
        bind(enc, &self.topk_out, 0, 1);
        set_val(enc, &groups_u32, 2);
        set_val(enc, &k_u32, 3);
        enc.dispatch_threadgroups(
            MTLSize {
                width: rows,
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
        Ok(())
    }

    #[must_use]
    #[expect(unsafe_code)]
    pub fn read_topk_row(&self, row: usize, k: usize) -> &[TopKCandidate] {
        let k = k.clamp(1, MAX_TOP_K);
        let base = row * MAX_TOP_K;
        let slice = unsafe { self.topk_out.as_slice::<TopKCandidate>() };
        &slice[base..base + k]
    }
}
