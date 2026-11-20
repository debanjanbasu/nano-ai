//! `DSpark` 5-Layer Semi-Autoregressive Speculative Drafter (`openbmb/MiniCPM-V-4.7-1B-DSpark`)
//! and Hardware-Aware Confidence Prefix Scheduler (arXiv:2607.05147).
//!
//! Architecture:
//! - Target tap layers: `[3, 7, 11, 15, 19]` concatenated `[n, 5120]` -> `fc` `[1024, 5120]` -> `hidden_norm` `[1024]`.
//! - Fused context KV precomputation across all 5 draft layers (`[5 * 2 * 512, 1024]`), stored in F16 KV cache.
//! - Block-parallel forward pass over up to `block_size` (default 8..15) positions: `[anchor_token, mask_token, ...]`,
//!   non-causal block attention over `context_len + block_len`.
//! - Shared output head (`lm_head`) with ID-order frequency cut (first 98,304 rows + special tokens) for 2.5x faster draft scoring.
//! - Sequential low-rank (`r = 256`) Markov head (`markov_w1`, `markov_w2`) + Confidence head (`σ(w^T [h_k; markov_w1[x_{k-1}]] + b)`),
//!   executed entirely on the GPU without host round-trips per draft position.
//! - Hardware-Aware Prefix Scheduler: selects optimal verification length `ℓ*` maximizing `Θ(ℓ) = (1 + Σ_{j=1}^ℓ ∏_{i≤j} c_i) / T_verify(1 + ℓ)`.

use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

use nano_metal::batch::CommandBatch;
use nano_metal::buffer::MetalBuffer;
use nano_metal::context::MetalContext;
use nano_metal::dspark::DsparkKernels;
use nano_metal::ops::NanoOps;
use nano_metal::projection::{AnyMatrix, ProjectionKernels, SignedHadamard512};
use nano_metal::shaders::ShaderLibrary;

use crate::Error;
use crate::artifact::{DSP_MAGIC, MappedArtifact, MatrixHandle};
use crate::model::{VOCAB, WIDTH};

pub const DSPARK_LAYERS: usize = 5;
pub const DSPARK_TAPS: [usize; 5] = [3, 7, 11, 15, 19];
pub const DSPARK_MAX_BLOCK: usize = 7;
/// Depth-0 score margin below which the chain's first token is treated as a coin flip and the
/// next drafted block is capped short (`recommended_forward_len`). The pick kernel's scores are
/// pre-softmax logits of one candidate set, so a gap under ~0.5 nats is indistinguishable noise.
pub(crate) const DEPTH0_MARGIN_AMBIGUOUS: f32 = 0.5;
pub const DSPARK_MASK_TOKEN: u32 = 248_076;
pub const DSPARK_RANK: usize = 256;
pub const DSPARK_HEAD_CUT: usize = 98_304;
pub const DSPARK_SPECIAL_FIRST: usize = 248_044;
pub const DSPARK_SPECIAL_COUNT: usize = 50;
pub const DSPARK_REDUCED_ANCHORS: usize = 4;
pub const DSPARK_EPSILON: f32 = 1e-6;
pub const DSPARK_ROPE_THETA: f32 = 10_000_000.0;
pub const DSPARK_KV_TOKEN_BYTES: usize = 544;
/// Context keys the drafter attends to besides its block.
///
/// Its acceptance collapses when it sees a long context (M4 Pro, 41K: 60 of 661 drafts accepted
/// with full attention, 152 of 306 with 1024 keys, 128 of 462 with 4096), and its cost grows.
pub const DSPARK_WINDOW: usize = 1024;
/// Positions held by each drafter KV cache, a ring (`DS_KV_RING` in dspark.metal).
///
/// It covers the window plus the block, with room to roll back up to `DSPARK_KV_RING -
/// DSPARK_WINDOW` tokens: 11 MiB in total at any context (a full-context cache was 1.4 GiB at 262K).
pub const DSPARK_KV_RING: usize = 2048;
pub const BASE_VERIFY_MS: [f32; DSPARK_MAX_BLOCK + 1] =
    [2.2, 2.5, 2.8, 3.1, 3.4, 3.6, 3.8, 4.0];
/// Hardware-Aware Prefix Scheduler (`Algorithm 1` of arXiv:2607.05147).
///
/// Given per-position conditional survival probabilities `c_1, ..., c_γ` and verify step times `verify_ms[b]`
/// for batch size `b = 1..=1+γ` (plus `draft_ms`), returns the prefix length `ℓ* ∈ 1..=γ` that maximizes
/// expected verified tokens per millisecond: `Θ(ℓ) = (1 + Σ_{j=1}^ℓ ∏_{i=1}^j c_i) / (draft_ms + verify_ms[ℓ])`.
#[must_use]
pub fn schedule_verification_length(
    confidences: &[f32],
    draft_ms: f32,
    verify_ms_table: &[f32],
) -> (usize, f32) {
    schedule_verification_length_with_rate(confidences, draft_ms, verify_ms_table, 0.0)
}

/// Kapitsa `KfBudget` margin scheduler: maximizes `E(ℓ) - rate * (draft_ms + verify_ms[ℓ])`
/// once the running token/ms `rate` is established, falling back to `E(ℓ) / (draft_ms + verify_ms[ℓ])` during warmup.
#[must_use]
pub fn schedule_verification_length_with_rate(
    confidences: &[f32],
    draft_ms: f32,
    verify_ms_table: &[f32],
    recent_rate: f32,
) -> (usize, f32) {
    if confidences.is_empty() {
        return (0, 1.0);
    }
    let max_l = confidences.len().min(verify_ms_table.len().saturating_sub(1));
    let mut survival = 1.0_f32;
    let mut expected_tokens = 1.0_f32;
    let mut best_l = 1_usize;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_expected = 1.0_f32;

    for l in 1..=max_l {
        let c = confidences[l - 1].clamp(0.0, 1.0);
        survival *= c;
        expected_tokens += survival;
        let total_ms = (draft_ms + verify_ms_table[l]).max(0.05);
        let score = if recent_rate > 0.01 {
            recent_rate.mul_add(-total_ms, expected_tokens)
        } else {
            expected_tokens / total_ms
        };
        if l == 1 || score >= best_score {
            best_score = score;
            best_l = l;
            best_expected = expected_tokens;
        } else {
            break;
        }
    }
    (best_l, best_expected)
}

/// Detect a degenerate cycle at the end of `history`: the shortest period `p` (1..=64) whose
/// suffix repeats `MIN_CYCLES` or more times back-to-back. Returns `(p, cycles)` — the loop
/// continuation is `history[history.len() - p]` — or `None` when the tail is not cycling.
/// Deliberately restricted to the last 512 tokens and short periods: legitimate structured
/// output (JSON keys, code syntax) repeats with long periods or few cycles; only verbatim
/// lock-in repeats a short unit many times in a row (arXiv 2610.24715: all surviving llama.cpp
/// loops had period g <= 7).
#[must_use]
pub(crate) fn detect_degenerate_cycle(history: &[u32]) -> Option<(usize, usize)> {
    const MIN_CYCLES: usize = 8;
    let n = history.len();
    if n < MIN_CYCLES {
        return None;
    }
    let window_start = n.saturating_sub(512);
    for p in 1..=64usize {
        if n < p * MIN_CYCLES {
            break;
        }
        // Count how many consecutive p-blocks ending at the tail match the last block.
        let last = &history[n - p..];
        let mut cycles = 1_usize;
        let mut end = n - p;
        while end >= window_start + p {
            if &history[end - p..end] != last {
                break;
            }
            cycles += 1;
            end -= p;
        }
        if cycles >= MIN_CYCLES {
            // The shortest qualifying period is returned; its multiples repeat the same
            // continuation, so one penalty entry covers them.
            return Some((p, cycles));
        }
    }
    None
}

/// Prompt-Lookup (N-gram suffix) zero-cost drafter.
///
/// Scans prompt and generated token history for the longest suffix match (≥ `min_match` tokens)
/// and proposes up to `max_draft` continuation tokens in < 0.05 ms on CPU.
#[must_use]
pub fn lookup_ngram_draft(history: &[u32], min_match: usize, max_draft: usize) -> Vec<u32> {
    let n = history.len();
    if n < min_match * 2 || max_draft == 0 {
        return Vec::new();
    }
    let suffix = &history[n - min_match..];
    let search_end = n - min_match;
    for pos in (0..search_end).rev() {
        if history[pos..pos + min_match] == *suffix {
            let cont_start = pos + min_match;
            // Never copy tokens from the active suffix itself (`n - min_match..n`),
            // preventing self-referential repetition loops!
            let cont_end = (cont_start + max_draft).min(n - min_match);
            if cont_end <= cont_start {
                continue;
            }
            return history[cont_start..cont_end].to_vec();
        }
    }
    Vec::new()
}

/// Adaptive Prompt-Lookup (N-gram suffix) zero-cost drafter.
///
/// Uses strict 6-gram matching while `<think>` reasoning is open to prevent reasoning loops,
/// and falls back to 5-gram and recent-window 4-gram matching after `</think>` (`closed_thinking`).
#[must_use]
pub fn lookup_ngram_draft_adaptive(
    history: &[u32],
    closed_thinking: bool,
    max_draft: usize,
) -> Vec<u32> {
    if max_draft == 0 {
        return Vec::new();
    }
    // A degenerate cycle at the tail means the best suffix match IS the loop: drafting its
    // continuation would feed the lock-in the cycle penalty is trying to break (arXiv
    // 2610.24715). Return nothing and let the penalty push the model out of the rut.
    if detect_degenerate_cycle(history).is_some() {
        return Vec::new();
    }
    let ng6 = lookup_ngram_draft(history, 6, max_draft);
    if !ng6.is_empty() || !closed_thinking {
        return ng6;
    }
    let ng5 = lookup_ngram_draft(history, 5, max_draft.min(5));
    if !ng5.is_empty() {
        return ng5;
    }
    let n = history.len();
    if n >= 16 && history[n - 1] != history[n - 2] {
        let window_start = n.saturating_sub(512);
        return lookup_ngram_draft(&history[window_start..], 4, max_draft.min(4));
    }
    Vec::new()
}

const CALIBRATOR_CAPACITY: usize = 256;
const CALIBRATOR_MIN_OBS: usize = 16;
const CALIBRATOR_REFIT_STRIDE: usize = 8;
const CALIBRATOR_NEWTON_STEPS: usize = 15;

/// Online damped Newton-Raphson logistic calibrator for `DSpark` confidence heads (`p = σ(α_d + β_d logit(q_d))`).
pub struct ConfidenceCalibrator {
    records: [VecDeque<(f32, bool)>; DSPARK_MAX_BLOCK],
    counts: [usize; DSPARK_MAX_BLOCK],
    pub alpha: [f32; DSPARK_MAX_BLOCK],
    pub beta: [f32; DSPARK_MAX_BLOCK],
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceCalibrator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: std::array::from_fn(|_| VecDeque::with_capacity(CALIBRATOR_CAPACITY)),
            counts: [0; DSPARK_MAX_BLOCK],
            alpha: [0.0; DSPARK_MAX_BLOCK],
            beta: [1.0; DSPARK_MAX_BLOCK],
        }
    }

    #[must_use]
    pub fn calibrate(&self, depth: usize, raw_conf: f32) -> f32 {
        let idx = depth.min(DSPARK_MAX_BLOCK - 1);
        let prob = raw_conf.clamp(1e-4, 1.0 - 1e-4);
        let logit_q = (prob / (1.0 - prob)).ln();
        let linear = self.beta[idx].mul_add(logit_q, self.alpha[idx]);
        1.0 / (1.0 + (-linear).exp())
    }

    /// Observed acceptance rate at `depth` over the recent ring, when enough rounds have been
    /// seen. This is the quantity the scheduler actually needs (E(ℓ) multiplies survival across
    /// depths), and it cannot be optimistic: it is what really happened. `None` during warmup.
    #[must_use]
    pub fn empirical_acceptance(&self, depth: usize) -> Option<f32> {
        let idx = depth.min(DSPARK_MAX_BLOCK - 1);
        let ring = &self.records[idx];
        if ring.len() < CALIBRATOR_MIN_OBS {
            return None;
        }
        let hits = ring.iter().filter(|&&(_, accepted)| accepted).count();
        Some(hits as f32 / ring.len() as f32)
    }

    pub fn observe(&mut self, raw_confs: &[f32], accepted_len: usize) {
        if raw_confs.is_empty() {
            return;
        }
        // Observe EVERY depth every round: depths beyond this round's drafted length count as
        // not accepted. Restricting observations to the depths a round actually reached makes
        // the deep-depth rings survivorship-biased — they only see rounds where the shallower
        // depths were accepted, so `empirical_acceptance(3+)` reads high and the scheduler
        // keeps drafting deep blocks that mostly roll back (measured on the A15: 4.0 drafted
        // per round at 21% overall acceptance, while a depth-1 cap ran 12% faster).
        let conf_at = |depth: usize| -> f32 {
            raw_confs
                .get(depth)
                .copied()
                .unwrap_or(0.5)
                .clamp(1e-4, 1.0 - 1e-4)
        };
        for depth in 0..DSPARK_MAX_BLOCK {
            let prob = conf_at(depth);
            let logit_q = (prob / (1.0 - prob)).ln();
            let accepted = depth < accepted_len;
            let ring = &mut self.records[depth];
            if ring.len() >= CALIBRATOR_CAPACITY {
                ring.pop_front();
            }
            ring.push_back((logit_q, accepted));
            self.counts[depth] += 1;
            if ring.len() >= CALIBRATOR_MIN_OBS
                && self.counts[depth].is_multiple_of(CALIBRATOR_REFIT_STRIDE)
            {
                let hits = ring.iter().filter(|&&(_, accepted)| accepted).count();
                if hits == 0 {
                    self.alpha[depth] = -4.0;
                    self.beta[depth] = 1.0;
                } else if hits == ring.len() {
                    self.alpha[depth] = 4.0;
                    self.beta[depth] = 1.0;
                } else if hits >= 3 && ring.len() - hits >= 3 {
                    self.refit_depth(depth);
                }
            }
        }
    }

    fn refit_depth(&mut self, depth: usize) {
        let ring = &self.records[depth];
        let mut cur_alpha = f64::from(self.alpha[depth]);
        let mut cur_beta = f64::from(self.beta[depth]);
        for _ in 0..CALIBRATOR_NEWTON_STEPS {
            let (mut grad0, mut grad1) = (1e-4 * cur_alpha, 1e-4 * (cur_beta - 1.0));
            let (mut h00, mut h01, mut h11) = (1e-4_f64, 0.0_f64, 1e-4_f64);
            for &(logit, hit) in ring {
                let x_val = f64::from(logit);
                let pred = 1.0 / (1.0 + (-cur_beta.mul_add(x_val, cur_alpha)).exp());
                let weight = pred.mul_add(1.0 - pred, 1e-6);
                let residual = pred - f64::from(u8::from(hit));
                grad0 += residual;
                grad1 = residual.mul_add(x_val, grad1);
                h00 += weight;
                h01 = weight.mul_add(x_val, h01);
                h11 = (weight * x_val).mul_add(x_val, h11);
            }
            let det = h00.mul_add(h11, -(h01 * h01));
            if det.abs() < 1e-18 {
                break;
            }
            let step0 = 0.5 * h11.mul_add(grad0, -(h01 * grad1)) / det;
            let step1 = 0.5 * h00.mul_add(grad1, -(h01 * grad0)) / det;
            cur_alpha = (cur_alpha - step0).clamp(-4.0, 4.0);
            cur_beta = (cur_beta - step1).clamp(0.1, 8.0);
            if step0.abs().max(step1.abs()) < 1e-6 {
                break;
            }
        }
        if cur_alpha.is_finite() && cur_beta.is_finite() {
            self.alpha[depth] = cur_alpha as f32;
            self.beta[depth] = cur_beta as f32;
        }
    }
}
/// One drafter layer's weights, resolved once; norms are F32 byte offsets in the payload.
#[derive(Clone, Copy)]
struct DraftLayer {
    q: MatrixHandle,
    k: MatrixHandle,
    v: MatrixHandle,
    o: MatrixHandle,
    gate: MatrixHandle,
    up: MatrixHandle,
    down: MatrixHandle,
    q_norm: usize,
    k_norm: usize,
    input_norm: usize,
    post_norm: usize,
}

impl DraftLayer {
    fn resolve(artifact: &MappedArtifact, layer: usize) -> crate::Result<Self> {
        let m = |name: &str| artifact.matrix_handle(&format!("layers.{layer}.{name}"));
        let f = |name: &str| Ok::<_, Error>(artifact.f32_slice(&format!("layers.{layer}.{name}"))?.1);
        Ok(Self {
            q: m("self_attn.q_proj.weight")?,
            k: m("self_attn.k_proj.weight")?,
            v: m("self_attn.v_proj.weight")?,
            o: m("self_attn.o_proj.weight")?,
            gate: m("mlp.gate_proj.weight")?,
            up: m("mlp.up_proj.weight")?,
            down: m("mlp.down_proj.weight")?,
            q_norm: f("self_attn.q_norm.weight")?,
            k_norm: f("self_attn.k_norm.weight")?,
            input_norm: f("input_layernorm.weight")?,
            post_norm: f("post_attention_layernorm.weight")?,
        })
    }
}

pub struct DsparkDrafter {
    artifact: MappedArtifact,
    layers: Vec<DraftLayer>,
    fc: MatrixHandle,
    hidden_norm: usize,
    final_norm: usize,
    w1_codes_off: usize,
    w1_groups_off: usize,
    w2_codes_off: usize,
    w2_groups_off: usize,
    conf_w_off: usize,
    conf_b_off: usize,
    kernels: DsparkKernels,
    k_caches: Vec<MetalBuffer>,
    v_caches: Vec<MetalBuffer>,
    context_len: usize,
    scored_anchors: usize,
    // Scratch buffers
    tap_proj: MetalBuffer,
    hidden: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attn_out: MetalBuffer,
    branch: MetalBuffer,
    ffn_gate: MetalBuffer,
    ffn_up: MetalBuffer,
    ffn_prod: MetalBuffer,
    positions: MetalBuffer,
    base_logits: MetalBuffer,
    special_logits: MetalBuffer,
    draft_tokens: MetalBuffer,
    conf_out: MetalBuffer,
    /// Calibrated verify cost table in ms for `0..=DSPARK_MAX_BLOCK` draft tokens.
    pub verify_ms_table: [f32; DSPARK_MAX_BLOCK + 1],
    pub draft_ms: f32,
    pub calibrator: ConfidenceCalibrator,
    last_raw_confs: Vec<f32>,
    /// Walk length of the last drafted block: the verify that ran had this many rows, so it
    /// indexes the per-block-length verify cost table.
    last_draft_block_len: usize,
    last_draft_ms: f32,
    recent_tok_sum: f32,
    recent_ms_sum: f32,
    rate_cycles: usize,
    /// Score gap between the best and second-best depth-0 candidate of the last drafted block
    /// (`conf_out[7]` from `ds_markov_walk_pick_int4`; `f32::INFINITY` before the first block).
    /// A tiny margin means the chain's first token was a coin flip, so a long verify block would
    /// mostly be discarded rows.
    last_depth0_margin: f32,
}
impl DsparkDrafter {
    pub fn open(ctx: &MetalContext, shaders: &ShaderLibrary, path: &Path) -> crate::Result<Self> {
        let artifact = MappedArtifact::open(ctx, path, DSP_MAGIC)?;
        Self::from_artifact(ctx, shaders, artifact)
    }

    /// Initialize `DsparkDrafter` directly from a shared `MappedArtifact` (for fused 3-in-1 bundles).
    pub fn from_artifact(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        artifact: MappedArtifact,
    ) -> crate::Result<Self> {
        let kernels = DsparkKernels::new(ctx, shaders)?;
        let layers = (0..DSPARK_LAYERS)
            .map(|layer| DraftLayer::resolve(&artifact, layer))
            .collect::<crate::Result<Vec<_>>>()?;
        let fc = artifact.matrix_handle("fc.weight")?;
        let hidden_norm = artifact.f32_slice("hidden_norm.weight")?.1;
        let final_norm = artifact.f32_slice("norm.weight")?.1;
        let w1_desc = artifact.descriptor("markov_head.markov_w1.weight")?;
        let w2_desc = artifact.descriptor("markov_head.markov_w2.weight")?;
        let (w1_codes_off, w1_groups_off) = (w1_desc.offset, w1_desc.aux_offset);
        let (w2_codes_off, w2_groups_off) = (w2_desc.offset, w2_desc.aux_offset);
        let conf_w_off = artifact.f32_slice("confidence_head.proj.weight")?.1;
        let conf_b_off = artifact.f32_slice("confidence_head.proj.bias")?.1;
        let dev = ctx.device();
        let kv_bytes = DSPARK_KV_RING * DSPARK_KV_TOKEN_BYTES;
        let mut k_caches = Vec::with_capacity(DSPARK_LAYERS);
        let mut v_caches = Vec::with_capacity(DSPARK_LAYERS);
        for _ in 0..DSPARK_LAYERS {
            k_caches.push(MetalBuffer::empty(dev, kv_bytes)?);
            v_caches.push(MetalBuffer::empty(dev, kv_bytes)?);
        }
        let max_rows = DSPARK_MAX_BLOCK.max(64);
        let tap_proj = MetalBuffer::empty(dev, max_rows * WIDTH * 4)?;
        let hidden = MetalBuffer::empty(dev, DSPARK_MAX_BLOCK * WIDTH * 4)?;
        let normed = MetalBuffer::empty(dev, max_rows * WIDTH * 4)?;
        let q = MetalBuffer::empty(dev, max_rows * 2048 * 4)?;
        let k = MetalBuffer::empty(dev, max_rows * 512 * 4)?;
        let v = MetalBuffer::empty(dev, max_rows * 512 * 4)?;
        let attn_out = MetalBuffer::empty(dev, DSPARK_MAX_BLOCK * 2048 * 4)?;
        let branch = MetalBuffer::empty(dev, DSPARK_MAX_BLOCK * WIDTH * 4)?;
        let ffn_gate = MetalBuffer::empty(dev, DSPARK_MAX_BLOCK * 3584 * 4)?;
        let ffn_up = MetalBuffer::empty(dev, DSPARK_MAX_BLOCK * 3584 * 4)?;
        let ffn_prod = MetalBuffer::empty(dev, DSPARK_MAX_BLOCK * 3584 * 4)?;
        let positions = MetalBuffer::empty(dev, max_rows * 4)?;
        let base_logits = MetalBuffer::empty(dev, DSPARK_MAX_BLOCK * VOCAB * 4)?;
        let special_logits = MetalBuffer::empty(dev, DSPARK_MAX_BLOCK * DSPARK_SPECIAL_COUNT * 4)?;
        let draft_tokens = MetalBuffer::empty(dev, (DSPARK_MAX_BLOCK + 3) * 4)?;
        let conf_out = MetalBuffer::empty(dev, (DSPARK_MAX_BLOCK + 2) * 4)?;
        Ok(Self {
            artifact,
            layers,
            fc,
            hidden_norm,
            final_norm,
            w1_codes_off,
            w1_groups_off,
            w2_codes_off,
            w2_groups_off,
            conf_w_off,
            conf_b_off,
            kernels,
            k_caches,
            v_caches,
            context_len: 0,
            scored_anchors: 0,
            tap_proj,
            hidden,
            normed,
            q,
            k,
            v,
            attn_out,
            branch,
            ffn_gate,
            ffn_up,
            ffn_prod,
            positions,
            base_logits,
            special_logits,
            draft_tokens,
            conf_out,
            verify_ms_table: BASE_VERIFY_MS,
            draft_ms: 0.9,
            calibrator: ConfidenceCalibrator::new(),
            last_raw_confs: Vec::with_capacity(DSPARK_MAX_BLOCK),
            last_draft_block_len: 1,
            last_draft_ms: 0.9,
            recent_tok_sum: 0.0,
            recent_ms_sum: 0.0,
            rate_cycles: 0,
            last_depth0_margin: f32::INFINITY,
        })
    }
    pub fn reset(&mut self) {
        self.context_len = 0;
        self.scored_anchors = 0;
        self.last_raw_confs.clear();
        self.last_depth0_margin = f32::INFINITY;
    }

    pub const fn truncate_context(&mut self, new_len: usize) {
        if new_len < self.context_len {
            self.context_len = new_len;
        }
    }

    #[must_use]
    pub const fn context_len(&self) -> usize {
        self.context_len
    }

    #[must_use]
    pub const fn scored_anchors(&self) -> usize {
        self.scored_anchors
    }

    pub const fn set_checkpoint_state(&mut self, context_len: usize, scored_anchors: usize) {
        self.context_len = context_len;
        self.scored_anchors = scored_anchors;
    }

    #[must_use]
    pub fn kv_caches(&self) -> (&[MetalBuffer], &[MetalBuffer]) {
        (&self.k_caches, &self.v_caches)
    }

    /// Project newly committed target hidden-state taps `[tokens, 5120]` into all 5 draft layers' KV caches.
    pub fn append_context(
        &mut self,
        ctx: &MetalContext,
        proj: &ProjectionKernels,
        ops: &NanoOps,
        taps: &MetalBuffer,
        tokens: usize,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let mut batch = CommandBatch::new_concurrent(ctx)?;
        self.encode_append_context(&mut batch, proj, ops, taps, tokens)?;
        batch.commit_and_wait()?;
        Ok(())
    }

    /// Encode `append_context` into an existing `CommandBatch` (fusing target forward + drafter KV append).
    pub fn encode_append_context(
        &mut self,
        batch: &mut CommandBatch,
        proj: &ProjectionKernels,
        ops: &NanoOps,
        taps: &MetalBuffer,
        tokens: usize,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let mut chunk_start = tokens.saturating_sub(DSPARK_KV_RING);
        while chunk_start < tokens {
            let chunk = (tokens - chunk_start).min(64);
            let (fc, _) = self.artifact.matrix_at(self.fc);
            let (h_norm, h_off) = (&self.artifact.payload_buffer, self.hidden_norm);
            let taps_offset = chunk_start * 5 * WIDTH * 4;
            proj.matmul_at(batch, fc, taps, taps_offset, &self.tap_proj, chunk as u32)?;
            ops.rms_norm(
                batch,
                &self.tap_proj,
                h_norm,
                h_off,
                &self.normed,
                WIDTH as u32,
                chunk as u32,
                DSPARK_EPSILON,
            );
            for layer in 0..DSPARK_LAYERS {
                let dl = self.layers[layer];
                let (k_proj, _) = self.artifact.matrix_at(dl.k);
                let (v_proj, _) = self.artifact.matrix_at(dl.v);
                let (k_norm, k_off) = (&self.artifact.payload_buffer, dl.k_norm);
                if let (AnyMatrix::Int4(km), AnyMatrix::Int4(vm)) = (k_proj, v_proj) {
                    proj.int4_matmul_concat2(
                        batch,
                        km,
                        vm,
                        &self.normed,
                        &self.k,
                        0,
                        &self.v,
                        0,
                        chunk as u32,
                    );
                } else {
                    batch.independent(|batch| {
                        proj.matmul(batch, k_proj, &self.normed, &self.k, chunk as u32)?;
                        proj.matmul(batch, v_proj, &self.normed, &self.v, chunk as u32)
                    })?;
                }
                self.kernels.k_norm_rope_kv_store_f16(
                    batch,
                    (&self.k, 0),
                    (&self.v, 0),
                    (k_norm, k_off),
                    &self.k_caches[layer],
                    &self.v_caches[layer],
                    (self.context_len + chunk_start) as u32,
                    chunk as u32,
                    DSPARK_EPSILON,
                    DSPARK_ROPE_THETA,
                );
            }
            chunk_start += chunk;
            if chunk_start < tokens {
                batch.submit_and_renew()?;
            }
        }
        self.context_len += tokens;
        Ok(())
    }


    /// Encode one `DSpark` semi-autoregressive drafting pass with post-Markov target repetition
    /// and frequency penalties (`0.10 * (cnt - 1)`) applied inside `ds_markov_walk_pick_int4`.
    #[expect(unsafe_code)]
    pub fn encode_draft_block_with_penalties(
        &mut self,
        batch: &mut CommandBatch,
        proj: &ProjectionKernels,
        ops: &NanoOps,
        rot512_1024: &SignedHadamard512,
        target_embed: (AnyMatrix<'_>, bool),
        target_lm_head: (AnyMatrix<'_>, bool),
        anchor_token: u32,
        max_draft: usize,
        recent_counts: &[[u32; 2]],
        rep_penalty: f32,
    ) -> crate::Result<usize> {
        let markov_len = max_draft.clamp(1, DSPARK_MAX_BLOCK);
        let block_len = DSPARK_MAX_BLOCK;
        unsafe {
            let toks = self.draft_tokens.as_mut_slice::<u32>();
            toks[0] = anchor_token;
            for slot in &mut toks[1..block_len] {
                *slot = DSPARK_MASK_TOKEN;
            }
            let pos = self.positions.as_mut_slice::<u32>();
            for (idx, slot) in pos[..block_len].iter_mut().enumerate() {
                *slot = (self.context_len + idx) as u32;
            }
        }

        match target_embed.0 {
            AnyMatrix::Bf16(mat) => {
                proj.embed_gather_bf16(
                    batch,
                    mat,
                    &self.draft_tokens,
                    &self.hidden,
                    block_len as u32,
                );
            }
            AnyMatrix::Int8(mat) => {
                proj.embed_gather_int8(
                    batch,
                    mat,
                    &self.draft_tokens,
                    &self.hidden,
                    block_len as u32,
                );
            }
            AnyMatrix::Ptq1(mat) => {
                if target_embed.1 {
                    proj.embed_gather_ptq1_fwht512(
                        batch,
                        mat,
                        &self.draft_tokens,
                        rot512_1024,
                        &self.hidden,
                        block_len as u32,
                    );
                } else {
                    proj.embed_gather_ptq1(
                        batch,
                        mat,
                        &self.draft_tokens,
                        &self.hidden,
                        block_len as u32,
                    );
                }
            }
            AnyMatrix::Int4(mat) => {
                proj.embed_gather_int4(
                    batch,
                    mat,
                    &self.draft_tokens,
                    &self.hidden,
                    block_len as u32,
                );
            }
        }

        let total_keys = (self.context_len + block_len) as u32;
        let payload = &self.artifact.payload_buffer;
        let (in_norm0, in_off0) = (payload, self.layers[0].input_norm);
        ops.rms_norm(
            batch,
            &self.hidden,
            in_norm0,
            in_off0,
            &self.normed,
            WIDTH as u32,
            block_len as u32,
            DSPARK_EPSILON,
        );
        for layer in 0..DSPARK_LAYERS {
            let dl = self.layers[layer];
            let (q_proj, _) = self.artifact.matrix_at(dl.q);
            let (k_proj, _) = self.artifact.matrix_at(dl.k);
            let (v_proj, _) = self.artifact.matrix_at(dl.v);
            let (o_proj, _) = self.artifact.matrix_at(dl.o);
            let (q_norm, q_off) = (payload, dl.q_norm);
            let (k_norm, k_off) = (payload, dl.k_norm);

            if let (AnyMatrix::Int4(qm), AnyMatrix::Int4(km), AnyMatrix::Int4(vm)) =
                (q_proj, k_proj, v_proj)
            {
                proj.int4_matmul_concat3(
                    batch,
                    qm,
                    km,
                    vm,
                    &self.normed,
                    &self.q,
                    &self.k,
                    &self.v,
                    block_len as u32,
                );
            } else {
                batch.independent(|batch| {
                    proj.matmul(batch, q_proj, &self.normed, &self.q, block_len as u32)?;
                    proj.matmul(batch, k_proj, &self.normed, &self.k, block_len as u32)?;
                    proj.matmul(batch, v_proj, &self.normed, &self.v, block_len as u32)
                })?;
            }
            self.kernels.qk_norm_rope_kv_store_f16(
                batch,
                &self.q,
                &self.k,
                &self.v,
                (q_norm, q_off),
                (k_norm, k_off),
                &self.positions,
                &self.k_caches[layer],
                &self.v_caches[layer],
                self.context_len as u32,
                block_len as u32,
                DSPARK_EPSILON,
                DSPARK_ROPE_THETA,
            );
            self.kernels.block_attention(
                batch,
                &self.q,
                &self.k_caches[layer],
                &self.v_caches[layer],
                &self.attn_out,
                block_len as u32,
                total_keys,
                self.context_len.saturating_sub(DSPARK_WINDOW) as u32,
            );
            proj.matmul(batch, o_proj, &self.attn_out, &self.branch, block_len as u32)?;

            let (post_norm, post_off) = (payload, dl.post_norm);
            ops.residual_add_rms(
                batch,
                &self.hidden,
                &self.branch,
                post_norm,
                post_off,
                &self.tap_proj,
                &self.normed,
                WIDTH as u32,
                block_len as u32,
                DSPARK_EPSILON,
            );

            let (gate_proj, _) = self.artifact.matrix_at(dl.gate);
            let (up_proj, _) = self.artifact.matrix_at(dl.up);
            let (down_proj, _) = self.artifact.matrix_at(dl.down);
            if matches!((gate_proj, up_proj), (AnyMatrix::Int4(_), AnyMatrix::Int4(_))) {
                proj.matmul_swiglu(
                    batch,
                    gate_proj,
                    up_proj,
                    &self.normed,
                    &self.ffn_prod,
                    block_len as u32,
                )?;
            } else {
                batch.independent(|batch| {
                    proj.matmul(batch, gate_proj, &self.normed, &self.ffn_gate, block_len as u32)?;
                    proj.matmul(batch, up_proj, &self.normed, &self.ffn_up, block_len as u32)
                })?;
                ops.swiglu(
                    batch,
                    &self.ffn_gate,
                    &self.ffn_up,
                    &self.ffn_prod,
                    (block_len * 3584) as u32,
                );
            }
            proj.matmul(batch, down_proj, &self.ffn_prod, &self.branch, block_len as u32)?;
            if layer + 1 < DSPARK_LAYERS || !target_lm_head.1 {
                let (next_norm, next_off) = if layer + 1 < DSPARK_LAYERS {
                    (payload, self.layers[layer + 1].input_norm)
                } else {
                    (payload, self.final_norm)
                };
                ops.residual_add_rms(
                    batch,
                    &self.tap_proj,
                    &self.branch,
                    next_norm,
                    next_off,
                    &self.hidden,
                    &self.normed,
                    WIDTH as u32,
                    block_len as u32,
                    DSPARK_EPSILON,
                );
            } else {
                ops.residual_add_rms_fwht512(
                    batch,
                    &self.tap_proj,
                    &self.branch,
                    payload,
                    self.final_norm,
                    &rot512_1024.signs,
                    &self.hidden,
                    &self.normed,
                    WIDTH as u32,
                    block_len as u32,
                    DSPARK_EPSILON,
                );
            }
        }

        let anchor = anchor_token as usize;
        self.scored_anchors = if (DSPARK_HEAD_CUT..DSPARK_SPECIAL_FIRST).contains(&anchor) {
            0
        } else {
            (self.scored_anchors + 1).min(DSPARK_REDUCED_ANCHORS)
        };
        let active_vocab = if self.scored_anchors >= DSPARK_REDUCED_ANCHORS {
            DSPARK_HEAD_CUT
        } else {
            VOCAB
        };
        let active_head = target_lm_head.0.slice_rows(0, active_vocab as u32);

        // Compute base logits [block_len, active_vocab] (plus special_logits when reduced) via target's shared lm_head
        let lm_in = &self.normed;
        if active_vocab == DSPARK_HEAD_CUT {
            let special_head = target_lm_head
                .0
                .slice_rows(DSPARK_SPECIAL_FIRST as u32, DSPARK_SPECIAL_COUNT as u32);
            batch.independent(|batch| {
                proj.matmul(batch, active_head, lm_in, &self.base_logits, markov_len as u32)?;
                proj.matmul(batch, special_head, lm_in, &self.special_logits, markov_len as u32)
            })?;
        } else {
            proj.matmul(
                batch,
                active_head,
                lm_in,
                &self.base_logits,
                markov_len as u32,
            )?;
        }
        // Sequential Markov + Confidence chain encoded into the same command buffer!
        let freq_penalty = if rep_penalty > 1.0001 { 0.10 } else { 0.0 };
        self.kernels.markov_walk_int4(
            batch,
            (payload, self.w1_codes_off),
            (payload, self.w1_groups_off),
            (payload, self.w2_codes_off),
            (payload, self.w2_groups_off),
            &self.normed,
            &self.base_logits,
            &self.special_logits,
            (payload, self.conf_w_off),
            (payload, self.conf_b_off),
            &self.draft_tokens,
            &self.conf_out,
            markov_len as u32,
            active_vocab as u32,
            recent_counts,
            rep_penalty,
            freq_penalty,
        );
        Ok(markov_len)
    }

    /// Read `draft_tokens` and `conf_out` after `batch.commit_and_wait()`, run calibration and
    /// `schedule_verification_length`, and return `(scheduled_draft_tokens, confidences)`.
    /// With `verified_all` the whole block was already verified, so every draft is kept.
    #[expect(unsafe_code)]
    pub fn finish_draft_block(
        &mut self,
        block_len: usize,
        draft_elapsed_ms: f32,
        verified_all: bool,
    ) -> (Vec<u32>, Vec<f32>) {
        let (all_drafts, raw_confs, margin) = unsafe {
            let draft_full = self.draft_tokens.as_slice::<u32>();
            let conf_full = self.conf_out.as_slice::<f32>();
            let margin = if draft_full[8] == u32::MAX {
                f32::INFINITY
            } else {
                conf_full[7]
            };
            (
                draft_full[1..=block_len].to_vec(),
                conf_full[..block_len].to_vec(),
                margin,
            )
        };
        self.last_depth0_margin = margin;
        let calibrated_confs: Vec<f32> = raw_confs
            .iter()
            .enumerate()
            .map(|(depth, &raw)| self.calibrator.calibrate(depth, raw))
            .collect();
        self.last_draft_ms = draft_elapsed_ms;
        let rate = if self.rate_cycles >= 16 && self.recent_ms_sum > 0.1 {
            self.recent_tok_sum / self.recent_ms_sum
        } else {
            0.0
        };
        let (scheduled_len, _) = schedule_verification_length_with_rate(
            &calibrated_confs,
            self.draft_ms,
            &self.verify_ms_table,
            rate,
        );
        let kept = if verified_all { block_len } else { scheduled_len.clamp(1, block_len) };
        // Keep the FULL block's raw confs: the calibrator must observe every depth every round
        // (undrafted depths as not-accepted), or the deep-depth acceptance rates become
        // survivorship-biased and the scheduler keeps drafting blocks that mostly roll back.
        self.last_raw_confs.clear();
        self.last_raw_confs.extend_from_slice(&raw_confs[..block_len.min(raw_confs.len())]);
        self.last_draft_block_len = block_len;
        (
            all_drafts[..kept].to_vec(),
            calibrated_confs[..kept].to_vec(),
        )
    }

    /// `[anchor, drafts...]` of the block last encoded by `encode_draft_block*`, on the GPU.
    #[must_use]
    pub const fn draft_tokens(&self) -> &MetalBuffer {
        &self.draft_tokens
    }

    /// Optimal number of Markov / `lm_head` draft steps (`3..=7`) to evaluate after the full 7-token block attention pass.
    #[must_use]
    pub fn recommended_forward_len(&self, max_draft: usize) -> usize {
        // Scheduler confidences per depth: the empirical acceptance rate (what really happened
        // in recent rounds) once enough observations exist, else the calibrated flat estimate.
        // The flat estimate through the learned logistic proved optimistic on dense-reasoning
        // prompts, where real acceptance at depth >= 2 collapses toward 0.2-0.3 while the
        // calibrated model still predicted ~0.5 - so the scheduler kept drafting 4-7 rows that
        // were mostly discarded. Empirical rates are unfoolable: they are the outcomes.
        let est_confs: [f32; DSPARK_MAX_BLOCK] = std::array::from_fn(|d| {
            self.calibrator
                .empirical_acceptance(d)
                .unwrap_or_else(|| self.calibrator.calibrate(d, 0.55))
        });
        let rate = if self.rate_cycles >= 16 && self.recent_ms_sum > 0.1 {
            self.recent_tok_sum / self.recent_ms_sum
        } else {
            0.0
        };
        let (sched, _) = schedule_verification_length_with_rate(
            &est_confs,
            self.draft_ms,
            &self.verify_ms_table,
            rate,
        );
        // The depth-0 score margin is a second signal the calibrated confidences do not capture:
        // a coin-flip first token predicts a mostly-discarded verify block regardless of the
        // confidence EMA, so cap the next block short and re-widen once a confident pick returns.
        let margin_cap = if self.last_depth0_margin < DEPTH0_MARGIN_AMBIGUOUS {
            3
        } else {
            DSPARK_MAX_BLOCK
        };
        // Once the token/ms rate is learned, trust the scheduler's optimum exactly: on prompts
        // where deep speculation loses (low acceptance, e.g. dense reasoning), both the old
        // `+2` premium and the `4` floor wasted verify rows and made speculation a net loss vs
        // plain decode (measured: 115 vs 134 tok/s). The premium dates from unfused rounds,
        // where extra walk steps overlapped the verify wait; under fusion the whole round is
        // one serial batch, so every drafted row costs its verify. During warmup the
        // conservative floor stays: with no rate data the flat-confidence estimate would
        // otherwise collapse the block to 1 and disable speculation before it calibrates.
        // `max_draft` drops below 4 near `max_new_tokens`; `clamp` would panic there.
        // The +2 premium pays only when deep rows are actually accepted (high-acceptance
        // prompts gained from it); with the learned rate and empirical depths in hand, apply it
        // only there, and trust the raw scheduler optimum otherwise.
        let deep_pays = self
            .calibrator
            .empirical_acceptance(3)
            .is_some_and(|a| a > 0.5);
        let sched = if self.rate_cycles < 16 || deep_pays {
            sched + 2
        } else {
            sched
        };
        let floor = if self.rate_cycles < 16 || deep_pays { 4 } else { 1 };
        sched
            .max(floor)
            .min(max_draft.min(margin_cap))
            .max(1)
    }

    /// Asynchronously run one `DSpark` semi-autoregressive drafting pass on the GPU with
    /// target repetition and frequency penalties applied post-Markov.
    pub async fn draft_block_async_with_penalties(
        &mut self,
        ctx: &MetalContext,
        proj: &ProjectionKernels,
        ops: &NanoOps,
        rot512_1024: &SignedHadamard512,
        target_embed: (AnyMatrix<'_>, bool),
        target_lm_head: (AnyMatrix<'_>, bool),
        anchor_token: u32,
        max_draft: usize,
        recent_counts: &[[u32; 2]],
        rep_penalty: f32,
    ) -> crate::Result<(Vec<u32>, Vec<f32>)> {
        let t_draft = Instant::now();
        let (block_len, fut) = {
            let mut batch = CommandBatch::new_concurrent(ctx)?;
            let block_len = self.encode_draft_block_with_penalties(
                &mut batch,
                proj,
                ops,
                rot512_1024,
                target_embed,
                target_lm_head,
                anchor_token,
                max_draft,
                recent_counts,
                rep_penalty,
            )?;
            (block_len, batch.commit_and_await())
        };
        fut.await?;
        let elapsed_ms = (t_draft.elapsed().as_secs_f64() * 1000.0) as f32;
        Ok(self.finish_draft_block(block_len, elapsed_ms, false))
    }

    /// Record verification outcome for the most recent `draft_block`, updating the online
    /// Newton-Raphson confidence calibrator and `draft_ms` / `verify_ms_table` EMAs.
    pub fn record_verification(&mut self, accepted_len: usize, verify_ms: f32, cycle_ms: f32) {
        if self.last_raw_confs.is_empty() {
            return;
        }
        let drafted_len = self.last_draft_block_len.min(DSPARK_MAX_BLOCK);
        self.calibrator.observe(&self.last_raw_confs, accepted_len);
        if self.last_draft_ms > 0.05 && self.last_draft_ms < 20.0 {
            self.draft_ms = 0.85_f32.mul_add(self.draft_ms, 0.15 * self.last_draft_ms);
        }
        if verify_ms > 0.1 && verify_ms < 50.0 {
            self.verify_ms_table[drafted_len] =
                0.85_f32.mul_add(self.verify_ms_table[drafted_len], 0.15 * verify_ms);
        } else if cycle_ms > 0.1 && cycle_ms < 200.0 {
            // Fused rounds cannot split draft from verify: the whole batch is the block-length
            // cycle cost. Attribute the draft EMA's share to the draft and store the remainder
            // as this block length's verify cost — otherwise the table keeps stale unfused
            // prices, which overstate fused verify costs where they matter and understate them
            // where they don't, biasing block depth either way.
            let fused_verify = (cycle_ms - self.draft_ms).max(0.2);
            self.verify_ms_table[drafted_len] =
                0.85_f32.mul_add(self.verify_ms_table[drafted_len], 0.15 * fused_verify);
        }
        // The token/ms rate learns from every round: fused rounds report their whole batch as
        // the cycle cost (they cannot split draft from verify), unfused rounds report
        // draft + verify. Without this, fusion would freeze `rate_cycles` at its warmup value
        // and the scheduler would never trust its own optimum.
        if cycle_ms > 0.1 && cycle_ms < 200.0 {
            let cyc_toks = (accepted_len + 1) as f32;
            self.recent_tok_sum = 0.92_f32.mul_add(self.recent_tok_sum, cyc_toks);
            self.recent_ms_sum = 0.92_f32.mul_add(self.recent_ms_sum, cycle_ms);
            self.rate_cycles += 1;
        }
        self.last_raw_confs.clear();
    }
}
