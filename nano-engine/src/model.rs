//! Native Metal execution of MiniCPM-V 4.7 1B (`openbmb/MiniCPM-V-4.7-1B`):
//! - 24-layer Qwen3.5-0.8B hybrid backbone (`width = 1024`, `ffn = 3584`, `vocab = 248,094`):
//!   - 18 Gated `DeltaNet` recurrent layers (`layers % 4 != 3`), F16 state `[16 * 128][128]` with zero-copy rollback
//!   - 6 Full-Attention layers (`layers 3, 7, 11, 15, 19, 23`), rotated Q8 KV cache (544 B/token/layer = 3.19 KiB/token total)
//! - Canvas 3D M-RoPE (`sections = [11, 11, 10]`, `partial_rotary_factor = 0.25`)
//! - Optional `SigLIP2` + Window-Attention Merger vision tower (`16x` and `4x` modes)
//! - Speculative decoding: `DSpark` 5-layer semi-autoregressive drafter + N-gram prompt-lookup drafter

use std::path::Path;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};
use std::time::Instant;
use nano_metal::batch::{BufferCopyRequest, CommandBatch};
use nano_metal::buffer::MetalBuffer;
use nano_metal::context::MetalContext;
use nano_metal::ops::{
    AttentionWorkspace, CONV_CHANNELS, GDN_DIM, GDN_HEADS, GDN_OUT_WIDTH, KvFormat, MODEL_FFN,
    MODEL_WIDTH, NanoOps, Rope,
};
use nano_metal::projection::{AnyMatrix, ProjectionKernels, SignedHadamard512};
use nano_metal::sampling::SamplingKernels;
use nano_metal::shaders::ShaderLibrary;
use serde::Serialize;
use crate::tokenizer::Tokenizer;

use crate::Error;
use crate::artifact::{LM_MAGIC, MappedArtifact, MatrixHandle};
use crate::ane_ffn::{
    ANE_MAX_ROWS, ANE_MIN_ROWS, AneFfnOffload, AneState, check_rows_and_tolerance,
    relative_difference,
};
use crate::dspark::{
    DSPARK_KV_RING, DSPARK_KV_TOKEN_BYTES, DSPARK_TAPS, DsparkDrafter, lookup_ngram_draft_adaptive,
};
use crate::prompt_cache::DiskCache;
use crate::quant::deterministic_signs;
use crate::sampler::{Sampler, SamplingParams};
use crate::vision::{
    DownsampleMode, IMAGE_END_ID, IMAGE_ID_END, IMAGE_ID_START, IMAGE_START_ID, IMAGE_TOKEN_ID,
    NEWLINE_ID, PreparedImage, SLICE_END_ID, SLICE_START_ID, VisionEncoder, compute_canvas_mrope,
};

pub const LAYERS: usize = 24;
pub const WIDTH: usize = MODEL_WIDTH as usize;
pub const FFN: usize = MODEL_FFN as usize;
pub const VOCAB: usize = 248_094;
pub const FULL_INTERVAL: usize = 4;
pub const EPSILON: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10_000_000.0;
pub const MAX_VERIFY_TOKENS: usize = 8;
pub const MAX_CONTEXT_TOKENS: usize = 262_144;
/// Rows per `score_tokens` pass (capped by `MemoryProfile::prefill_rows`). Wider blocks would take
/// the ANE's fp16 FFN (mean |Δ logprob| ~0.007 vs ~0.001 between GPU block sizes) and 2-4x the
/// logits buffer for little speed.
const SCORE_ROWS: usize = 128;
const _: () = assert!(SCORE_ROWS < ANE_MIN_ROWS);
/// KV caches start this small and double on demand up to `MAX_CONTEXT_TOKENS`.
pub const INITIAL_KV_CAPACITY: usize = 1_024;
/// Layers timed when deciding between the ANE and the GPU (a full 24-layer pass cost ~1 s on
/// an iPhone SE 3, where the GPU half runs at 512 rows).
const CALIBRATION_LAYERS: usize = 4;
pub const IM_START_ID: u32 = 248_045;
pub const IM_END_ID: u32 = 248_046;
pub const THINK_START_ID: u32 = 248_068;
pub const THINK_END_ID: u32 = 248_069;
const TOOL_CALL_START_ID: u32 = 248_058;
const FENCE_NEWLINE_ID: u32 = 74_694;
const FENCE_BARE_ID: u32 = 13_874;
struct ThreadParker(Thread);

impl Wake for ThreadParker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on_future<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let waker = Waker::from(Arc::new(ThreadParker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => thread::park(),
        }
    }
}

/// Top token, its log-probability and `next`'s log-probability under one row of logits.
fn row_score(row: &[f32], next: Option<u32>) -> TokenScore {
    let (top, max) = row
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(ti, tm), (i, &l)| if l > tm { (i, l) } else { (ti, tm) });
    let log_sum = row.iter().map(|&l| (l - max).exp()).sum::<f32>().ln() + max;
    TokenScore {
        top: top as u32,
        top_logprob: max - log_sum,
        next_logprob: next.map(|n| row[n as usize] - log_sum),
    }
}
/// Output of MiniCPM-V 4.7 1B generation with detailed timing and speculation telemetry.
#[derive(Clone, Debug, Serialize)]
pub struct GenerationReport {
    pub text: String,
    pub reasoning_text: Option<String>,
    pub prompt_tokens: usize,
    pub cached_prefix_tokens: usize,
    pub generated_tokens: usize,
    pub token_ids: Vec<u32>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub speculative_rounds: usize,
    pub drafted_tokens: usize,
    pub accepted_tokens: usize,
    pub ngram_accepted_tokens: usize,
}

/// Next-token statistics at one position of a teacher-forced prompt (`NanoEngine::score_tokens`).
#[derive(Clone, Copy, Debug)]
pub struct TokenScore {
    /// The model's most likely next token and its log-probability.
    pub top: u32,
    pub top_logprob: f32,
    /// Log-probability of the prompt's actual next token (`None` at the last position).
    pub next_logprob: Option<f32>,
}

/// Engine sizing derived from the memory the device lets this process use. The language model
/// always gets the full `MAX_CONTEXT_TOKENS` window; smaller budgets store its KV cache in 4 bits.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct MemoryProfile {
    /// Bytes available at open: the jetsam headroom on iOS, Metal's recommended working set elsewhere.
    pub budget_bytes: u64,
    /// Prompt rows per prefill pass; prefill scratch costs ~190 KiB per row.
    pub prefill_rows: usize,
    /// Full-attention KV precision: 8 bits when a full-window cache fits, else 4.
    pub kv_bits: u8,
    /// Prefix-cache checkpoints kept (~10.4 MiB of recurrent state each, plus their KV).
    pub prefix_checkpoints: usize,
    /// Byte budget of the on-disk prompt cache (`prompt_cache`), least recently used files evicted first.
    pub prompt_cache_bytes: u64,
    /// Keep the vision tower's ~128 MiB of scratch between images.
    pub keep_vision: bool,
}

impl MemoryProfile {
    /// Mac-class budgets keep the fastest settings; smaller ones (phones) trade prefill batch,
    /// KV precision and caches for headroom.
    #[must_use]
    pub fn for_budget(budget_bytes: u64) -> Self {
        // A doubling cache briefly holds its old and new copy: 1.5x its final size.
        let peak = |bytes_per_token: usize, tokens: usize| (bytes_per_token * tokens) as u64 * 3 / 2;
        let lm_q8 = peak(12 * KvFormat::Q8.token_bytes(), MAX_CONTEXT_TOKENS); // 6 layers × K, V
        let roomy = budget_bytes >= 8 << 30;
        Self {
            budget_bytes,
            prefill_rows: if roomy { 512 } else { 128 },
            kv_bits: if lm_q8 <= budget_bytes / 2 { 8 } else { 4 },
            prefix_checkpoints: if roomy { 12 } else { 2 },
            prompt_cache_bytes: if roomy { 4 << 30 } else { 512 << 20 },
            keep_vision: roomy,
        }
    }

    /// Profile for this device: its memory budget, and whether the ANE is reachable. The ANE's
    /// fixed per-call cost needs wide prefill chunks to pay off (iPhone SE 3, 864-token prompt:
    /// 252 tok/s at 128 rows, 419 at 512, GPU alone ~290 at any width), so with an ANE the
    /// chunks are as wide as its programs allow.
    #[must_use]
    pub fn detect() -> Self {
        let mut profile = Self::for_budget(nano_metal::memory::budget());
        if profile.budget_bytes >= 1 << 30 && nano_metal::ane::unavailable().is_none() {
            profile.prefill_rows = ANE_MAX_ROWS;
        }
        profile
    }
}

/// Saved recurrent + KV cache state at an exact prefix boundary in unified memory.
pub struct PrefixCheckpoint {
    pub tokens: Vec<u32>,
    pub rope_delta: i32,
    pub dspark_scored_anchors: usize,
    recurrent: Vec<(MetalBuffer, MetalBuffer)>,
    attn_kv: Vec<(MetalBuffer, MetalBuffer)>,
    dspark_kv: Vec<(MetalBuffer, MetalBuffer)>,
    last_used_tick: u64,
    pinned_system: bool,
}

struct RecurrentState {
    /// F16 state `[16 * 128][128]` (65,536 bytes)
    state: MetalBuffer,
    /// F16 rollback state for copy-free speculative verification
    rollback_state: MetalBuffer,
    /// F32 conv history `[6144 * 3]` (73,728 bytes)
    history: MetalBuffer,
    /// F32 rollback conv history
    rollback_history: MetalBuffer,
    /// Recorded verify block `qkv` `[8 * 6144]` (196,608 B) for exact conv history replay
    tape_qkv: MetalBuffer,
    /// Recorded verify block `convolved` `[8 * 6144]` (196,608 B) for zero-pass partial acceptance replay
    tape_convolved: MetalBuffer,
    /// Recorded verify block `decay` `[8 * 16]` (512 B)
    tape_decay: MetalBuffer,
    /// Recorded verify block `beta` `[8 * 16]` (512 B)
    tape_beta: MetalBuffer,
}

struct FullAttentionCache {
    k_cache: MetalBuffer,
    v_cache: MetalBuffer,
}

/// One layer's weights, resolved by name once at open so encoding a step does no lookups.
/// Norm and per-channel vectors are byte offsets of F32 tensors in the artifact payload.
#[derive(Clone, Copy)]
struct LayerWeights {
    input_norm: usize,
    post_norm: usize,
    mixer: Mixer,
    gate: MatrixHandle,
    up: MatrixHandle,
    down: MatrixHandle,
}

#[derive(Clone, Copy)]
enum Mixer {
    Attention {
        q: MatrixHandle,
        k: MatrixHandle,
        v: MatrixHandle,
        o: MatrixHandle,
        q_norm_at: usize,
        k_norm_at: usize,
    },
    Recurrent {
        qkv: MatrixHandle,
        z: MatrixHandle,
        a: MatrixHandle,
        b: MatrixHandle,
        out: MatrixHandle,
        conv_at: usize,
        a_log_at: usize,
        dt_bias_at: usize,
        norm_at: usize,
    },
}

impl LayerWeights {
    fn resolve(artifact: &MappedArtifact, layer: usize) -> crate::Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}");
        let m = |name: &str| artifact.matrix_handle(&format!("{prefix}.{name}"));
        let f = |name: &str| Ok::<_, Error>(artifact.f32_slice(&format!("{prefix}.{name}"))?.1);
        let mixer = if layer % FULL_INTERVAL == FULL_INTERVAL - 1 {
            Mixer::Attention {
                q: m("self_attn.q_proj.weight")?,
                k: m("self_attn.k_proj.weight")?,
                v: m("self_attn.v_proj.weight")?,
                o: m("self_attn.o_proj.weight")?,
                q_norm_at: f("self_attn.q_norm.weight")?,
                k_norm_at: f("self_attn.k_norm.weight")?,
            }
        } else {
            Mixer::Recurrent {
                qkv: m("linear_attn.in_proj_qkv.weight")?,
                z: m("linear_attn.in_proj_z.weight")?,
                a: m("linear_attn.in_proj_a.weight")?,
                b: m("linear_attn.in_proj_b.weight")?,
                out: m("linear_attn.out_proj.weight")?,
                conv_at: f("linear_attn.conv1d.weight")?,
                a_log_at: f("linear_attn.A_log")?,
                dt_bias_at: f("linear_attn.dt_bias")?,
                norm_at: f("linear_attn.norm.weight")?,
            }
        };
        Ok(Self {
            input_norm: f("input_layernorm.weight")?,
            post_norm: f("post_attention_layernorm.weight")?,
            mixer,
            gate: m("mlp.gate_proj.weight")?,
            up: m("mlp.up_proj.weight")?,
            down: m("mlp.down_proj.weight")?,
        })
    }
}

/// Prompt-cache directory key: the first 16 hex digits of the payload SHA-256. The header is
/// untrusted and the key names a directory, so anything but hex (`..`, `/`) disables the cache.
pub(crate) fn prompt_cache_key(payload_sha256: &str) -> Option<&str> {
    payload_sha256.get(..16).filter(|k| k.bytes().all(|b| b.is_ascii_hexdigit()))
}

pub struct NanoEngine {
    ctx: MetalContext,
    shaders: ShaderLibrary,
    proj: ProjectionKernels,
    ops: NanoOps,
    sampling: SamplingKernels,
    artifact: MappedArtifact,
    layers: Vec<LayerWeights>,
    embed: MatrixHandle,
    lm_head: MatrixHandle,
    /// Byte offset of the final norm's F32 weights in the artifact payload.
    final_norm: usize,
    tokenizer: Option<Tokenizer>,
    drafter: Option<DsparkDrafter>,
    vision: Option<VisionEncoder>,
    vision_path: Option<std::path::PathBuf>,
    ane: AneState,
    rope_rows_buf: MetalBuffer,
    rot512_1024: SignedHadamard512,
    recurrent: Vec<RecurrentState>,
    attention_caches: Vec<FullAttentionCache>,
    workspace: AttentionWorkspace,
    kv_format: KvFormat,
    position: usize,
    kv_capacity: usize,
    profile: MemoryProfile,
    rope_delta: i32,
    // Scratch buffers sized for up to `profile.prefill_rows` rows
    token_ids_buf: MetalBuffer,
    hidden: MetalBuffer,
    hidden_alt: MetalBuffer,
    normed: MetalBuffer,
    rotated_1024: MetalBuffer,
    branch: MetalBuffer,
    qkv: MetalBuffer,
    convolved: MetalBuffer,
    gdn_gate: MetalBuffer,
    alpha: MetalBuffer,
    raw_beta: MetalBuffer,
    decay: MetalBuffer,
    beta: MetalBuffer,
    gdn_out: MetalBuffer,
    attn_qg: MetalBuffer,
    attn_k: MetalBuffer,
    attn_v: MetalBuffer,
    attn_q: MetalBuffer,
    attn_gate: MetalBuffer,
    attn_out: MetalBuffer,
    ffn_gate: MetalBuffer,
    ffn_up: MetalBuffer,
    ffn_prod: MetalBuffer,
    logits: MetalBuffer,
    /// `SCORE_ROWS`-row logits for `score_tokens`, allocated on its first call.
    score_logits: Option<MetalBuffer>,
    taps_buf: MetalBuffer,
    rep_penalty: f32,
    recent_token_counts: Vec<[u32; 2]>,
    dry_scratch: Vec<[u32; 2]>,
    active_tokens: Vec<u32>,
    prefix_cache: Vec<PrefixCheckpoint>,
    cache_tick: u64,
    prefix_cache_enabled: bool,
    /// Checkpoints persisted next to the model; `None` when that directory is not writable.
    disk_cache: Option<DiskCache>,
}

impl NanoEngine {
    pub const DEFAULT_SYSTEM_PROMPT: &'static str = "You are a helpful assistant.";
    /// Open the model, sizing every buffer for this device's memory budget.
    pub fn open(model_path: &Path) -> crate::Result<Self> {
        Self::open_with_profile(model_path, MemoryProfile::detect())
    }

    pub(crate) fn open_with_profile(model_path: &Path, profile: MemoryProfile) -> crate::Result<Self> {
        let parent = model_path.parent();
        let find_sibling = |filename: &str| -> Option<std::path::PathBuf> {
            parent.map(|p| p.join(filename)).filter(|p| p.is_file())
        };
        // Map (or build once from `tokenizer.json`) the tokenizer while pipelines build.
        let tokenizer_dir = parent.map(Path::to_path_buf);
        let tokenizer_job =
            thread::spawn(move || tokenizer_dir.map_or(Ok(None), |dir| Tokenizer::load(&dir)));
        let ctx = MetalContext::new()?;
        let shaders = ShaderLibrary::new(ctx.device())?;
        let proj = ProjectionKernels::new(&shaders)?;
        let ops = NanoOps::new(&shaders)?;
        let sampling = SamplingKernels::new(&ctx, &shaders, VOCAB, MAX_VERIFY_TOKENS)?;
        let artifact = MappedArtifact::open(&ctx, model_path, LM_MAGIC)?;
        let layers = (0..LAYERS)
            .map(|layer| LayerWeights::resolve(&artifact, layer))
            .collect::<crate::Result<Vec<_>>>()?;
        let embed = artifact.matrix_handle("model.language_model.embed_tokens.weight")?;
        let lm_head = artifact.matrix_handle("lm_head.weight")?;
        let final_norm = artifact.f32_slice("model.language_model.norm.weight")?.1;
        let tokenizer = tokenizer_job
            .join()
            .map_err(|_| Error::Tokenizer("tokenizer loader thread panicked".into()))??;

        let drafter = if artifact
            .header
            .tensors
            .contains_key("markov_head.markov_w1.weight")
        {
            Some(DsparkDrafter::from_artifact(&ctx, &shaders, artifact.clone())?)
        } else if let Some(dp) = find_sibling("minicpm-v-4.7-1b-dspark.bin") {
            Some(DsparkDrafter::open(&ctx, &shaders, &dp)?)
        } else {
            None
        };

        // The vision tower and its scratch are built on the first image.
        let vision_path = find_sibling("minicpm-v-4.7-1b-vision.bin");
        let rot512_1024 = SignedHadamard512::new(&ctx, &deterministic_signs(1024))?;

        let dev = ctx.device();
        let kv_format = if profile.kv_bits == 4 { KvFormat::Q4 } else { KvFormat::Q8 };
        let state_bytes = (GDN_HEADS * GDN_DIM * GDN_DIM) as usize * 2; // F16 = 65,536 B
        let hist_bytes = (CONV_CHANNELS as usize) * 3 * 4; // F32 = 73,728 B

        let mut recurrent = Vec::with_capacity(18);
        for _ in 0..18 {
            let s = MetalBuffer::empty(dev, state_bytes)?;
            s.clear();
            let rs = MetalBuffer::empty(dev, state_bytes)?;
            rs.clear();
            let h = MetalBuffer::empty(dev, hist_bytes)?;
            h.clear();
            let rh = MetalBuffer::empty(dev, hist_bytes)?;
            rh.clear();
            let tq = MetalBuffer::empty(dev, MAX_VERIFY_TOKENS * CONV_CHANNELS as usize * 4)?;
            let tc = MetalBuffer::empty(dev, MAX_VERIFY_TOKENS * CONV_CHANNELS as usize * 4)?;
            let td = MetalBuffer::empty(dev, MAX_VERIFY_TOKENS * GDN_HEADS as usize * 4)?;
            let tb = MetalBuffer::empty(dev, MAX_VERIFY_TOKENS * GDN_HEADS as usize * 4)?;
            recurrent.push(RecurrentState {
                state: s,
                rollback_state: rs,
                history: h,
                rollback_history: rh,
                tape_qkv: tq,
                tape_convolved: tc,
                tape_decay: td,
                tape_beta: tb,
            });
        }

        let init_cap = INITIAL_KV_CAPACITY;
        let kv_bytes = init_cap * kv_format.token_bytes();
        let mut attention_caches = Vec::with_capacity(6);
        for _ in 0..6 {
            attention_caches.push(FullAttentionCache {
                k_cache: MetalBuffer::empty(dev, kv_bytes)?,
                v_cache: MetalBuffer::empty(dev, kv_bytes)?,
            });
        }

        let m = profile.prefill_rows;
        // 2048 partial rows (16.9 MiB, independent of context) let a 512-row prefill split 4 ways
        // and an 8-row verify 256 ways over a long prefix.
        let workspace = AttentionWorkspace::new(&ctx, m.max(2048) as u32)?;
        let rope_rows_buf = MetalBuffer::empty(dev, m * 16)?;
        let token_ids_buf = MetalBuffer::empty(dev, m * 4)?;
        let hidden = MetalBuffer::empty(dev, m * WIDTH * 4)?;
        let hidden_alt = MetalBuffer::empty(dev, m * WIDTH * 4)?;
        let normed = MetalBuffer::empty(dev, m * WIDTH * 4)?;
        let rotated_1024 = MetalBuffer::empty(dev, m * 1024 * 4)?;
        let branch = MetalBuffer::empty(dev, m * WIDTH * 4)?;
        let qkv = MetalBuffer::empty(dev, m * CONV_CHANNELS as usize * 4)?;
        let convolved = MetalBuffer::empty(dev, m * CONV_CHANNELS as usize * 4)?;
        let gdn_gate = MetalBuffer::empty(dev, m * GDN_OUT_WIDTH as usize * 4)?;
        let alpha = MetalBuffer::empty(dev, m * GDN_HEADS as usize * 4)?;
        let raw_beta = MetalBuffer::empty(dev, m * GDN_HEADS as usize * 4)?;
        let decay = MetalBuffer::empty(dev, m * GDN_HEADS as usize * 4)?;
        let beta = MetalBuffer::empty(dev, m * GDN_HEADS as usize * 4)?;
        let gdn_out = MetalBuffer::empty(dev, m * GDN_OUT_WIDTH as usize * 4)?;
        let attn_qg = MetalBuffer::empty(dev, m * 4096 * 4)?;
        let attn_k = MetalBuffer::empty(dev, m * 512 * 4)?;
        let attn_v = MetalBuffer::empty(dev, m * 512 * 4)?;
        let attn_q = MetalBuffer::empty(dev, m * 2048 * 4)?;
        let attn_gate = MetalBuffer::empty(dev, m * 2048 * 4)?;
        let attn_out = MetalBuffer::empty(dev, m * 2048 * 4)?;
        let ffn_gate = MetalBuffer::empty(dev, m * FFN * 4)?;
        let ffn_up = MetalBuffer::empty(dev, m * FFN * 4)?;
        let ffn_prod = MetalBuffer::empty(dev, m * FFN * 4)?;
        let logits = MetalBuffer::empty(dev, MAX_VERIFY_TOKENS * VOCAB * 4)?;
        let taps_buf = MetalBuffer::empty(dev, m * 5 * WIDTH * 4)?;

        let disk_cache = parent
            .zip(prompt_cache_key(&artifact.header.payload_sha256))
            .and_then(|(dir, key)| {
                DiskCache::open(&dir.join(".prompt-cache"), key, kv_format, profile.prompt_cache_bytes)
            });
        let ane = AneState::start();
        Ok(Self {
            shaders,
            proj,
            ops,
            sampling,
            vision: None,
            vision_path,
            ane,
            rope_rows_buf,
            artifact,
            layers,
            embed,
            lm_head,
            final_norm,
            tokenizer,
            drafter,
            rot512_1024,
            kv_capacity: init_cap,
            profile,
            recurrent,
            attention_caches,
            workspace,
            kv_format,
            position: 0,
            rope_delta: 0,
            token_ids_buf,
            hidden,
            hidden_alt,
            normed,
            rotated_1024,
            branch,
            qkv,
            convolved,
            gdn_gate,
            alpha,
            raw_beta,
            decay,
            beta,
            gdn_out,
            attn_qg,
            attn_k,
            attn_v,
            attn_q,
            attn_gate,
            attn_out,
            ffn_gate,
            ffn_up,
            ffn_prod,
            logits,
            score_logits: None,
            taps_buf,
            ctx,
            rep_penalty: 1.0,
            recent_token_counts: Vec::new(),
            dry_scratch: Vec::new(),
            active_tokens: Vec::new(),
            prefix_cache: Vec::with_capacity(profile.prefix_checkpoints),
            cache_tick: 0,
            prefix_cache_enabled: true,
            disk_cache,
        })
    }


    /// Attach the `SigLIP2` + Window-Attention Merger vision tower (`minicpm-v-4.7-1b-vision.bin`).
    pub fn load_vision(&mut self, vision_path: &Path) -> crate::Result<()> {
        self.vision = Some(VisionEncoder::open(&self.ctx, &self.shaders, vision_path)?);
        Ok(())
    }

    /// Switch the full-attention KV cache format (`KvFormat::Q8` or `KvFormat::Q4`) and clear prefix checkpoints.
    ///
    /// The KV buffers are sized for the current format's bytes per token, so a wider format needs
    /// new buffers: keeping the old ones would write past their end.
    pub fn set_kv_format(&mut self, format: KvFormat) {
        if format != self.kv_format
            && let Some(disk) = self.disk_cache.take()
        {
            self.disk_cache = disk.reopen(format);
        }
        self.kv_format = format;
        self.reallocate_kv_buffers();
        self.prefix_cache.clear();
        self.reset();
    }

    /// Reallocate the KV caches for the current format and capacity, dropping their contents.
    ///
    /// Each layer gets its own allocation, as it does at construction: `MetalBuffer` is
    /// handle-based, so handing the same buffer to two layers would make them overwrite each
    /// other's cache.
    ///
    /// If the allocation fails the old buffers stay in place, so the cache keeps its old size;
    /// the next `ensure_kv_capacity` then reallocates at the new format's size.
    fn reallocate_kv_buffers(&mut self) {
        let bytes = self.kv_capacity * self.kv_format.token_bytes();
        let dev = self.ctx.device();
        for cache in &mut self.attention_caches {
            let Ok(new_k) = MetalBuffer::empty(dev, bytes) else {
                return;
            };
            let Ok(new_v) = MetalBuffer::empty(dev, bytes) else {
                return;
            };
            cache.k_cache = new_k;
            cache.v_cache = new_v;
        }
    }

    /// The KV cache buffers and the capacity they are sized for, for tests that assert the
    /// buffers match the current KV format.
    #[cfg(test)]
    pub(crate) fn kv_cache_geometry(&self) -> (Vec<(usize, usize)>, usize) {
        (
            self.attention_caches
                .iter()
                .map(|c| (c.k_cache.length(), c.v_cache.length()))
                .collect(),
            self.kv_capacity,
        )
    }

    pub fn reset(&mut self) {
        self.position = 0;
        self.rope_delta = 0;
        self.active_tokens.clear();
        self.recent_token_counts.clear();
        self.dry_scratch.clear();
        for r in &self.recurrent {
            r.state.clear();
            r.history.clear();
        }
        if let Some(d) = &mut self.drafter {
            d.reset();
        }
    }

    pub const fn set_prefix_cache_enabled(&mut self, enabled: bool) {
        self.prefix_cache_enabled = enabled;
    }

    #[must_use]
    pub const fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache_enabled
    }

    pub fn clear_prefix_cache(&mut self) {
        self.prefix_cache.clear();
        self.active_tokens.clear();
    }

    #[must_use]
    pub fn cached_prefix_tokens(&self, prompt_ids: &[u32]) -> usize {
        if !self.prefix_cache_enabled || prompt_ids.len() <= 1 {
            return 0;
        }
        let max_reusable = prompt_ids.len() - 1;
        let resident_lcp = if self.position > 0
            && self.rope_delta == 0
            && self.active_tokens.len() == self.position
            && prompt_ids.starts_with(&self.active_tokens)
        {
            self.position.min(max_reusable)
        } else {
            0
        };
        let ckpt_lcp = self
            .prefix_cache
            .iter()
            .filter(|ckpt| ckpt.tokens.len() <= max_reusable && prompt_ids.starts_with(&ckpt.tokens))
            .map(|ckpt| ckpt.tokens.len())
            .max()
            .unwrap_or(0);
        let disk_lcp = self
            .disk_cache
            .as_ref()
            .map_or(0, |disk| disk.longest(prompt_ids, max_reusable));
        resident_lcp.max(ckpt_lcp).max(disk_lcp)
    }

    fn save_prefix_checkpoint(
        &mut self,
        prefix_tokens: &[u32],
        pinned_system: bool,
    ) -> crate::Result<()> {
        let prefix_len = prefix_tokens.len();
        if !self.prefix_cache_enabled
            || prefix_len < 8
            || prefix_len != self.position
            || self.rope_delta != 0
        {
            return Ok(());
        }
        self.cache_tick += 1;
        let tick = self.cache_tick;
        if let Some(idx) = self.prefix_cache.iter().position(|ckpt| ckpt.tokens == prefix_tokens) {
            let existing = &mut self.prefix_cache[idx];
            existing.last_used_tick = tick;
            existing.pinned_system |= pinned_system;
            self.persist_checkpoint(idx);
            return Ok(());
        }
        if !self.make_room_for_checkpoint(self.checkpoint_bytes(prefix_len)) {
            return Ok(());
        }
        let dev = self.ctx.device();
        let state_bytes = (GDN_HEADS * GDN_DIM * GDN_DIM) as usize * 2;
        let hist_bytes = (CONV_CHANNELS as usize) * 3 * 4;
        let attn_bytes = prefix_len * self.kv_format.token_bytes();
        let dspark_bytes = DSPARK_KV_RING * DSPARK_KV_TOKEN_BYTES;

        let mut rec_bufs = Vec::with_capacity(self.recurrent.len());
        for _ in &self.recurrent {
            rec_bufs.push((
                MetalBuffer::empty(dev, state_bytes)?,
                MetalBuffer::empty(dev, hist_bytes)?,
            ));
        }
        let mut attn_bufs = Vec::with_capacity(self.attention_caches.len());
        for _ in &self.attention_caches {
            attn_bufs.push((
                MetalBuffer::empty(dev, attn_bytes)?,
                MetalBuffer::empty(dev, attn_bytes)?,
            ));
        }
        let mut dspark_bufs = Vec::new();
        let mut dspark_scored_anchors = 0;
        if let Some(drafter) = &self.drafter {
            dspark_scored_anchors = drafter.scored_anchors();
            let (dk, _) = drafter.kv_caches();
            dspark_bufs.reserve(dk.len());
            for _ in dk {
                dspark_bufs.push((
                    MetalBuffer::empty(dev, dspark_bytes)?,
                    MetalBuffer::empty(dev, dspark_bytes)?,
                ));
            }
        }

        let mut batch = CommandBatch::new(&self.ctx)?;
        let mut copies = Vec::with_capacity(60);
        for (rstate, (dst_s, dst_h)) in self.recurrent.iter().zip(&rec_bufs) {
            copies.push(BufferCopyRequest {
                source: &rstate.state,
                source_offset: 0,
                destination: dst_s,
                destination_offset: 0,
                size: state_bytes,
            });
            copies.push(BufferCopyRequest {
                source: &rstate.history,
                source_offset: 0,
                destination: dst_h,
                destination_offset: 0,
                size: hist_bytes,
            });
        }
        for (acache, (dst_k, dst_v)) in self.attention_caches.iter().zip(&attn_bufs) {
            copies.push(BufferCopyRequest {
                source: &acache.k_cache,
                source_offset: 0,
                destination: dst_k,
                destination_offset: 0,
                size: attn_bytes,
            });
            copies.push(BufferCopyRequest {
                source: &acache.v_cache,
                source_offset: 0,
                destination: dst_v,
                destination_offset: 0,
                size: attn_bytes,
            });
        }
        if let Some(drafter) = &self.drafter {
            let (dk, dv) = drafter.kv_caches();
            for ((src_k, src_v), (dst_k, dst_v)) in dk.iter().zip(dv.iter()).zip(&dspark_bufs) {
                copies.push(BufferCopyRequest {
                    source: src_k,
                    source_offset: 0,
                    destination: dst_k,
                    destination_offset: 0,
                    size: dspark_bytes,
                });
                copies.push(BufferCopyRequest {
                    source: src_v,
                    source_offset: 0,
                    destination: dst_v,
                    destination_offset: 0,
                    size: dspark_bytes,
                });
            }
        }
        batch.blit_buffer_copies(copies)?;
        batch.commit_and_wait()?;
        self.prefix_cache.push(PrefixCheckpoint {
            tokens: prefix_tokens.to_vec(),
            rope_delta: self.rope_delta,
            dspark_scored_anchors,
            recurrent: rec_bufs,
            attn_kv: attn_bufs,
            dspark_kv: dspark_bufs,
            last_used_tick: tick,
            pinned_system,
        });
        self.persist_checkpoint(self.prefix_cache.len() - 1);
        Ok(())
    }

    /// Evict the least valuable in-memory checkpoints until one more of `incoming` bytes fits;
    /// `false` (nothing evicted) when it never can.
    fn make_room_for_checkpoint(&mut self, incoming: u64) -> bool {
        // A quarter of the budget: the live KV cache may take half (its 1.5x doubling peak is what
        // picks `kv_bits`), and weights plus prefill scratch need most of the rest.
        let cap = self.profile.budget_bytes / 4;
        if incoming > cap {
            return false;
        }
        let mut total: u64 = self.prefix_cache.iter().map(|c| self.checkpoint_bytes(c.tokens.len())).sum();
        while (self.prefix_cache.len() >= self.profile.prefix_checkpoints || total + incoming > cap)
            && let Some((evict_idx, _)) = self
                .prefix_cache
                .iter()
                .enumerate()
                .min_by_key(|(_, ckpt)| (ckpt.pinned_system, ckpt.last_used_tick))
        {
            total -= self.checkpoint_bytes(self.prefix_cache[evict_idx].tokens.len());
            self.prefix_cache.swap_remove(evict_idx);
        }
        true
    }

    fn checkpoint_bytes(&self, prefix_len: usize) -> u64 {
        self.checkpoint_section_lens(prefix_len).iter().sum::<usize>() as u64
    }

    /// Bytes of each checkpoint section for a `prefix_len` prefix, in `PrefixCheckpoint` order:
    /// (state, history) per GDN layer, (K, V) per attention layer, then (K, V) per drafter layer.
    fn checkpoint_section_lens(&self, prefix_len: usize) -> Vec<usize> {
        let state_bytes = (GDN_HEADS * GDN_DIM * GDN_DIM) as usize * 2;
        let hist_bytes = (CONV_CHANNELS as usize) * 3 * 4;
        let attn_bytes = prefix_len * self.kv_format.token_bytes();
        let dspark_bytes = DSPARK_KV_RING * DSPARK_KV_TOKEN_BYTES;
        let mut lens = Vec::with_capacity(60);
        for _ in &self.recurrent {
            lens.extend([state_bytes, hist_bytes]);
        }
        for _ in &self.attention_caches {
            lens.extend([attn_bytes, attn_bytes]);
        }
        if let Some(drafter) = &self.drafter {
            for _ in drafter.kv_caches().0 {
                lens.extend([dspark_bytes, dspark_bytes]);
            }
        }
        lens
    }

    /// Queue checkpoint `idx` for the on-disk cache (or mark its file used).
    fn persist_checkpoint(&self, idx: usize) {
        let (Some(disk), Some(ckpt)) = (&self.disk_cache, self.prefix_cache.get(idx)) else {
            return;
        };
        let buffers = ckpt
            .recurrent
            .iter()
            .chain(&ckpt.attn_kv)
            .chain(&ckpt.dspark_kv)
            .flat_map(|(a, b)| [a, b])
            .zip(self.checkpoint_section_lens(ckpt.tokens.len()))
            .map(|(buffer, len)| (buffer.clone(), len))
            .collect();
        disk.persist(
            &ckpt.tokens,
            ckpt.rope_delta,
            ckpt.dspark_scored_anchors,
            ckpt.pinned_system,
            buffers,
        );
    }

    fn restore_or_advance_prefix(&mut self, prompt_ids: &[u32]) -> crate::Result<usize> {
        if !self.prefix_cache_enabled || prompt_ids.len() <= 1 {
            self.reset();
            return Ok(0);
        }
        let max_reusable = prompt_ids.len() - 1;
        let resident_lcp = self
            .active_tokens
            .iter()
            .zip(prompt_ids)
            .take_while(|(a, b)| a == b)
            .count()
            .min(max_reusable);
        let mut best_ckpt_idx = self
            .prefix_cache
            .iter()
            .enumerate()
            .filter(|(_, ckpt)| {
                ckpt.tokens.len() <= max_reusable && prompt_ids.starts_with(&ckpt.tokens)
            })
            .max_by_key(|(_, ckpt)| ckpt.tokens.len())
            .map(|(idx, _)| idx);
        let mut best_ckpt_len = best_ckpt_idx.map_or(0, |idx| self.prefix_cache[idx].tokens.len());
        let resident_usable = resident_lcp == self.position && self.position > 0 && self.rope_delta == 0;

        // A longer prefix on disk is mapped zero-copy and joins the in-memory checkpoints.
        if let Some(disk) = &self.disk_cache
            && let Some(loaded) = disk.load(
                &self.ctx,
                prompt_ids,
                max_reusable,
                best_ckpt_len.max(if resident_usable { resident_lcp } else { 0 }),
                |len| self.checkpoint_section_lens(len),
            )
            && self.make_room_for_checkpoint(self.checkpoint_bytes(loaded.header.tokens.len()))
        {
            let mut buffers = loaded.buffers.into_iter();
            let mut pairs = std::iter::from_fn(|| Some((buffers.next()?, buffers.next()?)));
            let recurrent = pairs.by_ref().take(self.recurrent.len()).collect();
            let attn_kv = pairs.by_ref().take(self.attention_caches.len()).collect();
            best_ckpt_len = loaded.header.tokens.len();
            self.prefix_cache.push(PrefixCheckpoint {
                tokens: loaded.header.tokens,
                rope_delta: loaded.header.rope_delta,
                dspark_scored_anchors: loaded.header.dspark_scored_anchors,
                recurrent,
                attn_kv,
                dspark_kv: pairs.collect(),
                last_used_tick: self.cache_tick,
                pinned_system: loaded.header.pinned,
            });
            best_ckpt_idx = Some(self.prefix_cache.len() - 1);
        }

        if resident_usable && best_ckpt_len <= resident_lcp
        {
            self.active_tokens.truncate(resident_lcp);
            if let Some(drafter) = &mut self.drafter {
                drafter.truncate_context(resident_lcp);
            }
            return Ok(resident_lcp);
        }

        if let Some(ckpt_idx) = best_ckpt_idx {
            let matched = self.prefix_cache[ckpt_idx].tokens.len();
            self.ensure_kv_capacity(matched)?;
            self.cache_tick += 1;
            self.prefix_cache[ckpt_idx].last_used_tick = self.cache_tick;
            if let Some(disk) = &self.disk_cache {
                disk.touch(&self.prefix_cache[ckpt_idx].tokens);
            }
            let ckpt = &self.prefix_cache[ckpt_idx];
            let state_bytes = (GDN_HEADS * GDN_DIM * GDN_DIM) as usize * 2;
            let hist_bytes = (CONV_CHANNELS as usize) * 3 * 4;
            let attn_bytes = matched * self.kv_format.token_bytes();
            let dspark_bytes = DSPARK_KV_RING * DSPARK_KV_TOKEN_BYTES;

            let mut batch = CommandBatch::new(&self.ctx)?;
            let mut copies = Vec::with_capacity(60);
            for (rstate, (src_s, src_h)) in self.recurrent.iter().zip(&ckpt.recurrent) {
                copies.push(BufferCopyRequest {
                    source: src_s,
                    source_offset: 0,
                    destination: &rstate.state,
                    destination_offset: 0,
                    size: state_bytes,
                });
                copies.push(BufferCopyRequest {
                    source: src_h,
                    source_offset: 0,
                    destination: &rstate.history,
                    destination_offset: 0,
                    size: hist_bytes,
                });
            }
            if resident_lcp < matched {
                for (acache, (src_k, src_v)) in self.attention_caches.iter().zip(&ckpt.attn_kv) {
                    copies.push(BufferCopyRequest {
                        source: src_k,
                        source_offset: 0,
                        destination: &acache.k_cache,
                        destination_offset: 0,
                        size: attn_bytes,
                    });
                    copies.push(BufferCopyRequest {
                        source: src_v,
                        source_offset: 0,
                        destination: &acache.v_cache,
                        destination_offset: 0,
                        size: attn_bytes,
                    });
                }
            }
            // The drafter ring is not append-only: positions decoded past `matched` overwrite it.
            if let Some(drafter) = &self.drafter {
                let (dk, dv) = drafter.kv_caches();
                for ((dst_k, dst_v), (src_k, src_v)) in dk.iter().zip(dv.iter()).zip(&ckpt.dspark_kv) {
                    copies.push(BufferCopyRequest {
                        source: src_k,
                        source_offset: 0,
                        destination: dst_k,
                        destination_offset: 0,
                        size: dspark_bytes,
                    });
                    copies.push(BufferCopyRequest {
                        source: src_v,
                        source_offset: 0,
                        destination: dst_v,
                        destination_offset: 0,
                        size: dspark_bytes,
                    });
                }
            }
            if let Err(e) = batch.blit_buffer_copies(copies).and_then(|()| batch.commit_and_wait()) {
                // The blit may have partly run: neither the old sequence nor this checkpoint is trustworthy.
                self.reset();
                self.prefix_cache.swap_remove(ckpt_idx);
                return Err(e.into());
            }

            let rope_delta = ckpt.rope_delta;
            let scored_anchors = ckpt.dspark_scored_anchors;
            let restored_tokens = ckpt.tokens.clone();
            self.position = matched;
            self.rope_delta = rope_delta;
            self.active_tokens = restored_tokens;
            self.recent_token_counts.clear();
            self.dry_scratch.clear();
            if let Some(drafter) = &mut self.drafter {
                drafter.set_checkpoint_state(matched, scored_anchors);
            }
            return Ok(matched);
        }

        self.reset();
        Ok(0)
    }

    #[must_use]
    pub fn device_name(&self) -> String {
        self.ctx.device_name()
    }

    #[must_use]
    pub fn ane_status(&self) -> serde_json::Value {
        self.ane.status_json()
    }
    /// Sizing chosen for this device at open.
    #[must_use]
    pub const fn memory_profile(&self) -> MemoryProfile {
        self.profile
    }


    fn take_ane(&mut self) -> Option<Box<AneFfnOffload>> {
        match std::mem::replace(&mut self.ane, AneState::Busy) {
            AneState::Ready(offload) => Some(offload),
            AneState::Pending(rx) => {
                // Still compiling: this chunk runs on the GPU instead of waiting.
                let prog = match rx.try_recv() {
                    Ok(prog) => prog,
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        self.ane = AneState::Pending(rx);
                        return None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        Err(Error::InvalidFormat("ANE compile thread exited".into()))
                    }
                };
                let checked = prog.and_then(|prog| self.check_ane(&prog));
                match checked {
                    Ok(Ok(offload)) => Some(offload),
                    Ok(Err(reason)) => {
                        self.ane = AneState::Off(reason);
                        None
                    }
                    Err(e) => {
                        self.ane = AneState::Off(e.to_string());
                        None
                    }
                }
            }
            other => {
                self.ane = other;
                None
            }
        }
    }

    /// Run the first `CALIBRATION_LAYERS` FFNs on the ANE and on the GPU at the prefill batch
    /// size, each in one command batch as prefill encodes them. The ANE is kept (`Ok(Ok(_))`)
    /// only if it agrees with the GPU and is faster on this chip; otherwise `Ok(Err(reason))`.
    #[expect(unsafe_code)]
    fn check_ane(
        &mut self,
        prog: &nano_metal::ane::Program,
    ) -> crate::Result<Result<Box<AneFfnOffload>, String>> {
        let mut offload = AneFfnOffload::new(&self.ctx, &self.shaders, prog)?;
        let (_, tol) = check_rows_and_tolerance();
        let rows = self.profile.prefill_rows.min(ANE_MAX_ROWS);
        unsafe {
            let slice = &mut self.normed.as_mut_slice::<f32>()[..rows * WIDTH];
            for (i, v) in slice.iter_mut().enumerate() {
                *v = ((i as f32) * 0.017).sin() * 0.5;
            }
        }
        let (mut ane_s, mut gpu_s) = (f64::INFINITY, f64::INFINITY);
        // The first pass warms pipelines and the ANE model; keep each path's best time.
        for _ in 0..2 {
            offload.reset_status();
            let mut stage = CommandBatch::new(&self.ctx)?;
            if !offload.stage_layer(&mut stage, &self.artifact, 0, 0)? {
                return Ok(Err("layer 0 is not INT4_G64".into()));
            }
            stage.commit_and_wait()?;
            let start = Instant::now();
            let mut batch = CommandBatch::new(&self.ctx)?;
            for l in 0..CALIBRATION_LAYERS {
                // The GPU stages layer l + 1 while the ANE evaluates layer l.
                let next = (l + 1 < CALIBRATION_LAYERS).then_some(l + 1);
                if !offload.encode_layer_ffn(&mut batch, &self.artifact, l, next, &self.normed, &self.branch, rows as u32)? {
                    return Ok(Err(format!("no ANE function for {rows} rows")));
                }
            }
            batch.commit_and_wait()?;
            if !offload.finish_and_check() {
                return Ok(Err("ANE evaluation failed".into()));
            }
            ane_s = ane_s.min(start.elapsed().as_secs_f64());

            let start = Instant::now();
            let mut batch = CommandBatch::new(&self.ctx)?;
            for l in 0..CALIBRATION_LAYERS {
                let lw = self.layers[l];
                self.project_layer_matrix(&mut batch, lw.gate, &self.normed, &self.ffn_gate, rows as u32)?;
                self.project_layer_matrix(&mut batch, lw.up, &self.normed, &self.ffn_up, rows as u32)?;
                self.ops
                    .swiglu(&mut batch, &self.ffn_gate, &self.ffn_up, &self.ffn_prod, (rows * FFN) as u32);
                self.project_layer_matrix(&mut batch, lw.down, &self.ffn_prod, &self.hidden_alt, rows as u32)?;
            }
            batch.commit_and_wait()?;
            gpu_s = gpu_s.min(start.elapsed().as_secs_f64());
        }
        // Both paths end with the last layer's FFN of the same input.
        let (gpu, ane) = unsafe {
            (
                &self.hidden_alt.as_slice::<f32>()[..rows * WIDTH],
                &self.branch.as_slice::<f32>()[..rows * WIDTH],
            )
        };
        let err = relative_difference(gpu, ane);
        if err.is_nan() || err > tol {
            return Ok(Err(format!("ANE check relative diff {err:.4} > {tol}")));
        }
        let per_layer_ms = |s: f64| s * 1e3 / CALIBRATION_LAYERS as f64;
        offload.layer_ms = (per_layer_ms(gpu_s), per_layer_ms(ane_s));
        if ane_s >= gpu_s {
            return Ok(Err(format!(
                "GPU is faster on this chip: an FFN layer at {rows} rows takes {:.2} ms on the GPU, {:.2} ms with the ANE",
                offload.layer_ms.0, offload.layer_ms.1
            )));
        }
        Ok(Ok(Box::new(offload)))
    }

    /// Format a single-turn user prompt using MiniCPM-V 4.7's chat template (thinking always enabled).
    #[must_use]
    pub fn format_chat_prompt(prompt: &str) -> String {
        Self::format_chat_prompt_with_system(Self::DEFAULT_SYSTEM_PROMPT, prompt)
    }

    /// Format a single-turn user prompt with a custom `system_prompt`.
    #[must_use]
    pub fn format_chat_prompt_with_system(system_prompt: &str, prompt: &str) -> String {
        format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
    }

    /// Pre-prefill and pin a shared `prefix_text` (e.g. a system prompt + shared context) in `PrefixCache`
    /// so all subsequent agent prompts starting with `prefix_text` skip prefilling it!
    pub fn warm_prefix_cache(&mut self, prefix_text: &str) -> crate::Result<usize> {
        let ids = self.encode_text(prefix_text)?;
        if ids.len() < 8 {
            return Ok(0);
        }
        self.prefill_prefix(&ids)?;
        // Best-effort: the prefix stays resident even when no checkpoint fits.
        let _ = self.save_prefix_checkpoint(&ids, true);
        Ok(ids.len())
    }

    /// Prefill `ids` into the resident state without saving a checkpoint, reusing the resident
    /// tokens or a checkpoint for any shared prefix, so a prompt that extends `ids` (the user's
    /// text so far, while they type) only prefills its remainder.
    pub fn prefill_prefix(&mut self, ids: &[u32]) -> crate::Result<()> {
        if self.position == ids.len() && self.active_tokens == ids {
            return Ok(());
        }
        let mut start = self.restore_or_advance_prefix(ids)?;
        while start < ids.len() {
            let end = (start + self.profile.prefill_rows).min(ids.len());
            let chunk = &ids[start..end];
            self.forward_step(Some(chunk), chunk.len(), Rope::Text(self.rope_delta), false, 0, 0)?;
            self.position += chunk.len();
            self.active_tokens.extend_from_slice(chunk);
            start = end;
        }
        Ok(())
    }

    /// Teacher-forced scoring of `ids` from an empty context, one entry per position. Backs the
    /// `OpenAI` `/v1/completions` `echo` + `logprobs` mode that evaluation harnesses use.
    pub fn score_tokens(&mut self, ids: &[u32]) -> crate::Result<Vec<TokenScore>> {
        let rows = self.profile.prefill_rows.min(SCORE_ROWS);
        let mut logits = match self.score_logits.take() {
            Some(buf) => buf,
            None => MetalBuffer::empty(self.ctx.device(), rows * VOCAB * 4)?,
        };
        // Generation keeps its small buffer; scoring borrows the wide one and always swaps back.
        std::mem::swap(&mut self.logits, &mut logits);
        self.reset();
        let scores = self.score_blocks(ids, rows);
        std::mem::swap(&mut self.logits, &mut logits);
        self.score_logits = Some(logits);
        self.reset();
        scores
    }

    #[expect(unsafe_code)]
    fn score_blocks(&mut self, ids: &[u32], rows: usize) -> crate::Result<Vec<TokenScore>> {
        let threads = thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let empty = TokenScore {
            top: 0,
            top_logprob: 0.0,
            next_logprob: None,
        };
        let mut scores = vec![empty; ids.len()];
        for (b, block) in ids.chunks(rows).enumerate() {
            self.forward_step(Some(block), block.len(), Rope::Text(0), false, block.len(), 0)?;
            self.position += block.len();
            let logits = unsafe { &self.logits.as_slice::<f32>()[..block.len() * VOCAB] };
            let next = &ids[b * rows + 1..];
            let out = &mut scores[b * rows..b * rows + block.len()];
            // Single-threaded, these row statistics cost about as much as the GPU pass itself.
            let per = block.len().div_ceil(threads);
            thread::scope(|s| {
                for (c, (out, chunk)) in out.chunks_mut(per).zip(logits.chunks(per * VOCAB)).enumerate() {
                    s.spawn(move || {
                        for (r, (score, row)) in out.iter_mut().zip(chunk.as_chunks::<VOCAB>().0).enumerate() {
                            *score = row_score(row, next.get(c * per + r).copied());
                        }
                    });
                }
            });
        }
        Ok(scores)
    }

    /// Tokenize (if needed) and teacher-force `prompt`, returning each token's text with its score.
    pub fn score_prompt(&mut self, prompt: ScorePrompt) -> crate::Result<ScoredPrompt> {
        let ids = match prompt {
            ScorePrompt::Text(text) => self.encode_text(&text)?,
            ScorePrompt::Ids(ids) => ids,
        };
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= VOCAB) {
            return Err(Error::InvalidArgument("prompt must be 1+ valid token ids".into()));
        }
        let scores = self.score_tokens(&ids)?;
        let piece = |id: u32| {
            self.tokenizer.as_ref().map_or_else(
                || id.to_string(),
                |tok| String::from_utf8_lossy(tok.token_bytes(id)).into_owned(),
            )
        };
        let pieces = ids.iter().map(|&id| piece(id)).collect();
        let top_pieces = scores.iter().map(|s| piece(s.top)).collect();
        Ok(ScoredPrompt {
            ids,
            pieces,
            scores,
            top_pieces,
        })
    }

    /// Execute `agent_prompts` sharing a common `system_prompt` with automatic GPU `PrefixCache` reuse.
    #[expect(clippy::needless_pass_by_value)]
    pub fn run_agents(
        &mut self,
        system_prompt: &str,
        agent_prompts: &[String],
        max_new_tokens: usize,
        params: SamplingParams,
    ) -> crate::Result<Vec<GenerationReport>> {
        if agent_prompts.len() > 1 {
            let shared_prefix = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n");
            let _ = self.warm_prefix_cache(&shared_prefix);
        }
        let mut reports = Vec::with_capacity(agent_prompts.len());
        for agent_prompt in agent_prompts {
            let formatted = Self::format_chat_prompt_with_system(system_prompt, agent_prompt);
            let ids = self.encode_text(&formatted)?;
            reports.push(self.generate_ids(&ids, max_new_tokens, params.clone())?);
        }
        Ok(reports)
    }

    pub fn encode_text(&self, text: &str) -> crate::Result<Vec<u32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Tokenizer("no tokenizer loaded".into()))?;
        Ok(tok.encode(text))
    }

    /// Whether `id` decodes to visible text: ordinary tokens plus the non-special added tokens
    /// (`<tool_call>`, `</tool_call>`, fim, `<tool_response>`, `<think>`, `</think>`).
    const fn is_text_token(id: u32) -> bool {
        id < 248_044 || matches!(id, 248_058..=248_069)
    }

    /// Decode token IDs to UTF-8 text, dropping special tokens.
    pub fn decode_tokens(&self, ids: &[u32]) -> crate::Result<String> {
        let Some(tok) = &self.tokenizer else {
            return Ok(format!("{ids:?}"));
        };
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| Self::is_text_token(id))
            .collect();
        Ok(tok.decode(&filtered))
    }

    /// Split `generated` tokens at `</think>` (`248_069`) when `thinking` was requested or `<think>` is present.
    pub fn split_reasoning_and_text(&self, ids: &[u32]) -> crate::Result<(Option<String>, String)> {
        if let Some(end_pos) = ids.iter().rposition(|&id| id == THINK_END_ID) {
            let think_slice = ids[..end_pos]
                .iter()
                .copied()
                .filter(|&id| id != THINK_START_ID && id != THINK_END_ID)
                .collect::<Vec<_>>();
            let answer_slice = ids[end_pos + 1..]
                .iter()
                .copied()
                .filter(|&id| id != THINK_START_ID && id != THINK_END_ID)
                .collect::<Vec<_>>();
            let reasoning = self.decode_tokens(&think_slice)?;
            let answer = self.decode_tokens(&answer_slice)?;
            let r_trimmed = reasoning.trim().to_owned();
            Ok((
                (!r_trimmed.is_empty()).then_some(r_trimmed),
                answer.trim_start().to_owned(),
            ))
        } else if let Some(split_pos) = ids
            .iter()
            .position(|&id| matches!(id, TOOL_CALL_START_ID | FENCE_NEWLINE_ID | FENCE_BARE_ID))
        {
            let think_slice = ids[..split_pos]
                .iter()
                .copied()
                .filter(|&id| id != THINK_START_ID && id != THINK_END_ID)
                .collect::<Vec<_>>();
            let answer_slice = ids[split_pos..]
                .iter()
                .copied()
                .filter(|&id| id != THINK_START_ID && id != THINK_END_ID)
                .collect::<Vec<_>>();
            let reasoning = self.decode_tokens(&think_slice)?;
            let answer = self.decode_tokens(&answer_slice)?;
            let r_trimmed = reasoning.trim().to_owned();
            Ok((
                (!r_trimmed.is_empty()).then_some(r_trimmed),
                answer.trim_start().to_owned(),
            ))
        } else {
            Ok((None, self.decode_tokens(ids)?))
        }
    }

    /// Incrementally decode token `id` (whole UTF-8 characters only), keeping `<think>` and `</think>` visible.
    fn stream_token_piece(&self, pending: &mut Vec<u8>, id: u32) -> Option<String> {
        let tok = self.tokenizer.as_ref()?;
        if !Self::is_text_token(id) {
            return None;
        }
        tok.stream_piece(pending, id)
    }

    fn project_layer_matrix(
        &self,
        batch: &mut CommandBatch,
        handle: MatrixHandle,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        let (mat, rotated) = self.artifact.matrix_at(handle);
        if rotated {
            if mat.columns() != 1024 {
                return Err(Error::InvalidFormat(format!(
                    "unsupported rotated matrix width {}",
                    mat.columns()
                )));
            }
            self.proj
                .rotate512(batch, &self.rot512_1024, input, &self.rotated_1024, tokens, false)?;
            Ok(self.proj.matmul(batch, mat, &self.rotated_1024, output, tokens)?)
        } else {
            Ok(self.proj.matmul(batch, mat, input, output, tokens)?)
        }
    }

    /// Forward `tokens` rows (`hidden` already populated with input embeddings) through all 24 layers.
    fn ensure_kv_capacity(&mut self, needed: usize) -> crate::Result<()> {
        if needed <= self.kv_capacity {
            return Ok(());
        }
        if needed > MAX_CONTEXT_TOKENS {
            return Err(Error::InvalidArgument(format!(
                "sequence length {needed} exceeds the {MAX_CONTEXT_TOKENS}-token context window"
            )));
        }
        let new_cap = needed.next_power_of_two().min(MAX_CONTEXT_TOKENS);
        let new_bytes = new_cap * self.kv_format.token_bytes();
        let used_bytes = self.position * self.kv_format.token_bytes();
        let dev = self.ctx.device();
        let mut batch = CommandBatch::new(&self.ctx)?;
        let mut new_caches = Vec::with_capacity(6);
        for old in &self.attention_caches {
            let nk = MetalBuffer::empty(dev, new_bytes)?;
            let nv = MetalBuffer::empty(dev, new_bytes)?;
            if used_bytes > 0 {
                batch.blit_buffer_copies([
                    BufferCopyRequest {
                        source: &old.k_cache,
                        source_offset: 0,
                        destination: &nk,
                        destination_offset: 0,
                        size: used_bytes,
                    },
                    BufferCopyRequest {
                        source: &old.v_cache,
                        source_offset: 0,
                        destination: &nv,
                        destination_offset: 0,
                        size: used_bytes,
                    },
                ])?;
            }
            new_caches.push(FullAttentionCache {
                k_cache: nk,
                v_cache: nv,
            });
        }
        if used_bytes > 0 {
            batch.commit_and_wait()?;
        }
        self.attention_caches = new_caches;
        self.kv_capacity = new_cap;
        Ok(())
    }

    /// When `speculative_verify` is true, recurrent layers write to `rollback_state`/`rollback_history`
    /// leaving `state`/`history` at the pre-verify boundary for zero-copy rollback on rejection.
    /// `compute_logits_rows`: how many trailing rows of the block to project through `lm_head` (1 for prefill/decode, `tokens` for verify).
    pub fn forward_embeddings(
        &mut self,
        tokens: usize,
        rope: Rope<'_>,
        speculative_verify: bool,
        compute_logits_rows: usize,
    ) -> crate::Result<()> {
        self.forward_step(None, tokens, rope, speculative_verify, compute_logits_rows, 0)
    }

    fn forward_step(
        &mut self,
        token_ids: Option<&[u32]>,
        tokens: usize,
        rope: Rope<'_>,
        speculative_verify: bool,
        compute_logits_rows: usize,
        sample_top_k: usize,
    ) -> crate::Result<()> {
        let (batch, ane_offload) = self.encode_forward_step(
            None,
            token_ids,
            tokens,
            rope,
            speculative_verify,
            compute_logits_rows,
            sample_top_k,
            true,
        )?;
        batch.commit_and_wait()?;
        self.finish_ane(ane_offload)
    }

    /// Encode a forward step and await GPU completion asynchronously without blocking the worker thread.
    async fn forward_step_async(
        &mut self,
        token_ids: Option<&[u32]>,
        tokens: usize,
        rope: Rope<'_>,
        speculative_verify: bool,
        compute_logits_rows: usize,
        sample_top_k: usize,
    ) -> crate::Result<()> {
        let (fut, ane_offload) = {
            let (batch, ane_offload) = self.encode_forward_step(
                None,
                token_ids,
                tokens,
                rope,
                speculative_verify,
                compute_logits_rows,
                sample_top_k,
                !speculative_verify,
            )?;
            (batch.commit_and_await(), ane_offload)
        };
        fut.await?;
        self.finish_ane(ane_offload)
    }

    /// Hand a finished ANE offload back, or, when it failed, switch the ANE off and reset: the
    /// step's KV rows and in-place GDN state are garbage and must never be used or checkpointed.
    fn finish_ane(&mut self, ane_offload: Option<Box<AneFfnOffload>>) -> crate::Result<()> {
        if let Some(ane) = ane_offload {
            if !ane.finish_and_check() {
                self.ane = AneState::Off("ANE evaluation failed or produced non-finite values".into());
                self.reset();
                return Err(Error::AneFailed);
            }
            self.ane = AneState::Ready(ane);
        }
        Ok(())
    }

    /// Encode a forward step into `into` (or a fresh batch) and return the batch.
    fn encode_forward_step(
        &mut self,
        into: Option<CommandBatch>,
        token_ids: Option<&[u32]>,
        tokens: usize,
        rope: Rope<'_>,
        speculative_verify: bool,
        compute_logits_rows: usize,
        sample_top_k: usize,
        allow_ane: bool,
    ) -> crate::Result<(CommandBatch, Option<Box<AneFfnOffload>>)> {
        if tokens > self.profile.prefill_rows {
            return Err(Error::InvalidArgument(format!(
                "chunk tokens {tokens} exceeds prefill_rows {}",
                self.profile.prefill_rows
            )));
        }
        let tokens_u32 = tokens as u32;
        self.ensure_kv_capacity(self.position + tokens)?;
        if tokens > MAX_VERIFY_TOKENS {
            // Cheap when resident; re-reads weights the OS dropped while the app was idle.
            self.artifact.prefetch();
        }
        let mut ane_offload = if allow_ane
            && !speculative_verify
            && (ANE_MIN_ROWS..=ANE_MAX_ROWS).contains(&tokens)
        {
            self.take_ane()
        } else {
            None
        };
        // Decode and verify blocks are dependent chains that drain faster in a serial encoder;
        // prefill chunks gain from concurrent `independent` groups.
        let mut batch = match into {
            Some(batch) => batch,
            None if tokens > MAX_VERIFY_TOKENS && ane_offload.is_none() => {
                CommandBatch::new_concurrent(&self.ctx)?
            }
            None => CommandBatch::new(&self.ctx)?,
        };
        if let Some(ids) = token_ids {
            self.encode_embed_tokens(&mut batch, ids);
        }
        if let Some(ane) = &ane_offload {
            ane.reset_status();
            if !ane.stage_layer(&mut batch, &self.artifact, 0, 0).unwrap_or(false)
                && let Some(a) = ane_offload.take()
            {
                self.ane = AneState::Ready(a);
            }
        }
        let mut rec_idx = 0_usize;
        let mut attn_idx = 0_usize;
        let payload = &self.artifact.payload_buffer;
        for layer in 0..LAYERS {
            let lw = self.layers[layer];
            if layer == 0 {
                let (in_norm, in_off) = (payload, lw.input_norm);
                self.ops.rms_norm(
                    &mut batch,
                    &self.hidden,
                    in_norm,
                    in_off,
                    &self.normed,
                    WIDTH as u32,
                    tokens_u32,
                    EPSILON,
                );
            }

            if let Mixer::Attention { q, k, v, o, q_norm_at, k_norm_at } = lw.mixer {
                let (q_mat, q_rot) = self.artifact.matrix_at(q);
                let (k_mat, k_rot) = self.artifact.matrix_at(k);
                let (v_mat, v_rot) = self.artifact.matrix_at(v);
                if !q_rot
                    && !k_rot
                    && !v_rot
                    && let (AnyMatrix::Int4(qm), AnyMatrix::Int4(km), AnyMatrix::Int4(vm)) =
                        (q_mat, k_mat, v_mat)
                {
                    self.proj.int4_matmul_concat3(
                        &mut batch,
                        qm,
                        km,
                        vm,
                        &self.normed,
                        &self.attn_qg,
                        &self.attn_k,
                        &self.attn_v,
                        tokens_u32,
                    );
                } else if !q_rot && !k_rot && !v_rot {
                    batch.independent(|batch| {
                        self.proj.matmul(batch, q_mat, &self.normed, &self.attn_qg, tokens_u32)?;
                        self.proj.matmul(batch, k_mat, &self.normed, &self.attn_k, tokens_u32)?;
                        self.proj.matmul(batch, v_mat, &self.normed, &self.attn_v, tokens_u32)
                    })?;
                } else {
                    self.project_layer_matrix(
                        &mut batch,
                        q,
                        &self.normed,
                        &self.attn_qg,
                        tokens_u32,
                    )?;
                    self.project_layer_matrix(
                        &mut batch,
                        k,
                        &self.normed,
                        &self.attn_k,
                        tokens_u32,
                    )?;
                    self.project_layer_matrix(
                        &mut batch,
                        v,
                        &self.normed,
                        &self.attn_v,
                        tokens_u32,
                    )?;
                }
                let (q_norm, q_off) = (payload, q_norm_at);
                let (k_norm, k_off) = (payload, k_norm_at);
                let cache = &self.attention_caches[attn_idx];
                attn_idx += 1;

                self.ops.prepare_attention(
                    &mut batch,
                    &self.attn_qg,
                    &self.attn_k,
                    &self.attn_v,
                    (q_norm, q_off),
                    (k_norm, k_off),
                    &self.attn_q,
                    &self.attn_gate,
                    &cache.k_cache,
                    &cache.v_cache,
                    self.position as u32,
                    self.kv_format,
                    EPSILON,
                    ROPE_THETA,
                    tokens_u32,
                    rope,
                );

                self.ops.attention_causal_block(
                    &mut batch,
                    &self.attn_q,
                    &cache.k_cache,
                    &cache.v_cache,
                    &self.attn_gate,
                    &self.attn_out,
                    self.position as u32,
                    tokens_u32,
                    self.kv_format,
                    &self.workspace,
                );

                self.project_layer_matrix(
                    &mut batch,
                    o,
                    &self.attn_out,
                    &self.branch,
                    tokens_u32,
                )?;
            } else if let Mixer::Recurrent { qkv, z, a, b, out, conv_at, a_log_at, dt_bias_at, norm_at } =
                lw.mixer
            {
                // Gated DeltaNet recurrent layer
                let (qkv_mat, qkv_rot) = self.artifact.matrix_at(qkv);
                let (z_mat, z_rot) = self.artifact.matrix_at(z);
                let (a_mat, _) = self.artifact.matrix_at(a);
                let (b_mat, _) = self.artifact.matrix_at(b);
                let qkv_dst = if speculative_verify {
                    &self.recurrent[rec_idx].tape_qkv
                } else {
                    &self.qkv
                };
                if !qkv_rot
                    && !z_rot
                    && let (
                        AnyMatrix::Int4(qkvm),
                        AnyMatrix::Int4(zm),
                        AnyMatrix::Bf16(am),
                        AnyMatrix::Bf16(bm),
                    ) = (qkv_mat, z_mat, a_mat, b_mat)
                {
                    self.proj.int4_matmul_gdn(
                        &mut batch,
                        qkvm,
                        zm,
                        am,
                        bm,
                        &self.normed,
                        qkv_dst,
                        &self.gdn_gate,
                        &self.alpha,
                        &self.raw_beta,
                        tokens_u32,
                    )?;
                } else if !qkv_rot
                    && !z_rot
                    && let (AnyMatrix::Bf16(am), AnyMatrix::Bf16(bm)) = (a_mat, b_mat)
                {
                    batch.independent(|batch| {
                        self.proj.matmul(batch, qkv_mat, &self.normed, qkv_dst, tokens_u32)?;
                        self.proj.matmul(batch, z_mat, &self.normed, &self.gdn_gate, tokens_u32)?;
                        self.proj.bf16_matmul_pair(
                            batch,
                            am,
                            bm,
                            &self.normed,
                            &self.alpha,
                            &self.raw_beta,
                            tokens_u32,
                        );
                        Ok::<(), nano_metal::Error>(())
                    })?;
                } else {
                    self.project_layer_matrix(
                        &mut batch,
                        qkv,
                        &self.normed,
                        qkv_dst,
                        tokens_u32,
                    )?;
                    self.project_layer_matrix(
                        &mut batch,
                        z,
                        &self.normed,
                        &self.gdn_gate,
                        tokens_u32,
                    )?;
                    self.project_layer_matrix(
                        &mut batch,
                        a,
                        &self.normed,
                        &self.alpha,
                        tokens_u32,
                    )?;
                    self.project_layer_matrix(
                        &mut batch,
                        b,
                        &self.normed,
                        &self.raw_beta,
                        tokens_u32,
                    )?;
                }

                let (conv_w, conv_off) = (payload, conv_at);
                let (a_w, a_off) = (payload, a_log_at);
                let (dt_w, dt_off) = (payload, dt_bias_at);
                let (gdn_norm, gdn_norm_off) = (payload, norm_at);

                let rstate = &self.recurrent[rec_idx];
                rec_idx += 1;
                let (final_hist, final_st, conv_buf, dec_buf, beta_buf) = if speculative_verify {
                    (
                        &rstate.rollback_history,
                        &rstate.rollback_state,
                        &rstate.tape_convolved,
                        &rstate.tape_decay,
                        &rstate.tape_beta,
                    )
                } else {
                    (
                        &rstate.history,
                        &rstate.state,
                        &self.convolved,
                        &self.decay,
                        &self.beta,
                    )
                };

                self.ops.conv_l2_decay(
                    &mut batch,
                    qkv_dst,
                    (conv_w, conv_off),
                    &rstate.history,
                    final_hist,
                    conv_buf,
                    (a_w, a_off),
                    &self.alpha,
                    (dt_w, dt_off),
                    &self.raw_beta,
                    dec_buf,
                    beta_buf,
                    tokens_u32,
                    tokens_u32,
                    EPSILON,
                );
                self.ops.gdn_recurrence_f16(
                    &mut batch,
                    conv_buf,
                    dec_buf,
                    beta_buf,
                    &rstate.state,
                    final_st,
                    &self.gdn_out,
                    tokens_u32,
                    tokens_u32,
                );
                self.ops.gdn_postprocess(
                    &mut batch,
                    &self.gdn_out,
                    &self.gdn_gate,
                    (gdn_norm, gdn_norm_off),
                    &self.attn_out,
                    tokens_u32,
                    EPSILON,
                );
                self.project_layer_matrix(
                    &mut batch,
                    out,
                    &self.attn_out,
                    &self.branch,
                    tokens_u32,
                )?;
            }

            // Fused residual add + post-attention RMSNorm
            let (post_norm, post_off) = (payload, lw.post_norm);
            self.ops.residual_add_rms(
                &mut batch,
                &self.hidden,
                &self.branch,
                post_norm,
                post_off,
                &self.hidden_alt,
                &self.normed,
                WIDTH as u32,
                tokens_u32,
                EPSILON,
            );

            // Feed-forward SwiGLU (ANE offload for prefill chunks, or Metal GPU)
            let ran_on_ane = if let Some(ane) = &ane_offload {
                let next_l = (layer + 1 < LAYERS).then_some(layer + 1);
                ane.encode_layer_ffn(
                    &mut batch,
                    &self.artifact,
                    layer,
                    next_l,
                    &self.normed,
                    &self.branch,
                    tokens_u32,
                )?
            } else {
                false
            };
            if !ran_on_ane {
                let (gate_mat, gate_rot) = self.artifact.matrix_at(lw.gate);
                let (up_mat, up_rot) = self.artifact.matrix_at(lw.up);
                if !gate_rot && !up_rot {
                    self.proj.matmul_swiglu(
                        &mut batch,
                        gate_mat,
                        up_mat,
                        &self.normed,
                        &self.ffn_prod,
                        tokens_u32,
                    )?;
                } else {
                    self.project_layer_matrix(
                        &mut batch,
                        lw.gate,
                        &self.normed,
                        &self.ffn_gate,
                        tokens_u32,
                    )?;
                    self.project_layer_matrix(
                        &mut batch,
                        lw.up,
                        &self.normed,
                        &self.ffn_up,
                        tokens_u32,
                    )?;
                    self.ops.swiglu(
                        &mut batch,
                        &self.ffn_gate,
                        &self.ffn_up,
                        &self.ffn_prod,
                        (tokens * FFN) as u32,
                    );
                }
                self.project_layer_matrix(
                    &mut batch,
                    lw.down,
                    &self.ffn_prod,
                    &self.branch,
                    tokens_u32,
                )?;
            }
            let tap_slot = if self.drafter.is_some() {
                DSPARK_TAPS
                    .iter()
                    .position(|&l| l == layer)
                    .map(|idx| idx as u32)
            } else {
                None
            };
            if layer + 1 < LAYERS && ane_offload.is_none() {
                let (next_norm, next_off) = (payload, self.layers[layer + 1].input_norm);
                self.ops.residual_add_rms_tap(
                    &mut batch,
                    &self.hidden_alt,
                    &self.branch,
                    next_norm,
                    next_off,
                    &self.hidden,
                    &self.normed,
                    &self.taps_buf,
                    tap_slot,
                    WIDTH as u32,
                    tokens_u32,
                    EPSILON,
                );
            } else if layer + 1 == LAYERS
                && compute_logits_rows == tokens
                && ane_offload.is_none()
            {
                let (final_norm, final_off) = (payload, self.final_norm);
                if self.artifact.matrix_at(self.lm_head).1 {
                    self.ops.residual_add_rms_fwht512(
                        &mut batch,
                        &self.hidden_alt,
                        &self.branch,
                        final_norm,
                        final_off,
                        &self.rot512_1024.signs,
                        &self.hidden,
                        &self.rotated_1024,
                        WIDTH as u32,
                        tokens_u32,
                        EPSILON,
                    );
                } else {
                    self.ops.residual_add_rms(
                        &mut batch,
                        &self.hidden_alt,
                        &self.branch,
                        final_norm,
                        final_off,
                        &self.hidden,
                        &self.normed,
                        WIDTH as u32,
                        tokens_u32,
                        EPSILON,
                    );
                }
            } else {
                self.ops.residual_add(
                    &mut batch,
                    &self.hidden_alt,
                    &self.branch,
                    &self.hidden,
                    (tokens * WIDTH) as u32,
                );
                if let Some(slot) = tap_slot {
                    self.ops.copy_tap(
                        &mut batch,
                        &self.hidden,
                        &self.taps_buf,
                        slot,
                        tokens_u32,
                    );
                }
                if layer + 1 < LAYERS {
                    let (next_norm, next_off) = (payload, self.layers[layer + 1].input_norm);
                    self.ops.rms_norm(
                        &mut batch,
                        &self.hidden,
                        next_norm,
                        next_off,
                        &self.normed,
                        WIDTH as u32,
                        tokens_u32,
                        EPSILON,
                    );
                }
            }
        }

        if compute_logits_rows > 0 {
            let (final_norm, final_off) = (payload, self.final_norm);
            let (lm_mat, lm_rot) = self.artifact.matrix_at(self.lm_head);
            let (norm_off, norm_rows) = if compute_logits_rows == tokens {
                (0, tokens_u32)
            } else {
                ((tokens - 1) * WIDTH * 4, 1)
            };
            if lm_rot {
                if compute_logits_rows != tokens || ane_offload.is_some() {
                    self.ops.rms_norm_fwht512_at(
                        &mut batch,
                        &self.hidden,
                        norm_off,
                        final_norm,
                        final_off,
                        &self.rot512_1024.signs,
                        &self.rotated_1024,
                        WIDTH as u32,
                        norm_rows,
                        EPSILON,
                    );
                }
                self.proj
                    .matmul(&mut batch, lm_mat, &self.rotated_1024, &self.logits, norm_rows)?;
            } else {
                if compute_logits_rows != tokens || ane_offload.is_some() {
                    self.ops.rms_norm_at(
                        &mut batch,
                        &self.hidden,
                        norm_off,
                        final_norm,
                        final_off,
                        &self.normed,
                        WIDTH as u32,
                        norm_rows,
                        EPSILON,
                    );
                }
                self.proj
                    .matmul(&mut batch, lm_mat, &self.normed, &self.logits, norm_rows)?;
            }
        }

        if sample_top_k > 0 && compute_logits_rows > 0 {
            if !self.dry_scratch.is_empty() {
                let freq_pen = 0.10_f32;
                self.sampling.encode_repetition_penalty(
                    &mut batch,
                    &self.logits,
                    &self.dry_scratch,
                    VOCAB,
                    compute_logits_rows,
                    self.rep_penalty,
                    freq_pen,
                );
            }
            if sample_top_k == 1 {
                self.sampling.encode_greedy_rows(
                    &mut batch,
                    &self.logits,
                    VOCAB,
                    compute_logits_rows,
                )?;
            } else {
                self.sampling.encode_topk_rows(
                    &mut batch,
                    &self.logits,
                    VOCAB,
                    compute_logits_rows,
                    sample_top_k,
                )?;
            }
        }
        if !speculative_verify
            && let Some(d) = &mut self.drafter
        {
            d.encode_append_context(&mut batch, &self.proj, &self.ops, &self.taps_buf, tokens)?;
        }
        Ok((batch, ane_offload))
    }

    fn embed_tokens_into_hidden(&mut self, ids: &[u32]) -> crate::Result<()> {
        let mut batch = CommandBatch::new(&self.ctx)?;
        self.encode_embed_tokens(&mut batch, ids);
        Ok(batch.commit_and_wait()?)
    }

    #[expect(unsafe_code)]
    fn encode_embed_tokens(&mut self, batch: &mut CommandBatch, ids: &[u32]) {
        let count = ids.len();
        unsafe {
            let dst = &mut self.token_ids_buf.as_mut_slice::<u32>()[..count];
            dst.copy_from_slice(ids);
        }
        self.encode_embed_gather(batch, &self.token_ids_buf, count);
    }

    /// Gather the embeddings of the first `count` token ids in `ids` into `hidden`.
    fn encode_embed_gather(&self, batch: &mut CommandBatch, ids: &MetalBuffer, count: usize) {
        let (emb_mat, rotated) = self.artifact.matrix_at(self.embed);
        match emb_mat {
            AnyMatrix::Bf16(m) => {
                self.proj
                    .embed_gather_bf16(batch, m, ids, &self.hidden, count as u32);
            }
            AnyMatrix::Int8(m) => {
                self.proj
                    .embed_gather_int8(batch, m, ids, &self.hidden, count as u32);
            }
            AnyMatrix::Ptq1(m) => {
                if rotated {
                    self.proj.embed_gather_ptq1_fwht512(
                        batch,
                        m,
                        ids,
                        &self.rot512_1024,
                        &self.hidden,
                        count as u32,
                    );
                } else {
                    self.proj.embed_gather_ptq1(
                        batch,
                        m,
                        ids,
                        &self.hidden,
                        count as u32,
                    );
                }
            }
            AnyMatrix::Int4(m) => {
                self.proj
                    .embed_gather_int4(batch, m, ids, &self.hidden, count as u32);
            }
        }
    }

    /// Generate a completion from `prompt_ids` with N-gram lookup and `DSpark` speculation.
    pub fn generate_ids(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        params: SamplingParams,
    ) -> crate::Result<GenerationReport> {
        self.generate_ids_streaming(prompt_ids, max_new_tokens, params, |_| {})
    }
    fn record_recent_token(&mut self, token_id: u32, weight: u32) {
        let tagged = 0xFF00_0000 | (token_id & 0x00FF_FFFF);
        if let Some(entry) = self
            .recent_token_counts
            .iter_mut()
            .find(|pair| pair[0] == tagged)
        {
            entry[1] = entry[1].saturating_add(weight);
        } else if self.recent_token_counts.len() >= 768 {
            self.recent_token_counts.remove(0);
            self.recent_token_counts.push([tagged, weight]);
        } else {
            self.recent_token_counts.push([tagged, weight]);
        }
    }

    /// Build `dry_scratch` combining global `recent_token_counts` (`0xFF` row mask) with, when
    /// `ngrams` (reasoning still open), exact per-row DRY (Don't Repeat Yourself) exponential
    /// n-gram penalties for each row `r` of `block`. Answers, code and tool calls must copy
    /// identifiers verbatim, so they never get DRY. For row `r`, the context ends with `history` +
    /// `block[1..=r]`; any earlier match of length `L >= 3` in the last 512 tokens penalizes it.
    pub(crate) fn build_dry_penalties(
        &mut self,
        history: &[u32],
        block: &[u32],
        ngrams: bool,
        min_pos: usize,
    ) {
        self.dry_scratch.clear();
        if (self.rep_penalty - 1.0).abs() > 1e-4 {
            self.dry_scratch.extend_from_slice(&self.recent_token_counts);
        }
        let hist_len = history.len();
        let search_start = hist_len.saturating_sub(512).max(min_pos);
        if !ngrams || hist_len < search_start + 4 {
            return;
        }
        for row_idx in 0..block.len().min(MAX_VERIFY_TOKENS) {
            let extra = &block[1..=row_idx];
            let total_len = hist_len + extra.len();
            let token_at = |idx: usize| -> u32 {
                if idx < hist_len {
                    history[idx]
                } else {
                    extra[idx - hist_len]
                }
            };
            let last_tok = token_at(total_len - 1);
            let prev_tok = token_at(total_len - 2);
            let mut max_match_per_tok: Vec<(u32, usize)> = Vec::new();
            for end_pos in (search_start + 3..total_len - 1).rev() {
                if token_at(end_pos - 1) == last_tok
                    && token_at(end_pos - 2) == prev_tok
                    && token_at(end_pos - 3) == token_at(total_len - 3)
                {
                    let mut match_len = 3_usize;
                    while match_len < 16
                        && end_pos > match_len
                        && total_len > match_len
                        && token_at(end_pos - 1 - match_len) == token_at(total_len - 1 - match_len)
                    {
                        match_len += 1;
                    }
                    let cand = token_at(end_pos);
                    if let Some(existing) = max_match_per_tok.iter_mut().find(|p| p.0 == cand) {
                        existing.1 = existing.1.max(match_len);
                    } else if max_match_per_tok.len() < 16 {
                        max_match_per_tok.push((cand, match_len));
                    }
                }
            }
            for (cand, match_len) in max_match_per_tok {
                // Standard DRY allowed_length = 2: penalize matches of length >= 3
                let exp_shift = (match_len.saturating_sub(3)).min(7);
                let weight = (8_u32 << exp_shift).min(500);
                let tagged = ((row_idx as u32) << 24) | (cand & 0x00FF_FFFF);
                if self.dry_scratch.len() < 1024 {
                    self.dry_scratch.push([tagged, weight]);
                }
            }
        }
    }
    #[cfg(test)]
    #[must_use]
    pub(crate) fn dry_scratch_entries(&self) -> &[[u32; 2]] {
        &self.dry_scratch
    }

    pub fn generate_ids_streaming(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        params: SamplingParams,
        mut on_piece: impl FnMut(&str),
    ) -> crate::Result<GenerationReport> {
        if prompt_ids.is_empty() {
            return Err(Error::InvalidArgument("prompt_ids cannot be empty".into()));
        }
        let mut sampler = Sampler::new(params);

        // 1. Restore longest matching prefix checkpoint (or zero-copy resident continuation),
        // then prefill remaining tokens in chunks with automatic milestone checkpointing.
        let t_prefill_start = Instant::now();
        let lcp_with_resident = if self.prefix_cache_enabled {
            self.active_tokens
                .iter()
                .zip(prompt_ids)
                .take_while(|(a, b)| a == b)
                .count()
                .min(prompt_ids.len().saturating_sub(1))
        } else {
            0
        };
        let mut reused = self.restore_or_advance_prefix(prompt_ids)?;
        let mut milestones = Vec::with_capacity(3);
        if self.prefix_cache_enabled {
            if lcp_with_resident >= 8 && lcp_with_resident < prompt_ids.len() {
                milestones.push((lcp_with_resident, true));
            }
            // The first turn's end (the system prompt, however long) is the prefix agents resend.
            if let Some(pos) = prompt_ids
                .iter()
                .position(|&id| id == IM_END_ID)
            {
                let se = if prompt_ids.get(pos + 1) == Some(&NEWLINE_ID) {
                    pos + 2
                } else {
                    pos + 1
                };
                if se >= 8 && se < prompt_ids.len() {
                    milestones.push((se, true));
                }
            }
            if prompt_ids.len() >= 32 && prompt_ids.len() <= ANE_MIN_ROWS {
                milestones.push((prompt_ids.len() - 1, false));
            }
            milestones.sort_by_key(|&(pos, _)| pos);
            milestones.dedup_by_key(|m| m.0);
        }
        let mut start = reused;
        while start < prompt_ids.len() {
            let mut end = (start + self.profile.prefill_rows).min(prompt_ids.len());
            let mut hit_milestone = None;
            for &(m_pos, pinned) in &milestones {
                if m_pos > start
                    && m_pos < end
                    && !self
                        .prefix_cache
                        .iter()
                        .any(|ckpt| ckpt.tokens == prompt_ids[..m_pos])
                {
                    end = m_pos;
                    hit_milestone = Some((m_pos, pinned));
                    break;
                }
            }
            let chunk = &prompt_ids[start..end];
            let is_last = end == prompt_ids.len();
            self.dry_scratch.clear();
            match self.forward_step(
                Some(chunk),
                chunk.len(),
                Rope::Text(self.rope_delta),
                false,
                usize::from(is_last),
                if is_last { sampler.active_top_k() } else { 0 },
            ) {
                Ok(()) => {}
                // The ANE failed (now off) and the engine was reset: prefill it all again on the GPU.
                Err(Error::AneFailed) => {
                    (start, reused) = (0, 0);
                    continue;
                }
                Err(e) => return Err(e),
            }
            self.position += chunk.len();
            self.active_tokens.extend_from_slice(chunk);
            if let Some((m_pos, pinned)) = hit_milestone
                && !is_last
            {
                // Best-effort: a checkpoint that cannot be made only costs a later prefill.
                let _ = self.save_prefix_checkpoint(&prompt_ids[..m_pos], pinned);
            }
            start = end;
        }
        self.decode_from_prefilled_logits(
            prompt_ids,
            reused,
            t_prefill_start.elapsed().as_secs_f64() * 1000.0,
            max_new_tokens,
            &mut sampler,
            &mut on_piece,
        )
    }

    fn decode_from_prefilled_logits(
        &mut self,
        prompt_ids: &[u32],
        cached_prefix_tokens: usize,
        prefill_ms: f64,
        max_new_tokens: usize,
        sampler: &mut Sampler,
        on_piece: &mut impl FnMut(&str),
    ) -> crate::Result<GenerationReport> {
        block_on_future(self.decode_from_prefilled_logits_async_stoppable(
            prompt_ids,
            cached_prefix_tokens,
            prefill_ms,
            max_new_tokens,
            sampler,
            on_piece,
            || false,
        ))
    }

    /// Generate a multimodal response from an image (`image_bytes`) + text prompt (`prompt`),
    /// running the `SigLIP2` + Window-Attention Merger vision tower, Canvas 3D M-RoPE, and speculative decoding.
    pub fn generate_multimodal(
        &mut self,
        image_bytes: &[u8],
        prompt: &str,
        mode: DownsampleMode,
        max_new_tokens: usize,
        params: SamplingParams,
    ) -> crate::Result<GenerationReport> {
        self.generate_multimodal_streaming(image_bytes, prompt, mode, max_new_tokens, params, |_| {})
    }

    #[expect(unsafe_code)]
    pub fn generate_multimodal_streaming(
        &mut self,
        image_bytes: &[u8],
        prompt: &str,
        mode: DownsampleMode,
        max_new_tokens: usize,
        params: SamplingParams,
        mut on_piece: impl FnMut(&str),
    ) -> crate::Result<GenerationReport> {
        let prep = PreparedImage::from_bytes(image_bytes, 36)?;
        let mut crop_embeds = Vec::new();
        if self.vision.is_none() {
            if self
                .artifact
                .header
                .tensors
                .contains_key("model.vision_tower.embeddings.patch_embedding.weight")
            {
                self.vision = Some(VisionEncoder::from_artifact(
                    &self.ctx,
                    &self.shaders,
                    self.artifact.clone(),
                )?);
            } else if let Some(vp) = self.vision_path.clone() {
                self.load_vision(&vp)?;
            }
        }
        {
            let vis = self
                .vision
                .as_mut()
                .ok_or_else(|| Error::InvalidArgument("vision tower not loaded".into()))?;
            crop_embeds.push(vis.encode_crop(&self.ctx, &self.shaders, &self.proj, &prep.overview, mode)?);
            for s in &prep.slices {
                crop_embeds.push(vis.encode_crop(&self.ctx, &self.shaders, &self.proj, s, mode)?);
            }
        }
        if !self.profile.keep_vision {
            self.vision = None;
        }
        let flat_img_embeds: Vec<f32> = crop_embeds.into_iter().flatten().collect();

        // Build token sequence: `<|im_start|>user\n<image_id>0</image_id><image>[pads]</image>[slices]\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
        let mut prompt_ids = self.encode_text(&format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n", Self::DEFAULT_SYSTEM_PROMPT))?;
        let zero_ids = self.encode_text("0")?;
        prompt_ids.push(IMAGE_ID_START);
        prompt_ids.extend_from_slice(&zero_ids);
        prompt_ids.push(IMAGE_ID_END);
        prompt_ids.push(IMAGE_START_ID);
        prompt_ids.extend(std::iter::repeat_n(
            IMAGE_TOKEN_ID,
            prep.overview.llm_tokens(mode),
        ));
        prompt_ids.push(IMAGE_END_ID);
        if !prep.slices.is_empty() && prep.slice_rows > 0 && prep.slice_cols > 0 {
            for r in 0..prep.slice_rows {
                if r > 0 {
                    prompt_ids.push(NEWLINE_ID);
                }
                for c in 0..prep.slice_cols {
                    let s = &prep.slices[r * prep.slice_cols + c];
                    prompt_ids.push(SLICE_START_ID);
                    prompt_ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, s.llm_tokens(mode)));
                    prompt_ids.push(SLICE_END_ID);
                }
            }
        }
        prompt_ids.push(NEWLINE_ID);
        let suffix = format!("{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n");
        prompt_ids.extend(self.encode_text(&suffix)?);

        let target_sizes = prep.target_sizes();
        let (pos3d, rope_delta) = compute_canvas_mrope(&prompt_ids, &target_sizes, mode);
        self.reset();
        self.rope_delta = rope_delta;

        // Prefill multimodal prompt with Canvas 3D M-RoPE positions and scattered image features
        let t_prefill_start = Instant::now();
        let mut img_feat_cursor = 0_usize;
        let mut start = 0;
        let mut sampler = Sampler::new(params);
        while start < prompt_ids.len() {
            let end = (start + self.profile.prefill_rows).min(prompt_ids.len());
            let chunk = &prompt_ids[start..end];
            let is_last = end == prompt_ids.len();
            self.embed_tokens_into_hidden(chunk)?;
            unsafe {
                let h_slice = &mut self.hidden.as_mut_slice::<f32>()[..chunk.len() * WIDTH];
                for (t, &id) in chunk.iter().enumerate() {
                    if id == IMAGE_TOKEN_ID
                        && img_feat_cursor + WIDTH <= flat_img_embeds.len()
                    {
                        h_slice[t * WIDTH..(t + 1) * WIDTH].copy_from_slice(
                            &flat_img_embeds[img_feat_cursor..img_feat_cursor + WIDTH],
                        );
                        img_feat_cursor += WIDTH;
                    }
                }
                let r_slice = &mut self.rope_rows_buf.as_mut_slice::<[u32; 4]>()[..chunk.len()];
                r_slice.copy_from_slice(&pos3d[start..end]);
            }
            let rope_buf = self.rope_rows_buf.clone();
            self.dry_scratch.clear();
            self.forward_step(
                None,
                chunk.len(),
                Rope::Canvas(&rope_buf, 0),
                false,
                usize::from(is_last),
                if is_last { sampler.active_top_k() } else { 0 },
            )?;
            self.position += chunk.len();
            start = end;
        }
        let report = self.decode_from_prefilled_logits(
            &prompt_ids,
            0,
            t_prefill_start.elapsed().as_secs_f64() * 1000.0,
            max_new_tokens,
            &mut sampler,
            &mut on_piece,
        );
        self.active_tokens.clear();
        self.rope_delta = 0;
        self.position = 0;
        report
    }

    /// Asynchronously prefill and decode `prompt_ids`, awaiting GPU completion through Metal 4
    /// commit feedback.
    pub async fn generate_ids_streaming_async(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        params: SamplingParams,
        on_piece: impl FnMut(&str),
    ) -> crate::Result<GenerationReport> {
        self.generate_ids_streaming_async_stoppable(
            prompt_ids,
            max_new_tokens,
            params,
            on_piece,
            || false,
        )
        .await
    }

    async fn generate_ids_streaming_async_stoppable(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        params: SamplingParams,
        mut on_piece: impl FnMut(&str),
        mut should_stop: impl FnMut() -> bool,
    ) -> crate::Result<GenerationReport> {
        if prompt_ids.is_empty() {
            return Err(Error::InvalidArgument("prompt_ids cannot be empty".into()));
        }
        let mut sampler = Sampler::new(params);
        let t_prefill_start = Instant::now();
        let lcp_with_resident = if self.prefix_cache_enabled {
            self.active_tokens
                .iter()
                .zip(prompt_ids)
                .take_while(|(a, b)| a == b)
                .count()
                .min(prompt_ids.len().saturating_sub(1))
        } else {
            0
        };
        let mut reused = self.restore_or_advance_prefix(prompt_ids)?;
        let mut milestones = Vec::with_capacity(3);
        if self.prefix_cache_enabled {
            if lcp_with_resident >= 8 && lcp_with_resident < prompt_ids.len() {
                milestones.push((lcp_with_resident, true));
            }
            // The first turn's end (the system prompt, however long) is the prefix agents resend.
            if let Some(pos) = prompt_ids
                .iter()
                .position(|&id| id == IM_END_ID)
            {
                let se = if prompt_ids.get(pos + 1) == Some(&NEWLINE_ID) {
                    pos + 2
                } else {
                    pos + 1
                };
                if se >= 8 && se < prompt_ids.len() {
                    milestones.push((se, true));
                }
            }
            if prompt_ids.len() >= 32 && prompt_ids.len() <= ANE_MIN_ROWS {
                milestones.push((prompt_ids.len() - 1, false));
            }
            milestones.sort_by_key(|&(pos, _)| pos);
            milestones.dedup_by_key(|m| m.0);
        }
        let mut start = reused;
        while start < prompt_ids.len() {
            let mut end = (start + self.profile.prefill_rows).min(prompt_ids.len());
            let mut hit_milestone = None;
            for &(m_pos, pinned) in &milestones {
                if m_pos > start
                    && m_pos < end
                    && !self
                        .prefix_cache
                        .iter()
                        .any(|ckpt| ckpt.tokens == prompt_ids[..m_pos])
                {
                    end = m_pos;
                    hit_milestone = Some((m_pos, pinned));
                    break;
                }
            }
            let chunk = &prompt_ids[start..end];
            let is_last = end == prompt_ids.len();
            self.dry_scratch.clear();
            match self
                .forward_step_async(
                    Some(chunk),
                    chunk.len(),
                    Rope::Text(self.rope_delta),
                    false,
                    usize::from(is_last),
                    if is_last { sampler.active_top_k() } else { 0 },
                )
                .await
            {
                Ok(()) => {}
                // The ANE failed (now off) and the engine was reset: prefill it all again on the GPU.
                Err(Error::AneFailed) => {
                    (start, reused) = (0, 0);
                    continue;
                }
                Err(e) => return Err(e),
            }
            self.position += chunk.len();
            self.active_tokens.extend_from_slice(chunk);
            if let Some((m_pos, pinned)) = hit_milestone
                && !is_last
            {
                // Best-effort: a checkpoint that cannot be made only costs a later prefill.
                let _ = self.save_prefix_checkpoint(&prompt_ids[..m_pos], pinned);
            }
            start = end;
        }
        self.decode_from_prefilled_logits_async_stoppable(
            prompt_ids,
            reused,
            t_prefill_start.elapsed().as_secs_f64() * 1000.0,
            max_new_tokens,
            &mut sampler,
            &mut on_piece,
            &mut should_stop,
        )
        .await
    }

    async fn decode_from_prefilled_logits_async_stoppable(
        &mut self,
        prompt_ids: &[u32],
        cached_prefix_tokens: usize,
        prefill_ms: f64,
        max_new_tokens: usize,
        sampler: &mut Sampler,
        on_piece: &mut impl FnMut(&str),
        mut should_stop: impl FnMut() -> bool,
    ) -> crate::Result<GenerationReport> {
        let max_new_tokens = max_new_tokens.min(MAX_CONTEXT_TOKENS.saturating_sub(self.position));
        let mut history = prompt_ids.to_vec();
        let mut stream_state = Vec::new();
        self.rep_penalty = 1.0;
        self.recent_token_counts.clear();
        // The template leaves `<think>` open for reasoning or closes it up front (`<think>\n\n</think>`).
        let thinking_open = prompt_ids.iter().rposition(|&id| id == THINK_START_ID)
            > prompt_ids.iter().rposition(|&id| id == THINK_END_ID);
        let prompt_len = prompt_ids.len();
        let anchor_only = [*prompt_ids.last().unwrap_or(&0)];
        self.build_dry_penalties(&history, &anchor_only, thinking_open, prompt_len);
        let first_top_k = sampler.active_top_k();
        let first = if !self.dry_scratch.is_empty() {
            let freq_pen = 0.10_f32;
            let fut = {
                let mut batch = CommandBatch::new(&self.ctx)?;
                self.sampling.encode_repetition_penalty(
                    &mut batch,
                    &self.logits,
                    &self.dry_scratch,
                    VOCAB,
                    1,
                    self.rep_penalty,
                    freq_pen,
                );
                if first_top_k == 1 {
                    self.sampling.encode_greedy_rows(&mut batch, &self.logits, VOCAB, 1)?;
                } else {
                    self.sampling.encode_topk_rows(&mut batch, &self.logits, VOCAB, 1, first_top_k)?;
                }
                batch.commit_and_await()
            };
            fut.await?;
            if first_top_k == 1 {
                let token_id = self.sampling.read_greedy_rows(1)[0].best_id;
                crate::sampler::SamplingResult { token_id, is_eos: sampler.is_eos(token_id) }
            } else {
                sampler.select_from_candidates(self.sampling.read_topk_row(0, first_top_k))
            }
        } else if first_top_k == 1 {
            let token_id = self.sampling.read_greedy_rows(1)[0].best_id;
            crate::sampler::SamplingResult { token_id, is_eos: sampler.is_eos(token_id) }
        } else {
            sampler.select_from_candidates(self.sampling.read_topk_row(0, first_top_k))
        };
        let mut generated = vec![first.token_id];
        history.push(first.token_id);
        self.record_recent_token(first.token_id, 1);
        if !first.is_eos
            && let Some(piece) = self.stream_token_piece(&mut stream_state, first.token_id)
        {
            on_piece(&piece);
        }
        if first.is_eos || max_new_tokens <= 1 || should_stop() {
            self.active_tokens = prompt_ids[..self.position.min(prompt_ids.len())].to_vec();
            let (reasoning_text, text) = self.split_reasoning_and_text(&generated)?;
            return Ok(GenerationReport {
                text,
                reasoning_text,
                prompt_tokens: prompt_ids.len(),
                cached_prefix_tokens,
                generated_tokens: generated.len(),
                token_ids: generated,
                prefill_ms,
                decode_ms: 0.0,
                prefill_tok_per_s: (prompt_ids.len() as f64) / (prefill_ms / 1000.0).max(1e-6),
                decode_tok_per_s: 0.0,
                speculative_rounds: 0,
                drafted_tokens: 0,
                accepted_tokens: 0,
                ngram_accepted_tokens: 0,
            });
        }

        let t_decode_start = Instant::now();
        let mut speculative_rounds = 0_usize;
        let mut drafted_tokens = 0_usize;
        let mut accepted_tokens = 0_usize;
        let mut ngram_accepted_tokens = 0_usize;
        // Drafts for the next round, whether they came from the n-gram lookup, and whether their
        // verify block already ran in the previous round's batch.
        let mut pending_drafts: Option<(Vec<u32>, bool, bool)> = None;
        let mut closed_thinking = !thinking_open || first.token_id == THINK_END_ID;
        if closed_thinking {
            self.rep_penalty = sampler.repetition_penalty();
            self.recent_token_counts.clear();
        }
        let mut think_entropies: Vec<f32> = Vec::with_capacity(512);
        let mut global_ent_sum = 0.0_f32;
        let mut local_ent_sum = 0.0_f32;
        let mut rpdi_saturated = false;
        let rpdi_window = 32_usize;

        while generated.len() < max_new_tokens && !should_stop() {
            if !closed_thinking
                && generated.iter().any(|&id| {
                    matches!(
                        id,
                        THINK_END_ID | TOOL_CALL_START_ID | FENCE_NEWLINE_ID | FENCE_BARE_ID
                    )
                })
            {
                closed_thinking = true;
                self.rep_penalty = sampler.repetition_penalty();
                self.recent_token_counts.clear();
            }
            if !closed_thinking {
                let n_th = think_entropies.len();
                if n_th >= rpdi_window * 2 {
                    let gtf = (global_ent_sum / (n_th as f32)).max(0.05);
                    let ltf = local_ent_sum / (rpdi_window as f32);
                    let rpdi = ltf / gtf;
                    // Trigger saturation only when RPDI spikes (> 1.45 after 64 tokens)
                    if rpdi > 1.45 {
                        rpdi_saturated = true;
                    }
                }
            }
            let anchor = *generated.last().unwrap_or(&0);
            let remaining = max_new_tokens - generated.len();
            let mut used_ngram = false;
            let mut pre_verified = false;
            // Sampling verifies drafts by sampling each row (exact speculative sampling), but its
            // top-k pass per row made drafting a net loss (M4 Pro, T=0.7: 178 vs 261 tok/s), so
            // only greedy decoding drafts.
            let drafts = if let Some((d_toks, from_ngram, verified)) = pending_drafts.take() {
                used_ngram = from_ngram;
                pre_verified = verified;
                d_toks
            } else if remaining > 1 && sampler.is_greedy() {
                let ng = lookup_ngram_draft_adaptive(&history, closed_thinking, remaining.min(7));
                if !ng.is_empty() {
                    used_ngram = true;
                    ng
                } else if self.drafter.is_some() {
                    let emb = self.artifact.matrix_at(self.embed);
                    let lm_head = self.artifact.matrix_at(self.lm_head);
                    let fwd_len = self.drafter.as_ref().map_or(4, |d| d.recommended_forward_len(remaining.min(7)));
                    let fuse_initial = (closed_thinking || !rpdi_saturated) && sampler.is_greedy();
                    if fuse_initial {
                        let (blk_len, fut) = {
                            let mut batch = CommandBatch::new_concurrent(&self.ctx)?;
                            let blk_len = if let Some(drafter) = &mut self.drafter {
                                drafter.encode_draft_block_with_penalties(
                                    &mut batch,
                                    &self.proj,
                                    &self.ops,
                                    &self.rot512_1024,
                                    emb,
                                    lm_head,
                                    anchor,
                                    fwd_len,
                                    &self.recent_token_counts,
                                    self.rep_penalty,
                                )?
                            } else {
                                0
                            };
                            let rows = 1 + blk_len;
                            batch.submit_and_renew()?;
                            self.build_dry_penalties(&history, &[anchor], !closed_thinking, prompt_len);
                            if let Some(drafter) = &self.drafter {
                                self.encode_embed_gather(&mut batch, drafter.draft_tokens(), rows);
                            }
                            let (verify_batch, _) = self.encode_forward_step(
                                Some(batch),
                                None,
                                rows,
                                Rope::Text(self.rope_delta),
                                true,
                                rows,
                                sampler.active_top_k(),
                                false,
                            )?;
                            (blk_len, verify_batch.commit_and_await())
                        };
                        fut.await?;
                        pre_verified = true;
                        self.drafter.as_mut().map_or_else(Vec::new, |d| d.finish_draft_block(blk_len, 0.0, true).0)
                    } else if let Some(drafter) = &mut self.drafter {
                        drafter
                            .draft_block_async_with_penalties(
                                &self.ctx,
                                &self.proj,
                                &self.ops,
                                &self.rot512_1024,
                                emb,
                                lm_head,
                                anchor,
                                fwd_len,
                                &self.recent_token_counts,
                                self.rep_penalty,
                            )
                            .await?
                            .0
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            if !drafts.is_empty() {
                speculative_rounds += 1;
                let draft_count = drafts.len();
                drafted_tokens += draft_count;
                let mut block = Vec::with_capacity(1 + draft_count);
                block.push(anchor);
                block.extend_from_slice(&drafts);

                let top_k = sampler.active_top_k();
                let verify_ms = if pre_verified {
                    if !closed_thinking {
                        let has_loop = self
                            .dry_scratch
                            .iter()
                            .any(|&[t, w]| (t >> 24) < 0x80 && w >= 32);
                        if has_loop && generated.len() >= 64 {
                            rpdi_saturated = true;
                        }
                    }
                    0.0
                } else {
                    self.build_dry_penalties(&history, &block, !closed_thinking, prompt_len);
                    if !closed_thinking {
                        let has_loop = self
                            .dry_scratch
                            .iter()
                            .any(|&[t, w]| (t >> 24) < 0x80 && w >= 32);
                        if has_loop && generated.len() >= 64 {
                            rpdi_saturated = true;
                        }
                        if rpdi_saturated && generated.len() >= 128 {
                            let excess = generated.len().saturating_sub(128);
                            let base_boost = (80 + excess * 3).min(300) as u32;
                            for (r, &prev_tok) in block.iter().enumerate() {
                                let mult = if matches!(prev_tok, 13 | 198 | 271 | 624 | 382) {
                                    3
                                } else {
                                    1
                                };
                                if self.dry_scratch.len() < 1024 {
                                    let tagged = (0x80_u32 | (r as u32)) << 24 | THINK_END_ID;
                                    self.dry_scratch.push([tagged, (base_boost * mult).min(500)]);
                                }
                            }
                        }
                    }
                    let t_verify = Instant::now();
                    self.forward_step_async(
                        Some(&block),
                        block.len(),
                        Rope::Text(self.rope_delta),
                        true,
                        block.len(),
                        top_k,
                    )
                    .await?;
                    (t_verify.elapsed().as_secs_f64() * 1000.0) as f32
                };
                let greedy_rows =
                    (top_k == 1).then(|| self.sampling.read_greedy_rows(block.len()));
                let mut accepted = 0_usize;
                let mut hit_eos = false;
                for i in 0..draft_count {
                    let (target_pick, step_ent) = if let Some(rows) = &greedy_rows {
                        let margin = (rows[i].best_logit - rows[i].second_logit).max(0.0);
                        (rows[i].best_id, (-0.5 * margin).exp())
                    } else {
                        let cands = self.sampling.read_topk_row(i, top_k);
                        let margin = if cands.len() >= 2 {
                            (cands[0].logit - cands[1].logit).max(0.0)
                        } else {
                            4.0
                        };
                        (sampler.select_from_candidates(cands).token_id, (-0.5 * margin).exp())
                    };
                    if !closed_thinking && target_pick != THINK_END_ID {
                        global_ent_sum += step_ent;
                        local_ent_sum += step_ent;
                        if think_entropies.len() >= rpdi_window {
                            local_ent_sum -= think_entropies[think_entropies.len() - rpdi_window];
                        }
                        think_entropies.push(step_ent);
                    }
                    if target_pick == drafts[i] {
                        accepted += 1;
                        generated.push(target_pick);
                        history.push(target_pick);
                        self.record_recent_token(target_pick, 1);
                        if sampler.is_eos(target_pick) {
                            hit_eos = true;
                            break;
                        }
                        if let Some(piece) =
                            self.stream_token_piece(&mut stream_state, target_pick)
                        {
                            on_piece(&piece);
                        }
                        if generated.len() >= max_new_tokens || should_stop() {
                            hit_eos = true;
                            break;
                        }
                    } else {
                        generated.push(target_pick);
                        history.push(target_pick);
                        self.record_recent_token(target_pick, 1);
                        if sampler.is_eos(target_pick) {
                            hit_eos = true;
                        } else if let Some(piece) =
                            self.stream_token_piece(&mut stream_state, target_pick)
                        {
                            on_piece(&piece);
                        }
                        if should_stop() {
                            hit_eos = true;
                        }
                        break;
                    }
                }
                if !hit_eos && accepted == draft_count && generated.len() < max_new_tokens {
                    let (bonus, bonus_ent) = if let Some(rows) = &greedy_rows {
                        let r = rows[draft_count];
                        let margin = (r.best_logit - r.second_logit).max(0.0);
                        (r.best_id, (-0.5 * margin).exp())
                    } else {
                        let cands = self.sampling.read_topk_row(draft_count, top_k);
                        let margin = if cands.len() >= 2 {
                            (cands[0].logit - cands[1].logit).max(0.0)
                        } else {
                            4.0
                        };
                        (sampler.select_from_candidates(cands).token_id, (-0.5 * margin).exp())
                    };
                    if !closed_thinking && bonus != THINK_END_ID {
                        global_ent_sum += bonus_ent;
                        local_ent_sum += bonus_ent;
                        if think_entropies.len() >= rpdi_window {
                            local_ent_sum -= think_entropies[think_entropies.len() - rpdi_window];
                        }
                        think_entropies.push(bonus_ent);
                    }
                    generated.push(bonus);
                    history.push(bonus);
                    self.record_recent_token(bonus, 1);
                    if sampler.is_eos(bonus) {
                        hit_eos = true;
                    } else if let Some(piece) =
                        self.stream_token_piece(&mut stream_state, bonus)
                    {
                        on_piece(&piece);
                        if should_stop() {
                            hit_eos = true;
                        }
                    }
                }

                accepted_tokens += accepted;
                if used_ngram {
                    ngram_accepted_tokens += accepted;
                } else if let Some(d) = &mut self.drafter {
                    d.record_verification(accepted, verify_ms);
                }

                if accepted == draft_count {
                    for rstate in &mut self.recurrent {
                        std::mem::swap(&mut rstate.state, &mut rstate.rollback_state);
                        std::mem::swap(&mut rstate.history, &mut rstate.rollback_history);
                    }
                }
                let kept_len = if accepted == draft_count {
                    block.len()
                } else {
                    accepted + 1
                };
                if !hit_eos && generated.len() < max_new_tokens {
                    let kept_u32 = kept_len as u32;
                    let mut fused_draft_blk = None;
                    let mut has_work = false;
                    let next_rem = max_new_tokens - generated.len();
                    // N-gram drafts cost nothing on the GPU; when one exists, skip the DSpark block.
                    let ng_cpu = if next_rem > 1 && sampler.is_greedy() {
                        lookup_ngram_draft_adaptive(&history, closed_thinking, next_rem.min(7))
                    } else {
                        Vec::new()
                    };
                    // With reasoning closed, the verify penalties do not depend on the drafts, so
                    // the next round's verify rides in this batch: one commit and one wait per round.
                    let fuse_verify = (closed_thinking || !rpdi_saturated) && sampler.is_greedy();
                    let mut fused = false;
                    let fut_opt = {
                        let mut batch = CommandBatch::new_concurrent(&self.ctx)?;
                        if accepted < draft_count {
                            has_work = true;
                            batch.independent(|batch| {
                                for rstate in &self.recurrent {
                                    self.ops.gdn_replay_state_only(
                                        batch,
                                        &rstate.tape_qkv,
                                        &rstate.tape_convolved,
                                        &rstate.tape_decay,
                                        &rstate.tape_beta,
                                        &rstate.history,
                                        &rstate.rollback_history,
                                        &rstate.state,
                                        &rstate.rollback_state,
                                        kept_u32,
                                    );
                                }
                                Ok::<(), nano_metal::Error>(())
                            })?;
                            // Swap now rather than after the wait: a fused verify reads the replayed state.
                            for rstate in &mut self.recurrent {
                                std::mem::swap(&mut rstate.state, &mut rstate.rollback_state);
                                std::mem::swap(&mut rstate.history, &mut rstate.rollback_history);
                            }
                        }
                        if let Some(drafter) = &mut self.drafter {
                            has_work = true;
                            drafter.encode_append_context(
                                &mut batch,
                                &self.proj,
                                &self.ops,
                                &self.taps_buf,
                                kept_len,
                            )?;
                        }
                        self.position += kept_len;
                        let next_anchor = *generated.last().unwrap_or(&0);
                        if ng_cpu.is_empty()
                            && next_rem > 1
                            && sampler.is_greedy()
                            && let Some(drafter) = &mut self.drafter
                        {
                            let emb = self.artifact.matrix_at(self.embed);
                            let lm_head = self.artifact.matrix_at(self.lm_head);
                            let fwd_len = drafter.recommended_forward_len(next_rem.min(7));
                            let blk_len = drafter.encode_draft_block_with_penalties(
                                &mut batch,
                                &self.proj,
                                &self.ops,
                                &self.rot512_1024,
                                emb,
                                lm_head,
                                next_anchor,
                                fwd_len,
                                &self.recent_token_counts,
                                self.rep_penalty,
                            )?;
                            fused_draft_blk = Some(blk_len);
                        }
                        let next_drafts = if ng_cpu.is_empty() { fused_draft_blk } else { Some(ng_cpu.len()) };
                        if fuse_verify && let Some(drafts) = next_drafts {
                            let rows = 1 + drafts;
                            // The GPU starts on the replay, append and draft while the verify is encoded.
                            batch.submit_and_renew()?;
                            self.build_dry_penalties(&history, &[next_anchor], !closed_thinking, prompt_len);
                            if ng_cpu.is_empty() {
                                if let Some(drafter) = &self.drafter {
                                    // The drafter's token buffer already holds `[anchor, drafts...]`.
                                    self.encode_embed_gather(&mut batch, drafter.draft_tokens(), rows);
                                }
                            } else {
                                let block: Vec<u32> = std::iter::once(next_anchor).chain(ng_cpu.iter().copied()).collect();
                                self.encode_embed_tokens(&mut batch, &block);
                            }
                            let (verify_batch, _) = self.encode_forward_step(
                                Some(batch),
                                None,
                                rows,
                                Rope::Text(self.rope_delta),
                                true,
                                rows,
                                sampler.active_top_k(),
                                false,
                            )?;
                            fused = true;
                            Some(verify_batch.commit_and_await())
                        } else {
                            (has_work || fused_draft_blk.is_some()).then(|| batch.commit_and_await())
                        }
                    };
                    if let Some(fut) = fut_opt {
                        let t_batch = Instant::now();
                        fut.await?;
                        let batch_ms = (t_batch.elapsed().as_secs_f64() * 1000.0) as f32;
                        if let (Some(blk_len), Some(drafter)) = (fused_draft_blk, &mut self.drafter) {
                            // shortcut: fused rounds cannot split draft from verify time, so they
                            // leave the cost tables to unfused (reasoning) rounds.
                            let draft_ms = if fused { 0.0 } else { batch_ms * 0.7 };
                            let (d_toks, _) = drafter.finish_draft_block(blk_len, draft_ms, fused);
                            pending_drafts = Some((d_toks, false, fused));
                        }
                    }
                    if !ng_cpu.is_empty() {
                        pending_drafts = Some((ng_cpu, true, fused));
                    }
                    continue;
                }
                if accepted == draft_count {
                    if let Some(drafter) = &mut self.drafter {
                        drafter.append_context(
                            &self.ctx,
                            &self.proj,
                            &self.ops,
                            &self.taps_buf,
                            kept_len,
                        )?;
                    }
                    self.position += kept_len;
                }
                break;
            }

            let top_k = sampler.active_top_k();
            self.build_dry_penalties(&history, &[anchor], !closed_thinking, prompt_len);
            if !closed_thinking {
                let has_loop = self
                    .dry_scratch
                    .iter()
                    .any(|&[t, w]| (t >> 24) < 0x80 && w >= 32);
                if has_loop && generated.len() >= 64 {
                    rpdi_saturated = true;
                }
                if rpdi_saturated && generated.len() >= 128 && self.dry_scratch.len() < 1024 {
                    let excess = generated.len().saturating_sub(128);
                    let base_boost = (80 + excess * 3).min(300) as u32;
                    let mult = if matches!(anchor, 13 | 198 | 271 | 624 | 382) {
                        3
                    } else {
                        1
                    };
                    let tagged = (0x80_u32 << 24) | THINK_END_ID;
                    self.dry_scratch.push([tagged, (base_boost * mult).min(500)]);
                }
            }
            self.forward_step_async(
                Some(&[anchor]),
                1,
                Rope::Text(self.rope_delta),
                false,
                1,
                top_k,
            )
            .await?;
            self.position += 1;
            let (next, step_ent) = if top_k == 1 {
                let sel = self.sampling.read_greedy_rows(1)[0];
                let margin = (sel.best_logit - sel.second_logit).max(0.0);
                (
                    crate::sampler::SamplingResult {
                        token_id: sel.best_id,
                        is_eos: sampler.is_eos(sel.best_id),
                    },
                    (-0.5 * margin).exp(),
                )
            } else {
                let cands = self.sampling.read_topk_row(0, top_k);
                let margin = if cands.len() >= 2 {
                    (cands[0].logit - cands[1].logit).max(0.0)
                } else {
                    4.0
                };
                (sampler.select_from_candidates(cands), (-0.5 * margin).exp())
            };
            if !closed_thinking && next.token_id != THINK_END_ID {
                global_ent_sum += step_ent;
                local_ent_sum += step_ent;
                if think_entropies.len() >= rpdi_window {
                    local_ent_sum -= think_entropies[think_entropies.len() - rpdi_window];
                }
                think_entropies.push(step_ent);
            }
            generated.push(next.token_id);
            history.push(next.token_id);
            self.record_recent_token(next.token_id, 1);
            if next.is_eos {
                break;
            }
            if let Some(piece) = self.stream_token_piece(&mut stream_state, next.token_id) {
                on_piece(&piece);
                if should_stop() {
                    break;
                }
            }
        }

        self.active_tokens = history[..self.position.min(history.len())].to_vec();
        let decode_ms = t_decode_start.elapsed().as_secs_f64() * 1000.0;
        let decode_tokens = generated.len().saturating_sub(1);
        let (reasoning_text, text) = self.split_reasoning_and_text(&generated)?;
        Ok(GenerationReport {
            text,
            reasoning_text,
            prompt_tokens: prompt_ids.len(),
            cached_prefix_tokens,
            generated_tokens: generated.len(),
            token_ids: generated,
            prefill_ms,
            decode_ms,
            prefill_tok_per_s: (prompt_ids.len() as f64) / (prefill_ms / 1000.0).max(1e-6),
            decode_tok_per_s: if decode_ms > 0.0 {
                (decode_tokens as f64) / (decode_ms / 1000.0)
            } else {
                0.0
            },
            speculative_rounds,
            drafted_tokens,
            accepted_tokens,
            ngram_accepted_tokens,
        })
    }

    /// Asynchronously generate a multimodal response from an image (`image_bytes`) + text prompt (`prompt`),
    /// overlapping CPU image preparation with GPU vision encoding and async decode.
    pub async fn generate_multimodal_streaming_async(
        &mut self,
        image_bytes: &[u8],
        prompt: &str,
        mode: DownsampleMode,
        max_new_tokens: usize,
        params: SamplingParams,
        on_piece: impl FnMut(&str),
    ) -> crate::Result<GenerationReport> {
        self.generate_multimodal_streaming_async_stoppable(
            image_bytes,
            prompt,
            mode,
            max_new_tokens,
            params,
            on_piece,
            || false,
        )
        .await
    }

    #[expect(unsafe_code)]
    async fn generate_multimodal_streaming_async_stoppable(
        &mut self,
        image_bytes: &[u8],
        prompt: &str,
        mode: DownsampleMode,
        max_new_tokens: usize,
        params: SamplingParams,
        mut on_piece: impl FnMut(&str),
        mut should_stop: impl FnMut() -> bool,
    ) -> crate::Result<GenerationReport> {
        let owned_bytes = image_bytes.to_vec();
        let prep = tokio::task::spawn_blocking(move || PreparedImage::from_bytes(&owned_bytes, 36))
            .await
            .map_err(|e| Error::InvalidArgument(format!("image prep join failed: {e}")))??;
        let mut crop_embeds = Vec::new();
        if self.vision.is_none() {
            if self
                .artifact
                .header
                .tensors
                .contains_key("model.vision_tower.embeddings.patch_embedding.weight")
            {
                self.vision = Some(VisionEncoder::from_artifact(
                    &self.ctx,
                    &self.shaders,
                    self.artifact.clone(),
                )?);
            } else if let Some(vp) = self.vision_path.clone() {
                self.load_vision(&vp)?;
            }
        }
        {
            let vis = self
                .vision
                .as_mut()
                .ok_or_else(|| Error::InvalidArgument("vision tower not loaded".into()))?;
            crop_embeds.push(
                vis.encode_crop_async(&self.ctx, &self.shaders, &self.proj, &prep.overview, mode)
                    .await?,
            );
            for s in &prep.slices {
                crop_embeds.push(
                    vis.encode_crop_async(&self.ctx, &self.shaders, &self.proj, s, mode)
                        .await?,
                );
            }
        }
        if !self.profile.keep_vision {
            self.vision = None;
        }
        let flat_img_embeds: Vec<f32> = crop_embeds.into_iter().flatten().collect();

        let mut prompt_ids = self.encode_text(&format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n", Self::DEFAULT_SYSTEM_PROMPT))?;
        let zero_ids = self.encode_text("0")?;
        prompt_ids.push(IMAGE_ID_START);
        prompt_ids.extend_from_slice(&zero_ids);
        prompt_ids.push(IMAGE_ID_END);
        prompt_ids.push(IMAGE_START_ID);
        prompt_ids.extend(std::iter::repeat_n(
            IMAGE_TOKEN_ID,
            prep.overview.llm_tokens(mode),
        ));
        prompt_ids.push(IMAGE_END_ID);
        if !prep.slices.is_empty() && prep.slice_rows > 0 && prep.slice_cols > 0 {
            for r in 0..prep.slice_rows {
                if r > 0 {
                    prompt_ids.push(NEWLINE_ID);
                }
                for c in 0..prep.slice_cols {
                    let s = &prep.slices[r * prep.slice_cols + c];
                    prompt_ids.push(SLICE_START_ID);
                    prompt_ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, s.llm_tokens(mode)));
                    prompt_ids.push(SLICE_END_ID);
                }
            }
        }
        prompt_ids.push(NEWLINE_ID);
        let suffix = format!("{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n");
        prompt_ids.extend(self.encode_text(&suffix)?);

        let target_sizes = prep.target_sizes();
        let (pos3d, rope_delta) = compute_canvas_mrope(&prompt_ids, &target_sizes, mode);
        self.reset();
        self.rope_delta = rope_delta;

        let t_prefill_start = Instant::now();
        let mut img_feat_cursor = 0_usize;
        let mut start = 0;
        let mut sampler = Sampler::new(params);
        while start < prompt_ids.len() {
            let end = (start + self.profile.prefill_rows).min(prompt_ids.len());
            let chunk = &prompt_ids[start..end];
            let is_last = end == prompt_ids.len();
            self.embed_tokens_into_hidden(chunk)?;
            unsafe {
                let h_slice = &mut self.hidden.as_mut_slice::<f32>()[..chunk.len() * WIDTH];
                for (t, &id) in chunk.iter().enumerate() {
                    if id == IMAGE_TOKEN_ID && img_feat_cursor + WIDTH <= flat_img_embeds.len() {
                        h_slice[t * WIDTH..(t + 1) * WIDTH].copy_from_slice(
                            &flat_img_embeds[img_feat_cursor..img_feat_cursor + WIDTH],
                        );
                        img_feat_cursor += WIDTH;
                    }
                }
                let r_slice = &mut self.rope_rows_buf.as_mut_slice::<[u32; 4]>()[..chunk.len()];
                r_slice.copy_from_slice(&pos3d[start..end]);
            }
            let rope_buf = self.rope_rows_buf.clone();
            self.dry_scratch.clear();
            self.forward_step_async(
                None,
                chunk.len(),
                Rope::Canvas(&rope_buf, 0),
                false,
                usize::from(is_last),
                if is_last { sampler.active_top_k() } else { 0 },
            )
            .await?;
            self.position += chunk.len();
            start = end;
        }
        let report = self
            .decode_from_prefilled_logits_async_stoppable(
                &prompt_ids,
                0,
                t_prefill_start.elapsed().as_secs_f64() * 1000.0,
                max_new_tokens,
                &mut sampler,
                &mut on_piece,
                &mut should_stop,
            )
            .await;
        self.active_tokens.clear();
        self.rope_delta = 0;
        self.position = 0;
        report
    }

    #[must_use]
    #[expect(unsafe_code)]
    pub fn last_logits(&self) -> Vec<f32> {
        let slice = unsafe { self.logits.as_slice::<f32>() };
        slice[..VOCAB].to_vec()
    }
}

// ============================================================================
// Tokio Async Multi-Agent Engine Handle (`AsyncEngineHandle`)
// ============================================================================

/// Streaming event emitted per agent during `AsyncEngineHandle::run_agents_streaming`.
pub enum AgentStreamEvent {
    Header {
        agent_idx: usize,
        total_agents: usize,
    },
    Piece(String),
    AgentDone {
        device_name: String,
        ane_enabled: bool,
        report: Box<GenerationReport>,
    },
}

enum AsyncEngineCommand {
    ChatText {
        prompt_text: String,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        reply: tokio::sync::oneshot::Sender<crate::Result<GenerationReport>>,
    },
    ChatIds {
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        reply: tokio::sync::oneshot::Sender<crate::Result<GenerationReport>>,
    },
    Multimodal {
        image_bytes: Vec<u8>,
        prompt: String,
        mode: DownsampleMode,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        reply: tokio::sync::oneshot::Sender<crate::Result<GenerationReport>>,
    },
    WarmPrefix {
        prefix_text: String,
        reply: tokio::sync::oneshot::Sender<crate::Result<usize>>,
    },
    Score {
        prompt: ScorePrompt,
        reply: tokio::sync::oneshot::Sender<crate::Result<ScoredPrompt>>,
    },
    Agents {
        system_prompt: String,
        agent_prompts: Vec<String>,
        max_new_tokens: usize,
        params: SamplingParams,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<AgentStreamEvent>>,
        reply: tokio::sync::oneshot::Sender<crate::Result<Vec<GenerationReport>>>,
    },
}

/// A raw prompt to score: text the engine tokenizes, or token ids a client already produced.
pub enum ScorePrompt {
    Text(String),
    Ids(Vec<u32>),
}

/// A teacher-forced prompt: its tokens and their text, the per-position scores, and the text of
/// each position's top token.
pub struct ScoredPrompt {
    pub ids: Vec<u32>,
    pub pieces: Vec<String>,
    pub scores: Vec<TokenScore>,
    pub top_pieces: Vec<String>,
}

/// Thread-safe `Clone + Send + Sync` Tokio async handle to a dedicated `NanoEngine` worker.
///
/// Runs a single-threaded Tokio runtime that awaits non-blocking GPU completions and shares `PrefixCache`.
#[derive(Clone)]
pub struct AsyncEngineHandle {
    tx: tokio::sync::mpsc::UnboundedSender<AsyncEngineCommand>,
    device_name: String,
    ane_status: serde_json::Value,
}

impl AsyncEngineHandle {
    /// Open `NanoEngine` on a dedicated async worker thread and return a cloneable `AsyncEngineHandle`.
    pub async fn open(model_path: &Path) -> crate::Result<Self> {
        let path_buf = model_path.to_path_buf();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AsyncEngineCommand>();
        let (init_tx, init_rx) =
            tokio::sync::oneshot::channel::<crate::Result<(String, serde_json::Value)>>();
        std::thread::Builder::new()
            .name("nano-engine-tokio-worker".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(r) => r,
                    Err(err) => {
                        let _ = init_tx.send(Err(Error::Io(err)));
                        return;
                    }
                };
                let mut engine = match NanoEngine::open(&path_buf) {
                    Ok(eng) => {
                        let _ = init_tx.send(Ok((eng.device_name(), eng.ane_status())));
                        eng
                    }
                    Err(err) => {
                        let _ = init_tx.send(Err(err));
                        return;
                    }
                };
                rt.block_on(async move {
                    while let Some(cmd) = rx.recv().await {
                        match cmd {
                            AsyncEngineCommand::ChatText {
                                prompt_text,
                                max_new_tokens,
                                params,
                                piece_tx,
                                reply,
                            } => {
                                let res = match engine.encode_text(&prompt_text) {
                                    Ok(ids) => {
                                        engine
                                            .generate_ids_streaming_async_stoppable(
                                                &ids,
                                                max_new_tokens,
                                                params,
                                                |piece| {
                                                    if let Some(tx) = &piece_tx {
                                                        let _ = tx.send(piece.to_owned());
                                                    }
                                                },
                                                || piece_tx.as_ref().is_some_and(tokio::sync::mpsc::UnboundedSender::is_closed),
                                            )
                                            .await
                                    }
                                    Err(e) => Err(e),
                                };
                                let _ = reply.send(res);
                            }
                            AsyncEngineCommand::ChatIds {
                                prompt_ids,
                                max_new_tokens,
                                params,
                                piece_tx,
                                reply,
                            } => {
                                let res = engine
                                    .generate_ids_streaming_async_stoppable(
                                        &prompt_ids,
                                        max_new_tokens,
                                        params,
                                        |piece| {
                                            if let Some(tx) = &piece_tx {
                                                let _ = tx.send(piece.to_owned());
                                            }
                                        },
                                        || piece_tx.as_ref().is_some_and(tokio::sync::mpsc::UnboundedSender::is_closed),
                                    )
                                    .await;
                                let _ = reply.send(res);
                            }
                            AsyncEngineCommand::Multimodal {
                                image_bytes,
                                prompt,
                                mode,
                                max_new_tokens,
                                params,
                                piece_tx,
                                reply,
                            } => {
                                let res = engine
                                    .generate_multimodal_streaming_async_stoppable(
                                        &image_bytes,
                                        &prompt,
                                        mode,
                                        max_new_tokens,
                                        params,
                                        |piece| {
                                            if let Some(tx) = &piece_tx {
                                                let _ = tx.send(piece.to_owned());
                                            }
                                        },
                                        || piece_tx.as_ref().is_some_and(tokio::sync::mpsc::UnboundedSender::is_closed),
                                    )
                                    .await;
                                let _ = reply.send(res);
                            }
                            AsyncEngineCommand::WarmPrefix { prefix_text, reply } => {
                                let _ = reply.send(engine.warm_prefix_cache(&prefix_text));
                            }
                            AsyncEngineCommand::Score { prompt, reply } => {
                                let _ = reply.send(engine.score_prompt(prompt));
                            }
                            AsyncEngineCommand::Agents {
                                system_prompt,
                                agent_prompts,
                                max_new_tokens,
                                params,
                                event_tx,
                                reply,
                            } => {
                                let res = async {
                                    if agent_prompts.len() > 1 {
                                        let shared_prefix = format!(
                                            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
                                        );
                                        let _ = engine.warm_prefix_cache(&shared_prefix);
                                    }
                                    let total_agents = agent_prompts.len();
                                    let dev_name = engine.device_name();
                                    let ane_enabled =
                                        engine.ane_status()["enabled"].as_bool().unwrap_or(false);
                                    let mut reports = Vec::with_capacity(total_agents);
                                    for (agent_idx, agent_prompt) in
                                        agent_prompts.iter().enumerate()
                                    {
                                        if let Some(tx) = &event_tx {
                                            let _ = tx.send(AgentStreamEvent::Header {
                                                agent_idx,
                                                total_agents,
                                            });
                                        }
                                        let formatted = NanoEngine::format_chat_prompt_with_system(
                                            &system_prompt,
                                            agent_prompt,
                                        );
                                        let ids = engine.encode_text(&formatted)?;
                                        let rep = engine
                                            .generate_ids_streaming_async(
                                                &ids,
                                                max_new_tokens,
                                                params.clone(),
                                                |piece| {
                                                    if let Some(tx) = &event_tx {
                                                        let _ = tx.send(AgentStreamEvent::Piece(
                                                            piece.to_owned(),
                                                        ));
                                                    }
                                                },
                                            )
                                            .await?;
                                        if let Some(tx) = &event_tx {
                                            let _ = tx.send(AgentStreamEvent::AgentDone {
                                                device_name: dev_name.clone(),
                                                ane_enabled,
                                                report: Box::new(rep.clone()),
                                            });
                                        }
                                        reports.push(rep);
                                    }
                                    Ok(reports)
                                }
                                .await;
                                let _ = reply.send(res);
                            }
                        }
                    }
                });
            })
            .map_err(Error::Io)?;
        let (device_name, ane_status) = init_rx
            .await
            .map_err(|_| Error::InvalidArgument("worker init dropped".into()))??;
        Ok(Self {
            tx,
            device_name,
            ane_status,
        })
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    #[must_use]
    pub const fn ane_status(&self) -> &serde_json::Value {
        &self.ane_status
    }

    #[must_use]
    pub fn ane_enabled(&self) -> bool {
        self.ane_status["enabled"].as_bool().unwrap_or(false)
    }

    /// Asynchronously pre-prefill and pin `prefix_text` in the shared GPU `PrefixCache`.
    pub async fn warm_prefix_cache(&self, prefix_text: String) -> crate::Result<usize> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::WarmPrefix {
                prefix_text,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously teacher-force `prompt` (see `NanoEngine::score_tokens`).
    pub async fn score_prompt(&self, prompt: ScorePrompt) -> crate::Result<ScoredPrompt> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::Score {
                prompt,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously encode and generate a completion from `prompt_text`, streaming pieces to `piece_tx` if provided.
    pub async fn generate_text(
        &self,
        prompt_text: String,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> crate::Result<GenerationReport> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::ChatText {
                prompt_text,
                max_new_tokens,
                params,
                piece_tx,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously generate a completion from `prompt_ids`, streaming pieces to `piece_tx` if provided.
    pub async fn generate_ids(
        &self,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> crate::Result<GenerationReport> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::ChatIds {
                prompt_ids,
                max_new_tokens,
                params,
                piece_tx,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously generate a multimodal completion from `image_bytes` + `prompt`.
    pub async fn generate_multimodal(
        &self,
        image_bytes: Vec<u8>,
        prompt: String,
        mode: DownsampleMode,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> crate::Result<GenerationReport> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::Multimodal {
                image_bytes,
                prompt,
                mode,
                max_new_tokens,
                params,
                piece_tx,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously execute a batch of collaborating agents sharing `system_prompt` and `PrefixCache`.
    pub async fn run_agents(
        &self,
        system_prompt: String,
        agent_prompts: Vec<String>,
        max_new_tokens: usize,
        params: SamplingParams,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<AgentStreamEvent>>,
    ) -> crate::Result<Vec<GenerationReport>> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::Agents {
                system_prompt,
                agent_prompts,
                max_new_tokens,
                params,
                event_tx,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }
}

