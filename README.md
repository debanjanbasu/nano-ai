# nano-ai

Native Rust and Metal inference engine for **MiniCPM-V 4.7 1B** (`openbmb/MiniCPM-V-4.7-1B`) and its **DSpark** speculative drafter (`openbmb/MiniCPM-V-4.7-1B-DSpark`) on Apple Silicon Macs and iPhones/iPads.

Targets the latest Apple platforms only: macOS / iOS 27.2 (Xcode 27.2 SDKs), Metal 4 with the Metal Shading Language 4.1, Swift 6, on Apple7-family GPUs or newer (A14 / M1 and later; tested on an M4 Pro and an iPhone SE 3 with an A15).

## Architecture & Optimizations

1. **3-in-1 Fused Multimodal Artifact (`MCV47` = Language + Vision + DSpark), File-Backed and Zero-Copy**
   - All **842 tensors** across the 24-layer Language Model, 27-layer `SigLIP2` + Window-Attention Merger Vision Tower, and 5-layer `DSpark` speculative drafter live in **one file (`minicpm-v-4.7-1b-tern4.bin`)**. Every matrix is AWQ + WLS **`INT4_G64`** (4-bit, group-64 FP14 `(scale, bias)`), except the 16-row GDN `in_proj_a/b` (BF16) and norms/biases (F32).
   - **Tied Embedding / LM-Head:** one `[248094, 1024]` `INT4_G64` table serves `embed_tokens.weight` and `lm_head.weight`.
   - **Compressed on disk (`713.5 MiB` model + `2.8 MiB` `tokenizer.bin` = `716.3 MiB` total), zero-copy in Metal:** `nano-ai convert` rounds `INT4_G64` `(scale, bias)` to FP14 with `q` re-snapping, compresses the fused artifact with `zstd_col_delta` (8 MiB frames decoded across all CPU cores in ~150 ms on M4 Pro into a page-aligned `Mmap`, or mapped file-backed when stored raw), and Zstd-compresses `tokenizer.bin` (`2.8 MiB`, decoded in < 3 ms), eliminating `tokenizer.json` (`-19.1 MiB`).
2. **MiniCPM-V 4.7 1B Language Model: 24-Layer Hybrid with the Full 262,144-Token Context on Every Device**
   - **18 Gated DeltaNet Recurrent Layers:** Fused 6144-channel `conv1d` + L2 Q/K normalization + `softplus` decay/beta kernel (`mc_conv_l2_decay`), F16 recurrent state `[16 * 128][128]` with F32 register recurrence (`mc_gdn_rows_4_f16`), and zero-copy speculative state rollback.
   - **6 Full-Attention Layers:** Canvas 3D M-RoPE (`sections = [11, 11, 10]`, `partial_rotary_factor = 0.25`), Hadamard-256 rotated KV cache in 8 bits (**6.4 KiB/token** across all 6 layers, K + V) when a full-window cache fits the device's memory budget, else 4 bits (3.4 KiB/token), starting at 1,024 tokens and doubling on demand up to **`MAX_CONTEXT_TOKENS = 262,144`**.
   - **Attention kernels sized for long contexts:** prefill and speculative-verify blocks run GQA-packed flash attention (`mc_attn_flash`): each SIMD group owns 2 query rows × the 4 query heads that share a KV head, K/V tiles are dequantized once into threadgroup memory, and `simdgroup_half8x8` matrices run Q·Kᵀ and P·V under an exp2 online softmax. Single-row decode runs a key-parallel kernel (`mc_attn_decode`: one key per lane, then 8 output dims per lane). Both split the key prefix across threadgroups for parallelism and merge with `mc_attn_reduce`; the split partials take 16.9 MiB whatever the context (they took 67.6 MiB at 262K before).
   - **Bandwidth-bound 1–2 token `INT4_G64` kernels** (`int4_mv*`: 8 columns per lane, every weight byte read once for both tokens; ~255 GB/s on the 248,094-row `lm_head`) with fused Q/K/V, GDN (`qkv` + `z` + BF16 `a`/`b`) and SwiGLU variants; 3–8 token blocks use `simdgroup_float8x8` tiles (`int4_matmul_wide*`). Those split each tile's columns across 8 SIMD groups on M-series GPUs; on A-series (phone) GPUs they use 2 (`*_s2`) and 4-tile threadgroups for `lm_head`, which made the SE 3's verify and draft blocks ~25% cheaper (its 8-way cross-SIMD reduction cost more than the tiles).
   - **Prefill GEMMs on Metal Performance Primitives:** `INT4_G64` projections of 256+ rows dequantize each matrix to a half scratch and run MPP `matmul2d` (`shaders/mpp.metal`); gate + up share one pass with the SwiGLU fused into its store. Decode and verify blocks keep the `int4_*` kernels.
   - **Fused Residual + RMSNorm (`mc_add_rms`), Fused Gate/Up SwiGLU & Nightly `std::simd::f32x16`:** Eliminates intermediate memory round-trips; prefill projects only the final prompt row through `lm_head`.

3. **SigLIP2 + Intra-ViT 2x2 Window-Attention Merger Vision Tower**
   - Supports both `16x` (64 tokens per 448x448 crop via layer-6 `vit_merger` window attention) and `4x` (256 tokens per crop) modes.
   - Exact Python half-to-even `round()` LLaVA-UHD slicing and Canvas 3D M-RoPE (`compute_canvas_mrope`) verified against Hugging Face and Kaggle golden fixtures.

4. **DSpark Semi-Autoregressive Speculative Decoding + N-Gram Lookup**
   - Zero-cost N-gram prompt suffix lookup drafter (when it hits, the DSpark block is skipped) + 5-layer DSpark parallel block drafter with GPU-resident `INT4_G64` low-rank (`r = 256`) Markov transition head (`ds_markov_conf_step_int4`, `ds_markov_argmax_partial_int4`), `simdgroup_float8x8` multi-row `INT4_G64` block matmuls, and Hardware-Aware Confidence Prefix Scheduler (`schedule_verification_length`). Speculation applies to greedy decoding; sampled decoding runs one token per step (sample-and-compare verification is exact but its per-row top-k made drafting a net loss: 178 vs 261 tok/s at T = 0.7).
   - The drafter attends only to its last 1,024 context positions plus its block, and its KV is a 2,048-position ring (11 MiB at any context; a full-context cache was 1.4 GiB at 262K). Seeing a long context made it worse, not better: at 41K tokens it had 60 of 661 drafts accepted with full attention and 152 of 306 with the window, and decode went from 58 to 245 tok/s.
   - In both `<think>` reasoning (while not RPDI-saturated) and final answers/code/tool calls, each round's rollback replay, drafter KV append, next draft block and that block's verify go to the GPU as one submission (the verify embeds the drafter's token buffer directly), so a round costs one wait instead of two.
5. **Apple Neural Engine (`AppleNeuralEngine.framework`) on Every Chip That Has One, Mac and iPhone**
   - Language prefill FFN for chunks of 256–512 rows (MIL functions `ffn256`, `ffn512`: `1024 -> 3584 -> 1024` SwiGLU) and the vision MLP (`vmlp_{256,1024}`: `1152 -> 4304 -> 1152`, 27 SigLIP2 layers). Narrower chunks stay on the GPU, where they are faster.
   - The FFN runs in fp16 end to end (input, the exact `INT4_G64` weight values, the SwiGLU intermediate), with the `down_proj` input scaled ×32 and back so the ANE's fixed output error floor stays negligible (the Gemma 4 trick for fp16 inference). An earlier int8 path cost 0.13 wikitext perplexity (20.143 vs 20.009) and agreed with the GPU on only 94.9% of top-1 tokens; fp16 gives 20.009 and 99.6%, for ~12% of the ANE's prefill gain and +23 MB.
   - The programs compile in the background (cached in the user cache directory, or the app's `Library/Caches` on iOS) and prefill never waits for them. The first wide chunk runs four layers' FFN on both units: the ANE is kept only if it agrees with the GPU (relative error ≤ 0.01) and is faster on this chip (`ane.ffn_layer_ms` in the status).
   - Zero-copy `IOSurface` shared memory and `MTLSharedEvent` GPU-to-ANE-to-GPU handoffs overlap next-layer `INT4_G64` weight staging on the GPU with the ANE's current layer.
6. **Device-Adaptive Memory Profile (`MemoryProfile`)**
   - Sized at load from the memory the process may use: `os_proc_available_memory()` on iOS, Metal's recommended working set on macOS.

     | Setting | Mac (budget ≥ 8 GiB) | iPhone SE 3 (~2 GiB budget) |
     | --- | --- | --- |
     | Prefill chunk | 512 rows | 512 rows with the ANE, 128 without |
     | Full-attention KV | 8-bit | 4-bit (full 262,144-token window still fits) |
     | Drafter KV | 2,048-position ring (11 MiB) | 2,048-position ring (11 MiB) |
     | Prefix-cache checkpoints | 12 | 2 |
     | On-disk prompt cache | 4 GiB | 512 MiB |
     | Vision scratch (128 MiB) | kept between images | freed after each image |
   - The vision tower and its scratch are built on the first image, so text-only sessions never allocate them.
   - **Persistent prompt cache:** every prefix checkpoint (the system turn, shared agent prefixes, the latest conversation tail) is also written to `<model dir>/.prompt-cache/` off the decode path and found again after a restart, where the file is mapped zero-copy as Metal buffers. A ~4.1K-token system prompt restored in 39 ms after a server restart against 1,541 ms of prefill. Files are validated (token hash, sizes, layout) and evicted least-recently-used within the budget above.
7. **Compact Tokenizer (`tokenizer.bin`)**
   - The byte-level BPE (added tokens, NFC, the pre-tokenizer regex hand-coded over the same Unicode tables, merge-rank BPE with reusable scratch buffers, streaming UTF-8 decode) loads from a **2.8 MiB** Zstd-compressed binary (`6.5 MiB` uncompressed in memory), replacing ~100 MiB of `tokenizers` heap and `tokenizer.json`.

8. **OpenAI-Compatible Server for Agent and Coding Harnesses (`nano-ai serve`)**
   - `/v1/chat/completions` renders the model's own chat template byte for byte, including `tools`, assistant `tool_calls`, grouped `tool` results, `reasoning_content` and `enable_thinking` (**enabled by default**, matching the upstream `chat_template.jinja`, and dynamically bounded by RPDI entropy saturation and DRY n-gram loop detection). It parses `<tool_call><function=…><parameter=…>` (and JSON-body fallbacks) into `tool_calls`, typing each argument by its JSON schema, and supports `tool_choice` (`none`, `auto`, `required`, a named function), `stop`, and streaming with `delta.tool_calls`.
   - `/v1/completions` accepts string or token-id prompts. With `echo` / `logprobs` it scores the prompt teacher-forced in 128-row blocks, which is the mode lm-evaluation-harness uses for multiple-choice and perplexity tasks.
   - `/v1/chat/completions`, `chat` and `agents` default `max_tokens` to the full **`MAX_CONTEXT_TOKENS = 262,144`** window.
   - Decoding applies no hidden penalties to answers, code or tool calls: DRY n-gram penalties (bounded strictly to generated tokens so prompt identifiers are never penalized) and the dynamic RPDI reasoning loop breaker act only inside an open `<think>` block, and `repetition_penalty` defaults to `1.0`.
   - `Accept-Encoding: zstd-dict, zstd` and `Content-Encoding: zstd-dict, zstd` use a pre-trained Zstd dictionary (`GET /v1/zstd-dict`, `application/zstd-dictionary`) tailored for OpenAI JSON and SSE payloads.

9. **Metal 4 Runtime (macOS / iOS 27.2+)**
   - `MTL4CommandQueue` with recycled `MTL4CommandAllocator`s, argument tables and upload arenas per batch; buffers bind by GPU address and small constants go through the arena. Serial batches place a barrier after every dispatch, concurrent ones only around `independent` groups; each command buffer waits on earlier queue work.
   - Every `MetalBuffer` (weights, scratch, KV caches, ANE `IOSurface`s, prompt-cache maps) joins one `MTLResidencySet` for its whole life; dropped buffers leave it once no command buffer that might use them is in flight.
   - Completion arrives through `MTL4CommitFeedback`, for blocking and `async` waits alike; pipelines compile through `MTL4Compiler`.

## Benchmarks

Greedy decoding with DSpark + n-gram speculation; prompt-processing numbers are steady state (the first wide prompt also times the ANE against the GPU).

| Device | Workload | Prefill | Decode | Peak footprint |
| --- | --- | --- | --- | --- |
| M4 Pro | Text, 867-token prompt | **~3,570 tok/s** (ANE FFN) | **~540–608 tok/s** | 258 MB (`chat`) |
| M4 Pro | Image (bald eagle, `16x`) | 2,236 tok/s | 379 tok/s | 482 MB |
| iPhone SE 3 (A15, 4 GB) | Text, 867-token prompt | **~415 tok/s** (ANE FFN) | **48–58 tok/s** | ~220 MiB |
| iPhone SE 3 (A15, 4 GB) | Short prompt (31 tokens) | 274 tok/s (GPU) | 48 tok/s | 164 MiB |
| iPhone SE 3 (A15, 4 GB) | Image, steady state | 409 tok/s | 55 tok/s, ~5.1 s per answer | ~357 MiB |

On the SE 3 the ANE beats the GPU from 256-row chunks up (415 vs ~290 tok/s) and runs the vision MLP in ~3.2 s per image against ~5.3 s on the GPU alone.

SE 3 speculative decode with reasoning open (192 tokens, medians of 3 interleaved runs with 2-minute cool-downs): 87.2 tok/s on a step-by-step arithmetic prompt and 86.0 on a code prompt, against 68.2 and 64.5 with the 8-way wide kernels. The GPU is busy ~90% of a round; the rest is the CPU round trip between the draft and verify submissions.

Long context on the M4 Pro (needle-in-a-haystack over real Rust sources, needle at the halfway point, `evals/niah.py`): 16K prompt 3,039 tok/s prefill and 190 tok/s decode, 64K prompt 1,648 tok/s and 110 tok/s, 131K prompt 909 tok/s and 111 tok/s — the passphrase is retrieved exactly at every size up to 131K. At 254K tokens (the 262K window's edge) the answer degrades to the numeric half of the passphrase, and a 270K prompt is rejected with a context-window error.

## Quality

Standard harnesses against the shipped artifact on the M4 Pro (`evals/`; see `evals/README.md`):

| Task | Metric | INT4 (shipped) | BF16 |
| --- | --- | --- | --- |
| wikitext | word perplexity | 22.26 | 21.17 |
| arc_easy (0-shot) | acc / acc_norm | 0.652 / 0.588 | 0.666 / 0.600 |
| gsm8k (5-shot) | flexible / strict | 0.594 / 0.556 | 0.656 / 0.640 |
| mmlu (5-shot) | acc | 0.557 | — |
| HumanEval+ / MBPP+ | pass@1 | 0.384 / 0.405 | — |
| BFCL single turn (`fc`) | overall acc | 19.54% | 16.14% |

BFCL single turn, the shipped AWQ-seeded QAT (`int4-qat-awq`) against the previous build (same binary, same machine):
overall 19.54% vs 17.41%, non-live AST 59.60% vs 62.90%, non-live parallel-multiple AST 60.50% vs 55.00%,
live 55.81% vs 60.10%, irrelevance detection 79.96% vs 51.09%.

## Quick Start

```bash
# Run unit & GPU parity tests
cargo test --offline

# Fuse the converter outputs into the raw artifact and swap tokenizer.json for tokenizer.bin
cargo run --release --offline -p nano-ai -- convert --dir models/minicpm-v-4.7-1b

# Run 24-layer Metal + ANE benchmark on Apple Silicon
cargo run --release --offline -p nano-ai -- bench

# Interactive / single-turn multimodal chat
cargo run --release --offline -p nano-ai -- chat --image path/to/image.png "Describe this image."
# Text chat (DSpark + Vision + Tokenizer auto-loaded from model directory)
cargo run --release --offline -p nano-ai -- chat "Explain Gated DeltaNet in two sentences."

# Start local OpenAI-compatible server with zstd-dict compression
cargo run --release --offline -p nano-ai -- serve --host 127.0.0.1 --port 8080
# Fetch the trained Zstd dictionary for zstd-dict client compression/decompression
curl -s http://127.0.0.1:8080/v1/zstd-dict -o http_zstd.dict
```

### Quality harnesses

`evals/` runs the standard benchmarks against a live server (lm-eval, EvalPlus, BFCL, the
tool-call smoke and the long-context NIAH probe); see `evals/README.md` for the setup and the
measured scores.

```bash
evals/baseline.sh                      # the full gate for the shipped artifact
evals/lm_eval.sh 8080 "" wikitext arc_easy gsm8k
evals/variant.sh my-artifact path/to/minicpm-v-4.7-1b-tern4.bin
python3 evals/niah.py 8080 16384,65536
```

### iPhone / iPad

`ios/` holds a small SwiftUI host app (`NanoAI`) that links the engine through `nano-ios` (a static library with `nano_ai_run`, `nano_ai_warm` and `nano_ai_free`). It keeps one engine open across prompts. At launch `nano_ai_warm("")` opens the engine and pins the system turn (restored from the on-disk prompt cache on later launches). While you type, a debounced `nano_ai_warm(text)` prefills the prompt so far, so Run only prefills the last few tokens.

The project targets iOS 27.2 with Swift 6 (whole-module `-O`, LTO, dead-code stripping in Release) and links the engine built for `aarch64-apple-ios` with `target-cpu=apple-a14`. It requests the `increased-memory-limit` entitlement, which personal teams can sign; `extended-virtual-addressing` needs a paid team. Set `DEVELOPMENT_TEAM` in `ios/project.yml`, sign in to Xcode with that team, enable Developer Mode on the device, and trust the developer profile on it (Settings → General → VPN & Device Management).

```bash
cd ios && xcodegen generate && cd ..
UDID=<device id from `xcrun devicectl list devices`>
xcodebuild -project ios/NanoAI.xcodeproj -scheme NanoAI -configuration Release \
  -destination "id=$UDID" -derivedDataPath target/ios-dd -allowProvisioningUpdates build
xcrun devicectl device install app --device $UDID target/ios-dd/Build/Products/Release-iphoneos/NanoAI.app
# Model files live in the app's Documents (Finder file sharing works too)
for f in minicpm-v-4.7-1b-tern4.bin tokenizer.bin; do
  xcrun devicectl device copy to --device $UDID --domain-type appDataContainer \
    --domain-identifier com.debanjanbasu.nanoai --source models/minicpm-v-4.7-1b/$f --destination Documents/$f
done
# Headless run: one JSON report per --prompt (speed, memory, ANE status), then the app exits
xcrun devicectl device process launch --device $UDID --console --terminate-existing \
  com.debanjanbasu.nanoai --prompt "Explain Gated DeltaNet in two sentences." --max-tokens 200
```

## License

The **code** in this repository is dual-licensed under either of:

- [MIT](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

at your option.

The **model weights and tokenizer** in `models/` are converted artifacts of the third-party
[MiniCPM-V 4.7 1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) and
[MiniCPM-V-4.7-1B-DSpark](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-DSpark) models by OpenBMB.
Both upstream models are published under the **Apache License 2.0**, so those terms apply to the
converted weights as well; this repository's dual license covers the code, not the weights.

Everything under `models/` is tracked with [Git LFS](https://git-lfs.com/); run `git lfs install`
before cloning so the large files are fetched correctly.
