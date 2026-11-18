//! `nano-ai` entry point: dispatches to the CLI commands and hosts the tests.

pub mod cli;
pub mod server;

pub const DEFAULT_MODEL_PATH: &str = "models/minicpm-v-4.7-1b/minicpm-v-4.7-1b-tern4.bin";
/// Default `max_tokens` for `/v1/chat/completions`: the full 262K-token context window.
pub const DEFAULT_CHAT_MAX_TOKENS: usize = nano_engine::model::MAX_CONTEXT_TOKENS;

use std::path::PathBuf;
use std::process::ExitCode;

use nano_engine::model::NanoEngine;
use nano_engine::sampler::SamplingParams;

use cli::{run_agents, run_chat, usage};
use server::run_serve;

async fn run_bench(_args: &[String]) -> nano_engine::Result<()> {
    use nano_engine::artifact::write_synthetic_lm_artifact;

    let max_tokens = 64_usize;
    let dir = std::env::temp_dir().join(format!("nano_ai_bench_{}", std::process::id()));
    tokio::fs::create_dir_all(&dir).await?;
    let model_path = dir.join("bench_tern4.bin");
    let mp_clone = model_path.clone();
    let file_bytes =
        tokio::task::spawn_blocking(move || write_synthetic_lm_artifact(&mp_clone, false))
            .await
            .map_err(|e| nano_engine::Error::InvalidArgument(format!("bench init failed: {e}")))??;
    let file_mb = file_bytes as f64 / (1024.0 * 1024.0);

    let out = tokio::task::spawn_blocking(move || -> nano_engine::Result<serde_json::Value> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let mut engine = NanoEngine::open(&model_path)?;
        let prompt: Vec<u32> = (0..128).map(|i| i + 100).collect();
        let params = SamplingParams {
            repetition_penalty: 1.0,
            eos_tokens: vec![999_999],
            ..SamplingParams::default()
        };
        rt.block_on(async {
            engine.set_prefix_cache_enabled(false);
            let _ = engine
                .generate_ids_streaming_async(&prompt[..16], 8, params.clone(), |_| {})
                .await?;
            let plain = engine
                .generate_ids_streaming_async(&prompt, max_tokens, params.clone(), |_| {})
                .await?;
            engine.set_prefix_cache_enabled(true);
            let _ = engine
                .generate_ids_streaming_async(&prompt, 4, params.clone(), |_| {})
                .await?;
            let cached = engine
                .generate_ids_streaming_async(&prompt, 4, params.clone(), |_| {})
                .await?;
            let spec_prompt = vec![
                10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
            ];
            let spec = engine
                .generate_ids_streaming_async(&spec_prompt, max_tokens, params, |_| {})
                .await?;
            Ok(serde_json::json!({
                "device": engine.device_name(),
                "ane": engine.ane_status(),
                "artifact_mb": (file_mb * 100.0).round() / 100.0,
                "plain_128_prefill_tok_per_s": (plain.prefill_tok_per_s * 10.0).round() / 10.0,
                "cached_128_prefill_tok_per_s": (cached.prefill_tok_per_s * 10.0).round() / 10.0,
                "cached_prefix_tokens": cached.cached_prefix_tokens,
                "plain_decode_tok_per_s": (plain.decode_tok_per_s * 10.0).round() / 10.0,
                "speculative_decode_tok_per_s": (spec.decode_tok_per_s * 10.0).round() / 10.0,
                "speculative_accepted": format!("{}/{}", spec.accepted_tokens, spec.drafted_tokens),
            }))
        })
    })
    .await
    .map_err(|e| nano_engine::Error::InvalidArgument(format!("worker join failed: {e}")))??;

    let _ = tokio::fs::remove_dir_all(&dir).await;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

async fn run_convert(args: &[String]) -> nano_engine::Result<()> {
    use nano_engine::artifact::fuse_and_requantize_suite;

    let mut model_dir = PathBuf::from("models/minicpm-v-4.7-1b");
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--dir" && i + 1 < args.len() {
            model_dir = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }

    let summary = tokio::task::spawn_blocking(move || fuse_and_requantize_suite(&model_dir))
        .await
        .map_err(|e| nano_engine::Error::InvalidArgument(format!("convert join: {e}")))??;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        usage();
        return ExitCode::FAILURE;
    };
    match cmd.as_str() {
        "chat" => match run_chat(rest).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        "agents" => match run_agents(rest).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        "serve" => match run_serve(rest).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        "bench" => match run_bench(rest).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        "convert" => match run_convert(rest).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        "--help" | "-h" => {
            usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("error: unknown command {other:?}");
            usage();
            ExitCode::FAILURE
        }
    }
}
