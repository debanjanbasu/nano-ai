//! Probe: ANE vs GPU per-layer FFN timing at 512 rows on the real artifact.
//!
//! Measured (M2): ANE 3.50 ms/layer vs GPU 4.51 ms/layer at 512 rows. A split
//! FFN (ANE first half + GPU second half, concurrent) was implemented and
//! benchmarked NEUTRAL on both M2 and A15 — the ANE's 256-row function runs at
//! roughly half the 512-row efficiency, so concurrent halves ≈ one sequential
//! whole, and the measured prefill distributions overlapped on both devices.
//! The split was reverted; this probe stays so the numbers can be re-checked
//! if the ANE function set (256-row efficiency) or chunk sizes change.
use nano_engine::model::NanoEngine;
use nano_engine::sampler::SamplingParams;
use nano_metal::context::MetalContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "models/minicpm-v-4.7-1b/minicpm-v-4.7-1b-tern4.bin";
    let _ctx = MetalContext::new()?;
    let mut engine = NanoEngine::open(std::path::Path::new(path))?;
    let ids: Vec<u32> = (0..512).map(|i| (i % 30_000) + 1_000).collect();
    let params = SamplingParams { eos_tokens: vec![999_999], ..SamplingParams::default() };
    let _ = engine.generate_ids(&ids, 4, params)?;
    println!("{}", engine.ane_status());
    Ok(())
}
