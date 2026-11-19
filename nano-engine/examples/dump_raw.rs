//! Dump the canonical (decompressed, de-shuffled) payload of an MCV47 artifact so offline
//! analysis can measure the true entropy of the weight codes. `cargo run -p nano-engine
//! --example dump_raw -- <artifact> <out.bin>`
use nano_engine::artifact::{LM_MAGIC, MappedArtifact};
use nano_metal::context::MetalContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: dump_raw <artifact> <out>");
    let out = args.next().expect("usage: dump_raw <artifact> <out>");
    let ctx = MetalContext::new()?;
    let mapped = MappedArtifact::open(&ctx, std::path::Path::new(&path), LM_MAGIC)?;
    let payload = unsafe { mapped.payload_buffer.as_slice::<u8>() };
    std::fs::write(&out, payload)?;
    println!("wrote {} canonical bytes to {out}", payload.len());
    Ok(())
}
