//! Measure the entropy of the *post-transform* shuffled planes (what a coder would actually
//! see) and the conditional entropy under a row-stride context, per tensor kind.
//! `cargo run -p nano-engine --example plane_entropy -- <artifact>`
#![allow(clippy::expect_used, clippy::unwrap_used)]
use std::collections::BTreeMap;

use nano_engine::artifact::{
    byte_shuffle4_encode, col_delta_pre_encode_in_place, LM_MAGIC, MappedArtifact, TensorDtype,
};
use nano_metal::context::MetalContext;

fn entropy_and_conditional(bytes: &[u8], stride: usize, label: &str) {
    let len = bytes.len();
    if len < stride * 2 {
        return;
    }
    // H(b)
    let mut hist = [0_u64; 256];
    for &b in bytes {
        hist[b as usize] += 1;
    }
    let h0: f64 = hist.iter().filter(|&&c| c > 0).map(|&c| {
        let p = c as f64 / len as f64;
        -p * p.log2()
    }).sum();
    // H(b | above) with both a 16-bucket and a full-256 context.
    let mut joint16 = vec![[0_u64; 256]; 16];
    let mut joint256 = vec![[0_u64; 256]; 256];
    let mut ctx_total16 = [0_u64; 16];
    let mut ctx_total256 = [0_u64; 256];
    let n = len - stride;
    for i in 0..n {
        let above = bytes[i] as usize;
        let cur = bytes[i + stride] as usize;
        joint16[above >> 4][cur] += 1;
        ctx_total16[above >> 4] += 1;
        joint256[above][cur] += 1;
        ctx_total256[above] += 1;
    }
    let conditional = |joint: &[[u64; 256]], totals: &[u64]| {
        let mut h = 0.0_f64;
        for (bucket, &total) in totals.iter().enumerate() {
            if total == 0 {
                continue;
            }
            #[allow(clippy::suboptimal_flops)] // the mul_add form reads worse for an entropy sum
            for &count in &joint[bucket] {
                if count > 0 {
                    let p = count as f64 / total as f64;
                    h += (total as f64 / n as f64) * -p.log2() * p;
                }
            }
        }
        h
    };
    let h16 = conditional(&joint16, &ctx_total16);
    let h256 = conditional(&joint256, &ctx_total256);
    println!(
        "{label}: bytes {len}  H0={h0:.3}  H(b|ctx16)={h16:.3}  H(b|ctx256)={h256:.3}  saving: 16-bucket {:.1}% / 256 {:.1}%",
        (1.0 - h16 / 8.0) * 100.0,
        (1.0 - h256 / 8.0) * 100.0,
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: plane_entropy <artifact>");
    let ctx = MetalContext::new()?;
    let mapped = MappedArtifact::open(&ctx, std::path::Path::new(&path), LM_MAGIC)?;
    #[expect(unsafe_code)]
    let mut canonical = unsafe { mapped.payload_buffer.as_slice::<u8>() }.to_vec();

    // RAW canonical codes (no col_delta transform): the "byte above" is cols/2 bytes back.
    {
        let mut groups: BTreeMap<String, (usize, Vec<(usize, usize)>)> = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        for td in mapped.header.tensors.values() {
            if !seen.insert(td.offset) || td.dtype != TensorDtype::Int4G64 || td.aux_length == 0 {
                continue;
            }
            let [rows, cols] = td.shape;
            let kind = if rows >= 65_536 {
                "big-vocab".into()
            } else if cols == 1024 {
                "lm-shaped".into()
            } else {
                format!("{cols}-col")
            };
            let entry = groups.entry(kind).or_insert((cols / 2, Vec::new()));
            entry.1.push((td.offset, td.length));
        }
        println!("== RAW canonical codes (pre-transform) ==");
        for (kind, (stride, regions)) in &groups {
            let mut raw = Vec::with_capacity(regions.iter().map(|r| r.1).sum::<usize>());
            for &(off, len) in regions {
                raw.extend_from_slice(&canonical[off..off + len]);
            }
            entropy_and_conditional(&raw, *stride, &format!("{kind} [stride={stride}]"));
        }
    }
    col_delta_pre_encode_in_place(&mut canonical, &mapped.header.tensors);
    println!("== post col_delta + shuffle4 planes ==");
    col_delta_pre_encode_in_place(&mut canonical, &mapped.header.tensors);

    // Group tensors by (kind, shape-signature) so each group gets the right stride.
    let mut groups: BTreeMap<String, (usize, Vec<(usize, usize)>)> = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    for td in mapped.header.tensors.values() {
        if !seen.insert(td.offset) || td.dtype != TensorDtype::Int4G64 || td.aux_length == 0 {
            continue;
        }
        let [rows, cols] = td.shape;
        let kind = if rows >= 65_536 {
            "big-vocab(transposed)".into()
        } else if cols == 1024 {
            "lm-shaped".into()
        } else {
            format!("{cols}-col")
        };
        // Plane-domain context stride: original "row above" delta is cols/2 bytes; each plane
        // byte covers 4 original bytes, so the stride in a plane is cols/8. For the transposed
        // big-vocab tensors the transform makes original rows plane-adjacent: stride 1.
        let stride = if rows >= 65_536 { 1 } else { cols / 8 };
        let entry = groups.entry(kind).or_insert((stride, Vec::new()));
        entry.1.push((td.offset, td.length));
    }
    for (kind, (stride, regions)) in &groups {
        // Concatenate this group's shuffled code regions.
        let mut plane_bytes = Vec::with_capacity(regions.iter().map(|r| r.1).sum::<usize>());
        for &(off, len) in regions {
            plane_bytes.extend_from_slice(&byte_shuffle4_encode(
                &canonical[off..off + len],
            ));
        }
        entropy_and_conditional(&plane_bytes, *stride, &format!("{kind} [stride={stride}]"));
    }
    Ok(())
}
