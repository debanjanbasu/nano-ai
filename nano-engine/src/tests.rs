//! Unit and GPU smoke tests for `nano-engine`:
//! 1. Golden Canvas 3D M-RoPE layout tests (matching Hugging Face `GOLDEN_CANVAS_LAYOUTS` and Kaggle reference).
//! 2. `PTQ1_0` ternary L2-optimal + 512-block signed-Hadamard round-trip and Metal kernel agreement.
//! 3. `INT4_G64` group-64 4-bit quantization + Metal kernel agreement (1, 2, 4, 8 rows and fused `SwiGLU`).
//! 4. End-to-end 24-layer MiniCPM-V 4.7 1B Metal execution + N-gram / `DSpark` speculative decoding smoke run.

use nano_metal::batch::CommandBatch;
use nano_metal::buffer::MetalBuffer;
use nano_metal::context::MetalContext;
use nano_metal::ops::{AttentionWorkspace, KvFormat, NanoOps, Rope};
use nano_metal::projection::{
    AnyMatrix, Bf16Matrix, Int4Matrix, ProjectionKernels, Ptq1Matrix, SignedHadamard512,
    decode_ptq1_row,
};
use nano_metal::shaders::ShaderLibrary;

use crate::artifact::{
    LM_MAGIC, MappedArtifact, http_zstd_dict, write_synthetic_lm_artifact,
    zstd_compress_with_dict, zstd_decompress_with_dict,
};
use crate::dspark::{lookup_ngram_draft, schedule_verification_length};
use crate::model::{AsyncEngineHandle, NanoEngine, prompt_cache_key};
use crate::quant::{
    deterministic_signs, fwht512_forward_row, fwht512_inverse_row, quantize_matrix_int4_g64,
    quantize_matrix_ptq1,
};
use crate::sampler::SamplingParams;
use crate::vision::{
    DownsampleMode, IMAGE_END_ID, IMAGE_ID_END, IMAGE_ID_START, IMAGE_START_ID, IMAGE_TOKEN_ID,
    NEWLINE_ID, SLICE_END_ID, SLICE_START_ID, compute_canvas_mrope,
};

/// Switching the KV format must resize the cache buffers to match the new format.
///
/// Each format stores a different number of bytes per token (F16 1024, Q8 544, Q4 288), and the
/// attention kernels write `format.token_bytes()` bytes per cached token. Reusing the buffers
/// from the previous format would write past the end of the allocation.
#[test]
fn kv_cache_buffers_are_resized_when_the_format_changes() {
    if MetalContext::new().is_err() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("nano_ai_kv_format_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let model_path = dir.join("kv_format.bin");
    write_synthetic_lm_artifact(&model_path, true).unwrap();
    let mut engine = NanoEngine::open(&model_path).unwrap();

    for format in [KvFormat::F16, KvFormat::Q8, KvFormat::Q4, KvFormat::F16] {
        engine.set_kv_format(format);
        let (buffers, capacity) = engine.kv_cache_geometry();
        let expected = capacity * format.token_bytes();
        assert!(!buffers.is_empty(), "expected at least one attention layer");
        for (i, (k, v)) in buffers.iter().enumerate() {
            assert_eq!(
                *k, expected,
                "layer {i} K cache holds {k} bytes, but {format:?} needs {expected}"
            );
            assert_eq!(
                *v, expected,
                "layer {i} V cache holds {v} bytes, but {format:?} needs {expected}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The 1- and 2-token fused kernels (Q/K/V concat, Gated `DeltaNet` input projection with BF16
/// `a`/`b`, `SwiGLU`) must agree with the multi-token kernels and a CPU BF16 reference.
#[test]
#[expect(unsafe_code)]
fn small_batch_fused_int4_kernels_match_wide_kernels() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let shaders = ShaderLibrary::new(ctx.device()).unwrap();
    let proj = ProjectionKernels::new(&shaders).unwrap();
    let cols = 1024;
    // Q/K/V-like concat triple, plus an `up` matrix pairing with matrix 1 as `gate`.
    let rows = [64_usize, 32, 16, 32];
    let weights: Vec<(Vec<u8>, Vec<u8>)> = rows
        .iter()
        .enumerate()
        .map(|(m, &r)| {
            let w: Vec<f32> = (0..r * cols)
                .map(|i| ((i as f32).mul_add(0.0371, m as f32)).sin() * 0.3)
                .collect();
            quantize_matrix_int4_g64(&w, r, cols)
        })
        .collect();
    let bufs: Vec<(MetalBuffer, MetalBuffer)> = weights
        .iter()
        .map(|(c, g)| {
            (
                MetalBuffer::from_slice(ctx.device(), c).unwrap(),
                MetalBuffer::from_slice(ctx.device(), g).unwrap(),
            )
        })
        .collect();
    let mats: Vec<Int4Matrix<'_>> = bufs
        .iter()
        .zip(rows)
        .map(|((codes, groups), r)| Int4Matrix {
            codes,
            codes_offset: 0,
            groups,
            groups_offset: 0,
            rows: r as u32,
            columns: cols as u32,
        })
        .collect();
    let bf16_rows = 16;
    let bf16: Vec<Vec<u16>> = (0..2)
        .map(|m| {
            (0..bf16_rows * cols)
                .map(|i| half::bf16::from_f32(((i * (m + 3)) as f32 * 0.013).cos() * 0.1).to_bits())
                .collect()
        })
        .collect();
    let bf16_bufs: Vec<MetalBuffer> = bf16
        .iter()
        .map(|w| MetalBuffer::from_slice(ctx.device(), w).unwrap())
        .collect();
    let bf16_mats: Vec<Bf16Matrix<'_>> = bf16_bufs
        .iter()
        .map(|buffer| Bf16Matrix {
            buffer,
            offset: 0,
            rows: bf16_rows as u32,
            columns: cols as u32,
        })
        .collect();
    let x: Vec<f32> = (0..4 * cols).map(|i| ((i as f32) * 0.71).sin()).collect();
    let x_buf = MetalBuffer::from_slice(ctx.device(), &x).unwrap();
    let out = |n: usize| MetalBuffer::empty(ctx.device(), 4 * n * 4).unwrap();
    let wide: Vec<MetalBuffer> = rows.iter().map(|&r| out(r)).collect();
    let mut batch = CommandBatch::new(&ctx).unwrap();
    for (m, w) in mats.iter().zip(&wide) {
        proj.int4_matmul(&mut batch, *m, &x_buf, w, 4).unwrap();
    }
    batch.commit_and_wait().unwrap();
    let read = |b: &MetalBuffer, n: usize| unsafe { b.as_slice::<f32>()[..n].to_vec() };
    let reference: Vec<Vec<f32>> = wide.iter().zip(rows).map(|(b, r)| read(b, 4 * r)).collect();
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let swiglu_ref: Vec<f32> = reference[1]
        .iter()
        .zip(&reference[3])
        .map(|(&g, &u)| silu(g) * u)
        .collect();
    for tokens in [1_usize, 2] {
        let fused: Vec<MetalBuffer> = rows.iter().map(|&r| out(r)).collect();
        let ab: Vec<MetalBuffer> = (0..2).map(|_| out(bf16_rows)).collect();
        let swiglu = out(rows[1]);
        let mut batch = CommandBatch::new(&ctx).unwrap();
        proj.matmul_swiglu(
            &mut batch,
            AnyMatrix::Int4(mats[1]),
            AnyMatrix::Int4(mats[3]),
            &x_buf,
            &swiglu,
            tokens as u32,
        )
        .unwrap();
        proj.int4_matmul_concat3(
            &mut batch, mats[0], mats[1], mats[2], &x_buf, &fused[0], &fused[1], &fused[2],
            tokens as u32,
        );
        let gdn: Vec<MetalBuffer> = rows[..2].iter().map(|&r| out(r)).collect();
        proj.int4_matmul_gdn(
            &mut batch, mats[0], mats[1], bf16_mats[0], bf16_mats[1], &x_buf, &gdn[0], &gdn[1],
            &ab[0], &ab[1], tokens as u32,
        )
        .unwrap();
        batch.commit_and_wait().unwrap();
        for (m, &r) in rows[..3].iter().enumerate() {
            let got = read(&fused[m], tokens * r);
            for (i, (g, e)) in got.iter().zip(&reference[m]).enumerate() {
                assert!((g - e).abs() < 1e-3, "concat3 matrix {m} token {} row {}: {g} vs {e}", i / r, i % r);
            }
            if m < 2 {
                let got = read(&gdn[m], tokens * r);
                for (g, e) in got.iter().zip(&reference[m]) {
                    assert!((g - e).abs() < 1e-3, "gdn matrix {m}: {g} vs {e}");
                }
            }
        }
        for (i, (g, e)) in read(&swiglu, tokens * rows[1]).iter().zip(&swiglu_ref).enumerate() {
            assert!((g - e).abs() < 1e-3, "swiglu token {} row {}: {g} vs {e}", i / rows[1], i % rows[1]);
        }
        for (m, w) in bf16.iter().enumerate() {
            let got = read(&ab[m], tokens * bf16_rows);
            for t in 0..tokens {
                for r in 0..bf16_rows {
                    let e: f32 = (0..cols)
                        .map(|c| half::bf16::from_bits(w[r * cols + c]).to_f32() * x[t * cols + c])
                        .sum();
                    let g = got[t * bf16_rows + r];
                    assert!((g - e).abs() < 1e-3, "bf16 {m} token {t} row {r}: {g} vs {e}");
                }
            }
        }
    }
}

/// The `SigLIP2` MLP runs `fc1` (1152 -> 4304, GELU) and `fc2` (4304 -> 1152, residual add) through
/// the vision INT4 GEMM when the ANE is unavailable; 4304 is not a multiple of 64 or 128, so the
/// padded last group and partial K chunk must match a CPU dequantized reference.
#[test]
#[expect(unsafe_code)]
fn vision_int4_gemm_matches_cpu_reference() {
    use nano_metal::vision::{Epilogue, VisionKernels};
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let shaders = ShaderLibrary::new(ctx.device()).unwrap();
    let kernels = VisionKernels::new(&shaders).unwrap();
    let tokens = 37_usize;
    for (rows, cols, epilogue) in [(4304_usize, 1152_usize, Epilogue::GeluTanh), (1152, 4304, Epilogue::Accumulate)] {
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.0173).sin() * 0.2).collect();
        let (codes, groups) = quantize_matrix_int4_g64(&w, rows, cols);
        let bias: Vec<f32> = (0..rows).map(|r| (r as f32 * 0.31).cos() * 0.05).collect();
        let x: Vec<f32> = (0..tokens * cols).map(|i| ((i as f32) * 0.71).sin()).collect();
        let residual: Vec<f32> = (0..tokens * rows).map(|i| ((i as f32) * 0.13).cos()).collect();
        let codes_buf = MetalBuffer::from_slice(ctx.device(), &codes).unwrap();
        let groups_buf = MetalBuffer::from_slice(ctx.device(), &groups).unwrap();
        let bias_buf = MetalBuffer::from_slice(ctx.device(), &bias).unwrap();
        let x_buf = MetalBuffer::from_slice(ctx.device(), &x).unwrap();
        let out = MetalBuffer::from_slice(ctx.device(), &residual).unwrap();
        let mat = Int4Matrix {
            codes: &codes_buf,
            codes_offset: 0,
            groups: &groups_buf,
            groups_offset: 0,
            rows: rows as u32,
            columns: cols as u32,
        };
        let mut batch = CommandBatch::new(&ctx).unwrap();
        kernels.int4_gemm_epilogue(&mut batch, mat, (&bias_buf, 0), &x_buf, &out, tokens as u32, epilogue);
        batch.commit_and_wait().unwrap();
        let got = unsafe { out.as_slice::<f32>()[..tokens * rows].to_vec() };

        let row_groups = cols.div_ceil(64);
        let weight = |r: usize, c: usize| {
            let g = (r * row_groups + c / 64) * 4;
            let scale = half::f16::from_le_bytes([groups[g], groups[g + 1]]).to_f32();
            let shift = half::f16::from_le_bytes([groups[g + 2], groups[g + 3]]).to_f32();
            let byte = codes[r * row_groups * 32 + c / 2];
            let q = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            f32::from(q).mul_add(scale, shift)
        };
        let gelu = |v: f32| 0.5 * v * (1.0 + (0.797_884_6 * (0.044_715 * v * v).mul_add(v, v)).tanh());
        let mut max_err = 0.0_f32;
        for t in 0..tokens {
            for r in 0..rows {
                let dot: f32 = (0..cols).map(|c| weight(r, c) * x[t * cols + c]).sum::<f32>() + bias[r];
                let want = match epilogue {
                    Epilogue::GeluTanh => gelu(dot),
                    _ => residual[t * rows + r] + dot,
                };
                max_err = max_err.max((got[t * rows + r] - want).abs() / want.abs().max(1.0));
            }
        }
        assert!(max_err < 1e-3, "{rows}x{cols} {epilogue:?}: max relative error {max_err}");
    }
}

#[test]
fn canvas_mrope_matches_golden_single_and_sliced_layouts() {
    // 1. Single image, no slices (from transformers GOLDEN_CANVAS_LAYOUTS):
    // [bos(1), im_start, 4 visual patches, im_end, eos(2)] with 8x8 patch grid -> 2x2 tokens at 16x
    let ids = [
        1,
        IMAGE_START_ID,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        IMAGE_END_ID,
        2,
    ];
    let (pos, delta) = compute_canvas_mrope(&ids, &[[8, 8]], DownsampleMode::Mode16x);
    let t: Vec<u32> = pos.iter().map(|p| p[0]).collect();
    let h: Vec<u32> = pos.iter().map(|p| p[1]).collect();
    let w: Vec<u32> = pos.iter().map(|p| p[2]).collect();
    assert_eq!(t, vec![0, 1, 1, 1, 1, 1, 1, 4]);
    assert_eq!(h, vec![0, 0, 1, 1, 2, 2, 3, 4]);
    assert_eq!(w, vec![0, 0, 1, 2, 1, 2, 3, 4]);
    assert_eq!(delta, -3);

    // 2. Image with a 2x2 slice grid, rows split by a newline (from GOLDEN_CANVAS_LAYOUTS)
    let sliced_ids = [
        1,
        IMAGE_START_ID,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        IMAGE_END_ID,
        SLICE_START_ID,
        IMAGE_TOKEN_ID,
        SLICE_END_ID,
        SLICE_START_ID,
        IMAGE_TOKEN_ID,
        SLICE_END_ID,
        NEWLINE_ID,
        SLICE_START_ID,
        IMAGE_TOKEN_ID,
        SLICE_END_ID,
        SLICE_START_ID,
        IMAGE_TOKEN_ID,
        SLICE_END_ID,
        2,
    ];
    let (pos_s, delta_s) = compute_canvas_mrope(
        &sliced_ids,
        &[[8, 8], [4, 4], [4, 4], [4, 4], [4, 4]],
        DownsampleMode::Mode16x,
    );
    let ts: Vec<u32> = pos_s.iter().map(|p| p[0]).collect();
    let hs: Vec<u32> = pos_s.iter().map(|p| p[1]).collect();
    let ws: Vec<u32> = pos_s.iter().map(|p| p[2]).collect();
    assert_eq!(
        ts,
        vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4]
    );
    assert_eq!(
        hs,
        vec![0, 0, 1, 1, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4]
    );
    assert_eq!(
        ws,
        vec![0, 0, 1, 2, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 2, 2, 2, 4]
    );
    assert_eq!(delta_s, -16);
}

#[test]
fn canvas_mrope_matches_kaggle_reference_16x_and_4x() {
    // Verify against the exact 448x448 image reference captured from MiniCPM-V-4.7-1B on Kaggle
    // In 16x mode: 32x32 patches -> 8x8 = 64 tokens, rope_delta = -58
    let mut ids_16x = vec![248_045, 846, 198, IMAGE_ID_START, 15, IMAGE_ID_END, IMAGE_START_ID];
    ids_16x.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, 64));
    ids_16x.extend_from_slice(&[
        IMAGE_END_ID, 198, 72240, 279, 20182, 321, 7736, 303, 411, 2099, 13, 248_046, 198,
        248_045, 74455, 198, 248_068, 271, 248_069, 271,
    ]);
    let (pos_16x, delta_16x) = compute_canvas_mrope(&ids_16x, &[[32, 32]], DownsampleMode::Mode16x);
    assert_eq!(delta_16x, -58);
    assert_eq!(pos_16x[6], [6, 5, 5, 0]); // <image> halo
    assert_eq!(pos_16x[7], [6, 6, 6, 0]); // top-left visual token
    assert_eq!(pos_16x[70], [6, 13, 13, 0]); // bottom-right visual token
    assert_eq!(pos_16x[71], [6, 14, 14, 0]); // </image> far corner
    assert_eq!(pos_16x[72], [6, 6, 6, 0]); // trailing \n inside frame group
    assert_eq!(pos_16x[73], [15, 15, 15, 0]); // first text token after image

    // In 4x mode: 32x32 patches -> 16x16 = 256 tokens, rope_delta = -242
    let mut ids_4x = vec![248_045, 846, 198, IMAGE_ID_START, 15, IMAGE_ID_END, IMAGE_START_ID];
    ids_4x.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, 256));
    ids_4x.extend_from_slice(&[
        IMAGE_END_ID, 198, 72240, 279, 20182, 321, 7736, 303, 411, 2099, 13, 248_046, 198,
        248_045, 74455, 198, 248_068, 271, 248_069, 271,
    ]);
    let (pos_4x, delta_4x) = compute_canvas_mrope(&ids_4x, &[[32, 32]], DownsampleMode::Mode4x);
    assert_eq!(delta_4x, -242);
    assert_eq!(pos_4x.last().copied().unwrap(), [40, 40, 40, 0]);
}

#[test]
fn dspark_scheduler_and_ngram_lookup_work() {
    let verify_table = [
        2.2, 2.5, 2.8, 3.1, 3.4, 3.6, 3.8, 4.0, 4.5, 4.9, 5.3, 5.7, 6.1, 6.5, 6.9, 7.3,
    ];
    // High confidence prefix followed by sharp drop
    let confs = [0.95, 0.92, 0.90, 0.88, 0.85, 0.15, 0.05, 0.01];
    let (len, expected) = schedule_verification_length(&confs, 0.9, &verify_table);
    assert!((4..=6).contains(&len), "expected truncate near drop, got {len}");
    assert!(expected > 4.0);

    // N-gram lookup (stops before overlapping the matched suffix at `n - min_match`)
    let history = vec![10, 20, 30, 40, 50, 60, 70, 99, 10, 20, 30, 40];
    let draft = lookup_ngram_draft(&history, 4, 5);
    assert_eq!(draft, vec![50, 60, 70, 99]);
}

#[test]
#[expect(unsafe_code)]
fn metal_ptq1_int4_and_fwht512_match_cpu_reference() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let shaders = ShaderLibrary::new(ctx.device()).unwrap();
    let proj = ProjectionKernels::new(&shaders).unwrap();

    // 1. Test 512-block signed Hadamard forward + inverse roundtrip on GPU vs CPU
    let cols = 1024;
    let signs = deterministic_signs(cols);
    let rot = SignedHadamard512::new(&ctx, &signs).unwrap();
    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin()).collect();
    let mut cpu_rot = x.clone();
    fwht512_forward_row(&mut cpu_rot, &signs);

    let x_buf = MetalBuffer::from_slice(ctx.device(), &x).unwrap();
    let y_buf = MetalBuffer::empty(ctx.device(), cols * 4).unwrap();
    let z_buf = MetalBuffer::empty(ctx.device(), cols * 4).unwrap();
    let mut batch = CommandBatch::new(&ctx).unwrap();
    proj.rotate512(&mut batch, &rot, &x_buf, &y_buf, 1, false).unwrap();
    proj.rotate512(&mut batch, &rot, &y_buf, &z_buf, 1, true).unwrap();
    batch.commit_and_wait().unwrap();

    let gpu_rot = unsafe { y_buf.as_slice::<f32>() };
    let gpu_inv = unsafe { z_buf.as_slice::<f32>() };
    for i in 0..cols {
        assert!(
            (gpu_rot[i] - cpu_rot[i]).abs() < 1e-5,
            "fwht512 mismatch at {i}: {} vs {}",
            gpu_rot[i],
            cpu_rot[i]
        );
        assert!(
            (gpu_inv[i] - x[i]).abs() < 1e-5,
            "fwht512 roundtrip mismatch at {i}"
        );
    }
    let mut cpu_inv = cpu_rot.clone();
    fwht512_inverse_row(&mut cpu_inv, &signs);
    for i in 0..cols {
        assert!((cpu_inv[i] - x[i]).abs() < 1e-5);
    }

    // 2. Test PTQ1_0 matvec and multi-row matmul on GPU vs CPU decoded reference
    let rows = 64;
    let w: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.037).cos() * 0.2)
        .collect();
    let packed = quantize_matrix_ptq1(&w, rows, cols, None);
    let mut decoded_w = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        decode_ptq1_row(&packed[r * 224..(r + 1) * 224], &mut decoded_w[r * cols..(r + 1) * cols])
            .unwrap();
    }
    let w_buf = MetalBuffer::from_slice(ctx.device(), &packed).unwrap();
    let out1_buf = MetalBuffer::empty(ctx.device(), rows * 4).unwrap();
    let ptq1_mat = Ptq1Matrix {
        buffer: &w_buf,
        offset: 0,
        rows: rows as u32,
        columns: cols as u32,
    };
    let mut batch = CommandBatch::new(&ctx).unwrap();
    proj.ptq1_matmul(&mut batch, ptq1_mat, &x_buf, &out1_buf, 1)
        .unwrap();
    batch.commit_and_wait().unwrap();
    let gpu_out1 = unsafe { out1_buf.as_slice::<f32>() };
    for r in 0..rows {
        let mut expected = 0.0_f32;
        for c in 0..cols {
            expected = decoded_w[r * cols + c].mul_add(x[c], expected);
        }
        assert!(
            (gpu_out1[r] - expected).abs() < 1e-2,
            "PTQ1 matvec row {r}: {} vs {expected}",
            gpu_out1[r]
        );
    }
    let out4_ptq1 = MetalBuffer::empty(ctx.device(), 4 * rows * 4).unwrap();
    let mut x4_ptq1 = Vec::with_capacity(4 * cols);
    for t in 0..4 {
        for c in 0..cols {
            x4_ptq1.push(x[c] * (t as f32 + 1.0) * 0.5);
        }
    }
    let x4_ptq1_buf = MetalBuffer::from_slice(ctx.device(), &x4_ptq1).unwrap();
    let mut batch = CommandBatch::new(&ctx).unwrap();
    proj.ptq1_matmul(&mut batch, ptq1_mat, &x4_ptq1_buf, &out4_ptq1, 4)
        .unwrap();
    batch.commit_and_wait().unwrap();
    let gpu_ptq1_4 = unsafe { out4_ptq1.as_slice::<f32>() };
    for r in 0..rows {
        assert!(
            0.5_f32.mul_add(-gpu_out1[r], gpu_ptq1_4[r]).abs() < 1e-2,
            "PTQ1 wide row {r} token 0 mismatch: {} vs {}",
            gpu_ptq1_4[r],
            0.5 * gpu_out1[r]
        );
        assert!(
            (gpu_ptq1_4[rows + r] - gpu_out1[r]).abs() < 1e-2,
            "PTQ1 wide row {r} token 1 mismatch: {} vs {}",
            gpu_ptq1_4[rows + r],
            gpu_out1[r]
        );
    }

    // 3. Test INT4_G64 matvec (1 row) and wide matmul (4 rows) on GPU vs CPU
    let (codes, groups) = quantize_matrix_int4_g64(&w, rows, cols);
    let codes_buf = MetalBuffer::from_slice(ctx.device(), &codes).unwrap();
    let groups_buf = MetalBuffer::from_slice(ctx.device(), &groups).unwrap();
    let int4_mat = Int4Matrix {
        codes: &codes_buf,
        codes_offset: 0,
        groups: &groups_buf,
        groups_offset: 0,
        rows: rows as u32,
        columns: cols as u32,
    };
    let mut x4 = Vec::with_capacity(4 * cols);
    for t in 0..4 {
        for c in 0..cols {
            x4.push(x[c] * (t as f32 + 1.0) * 0.5);
        }
    }
    let x4_buf = MetalBuffer::from_slice(ctx.device(), &x4).unwrap();
    let out_int4_1 = MetalBuffer::empty(ctx.device(), rows * 4).unwrap();
    let out_int4_2 = MetalBuffer::empty(ctx.device(), 2 * rows * 4).unwrap();
    let out_int4_4 = MetalBuffer::empty(ctx.device(), 4 * rows * 4).unwrap();
    let mut batch = CommandBatch::new(&ctx).unwrap();
    proj.int4_matmul(&mut batch, int4_mat, &x4_buf, &out_int4_1, 1)
        .unwrap();
    proj.int4_matmul(&mut batch, int4_mat, &x4_buf, &out_int4_2, 2)
        .unwrap();
    proj.int4_matmul(&mut batch, int4_mat, &x4_buf, &out_int4_4, 4)
        .unwrap();
    batch.commit_and_wait().unwrap();
    let g1 = unsafe { out_int4_1.as_slice::<f32>() };
    let g2 = unsafe { out_int4_2.as_slice::<f32>() };
    let g4 = unsafe { out_int4_4.as_slice::<f32>() };
    for r in 0..rows {
        assert!(
            (g1[r] - g4[r]).abs() < 1e-3,
            "int4 vec vs wide mismatch at row {r}: {} vs {}",
            g1[r],
            g4[r]
        );
        assert!(
            2.0_f32.mul_add(-g1[r], g4[rows + r]).abs() < 1e-2,
            "int4 wide row 1 scaling mismatch"
        );
        // The 2-token scalar kernel must agree with the 8-token-tile kernel on both tokens.
        for t in 0..2 {
            assert!(
                (g2[t * rows + r] - g4[t * rows + r]).abs() < 1e-3,
                "int4 2-token vs wide mismatch at token {t} row {r}: {} vs {}",
                g2[t * rows + r],
                g4[t * rows + r]
            );
        }
    }
}

/// Dequantized `(token, kv_head)` row of a KV cache, as `mc_flash_stage` stages it (half-rounded).
fn kv_row(cache: &[u8], format: KvFormat, token: usize, kv_head: usize) -> Vec<f64> {
    let row = &cache[token * format.token_bytes()..(token + 1) * format.token_bytes()];
    let f16 = |at: usize| f64::from(half::f16::from_le_bytes([row[at], row[at + 1]]).to_f32());
    let round = |x: f32| f64::from(half::f16::from_f32(x).to_f32());
    (0..256)
        .map(|d| match format {
            KvFormat::F16 => f16((kv_head * 256 + d) * 2),
            KvFormat::Q8 => {
                let scale = f16(512 + (kv_head * 8 + d / 32) * 2) as f32;
                round(f32::from(i8::from_ne_bytes([row[kv_head * 256 + d]])) * scale)
            }
            KvFormat::Q4 => {
                let scale = f16(256 + (kv_head * 8 + d / 32) * 2) as f32;
                let byte = row[kv_head * 128 + d / 2];
                let nibble = if d % 2 == 0 { byte & 15 } else { byte >> 4 };
                round((f32::from(nibble) - 8.0) * scale)
            }
        })
        .collect()
}

/// `attention_causal_block` (GQA-packed flash attention + split-K reduce) against an f64 CPU
/// reference over the same quantized cache: decode, verify and prefill widths, short and split
/// prefixes, every KV format.
#[test]
#[expect(unsafe_code)]
fn flash_attention_matches_cpu_reference() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let shaders = ShaderLibrary::new(ctx.device()).unwrap();
    let ops = NanoOps::new(&shaders).unwrap();
    let workspace = AttentionWorkspace::new(&ctx, 512).unwrap();
    let noise = |n: usize, seed: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32).mul_add(0.7193, seed)).sin() * 1.5).collect()
    };
    let norm = MetalBuffer::from_slice(ctx.device(), &vec![1.0_f32; 256]).unwrap();
    for format in [KvFormat::Q8, KvFormat::Q4, KvFormat::F16] {
        for (history, rows) in [(0_usize, 37_usize), (1100, 1), (1100, 8), (1100, 9), (3000, 1), (3000, 8)] {
            let block = history.max(rows);
            let qg = MetalBuffer::from_slice(ctx.device(), &noise(block * 4096, 0.1)).unwrap();
            let key = MetalBuffer::from_slice(ctx.device(), &noise(block * 512, 0.2)).unwrap();
            let value = MetalBuffer::from_slice(ctx.device(), &noise(block * 512, 0.3)).unwrap();
            let q_out = MetalBuffer::empty(ctx.device(), block * 2048 * 4).unwrap();
            let gate_out = MetalBuffer::empty(ctx.device(), block * 2048 * 4).unwrap();
            let k_cache = MetalBuffer::empty(ctx.device(), (history + rows) * format.token_bytes()).unwrap();
            let v_cache = MetalBuffer::empty(ctx.device(), (history + rows) * format.token_bytes()).unwrap();
            let out = MetalBuffer::empty(ctx.device(), rows * 2048 * 4).unwrap();
            let mut batch = CommandBatch::new(&ctx).unwrap();
            for (position, tokens) in [(0, history), (history, rows)] {
                if tokens > 0 {
                    ops.prepare_attention(
                        &mut batch, &qg, &key, &value, (&norm, 0), (&norm, 0), &q_out, &gate_out,
                        &k_cache, &v_cache, position as u32, format, 1e-6, 10_000_000.0,
                        tokens as u32, Rope::TEXT,
                    );
                }
            }
            ops.attention_causal_block(
                &mut batch, &q_out, &k_cache, &v_cache, &gate_out, &out, history as u32,
                rows as u32, format, &workspace,
            );
            batch.commit_and_wait().unwrap();
            let (q, gate) = unsafe { (q_out.as_slice::<f32>(), gate_out.as_slice::<f32>()) };
            let (kc, vc) = unsafe { (k_cache.as_slice::<u8>(), v_cache.as_slice::<u8>()) };
            let got = unsafe { out.as_slice::<f32>() };
            let keys: Vec<[Vec<f64>; 2]> = (0..history + rows)
                .map(|t| [kv_row(kc, format, t, 0), kv_row(kc, format, t, 1)])
                .collect();
            let values: Vec<[Vec<f64>; 2]> = (0..history + rows)
                .map(|t| [kv_row(vc, format, t, 0), kv_row(vc, format, t, 1)])
                .collect();
            let (mut worst, mut scale) = (0.0_f64, 0.0_f64);
            for row in 0..rows {
                for head in 0..8 {
                    let qh = &q[(row * 8 + head) * 256..][..256];
                    let visible = history + row + 1;
                    let scores: Vec<f64> = (0..visible)
                        .map(|t| {
                            let k = &keys[t][head / 4];
                            qh.iter().zip(k).map(|(&a, &b)| f64::from(a) * b).sum::<f64>() / 16.0
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    let mut acc = [0.0_f64; 256];
                    for (t, w) in weights.iter().enumerate() {
                        for (a, v) in acc.iter_mut().zip(&values[t][head / 4]) {
                            *a += w / total * v;
                        }
                    }
                    if format != KvFormat::F16 {
                        let mut h = 1;
                        while h < 256 {
                            for i in (0..256).step_by(2 * h) {
                                for j in i..i + h {
                                    (acc[j], acc[j + h]) = (acc[j] + acc[j + h], acc[j] - acc[j + h]);
                                }
                            }
                            h *= 2;
                        }
                        for a in &mut acc {
                            *a /= 16.0;
                        }
                    }
                    for (d, a) in acc.iter().enumerate() {
                        let index = (row * 8 + head) * 256 + d;
                        let expected = a / (1.0 + (-f64::from(gate[index])).exp());
                        worst = worst.max((f64::from(got[index]) - expected).abs());
                        scale = scale.max(expected.abs());
                    }
                }
            }
            assert!(scale > 1e-3, "{format:?} {history}+{rows}: attention output unexpectedly ~0");
            assert!(
                worst <= 4e-3 * scale,
                "{format:?} {history}+{rows}: max error {worst} vs output scale {scale}"
            );
        }
    }
}

#[test]
#[expect(unsafe_code)]
fn end_to_end_24_layer_engine_smoke_and_speculative_verify() {
    use sha2::Digest as _;
    if MetalContext::new().is_err() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("nano_ai_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let model_path = dir.join("test_mcv47.bin");
    write_synthetic_lm_artifact(&model_path, true).unwrap();
    {
        // The compressed artifact must decode (multi-frame zstd + col_delta inverse) to exactly
        // the payload whose SHA-256 the writer recorded.
        let ctx = MetalContext::new().unwrap();
        let mapped = MappedArtifact::open(&ctx, &model_path, LM_MAGIC).unwrap();
        assert!(mapped.header.zstd_frames.len() > 1, "expected a multi-frame payload");
        let payload = unsafe { mapped.payload_buffer.as_slice::<u8>() };
        let digest = sha2::Sha256::digest(payload)
            .iter()
            .fold(String::new(), |mut hex, b| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{b:02x}");
                hex
            });
        assert_eq!(digest, mapped.header.payload_sha256);
    }
    let mut engine = NanoEngine::open(&model_path).unwrap();
    let history = [1, 2, 3, 11, 22, 33, 44, 99, 1, 2, 3, 11, 22, 33];
    engine.build_dry_penalties(&history, &[33], true, 0);
    let dry_entries = engine.dry_scratch_entries();
    assert!(
        dry_entries.iter().any(|&[tagged, weight]| (tagged & 0x00FF_FFFF) == 44 && weight >= 8),
        "expected DRY penalty on repeated 6-gram continuation token 44, got {dry_entries:?}"
    );
    // Answers, code and tool calls (reasoning closed) must be free to repeat verbatim.
    engine.build_dry_penalties(&history, &[33], false, 0);
    assert_eq!(engine.dry_scratch_entries(), &[] as &[[u32; 2]]);
    // Prompt with repeated pattern to exercise both prefill and N-gram speculative verification
    let prompt = vec![10, 20, 30, 40, 0, 0, 10, 20, 30, 40];
    let params = SamplingParams {
        eos_tokens: vec![999_999],
        ..SamplingParams::default()
    };
    let report = engine.generate_ids(&prompt, 8, params).unwrap();
    assert_eq!(report.generated_tokens, 8);
    assert!(report.prefill_ms > 0.0);
    assert!(report.decode_ms > 0.0);

    // Exercise >= 64-token prefill chunk to trigger ANE FFN offload (or GPU fallback)
    let long_prompt: Vec<u32> = (0..64).map(|i| (i % 16) + 1).collect();
    let params2 = SamplingParams {
        eos_tokens: vec![999_999],
        ..SamplingParams::default()
    };
    let report2 = engine.generate_ids(&long_prompt, 4, params2.clone()).unwrap();
    assert_eq!(report2.generated_tokens, 4);
    assert_eq!(report2.cached_prefix_tokens, 0);
    // Repeat the 64-token prompt to verify PrefixCache hits 63 cached tokens and produces identical token_ids!
    let report2_cached = engine.generate_ids(&long_prompt, 4, params2.clone()).unwrap();
    assert_eq!(report2_cached.cached_prefix_tokens, 63);
    assert_eq!(report2_cached.token_ids, report2.token_ids);
    // Prefill while typing: a prompt that extends the prefilled text only prefills the rest,
    // and generates exactly what a cold prefill does.
    let typed: Vec<u32> = (0..48).map(|i| (i % 13) + 20).collect();
    engine.prefill_prefix(&typed[..30]).unwrap();
    let warmed = engine.generate_ids(&typed, 4, params2.clone()).unwrap();
    assert_eq!(warmed.cached_prefix_tokens, 30);
    engine.set_prefix_cache_enabled(false);
    let cold = engine.generate_ids(&typed, 4, params2).unwrap();
    engine.set_prefix_cache_enabled(true);
    assert_eq!(cold.cached_prefix_tokens, 0);
    assert_eq!(warmed.token_ids, cold.token_ids);

    // Exercise SAW-INT4 Hadamard-256 rotated 4-bit KV cache (`KvFormat::Q4`, 288 B/token)
    engine.set_kv_format(nano_metal::ops::KvFormat::Q4);
    let report_q4 = engine
        .generate_ids(
            &long_prompt,
            4,
            SamplingParams {
                eos_tokens: vec![999_999],
                ..SamplingParams::default()
            },
        )
        .unwrap();
    assert_eq!(report_q4.generated_tokens, 4);
    assert!(report_q4.token_ids.iter().all(|&id| (id as usize) < crate::model::VOCAB));
    engine.set_kv_format(nano_metal::ops::KvFormat::Q8);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let async_rep = rt.block_on(async {
        let handle = AsyncEngineHandle::open(&model_path).await.unwrap();
        handle
            .generate_ids(
                long_prompt.clone(),
                4,
                SamplingParams {
                    eos_tokens: vec![999_999],
                    ..SamplingParams::default()
                },
                None,
            )
            .await
            .unwrap()
    });
    assert_eq!(async_rep.token_ids, report2.token_ids);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Checkpoints persist to `.prompt-cache` and restore in a fresh engine to the same greedy output
/// as a cold prefill and an in-memory restore; corrupt files are ignored and replaced or removed.
#[test]
fn prompt_cache_restores_across_engines_and_drops_corrupt_files() {
    if MetalContext::new().is_err() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("nano_ai_prompt_cache_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("test_mcv47.bin");
    write_synthetic_lm_artifact(&model_path, true).unwrap();
    let prompt: Vec<u32> = (0..200).map(|i| (i * 7 % 31) + 1).collect();
    let params = SamplingParams {
        eos_tokens: vec![999_999],
        ..SamplingParams::default()
    };
    let cache_files = || -> Vec<std::path::PathBuf> {
        std::fs::read_dir(dir.join(".prompt-cache"))
            .into_iter()
            .flatten()
            .flatten()
            .flat_map(|d| std::fs::read_dir(d.path()).into_iter().flatten().flatten())
            .map(|f| f.path())
            .collect()
    };
    let restored = || {
        let mut engine = NanoEngine::open(&model_path).unwrap();
        engine.generate_ids(&prompt, 6, params.clone()).unwrap()
    };

    // Cold prefill, then an in-memory restore; dropping the engine finishes the disk writes.
    let (cold, in_memory) = {
        let mut engine = NanoEngine::open(&model_path).unwrap();
        let cold = engine.generate_ids(&prompt, 6, params.clone()).unwrap();
        (cold, engine.generate_ids(&prompt, 6, params.clone()).unwrap())
    };
    assert_eq!(cold.cached_prefix_tokens, 0);
    assert_eq!(in_memory.cached_prefix_tokens, prompt.len() - 1);
    assert_eq!(in_memory.token_ids, cold.token_ids);
    let files = cache_files();
    assert_eq!(files.len(), 1, "expected one persisted checkpoint, got {files:?}");

    let from_disk = restored();
    assert_eq!(from_disk.cached_prefix_tokens, prompt.len() - 1);
    assert_eq!(from_disk.token_ids, cold.token_ids);

    // Truncated after the index was built: the load fails, prefill runs cold, and the file is
    // rewritten from the fresh checkpoint.
    {
        let mut engine = NanoEngine::open(&model_path).unwrap();
        std::fs::OpenOptions::new().write(true).open(&files[0]).unwrap().set_len(20_000).unwrap();
        let report = engine.generate_ids(&prompt, 6, params.clone()).unwrap();
        assert_eq!(report.cached_prefix_tokens, 0);
        assert_eq!(report.token_ids, cold.token_ids);
    }
    let rewritten = restored();
    assert_eq!(rewritten.cached_prefix_tokens, prompt.len() - 1);
    assert_eq!(rewritten.token_ids, cold.token_ids);

    // A garbled header is dropped while indexing at open.
    std::fs::write(&files[0], b"not a prompt cache file").unwrap();
    let engine = NanoEngine::open(&model_path).unwrap();
    assert_eq!(cache_files(), [] as [std::path::PathBuf; 0]);
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn http_zstd_dictionary_roundtrip_and_compression_gain() {
    let dict = http_zstd_dict();
    assert!(!dict.is_empty(), "expected non-empty trained HTTP Zstd dictionary");
    let sse_chunk = br#"data: {"id":"chatcmpl-nano","object":"chat.completion.chunk","model":"minicpm-v-4.7-1b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
    let compressed = zstd_compress_with_dict(sse_chunk, dict, 3).unwrap();
    let roundtrip = zstd_decompress_with_dict(&compressed, dict, 1 << 20).unwrap();
    assert_eq!(roundtrip, sse_chunk);
    assert!(
        compressed.len() < sse_chunk.len(),
        "expected dictionary-compressed SSE chunk ({}) < raw len ({})",
        compressed.len(),
        sse_chunk.len()
    );
}

#[test]
fn zstd_decompression_bomb_is_rejected_at_limit() {
    use std::io::Write as _;
    // 256 MiB of zeros compresses to a few KiB; decoding must stop at the 16 MiB limit.
    let mut enc = zstd::stream::write::Encoder::new(Vec::new(), 3).unwrap();
    let zeros = vec![0_u8; 1 << 20];
    for _ in 0..256 {
        enc.write_all(&zeros).unwrap();
    }
    let bomb = enc.finish().unwrap();
    assert!(bomb.len() < 64 * 1024, "bomb is {} bytes", bomb.len());
    assert!(zstd_decompress_with_dict(&bomb, &[], 16 << 20).is_err());
}

#[test]
fn prompt_cache_key_requires_hex() {
    assert_eq!(prompt_cache_key("0123456789abcdefABCD"), Some("0123456789abcdef"));
    assert_eq!(prompt_cache_key("../../../tmp/x-q8"), None);
    assert_eq!(prompt_cache_key("/etc/0123456789ab"), None);
    assert_eq!(prompt_cache_key("abc"), None);
}

/// The compact Zstd-compressed `tokenizer.bin` must encode, decode and incrementally stream
/// added tokens, NFC, contractions, Unicode scripts and code losslessly.
#[test]
fn compact_tokenizer_roundtrips_and_matches_golden_ids() {
    let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/minicpm-v-4.7-1b");
    let Ok(Some(tok)) = crate::tokenizer::Tokenizer::load(&model_dir) else {
        return;
    };
    let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let ids = tok.encode(prompt);
    assert_eq!(ids.first().copied(), Some(248_045));
    assert!(ids.contains(&248_046));
    assert_eq!(ids.last().copied(), Some(198));
    assert_eq!(tok.decode(&ids), prompt);

    let samples = [
        "don't 's 're 'll 've 'm 'd 'S 'RE",
        "e\u{301} \u{e9} \u{f1} 日本語 中文 한국어 Привет مرحبا हिन्दी 🙂 👩\u{200d}💻",
        "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>",
        include_str!("../../README.md"),
    ];
    for text in samples {
        let encoded = tok.encode(text);
        let expected: String = unicode_normalization_alignments::UnicodeNormalization::nfc(text)
            .map(|(c, _)| c)
            .collect();
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, expected);
        let mut pending = Vec::new();
        let streamed: String = encoded
            .iter()
            .filter_map(|&id| tok.stream_piece(&mut pending, id))
            .collect();
        let tail = String::from_utf8_lossy(&pending);
        assert_eq!(streamed + &tail, decoded);
    }
}

/// Prefill-sized INT4 GEMMs (dequant + MPP `matmul2d`) must match the wide int4 kernels, which the
/// same calls take inside a concurrent `independent` group.
#[test]
#[expect(unsafe_code)]
fn mpp_int4_gemms_match_wide_kernels() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let shaders = ShaderLibrary::new(ctx.device()).unwrap();
    let proj = ProjectionKernels::new(&shaders).unwrap();
    let cols = 1024;
    // Concat triple (208 rows: a partial 64-row MPP tile) plus `up` pairing with matrix 0 as `gate`.
    let rows = [208_usize, 64, 48, 208];
    let bufs: Vec<(MetalBuffer, MetalBuffer)> = rows
        .iter()
        .enumerate()
        .map(|(m, &r)| {
            let w: Vec<f32> = (0..r * cols)
                .map(|i| ((i as f32).mul_add(0.0173, m as f32)).sin() * 0.3)
                .collect();
            let (codes, groups) = quantize_matrix_int4_g64(&w, r, cols);
            (
                MetalBuffer::from_slice(ctx.device(), &codes).unwrap(),
                MetalBuffer::from_slice(ctx.device(), &groups).unwrap(),
            )
        })
        .collect();
    let mats: Vec<Int4Matrix<'_>> = bufs
        .iter()
        .zip(rows)
        .map(|((codes, groups), r)| Int4Matrix {
            codes,
            codes_offset: 0,
            groups,
            groups_offset: 0,
            rows: r as u32,
            columns: cols as u32,
        })
        .collect();
    let ab: Vec<u16> = (0..16 * cols)
        .map(|i| half::bf16::from_f32((i as f32 * 0.013).cos() * 0.1).to_bits())
        .collect();
    let ab_buf = MetalBuffer::from_slice(ctx.device(), &ab).unwrap();
    let ab_mat = Bf16Matrix { buffer: &ab_buf, offset: 0, rows: 16, columns: cols as u32 };
    for tokens in [300_usize, 512] {
        let x: Vec<f32> = (0..tokens * cols).map(|i| ((i as f32) * 0.71).sin()).collect();
        let x_buf = MetalBuffer::from_slice(ctx.device(), &x).unwrap();
        let t = tokens as u32;
        let run = |wide: bool| -> Vec<Vec<f32>> {
            // plain, concat3 x3, gdn qkv/z, swiglu, gdn a/b
            let out_rows = [rows[0], rows[0], rows[1], rows[2], rows[0], rows[1], rows[0], 16, 16];
            let outs: Vec<MetalBuffer> = out_rows
                .iter()
                .map(|&r| MetalBuffer::empty(ctx.device(), tokens * r * 4).unwrap())
                .collect();
            let encode = |batch: &mut CommandBatch| -> nano_metal::Result<()> {
                proj.int4_matmul(batch, mats[0], &x_buf, &outs[0], t)?;
                proj.int4_matmul_concat3(
                    batch, mats[0], mats[1], mats[2], &x_buf, &outs[1], &outs[2], &outs[3], t,
                );
                proj.int4_matmul_gdn(
                    batch, mats[0], mats[1], ab_mat, ab_mat, &x_buf, &outs[4], &outs[5], &outs[7],
                    &outs[8], t,
                )?;
                proj.matmul_swiglu(
                    batch,
                    AnyMatrix::Int4(mats[0]),
                    AnyMatrix::Int4(mats[3]),
                    &x_buf,
                    &outs[6],
                    t,
                )
            };
            let mut batch = CommandBatch::new_concurrent(&ctx).unwrap();
            if wide {
                batch.independent(encode).unwrap();
            } else {
                encode(&mut batch).unwrap();
            }
            batch.commit_and_wait().unwrap();
            outs.iter().map(|b| unsafe { b.as_slice::<f32>().to_vec() }).collect()
        };
        let (mpp, wide) = (run(false), run(true));
        for (i, (got, want)) in mpp.iter().zip(&wide).enumerate().take(7) {
            let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
            let norm: f32 = want.iter().map(|w| w * w).sum();
            // Half-rounded weights (and the near-cancelling sums of this data) cost ~4e-3 relative
            // RMS; tiling or layout bugs cost O(1).
            let rel = (err / norm).sqrt();
            assert!(rel < 1e-2, "tokens {tokens} output {i}: relative RMS error {rel}");
            assert!(err > 0.0, "tokens {tokens} output {i}: MPP path not taken");
        }
    }
}

/// One-time prefill-width calibration at open must leave the engine in its just-constructed
/// state: the timed synthetic prefill passes wrote KV rows and GDN state, and `reset()` has to
/// unwind all of it before the first real prompt.
#[test]
fn prefill_rows_calibration_keeps_state_clean() {
    if MetalContext::new().is_err() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("nano_ai_prefill_calib_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let model_path = dir.join("prefill_calib.bin");
    write_synthetic_lm_artifact(&model_path, true).unwrap();
    let mut engine = NanoEngine::open(&model_path).unwrap();

    let (position, rope_delta, active_tokens, checkpoints, prefill_rows) =
        engine.calibration_test_state();
    assert_eq!(position, 0, "calibration must not advance the position");
    assert_eq!(active_tokens, 0, "calibration must not leave resident tokens");
    assert_eq!(checkpoints, 0, "calibration must not create prefix checkpoints");
    assert_eq!(rope_delta, 0, "calibration must not set a rope delta");
    assert!(
        [128, 256, 512].contains(&prefill_rows),
        "calibrated prefill_rows {prefill_rows} is not one of 128/256/512"
    );
    // The engine must still work end to end after calibration rewrote the profile.
    let report = engine
        .generate_ids(&[10, 20, 30, 40, 50], 2, SamplingParams::default())
        .unwrap();
    assert_eq!(report.generated_tokens, 2);
}

/// Measures whether a Cauchy-Schwarz `lm_head` gate could skip the full-vocab pass on decode:
/// `logit_row <= ||x||_2 * ||w_row||_2`, and after the final `RMSNorm`
/// `||x||_2 <= sqrt(WIDTH) * ||final_norm||_inf`, both valid bounds. If the decode top-1 logit
/// exceeds `||x||_2_bound * max_cold_row_norm`, the hot-pick is PROVABLY the vocab max and the
/// cold 127 MB pass could be skipped. Run with:
/// `cargo test -p nano-engine lm_head_norm_gate_probe -- --ignored --nocapture`
/// (needs models/minicpm-v-4.7-1b/minicpm-v-4.7-1b-tern4.bin on disk).
#[test]
#[ignore = "measurement probe: needs the real artifact, run with --ignored --nocapture"]
fn lm_head_norm_gate_probe() {
    use crate::model::VOCAB;
    use std::time::Instant;

    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/minicpm-v-4.7-1b/minicpm-v-4.7-1b-tern4.bin");
    if !model_path.is_file() {
        eprintln!("probe skipped: real artifact not found at {model_path:?}");
        return;
    }
    let mut engine = NanoEngine::open(&model_path).unwrap();

    // 1. Per-row L2 norms of the tied INT4_G64 lm_head (dequantized on CPU, once).
    let artifact = engine.artifact_for_probe();
    let lm = artifact.matrix_handle("lm_head.weight").unwrap();
    let AnyMatrix::Int4(m) = artifact.matrix_at(lm).0 else {
        panic!("probe assumes an INT4_G64 lm_head");
    };
    assert_eq!(m.rows as usize, VOCAB);
    let cols = m.columns as usize;
    #[expect(unsafe_code)]
    let codes = unsafe { &m.codes.as_slice::<u8>()[m.codes_offset..] };
    // `groups_offset` is a byte offset; `as_slice::<u32>` indexes 4-byte elements.
    #[expect(unsafe_code)]
    let groups = unsafe { &m.groups.as_slice::<u32>()[m.groups_offset / 4..] };
    let t = Instant::now();
    let mut row_norm_sq = vec![0.0_f32; VOCAB];
    for row in 0..VOCAB {
        let row_codes = &codes[row * cols / 2..(row + 1) * cols / 2];
        let row_groups = &groups[row * (cols / 64)..];
        let mut sq = 0.0_f64;
        for (c, &byte) in row_codes.iter().enumerate() {
            // Each `u32` group entry packs two f16s: (scale, bias) for 64 columns.
            let pair = row_groups[c / 32];
            let s = half::f16::from_bits(pair as u16).to_f32();
            let b = half::f16::from_bits((pair >> 16) as u16).to_f32();
            let lo = f64::from(f32::from(byte & 0x0F));
            let hi = f64::from(f32::from(byte >> 4));
            let slo = f64::from(s);
            let blo = f64::from(b);
            sq = hi.mul_add(hi.mul_add(slo, blo), lo.mul_add(lo.mul_add(slo, blo), sq));
        }
        row_norm_sq[row] = sq as f32;
    }
    let max_full_norm_sq = row_norm_sq.iter().copied().fold(0.0_f32, f32::max);
    println!(
        "row norms in {:.1?}: max ||w_row||_2 = {:.3}",
        t.elapsed(),
        max_full_norm_sq.sqrt()
    );

    // 2. Static ||x||_2 bound: sqrt(WIDTH) * ||final_norm||_inf.
    let (norm_buf, norm_off, norm_len) = artifact.f32_slice("model.language_model.norm.weight").unwrap();
    #[expect(unsafe_code)]
    let norm_w = unsafe { &norm_buf.as_slice::<f32>()[norm_off / 4..norm_off / 4 + norm_len] };
    let norm_inf = norm_w.iter().fold(0.0_f32, |a, &w| a.max(w.abs()));
    let x_norm_bound = (cols as f32).sqrt() * norm_inf;
    println!("||x||_2 bound = sqrt({cols}) * {norm_inf:.3} = {x_norm_bound:.3}");

    // 3. Decode with per-step top-1 logits, counting gate fires against progressively larger
    //    hot sets (hot = most frequent generated ids so far; cold = the rest).
    let prompt: Vec<u32> = crate::quant::deterministic_signs(8)
        .iter()
        .map(|&s| (s.abs() * 1000.0) as u32 % 50_000)
        .collect();
    let mut history = prompt;
    let mut fired_full = 0_usize;
    let mut fired_hot16k = 0_usize;
    let mut fired_hot64k = 0_usize;
    let steps = 32;
    for _ in 0..steps {
        let params = SamplingParams {
            eos_tokens: vec![999_999],
            ..SamplingParams::default()
        };
        let rep = engine.generate_ids(&history, 1, params).unwrap();
        let logits = engine.last_logits();
        let top1 = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let bound = x_norm_bound * max_full_norm_sq.sqrt();
        if top1 >= bound {
            fired_full += 1;
        }
        // Hot-set variants: exclude the K most frequent history ids from the cold set.
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for &id in &history {
            *counts.entry(id).or_default() += 1;
        }
        let mut by_freq: Vec<(u32, usize)> = counts.into_iter().collect();
        by_freq.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        for (hot_k, fired) in [(16_384, &mut fired_hot16k), (65_536, &mut fired_hot64k)] {
            let hot: std::collections::HashSet<u32> =
                by_freq.iter().take(hot_k).map(|&(id, _)| id).collect();
            let max_cold = row_norm_sq
                .iter()
                .enumerate()
                .filter(|(row, _)| !hot.contains(&(*row as u32)))
                .map(|(_, &sq)| sq)
                .fold(0.0_f32, f32::max)
                .sqrt();
            if top1 >= x_norm_bound * max_cold {
                *fired += 1;
            }
        }
        history.push(rep.token_ids[0]);
    }
    println!(
        "gate fires over {steps} decode steps: full-vocab {fired_full}, hot-16k {fired_hot16k}, hot-64k {fired_hot64k}"
    );
    println!(
        "gate bound = {x_norm_bound:.1} * {:.3} = {:.1}; typical top-1 logits are far below it, so the static bound never fires",
        max_full_norm_sq.sqrt(),
        x_norm_bound * max_full_norm_sq.sqrt()
    );
}
