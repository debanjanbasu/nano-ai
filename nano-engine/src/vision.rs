//! Canvas 3D M-RoPE position computation and `SigLIP2` + Window-Attention Merger vision encoding
//! for MiniCPM-V 4.7 1B.

use std::path::Path;
use std::simd::f32x16;

use image::GenericImageView as _;
use nano_metal::batch::CommandBatch;
use nano_metal::buffer::MetalBuffer;
use nano_metal::context::MetalContext;
use nano_metal::projection::{AnyMatrix, ProjectionKernels};
use nano_metal::shaders::ShaderLibrary;
use nano_metal::vision::{Epilogue, Linear, VISION_WIDTH, VisionKernels};

use crate::Error;
use crate::artifact::{MappedArtifact, TensorDtype, VIS_MAGIC};
use crate::ane_ffn::{VisionAneOffload, VisionAneState};
use crate::model::WIDTH;
pub const PATCH_SIZE: usize = 14;
pub const SCALE_RESOLUTION: usize = 448;
pub const SLICE_ALIGN: usize = 56; // 14 * 4

/// Special token IDs pinned in MiniCPM-V 4.7 1B's `config.json`.
pub const IMAGE_TOKEN_ID: u32 = 248_056;
pub const VIDEO_TOKEN_ID: u32 = 248_057;
pub const IMAGE_START_ID: u32 = 248_078;
pub const IMAGE_END_ID: u32 = 248_079;
pub const SLICE_START_ID: u32 = 248_088;
pub const SLICE_END_ID: u32 = 248_089;
pub const IMAGE_ID_START: u32 = 248_090;
pub const IMAGE_ID_END: u32 = 248_091;
pub const NEWLINE_ID: u32 = 198;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DownsampleMode {
    #[default]
    Mode16x,
    Mode4x,
}

impl DownsampleMode {
    #[must_use]
    pub const fn merge_factor(self) -> usize {
        match self {
            Self::Mode16x => 4,
            Self::Mode4x => 2,
        }
    }

    #[must_use]
    pub const fn token_divisor(self) -> usize {
        match self {
            Self::Mode16x => 16,
            Self::Mode4x => 4,
        }
    }
}

/// Python half-to-even rounding (`round()`), matching `MiniCPMV4_7ImageProcessorPil`.
#[must_use]
pub fn round_half_to_even(v: f64) -> i64 {
    let fl = v.floor();
    let diff = v - fl;
    if diff > 0.5 {
        fl as i64 + 1
    } else if diff < 0.5 {
        fl as i64
    } else {
        let lo = fl as i64;
        if lo % 2 == 0 { lo } else { lo + 1 }
    }
}

#[must_use]
fn ensure_divide(length: usize, align: usize) -> usize {
    let rounded = round_half_to_even(length as f64 / align as f64).max(1) as usize;
    rounded * align
}

/// One preprocessed crop (thumbnail or slice) of an image.
#[derive(Clone, Debug)]
pub struct ImageCrop {
    /// Patch grid `(grid_h, grid_w)` in 14x14 patches (each divisible by 4).
    pub grid_h: usize,
    pub grid_w: usize,
    /// Flattened patch pixels `[grid_h * grid_w, 3 * 14 * 14]`, normalized `(x / 255 - 0.5) / 0.5`.
    pub patches: Vec<f32>,
}

impl ImageCrop {
    #[must_use]
    pub const fn num_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }

    #[must_use]
    pub const fn llm_tokens(&self, mode: DownsampleMode) -> usize {
        (self.grid_h * self.grid_w) / mode.token_divisor()
    }
}

/// Preprocessed image ready for vision encoding and Canvas M-RoPE layout.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub overview: ImageCrop,
    pub slices: Vec<ImageCrop>,
    pub slice_rows: usize,
    pub slice_cols: usize,
}

impl PreparedImage {
    /// Decode and preprocess an image from disk or bytes (single-overview for ≤ 448x448, or LLaVA-UHD grid).
    pub fn from_bytes(bytes: &[u8], max_slice_nums: usize) -> crate::Result<Self> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| Error::InvalidArgument(format!("image decode failed: {e}")))?;
        let (orig_w, orig_h) = img.dimensions();
        let (orig_w, orig_h) = (orig_w as usize, orig_h as usize);
        let ratio = (orig_w * orig_h) as f64 / (SCALE_RESOLUTION * SCALE_RESOLUTION) as f64;

        let scale_w = ensure_divide(
            (SCALE_RESOLUTION as f64 * (orig_w as f64 / orig_h as f64).sqrt()).round() as usize,
            SLICE_ALIGN,
        );
        let scale_h = ensure_divide(
            (scale_w as f64 * orig_h as f64 / orig_w as f64).round() as usize,
            SLICE_ALIGN,
        );
        let overview = Self::crop_to_patches(&img, scale_w, scale_h);

        if ratio <= 1.0 || max_slice_nums <= 1 {
            return Ok(Self {
                overview,
                slices: Vec::new(),
                slice_rows: 0,
                slice_cols: 0,
            });
        }

        // LLaVA-UHD grid selection
        let multiple = (ratio.ceil() as usize).min(max_slice_nums);
        let log_ratio = (orig_w as f64 / orig_h as f64).ln();
        let mut best_grid = (1_usize, 1_usize);
        let mut best_err = f64::INFINITY;
        for n in [multiple.saturating_sub(1), multiple, multiple + 1] {
            if n <= 1 || n > max_slice_nums {
                continue;
            }
            for r in 1..=n {
                if n % r == 0 {
                    let c = n / r;
                    let err = (log_ratio - (c as f64 / r as f64).ln()).abs();
                    if err < best_err {
                        best_err = err;
                        best_grid = (r, c);
                    }
                }
            }
        }
        let (rows, cols) = best_grid;
        if rows * cols <= 1 {
            return Ok(Self {
                overview,
                slices: Vec::new(),
                slice_rows: 0,
                slice_cols: 0,
            });
        }
        let refine_w = ensure_divide(orig_w, cols * SLICE_ALIGN);
        let refine_h = ensure_divide(orig_h, rows * SLICE_ALIGN);
        let refined = img.resize_exact(
            refine_w as u32,
            refine_h as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let cell_w = refine_w / cols;
        let cell_h = refine_h / rows;
        let slices: Vec<ImageCrop> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(rows * cols);
            for ry in 0..rows {
                for cx in 0..cols {
                    let sub = refined.crop_imm(
                        (cx * cell_w) as u32,
                        (ry * cell_h) as u32,
                        cell_w as u32,
                        cell_h as u32,
                    );
                    handles.push(s.spawn(move || Self::crop_to_patches(&sub, cell_w, cell_h)));
                }
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| ImageCrop {
                        grid_h: 0,
                        grid_w: 0,
                        patches: Vec::new(),
                    })
                })
                .collect()
        });
        Ok(Self {
            overview,
            slices,
            slice_rows: rows,
            slice_cols: cols,
        })
    }

    fn crop_to_patches(img: &image::DynamicImage, target_w: usize, target_h: usize) -> ImageCrop {
        let resized = img
            .resize_exact(
                target_w as u32,
                target_h as u32,
                image::imageops::FilterType::CatmullRom,
            )
            .to_rgb8();
        let grid_h = target_h / PATCH_SIZE;
        let grid_w = target_w / PATCH_SIZE;
        let raw = resized.as_raw();
        let mut patches = Vec::with_capacity(grid_h * grid_w * 3 * PATCH_SIZE * PATCH_SIZE);
        let threads = std::thread::available_parallelism()
            .map_or(8, usize::from)
            .min(8)
            .min(grid_h.max(1));
        if grid_h >= 8 && threads > 1 {
            let patch_floats = 3 * PATCH_SIZE * PATCH_SIZE;
            let row_floats = grid_w * patch_floats;
            patches.resize(grid_h * row_floats, 0.0);
            let rows_per_chunk = grid_h.div_ceil(threads);
            std::thread::scope(|s| {
                for (chunk_idx, out_chunk) in
                    patches.chunks_mut(rows_per_chunk * row_floats).enumerate()
                {
                    let py_start = chunk_idx * rows_per_chunk;
                    let py_count = out_chunk.len() / row_floats;
                    s.spawn(move || {
                        let scale = f32x16::splat(2.0 / 255.0);
                        let bias = f32x16::splat(-1.0);
                        let mut chan_buf = [[0.0_f32; PATCH_SIZE * PATCH_SIZE]; 3];
                        for local_y in 0..py_count {
                            let py = py_start + local_y;
                            for px in 0..grid_w {
                                for dy in 0..PATCH_SIZE {
                                    let row_base =
                                        ((py * PATCH_SIZE + dy) * target_w + px * PATCH_SIZE) * 3;
                                    for dx in 0..PATCH_SIZE {
                                        let idx = row_base + dx * 3;
                                        let k = dy * PATCH_SIZE + dx;
                                        chan_buf[0][k] = f32::from(raw[idx]);
                                        chan_buf[1][k] = f32::from(raw[idx + 1]);
                                        chan_buf[2][k] = f32::from(raw[idx + 2]);
                                    }
                                }
                                let dst_base = (local_y * grid_w + px) * patch_floats;
                                for (c_idx, ch) in chan_buf.iter_mut().enumerate() {
                                    let mut i = 0;
                                    while i + 16 <= PATCH_SIZE * PATCH_SIZE {
                                        let v = f32x16::from_slice(&ch[i..i + 16]) * scale + bias;
                                        v.copy_to_slice(&mut ch[i..i + 16]);
                                        i += 16;
                                    }
                                    for v in &mut ch[i..] {
                                        *v = v.mul_add(2.0 / 255.0, -1.0);
                                    }
                                    let c_off = dst_base + c_idx * PATCH_SIZE * PATCH_SIZE;
                                    out_chunk[c_off..c_off + PATCH_SIZE * PATCH_SIZE]
                                        .copy_from_slice(ch);
                                }
                            }
                        }
                    });
                }
            });
        } else {
            let scale = f32x16::splat(2.0 / 255.0);
            let bias = f32x16::splat(-1.0);
            let mut chan_buf = [[0.0_f32; PATCH_SIZE * PATCH_SIZE]; 3];
            for py in 0..grid_h {
                for px in 0..grid_w {
                    for dy in 0..PATCH_SIZE {
                        let row_base = ((py * PATCH_SIZE + dy) * target_w + px * PATCH_SIZE) * 3;
                        for dx in 0..PATCH_SIZE {
                            let idx = row_base + dx * 3;
                            let k = dy * PATCH_SIZE + dx;
                            chan_buf[0][k] = f32::from(raw[idx]);
                            chan_buf[1][k] = f32::from(raw[idx + 1]);
                            chan_buf[2][k] = f32::from(raw[idx + 2]);
                        }
                    }
                    for ch in &mut chan_buf {
                        let mut i = 0;
                        while i + 16 <= PATCH_SIZE * PATCH_SIZE {
                            let v = f32x16::from_slice(&ch[i..i + 16]) * scale + bias;
                            v.copy_to_slice(&mut ch[i..i + 16]);
                            i += 16;
                        }
                        for v in &mut ch[i..] {
                            *v = v.mul_add(2.0 / 255.0, -1.0);
                        }
                        patches.extend_from_slice(ch);
                    }
                }
            }
        }
        ImageCrop {
            grid_h,
            grid_w,
            patches,
        }
    }

    /// All `(grid_h, grid_w)` crop sizes in sequence order (overview first, then slices).
    #[must_use]
    pub fn target_sizes(&self) -> Vec<[usize; 2]> {
        let mut out = vec![[self.overview.grid_h, self.overview.grid_w]];
        for s in &self.slices {
            out.push([s.grid_h, s.grid_w]);
        }
        out
    }
}

/// Compute Canvas 3D M-RoPE positions `(Vec<[u32; 4]>, rope_delta)` for a prompt's `input_ids`,
/// matching `MiniCPMV4_7Model::get_rope_index` bit-for-bit.
struct CanvasFrame {
    thumb: (usize, usize, usize),
    slices: Vec<(usize, usize, usize)>,
    end_idx: usize,
}

#[must_use]
pub fn compute_canvas_mrope(
    input_ids: &[u32],
    target_sizes: &[[usize; 2]],
    mode: DownsampleMode,
) -> (Vec<[u32; 4]>, i32) {
    let seq_len = input_ids.len();
    let merge_factor = mode.merge_factor();
    let mut pos3d = vec![[0_u32; 4]; seq_len];
    for (i, p) in pos3d.iter_mut().enumerate() {
        let idx = i as u32;
        *p = [idx, idx, idx, 0];
    }
    if target_sizes.is_empty() {
        return (pos3d, 0);
    }

    let markers = [
        SLICE_START_ID,
        SLICE_END_ID,
        IMAGE_START_ID,
        IMAGE_END_ID,
        NEWLINE_ID,
    ];
    let is_marker = |id: u32| markers.contains(&id);

    // Locate all visual runs (IMAGE_TOKEN_ID or VIDEO_TOKEN_ID)
    let mut runs = Vec::new();
    let mut i = 0;
    while i < seq_len {
        let id = input_ids[i];
        if id == IMAGE_TOKEN_ID || id == VIDEO_TOKEN_ID {
            let start = i;
            while i < seq_len && input_ids[i] == id {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    if runs.len() != target_sizes.len() {
        return (pos3d, 0);
    }

    // Group runs into frames (thumbnail + slices) and adjacent frame groups
    let mut groups: Vec<(usize, usize, Vec<CanvasFrame>)> = Vec::new();
    let mut prev_end = 0_usize;

    for (crop_idx, &(start, end)) in runs.iter().enumerate() {
        let is_slice = start > 0 && input_ids[start - 1] == SLICE_START_ID;
        if is_slice && let Some(last_group) = groups.last_mut()
            && let Some(last_frame) = last_group.2.last_mut()
        {
            last_frame.slices.push((start, end, crop_idx));
            prev_end = end;
            continue;
        }
        let gap = if start > 0 && prev_end <= start - 1 {
            &input_ids[prev_end..start - 1]
        } else {
            &[]
        };
        let adjacent = !groups.is_empty() && gap.iter().copied().all(is_marker);
        let frame = CanvasFrame {
            thumb: (start, end, crop_idx),
            slices: Vec::new(),
            end_idx: end,
        };
        if adjacent {
            if let Some(last_group) = groups.last_mut() {
                last_group.2.push(frame);
            }
        } else {
            groups.push((start.saturating_sub(1), 0, vec![frame]));
        }
        prev_end = end;
    }

    for group in &mut groups {
        for frame in &mut group.2 {
            let last_crop_end = frame.slices.last().map_or(frame.thumb.1, |s| s.1);
            let mut end_idx = last_crop_end;
            while end_idx < seq_len && is_marker(input_ids[end_idx]) {
                end_idx += 1;
            }
            frame.end_idx = end_idx;
        }
        if let Some(last) = group.2.last() {
            group.1 = last.end_idx;
        }
    }

    let linspace_round = |coord: usize, grid: usize, canvas: usize| -> u32 {
        if grid <= 1 || canvas <= 1 {
            return 0;
        }
        let v = (coord as f64) * ((canvas - 1) as f64) / ((grid - 1) as f64);
        round_half_to_even(v).clamp(0, (canvas - 1).cast_signed() as i64) as u32
    };

    let mut current_pos = 0_u32;
    let mut current_idx = 0_usize;

    for (group_start, group_end, frames) in groups {
        if group_start > current_idx {
            for idx in current_idx..group_start {
                pos3d[idx] = [current_pos, current_pos, current_pos, 0];
                current_pos += 1;
            }
        }
        let mut frame_cursor = group_start;
        for frame in frames {
            let (thumb_start, thumb_end, thumb_gi) = frame.thumb;
            let frame_start = thumb_start.saturating_sub(1);
            if frame_start > frame_cursor {
                for idx in frame_cursor..frame_start {
                    pos3d[idx] = [current_pos, current_pos, current_pos, 0];
                    current_pos += 1;
                }
                current_pos += 1;
            }
            let frame_end = frame.end_idx.min(group_end);
            let origin = current_pos;
            let halo = origin.saturating_sub(1);

            let ts_thumb = target_sizes[thumb_gi];
            let thumb_h = ts_thumb[0] / merge_factor;
            let thumb_w = ts_thumb[1] / merge_factor;
            let mut canvas_h = thumb_h;
            let mut canvas_w = thumb_w;
            let mut slice_h = 0_usize;
            let mut slice_w = 0_usize;
            let mut num_cols = 0_usize;

            if !frame.slices.is_empty() {
                num_cols = frame.slices.len();
                for k in 0..frame.slices.len() - 1 {
                    if frame.slices[k + 1].0 - frame.slices[k].1 > 2 {
                        num_cols = k + 1;
                        break;
                    }
                }
                let mut num_rows = frame.slices.len().checked_div(num_cols).unwrap_or(1);
                if num_rows * num_cols != frame.slices.len() {
                    num_rows = 1;
                    num_cols = frame.slices.len();
                }
                let ts_s0 = target_sizes[frame.slices[0].2];
                slice_h = ts_s0[0] / merge_factor;
                slice_w = ts_s0[1] / merge_factor;
                canvas_h = num_rows * slice_h;
                canvas_w = num_cols * slice_w;
            }

            // Base coat
            for p in &mut pos3d[frame_start..frame_end] {
                *p = [origin, origin, origin, 0];
            }
            pos3d[frame_start][1] = halo;
            pos3d[frame_start][2] = halo;

            if thumb_end < frame_end {
                pos3d[thumb_end][1] = origin + canvas_h as u32;
                pos3d[thumb_end][2] = origin + canvas_w as u32;
            }

            // Thumbnail tokens
            for r in 0..thumb_h {
                let hc = linspace_round(r, thumb_h, canvas_h);
                for c in 0..thumb_w {
                    let wc = linspace_round(c, thumb_w, canvas_w);
                    let idx = thumb_start + r * thumb_w + c;
                    if idx < thumb_end {
                        pos3d[idx] = [origin, origin + hc, origin + wc, 0];
                    }
                }
            }

            // Slice tokens & markers
            for (k, &(s_start, s_end, s_gi)) in frame.slices.iter().enumerate() {
                let h_off = (k / num_cols) * slice_h;
                let w_off = (k % num_cols) * slice_w;
                let ts_s = target_sizes[s_gi];
                let sh = ts_s[0] / merge_factor;
                let sw = ts_s[1] / merge_factor;

                let s_marker = s_start.saturating_sub(1);
                if s_marker >= frame_start {
                    pos3d[s_marker][1] = origin + h_off as u32;
                    pos3d[s_marker][2] = origin + w_off as u32;
                }
                if s_end < frame_end {
                    pos3d[s_end][1] = origin + (h_off + sh.saturating_sub(1)) as u32;
                    pos3d[s_end][2] = origin + (w_off + sw.saturating_sub(1)) as u32;
                }
                for r in 0..sh {
                    for c in 0..sw {
                        let idx = s_start + r * sw + c;
                        if idx < s_end {
                            pos3d[idx] = [
                                origin,
                                origin + (h_off + r) as u32,
                                origin + (w_off + c) as u32,
                                0,
                            ];
                        }
                    }
                }
            }

            // Inter-row newline positions
            if !frame.slices.is_empty() {
                for k in 0..frame.slices.len() - 1 {
                    let gap_s = frame.slices[k].1;
                    let gap_e = frame.slices[k + 1].0;
                    if gap_e - gap_s <= 2 {
                        continue;
                    }
                    let boundary_h = ((k / num_cols + 1) * slice_h).saturating_sub(1) as u32;
                    let right_edge_w = (num_cols * slice_w) as u32;
                    for nl_idx in (gap_s + 1)..(gap_e - 1) {
                        pos3d[nl_idx][1] = origin + boundary_h;
                        pos3d[nl_idx][2] = origin + right_edge_w;
                    }
                }
            }

            current_pos = origin + canvas_h.max(canvas_w) as u32 + 1;
            frame_cursor = frame_end;
        }
        if frame_cursor < group_end {
            for idx in frame_cursor..group_end {
                pos3d[idx] = [current_pos, current_pos, current_pos, 0];
                current_pos += 1;
            }
        }
        current_idx = group_end;
    }

    if current_idx < seq_len {
        for idx in current_idx..seq_len {
            pos3d[idx] = [current_pos, current_pos, current_pos, 0];
            current_pos += 1;
        }
    }

    let max_pos = pos3d
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .max()
        .unwrap_or(0)
        .cast_signed();
    let rope_delta = max_pos + 1 - (seq_len as u32).cast_signed();
    (pos3d, rope_delta)
}

pub const VISION_LAYERS: usize = 27;
pub const VISION_FFN: usize = 4304;
pub const VISION_MERGER_HIDDEN: usize = 4608;
pub const VISION_MERGER_FFN: usize = 17_216;
pub const VISION_INSERT_LAYER: usize = 6;
pub const VISION_EPSILON: f32 = 1e-6;
pub const MAX_CROP_PATCHES: usize = 1600;

/// Native Metal execution of MiniCPM-V 4.7 1B's 27-layer `SigLIP2` + Window-Attention Merger vision tower.
pub struct VisionEncoder {
    artifact: MappedArtifact,
    kernels: VisionKernels,
    ane: VisionAneState,
    patch_in: MetalBuffer,
    hidden: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attn_out: MetalBuffer,
    ffn_mid: MetalBuffer,
    window_buf: MetalBuffer,
    dummy_mean: MetalBuffer,
    merger_mid: MetalBuffer,
    out_buf: MetalBuffer,
}

impl VisionEncoder {
    pub fn open(ctx: &MetalContext, shaders: &ShaderLibrary, path: &Path) -> crate::Result<Self> {
        let artifact = MappedArtifact::open(ctx, path, VIS_MAGIC)?;
        Self::from_artifact(ctx, shaders, artifact)
    }

    /// Initialize `VisionEncoder` directly from a shared `MappedArtifact` (for fused 3-in-1 bundles).
    pub fn from_artifact(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        artifact: MappedArtifact,
    ) -> crate::Result<Self> {
        let kernels = VisionKernels::new(shaders)?;
        let dev = ctx.device();
        let max_p = MAX_CROP_PATCHES;
        let vis_w = VISION_WIDTH as usize;
        let patch_in = MetalBuffer::empty(dev, max_p * 3 * PATCH_SIZE * PATCH_SIZE * 4)?;
        let hidden = MetalBuffer::empty(dev, max_p * vis_w * 4)?;
        let normed = MetalBuffer::empty(dev, max_p * VISION_MERGER_HIDDEN * 4)?;
        let q = MetalBuffer::empty(dev, max_p * vis_w * 4)?;
        let k = MetalBuffer::empty(dev, max_p * vis_w * 4)?;
        let v = MetalBuffer::empty(dev, max_p * vis_w * 4)?;
        let attn_out = MetalBuffer::empty(dev, max_p * vis_w * 4)?;
        let ffn_mid = MetalBuffer::empty(dev, max_p * VISION_FFN * 4)?;
        let window_buf = MetalBuffer::empty(dev, max_p * vis_w * 4)?;
        let dummy_mean = MetalBuffer::empty(dev, 16)?;
        let merger_mid = MetalBuffer::empty(dev, (max_p / 4) * VISION_MERGER_FFN * 4)?;
        let out_buf = MetalBuffer::empty(dev, (max_p / 4) * WIDTH * 4)?;
        Ok(Self {
            artifact,
            kernels,
            ane: VisionAneState::start(),
            patch_in,
            hidden,
            normed,
            q,
            k,
            v,
            attn_out,
            ffn_mid,
            window_buf,
            dummy_mean,
            merger_mid,
            out_buf,
        })
    }

    fn take_ane(&mut self, ctx: &MetalContext, shaders: &ShaderLibrary) -> Option<Box<VisionAneOffload>> {
        match std::mem::replace(&mut self.ane, VisionAneState::Busy) {
            VisionAneState::Ready(offload) => Some(offload),
            VisionAneState::Pending(rx) => {
                let built = rx
                    .recv()
                    .map_err(|_| Error::InvalidFormat("Vision ANE compile thread exited".into()))
                    .and_then(|prog| prog)
                    .and_then(|prog| Ok(Box::new(VisionAneOffload::new(ctx, shaders, &prog)?)));
                match built {
                    Ok(offload) => Some(offload),
                    Err(e) => {
                        self.ane = VisionAneState::Off(e.to_string());
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

    fn project_linear(
        &self,
        batch: &mut CommandBatch,
        proj: &ProjectionKernels,
        weight_name: &str,
        bias_name: &str,
        input: &MetalBuffer,
        scratch: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
        epilogue: Epilogue,
    ) -> crate::Result<()> {
        let desc = self.artifact.descriptor(weight_name)?;
        let (bias_buf, bias_off, _) = self.artifact.f32_slice(bias_name)?;
        if desc.dtype == TensorDtype::F32 {
            let (rows, columns) = if desc.shape[0] == 1 && desc.shape[1] == VISION_WIDTH as usize * VISION_FFN {
                (VISION_WIDTH, VISION_FFN as u32)
            } else if desc.shape[0] == 1 && desc.shape[1] == VISION_WIDTH as usize * VISION_MERGER_FFN {
                (VISION_WIDTH, VISION_MERGER_FFN as u32)
            } else {
                (desc.shape[0] as u32, desc.shape[1] as u32)
            };
            self.kernels.gemm(
                batch,
                Linear {
                    f32: true,
                    weights: &self.artifact.payload_buffer,
                    weights_offset: desc.offset,
                    bias: bias_buf,
                    bias_offset: bias_off,
                    rows,
                    columns,
                },
                input,
                output,
                tokens,
                epilogue,
            );
            return Ok(());
        }
        let (mat, _) = self.artifact.matrix(weight_name)?;
        if let AnyMatrix::Int4(m) = mat {
            self.kernels.int4_gemm_epilogue(
                batch,
                m,
                (bias_buf, bias_off),
                input,
                output,
                tokens,
                epilogue,
            );
        } else {
            let target = if epilogue == Epilogue::Accumulate {
                scratch
            } else {
                output
            };
            proj.matmul(batch, mat, input, target, tokens)?;
            self.kernels.bias_epilogue(
                batch,
                target,
                (bias_buf, bias_off),
                output,
                mat.rows(),
                tokens,
                epilogue,
            );
        }
        Ok(())
    }

    /// Encode one `ImageCrop` into `[llm_tokens, 1024]` decoder embeddings.
    pub fn encode_crop(
        &mut self,
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        proj: &ProjectionKernels,
        crop: &ImageCrop,
        mode: DownsampleMode,
    ) -> crate::Result<Vec<f32>> {
        let mut vis_ane = (crop.num_patches() <= 1024).then(|| self.take_ane(ctx, shaders)).flatten();
        if let Some(ane) = vis_ane.take() {
            if let Ok(out) = self.encode_crop_inner(ctx, proj, crop, mode, Some(&ane)) {
                self.ane = VisionAneState::Ready(ane);
                return Ok(out);
            }
            self.ane = VisionAneState::Off("Vision ANE check failed; falling back to GPU".into());
        }
        self.encode_crop_inner(ctx, proj, crop, mode, None)
    }

    /// Asynchronously encode one `ImageCrop` into `[llm_tokens, 1024]` decoder embeddings,
    /// awaiting GPU completion through Metal 4 commit feedback.
    pub async fn encode_crop_async(
        &mut self,
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        proj: &ProjectionKernels,
        crop: &ImageCrop,
        mode: DownsampleMode,
    ) -> crate::Result<Vec<f32>> {
        let mut vis_ane = (crop.num_patches() <= 1024).then(|| self.take_ane(ctx, shaders)).flatten();
        if let Some(ane) = vis_ane.take() {
            let (llm_tokens, fut) = {
                let (llm_tokens, batch) =
                    self.build_crop_batch(ctx, proj, crop, mode, Some(&ane))?;
                (llm_tokens, batch.commit_and_await())
            };
            let res = fut.await;
            if res.is_ok() && ane.finish_and_check() {
                self.ane = VisionAneState::Ready(ane);
                return self.read_crop_output(llm_tokens);
            }
            self.ane = VisionAneState::Off("Vision ANE check failed; falling back to GPU".into());
        }
        let (llm_tokens, fut) = {
            let (llm_tokens, batch) = self.build_crop_batch(ctx, proj, crop, mode, None)?;
            (llm_tokens, batch.commit_and_await())
        };
        fut.await?;
        self.read_crop_output(llm_tokens)
    }

    fn encode_crop_inner(
        &mut self,
        ctx: &MetalContext,
        proj: &ProjectionKernels,
        crop: &ImageCrop,
        mode: DownsampleMode,
        vis_ane: Option<&VisionAneOffload>,
    ) -> crate::Result<Vec<f32>> {
        let (llm_tokens, batch) = self.build_crop_batch(ctx, proj, crop, mode, vis_ane)?;
        batch.commit_and_wait()?;
        if let Some(ane) = vis_ane
            && !ane.finish_and_check()
        {
            return Err(Error::InvalidFormat("Vision ANE step failed".into()));
        }
        self.read_crop_output(llm_tokens)
    }

    #[expect(unsafe_code)]
    fn read_crop_output(&self, llm_tokens: usize) -> crate::Result<Vec<f32>> {
        let out = unsafe { &self.out_buf.as_slice::<f32>()[..llm_tokens * WIDTH] };
        if out.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidFormat(
                "VisionEncoder produced non-finite embeddings".into(),
            ));
        }
        Ok(out.to_vec())
    }

    #[expect(unsafe_code)]
    fn build_crop_batch(
        &mut self,
        ctx: &MetalContext,
        proj: &ProjectionKernels,
        crop: &ImageCrop,
        mode: DownsampleMode,
        vis_ane: Option<&VisionAneOffload>,
    ) -> crate::Result<(usize, CommandBatch)> {
        let mut cur_h = crop.grid_h;
        let mut cur_w = crop.grid_w;
        let mut count = cur_h * cur_w;
        if count > MAX_CROP_PATCHES {
            return Err(Error::InvalidArgument("crop exceeds MAX_CROP_PATCHES".into()));
        }
        // 1. Patch embedding + GPU 70x70 BF16 position embedding + 27 SigLIP2 layers in ONE CommandBatch!
        unsafe {
            let dst = &mut self.patch_in.as_mut_slice::<f32>()[..crop.patches.len()];
            dst.copy_from_slice(&crop.patches);
        }
        let (patch_w, patch_w_off, _) = self
            .artifact
            .f32_slice("model.vision_tower.embeddings.patch_embedding.weight")?;
        let (patch_b, patch_b_off, _) = self
            .artifact
            .f32_slice("model.vision_tower.embeddings.patch_embedding.bias")?;
        let pos_desc = self
            .artifact
            .descriptor("model.vision_tower.embeddings.position_embedding.weight")?;
        let mut batch = CommandBatch::new_concurrent(ctx)?;
        self.kernels.gemm(
            &mut batch,
            Linear {
                f32: true,
                weights: patch_w,
                weights_offset: patch_w_off,
                bias: patch_b,
                bias_offset: patch_b_off,
                rows: VISION_WIDTH,
                columns: (3 * PATCH_SIZE * PATCH_SIZE) as u32,
            },
            &self.patch_in,
            &self.hidden,
            count as u32,
            Epilogue::Store,
        );
        self.kernels.add_positions_bf16(
            &mut batch,
            &self.hidden,
            (&self.artifact.payload_buffer, pos_desc.offset),
            cur_h as u32,
            cur_w as u32,
        );
        if let Some(ane) = vis_ane {
            ane.reset_status();
            let _ = ane.stage_layer(&mut batch, &self.artifact, 0, 0)?;
        }
        for layer in 0..VISION_LAYERS {
            let p = format!("model.vision_tower.encoder.layers.{layer}");
            let (ln1_w, ln1_w_off, _) = self.artifact.f32_slice(&format!("{p}.layer_norm1.weight"))?;
            let (ln1_b, ln1_b_off, _) = self.artifact.f32_slice(&format!("{p}.layer_norm1.bias"))?;
            self.kernels.layernorm(
                &mut batch,
                &self.hidden,
                (ln1_w, ln1_w_off),
                (ln1_b, ln1_b_off),
                &self.normed,
                VISION_WIDTH,
                count as u32,
                VISION_EPSILON,
            );
            let (qm, _) = self.artifact.matrix(&format!("{p}.self_attn.q_proj.weight"))?;
            let (km, _) = self.artifact.matrix(&format!("{p}.self_attn.k_proj.weight"))?;
            let (vm_mat, _) = self.artifact.matrix(&format!("{p}.self_attn.v_proj.weight"))?;
            if let (AnyMatrix::Int4(q4), AnyMatrix::Int4(k4), AnyMatrix::Int4(v4)) = (qm, km, vm_mat) {
                let (bq, bq_off, _) = self.artifact.f32_slice(&format!("{p}.self_attn.q_proj.bias"))?;
                let (bk, bk_off, _) = self.artifact.f32_slice(&format!("{p}.self_attn.k_proj.bias"))?;
                let (bv, bv_off, _) = self.artifact.f32_slice(&format!("{p}.self_attn.v_proj.bias"))?;
                self.kernels.int4_qkv_bias(
                    &mut batch,
                    q4,
                    (bq, bq_off),
                    k4,
                    (bk, bk_off),
                    v4,
                    (bv, bv_off),
                    &self.normed,
                    &self.q,
                    &self.k,
                    &self.v,
                    count as u32,
                );
            } else {
                // Not `independent`: each bias epilogue must see its own matmul's output.
                for (name, out) in [("q", &self.q), ("k", &self.k), ("v", &self.v)] {
                    self.project_linear(
                        &mut batch,
                        proj,
                        &format!("{p}.self_attn.{name}_proj.weight"),
                        &format!("{p}.self_attn.{name}_proj.bias"),
                        &self.normed,
                        out,
                        out,
                        count as u32,
                        Epilogue::Store,
                    )?;
                }
            }
            self.kernels.attention(
                &mut batch,
                &self.q,
                &self.k,
                &self.v,
                &self.attn_out,
                count as u32,
                0,
            );
            self.project_linear(
                &mut batch,
                proj,
                &format!("{p}.self_attn.out_proj.weight"),
                &format!("{p}.self_attn.out_proj.bias"),
                &self.attn_out,
                &self.q,
                &self.hidden,
                count as u32,
                Epilogue::Accumulate,
            )?;


            let (ln2_w, ln2_w_off, _) = self.artifact.f32_slice(&format!("{p}.layer_norm2.weight"))?;
            let (ln2_b, ln2_b_off, _) = self.artifact.f32_slice(&format!("{p}.layer_norm2.bias"))?;
            self.kernels.layernorm(
                &mut batch,
                &self.hidden,
                (ln2_w, ln2_w_off),
                (ln2_b, ln2_b_off),
                &self.normed,
                VISION_WIDTH,
                count as u32,
                VISION_EPSILON,
            );
            let ran_mlp_on_ane = if let Some(ane) = vis_ane {
                let next_l = (layer + 1 < VISION_LAYERS).then_some(layer + 1);
                ane.encode_layer_mlp(
                    &mut batch,
                    &self.artifact,
                    layer,
                    next_l,
                    &self.normed,
                    &self.hidden,
                    count as u32,
                )?
            } else {
                false
            };
            if !ran_mlp_on_ane {
                self.project_linear(
                    &mut batch,
                    proj,
                    &format!("{p}.mlp.fc1.weight"),
                    &format!("{p}.mlp.fc1.bias"),
                    &self.normed,
                    &self.ffn_mid,
                    &self.ffn_mid,
                    count as u32,
                    Epilogue::GeluTanh,
                )?;
                self.project_linear(
                    &mut batch,
                    proj,
                    &format!("{p}.mlp.fc2.weight"),
                    &format!("{p}.mlp.fc2.bias"),
                    &self.ffn_mid,
                    &self.q,
                    &self.hidden,
                    count as u32,
                    Epilogue::Accumulate,
                )?;
                if batch.dispatch_count() >= 64 {
                    batch.submit_and_renew()?;
                }
            }

            // Intra-ViT 2x2 Window-Attention Merger after layer 6 in 16x mode
            if mode == DownsampleMode::Mode16x && layer == VISION_INSERT_LAYER {
                let vm = "model.vision_tower.vit_merger";
                // Gather 2x2 windows of residual `hidden` into `window_buf`
                self.kernels.window_gather(
                    &mut batch,
                    &self.hidden,
                    &self.window_buf,
                    &self.dummy_mean,
                    cur_h as u32,
                    cur_w as u32,
                );
                let (vln_w, vln_w_off, _) = self.artifact.f32_slice(&format!("{vm}.layer_norm1.weight"))?;
                let (vln_b, vln_b_off, _) = self.artifact.f32_slice(&format!("{vm}.layer_norm1.bias"))?;
                self.kernels.layernorm(
                    &mut batch,
                    &self.window_buf,
                    (vln_w, vln_w_off),
                    (vln_b, vln_b_off),
                    &self.normed,
                    VISION_WIDTH,
                    count as u32,
                    VISION_EPSILON,
                );
                let (qm, _) = self.artifact.matrix(&format!("{vm}.self_attn.q_proj.weight"))?;
                let (km, _) = self.artifact.matrix(&format!("{vm}.self_attn.k_proj.weight"))?;
                let (vm_mat, _) = self.artifact.matrix(&format!("{vm}.self_attn.v_proj.weight"))?;
                if let (AnyMatrix::Int4(q4), AnyMatrix::Int4(k4), AnyMatrix::Int4(v4)) = (qm, km, vm_mat) {
                    let (bq, bq_off, _) = self.artifact.f32_slice(&format!("{vm}.self_attn.q_proj.bias"))?;
                    let (bk, bk_off, _) = self.artifact.f32_slice(&format!("{vm}.self_attn.k_proj.bias"))?;
                    let (bv, bv_off, _) = self.artifact.f32_slice(&format!("{vm}.self_attn.v_proj.bias"))?;
                    self.kernels.int4_qkv_bias(
                        &mut batch,
                        q4,
                        (bq, bq_off),
                        k4,
                        (bk, bk_off),
                        v4,
                        (bv, bv_off),
                        &self.normed,
                        &self.q,
                        &self.k,
                        &self.v,
                        count as u32,
                    );
                } else {
                    // Not `independent`: each bias epilogue must see its own matmul's output.
                    for (name, out) in [("q", &self.q), ("k", &self.k), ("v", &self.v)] {
                        self.project_linear(
                            &mut batch,
                            proj,
                            &format!("{vm}.self_attn.{name}_proj.weight"),
                            &format!("{vm}.self_attn.{name}_proj.bias"),
                            &self.normed,
                            out,
                            out,
                            count as u32,
                            Epilogue::Store,
                        )?;
                    }
                }
                // Window size = 4 (each 2x2 patch window attends only to itself)
                self.kernels.attention(
                    &mut batch,
                    &self.q,
                    &self.k,
                    &self.v,
                    &self.attn_out,
                    count as u32,
                    4,
                );
                self.project_linear(
                    &mut batch,
                    proj,
                    &format!("{vm}.self_attn.out_proj.weight"),
                    &format!("{vm}.self_attn.out_proj.bias"),
                    &self.attn_out,
                    &self.q,
                    &self.window_buf,
                    count as u32,
                    Epilogue::Accumulate,
                )?;

                // Compute mean of updated 2x2 windows (post-window-attention) directly into `self.hidden`!
                cur_h /= 2;
                cur_w /= 2;
                count = cur_h * cur_w;
                self.kernels.window_mean(
                    &mut batch,
                    &self.window_buf,
                    &self.hidden,
                    count as u32,
                );
                let (pn_w, pn_w_off, _) = self.artifact.f32_slice(&format!("{vm}.pre_norm.weight"))?;
                let (pn_b, pn_b_off, _) = self.artifact.f32_slice(&format!("{vm}.pre_norm.bias"))?;
                self.kernels.layernorm(
                    &mut batch,
                    &self.window_buf,
                    (pn_w, pn_w_off),
                    (pn_b, pn_b_off),
                    &self.normed,
                    VISION_MERGER_HIDDEN as u32,
                    count as u32,
                    VISION_EPSILON,
                );
                self.project_linear(
                    &mut batch,
                    proj,
                    &format!("{vm}.linear_1.weight"),
                    &format!("{vm}.linear_1.bias"),
                    &self.normed,
                    &self.merger_mid,
                    &self.merger_mid,
                    count as u32,
                    Epilogue::GeluTanh,
                )?;
                self.project_linear(
                    &mut batch,
                    proj,
                    &format!("{vm}.linear_2.weight"),
                    &format!("{vm}.linear_2.bias"),
                    &self.merger_mid,
                    &self.q,
                    &self.hidden,
                    count as u32,
                    Epilogue::Accumulate,
                )?;
                if vis_ane.is_none() {
                    batch.submit_and_renew()?;
                }
            }
        }

        // 3. Post-LayerNorm + Final 2x2 DownsampleMLP Merger -> [count / 4, 1024]
        let (pln_w, pln_w_off, _) = self
            .artifact
            .f32_slice("model.vision_tower.post_layernorm.weight")?;
        let (pln_b, pln_b_off, _) = self
            .artifact
            .f32_slice("model.vision_tower.post_layernorm.bias")?;
        self.kernels.layernorm(
            &mut batch,
            &self.hidden,
            (pln_w, pln_w_off),
            (pln_b, pln_b_off),
            &self.q,
            VISION_WIDTH,
            count as u32,
            VISION_EPSILON,
        );
        self.kernels.window_gather(
            &mut batch,
            &self.q,
            &self.window_buf,
            &self.dummy_mean,
            cur_h as u32,
            cur_w as u32,
        );
        let llm_tokens = count / 4;
        let mg = "model.merger.mlp.0";
        let (mpn_w, mpn_w_off, _) = self.artifact.f32_slice(&format!("{mg}.pre_norm.weight"))?;
        let (mpn_b, mpn_b_off, _) = self.artifact.f32_slice(&format!("{mg}.pre_norm.bias"))?;
        self.kernels.layernorm(
            &mut batch,
            &self.window_buf,
            (mpn_w, mpn_w_off),
            (mpn_b, mpn_b_off),
            &self.normed,
            VISION_MERGER_HIDDEN as u32,
            llm_tokens as u32,
            VISION_EPSILON,
        );
        self.project_linear(
            &mut batch,
            proj,
            &format!("{mg}.linear_1.weight"),
            &format!("{mg}.linear_1.bias"),
            &self.normed,
            &self.merger_mid,
            &self.merger_mid,
            llm_tokens as u32,
            Epilogue::GeluErf,
        )?;
        self.project_linear(
            &mut batch,
            proj,
            &format!("{mg}.linear_2.weight"),
            &format!("{mg}.linear_2.bias"),
            &self.merger_mid,
            &self.q,
            &self.out_buf,
            llm_tokens as u32,
            Epilogue::Store,
        )?;
        Ok((llm_tokens, batch))
    }
}
