//! Apple Neural Engine ("ANE") `SwiGLU` FFN offload for MiniCPM-V 4.7 1B (`WIDTH = 1024`, `FFN = 3584`).
//!
//! - Compiles a MIL program (`program(1.3)`) with functions `ffn256` and `ffn512` in a background
//!   thread at engine startup (`<user cache dir>/micro-ai-minicpm-ffn/<hash>/`).
//! - Stages `INT4_G64` `gate_proj`, `up_proj`, and `down_proj` into two alternating fp16 `IOSurface`
//!   weight sets (their exact dequantized values) while the ANE evaluates the previous layer.
//! - Uses `MTLSharedEvent` GPU->ANE (`ready`) and ANE->GPU (`done`) handoffs so no host thread
//!   blocks mid-layer, and checks layer 0 against the Metal GPU FFN (result and speed) before enabling.
//! - Falls back transparently to the Metal GPU FFN if the private framework is unavailable,
//!   compilation fails, the GPU is faster on this chip, or any evaluation times out / produces
//!   non-finite rows.

use std::sync::{Arc, mpsc};
use std::time::Duration;

use micro_metal::ane::{self, AneKernels, Binding, Handoff, Program, Surface, SurfaceLayout};
use micro_metal::batch::CommandBatch;
use micro_metal::buffer::MetalBuffer;
use micro_metal::context::MetalContext;
use micro_metal::projection::AnyMatrix;
use micro_metal::shaders::ShaderLibrary;

use crate::artifact::MappedArtifact;
use crate::model::{FFN, WIDTH};

pub const ANE_FUNCTION_ROWS: [u32; 2] = [256, 512];
/// Narrowest prefill chunk the ANE takes; narrower chunks stay on the GPU.
///
/// The ANE's fixed per-call cost and the serial command encoding it needs lost to the GPU end
/// to end at 128 rows on an iPhone SE 3 (252 vs 328 tok/s) and won from 256 rows up (346–419 vs
/// ~290 tok/s).
pub const ANE_MIN_ROWS: usize = 256;
pub const ANE_MAX_ROWS: usize = 512;
pub const VIS_ANE_ROWS: [u32; 2] = [256, 1024];
const VIS_CACHE_TAG: &str = "micro-ai-minicpm-vision-v2";
const CACHE_TAG: &str = "micro-ai-minicpm-ffn";
const COMPILE_LIMIT: Duration = Duration::from_secs(120);
const JOB_BOUND: Duration = Duration::from_secs(2);
const CHECK_ROWS: u32 = 128;
/// fp16 keeps the ANE FFN within ~0.2% of the GPU's; the old `W8A8` path's ~4% would fail.
const CHECK_TOLERANCE: f64 = 0.01;

const X_LAYOUT: SurfaceLayout = SurfaceLayout::new(WIDTH as u32, ANE_MAX_ROWS as u32, true);
const Y_LAYOUT: SurfaceLayout = SurfaceLayout::new(WIDTH as u32, ANE_MAX_ROWS as u32, true);
const GATE_LAYOUT: SurfaceLayout = SurfaceLayout::new(FFN as u32, WIDTH as u32, true);
const DOWN_LAYOUT: SurfaceLayout = SurfaceLayout::new(WIDTH as u32, FFN as u32, true);
/// The ANE's `down_proj` matmul adds a fixed ~4e-4 rms absolute error at its output, ~1.5% of this
/// FFN's typical output (rms ~0.02); scaling its input by 32 (and the join by 1/32) leaves fp16
/// rounding (~0.1%) as the only error, with |y| up to 2047 before fp16 overflows.
const DOWN_INPUT_SCALE: f64 = 32.0;

fn hex(value: f64) -> String {
    let bits = value.to_bits();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff).cast_signed() - 1023;
    let fraction = format!("{:013x}", bits & ((1 << 52) - 1));
    let fraction = fraction.trim_end_matches('0');
    if fraction.is_empty() {
        format!("{sign}0x1p{exponent:+}")
    } else {
        format!("{sign}0x1.{fraction}p{exponent:+}")
    }
}

fn fp16(value: f64) -> String {
    format!("fp16({})", hex(value))
}

struct MilFunction {
    rows: u32,
    params: Vec<String>,
    body: String,
}

impl MilFunction {
    fn line(&mut self, text: &str) {
        self.body.push_str("        ");
        self.body.push_str(text);
        self.body.push_str(";\n");
    }

    fn tensor(ty: &str, rows: u32, width: u32) -> String {
        format!("tensor<{ty}, [1, 1, {rows}, {width}]>")
    }

    fn half(&mut self, name: &str, rows: u32, width: u32, expression: &str) {
        let tensor = Self::tensor("fp16", rows, width);
        self.line(&format!("{tensor} {name} = {expression}"));
    }

    fn input(&mut self, name: &str, layout: SurfaceLayout, rows: u32, width: u32) {
        self.params
            .push(format!("{} {name}", layout.buffer_type(rows, width)));
        let tensor = Self::tensor(layout.mil_type(), rows, width);
        self.line(&format!(
            "{tensor} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})"
        ));
    }

    fn dequantize(&mut self, name: &str, rows: u32, width: u32) {
        let unit = fp16(1.0 / 128.0);
        self.half(
            &format!("{name}_d"),
            rows,
            width,
            &format!("dequantize(input = {name}_t, scale = {unit})"),
        );
    }

    fn matmul(&mut self, name: &str, weights: &str, input: &str, rows: u32) {
        let tensor = Self::tensor("fp16", rows, self.rows);
        self.line(&format!(
            "{tensor} {name} = matmul(transpose_x = bool(false), transpose_y = bool(false), x = {weights}, y = {input})"
        ));
    }
}

/// Generate MIL function `ffn{rows}<ios18>` for MiniCPM-V 4.7 1B's 1024 -> 3584 -> 1024 `SwiGLU` FFN.
/// Everything stays fp16: int8 activations, per-row int8 weights and an int8 `SwiGLU` intermediate
/// (`W8A8`) cost ~4% relative error per layer and +0.7% wikitext perplexity.
fn ffn_mil_function(rows: u32) -> String {
    let (width, ffn) = (WIDTH as u32, FFN as u32);
    let mut f = MilFunction {
        rows,
        params: Vec::new(),
        body: String::new(),
    };
    f.input("x0", X_LAYOUT, width, rows);
    f.input("wg", GATE_LAYOUT, ffn, width);
    f.input("wu", GATE_LAYOUT, ffn, width);
    f.input("wd", DOWN_LAYOUT, width, ffn);
    f.matmul("gm", "wg_t", "x0_t", ffn);
    f.matmul("um", "wu_t", "x0_t", ffn);

    // SiLU(g) = (g / 2) * (1 + tanh(g / 2))
    f.half("gh", ffn, rows, &format!("mul(x = gm, y = {})", fp16(0.5)));
    f.half("th", ffn, rows, "tanh(x = gh)");
    f.half("tp", ffn, rows, &format!("add(x = th, y = {})", fp16(1.0)));
    f.half("si", ffn, rows, "mul(x = gh, y = tp)");
    f.half("h", ffn, rows, "mul(x = si, y = um)");
    f.half("hs", ffn, rows, &format!("mul(x = h, y = {})", fp16(DOWN_INPUT_SCALE)));
    f.matmul("yt", "wd_t", "hs", width);

    let stride = Y_LAYOUT.stride / Y_LAYOUT.element_bytes();
    let plane = stride * Y_LAYOUT.rows as usize;
    f.line(&format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{plane}, {plane}, {stride}, 1]))",
        Y_LAYOUT.buffer_type(width, rows)
    ));
    format!(
        "    func ffn{rows}<ios18>({}) {{\n{}    }} -> (y);\n",
        f.params.join(", "),
        f.body
    )
}

#[must_use]
pub fn ffn_program_text() -> String {
    let mut text = String::from("program(1.3)\n{\n");
    for rows in ANE_FUNCTION_ROWS {
        text.push_str(&ffn_mil_function(rows));
    }
    text.push_str("}\n");
    text
}

fn placeholder_blob() -> Vec<u8> {
    let mut blob = vec![0_u8; 192];
    blob[0..4].copy_from_slice(&1_u32.to_le_bytes());
    blob[4..8].copy_from_slice(&2_u32.to_le_bytes());
    blob[64..68].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
    blob[68..72].copy_from_slice(&1_u32.to_le_bytes());
    blob[72..80].copy_from_slice(&64_u64.to_le_bytes());
    blob[80..88].copy_from_slice(&128_u64.to_le_bytes());
    blob
}

fn function_for(rows: u32) -> Option<usize> {
    ANE_FUNCTION_ROWS
        .iter()
        .position(|&capacity| capacity >= rows)
}

struct WeightSet {
    wg: Surface,
    wu: Surface,
    wd: Surface,
}

impl WeightSet {
    fn new(ctx: &MetalContext) -> crate::Result<Self> {
        Ok(Self {
            wg: Surface::new(ctx, GATE_LAYOUT)?,
            wu: Surface::new(ctx, GATE_LAYOUT)?,
            wd: Surface::new(ctx, DOWN_LAYOUT)?,
        })
    }

    fn input<'a>(&'a self, name: &str, x0: &'a Surface) -> Option<&'a Surface> {
        Some(match name {
            "x0" => x0,
            "wg" => &self.wg,
            "wu" => &self.wu,
            "wd" => &self.wd,
            _ => return None,
        })
    }
}

pub struct AneFfnOffload {
    handoff: Handoff,
    kernels: AneKernels,
    x0: Surface,
    y: Surface,
    sets: [WeightSet; 2],
    /// `[set][func_idx]`
    bindings: [[Arc<Binding>; 2]; 2],
    pub status: MetalBuffer,
    /// Per-layer FFN time `(gpu, ane)` in ms measured when the offload was enabled.
    pub layer_ms: (f64, f64),
}

impl AneFfnOffload {
    pub fn new(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        program: &Program,
    ) -> crate::Result<Self> {
        let handoff = Handoff::new(ctx, JOB_BOUND)?;
        let kernels = AneKernels::new(shaders)?;
        let x0 = Surface::new(ctx, X_LAYOUT)?;
        let y = Surface::new(ctx, Y_LAYOUT)?;
        let sets = [WeightSet::new(ctx)?, WeightSet::new(ctx)?];

        let bind_fn = |set_idx: usize, func_idx: usize| -> crate::Result<Arc<Binding>> {
            let rows = ANE_FUNCTION_ROWS[func_idx];
            let fname = format!("ffn{rows}");
            let proc_idx = program
                .procedure(&fname)
                .ok_or_else(|| crate::Error::InvalidFormat(format!("missing ANE proc {fname}")))?;
            let names = program.inputs(proc_idx);
            let mut inputs = Vec::with_capacity(names.len());
            for n in names {
                let s = sets[set_idx]
                    .input(n, &x0)
                    .ok_or_else(|| crate::Error::InvalidFormat(format!("unknown ANE input {n}")))?;
                inputs.push(s);
            }
            Ok(Arc::new(program.bind(proc_idx, &inputs, &y)?))
        };

        let bindings = [
            [bind_fn(0, 0)?, bind_fn(0, 1)?],
            [bind_fn(1, 0)?, bind_fn(1, 1)?],
        ];
        let status = MetalBuffer::empty(ctx.device(), 16)?;
        status.clear();
        Ok(Self {
            handoff,
            kernels,
            x0,
            y,
            sets,
            bindings,
            status,
            layer_ms: (0.0, 0.0),
        })
    }

    /// Stage `layer`'s `gate_proj`, `up_proj`, and `down_proj` into weight set `set_idx`.
    pub fn stage_layer(
        &self,
        batch: &mut CommandBatch,
        artifact: &MappedArtifact,
        layer: usize,
        set_idx: usize,
    ) -> crate::Result<bool> {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let (AnyMatrix::Int4(gate), false) = artifact.matrix(&format!("{prefix}.gate_proj.weight"))?
        else {
            return Ok(false);
        };
        let (AnyMatrix::Int4(up), false) = artifact.matrix(&format!("{prefix}.up_proj.weight"))?
        else {
            return Ok(false);
        };
        let (AnyMatrix::Int4(down), false) = artifact.matrix(&format!("{prefix}.down_proj.weight"))?
        else {
            return Ok(false);
        };
        let set = &self.sets[set_idx];
        self.kernels.stage_int4_half(batch, gate, &set.wg)?;
        self.kernels.stage_int4_half(batch, up, &set.wu)?;
        self.kernels.stage_int4_half(batch, down, &set.wd)?;
        Ok(true)
    }

    /// Encode one layer's FFN on the ANE for `tokens` rows (`normed` -> `branch`), while
    /// simultaneously staging `next_layer` into the alternate weight set on the GPU!
    pub fn encode_layer_ffn(
        &self,
        batch: &mut CommandBatch,
        artifact: &MappedArtifact,
        layer: usize,
        next_layer: Option<usize>,
        normed: &MetalBuffer,
        branch: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<bool> {
        let Some(func_idx) = function_for(tokens) else {
            return Ok(false);
        };
        let func_rows = ANE_FUNCTION_ROWS[func_idx];
        let set_idx = layer & 1;
        self.kernels
            .pack_half(batch, normed, WIDTH as u32, tokens, func_rows, &self.x0)?;
        let ready = self.handoff.next_ready();
        let done = self.handoff.next_done();
        self.handoff
            .queue(&self.bindings[set_idx][func_idx], ready, done);
        batch.signal_event(self.handoff.ready_event(), ready)?;

        // Overlap GPU staging of `next_layer`'s weights into the other weight set while ANE runs!
        if let Some(next_l) = next_layer {
            let _ = self.stage_layer(batch, artifact, next_l, (layer + 1) & 1)?;
        }

        batch.wait_event(self.handoff.done_event(), done)?;
        self.kernels.join(
            batch,
            &self.y,
            branch,
            &self.status,
            WIDTH as u32,
            tokens,
            (1.0 / DOWN_INPUT_SCALE) as f32,
        )?;
        Ok(true)
    }

    #[must_use]
    #[expect(unsafe_code)]
    pub fn finish_and_check(&self) -> bool {
        if !self.handoff.finish() {
            return false;
        }
        let status = unsafe { self.status.as_slice::<u32>()[0] };
        status == 0
    }

    pub fn reset_status(&self) {
        self.status.clear();
    }
}

pub enum AneState {
    Off(String),
    Pending(mpsc::Receiver<crate::Result<Program>>),
    Ready(Box<AneFfnOffload>),
    Busy,
}

impl AneState {
    /// Compile the FFN program in the background on any chip; the first offloaded layer is
    /// checked against the GPU and the engine falls back to the GPU if they disagree.
    #[must_use]
    pub fn start() -> Self {
        if let Some(reason) = ane::unavailable() {
            return Self::Off(reason);
        }
        let (sender, receiver) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("micro-ane-compile".into())
            .spawn(move || {
                let prog = Program::compile(
                    &ffn_program_text(),
                    &placeholder_blob(),
                    CACHE_TAG,
                    COMPILE_LIMIT,
                )
                .map_err(crate::Error::Metal);
                let _ = sender.send(prog);
            });
        match spawned {
            Ok(_) => Self::Pending(receiver),
            Err(e) => Self::Off(format!("compile thread: {e}")),
        }
    }

    #[must_use]
    pub const fn is_active(&self) -> bool {
        !matches!(self, Self::Off(_))
    }

    #[must_use]
    pub fn status_json(&self) -> serde_json::Value {
        match self {
            Self::Off(reason) => serde_json::json!({"enabled": false, "reason": reason}),
            Self::Ready(offload) => serde_json::json!({
                "enabled": true,
                "min_rows": ANE_MIN_ROWS,
                "max_rows": ANE_MAX_ROWS,
                "ffn_layer_ms": { "gpu": offload.layer_ms.0, "ane": offload.layer_ms.1 },
            }),
            Self::Pending(_) | Self::Busy => serde_json::json!({
                "enabled": true,
                "min_rows": ANE_MIN_ROWS,
                "max_rows": ANE_MAX_ROWS,
            }),
        }
    }
}

#[must_use]
pub fn relative_difference(gpu: &[f32], ane: &[f32]) -> f64 {
    let (diff, reference) =
        gpu.iter()
            .zip(ane)
            .fold((0.0_f64, 0.0_f64), |(d, r), (&g, &a)| {
                let (g, a) = (f64::from(g), f64::from(a));
                ((g - a).mul_add(g - a, d), g.mul_add(g, r))
            });
    if reference == 0.0 {
        0.0
    } else {
        (diff / reference).sqrt()
    }
}

#[must_use]
pub const fn check_rows_and_tolerance() -> (u32, f64) {
    (CHECK_ROWS, CHECK_TOLERANCE)
}

// ============================================================================
// Vision Tower ANE Offload (27-Layer SigLIP2 `fc1` [4304, 1152] + `fc2` [1152, 4304])
// ============================================================================

const VIS_WIDTH: u32 = 1152;
const VIS_FFN: u32 = 4304;
const VIS_MAX_ROWS: u32 = 1024;

const VX_LAYOUT: SurfaceLayout = SurfaceLayout::new(VIS_WIDTH, VIS_MAX_ROWS, false);
const VTX_LAYOUT: SurfaceLayout = SurfaceLayout::new(1, VIS_MAX_ROWS, true);
const VFC1_W_LAYOUT: SurfaceLayout = SurfaceLayout::new(VIS_FFN, VIS_WIDTH, false);
const VFC1_S_LAYOUT: SurfaceLayout = SurfaceLayout::new(VIS_FFN, 1, true);
const VFC2_W_LAYOUT: SurfaceLayout = SurfaceLayout::new(VIS_WIDTH, VIS_FFN, false);
const VFC2_S_LAYOUT: SurfaceLayout = SurfaceLayout::new(VIS_WIDTH, 1, true);
const VY_LAYOUT: SurfaceLayout = SurfaceLayout::new(VIS_WIDTH + 1, VIS_MAX_ROWS, true);

fn vmlp_mil_function(rows: u32) -> String {
    let (width, ffn) = (VIS_WIDTH, VIS_FFN);
    let mut f = MilFunction {
        rows,
        params: Vec::new(),
        body: String::new(),
    };
    f.input("x0", VX_LAYOUT, width, rows);
    f.input("tx", VTX_LAYOUT, 1, rows);
    f.input("w1", VFC1_W_LAYOUT, ffn, width);
    f.input("s1", VFC1_S_LAYOUT, ffn, 1);
    f.input("b1", VFC1_S_LAYOUT, ffn, 1);
    f.input("w2", VFC2_W_LAYOUT, width, ffn);
    f.input("s2", VFC2_S_LAYOUT, width, 1);

    f.dequantize("x0", width, rows);
    f.dequantize("w1", ffn, width);
    f.matmul("m1", "w1_d", "x0_d", ffn);
    f.half("m1s", ffn, rows, "mul(x = m1, y = s1_t)");
    f.half("m1t", ffn, rows, "mul(x = m1s, y = tx_t)");
    f.half("z", ffn, rows, "add(x = m1t, y = b1_t)");

    f.half("z2", ffn, rows, "mul(x = z, y = z)");
    f.half("z3", ffn, rows, "mul(x = z2, y = z)");
    f.half("zc", ffn, rows, &format!("mul(x = z3, y = {})", fp16(0.044_715)));
    f.half("zi", ffn, rows, "add(x = z, y = zc)");
    f.half(
        "arg",
        ffn,
        rows,
        &format!("mul(x = zi, y = {})", fp16(0.797_884_560_802_865_4)),
    );
    f.half("th", ffn, rows, "tanh(x = arg)");
    f.half("tp", ffn, rows, &format!("add(x = th, y = {})", fp16(1.0)));
    f.half("zh", ffn, rows, &format!("mul(x = z, y = {})", fp16(0.5)));
    f.half("h", ffn, rows, "mul(x = zh, y = tp)");

    f.half("ha", ffn, rows, "abs(x = h)");
    f.half(
        "pk",
        1,
        rows,
        "reduce_max(x = ha, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))",
    );
    f.half(
        "fl",
        1,
        rows,
        &format!("maximum(x = pk, y = {})", fp16(f64::powi(2.0, -9))),
    );
    f.half(
        "iv",
        1,
        rows,
        &format!("real_div(x = {}, y = fl)", fp16(127.0)),
    );
    f.half("hs", ffn, rows, "mul(x = h, y = iv)");
    let q_ty = MilFunction::tensor("int8", ffn, rows);
    f.line(&format!(
        "{q_ty} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))"
    ));
    f.half(
        "hd",
        ffn,
        rows,
        &format!("dequantize(input = hq, scale = {})", fp16(1.0 / 128.0)),
    );
    f.half(
        "hsc",
        1,
        rows,
        &format!("mul(x = fl, y = {})", fp16(128.0 / 127.0)),
    );

    f.dequantize("w2", width, ffn);
    f.matmul("m2", "w2_d", "hd", width);
    f.half("ds", width, rows, "mul(x = m2, y = s2_t)");
    f.half(
        "yt",
        width + 1,
        rows,
        "concat(axis = int32(2), interleave = bool(false), values = (ds, hsc))",
    );

    let stride = VY_LAYOUT.stride / VY_LAYOUT.element_bytes();
    let plane = stride * VY_LAYOUT.rows as usize;
    f.line(&format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([{plane}, {plane}, {stride}, 1]))",
        VY_LAYOUT.buffer_type(width + 1, rows)
    ));
    format!(
        "    func vmlp_{rows}<ios18>({}) {{\n{}    }} -> (y);\n",
        f.params.join(", "),
        f.body
    )
}

#[must_use]
pub fn vision_program_text() -> String {
    let mut text = String::from("program(1.3)\n{\n");
    for rows in VIS_ANE_ROWS {
        text.push_str(&vmlp_mil_function(rows));
    }
    text.push_str("}\n");
    text
}

struct VisWeightSet {
    w1: Surface,
    s1: Surface,
    b1: Surface,
    w2: Surface,
    s2: Surface,
}

impl VisWeightSet {
    fn new(ctx: &MetalContext) -> crate::Result<Self> {
        Ok(Self {
            w1: Surface::new(ctx, VFC1_W_LAYOUT)?,
            s1: Surface::new(ctx, VFC1_S_LAYOUT)?,
            b1: Surface::new(ctx, VFC1_S_LAYOUT)?,
            w2: Surface::new(ctx, VFC2_W_LAYOUT)?,
            s2: Surface::new(ctx, VFC2_S_LAYOUT)?,
        })
    }

    fn input<'a>(&'a self, name: &str, x0: &'a Surface, tx: &'a Surface) -> Option<&'a Surface> {
        Some(match name {
            "x0" => x0,
            "tx" => tx,
            "w1" => &self.w1,
            "s1" => &self.s1,
            "b1" => &self.b1,
            "w2" => &self.w2,
            "s2" => &self.s2,
            _ => return None,
        })
    }
}

pub struct VisionAneOffload {
    handoff: Handoff,
    kernels: AneKernels,
    x0: Surface,
    tx: Surface,
    y: Surface,
    sets: [VisWeightSet; 2],
    /// `[set][func_idx]` for `vmlp_{256, 1024}`
    bindings: [[Arc<Binding>; 2]; 2],
    pub status: MetalBuffer,
}

impl VisionAneOffload {
    pub fn new(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        program: &Program,
    ) -> crate::Result<Self> {
        let handoff = Handoff::new(ctx, JOB_BOUND)?;
        let kernels = AneKernels::new(shaders)?;
        let x0 = Surface::new(ctx, VX_LAYOUT)?;
        let tx = Surface::new(ctx, VTX_LAYOUT)?;
        let y = Surface::new(ctx, VY_LAYOUT)?;
        let sets = [VisWeightSet::new(ctx)?, VisWeightSet::new(ctx)?];

        let bind_vmlp = |fname: &str, set: &VisWeightSet| -> crate::Result<Arc<Binding>> {
            let proc_idx = program
                .procedure(fname)
                .ok_or_else(|| crate::Error::InvalidFormat(format!("missing ANE proc {fname}")))?;
            let names = program.inputs(proc_idx);
            let mut inputs = Vec::with_capacity(names.len());
            for n in names {
                let s = set.input(n, &x0, &tx).ok_or_else(|| {
                    crate::Error::InvalidFormat(format!("unknown ANE input {n}"))
                })?;
                inputs.push(s);
            }
            Ok(Arc::new(program.bind(proc_idx, &inputs, &y)?))
        };

        let bindings = [
            [
                bind_vmlp("vmlp_256", &sets[0])?,
                bind_vmlp("vmlp_1024", &sets[0])?,
            ],
            [
                bind_vmlp("vmlp_256", &sets[1])?,
                bind_vmlp("vmlp_1024", &sets[1])?,
            ],
        ];
        let status = MetalBuffer::empty(ctx.device(), 16)?;
        status.clear();
        Ok(Self {
            handoff,
            kernels,
            x0,
            tx,
            y,
            sets,
            bindings,
            status,
        })
    }

    pub fn stage_layer(
        &self,
        batch: &mut CommandBatch,
        artifact: &MappedArtifact,
        layer: usize,
        set_idx: usize,
    ) -> crate::Result<bool> {
        let p = format!("model.vision_tower.encoder.layers.{layer}.mlp");
        let (AnyMatrix::Int4(fc1), false) = artifact.matrix(&format!("{p}.fc1.weight"))? else {
            return Ok(false);
        };
        let (b1_buf, b1_off, _) = artifact.f32_slice(&format!("{p}.fc1.bias"))?;
        let fc2_name = format!("{p}.fc2.weight");
        let set = &self.sets[set_idx];
        let desc2 = artifact.descriptor(&fc2_name)?;
        let fc2_mat = if desc2.dtype == crate::artifact::TensorDtype::F32 {
            None
        } else if let (AnyMatrix::Int4(fc2), false) = artifact.matrix(&fc2_name)? {
            Some(fc2)
        } else {
            return Ok(false);
        };
        batch.independent(|batch| {
            self.kernels.stage_int4(batch, fc1, &set.w1, &set.s1)?;
            self.kernels
                .stage_bias(batch, (b1_buf, b1_off), VIS_FFN, &set.b1)?;
            if let Some(fc2) = fc2_mat {
                self.kernels.stage_int4(batch, fc2, &set.w2, &set.s2)?;
            } else {
                self.kernels.stage_f32(
                    batch,
                    (&artifact.payload_buffer, desc2.offset),
                    VIS_WIDTH,
                    VIS_FFN,
                    &set.w2,
                    &set.s2,
                )?;
            }
            Ok(true)
        })
    }

    /// Execute `SigLIP2` layer `layer`'s `fc1` (with `gelu_tanh` into `ffn_mid`) and `fc2`
    /// (accumulated into `hidden`) on the ANE while overlapping next-layer weight staging on the Metal GPU!
    pub fn encode_layer_mlp(
        &self,
        batch: &mut CommandBatch,
        artifact: &MappedArtifact,
        layer: usize,
        next_layer: Option<usize>,
        normed: &MetalBuffer,
        hidden: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<bool> {
        let Some(func_idx) = VIS_ANE_ROWS.iter().position(|&r| r >= tokens) else {
            return Ok(false);
        };
        let func_rows = VIS_ANE_ROWS[func_idx];
        let set_idx = layer & 1;
        let p = format!("model.vision_tower.encoder.layers.{layer}.mlp");
        let (b2_buf, b2_off, _) = artifact.f32_slice(&format!("{p}.fc2.bias"))?;

        // 1. Pack `normed` -> `x0`/`tx` and dispatch fused `vmlp_{256,1024}` on ANE
        self.kernels
            .pack(batch, normed, VIS_WIDTH, tokens, func_rows, &self.tx, &self.x0)?;
        let ready = self.handoff.next_ready();
        let done = self.handoff.next_done();
        self.handoff
            .queue(&self.bindings[set_idx][func_idx], ready, done);
        batch.signal_event(self.handoff.ready_event(), ready)?;

        // Overlap next-layer weight staging into the alternate weight set while ANE runs `vmlp`
        if let Some(nl) = next_layer {
            let _ = self.stage_layer(batch, artifact, nl, (layer + 1) & 1)?;
        }

        // 2. Join `vmlp` output (`y[0..1152] * y[1152] + b2`) accumulated into `hidden` (mode = 4)
        batch.wait_event(self.handoff.done_event(), done)?;
        self.kernels.join_linear(
            batch,
            &self.y,
            &self.tx,
            (b2_buf, b2_off),
            hidden,
            &self.status,
            VIS_WIDTH,
            tokens,
            4,
        )?;
        Ok(true)
    }

    #[must_use]
    #[expect(unsafe_code)]
    pub fn finish_and_check(&self) -> bool {
        if !self.handoff.finish() {
            return false;
        }
        let status = unsafe { self.status.as_slice::<u32>()[0] };
        status == 0
    }

    pub fn reset_status(&self) {
        self.status.clear();
    }
}

pub enum VisionAneState {
    Off(String),
    Pending(mpsc::Receiver<crate::Result<Program>>),
    Ready(Box<VisionAneOffload>),
    Busy,
}

impl VisionAneState {
    /// Compile the vision MLP program in the background on any chip; failures fall back to the GPU.
    /// (iPhone SE 3: ~3.2 s per image with the ANE against ~5.3 s on the GPU alone.)
    #[must_use]
    pub fn start() -> Self {
        if let Some(reason) = ane::unavailable() {
            return Self::Off(reason);
        }
        let (sender, receiver) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("micro-vis-ane-compile".into())
            .spawn(move || {
                let prog = Program::compile(
                    &vision_program_text(),
                    &placeholder_blob(),
                    VIS_CACHE_TAG,
                    COMPILE_LIMIT,
                )
                .map_err(crate::Error::Metal);
                let _ = sender.send(prog);
            });
        match spawned {
            Ok(_) => Self::Pending(receiver),
            Err(e) => Self::Off(format!("compile thread: {e}")),
        }
    }
}
