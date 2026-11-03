//! C ABI for the iOS app: run prompts against an engine opened once per model directory and
//! report speed and memory (the physical footprint jetsam enforces) as JSON.

use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::sync::{Mutex, PoisonError, mpsc};
use std::time::Instant;

use micro_engine::model::MicroEngine;
use micro_engine::sampler::SamplingParams;
use micro_engine::vision::DownsampleMode;
use micro_metal::memory::{budget, usage};

const MODEL_FILE: &str = "minicpm-v-4.7-1b-tern4.bin";

/// The open engine and its model directory, kept so later prompts skip loading and reuse the
/// prefix cache and ANE decision.
type Slot = Option<(String, MicroEngine)>;
type Job = Box<dyn FnOnce(&mut Slot) + Send>;

/// Jobs for the one thread that owns the engine (its Metal objects cannot cross threads).
static WORKER: Mutex<Option<mpsc::Sender<Job>>> = Mutex::new(None);

fn on_engine_thread<R: Send + 'static>(
    job: impl FnOnce(&mut Slot) -> R + Send + 'static,
) -> Option<R> {
    let (reply, result) = mpsc::channel();
    let mut worker = WORKER.lock().unwrap_or_else(PoisonError::into_inner);
    let jobs = worker.get_or_insert_with(|| {
        let (jobs, inbox) = mpsc::channel::<Job>();
        std::thread::spawn(move || {
            let mut slot = None;
            for job in inbox {
                job(&mut slot);
            }
        });
        jobs
    });
    jobs.send(Box::new(move |slot| {
        let _ = reply.send(job(slot));
    }))
    .ok()?;
    drop(worker);
    result.recv().ok()
}

/// Footprint (MiB) and page-ins so far; page-ins growing while decoding means evicted weights
/// are being re-read from storage.
fn memory() -> serde_json::Value {
    usage().map_or(serde_json::Value::Null, |u| {
        serde_json::json!({
            "footprint_mib": u.footprint >> 20,
            "peak_mib": u.peak_footprint >> 20,
            "page_ins": u.page_ins,
        })
    })
}

/// The engine for `model_dir`, opened on first use (or when the directory changes).
fn engine<'a>(slot: &'a mut Slot, model_dir: &str) -> micro_engine::Result<&'a mut MicroEngine> {
    if slot.as_ref().is_none_or(|(dir, _)| dir != model_dir) {
        *slot = None;
        *slot = Some((
            model_dir.to_owned(),
            MicroEngine::open(&Path::new(model_dir).join(MODEL_FILE))?,
        ));
    }
    slot.as_mut()
        .map(|(_, engine)| engine)
        .ok_or_else(|| micro_engine::Error::InvalidArgument("engine not open".into()))
}

/// Prefill the chat prompt up to the user's `typed` text. The last two tokens stay out, since
/// more typing can re-merge them. Empty `typed` pins the system turn in the prompt cache (memory
/// and disk), so it also restores instantly on later launches.
fn warm(slot: &mut Slot, model_dir: &str, typed: &str) -> micro_engine::Result<()> {
    let engine = engine(slot, model_dir)?;
    let full = MicroEngine::format_chat_prompt(typed);
    let Some(user_turn) = full.strip_suffix("<|im_end|>\n<|im_start|>assistant\n<think>\n") else {
        return Ok(());
    };
    if typed.is_empty() {
        let system_turn = user_turn.strip_suffix("<|im_start|>user\n").unwrap_or(user_turn);
        engine.warm_prefix_cache(system_turn)?;
        return Ok(());
    }
    let ids = engine.encode_text(user_turn)?;
    engine.prefill_prefix(&ids[..ids.len().saturating_sub(2)])
}

fn run(
    slot: &mut Slot,
    model_dir: &str,
    prompt: &str,
    image: Option<&str>,
    max_tokens: usize,
) -> micro_engine::Result<serde_json::Value> {
    let start = Instant::now();
    let engine = engine(slot, model_dir)?;
    let open_ms = start.elapsed().as_secs_f64() * 1e3;
    let before = memory();
    let run_start = Instant::now();
    let report = if let Some(image) = image {
        let bytes = std::fs::read(Path::new(model_dir).join(image))?;
        engine.generate_multimodal(
            &bytes,
            prompt,
            DownsampleMode::Mode16x,
            max_tokens,
            SamplingParams::default(),
        )?
    } else {
        let ids = engine.encode_text(&MicroEngine::format_chat_prompt(prompt))?;
        engine.generate_ids(&ids, max_tokens, SamplingParams::default())?
    };
    let mut report = serde_json::to_value(report)?;
    if let Some(fields) = report.as_object_mut() {
        fields.remove("token_ids");
    }
    Ok(serde_json::json!({
        "device": engine.device_name(),
        "ane": engine.ane_status(),
        "profile": engine.memory_profile(),
        "budget_mib": budget() >> 20,
        "open_ms": open_ms,
        // Includes image preparation and the vision tower, which prefill_ms does not.
        "run_ms": run_start.elapsed().as_secs_f64() * 1e3,
        "memory": { "before": before, "after": memory() },
        "report": report,
    }))
}

/// Run one prompt and return its JSON report, to release with `micro_ai_free`.
///
/// The model is the one in `model_dir`; `image_path` (relative to `model_dir`, may be null) adds an
/// image; at most `max_tokens` tokens are generated. Failures come back as `{"error": …}`.
///
/// # Safety
///
/// `model_dir` and `prompt` must be valid NUL-terminated strings; `image_path` may be null.
#[expect(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn micro_ai_run(
    model_dir: *const c_char,
    prompt: *const c_char,
    image_path: *const c_char,
    max_tokens: u32,
) -> *mut c_char {
    // SAFETY: the caller passes NUL-terminated strings (image_path may be null).
    let (dir, prompt, image) = unsafe {
        (
            CStr::from_ptr(model_dir).to_string_lossy().into_owned(),
            CStr::from_ptr(prompt).to_string_lossy().into_owned(),
            (!image_path.is_null())
                .then(|| CStr::from_ptr(image_path).to_string_lossy().into_owned()),
        )
    };
    let json = on_engine_thread(move |slot| {
        run(slot, &dir, &prompt, image.as_deref(), max_tokens as usize)
            .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string(), "memory": memory() }))
    })
    .unwrap_or_else(|| serde_json::json!({ "error": "engine thread stopped" }));
    CString::new(json.to_string()).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Prefill the prompt up to `typed` (the user's text so far) while they type.
///
/// The next `micro_ai_run` whose prompt extends it only prefills the rest. Call it with "" at
/// launch to open the engine and pin the system turn. Returns false on failure.
///
/// # Safety
///
/// `model_dir` and `typed` must be valid NUL-terminated strings.
#[expect(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn micro_ai_warm(model_dir: *const c_char, typed: *const c_char) -> bool {
    // SAFETY: the caller passes NUL-terminated strings.
    let (dir, typed) = unsafe {
        (
            CStr::from_ptr(model_dir).to_string_lossy().into_owned(),
            CStr::from_ptr(typed).to_string_lossy().into_owned(),
        )
    };
    on_engine_thread(move |slot| warm(slot, &dir, &typed).is_ok()).unwrap_or(false)
}

/// Release a report returned by `micro_ai_run`.
///
/// # Safety
///
/// `report` must come from `micro_ai_run` (or be null) and not be used afterwards.
#[expect(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn micro_ai_free(report: *mut c_char) {
    if !report.is_null() {
        // SAFETY: `report` was produced by `CString::into_raw` in `micro_ai_run`.
        drop(unsafe { CString::from_raw(report) });
    }
}
