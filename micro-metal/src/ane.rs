//! Client of Apple's private `AppleNeuralEngine.framework`, the Neural Engine ("ANE"),
//! plus Metal pack/stage/join kernels (`shaders/ane.metal`) for MiniCPM-V 4.7 1B prefill & vision offload.
//!
//! Adapted from `local-ai/local-metal/src/ane.rs` and `bonsai_ane.rs` (Kapitsa / Splash Apache-2.0):
//! - Every selector is checked against its arm64 type encoding before the first call.
//! - Programs are MIL text whose weights are `IOSurface`-backed `tensor_buffer` inputs cached under
//!   `$(DARWIN_USER_CACHE_DIR)/<tag>/<hash>/`.
//! - `IOSurface`s are shared zero-copy with Metal ([`Surface`]).
//! - Evaluations are ordered with the GPU by two `MTLSharedEvent`s ([`Handoff`]) with a watchdog.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, PoisonError, mpsc};
use std::time::{Duration, Instant};

use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyClass, AnyObject, Bool, ProtocolObject, Sel};
use objc2::{msg_send, sel};
use objc2_core_foundation::{CFDictionary, CFRetained, CFString};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use objc2_io_surface::{
    IOSurfaceRef, kIOSurfaceAllocSize, kIOSurfaceBytesPerElement, kIOSurfaceBytesPerRow,
    kIOSurfaceHeight, kIOSurfacePixelFormat, kIOSurfaceWidth,
};
use objc2_metal::{MTLDevice, MTLSharedEvent, MTLSharedEventListener, MTLSize};

use crate::Error;
use crate::batch::{CommandBatch, ComputeEncoder};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::projection::Int4Matrix;
use crate::shaders::{Pipeline, ShaderLibrary};

const FRAMEWORK: &CStr =
    c"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
const RTLD_NOW: i32 = 2;
const CS_DARWIN_USER_CACHE_DIR: i32 = 65_538;
const QOS: u32 = 0x15;
const PIXEL_FP16: u32 = 0x4c30_3068;
const PIXEL_INT8: u32 = 0x4c30_3038;
const PAGE: usize = 16_384;

#[expect(unsafe_code)]
unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn confstr(name: i32, buffer: *mut c_char, length: usize) -> usize;
}

fn ane_error(message: impl Into<String>) -> Error {
    Error::NeuralEngine(message.into())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Shared<T: ?Sized>(Retained<T>);

#[expect(unsafe_code, clippy::non_send_fields_in_send_ty)]
unsafe impl<T: ?Sized> Send for Shared<T> {}
#[expect(unsafe_code)]
unsafe impl<T: ?Sized> Sync for Shared<T> {}

const SIGNATURES: &[(&str, bool, &str, &str)] = &[
    ("_ANEModel", true, "modelAtURL:key:", "@@:@@"),
    ("_ANEModel", false, "modelAttributes", "@@:"),
    ("_ANEClient", true, "sharedConnection", "@@:"),
    (
        "_ANEClient",
        false,
        "compileModel:options:qos:error:",
        "B@:@@I^@",
    ),
    ("_ANEClient", false, "compiledModelExistsFor:", "B@:@"),
    ("_ANEClient", false, "purgeCompiledModel:", "v@:@"),
    (
        "_ANEClient",
        false,
        "loadModel:options:qos:error:",
        "B@:@@I^@",
    ),
    (
        "_ANEClient",
        false,
        "unloadModel:options:qos:error:",
        "B@:@@I^@",
    ),
    (
        "_ANEClient",
        false,
        "evaluateWithModel:options:request:qos:error:",
        "B@:@@@I^@",
    ),
    (
        "_ANEIOSurfaceObject",
        true,
        "objectWithIOSurface:",
        "@@:^{__IOSurface=}",
    ),
    (
        "_ANERequest",
        true,
        "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:",
        "@@:@@@@@@@@@",
    ),
    ("_ANERequest", false, "setCompletionHandler:", "v@:@?"),
    (
        "_ANESharedWaitEvent",
        true,
        "waitEventWithValue:sharedEvent:eventType:",
        "@@:Q@Q",
    ),
    (
        "_ANESharedSignalEvent",
        true,
        "signalEventWithValue:symbolIndex:eventType:sharedEvent:",
        "@@:QIq@",
    ),
    (
        "_ANESharedEvents",
        true,
        "sharedEventsWithSignalEvents:waitEvents:",
        "@@:@@",
    ),
];

struct Api {
    client: Shared<AnyObject>,
    model: &'static AnyClass,
    surface: &'static AnyClass,
    request: &'static AnyClass,
    events: &'static AnyClass,
    signal_event: &'static AnyClass,
    wait_event: &'static AnyClass,
}

fn class(name: &str) -> Result<&'static AnyClass, String> {
    let name = CString::new(name).map_err(|error| error.to_string())?;
    AnyClass::get(&name)
        .ok_or_else(|| format!("AppleNeuralEngine lacks {}", name.to_string_lossy()))
}

#[expect(unsafe_code)]
fn check_signatures() -> Result<(), String> {
    for &(class_name, class_method, selector, expected) in SIGNATURES {
        let class = class(class_name)?;
        let name = CString::new(selector).map_err(|error| error.to_string())?;
        let sel = Sel::register(&name);
        let kind = if class_method { '+' } else { '-' };
        let method = if class_method {
            class.class_method(sel)
        } else {
            class.instance_method(sel)
        }
        .ok_or_else(|| format!("AppleNeuralEngine lacks {kind}[{class_name} {selector}]"))?;
        let encoding = unsafe {
            let raw = objc2::ffi::method_getTypeEncoding(method);
            if raw.is_null() {
                String::new()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            }
        };
        let stripped = encoding
            .chars()
            .filter(|c| !c.is_ascii_digit())
            .collect::<String>();
        if stripped != expected {
            return Err(format!(
                "AppleNeuralEngine changed {kind}[{class_name} {selector}]: {stripped}"
            ));
        }
    }
    Ok(())
}

#[expect(unsafe_code)]
fn resolve() -> Result<Api, String> {
    if unsafe { dlopen(FRAMEWORK.as_ptr(), RTLD_NOW) }.is_null() {
        let reason = unsafe {
            let raw = dlerror();
            if raw.is_null() {
                "unknown error".to_owned()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            }
        };
        return Err(format!("AppleNeuralEngine does not load: {reason}"));
    }
    check_signatures()?;
    let client_class = class("_ANEClient")?;
    let client = catch("_ANEClient sharedConnection", || {
        let client: Option<Retained<AnyObject>> =
            unsafe { msg_send![client_class, sharedConnection] };
        client
    })
    .map_err(|error| error.to_string())?
    .ok_or_else(|| "no _ANEClient shared connection".to_owned())?;
    Ok(Api {
        client: Shared(client),
        model: class("_ANEModel")?,
        surface: class("_ANEIOSurfaceObject")?,
        request: class("_ANERequest")?,
        events: class("_ANESharedEvents")?,
        signal_event: class("_ANESharedSignalEvent")?,
        wait_event: class("_ANESharedWaitEvent")?,
    })
}

static API: LazyLock<Result<Api, String>> = LazyLock::new(|| autoreleasepool(|_| resolve()));

fn api() -> Result<&'static Api, Error> {
    API.as_ref().map_err(|reason| ane_error(reason.clone()))
}

#[must_use]
pub fn unavailable() -> Option<String> {
    api().err().map(|error| match error {
        Error::NeuralEngine(reason) => reason,
        other => other.to_string(),
    })
}

fn catch<R>(what: &str, body: impl FnOnce() -> R) -> crate::Result<R> {
    objc2::exception::catch(AssertUnwindSafe(body)).map_err(|exception| {
        let reason = exception.map_or_else(|| "nil".to_owned(), |e| format!("{e:?}"));
        ane_error(format!("{what} raised {reason}"))
    })
}

fn error_text(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

#[must_use]
pub fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    bytes.iter().fold(seed, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
    })
}

pub const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;

/// Per-user cache directory: `confstr(_CS_DARWIN_USER_CACHE_DIR)` on macOS; sandboxed iOS apps
/// get no such directory, so they use their container's `Library/Caches`.
#[expect(unsafe_code)]
pub fn user_cache_dir() -> crate::Result<PathBuf> {
    let mut buffer = vec![0 as c_char; 1024];
    let written = unsafe { confstr(CS_DARWIN_USER_CACHE_DIR, buffer.as_mut_ptr(), buffer.len()) };
    if written > 0 && written <= buffer.len() {
        let path = unsafe { CStr::from_ptr(buffer.as_ptr()) };
        return Ok(PathBuf::from(path.to_string_lossy().into_owned()));
    }
    std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join("Library/Caches"))
        .ok_or_else(|| ane_error("no DARWIN_USER_CACHE_DIR or HOME"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceLayout {
    pub rows: u32,
    pub width: u32,
    pub fp16: bool,
    pub stride: usize,
    pub bytes: usize,
}

impl SurfaceLayout {
    #[must_use]
    pub const fn new(rows: u32, width: u32, fp16: bool) -> Self {
        let element = if fp16 { 2 } else { 1 };
        let stride = (width as usize * element).div_ceil(64) * 64;
        let bytes = (stride * rows as usize).div_ceil(PAGE) * PAGE;
        Self {
            rows,
            width,
            fp16,
            stride,
            bytes,
        }
    }

    #[must_use]
    pub const fn element_bytes(self) -> usize {
        if self.fp16 { 2 } else { 1 }
    }

    #[must_use]
    pub const fn mil_type(self) -> &'static str {
        if self.fp16 { "fp16" } else { "int8" }
    }

    #[must_use]
    pub fn strides(self) -> String {
        let stride = self.stride / self.element_bytes();
        let plane = stride * self.rows as usize;
        format!("[{plane}, {plane}, {stride}, 1]")
    }

    #[must_use]
    pub fn buffer_type(self, rows: u32, width: u32) -> String {
        format!(
            "tensor_buffer<{}, shape=[1, 1, {rows}, {width}], strides={}, interleave_factors=[1, 1, 1, 1]>",
            self.mil_type(),
            self.strides()
        )
    }
}

struct SurfaceRef(CFRetained<IOSurfaceRef>);

#[expect(unsafe_code, clippy::non_send_fields_in_send_ty)]
unsafe impl Send for SurfaceRef {}
#[expect(unsafe_code)]
unsafe impl Sync for SurfaceRef {}

pub struct Surface {
    owner: Arc<SurfaceRef>,
    buffer: MetalBuffer,
    layout: SurfaceLayout,
}

impl Surface {
    #[expect(unsafe_code)]
    pub fn new(context: &MetalContext, layout: SurfaceLayout) -> crate::Result<Self> {
        let pixel = if layout.fp16 { PIXEL_FP16 } else { PIXEL_INT8 };
        let values = [
            NSNumber::new_u32(layout.width),
            NSNumber::new_u32(layout.rows),
            NSNumber::new_u32(layout.element_bytes() as u32),
            NSNumber::new_usize(layout.stride),
            NSNumber::new_usize(layout.bytes),
            NSNumber::new_u32(pixel),
        ];
        let keys: [&NSString; 6] = unsafe {
            [
                kIOSurfaceWidth,
                kIOSurfaceHeight,
                kIOSurfaceBytesPerElement,
                kIOSurfaceBytesPerRow,
                kIOSurfaceAllocSize,
                kIOSurfacePixelFormat,
            ]
            .map(|key: &CFString| &*std::ptr::from_ref(key).cast::<NSString>())
        };
        let refs = values.iter().map(|value| &**value).collect::<Vec<_>>();
        let properties = NSDictionary::from_slices(&keys, &refs);
        let surface =
            unsafe { IOSurfaceRef::new(&*Retained::as_ptr(&properties).cast::<CFDictionary>()) }
                .ok_or_else(|| {
                    ane_error(format!(
                        "IOSurfaceCreate {}x{} failed",
                        layout.rows, layout.width
                    ))
                })?;
        if surface.bytes_per_row() != layout.stride || surface.alloc_size() < layout.bytes {
            return Err(ane_error("IOSurface took another row stride or size"));
        }
        let base = surface.base_address().cast::<u8>();
        let surface = Arc::new(SurfaceRef(surface));
        let owner: Arc<dyn Send + Sync> = surface.clone();
        let buffer = unsafe {
            MetalBuffer::from_bytes_no_copy(context.device(), base, layout.bytes, owner)?
        };
        Ok(Self {
            owner: surface,
            buffer,
            layout,
        })
    }

    #[must_use]
    pub const fn buffer(&self) -> &MetalBuffer {
        &self.buffer
    }

    #[must_use]
    pub const fn layout(&self) -> SurfaceLayout {
        self.layout
    }

    fn raw(&self) -> &IOSurfaceRef {
        &self.owner.0
    }
}

struct Procedure {
    name: String,
    inputs: Vec<String>,
    input_symbols: Vec<u32>,
    output_symbol: u32,
}

struct ProgramInner {
    model: Shared<AnyObject>,
}

impl Drop for ProgramInner {
    #[expect(unsafe_code)]
    fn drop(&mut self) {
        let model = Shared(self.model.0.clone());
        let _ = on_service_thread("ANE unload", Duration::from_secs(10), move || {
            let model = model;
            let api = api()?;
            catch("ANE unload", || {
                let _: Result<(), Retained<NSError>> = unsafe {
                    msg_send![&*api.client.0, unloadModel: &*model.0, options: &*NSDictionary::<NSString, AnyObject>::new(), qos: QOS, error: _]
                };
            })
        });
    }
}

fn on_service_thread<R: Send + 'static>(
    what: &'static str,
    limit: Duration,
    body: impl FnOnce() -> crate::Result<R> + Send + 'static,
) -> crate::Result<R> {
    static SERVICE: Mutex<()> = Mutex::new(());
    let (sender, receiver) = mpsc::channel();
    std::thread::Builder::new()
        .name("ane-service".into())
        .spawn(move || {
            let _serial = lock(&SERVICE);
            let result = autoreleasepool(|_| body());
            let _ = sender.send(result);
        })
        .map_err(|error| ane_error(format!("{what}: {error}")))?;
    receiver.recv_timeout(limit).map_err(|_| {
        ane_error(format!(
            "{what} did not finish within {} s",
            limit.as_secs()
        ))
    })?
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> crate::Result<()> {
    static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    let write = WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = path.with_extension(format!("tmp-{}-{write}", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ProgramReport {
    pub cached: bool,
    pub recompiled: bool,
    pub compile: Duration,
    pub load: Duration,
    pub directory: PathBuf,
}

pub struct Program {
    inner: Arc<ProgramInner>,
    procedures: Arc<Vec<Procedure>>,
    report: ProgramReport,
}

struct Loaded {
    model: Shared<AnyObject>,
    procedures: Vec<Procedure>,
    cached: bool,
    recompiled: bool,
    compile: Duration,
    load: Duration,
}

#[expect(unsafe_code)]
fn object_for_key(dictionary: &AnyObject, key: &str) -> Option<Retained<AnyObject>> {
    let key = NSString::from_str(key);
    unsafe { msg_send![dictionary, objectForKey: &*key] }
}

#[expect(unsafe_code)]
fn array_items(array: &AnyObject) -> Vec<Retained<AnyObject>> {
    let count: usize = unsafe { msg_send![array, count] };
    (0..count)
        .filter_map(|index| {
            let item: Option<Retained<AnyObject>> =
                unsafe { msg_send![array, objectAtIndex: index] };
            item
        })
        .collect()
}

#[expect(unsafe_code)]
fn number(object: &AnyObject) -> u32 {
    unsafe { msg_send![object, unsignedIntValue] }
}

#[expect(unsafe_code)]
fn string(object: &AnyObject) -> String {
    unsafe {
        let raw: *const c_char = msg_send![object, UTF8String];
        if raw.is_null() {
            String::new()
        } else {
            CStr::from_ptr(raw).to_string_lossy().into_owned()
        }
    }
}

#[expect(unsafe_code)]
fn describe(model: &AnyObject) -> crate::Result<Vec<Procedure>> {
    let missing = || ane_error("the ANE model does not describe its procedures");
    let attributes: Option<Retained<AnyObject>> = unsafe { msg_send![model, modelAttributes] };
    let attributes = attributes.ok_or_else(missing)?;
    let description = object_for_key(&attributes, "ANEFModelDescription").ok_or_else(missing)?;
    let symbols = object_for_key(&description, "kANEFModelInputSymbolsArrayKey")
        .map(|array| array_items(&array))
        .ok_or_else(missing)?;
    let names =
        object_for_key(&description, "kANEFModelProcedureNameToIDMapKey").ok_or_else(missing)?;
    let entries = object_for_key(&description, "ANEFModelProcedures")
        .map(|array| array_items(&array))
        .ok_or_else(missing)?;
    let mut procedures = (0..entries.len())
        .map(|_| Procedure {
            name: String::new(),
            inputs: Vec::new(),
            input_symbols: Vec::new(),
            output_symbol: 0,
        })
        .collect::<Vec<_>>();
    let keys: Option<Retained<AnyObject>> = unsafe { msg_send![&*names, allKeys] };
    for key in keys.map_or_default(|keys| array_items(&keys)) {
        let id = object_for_key(&names, &string(&key)).map(|id| number(&id));
        let procedure = id
            .and_then(|id| procedures.get_mut(id as usize))
            .ok_or_else(|| ane_error("ANE procedure id out of range"))?;
        procedure.name = string(&key);
    }
    for entry in entries {
        let id = object_for_key(&entry, "ANEFModelProcedureID")
            .map(|id| number(&id))
            .ok_or_else(missing)?;
        let procedure = procedures
            .get_mut(id as usize)
            .ok_or_else(|| ane_error("ANE procedure id out of range"))?;
        let outputs = object_for_key(&entry, "ANEFModelOutputSymbolIndexArray")
            .map_or_default(|array| array_items(&array));
        let [output] = outputs.as_slice() else {
            return Err(ane_error("ANE procedure with other than one output"));
        };
        procedure.output_symbol = number(output);
        for symbol in object_for_key(&entry, "ANEFModelInputSymbolIndexArray")
            .map_or_default(|array| array_items(&array))
        {
            let index = number(&symbol);
            let name = symbols
                .get(index as usize)
                .ok_or_else(|| ane_error("ANE input symbol out of range"))?;
            procedure.input_symbols.push(index);
            procedure.inputs.push(string(name));
        }
    }
    Ok(procedures)
}

#[expect(unsafe_code)]
fn compile_and_load(directory: &Path, key: &str) -> crate::Result<Loaded> {
    let api = api()?;
    let url =
        NSURL::fileURLWithPath_isDirectory(&NSString::from_str(&directory.to_string_lossy()), true);
    let key = NSString::from_str(key);
    let model = catch("_ANEModel modelAtURL:key:", || {
        let model: Option<Retained<AnyObject>> =
            unsafe { msg_send![api.model, modelAtURL: &*url, key: &*key] };
        model
    })?
    .ok_or_else(|| ane_error("_ANEModel modelAtURL:key: returned nil"))?;
    let options = NSDictionary::from_slices(
        &[
            &*NSString::from_str("kANEFModelType"),
            &*NSString::from_str("kANEFNetPlistFilenameKey"),
        ],
        &[
            &*NSString::from_str("kANEFModelMIL"),
            &*NSString::from_str("model.mil"),
        ],
    );
    let empty = NSDictionary::<NSString, AnyObject>::new();
    let client = &*api.client.0;
    let compile = || -> crate::Result<()> {
        catch("ANE compile", || {
            let result: Result<(), Retained<NSError>> = unsafe {
                msg_send![client, compileModel: &*model, options: &*options, qos: QOS, error: _]
            };
            result
        })?
        .map_err(|error| ane_error(format!("ANE compile failed: {}", error_text(&error))))
    };
    let load = || -> crate::Result<Result<(), String>> {
        catch("ANE load", || {
            let result: Result<(), Retained<NSError>> = unsafe {
                msg_send![client, loadModel: &*model, options: &*empty, qos: QOS, error: _]
            };
            result.map_err(|error| error_text(&error))
        })
    };
    let started = Instant::now();
    let cached = catch("ANE compiledModelExistsFor", || {
        let exists: bool = unsafe { msg_send![client, compiledModelExistsFor: &*model] };
        exists
    })?;
    if !cached {
        compile()?;
    }
    let compile_time = started.elapsed();
    let started = Instant::now();
    let recompiled = if let Err(reason) = load()? {
        eprintln!("ane: load failed ({reason}); purging and recompiling");
        catch("ANE purge", || {
            let (): () = unsafe { msg_send![client, purgeCompiledModel: &*model] };
        })?;
        compile()?;
        load()?.map_err(|reason| ane_error(format!("ANE load failed: {reason}")))?;
        true
    } else {
        false
    };
    let load_time = started.elapsed();
    let procedures = catch("ANE modelAttributes", || describe(&model))??;
    Ok(Loaded {
        model: Shared(model),
        procedures,
        cached: cached && !recompiled,
        recompiled,
        compile: compile_time,
        load: load_time,
    })
}

impl Program {
    pub fn compile(mil: &str, blob: &[u8], tag: &str, limit: Duration) -> crate::Result<Self> {
        api()?;
        let key = format!("{:016x}", fnv1a(blob, fnv1a(mil.as_bytes(), FNV_OFFSET)));
        let tag_dir = user_cache_dir()?.join(tag);
        let directory = tag_dir.join(&key);
        std::fs::create_dir_all(&directory)?;
        // Each MIL/weights revision gets a new key; drop the stale ones so the cache stays one entry.
        if let Ok(entries) = std::fs::read_dir(&tag_dir) {
            for entry in entries.flatten() {
                if entry.file_name() != key.as_str() {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
        write_if_changed(&directory.join("model.mil"), mil.as_bytes())?;
        write_if_changed(&directory.join("weights.bin"), blob)?;
        let path = directory.clone();
        let loaded = on_service_thread("ANE compile/load", limit, move || {
            compile_and_load(&path, &key)
        })?;
        Ok(Self {
            inner: Arc::new(ProgramInner {
                model: loaded.model,
            }),
            procedures: Arc::new(loaded.procedures),
            report: ProgramReport {
                cached: loaded.cached,
                recompiled: loaded.recompiled,
                compile: loaded.compile,
                load: loaded.load,
                directory,
            },
        })
    }

    #[must_use]
    pub const fn report(&self) -> &ProgramReport {
        &self.report
    }

    #[must_use]
    pub fn procedure(&self, name: &str) -> Option<usize> {
        self.procedures
            .iter()
            .position(|procedure| procedure.name == name)
    }

    #[must_use]
    pub fn inputs(&self, index: usize) -> &[String] {
        self.procedures
            .get(index)
            .map_or(&[], |procedure| procedure.inputs.as_slice())
    }

    #[expect(unsafe_code)]
    pub fn bind(
        &self,
        index: usize,
        inputs: &[&Surface],
        output: &Surface,
    ) -> crate::Result<Binding> {
        let api = api()?;
        let procedure = self
            .procedures
            .get(index)
            .ok_or_else(|| ane_error("ANE bind: no such procedure"))?;
        if inputs.len() != procedure.inputs.len() {
            return Err(ane_error("ANE bind: input count mismatch"));
        }
        autoreleasepool(|_| {
            let wrap = |surface: &Surface| -> crate::Result<Retained<AnyObject>> {
                catch("_ANEIOSurfaceObject objectWithIOSurface:", || {
                    let object: Option<Retained<AnyObject>> =
                        unsafe { msg_send![api.surface, objectWithIOSurface: surface.raw()] };
                    object
                })?
                .ok_or_else(|| ane_error("_ANEIOSurfaceObject returned nil"))
            };
            let objects = inputs
                .iter()
                .map(|surface| wrap(surface))
                .collect::<crate::Result<Vec<_>>>()?;
            let indices = procedure
                .input_symbols
                .iter()
                .map(|&symbol| NSNumber::new_u32(symbol))
                .collect::<Vec<_>>();
            Ok(Binding {
                inputs: Shared(NSArray::from_retained_slice(&objects)),
                input_indices: Shared(NSArray::from_retained_slice(&indices)),
                outputs: Shared(NSArray::from_retained_slice(&[wrap(output)?])),
                output_indices: Shared(NSArray::from_retained_slice(&[NSNumber::new_u32(
                    procedure.output_symbol,
                )])),
                procedure: Shared(NSNumber::new_usize(index)),
                program: Arc::clone(&self.inner),
            })
        })
    }
}

pub struct Binding {
    inputs: Shared<NSArray<AnyObject>>,
    input_indices: Shared<NSArray<NSNumber>>,
    outputs: Shared<NSArray<AnyObject>>,
    output_indices: Shared<NSArray<NSNumber>>,
    procedure: Shared<NSNumber>,
    program: Arc<ProgramInner>,
}

impl Binding {
    #[expect(unsafe_code)]
    fn request(&self, api: &Api, events: Option<&AnyObject>) -> crate::Result<Retained<AnyObject>> {
        let none: Option<&AnyObject> = None;
        catch("_ANERequest creation", || {
            let request: Option<Retained<AnyObject>> = unsafe {
                msg_send![
                    api.request,
                    requestWithInputs: &*self.inputs.0,
                    inputIndices: &*self.input_indices.0,
                    outputs: &*self.outputs.0,
                    outputIndices: &*self.output_indices.0,
                    weightsBuffer: none,
                    perfStats: none,
                    procedureIndex: &*self.procedure.0,
                    sharedEvents: events,
                    transactionHandle: none
                ]
            };
            request
        })?
        .ok_or_else(|| ane_error("ANE request creation failed"))
    }

    #[expect(unsafe_code)]
    pub fn evaluate(&self) -> crate::Result<Duration> {
        let api = api()?;
        autoreleasepool(|_| {
            let request = self.request(api, None)?;
            let empty = NSDictionary::<NSString, AnyObject>::new();
            let started = Instant::now();
            catch("ANE evaluate", || {
                let result: Result<(), Retained<NSError>> = unsafe {
                    msg_send![&*api.client.0, evaluateWithModel: &*self.program.model.0, options: &*empty, request: &*request, qos: QOS, error: _]
                };
                result
            })?
            .map_err(|error| ane_error(format!("ANE evaluate failed: {}", error_text(&error))))?;
            Ok(started.elapsed())
        })
    }

    #[expect(unsafe_code)]
    fn enqueue(
        &self,
        wait_event: &ProtocolObject<dyn MTLSharedEvent>,
        wait: u64,
        signal_event: &ProtocolObject<dyn MTLSharedEvent>,
        signal: u64,
        done: Box<dyn FnOnce(bool) + Send>,
    ) -> crate::Result<()> {
        let api = api()?;
        let wait_object: &AnyObject = wait_event.as_ref();
        let signal_object: &AnyObject = signal_event.as_ref();
        let port = sel!(eventPort);
        if !wait_object.class().responds_to(port) || !signal_object.class().responds_to(port) {
            return Err(ane_error(
                "the Metal shared event has no eventPort (Metal API validation on?)",
            ));
        }
        autoreleasepool(|_| {
            let signal_entry = catch("_ANESharedSignalEvent", || {
                let entry: Option<Retained<AnyObject>> = unsafe {
                    msg_send![api.signal_event, signalEventWithValue: signal, symbolIndex: 0_u32, eventType: 0_i64, sharedEvent: signal_object]
                };
                entry
            })?
            .ok_or_else(|| ane_error("ANE signal event creation failed"))?;
            let wait_entry = catch("_ANESharedWaitEvent", || {
                let entry: Option<Retained<AnyObject>> = unsafe {
                    msg_send![api.wait_event, waitEventWithValue: wait, sharedEvent: wait_object, eventType: 0_u64]
                };
                entry
            })?
            .ok_or_else(|| ane_error("ANE wait event creation failed"))?;
            let signals = NSArray::from_retained_slice(&[signal_entry]);
            let waits = NSArray::from_retained_slice(&[wait_entry]);
            let events = catch("_ANESharedEvents", || {
                let events: Option<Retained<AnyObject>> = unsafe {
                    msg_send![api.events, sharedEventsWithSignalEvents: &*signals, waitEvents: &*waits]
                };
                events
            })?
            .ok_or_else(|| ane_error("ANE shared events creation failed"))?;
            let request = self.request(api, Some(&events))?;
            let done = Mutex::new(Some(done));
            let handler = RcBlock::new(move |success: Bool, _error: *mut AnyObject| {
                let once = lock(&done).take();
                if let Some(done) = once {
                    done(success.as_bool());
                }
            });
            catch("ANE setCompletionHandler:", || {
                let (): () = unsafe { msg_send![&*request, setCompletionHandler: &*handler] };
            })?;
            let empty = NSDictionary::<NSString, AnyObject>::new();
            catch("ANE evaluate (async)", || {
                let result: Result<(), Retained<NSError>> = unsafe {
                    msg_send![&*api.client.0, evaluateWithModel: &*self.program.model.0, options: &*empty, request: &*request, qos: QOS, error: _]
                };
                result
            })?
            .map_err(|error| {
                ane_error(format!("ANE evaluate (async) failed: {}", error_text(&error)))
            })
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    Started(Instant),
    Succeeded,
    Failed,
}

struct HandoffInner {
    ready_value: u64,
    done_value: u64,
    jobs: HashMap<u64, Phase>,
    next_job: u64,
    outstanding: usize,
    retired: Option<String>,
    stop: bool,
    deadline: Option<Instant>,
}

struct HandoffState {
    ready: Shared<ProtocolObject<dyn MTLSharedEvent>>,
    done: Shared<ProtocolObject<dyn MTLSharedEvent>>,
    listener: Shared<MTLSharedEventListener>,
    bound: Duration,
    inner: Mutex<HandoffInner>,
    changed: Condvar,
}

impl HandoffState {
    fn retire_locked(&self, inner: &mut HandoffInner, why: &str) {
        if inner.retired.is_none() {
            inner.retired = Some(why.to_owned());
        }
        if self.ready.0.signaledValue() < inner.ready_value {
            self.ready.0.setSignaledValue(inner.ready_value);
        }
        if self.done.0.signaledValue() < inner.done_value {
            self.done.0.setSignaledValue(inner.done_value);
        }
        self.changed.notify_all();
    }

    fn report(&self, job: u64, ok: bool) {
        let mut inner = lock(&self.inner);
        inner.outstanding = inner.outstanding.saturating_sub(1);
        let phase = inner.jobs.get(&job).copied();
        if matches!(phase, Some(Phase::Queued | Phase::Started(_))) {
            if ok {
                inner.jobs.insert(job, Phase::Succeeded);
            } else {
                inner.jobs.insert(job, Phase::Failed);
                self.retire_locked(&mut inner, "a Neural Engine evaluation failed");
            }
        }
        drop(inner);
        self.changed.notify_all();
    }

    fn watch(&self) {
        let mut inner = lock(&self.inner);
        loop {
            if inner.stop {
                return;
            }
            let now = Instant::now();
            let mut earliest: Option<Instant> = None;
            let mut expired = false;
            for phase in inner.jobs.values_mut() {
                if let Phase::Started(started) = *phase {
                    let deadline = started + self.bound;
                    if deadline <= now {
                        *phase = Phase::Failed;
                        expired = true;
                    } else {
                        earliest = Some(earliest.map_or(deadline, |e| e.min(deadline)));
                    }
                }
            }
            if expired {
                let why = format!(
                    "a Neural Engine evaluation did not complete within {} ms",
                    self.bound.as_millis()
                );
                self.retire_locked(&mut inner, &why);
            }
            match inner.deadline {
                Some(deadline) if deadline <= now => {
                    inner.deadline = None;
                    self.retire_locked(&mut inner, "a Neural Engine chunk passed its deadline");
                }
                Some(deadline) => earliest = Some(earliest.map_or(deadline, |e| e.min(deadline))),
                None => {}
            }
            inner = match earliest {
                Some(deadline) => {
                    self.changed
                        .wait_timeout(inner, deadline.saturating_duration_since(now))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
                None => self
                    .changed
                    .wait(inner)
                    .unwrap_or_else(PoisonError::into_inner),
            };
        }
    }
}

pub struct Handoff {
    state: Arc<HandoffState>,
    watchdog: Option<std::thread::JoinHandle<()>>,
}

impl Handoff {
    pub fn new(context: &MetalContext, bound: Duration) -> crate::Result<Self> {
        let event = || {
            context
                .device()
                .newSharedEvent()
                .map(Shared)
                .ok_or_else(|| ane_error("newSharedEvent returned nil"))
        };
        let ready = event()?;
        let done = event()?;
        let inner = HandoffInner {
            ready_value: ready.0.signaledValue(),
            done_value: done.0.signaledValue(),
            jobs: HashMap::new(),
            next_job: 0,
            outstanding: 0,
            retired: None,
            stop: false,
            deadline: None,
        };
        let state = Arc::new(HandoffState {
            ready,
            done,
            listener: Shared(MTLSharedEventListener::sharedListener()),
            bound,
            inner: Mutex::new(inner),
            changed: Condvar::new(),
        });
        let watched = Arc::clone(&state);
        let watchdog = std::thread::Builder::new()
            .name("ane-watchdog".into())
            .spawn(move || watched.watch())
            .map_err(|error| ane_error(format!("watchdog thread: {error}")))?;
        Ok(Self {
            state,
            watchdog: Some(watchdog),
        })
    }

    #[must_use]
    pub fn ready_event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.state.ready.0
    }

    #[must_use]
    pub fn done_event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.state.done.0
    }

    #[must_use]
    pub fn next_ready(&self) -> u64 {
        let mut inner = lock(&self.state.inner);
        inner.ready_value += 1;
        if inner.retired.is_some() && self.state.ready.0.signaledValue() < inner.ready_value {
            self.state.ready.0.setSignaledValue(inner.ready_value);
        }
        inner.ready_value
    }

    #[must_use]
    pub fn next_done(&self) -> u64 {
        let mut inner = lock(&self.state.inner);
        inner.done_value += 1;
        if inner.retired.is_some() && self.state.done.0.signaledValue() < inner.done_value {
            self.state.done.0.setSignaledValue(inner.done_value);
        }
        inner.done_value
    }

    #[expect(unsafe_code)]
    pub fn queue(&self, binding: &Binding, ready: u64, done: u64) {
        let state = &self.state;
        let job = {
            let mut inner = lock(&state.inner);
            inner.next_job += 1;
            let job = inner.next_job;
            if inner.retired.is_some() {
                inner.jobs.insert(job, Phase::Failed);
                return;
            }
            inner.jobs.insert(job, Phase::Queued);
            inner.outstanding += 1;
            job
        };
        let reporter = Arc::clone(state);
        let started = binding.enqueue(
            &state.ready.0,
            ready,
            &state.done.0,
            done,
            Box::new(move |ok| reporter.report(job, ok)),
        );
        if let Err(error) = started {
            let mut inner = lock(&state.inner);
            inner.outstanding = inner.outstanding.saturating_sub(1);
            inner.jobs.insert(job, Phase::Failed);
            let why = format!("a Neural Engine evaluation could not start: {error}");
            state.retire_locked(&mut inner, &why);
            drop(inner);
            return;
        }
        let watched = Arc::clone(state);
        let notify = RcBlock::new(
            move |_event: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _value: u64| {
                let mut inner = lock(&watched.inner);
                let Some(phase @ Phase::Queued) = inner.jobs.get_mut(&job) else {
                    return;
                };
                *phase = Phase::Started(Instant::now());
                drop(inner);
                watched.changed.notify_all();
            },
        );
        unsafe {
            state.ready.0.notifyListener_atValue_block(
                &state.listener.0,
                ready,
                RcBlock::as_ptr(&notify),
            );
        }
    }

    pub fn finish(&self) -> bool {
        let state = &self.state;
        let mut inner = lock(&state.inner);
        inner.deadline = None;
        let settled = |inner: &HandoffInner| {
            inner.jobs.values().any(|phase| *phase == Phase::Failed)
                || inner.jobs.values().all(|phase| *phase == Phase::Succeeded)
        };
        let deadline = Instant::now() + state.bound;
        while !settled(&inner) {
            let now = Instant::now();
            if now >= deadline {
                let why = format!(
                    "a Neural Engine evaluation did not report within {} ms of its command",
                    state.bound.as_millis()
                );
                state.retire_locked(&mut inner, &why);
                break;
            }
            inner = state
                .changed
                .wait_timeout(inner, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        let ok =
            inner.retired.is_none() && inner.jobs.values().all(|phase| *phase == Phase::Succeeded);
        inner.jobs.clear();
        ok
    }

    pub fn retire(&self, why: &str) {
        let mut inner = lock(&self.state.inner);
        self.state.retire_locked(&mut inner, why);
    }

    #[must_use]
    pub fn retired(&self) -> Option<String> {
        lock(&self.state.inner).retired.clone()
    }
}

impl Drop for Handoff {
    fn drop(&mut self) {
        let state = &self.state;
        let mut inner = lock(&state.inner);
        state.retire_locked(&mut inner, "the handoff ended");
        let deadline = Instant::now() + state.bound;
        while inner.outstanding > 0 {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            inner = state
                .changed
                .wait_timeout(inner, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        inner.stop = true;
        state.changed.notify_all();
        drop(inner);
        if let Some(watchdog) = self.watchdog.take() {
            let _ = watchdog.join();
        }
    }
}

// ============================================================================
// Metal Pack / Stage / Join Kernels (`shaders/ane.metal`)
// ============================================================================

#[derive(Clone, Copy)]
#[repr(C)]
struct ScaleParams {
    tokens: u32,
    width: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PackParams {
    tokens: u32,
    width: u32,
    stride: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct StageInt4Params {
    rows: u32,
    columns: u32,
    stride: u32,
    scale_stride: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct JoinParams {
    tokens: u32,
    width: u32,
    ystride: u32,
    scale: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct JoinLinearParams {
    tokens: u32,
    rows: u32,
    ystride: u32,
    mode: u32,
}

fn bytes<T: Copy>(encoder: &ComputeEncoder, value: &T, index: usize) {
    encoder.set_val(value, index);
}

fn bind_buf(encoder: &ComputeEncoder, buffer: &MetalBuffer, offset: usize, index: usize) {
    encoder.bind(buffer, offset, index);
}

fn dispatch(encoder: &ComputeEncoder, groups: (usize, usize), threads: (usize, usize)) {
    encoder.dispatch_threadgroups(
        MTLSize {
            width: groups.0,
            height: groups.1,
            depth: 1,
        },
        MTLSize {
            width: threads.0,
            height: threads.1,
            depth: 1,
        },
    );
}

pub struct AneKernels {
    token_scales: Pipeline,
    pack: Pipeline,
    pack_half: Pipeline,
    stage_int4: Pipeline,
    stage_int4_half: Pipeline,
    join: Pipeline,
    stage_f32: Pipeline,
    join_linear: Pipeline,
    stage_bias: Pipeline,
}

impl AneKernels {
    pub fn new(shaders: &ShaderLibrary) -> crate::Result<Self> {
        let pipeline = |name: &str| shaders.pipeline(name);
        Ok(Self {
            token_scales: pipeline("mc_ane_token_scales")?,
            pack: pipeline("mc_ane_pack")?,
            pack_half: pipeline("mc_ane_pack_half")?,
            stage_int4: pipeline("mc_ane_stage_int4")?,
            stage_int4_half: pipeline("mc_ane_stage_int4_half")?,
            join: pipeline("mc_ane_join")?,
            stage_f32: pipeline("mc_ane_stage_f32")?,
            join_linear: pipeline("mc_ane_join_linear")?,
            stage_bias: pipeline("mc_ane_stage_bias")?,
        })
    }

    /// Pack `tokens` F32 rows of `input` (`width` wide) into channel-major int8 `output` (`[width, rows]`)
    /// and per-token fp16 `scales` (`[1, rows]`).
    pub fn pack(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        width: u32,
        tokens: u32,
        rows: u32,
        scales: &Surface,
        output: &Surface,
    ) -> crate::Result<()> {
        let stride = output.layout().stride;
        if tokens == 0
            || tokens > rows
            || !rows.is_multiple_of(32)
            || !width.is_multiple_of(32)
            || !scales.layout().fp16
            || output.layout().fp16
            || output.layout().rows != width
            || output.layout().width < rows
        {
            return Err(Error::InvalidArgument("ANE pack surface mismatch".into()));
        }
        let enc = batch.encoder();
        enc.set_pipeline(&self.token_scales);
        let sp = ScaleParams { tokens, width };
        bind_buf(enc, input, 0, 0);
        bind_buf(enc, scales.buffer(), 0, 1);
        bytes(enc, &sp, 2);
        dispatch(enc, (rows as usize, 1), (256, 1));
        batch.record_dispatch();

        let enc = batch.encoder();
        enc.set_pipeline(&self.pack);
        let pp = PackParams {
            tokens,
            width,
            stride: stride as u32,
        };
        bind_buf(enc, input, 0, 0);
        bind_buf(enc, scales.buffer(), 0, 1);
        bind_buf(enc, output.buffer(), 0, 2);
        bytes(enc, &pp, 3);
        dispatch(enc, ((rows / 32) as usize, (width / 32) as usize), (32, 8));
        batch.record_dispatch();
        Ok(())
    }

    /// Pack `tokens` F32 rows of `input` (`width` wide) into channel-major fp16 `output` (`[width, rows]`).
    pub fn pack_half(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        width: u32,
        tokens: u32,
        rows: u32,
        output: &Surface,
    ) -> crate::Result<()> {
        let layout = output.layout();
        if tokens == 0
            || tokens > rows
            || !rows.is_multiple_of(32)
            || !width.is_multiple_of(32)
            || !layout.fp16
            || layout.rows != width
            || layout.width < rows
        {
            return Err(Error::InvalidArgument(
                "ANE pack_half surface mismatch".into(),
            ));
        }
        let params = PackParams {
            tokens,
            width,
            stride: (layout.stride / 2) as u32,
        };
        let enc = batch.encoder();
        enc.set_pipeline(&self.pack_half);
        bind_buf(enc, input, 0, 0);
        bind_buf(enc, output.buffer(), 0, 1);
        bytes(enc, &params, 2);
        dispatch(enc, ((rows / 32) as usize, (width / 32) as usize), (32, 8));
        batch.record_dispatch();
        Ok(())
    }

    /// Stage an `INT4_G64` matrix into per-row int8 `output` (`[rows, columns]`) and fp16 `scales` (`[rows, 1]`).
    pub fn stage_int4(
        &self,
        batch: &mut CommandBatch,
        matrix: Int4Matrix<'_>,
        output: &Surface,
        scales: &Surface,
    ) -> crate::Result<()> {
        let layout = output.layout();
        if layout.fp16
            || layout.rows < matrix.rows
            || layout.width != matrix.columns
            || !scales.layout().fp16
            || scales.layout().rows < matrix.rows
        {
            return Err(Error::InvalidArgument(
                "ANE stage_int4 surface mismatch".into(),
            ));
        }
        let params = StageInt4Params {
            rows: matrix.rows,
            columns: matrix.columns,
            stride: layout.stride as u32,
            scale_stride: (scales.layout().stride / 2) as u32,
        };
        let enc = batch.encoder();
        enc.set_pipeline(&self.stage_int4);
        bind_buf(enc, matrix.codes, matrix.codes_offset, 0);
        bind_buf(enc, matrix.groups, matrix.groups_offset, 1);
        bind_buf(enc, output.buffer(), 0, 2);
        bind_buf(enc, scales.buffer(), 0, 3);
        bytes(enc, &params, 4);
        dispatch(enc, (matrix.rows.div_ceil(8) as usize, 1), (256, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// Stage an `INT4_G64` matrix (`columns` a multiple of 128) into fp16 `output` (`[rows, columns]`)
    /// holding its dequantized values.
    pub fn stage_int4_half(
        &self,
        batch: &mut CommandBatch,
        matrix: Int4Matrix<'_>,
        output: &Surface,
    ) -> crate::Result<()> {
        let layout = output.layout();
        if !layout.fp16
            || layout.rows < matrix.rows
            || layout.width != matrix.columns
            || !matrix.columns.is_multiple_of(128)
        {
            return Err(Error::InvalidArgument(
                "ANE stage_int4_half surface mismatch".into(),
            ));
        }
        let params = StageInt4Params {
            rows: matrix.rows,
            columns: matrix.columns,
            stride: (layout.stride / 2) as u32,
            scale_stride: 0,
        };
        let enc = batch.encoder();
        enc.set_pipeline(&self.stage_int4_half);
        bind_buf(enc, matrix.codes, matrix.codes_offset, 0);
        bind_buf(enc, matrix.groups, matrix.groups_offset, 1);
        bind_buf(enc, output.buffer(), 0, 2);
        bytes(enc, &params, 3);
        dispatch(enc, (matrix.rows.div_ceil(8) as usize, 1), (256, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// Stage an `F32` matrix `[rows, columns]` into per-row int8 `output` and fp16 `scales`.
    pub fn stage_f32(
        &self,
        batch: &mut CommandBatch,
        weights: (&MetalBuffer, usize),
        rows: u32,
        columns: u32,
        output: &Surface,
        scales: &Surface,
    ) -> crate::Result<()> {
        let layout = output.layout();
        if layout.fp16
            || layout.rows < rows
            || layout.width != columns
            || !scales.layout().fp16
            || scales.layout().rows < rows
        {
            return Err(Error::InvalidArgument(
                "ANE stage_f32 surface mismatch".into(),
            ));
        }
        let params = StageInt4Params {
            rows,
            columns,
            stride: layout.stride as u32,
            scale_stride: (scales.layout().stride / 2) as u32,
        };
        let enc = batch.encoder();
        enc.set_pipeline(&self.stage_f32);
        bind_buf(enc, weights.0, weights.1, 0);
        bind_buf(enc, output.buffer(), 0, 1);
        bind_buf(enc, scales.buffer(), 0, 2);
        bytes(enc, &params, 3);
        dispatch(enc, (rows.div_ceil(8) as usize, 1), (256, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// Stage an `F32` bias vector `[rows]` into an `FP16` surface `output` (`[rows, 1]`).
    pub fn stage_bias(
        &self,
        batch: &mut CommandBatch,
        bias: (&MetalBuffer, usize),
        rows: u32,
        output: &Surface,
    ) -> crate::Result<()> {
        let layout = output.layout();
        if !layout.fp16 || layout.rows < rows {
            return Err(Error::InvalidArgument(
                "ANE stage_bias surface mismatch".into(),
            ));
        }
        let params = [rows, (layout.stride / 2) as u32];
        let enc = batch.encoder();
        enc.set_pipeline(&self.stage_bias);
        bind_buf(enc, bias.0, bias.1, 0);
        bind_buf(enc, output.buffer(), 0, 1);
        bytes(enc, &params, 2);
        dispatch(enc, (rows.div_ceil(256) as usize, 1), (256, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// Join ANE linear output `partial` (`[rows, grid_tokens]`) + `bias` (`[rows]`) into `output` (`[tokens, rows]`).
    pub fn join_linear(
        &self,
        batch: &mut CommandBatch,
        partial: &Surface,
        scales: &Surface,
        bias: (&MetalBuffer, usize),
        output: &MetalBuffer,
        status: &MetalBuffer,
        rows: u32,
        tokens: u32,
        mode: u32,
    ) -> crate::Result<()> {
        let layout = partial.layout();
        if tokens == 0
            || !layout.fp16
            || layout.rows < rows
            || output.length() < tokens as usize * rows as usize * size_of::<f32>()
            || status.length() < size_of::<u32>()
        {
            return Err(Error::InvalidArgument(
                "ANE join_linear buffer mismatch".into(),
            ));
        }
        let params = JoinLinearParams {
            tokens,
            rows,
            ystride: (layout.stride / 2) as u32,
            mode,
        };
        let enc = batch.encoder();
        enc.set_pipeline(&self.join_linear);
        bind_buf(enc, partial.buffer(), 0, 0);
        bind_buf(enc, scales.buffer(), 0, 1);
        bind_buf(enc, bias.0, bias.1, 2);
        bind_buf(enc, output, 0, 3);
        bind_buf(enc, status, 0, 4);
        bytes(enc, &params, 5);
        dispatch(
            enc,
            (tokens.div_ceil(32) as usize, rows.div_ceil(32) as usize),
            (32, 8),
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Join ANE FFN output `partial` (`[width, rows]`) times `scale` into `output` (`[tokens, width]`).
    pub fn join(
        &self,
        batch: &mut CommandBatch,
        partial: &Surface,
        output: &MetalBuffer,
        status: &MetalBuffer,
        width: u32,
        tokens: u32,
        scale: f32,
    ) -> crate::Result<()> {
        let layout = partial.layout();
        if tokens == 0
            || !width.is_multiple_of(32)
            || !layout.fp16
            || layout.rows != width
            || output.length() < tokens as usize * width as usize * size_of::<f32>()
            || status.length() < size_of::<u32>()
        {
            return Err(Error::InvalidArgument("ANE join buffer mismatch".into()));
        }
        let params = JoinParams {
            tokens,
            width,
            ystride: (layout.stride / 2) as u32,
            scale,
        };
        let enc = batch.encoder();
        enc.set_pipeline(&self.join);
        bind_buf(enc, partial.buffer(), 0, 0);
        bind_buf(enc, output, 0, 1);
        bind_buf(enc, status, 0, 2);
        bytes(enc, &params, 3);
        dispatch(
            enc,
            (tokens.div_ceil(32) as usize, (width / 32) as usize),
            (32, 8),
        );
        batch.record_dispatch();
        Ok(())
    }
}
