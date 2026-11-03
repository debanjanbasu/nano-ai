use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4CommandQueue, MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResidencySet,
    MTLResidencySetDescriptor,
};

use crate::Error;
use crate::batch::Frame;

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    /// Allocators, command buffers, argument tables and upload arenas of completed batches.
    frames: Arc<Mutex<Vec<Frame>>>,
}

impl MetalContext {
    pub fn new() -> crate::Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoMetalDevice)?;
        let queue = device.newMTL4CommandQueue().ok_or(Error::NoCommandQueue)?;
        with_residency(&device, |r| queue.addResidencySet(&r.set))?;
        Ok(Self {
            device,
            queue,
            frames: Arc::default(),
        })
    }

    #[must_use]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    #[must_use]
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(crate) fn queue(&self) -> &Retained<ProtocolObject<dyn MTL4CommandQueue>> {
        &self.queue
    }

    pub(crate) const fn frames(&self) -> &Arc<Mutex<Vec<Frame>>> {
        &self.frames
    }
}

/// One residency set holds every live `MetalBuffer` (Metal 4 binds by GPU address, so nothing is
/// made resident implicitly). Dropped buffers leave the set only once every command buffer that
/// was being encoded or in flight when they dropped has completed.
pub(crate) struct Residency {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    dirty: bool,
    added: bool,
    next_serial: u64,
    outstanding: BTreeSet<u64>,
    retired: Vec<(u64, Retained<ProtocolObject<dyn MTLBuffer>>)>,
}

// Every access goes through `RESIDENCY`'s mutex.
#[expect(unsafe_code, clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Residency {}

static RESIDENCY: Mutex<Option<Residency>> = Mutex::new(None);

fn lock() -> MutexGuard<'static, Option<Residency>> {
    RESIDENCY.lock().unwrap_or_else(PoisonError::into_inner)
}

fn with_residency<T>(
    device: &ProtocolObject<dyn MTLDevice>,
    f: impl FnOnce(&mut Residency) -> T,
) -> crate::Result<T> {
    let mut guard = lock();
    if guard.is_none() {
        let set = device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|e| Error::CommandBuffer(format!("residency set: {e}")))?;
        set.requestResidency();
        *guard = Some(Residency {
            set,
            dirty: false,
            added: false,
            next_serial: 0,
            outstanding: BTreeSet::new(),
            retired: Vec::new(),
        });
    }
    guard
        .as_mut()
        .map(f)
        .ok_or_else(|| Error::CommandBuffer("residency set missing".into()))
}

/// Make a new buffer resident for every command buffer committed from now on.
pub(crate) fn track(
    device: &ProtocolObject<dyn MTLDevice>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
) -> crate::Result<()> {
    with_residency(device, |r| {
        r.set.addAllocation(ProtocolObject::from_ref(buffer));
        r.dirty = true;
        r.added = true;
    })
}

pub(crate) fn retire(buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
    let idle = lock().as_mut().is_some_and(|r| {
        r.retired.push((r.next_serial, buffer));
        r.outstanding.is_empty()
    });
    // No command buffer can reference it: release it now rather than at the next commit.
    if idle {
        flush();
    }
}

pub(crate) fn begin_segment() -> u64 {
    lock().as_mut().map_or(0, |r| {
        let serial = r.next_serial;
        r.next_serial += 1;
        r.outstanding.insert(serial);
        serial
    })
}

pub(crate) fn end_segment(serial: u64) {
    if let Some(r) = lock().as_mut() {
        r.outstanding.remove(&serial);
    }
}

/// Apply pending additions and the removals no in-flight command buffer can still reference.
pub(crate) fn flush() {
    if let Some(r) = lock().as_mut() {
        let oldest = r.outstanding.first().copied().unwrap_or(u64::MAX);
        let set = r.set.clone();
        let before = r.retired.len();
        r.retired.retain(|(tag, buffer)| {
            let keep = *tag > oldest;
            if !keep {
                set.removeAllocation(ProtocolObject::from_ref(&**buffer));
            }
            keep
        });
        if r.retired.len() != before {
            r.dirty = true;
        }
        if r.dirty {
            r.set.commit();
            r.dirty = false;
        }
        if r.added {
            r.set.requestResidency();
            r.added = false;
        }
    }
}
