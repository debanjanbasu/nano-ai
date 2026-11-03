use core::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::Error;

/// Safe wrapper around a Metal buffer, resident for its whole life (see `context::track`).
#[derive(Clone)]
pub struct MetalBuffer {
    inner: Arc<Tracked>,
}

struct Tracked(Retained<ProtocolObject<dyn MTLBuffer>>);

impl Drop for Tracked {
    fn drop(&mut self) {
        crate::context::retire(self.0.clone());
    }
}

impl std::ops::Deref for Tracked {
    type Target = ProtocolObject<dyn MTLBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl MetalBuffer {
    #[expect(unsafe_code)]
    pub fn from_slice<T: bytemuck::NoUninit>(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
    ) -> crate::Result<Self> {
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let len = bytes.len();
        let inner = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(bytes.as_ptr().cast_mut().cast::<c_void>()),
                len,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked,
            )
        }
        .ok_or_else(|| Error::BufferAllocation {
            size: len,
            reason: "newBufferWithBytes returned nil".to_owned(),
        })?;
        Self::track(device, inner)
    }

    pub fn empty(device: &ProtocolObject<dyn MTLDevice>, byte_len: usize) -> crate::Result<Self> {
        let inner = device
            .newBufferWithLength_options(
                byte_len,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked,
            )
            .ok_or_else(|| Error::BufferAllocation {
                size: byte_len,
                reason: "newBufferWithLength returned nil".to_owned(),
            })?;
        Self::track(device, inner)
    }

    /// Wrap page-aligned, externally-owned shared memory without copying it.
    ///
    /// # Safety
    ///
    /// `owner` must keep `pointer..pointer+byte_len` readable and page-backed
    /// for its lifetime, and `pointer` and `byte_len` must be 16 KiB page-aligned.
    #[expect(unsafe_code)]
    pub unsafe fn from_bytes_no_copy(
        device: &ProtocolObject<dyn MTLDevice>,
        pointer: NonNull<u8>,
        byte_len: usize,
        owner: Arc<dyn Send + Sync>,
    ) -> crate::Result<Self> {
        let raw = pointer.cast::<c_void>();
        let deallocator = RcBlock::new(move |_pointer: NonNull<c_void>, _length: usize| {
            let _ = &owner;
        });
        let inner = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                raw,
                byte_len,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked,
                Some(&deallocator),
            )
        }
        .ok_or_else(|| Error::BufferAllocation {
            size: byte_len,
            reason: "newBufferWithBytesNoCopy returned nil".to_owned(),
        })?;
        Self::track(device, inner)
    }

    fn track(
        device: &ProtocolObject<dyn MTLDevice>,
        inner: Retained<ProtocolObject<dyn MTLBuffer>>,
    ) -> crate::Result<Self> {
        crate::context::track(device, &inner)?;
        Ok(Self {
            // Atomic count: callers move buffers across threads behind their own Send wrappers.
            #[expect(clippy::arc_with_non_send_sync)]
            inner: Arc::new(Tracked(inner)),
        })
    }

    #[must_use]
    pub fn gpu_address(&self) -> u64 {
        self.inner.0.gpuAddress()
    }

    #[must_use]
    pub fn length(&self) -> usize {
        self.inner.length()
    }

    /// # Safety
    ///
    /// No concurrent GPU or CPU write may occur while the returned slice lives.
    #[must_use]
    #[expect(unsafe_code)]
    pub unsafe fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        let ptr = self.inner.contents().as_ptr().cast::<u8>();
        let len = self.inner.length();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        bytemuck::cast_slice(bytes)
    }

    /// # Safety
    ///
    /// Exclusive CPU/GPU access is required while the mutable slice lives.
    #[expect(unsafe_code)]
    pub unsafe fn as_mut_slice<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        let ptr = self.inner.contents().as_ptr().cast::<u8>();
        let len = self.inner.length();
        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        bytemuck::cast_slice_mut(bytes)
    }

    /// # Safety
    ///
    /// Valid for `self.length()` bytes while buffer lives.
    #[must_use]
    #[expect(unsafe_code)]
    pub unsafe fn contents_mut_ptr(&self) -> *mut u8 {
        self.inner.contents().as_ptr().cast::<u8>()
    }

    #[expect(unsafe_code)]
    pub fn clear(&self) {
        unsafe { std::ptr::write_bytes(self.contents_mut_ptr(), 0, self.length()) };
    }

    #[must_use]
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.inner.0
    }
}
