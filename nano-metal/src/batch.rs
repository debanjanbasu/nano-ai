use std::cell::RefCell;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use block2::RcBlock;
use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4CommitFeedback, MTL4CommitOptions,
    MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLComputePipelineState, MTLDevice,
    MTLSharedEvent, MTLSize, MTLStages,
};

use crate::Error;
use crate::buffer::MetalBuffer;
use crate::context::{self, MetalContext};

/// Highest `[[buffer(n)]]` index any kernel uses, plus one.
const MAX_BUFFER_BINDINGS: usize = 16;
const UPLOAD_CHUNK: usize = 64 * 1024;
/// Constants are read through `constant T&`; 16 bytes covers every vector type.
const UPLOAD_ALIGN: usize = 16;
const STAGES: MTLStages = MTLStages::Dispatch.union(MTLStages::Blit);

/// Reusable per-batch resources; recycled once every command buffer that used them completed.
pub(crate) struct Frame {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    cmd_bufs: Vec<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    upload: Upload,
}

// A frame is only touched by the batch that owns it or, once idle, through the pool's mutex.
#[expect(unsafe_code, clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Frame {}

impl Frame {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> crate::Result<Self> {
        let allocator = device
            .newCommandAllocator()
            .ok_or_else(|| Error::CommandBuffer("Failed to create command allocator".into()))?;
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(MAX_BUFFER_BINDINGS);
        let table = device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|e| Error::CommandBuffer(format!("argument table: {e}")))?;
        Ok(Self {
            allocator,
            cmd_bufs: Vec::new(),
            table,
            upload: Upload::default(),
        })
    }
}

/// Bump allocator for the small constants `setBytes` used to carry.
#[derive(Default)]
struct Upload {
    chunks: Vec<MetalBuffer>,
    chunk: usize,
    offset: usize,
}

impl Upload {
    /// Returns the GPU address and CPU pointer of `size` fresh bytes.
    fn alloc(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        size: usize,
    ) -> crate::Result<(u64, *mut u8)> {
        let start = self.offset.next_multiple_of(UPLOAD_ALIGN);
        if self.chunk < self.chunks.len() && start + size <= UPLOAD_CHUNK {
            self.offset = start + size;
        } else {
            // Only advance once the chunk exists, so a failed allocation leaves `chunk` valid.
            let next = if self.chunks.is_empty() { 0 } else { self.chunk + 1 };
            if next == self.chunks.len() {
                self.chunks
                    .push(MetalBuffer::empty(device, UPLOAD_CHUNK.max(size))?);
            }
            self.chunk = next;
            self.offset = size;
        }
        let chunk = &self.chunks[self.chunk];
        let at = self.offset - size;
        #[expect(unsafe_code)]
        let cpu = unsafe { chunk.contents_mut_ptr().add(at) };
        Ok((chunk.gpu_address() + at as u64, cpu))
    }
}

/// A Metal 4 compute encoder plus the argument table and upload arena its dispatches bind through.
pub struct ComputeEncoder {
    raw: Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    upload: RefCell<Upload>,
    /// First upload allocation failure; reported (and the batch not committed) at commit.
    failed: RefCell<Option<Error>>,
}

impl ComputeEncoder {
    pub fn set_pipeline(&self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.raw.setComputePipelineState(pipeline);
    }

    /// Bind `buffer` at byte `offset` to `[[buffer(index)]]`.
    #[expect(unsafe_code)]
    pub fn bind(&self, buffer: &MetalBuffer, offset: usize, index: usize) {
        // `MetalBuffer`s are always resident, and `index` is below the table's bind count.
        unsafe {
            self.table
                .setAddress_atIndex(buffer.gpu_address() + offset as u64, index);
        }
    }

    /// Copy `value` into the batch's upload arena and bind it to `[[buffer(index)]]`.
    #[expect(unsafe_code)]
    pub fn set_val<T: Copy>(&self, value: &T, index: usize) {
        let size = size_of::<T>();
        match self.upload.borrow_mut().alloc(&self.device, size) {
            Ok((gpu, cpu)) => unsafe {
                std::ptr::copy_nonoverlapping(std::ptr::from_ref(value).cast::<u8>(), cpu, size);
                self.table.setAddress_atIndex(gpu, index);
            },
            Err(error) => {
                self.failed.borrow_mut().get_or_insert(error);
            }
        }
    }

    #[expect(unsafe_code)]
    pub fn set_threadgroup_memory(&self, length: usize, index: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) };
    }

    pub fn dispatch_threadgroups(&self, groups: MTLSize, threads: MTLSize) {
        self.raw
            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    }

    fn barrier(&self) {
        self.raw
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                STAGES,
                STAGES,
                MTL4VisibilityOptions::Device,
            );
    }
}

pub struct BufferCopyRequest<'a> {
    pub source: &'a MetalBuffer,
    pub source_offset: usize,
    pub destination: &'a MetalBuffer,
    pub destination_offset: usize,
    pub size: usize,
}

/// Records dispatches into Metal 4 command buffers.
///
/// A serial batch places a barrier after every dispatch; a concurrent one only between
/// dispatches outside `independent` groups and after each group. Every command buffer waits
/// for all earlier work on the queue.
pub struct CommandBatch {
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    cmd_bufs: Vec<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>,
    /// Index into `cmd_bufs` of the command buffer being encoded.
    current: usize,
    /// Residency serial of the command buffer being encoded.
    serial: u64,
    encoder: ComputeEncoder,
    completion: Arc<Completion>,
    sealed: bool,
    dispatch_count: u32,
    concurrent: bool,
    independent: bool,
}

impl CommandBatch {
    pub fn new(ctx: &MetalContext) -> crate::Result<Self> {
        Self::with_dispatch(ctx, false)
    }

    pub fn new_concurrent(ctx: &MetalContext) -> crate::Result<Self> {
        Self::with_dispatch(ctx, true)
    }

    fn with_dispatch(ctx: &MetalContext, concurrent: bool) -> crate::Result<Self> {
        let pool = Arc::clone(ctx.frames());
        let recycled = lock(&pool).pop();
        let mut frame = match recycled {
            Some(frame) => {
                frame.allocator.reset();
                frame
            }
            None => Frame::new(ctx.device())?,
        };
        frame.upload.chunk = 0;
        frame.upload.offset = 0;
        if frame.cmd_bufs.is_empty() {
            frame.cmd_bufs.push(new_command_buffer(ctx.device())?);
        }
        let cmd_buf = frame.cmd_bufs[0].clone();
        let serial = context::begin_segment();
        let raw = match begin(&cmd_buf, &frame.allocator, &frame.table) {
            Ok(raw) => raw,
            Err(error) => {
                context::end_segment(serial);
                return Err(error);
            }
        };
        let completion = Arc::new(Completion {
            state: Mutex::new(CompletionState {
                pending: 0,
                sealed: false,
                finished: false,
                frame: None,
                error: None,
                waker: None,
                pool,
            }),
            done: Condvar::new(),
        });
        Ok(Self {
            queue: ctx.queue().clone(),
            allocator: frame.allocator,
            cmd_bufs: frame.cmd_bufs,
            current: 0,
            serial,
            encoder: ComputeEncoder {
                raw,
                table: frame.table,
                device: ctx.device().retain(),
                upload: RefCell::new(frame.upload),
                failed: RefCell::new(None),
            },
            completion,
            sealed: false,
            dispatch_count: 0,
            concurrent,
            independent: false,
        })
    }

    #[must_use]
    pub const fn encoder(&self) -> &ComputeEncoder {
        &self.encoder
    }

    pub fn record_dispatch(&mut self) {
        self.dispatch_count += 1;
        if !self.concurrent || !self.independent {
            self.encoder.barrier();
        }
    }

    pub fn independent<T, E>(
        &mut self,
        encode: impl FnOnce(&mut Self) -> Result<T, E>,
    ) -> Result<T, E> {
        let nested = std::mem::replace(&mut self.independent, true);
        let result = encode(self);
        self.independent = nested;
        if self.concurrent && !nested {
            self.encoder.barrier();
        }
        result
    }

    #[must_use]
    pub const fn dispatch_count(&self) -> u32 {
        self.dispatch_count
    }

    /// Whether dispatches recorded now may overlap (inside an `independent` group of a
    /// concurrent batch).
    #[must_use]
    pub const fn overlapping(&self) -> bool {
        self.concurrent && self.independent
    }

    #[expect(unsafe_code)]
    pub fn blit_buffer_copies<'a, I>(&mut self, copies: I) -> crate::Result<()>
    where
        I: IntoIterator<Item = BufferCopyRequest<'a>>,
    {
        self.encoder.barrier();
        for copy in copies {
            if copy
                .source_offset
                .checked_add(copy.size)
                .is_none_or(|end| end > copy.source.length())
                || copy
                    .destination_offset
                    .checked_add(copy.size)
                    .is_none_or(|end| end > copy.destination.length())
            {
                return Err(Error::CommandBuffer("Buffer copy out of bounds".into()));
            }
            // Bounds checked above; both buffers are resident `MetalBuffer`s.
            unsafe {
                self.encoder
                    .raw
                    .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        copy.source.raw(),
                        copy.source_offset,
                        copy.destination.raw(),
                        copy.destination_offset,
                        copy.size,
                    );
            }
        }
        self.encoder.barrier();
        Ok(())
    }

    /// GPU work encoded so far signals `event` with `value` once it completes.
    pub fn signal_event(
        &mut self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> crate::Result<()> {
        self.split(|queue| queue.signalEvent_value(ProtocolObject::from_ref(event), value))
    }

    /// GPU work encoded from now on waits until `event` reaches `value`.
    pub fn wait_event(
        &mut self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> crate::Result<()> {
        self.split(|queue| queue.waitForEvent_value(ProtocolObject::from_ref(event), value))
    }

    /// Start GPU work on what is encoded so far and keep encoding into a fresh command buffer.
    pub fn submit_and_renew(&mut self) -> crate::Result<()> {
        self.dispatch_count = 0;
        self.split(|_| {})
    }

    /// Commit the open command buffer, run `between` on the queue, and open the next one.
    fn split(
        &mut self,
        between: impl FnOnce(&ProtocolObject<dyn MTL4CommandQueue>),
    ) -> crate::Result<()> {
        if self.sealed {
            return Err(Error::CommandBuffer("command batch already failed".into()));
        }
        if let Err(error) = self.commit_segment() {
            self.seal_failed(&error);
            return Err(error);
        }
        between(&self.queue);
        self.begin_segment()
    }

    pub fn commit_and_wait(mut self) -> crate::Result<()> {
        self.finish().wait()
    }

    /// Commit now and return a future that resolves when the GPU finishes, so CPU work between
    /// this call and `.await` overlaps GPU execution.
    #[must_use = "the GPU work runs regardless; await the future to observe completion and errors"]
    pub fn commit_and_await(mut self) -> GpuCompletionFuture {
        GpuCompletionFuture {
            completion: self.finish(),
        }
    }

    fn finish(&mut self) -> Arc<Completion> {
        if !self.sealed {
            if let Err(error) = self.commit_segment() {
                self.completion.fail(error);
            }
            self.seal();
        }
        Arc::clone(&self.completion)
    }

    /// Seal after an encoding failure, recording it so the commit reports it instead of `Ok`.
    fn seal_failed(&mut self, error: &Error) {
        self.completion.fail(Error::CommandBuffer(error.to_string()));
        self.seal();
    }

    /// Hand the frame to the completion, which recycles it once the GPU is done with it.
    fn seal(&mut self) {
        self.sealed = true;
        let frame = Frame {
            allocator: self.allocator.clone(),
            cmd_bufs: std::mem::take(&mut self.cmd_bufs),
            table: self.encoder.table.clone(),
            upload: std::mem::take(&mut *self.encoder.upload.borrow_mut()),
        };
        self.completion.seal(frame);
    }

    /// On failure the batch is sealed: nothing is open, and work already committed finishes.
    fn begin_segment(&mut self) -> crate::Result<()> {
        if self.current + 1 == self.cmd_bufs.len() {
            match new_command_buffer(&self.encoder.device) {
                Ok(cmd_buf) => self.cmd_bufs.push(cmd_buf),
                Err(error) => {
                    self.seal_failed(&error);
                    return Err(error);
                }
            }
        }
        let serial = context::begin_segment();
        match begin(
            &self.cmd_bufs[self.current + 1],
            &self.allocator,
            &self.encoder.table,
        ) {
            Ok(raw) => {
                self.current += 1;
                self.serial = serial;
                self.encoder.raw = raw;
                Ok(())
            }
            Err(error) => {
                context::end_segment(serial);
                self.seal_failed(&error);
                Err(error)
            }
        }
    }

    /// End the open command buffer and commit it, unless encoding failed.
    #[expect(unsafe_code)]
    fn commit_segment(&mut self) -> crate::Result<()> {
        self.encoder.raw.endEncoding();
        let cmd_buf = &self.cmd_bufs[self.current];
        cmd_buf.endCommandBuffer();
        if let Some(error) = self.encoder.failed.borrow_mut().take() {
            context::end_segment(self.serial);
            return Err(error);
        }
        context::flush();
        self.completion.lock().pending += 1;
        let completion = Arc::clone(&self.completion);
        let serial = self.serial;
        let handler = RcBlock::new(
            move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                // Metal passes a valid feedback object for the duration of the call.
                let error = unsafe { feedback.as_ref() }
                    .error()
                    .map(|e| Error::CommandBuffer(format!("GPU error: {e}")));
                context::end_segment(serial);
                completion.segment_done(error);
            },
        );
        let options = MTL4CommitOptions::new();
        let mut list = [NonNull::from(&**cmd_buf)];
        // `handler` is copied by Metal; `list` holds one live, ended command buffer.
        unsafe {
            options.addFeedbackHandler(RcBlock::as_ptr(&handler));
            self.queue
                .commit_count_options(NonNull::from(&mut list).cast(), 1, &options);
        }
        Ok(())
    }
}

impl Drop for CommandBatch {
    /// An uncommitted batch is discarded; work it already submitted is waited for.
    fn drop(&mut self) {
        if self.sealed {
            return;
        }
        self.encoder.raw.endEncoding();
        self.cmd_bufs[self.current].endCommandBuffer();
        context::end_segment(self.serial);
        self.seal();
        let _ = self.completion.wait();
    }
}

fn new_command_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
) -> crate::Result<Retained<ProtocolObject<dyn MTL4CommandBuffer>>> {
    device
        .newCommandBuffer()
        .ok_or_else(|| Error::CommandBuffer("Failed to create command buffer".into()))
}

fn begin(
    cmd_buf: &ProtocolObject<dyn MTL4CommandBuffer>,
    allocator: &ProtocolObject<dyn MTL4CommandAllocator>,
    table: &ProtocolObject<dyn MTL4ArgumentTable>,
) -> crate::Result<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>> {
    cmd_buf.beginCommandBufferWithAllocator(allocator);
    let Some(encoder) = cmd_buf.computeCommandEncoder() else {
        cmd_buf.endCommandBuffer();
        return Err(Error::CommandBuffer(
            "Failed to create compute encoder".into(),
        ));
    };
    encoder.setArgumentTable(Some(table));
    // Batches and their command buffers run in submission order.
    encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
        STAGES,
        STAGES,
        MTL4VisibilityOptions::Device,
    );
    Ok(encoder)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Completion {
    state: Mutex<CompletionState>,
    done: Condvar,
}

struct CompletionState {
    /// Committed command buffers whose feedback has not arrived.
    pending: usize,
    /// No more command buffers will be committed.
    sealed: bool,
    finished: bool,
    frame: Option<Frame>,
    error: Option<Error>,
    waker: Option<Waker>,
    pool: Arc<Mutex<Vec<Frame>>>,
}

impl Completion {
    fn lock(&self) -> MutexGuard<'_, CompletionState> {
        lock(&self.state)
    }

    fn fail(&self, error: Error) {
        self.lock().error.get_or_insert(error);
    }

    fn segment_done(&self, error: Option<Error>) {
        let mut state = self.lock();
        if let Some(error) = error {
            state.error.get_or_insert(error);
        }
        state.pending -= 1;
        self.finish_if_done(state);
    }

    fn seal(&self, frame: Frame) {
        let mut state = self.lock();
        state.sealed = true;
        state.frame = Some(frame);
        self.finish_if_done(state);
    }

    fn finish_if_done(&self, mut state: MutexGuard<'_, CompletionState>) {
        if !state.sealed || state.pending > 0 || state.finished {
            return;
        }
        state.finished = true;
        if let Some(frame) = state.frame.take() {
            lock(&state.pool).push(frame);
        }
        let waker = state.waker.take();
        drop(state);
        self.done.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn wait(&self) -> crate::Result<()> {
        let mut state = self.lock();
        while !state.finished {
            state = self
                .done
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.error.take().map_or(Ok(()), Err)
    }
}

pub struct GpuCompletionFuture {
    completion: Arc<Completion>,
}

impl Future for GpuCompletionFuture {
    type Output = crate::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.completion.lock();
        if state.finished {
            Poll::Ready(state.error.take().map_or(Ok(()), Err))
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
