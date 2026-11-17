//! On-disk prompt cache: prefix checkpoints persisted next to the model, so long repeated prefixes
//! (system prompts, tool definitions) survive restarts and idle checkpoints live in evictable page
//! cache instead of RAM.
//!
//! Directory `<model dir>/.prompt-cache/<16 hex of payload SHA-256>-<kv format>/`, one file per
//! checkpoint named after the SHA-256 of its token ids (`u32` little-endian):
//! - `0..8`: magic `MAIPC001`; `8..16`: header length (`u64` LE); then the JSON [`Header`].
//! - Raw sections from the next 16 KiB boundary, each at a 16 KiB-aligned offset and padded to a
//!   16 KiB multiple, so a mapped file wraps zero-copy as Metal buffers.
//!
//! Files are written (temp file + rename) on a background thread. The directory stays under a
//! byte budget by deleting the least recently used files first (modification time, touched on use).

use std::fs::{self, File};
use std::io::{BufReader, Read, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::SystemTime;

use memmap2::Mmap;
use nano_metal::buffer::MetalBuffer;
use nano_metal::context::MetalContext;
use nano_metal::ops::KvFormat;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::artifact::METAL_PAGE;

const MAGIC: [u8; 8] = *b"MAIPC001";
/// Queue slots (touches and writes); the RAM queued writes pin is bounded by bytes separately.
const MAX_PENDING_WRITES: usize = 4;

#[derive(Serialize, Deserialize)]
pub struct Header {
    pub tokens: Vec<u32>,
    pub rope_delta: i32,
    pub dspark_scored_anchors: usize,
    pub kv_format: u32,
    pub pinned: bool,
    /// `[offset, length]` of each raw section, offsets relative to the first section.
    pub sections: Vec<[usize; 2]>,
}

/// A mapped checkpoint file: its header and one zero-copy buffer per section.
pub struct Loaded {
    pub header: Header,
    pub buffers: Vec<MetalBuffer>,
}

struct Entry {
    name: String,
    tokens: Vec<u32>,
    pinned: bool,
    bytes: u64,
    used: SystemTime,
}

/// A checkpoint section handed to the writer thread.
struct SendBuffer(MetalBuffer);
// SAFETY: `MTLBuffer` objects are thread-safe, and checkpoint buffers are never written after the
// blit that filled them completed, so the writer only ever reads settled memory.
#[expect(unsafe_code, clippy::non_send_fields_in_send_ty)]
unsafe impl Send for SendBuffer {}

enum Job {
    Write { name: String, header: Header, buffers: Vec<SendBuffer>, bytes: u64 },
    Touch(PathBuf),
}

pub struct DiskCache {
    root: PathBuf,
    key: String,
    budget: u64,
    dir: PathBuf,
    kv_format: KvFormat,
    entries: Arc<Mutex<Vec<Entry>>>,
    jobs: Option<SyncSender<Job>>,
    /// File bytes of queued writes; their buffers stay alive (pinning RAM) until written.
    pending: Arc<AtomicU64>,
    writer: Option<JoinHandle<()>>,
}

fn token_name(tokens: &[u32]) -> String {
    crate::artifact::hex(&Sha256::digest(bytemuck::cast_slice::<u32, u8>(tokens)))
}

/// Bytes a section occupies in the file: a whole number of (at least one) Metal pages.
const fn padded(len: usize) -> usize {
    let pages = len.div_ceil(METAL_PAGE);
    if pages == 0 { METAL_PAGE } else { pages * METAL_PAGE }
}

/// Parse magic + header; returns it with the file offset of the first section.
fn read_header(mut reader: impl Read) -> Option<(Header, usize)> {
    let mut fixed = [0_u8; 16];
    reader.read_exact(&mut fixed).ok()?;
    let (magic, len) = fixed.split_at(8);
    if magic != MAGIC {
        return None;
    }
    let len = usize::try_from(u64::from_le_bytes(len.try_into().ok()?)).ok()?;
    if len > 64 << 20 {
        return None;
    }
    let mut json = vec![0_u8; len];
    reader.read_exact(&mut json).ok()?;
    Some((serde_json::from_slice(&json).ok()?, (16 + len).next_multiple_of(METAL_PAGE)))
}

/// Sanity of a header against its file name, this cache's KV format and the file length.
fn header_ok(header: &Header, name: &str, kv_format: KvFormat, data_start: usize, file_len: u64) -> bool {
    let Ok(file_len) = usize::try_from(file_len) else { return false };
    name == token_name(&header.tokens)
        && header.kv_format == kv_format.code()
        && header.dspark_scored_anchors <= header.tokens.len()
        && !header.sections.is_empty()
        && header.sections.iter().all(|&[offset, len]| {
            offset.is_multiple_of(METAL_PAGE)
                && data_start
                    .checked_add(offset)
                    .and_then(|o| o.checked_add(padded(len)))
                    .is_some_and(|end| end <= file_len)
        })
}

fn lock(entries: &Mutex<Vec<Entry>>) -> MutexGuard<'_, Vec<Entry>> {
    entries.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Delete least recently used files until the directory fits `budget`: unpinned files first,
/// pinned ones only when they alone exceed it.
fn evict(entries: &mut Vec<Entry>, dir: &Path, budget: u64) {
    let mut total: u64 = entries.iter().map(|e| e.bytes).sum();
    // Most evictable last: unpinned before pinned, then oldest.
    entries.sort_by_key(|e| (!e.pinned, std::cmp::Reverse(e.used)));
    while total > budget
        && let Some(oldest) = entries.pop()
    {
        total -= oldest.bytes;
        let _ = fs::remove_file(dir.join(&oldest.name));
    }
}

#[expect(unsafe_code)]
fn write_file(dir: &Path, name: &str, header: &Header, buffers: &[SendBuffer]) -> std::io::Result<u64> {
    let json = serde_json::to_vec(header)?;
    let data_start = (16 + json.len()).next_multiple_of(METAL_PAGE);
    let tmp = dir.join(format!("{name}.tmp.{}", std::process::id()));
    let written = (|| -> std::io::Result<u64> {
        let mut out = File::create(&tmp)?;
        out.write_all(&MAGIC)?;
        out.write_all(&(json.len() as u64).to_le_bytes())?;
        out.write_all(&json)?;
        let mut end = data_start;
        for (buffer, &[offset, len]) in buffers.iter().zip(&header.sections) {
            // SAFETY: see `SendBuffer`; `persist` checked `len <= length()`.
            let bytes = unsafe { buffer.0.as_slice::<u8>() };
            out.seek(SeekFrom::Start((data_start + offset) as u64))?;
            out.write_all(&bytes[..len])?;
            end = data_start + offset + padded(len);
        }
        // Gaps and the tail stay sparse zeros.
        out.set_len(end as u64)?;
        Ok(end as u64)
    })();
    match written {
        Ok(bytes) => fs::rename(&tmp, dir.join(name)).map(|()| bytes),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

impl DiskCache {
    /// Open (creating it) and index the cache directory for one model payload and KV format.
    /// `None` when the directory is not writable, e.g. inside a read-only app bundle.
    pub fn open(root: &Path, key: &str, kv_format: KvFormat, budget: u64) -> Option<Self> {
        let dir = root.join(format!("{key}-{kv_format:?}").to_lowercase());
        fs::create_dir_all(&dir).ok()?;
        let probe = dir.join(format!(".probe.{}", std::process::id()));
        File::create(&probe).ok()?;
        let _ = fs::remove_file(&probe);

        let mut entries = Vec::new();
        for item in fs::read_dir(&dir).ok()?.flatten() {
            let path = item.path();
            let indexed = item.file_name().into_string().ok().and_then(|name| {
                let meta = item.metadata().ok()?;
                let (header, start) = read_header(BufReader::new(File::open(&path).ok()?))?;
                header_ok(&header, &name, kv_format, start, meta.len()).then(|| Entry {
                    name,
                    tokens: header.tokens,
                    pinned: header.pinned,
                    bytes: meta.len(),
                    used: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                })
            });
            // Anything else (corrupt files, stale temp files of a killed process) is dropped.
            match indexed {
                Some(entry) => entries.push(entry),
                None => drop(fs::remove_file(&path)),
            }
        }
        evict(&mut entries, &dir, budget);
        let entries = Arc::new(Mutex::new(entries));

        let (jobs, rx) = sync_channel::<Job>(MAX_PENDING_WRITES);
        let pending = Arc::new(AtomicU64::new(0));
        let (writer_dir, writer_entries, writer_pending) = (dir.clone(), Arc::clone(&entries), Arc::clone(&pending));
        let writer = std::thread::Builder::new()
            .name("prompt-cache-writer".into())
            .spawn(move || {
                for job in rx {
                    match job {
                        Job::Touch(path) => {
                            let _ = File::open(&path).and_then(|f| f.set_modified(SystemTime::now()));
                        }
                        Job::Write { name, header, buffers, bytes } => {
                            let written = write_file(&writer_dir, &name, &header, &buffers);
                            drop(buffers);
                            writer_pending.fetch_sub(bytes, Ordering::Relaxed);
                            let Ok(bytes) = written else {
                                continue;
                            };
                            let mut entries = lock(&writer_entries);
                            // Keep only the latest conversation checkpoint: drop unpinned ones it extends.
                            entries.retain(|e| {
                                let superseded = e.name != name
                                    && !header.pinned
                                    && !e.pinned
                                    && header.tokens.starts_with(&e.tokens);
                                if superseded {
                                    let _ = fs::remove_file(writer_dir.join(&e.name));
                                }
                                e.name != name && !superseded
                            });
                            entries.push(Entry {
                                name,
                                tokens: header.tokens,
                                pinned: header.pinned,
                                bytes,
                                used: SystemTime::now(),
                            });
                            evict(&mut entries, &writer_dir, budget);
                            drop(entries);
                        }
                    }
                }
            })
            .ok()?;
        Some(Self {
            root: root.to_path_buf(),
            key: key.to_owned(),
            budget,
            dir,
            kv_format,
            entries,
            jobs: Some(jobs),
            pending,
            writer: Some(writer),
        })
    }

    /// The cache for the same model under another KV format.
    pub fn reopen(&self, kv_format: KvFormat) -> Option<Self> {
        Self::open(&self.root, &self.key, kv_format, self.budget)
    }

    /// Length of the longest cached prefix of `prompt` no longer than `max_len`.
    pub fn longest(&self, prompt: &[u32], max_len: usize) -> usize {
        lock(&self.entries)
            .iter()
            .filter(|e| e.tokens.len() <= max_len && prompt.starts_with(&e.tokens))
            .map(|e| e.tokens.len())
            .max()
            .unwrap_or(0)
    }

    /// Queue `buffers` (`(buffer, bytes used)`) for writing unless the file exists, in which case
    /// it is touched. Never blocks: a full queue drops the write.
    pub fn persist(
        &self,
        tokens: &[u32],
        rope_delta: i32,
        dspark_scored_anchors: usize,
        pinned: bool,
        buffers: Vec<(MetalBuffer, usize)>,
    ) {
        let name = token_name(tokens);
        if self.touch_name(&name) || buffers.iter().any(|(b, len)| *len > b.length()) {
            return;
        }
        let mut offset = 0;
        let sections = buffers
            .iter()
            .map(|&(_, len)| {
                let section = [offset, len];
                offset += padded(len);
                section
            })
            .collect();
        // A file over the budget would only evict everything else, then itself.
        let bytes = offset as u64;
        if bytes > self.budget {
            return;
        }
        let header = Header {
            tokens: tokens.to_vec(),
            rope_delta,
            dspark_scored_anchors,
            kv_format: self.kv_format.code(),
            pinned,
            sections,
        };
        let buffers = buffers.into_iter().map(|(b, _)| SendBuffer(b)).collect();
        // Queued writes pin at most one budget of RAM; beyond that the write is dropped.
        let reserved = self.pending.try_update(Ordering::Relaxed, Ordering::Relaxed, |p| {
            (p + bytes <= self.budget).then_some(p + bytes)
        });
        if let (Some(jobs), Ok(_)) = (&self.jobs, reserved)
            && jobs.try_send(Job::Write { name, header, buffers, bytes }).is_err()
        {
            self.pending.fetch_sub(bytes, Ordering::Relaxed);
        }
    }

    /// Mark the file for `tokens` as just used.
    pub fn touch(&self, tokens: &[u32]) {
        self.touch_name(&token_name(tokens));
    }

    fn touch_name(&self, name: &str) -> bool {
        if let Some(entry) = lock(&self.entries).iter_mut().find(|e| e.name == name) {
            entry.used = SystemTime::now();
        } else {
            return false;
        }
        if let Some(jobs) = &self.jobs {
            let _ = jobs.try_send(Job::Touch(self.dir.join(name)));
        }
        true
    }

    /// Map the longest cached prefix of `prompt` that is longer than `floor` and at most
    /// `max_len` tokens, whose sections have exactly the lengths `expected(prefix_len)`.
    /// Files that fail validation are deleted and the next candidate is tried.
    pub fn load(
        &self,
        ctx: &MetalContext,
        prompt: &[u32],
        max_len: usize,
        floor: usize,
        expected: impl Fn(usize) -> Vec<usize>,
    ) -> Option<Loaded> {
        loop {
            let (name, len) = lock(&self.entries)
                .iter()
                .filter(|e| {
                    e.tokens.len() > floor && e.tokens.len() <= max_len && prompt.starts_with(&e.tokens)
                })
                .max_by_key(|e| e.tokens.len())
                .map(|e| (e.name.clone(), e.tokens.len()))?;
            if let Some(loaded) = self.map_file(ctx, &name, len, &expected(len)) {
                return Some(loaded);
            }
            lock(&self.entries).retain(|e| e.name != name);
            let _ = fs::remove_file(self.dir.join(&name));
        }
    }

    #[expect(unsafe_code)]
    fn map_file(&self, ctx: &MetalContext, name: &str, len: usize, expected: &[usize]) -> Option<Loaded> {
        let file = File::open(self.dir.join(name)).ok()?;
        // SAFETY: files are only ever replaced by rename, never modified in place.
        let map = Arc::new(unsafe { Mmap::map(&file) }.ok()?);
        let (header, start) = read_header(&map[..])?;
        if !header_ok(&header, name, self.kv_format, start, map.len() as u64)
            || header.tokens.len() != len
            || !header.sections.iter().map(|s| s[1]).eq(expected.iter().copied())
        {
            return None;
        }
        // The restore blit reads every byte right away; fault the file in ahead of the GPU.
        let _ = map.advise(memmap2::Advice::WillNeed);
        let owner: Arc<dyn Send + Sync> = Arc::clone(&map) as Arc<dyn Send + Sync>;
        let buffers = header
            .sections
            .iter()
            .map(|&[offset, len]| {
                let ptr = NonNull::new(map[start + offset..].as_ptr().cast_mut())?;
                // SAFETY: `header_ok` bounds-checked the page-aligned range; `owner` keeps it mapped.
                unsafe { MetalBuffer::from_bytes_no_copy(ctx.device(), ptr, padded(len), Arc::clone(&owner)) }
                    .ok()
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Loaded { header, buffers })
    }
}

impl Drop for DiskCache {
    /// Finish queued writes so a reopened cache sees them.
    fn drop(&mut self) {
        drop(self.jobs.take());
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}
