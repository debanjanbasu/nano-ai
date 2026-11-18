//! Byte-level BPE tokenizer for the MiniCPM-V 4.7 1B vocabulary.
//!
//! It runs from a compact `tokenizer.bin` mapped from disk instead of the ~100 MiB of heap the
//! Hugging Face `tokenizers` crate builds from `tokenizer.json`.
//!
//! The pipeline reproduces `tokenizer.json` exactly (checked against `tokenizers` in the tests):
//! added tokens are split out verbatim, the rest is NFC-normalized, cut by the model's pre-tokenizer
//! regex (hand-coded below over the same Unicode classes `fancy-regex` uses), then merged by
//! byte-level BPE in merge-rank order. Decoding concatenates token bytes as lossy UTF-8.
//!
//! `tokenizer.bin` layout (little-endian, every section 4-byte aligned):
//! magic `MCTOK\0\0\x01`, then `u32` counts `tokens`, `merges`, `added`, `blob_len`;
//! `[u32; 256]` token id of each byte; `[u32; tokens + 1]` offsets into the blob of token bytes;
//! the blob (padded to 4 bytes); merge keys `[[u32; 2]; merges]` (`left`, `right`) sorted;
//! merge values `[[u32; 2]; merges]` (`rank`, `merged id`); `[u32; added]` added-token ids.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::LazyLock;

use memmap2::{Mmap, MmapMut};
use regex_syntax::hir::{Class, HirKind};
use unicode_normalization_alignments::UnicodeNormalization;

use crate::Error;

const MAGIC: &[u8; 8] = b"MCTOK\0\0\x01";
const HEADER_WORDS: usize = 6; // magic (2 words) + 4 counts
/// The pre-tokenizer this module implements by hand; any other `tokenizer.json` is rejected.
const SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    map: Mmap,
    tokens: usize,
    offsets_at: usize,
    blob_at: usize,
    keys_at: usize,
    values_at: usize,
    added_at: usize,
    merges: usize,
    added: usize,
    /// Cached `(offset, len, id)` of every added token's bytes and a first-byte -> index bucket
    /// index, built once so `encode` does not re-derive both per call.
    added_table: Vec<(usize, usize, u32)>,
    added_buckets: BTreeMap<u8, Range<usize>>,
}

#[derive(Default)]
struct BpeScratch {
    ids: Vec<u32>,
    next: Vec<usize>,
    prev: Vec<usize>,
    heap: BinaryHeap<Reverse<(u32, usize, u32)>>,
}

impl Tokenizer {
    /// Open `dir/tokenizer.bin` (raw or Zstd-compressed), building it from `dir/tokenizer.json`
    /// first when only the JSON exists (kept in memory if the directory is read-only).
    pub fn load(dir: &Path) -> crate::Result<Option<Self>> {
        let bin = dir.join("tokenizer.bin");
        let json = dir.join("tokenizer.json");
        if !bin.is_file() {
            if !json.is_file() {
                return Ok(None);
            }
            let bytes = build_binary(&json)?;
            let compressed = zstd::stream::encode_all(bytes.as_slice(), 19).unwrap_or_else(|_| bytes.clone());
            let _ = write_atomic(&bin, &compressed);
            let mut map = MmapMut::map_anon(bytes.len())?;
            map.copy_from_slice(&bytes);
            return Self::from_map(map.make_read_only()?).map(Some);
        }
        #[expect(unsafe_code)]
        // SAFETY: read-only mapping of a file this process does not modify.
        let map = unsafe { Mmap::map(&File::open(&bin)?)? };
        if map.get(..4) == Some(b"\x28\xb5\x2f\xfd") {
            let decoded = crate::artifact::zstd_decompress_with_dict(&map, &[], 32 << 20)?;
            let mut anon = MmapMut::map_anon(decoded.len())?;
            anon.copy_from_slice(&decoded);
            return Self::from_map(anon.make_read_only()?).map(Some);
        }
        Self::from_map(map).map(Some)
    }
    fn from_map(map: Mmap) -> crate::Result<Self> {
        let bad = || Error::Tokenizer("corrupt tokenizer.bin".into());
        let words: &[u32] = bytemuck::try_cast_slice(map.get(..HEADER_WORDS * 4).ok_or_else(bad)?)
            .map_err(|_| bad())?;
        if map.get(..8) != Some(MAGIC.as_slice()) {
            return Err(bad());
        }
        let [tokens, merges, added, blob_len] =
            [words[2], words[3], words[4], words[5]].map(|w| w as usize);
        let offsets_at = (HEADER_WORDS + 256) * 4;
        let blob_at = offsets_at + (tokens + 1) * 4;
        let keys_at = blob_at + blob_len;
        let values_at = keys_at + merges * 8;
        let added_at = values_at + merges * 8;
        if blob_len % 4 != 0 || map.len() != added_at + added * 4 {
            return Err(bad());
        }
        let tok = Self {
            map,
            tokens,
            offsets_at,
            blob_at,
            keys_at,
            values_at,
            added_at,
            merges,
            added,
            added_table: Vec::new(),
            added_buckets: BTreeMap::new(),
        };
        let offsets = tok.words(tok.offsets_at, tokens + 1);
        if offsets.windows(2).any(|w| w[0] > w[1]) || offsets[tokens] as usize > blob_len {
            return Err(bad());
        }
        let mut tok = tok;
        tok.build_added_table();
        Ok(tok)
    }

    /// Cache every added token's `(blob offset, byte length, id)` and bucket the entries by
    /// first byte, so `encode`'s leftmost-longest scan touches only candidates that can match.
    /// Within a bucket entries are longest-first, so the first `starts_with` hit is the longest
    /// match.
    fn build_added_table(&mut self) {
        let map_start = self.map.as_ptr() as usize;
        self.added_table = self
            .words(self.added_at, self.added)
            .iter()
            .map(|&id| {
                let bytes = self.token_bytes(id);
                (bytes.as_ptr() as usize - map_start, bytes.len(), id)
            })
            .collect();
        self.added_buckets.clear();
        for (idx, (offset, len, _)) in self.added_table.iter().enumerate() {
            if *len > 0 {
                let b = self.map[*offset];
                let entry = self.added_buckets.entry(b).or_insert(idx..idx);
                entry.end = idx + 1;
            }
        }
        self.added_table
            .sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));
    }

    fn words(&self, at: usize, len: usize) -> &[u32] {
        // Offsets are validated in `from_map` and 4-byte aligned on a page-aligned mapping.
        bytemuck::cast_slice(&self.map[at..at + len * 4])
    }

    fn pairs(&self, at: usize) -> &[[u32; 2]] {
        bytemuck::cast_slice(&self.map[at..at + self.merges * 8])
    }

    /// UTF-8 bytes of token `id` (empty for ids outside the vocabulary).
    #[must_use]
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let id = id as usize;
        if id >= self.tokens {
            return &[];
        }
        let offsets = self.words(self.offsets_at, self.tokens + 1);
        &self.map[self.blob_at + offsets[id] as usize..self.blob_at + offsets[id + 1] as usize]
    }

    fn merge(&self, left: u32, right: u32) -> Option<[u32; 2]> {
        let keys = self.pairs(self.keys_at);
        keys.binary_search(&[left, right])
            .ok()
            .map(|i| self.pairs(self.values_at)[i])
    }

    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        // Leftmost-longest added-token match, like `tokenizers`' added vocabulary: at each
        // position, only the bucket of added tokens sharing the byte there can match, and the
        // table is sorted longest-first within that bucket.
        while let Some((start, len, id)) = (0..rest.len()).find_map(|at| {
            let first = rest.as_bytes()[at];
            self.added_buckets
                .get(&first)
                .and_then(|range| {
                    let hay = rest.as_bytes();
                    self.added_table[range.clone()].iter().find(|&(offset, len, _)| {
                        hay[at..].get(..*len).is_some_and(|window| window == &self.map[*offset..*offset + *len])
                    })
                })
                .map(|&(_, len, id)| (at, len, id))
        }) {
            let (head, tail) = rest.split_at(start);
            self.encode_plain(head, &mut ids);
            ids.push(id);
            rest = tail.split_at(len).1;
        }
        self.encode_plain(rest, &mut ids);
        ids
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        let normalized;
        let text = if text.is_ascii() {
            text
        } else {
            normalized = text.nfc().map(|(c, _)| c).collect::<String>();
            &normalized
        };
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut scratch = BpeScratch::default();
        let mut i = 0;
        while i < chars.len() {
            let end = piece_end(&chars, i);
            let from = chars[i].0;
            let to = chars.get(end).map_or(text.len(), |&(at, _)| at);
            self.bpe(&text.as_bytes()[from..to], &mut scratch, ids);
            i = end;
        }
    }

    /// Byte-level BPE: repeatedly apply the lowest-rank (then leftmost) merge, as `tokenizers` does.
    fn bpe(&self, piece: &[u8], scratch: &mut BpeScratch, out: &mut Vec<u32>) {
        let byte_ids = self.words(HEADER_WORDS * 4, 256);
        let n = piece.len();
        scratch.ids.clear();
        scratch.ids.extend(piece.iter().map(|&b| byte_ids[b as usize]));
        scratch.next.clear();
        scratch.next.extend(1..=n);
        scratch.prev.clear();
        scratch.prev.extend((0..n).map(|i| i.wrapping_sub(1)));
        scratch.heap.clear();
        for pos in 0..n.saturating_sub(1) {
            if let Some([rank, merged]) = self.merge(scratch.ids[pos], scratch.ids[pos + 1]) {
                scratch.heap.push(Reverse((rank, pos, merged)));
            }
        }
        while let Some(Reverse((_, pos, merged))) = scratch.heap.pop() {
            let right = scratch.next[pos];
            // Skip entries made stale by an earlier merge.
            if scratch.ids[pos] == u32::MAX
                || right >= n
                || self.merge(scratch.ids[pos], scratch.ids[right]).map(|m| m[1]) != Some(merged)
            {
                continue;
            }
            scratch.ids[pos] = merged;
            scratch.ids[right] = u32::MAX;
            scratch.next[pos] = scratch.next[right];
            if scratch.next[pos] < n {
                scratch.prev[scratch.next[pos]] = pos;
            }
            if let Some(&left) = scratch.prev.get(pos).filter(|&&p| p < n)
                && let Some([rank, m]) = self.merge(scratch.ids[left], merged)
            {
                scratch.heap.push(Reverse((rank, left, m)));
            }
            if scratch.next[pos] < n
                && let Some([rank, m]) = self.merge(merged, scratch.ids[scratch.next[pos]])
            {
                scratch.heap.push(Reverse((rank, pos, m)));
            }
        }
        out.extend(scratch.ids.iter().copied().filter(|&id| id != u32::MAX));
    }

    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.token_bytes(id))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Append token `id` to `pending` and return the text once it ends on a whole UTF-8 sequence
    /// (the incremental decode of `tokenizers::step_decode_stream` for byte-level vocabularies).
    pub fn stream_piece(&self, pending: &mut Vec<u8>, id: u32) -> Option<String> {
        pending.extend_from_slice(self.token_bytes(id));
        let text = String::from_utf8_lossy(pending);
        if text.is_empty() || text.ends_with('\u{FFFD}') {
            return None;
        }
        let text = text.into_owned();
        pending.clear();
        Some(text)
    }
}

/// Unicode classes of the pre-tokenizer, from the same `regex-syntax` tables `fancy-regex` uses.
struct Classes {
    letter: Vec<(char, char)>,
    mark: Vec<(char, char)>,
    number: Vec<(char, char)>,
    space: Vec<(char, char)>,
    /// Case-insensitive matches of the contraction letters `s t r v m l d e`.
    folds: Vec<(char, Vec<(char, char)>)>,
}

static CLASSES: LazyLock<Classes> = LazyLock::new(|| {
    let ranges =
        |pattern: &str| match regex_syntax::parse(pattern).map(regex_syntax::hir::Hir::into_kind) {
            Ok(HirKind::Class(Class::Unicode(class))) => class
                .ranges()
                .iter()
                .map(|r| (r.start(), r.end()))
                .collect(),
            _ => Vec::new(),
        };
    Classes {
        letter: ranges(r"\p{L}"),
        mark: ranges(r"\p{M}"),
        number: ranges(r"\p{N}"),
        space: ranges(r"\s"),
        folds: "strvmlde"
            .chars()
            .map(|c| (c, ranges(&format!("(?i:{c})"))))
            .collect(),
    }
});

fn in_ranges(ranges: &[(char, char)], c: char) -> bool {
    ranges
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// End (char index) of the pre-tokenizer piece starting at `i`: the first alternative of
/// `SPLIT_PATTERN` that matches, with the regex's greedy/backtracking semantics resolved by hand.
fn piece_end(chars: &[(usize, char)], i: usize) -> usize {
    let cls = &*CLASSES;
    let at = |k: usize| chars.get(k).map(|&(_, c)| c);
    let letter = |c: char| in_ranges(&cls.letter, c);
    let lm = |c: char| letter(c) || in_ranges(&cls.mark, c);
    let number = |c: char| in_ranges(&cls.number, c);
    let space = |c: char| in_ranges(&cls.space, c);
    let newline = |c: char| c == '\r' || c == '\n';
    let other = |c: char| !(space(c) || lm(c) || number(c));
    let run = |from: usize, pred: &dyn Fn(char) -> bool| {
        (from..chars.len())
            .find(|&k| !pred(chars[k].1))
            .unwrap_or(chars.len())
    };
    let fold = |k: usize| {
        at(k).and_then(|c| {
            cls.folds
                .iter()
                .find(|(_, r)| in_ranges(r, c))
                .map(|&(l, _)| l)
        })
    };
    let c = chars[i].1;
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c == '\'' {
        match fold(i + 1) {
            Some('s' | 't' | 'm' | 'd') => return i + 2,
            Some('r' | 'v') if fold(i + 2) == Some('e') => return i + 3,
            Some('l') if fold(i + 2) == Some('l') => return i + 3,
            _ => {}
        }
    }
    // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if lm(c) {
        return run(i, &lm);
    }
    if !(newline(c) || number(c)) && at(i + 1).is_some_and(lm) {
        return run(i + 1, &lm);
    }
    // \p{N}
    if number(c) {
        return i + 1;
    }
    // ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
    let symbols = if other(c) {
        Some(i)
    } else if c == ' ' && at(i + 1).is_some_and(other) {
        Some(i + 1)
    } else {
        None
    };
    if let Some(from) = symbols {
        return run(run(from, &other), &newline);
    }
    // Only whitespace is left. `\s*[\r\n]+` ends after the run's last line break.
    let end = run(i, &space);
    if let Some(last) = (i..end).rev().find(|&k| newline(chars[k].1)) {
        return last + 1;
    }
    // `\s+(?!\S)` keeps the last space for the next word; `\s+` takes a lone space.
    if end == chars.len() || end - i == 1 {
        end
    } else {
        end - 1
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Replace `dir/tokenizer.json` (~20 MB) or uncompressed `dir/tokenizer.bin` (~6.8 MB) with a
/// Zstd-compressed `tokenizer.bin` (~2.9 MB). Returns whether any file was compacted.
pub fn compact_in_place(dir: &Path) -> crate::Result<bool> {
    let json = dir.join("tokenizer.json");
    let bin = dir.join("tokenizer.bin");
    if json.is_file() {
        let raw = build_binary(&json)?;
        let compressed = zstd::stream::encode_all(raw.as_slice(), 19)?;
        write_atomic(&bin, &compressed)?;
        std::fs::remove_file(&json)?;
        return Ok(true);
    }
    if bin.is_file() {
        let data = std::fs::read(&bin)?;
        if data.get(..4) != Some(b"\x28\xb5\x2f\xfd") {
            let compressed = zstd::stream::encode_all(data.as_slice(), 19)?;
            write_atomic(&bin, &compressed)?;
            return Ok(true);
        }
    }
    Ok(false)
}

/// GPT-2 byte-to-character map of byte-level BPE vocabularies.
fn byte_chars() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut extra = 256_u32;
    for (b, slot) in map.iter_mut().enumerate() {
        *slot = if matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF) {
            char::from(b as u8)
        } else {
            extra += 1;
            char::from_u32(extra - 1).unwrap_or('\0')
        };
    }
    map
}

/// Build `tokenizer.bin` from MiniCPM-V 4.7's `tokenizer.json`, rejecting any other pipeline.
pub fn build_binary(json_path: &Path) -> crate::Result<Vec<u8>> {
    let bad = |what: &str| {
        Error::Tokenizer(format!(
            "{}: unsupported tokenizer ({what})",
            json_path.display()
        ))
    };
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(json_path)?)?;
    let pre = &v["pre_tokenizer"]["pretokenizers"];
    let expected = [
        (v["model"]["type"] == "BPE", "model"),
        (v["normalizer"]["type"] == "NFC", "normalizer"),
        (
            pre[0]["pattern"]["Regex"] == SPLIT_PATTERN && pre[0]["behavior"] == "Isolated",
            "pre-tokenizer",
        ),
        (
            pre[1]["type"] == "ByteLevel"
                && pre[1]["use_regex"] == false
                && pre[1]["add_prefix_space"] == false,
            "byte level",
        ),
        (v["decoder"]["type"] == "ByteLevel", "decoder"),
        (
            v["model"]["byte_fallback"] != true && v["model"]["ignore_merges"] != true,
            "merges",
        ),
    ];
    if let Some((_, what)) = expected.iter().find(|(ok, _)| !ok) {
        return Err(bad(what));
    }
    let vocab = v["model"]["vocab"]
        .as_object()
        .ok_or_else(|| bad("vocab"))?;
    let added = v["added_tokens"]
        .as_array()
        .ok_or_else(|| bad("added tokens"))?;
    let chars = byte_chars();
    let char_byte = |c: char| chars.iter().position(|&m| m == c).map(|b| b as u8);
    let mut strings: Vec<Option<String>> = Vec::new();
    let mut put = |id: usize, s: &str| {
        if strings.len() <= id {
            strings.resize(id + 1, None);
        }
        strings[id] = Some(s.to_owned());
    };
    for (token, id) in vocab {
        put(id.as_u64().ok_or_else(|| bad("vocab id"))? as usize, token);
    }
    let mut added_ids = Vec::new();
    for token in added {
        let id = token["id"].as_u64().ok_or_else(|| bad("added id"))? as usize;
        put(
            id,
            token["content"]
                .as_str()
                .ok_or_else(|| bad("added content"))?,
        );
        added_ids.push(id as u32);
    }
    let ids: std::collections::HashMap<&str, u32> = strings
        .iter()
        .enumerate()
        .filter_map(|(id, s)| s.as_deref().map(|s| (s, id as u32)))
        .collect();
    let mut merges: Vec<([u32; 2], [u32; 2])> = Vec::new();
    for (rank, pair) in v["model"]["merges"]
        .as_array()
        .ok_or_else(|| bad("merges"))?
        .iter()
        .enumerate()
    {
        let (l, r) = match (pair[0].as_str(), pair[1].as_str(), pair.as_str()) {
            (Some(l), Some(r), _) => (l, r),
            (_, _, Some(s)) => s.split_once(' ').ok_or_else(|| bad("merge"))?,
            _ => return Err(bad("merge")),
        };
        let (Some(&li), Some(&ri), Some(&mi)) =
            (ids.get(l), ids.get(r), ids.get(format!("{l}{r}").as_str()))
        else {
            return Err(bad("merge token"));
        };
        merges.push(([li, ri], [rank as u32, mi]));
    }
    merges.sort_unstable_by_key(|m| m.0);

    let mut blob = Vec::new();
    let mut offsets = vec![0_u32];
    for s in &strings {
        let s = s.as_deref().unwrap_or("");
        // Byte-level tokens map back char by char; anything else (added tokens) is its own UTF-8.
        match s.chars().map(char_byte).collect::<Option<Vec<u8>>>() {
            Some(bytes) => blob.extend_from_slice(&bytes),
            None => blob.extend_from_slice(s.as_bytes()),
        }
        offsets.push(blob.len() as u32);
    }
    blob.resize(blob.len().next_multiple_of(4), 0);
    let byte_ids = chars
        .iter()
        .map(|c| {
            ids.get(c.to_string().as_str())
                .copied()
                .ok_or_else(|| bad("byte token"))
        })
        .collect::<crate::Result<Vec<u32>>>()?;

    let mut out = MAGIC.to_vec();
    for w in [strings.len(), merges.len(), added_ids.len(), blob.len()] {
        out.extend_from_slice(&(w as u32).to_le_bytes());
    }
    let words = |out: &mut Vec<u8>, ws: &mut dyn Iterator<Item = u32>| {
        ws.for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
    };
    words(&mut out, &mut byte_ids.into_iter());
    words(&mut out, &mut offsets.into_iter());
    out.extend_from_slice(&blob);
    words(&mut out, &mut merges.iter().flat_map(|m| m.0));
    words(&mut out, &mut merges.iter().flat_map(|m| m.1));
    words(&mut out, &mut added_ids.into_iter());
    Ok(out)
}
