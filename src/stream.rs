//! Streaming checkpoint loader: read a safetensors file group-at-a-time
//! instead of all at once.
//!
//! The reason this exists: VoxCPM2's `model.safetensors` is 4.37 GB, and
//! `wasm32-unknown-unknown` has a 4 GB address space. The loader in
//! [`crate::weights`] reads the whole file, converts every tensor to the
//! backend dtype into a second full-size map, re-serializes that into a
//! third full-size buffer, and only then hands it to burn-store. At F32
//! that peaks around 22 GB of linear memory. In a browser it is not a
//! question of being slow; it cannot be allocated.
//!
//! So instead of one pass over the whole file:
//!
//! ```text
//!  read 0..8         → header length (u64 LE)
//!  read 8..8+N       → JSON header  { name: {dtype, shape, data_offsets} }
//!                      ├─ group tensors by module  (see `group_key`)
//!                      └─ batch groups under a byte budget
//!  for each batch:
//!      read its byte runs
//!      convert dtype / materialize weight_norm / fuse qkv + gate_up
//!      burn-store apply, allow_partial(true)   → uploads to the backend
//!      drop everything
//! ```
//!
//! Peak heap is then a function of the batch budget, not of the model.
//!
//! This module is platform-independent and sits behind the [`ByteSource`]
//! trait, so the same code drives:
//!
//! * `browser::fetch::HttpRangeSource` — HTTP `Range` requests from a
//!   browser tab (the reason this was written; that module only exists on
//!   `wasm32`, so this is not a link),
//! * [`FileSource`] — `seek` + `read` on a native file, which also makes
//!   the whole path testable without a browser,
//! * [`MemorySource`] — bytes already in memory.
//!
//! ## Why offsets are `u64` here
//!
//! `safetensors::tensor::TensorInfo` types `data_offsets` as
//! `(usize, usize)`. On `wasm32` `usize` is 32-bit, so it tops out at
//! 4294967295 — *smaller than this checkpoint*. Tensors past the 4 GB mark
//! would silently wrap. The header is therefore parsed into a local
//! [`TensorEntry`] with `u64` offsets rather than reusing `TensorInfo`.
//! (`safetensors::SafeTensors::read_metadata` is also unusable: it asserts
//! the buffer holds the entire file.)

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;

use burn::prelude::*;
use burn_store::{ApplyResult, ModuleSnapshot};
use safetensors::Dtype;
use serde::Deserialize;

use crate::weights::{self, Namespace, RawTensor};
use crate::{Error, Result};

/// Default per-batch budget of *source* bytes.
///
/// Peak heap is roughly `budget` (raw) + `budget × dst/src` (converted) +
/// the same again for the re-serialized buffer. At 64 MiB with a BF16 →
/// F32 conversion that is ~320 MB, which leaves plenty of headroom. Groups
/// are never split, so a single group larger than the budget (the token
/// embedding, at 301 MB) still goes through in one piece — those take the
/// single-tensor fast path, which skips the re-serialize copy.
pub const DEFAULT_BATCH_BUDGET: u64 = 64 * 1024 * 1024;

/// Safetensors' own cap on header size; a sane bound before we allocate.
const MAX_HEADER_BYTES: u64 = 100_000_000;

// ---------------------------------------------------------------------------
// ByteSource
// ---------------------------------------------------------------------------

/// A random-access source of checkpoint bytes.
///
/// Deliberately range-oriented rather than file-oriented: the whole point
/// of this module is that "give me the whole file" is not an operation that
/// can succeed for a 4.37 GB checkpoint in a 4 GB address space.
///
/// `async fn` in trait is used for static dispatch only — [`stream_into`]
/// is generic over the implementation, so there is no `dyn` requirement and
/// no `Send` bound (the browser runs this single-threaded).
#[allow(async_fn_in_trait)]
pub trait ByteSource {
    /// Total length of the underlying resource.
    fn len(&self) -> u64;

    /// Whether the resource is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read exactly `len` bytes starting at `offset`.
    async fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// A label for error messages and progress events.
    fn label(&self) -> &str;
}

/// A [`ByteSource`] over a buffer already in memory.
#[derive(Debug, Clone)]
pub struct MemorySource {
    label: String,
    bytes: Vec<u8>,
}

impl MemorySource {
    /// Wrap an in-memory buffer.
    pub fn new(label: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            label: label.into(),
            bytes,
        }
    }

    /// The wrapped bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl ByteSource for MemorySource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    async fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let start = offset as usize;
        let end = start
            .checked_add(len as usize)
            .ok_or_else(|| Error::Other(format!("{}: range overflow", self.label)))?;
        if end > self.bytes.len() {
            return Err(Error::Other(format!(
                "{}: read {len} bytes at {offset} is past the end ({})",
                self.label,
                self.bytes.len()
            )));
        }
        Ok(self.bytes[start..end].to_vec())
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// A [`ByteSource`] over a file on disk, via `seek` + `read`.
///
/// The native counterpart of the browser's HTTP-Range source. It exists for
/// two reasons: it makes this whole module testable without a browser (see
/// `examples/stream_load.rs`), and it gives native builds the same bounded
/// peak memory — useful well before any browser is involved, since the
/// default loader needs ~22 GB of RAM to apply this checkpoint at F32.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct FileSource {
    label: String,
    len: u64,
    file: std::cell::RefCell<std::fs::File>,
}

#[cfg(not(target_arch = "wasm32"))]
impl FileSource {
    /// Open `path` for random reads.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            label: path.display().to_string(),
            len,
            file: std::cell::RefCell::new(file),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ByteSource for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    async fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        if offset + len > self.len {
            return Err(Error::Other(format!(
                "{}: read {len} bytes at {offset} is past the end ({})",
                self.label, self.len
            )));
        }
        let mut file = self.file.borrow_mut();
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// Drive a future to completion on the current thread.
///
/// Sized for exactly one job: running [`stream_into`] over a
/// [`FileSource`] or [`MemorySource`], where every `await` resolves on its
/// first poll because the underlying read is plain synchronous work. It
/// busy-polls with a no-op waker, so it must not be used on a future that
/// genuinely parks — in the browser the real executor
/// (`wasm_bindgen_futures`) does this job.
#[cfg(not(target_arch = "wasm32"))]
pub fn block_on_ready<F: Future>(future: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct Noop;
    impl Wake for Noop {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Raw JSON shape of one header entry, with `u64` offsets.
#[derive(Debug, Deserialize)]
struct HeaderEntry {
    dtype: Dtype,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

/// One tensor in a checkpoint, located but not yet read.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Bare checkpoint key.
    pub name: String,
    /// On-disk dtype.
    pub dtype: Dtype,
    /// Logical shape.
    pub shape: Vec<usize>,
    /// Absolute offset in the file (data-blob offset + header size).
    pub offset: u64,
    /// Byte length.
    pub nbytes: u64,
}

/// A parsed safetensors header: every tensor located, no tensor read.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// All tensors, sorted by file offset.
    pub entries: Vec<TensorEntry>,
    /// Where the data blob starts (`8 + header_len`).
    pub data_start: u64,
    /// Total file size.
    pub file_size: u64,
}

impl Checkpoint {
    /// Read and parse just the header of `src`.
    ///
    /// Two small range reads: 8 bytes for the header length, then the
    /// header itself. Nothing else is touched.
    pub async fn open<S: ByteSource>(src: &S) -> Result<Self> {
        let file_size = src.len();
        if file_size < 8 {
            return Err(Error::Other(format!(
                "{}: {file_size} bytes is too small to be a safetensors file",
                src.label()
            )));
        }

        let len_bytes = src.read_at(0, 8).await?;
        let header_len = u64::from_le_bytes(
            len_bytes[..8]
                .try_into()
                .map_err(|_| Error::Other("short header-length read".into()))?,
        );
        if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len + 8 > file_size {
            return Err(Error::Other(format!(
                "{}: implausible safetensors header length {header_len} (file is {file_size} bytes)",
                src.label()
            )));
        }

        let header_bytes = src.read_at(8, header_len).await?;
        let header_str = std::str::from_utf8(&header_bytes)
            .map_err(|e| Error::Other(format!("{}: header is not UTF-8: {e}", src.label())))?;
        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)?;

        let data_start = 8 + header_len;
        let mut entries: Vec<TensorEntry> = Vec::with_capacity(raw.len());
        for (name, value) in raw {
            // `__metadata__` is a string map, not a tensor.
            if name == "__metadata__" {
                continue;
            }
            let entry: HeaderEntry = serde_json::from_value(value).map_err(|e| {
                Error::Other(format!("{}: bad header entry `{name}`: {e}", src.label()))
            })?;
            let (start, end) = entry.data_offsets;
            if end < start {
                return Err(Error::Other(format!(
                    "{}: tensor `{name}` has inverted offsets {start}..{end}",
                    src.label()
                )));
            }
            let nbytes = end - start;
            let offset = data_start + start;
            if offset + nbytes > file_size {
                return Err(Error::Other(format!(
                    "{}: tensor `{name}` ends at {} but the file is {file_size} bytes",
                    src.label(),
                    offset + nbytes
                )));
            }
            entries.push(TensorEntry {
                name,
                dtype: entry.dtype,
                shape: entry.shape.iter().map(|d| *d as usize).collect(),
                offset,
                nbytes,
            });
        }

        if entries.is_empty() {
            return Err(Error::Other(format!(
                "{}: safetensors header contains no tensors",
                src.label()
            )));
        }

        entries.sort_by_key(|e| e.offset);
        log::info!(
            "{}: {} tensors, {:.2} MB of weight data",
            src.label(),
            entries.len(),
            (file_size - data_start) as f64 / 1048576.0
        );
        Ok(Self {
            entries,
            data_start,
            file_size,
        })
    }

    /// Total bytes of tensor data (excludes the header).
    pub fn data_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.nbytes).sum()
    }

    /// Group tensors into batches that are safe to apply independently.
    ///
    /// See [`group_key`] for the grouping rule and why it is what it is.
    pub fn plan(&self, budget: u64) -> Vec<Batch> {
        // Group, preserving first-seen (offset) order so batches stay
        // roughly sequential in the file.
        let mut order: Vec<String> = Vec::new();
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, e) in self.entries.iter().enumerate() {
            let key = group_key(&e.name);
            groups.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                Vec::new()
            });
            groups.get_mut(&key).expect("just inserted").push(i);
        }

        let budget = budget.max(1);
        let mut batches: Vec<Batch> = Vec::new();
        let mut current = Batch::default();

        for key in order {
            let indices = groups.remove(&key).expect("key came from `order`");
            let nbytes: u64 = indices.iter().map(|i| self.entries[*i].nbytes).sum();

            // Flush first so a group is never split across batches — the
            // qkv/gate-up fusion and weight_norm materialization all need
            // their siblings present in the same apply.
            if !current.indices.is_empty() && current.nbytes + nbytes > budget {
                batches.push(std::mem::take(&mut current));
            }
            current.labels.push(key);
            current.indices.extend(indices);
            current.nbytes += nbytes;
        }
        if !current.indices.is_empty() {
            batches.push(current);
        }
        batches
    }
}

/// One unit of "fetch, convert, apply, drop".
#[derive(Debug, Default, Clone)]
pub struct Batch {
    /// Group keys in this batch, for logging and progress text.
    pub labels: Vec<String>,
    /// Indices into [`Checkpoint::entries`].
    pub indices: Vec<usize>,
    /// Source bytes this batch will read.
    pub nbytes: u64,
}

impl Batch {
    /// A short human-readable name — the first group key, plus a count.
    pub fn label(&self) -> String {
        match self.labels.len() {
            0 => "(empty)".to_string(),
            1 => self.labels[0].clone(),
            n => format!("{} (+{} more)", self.labels[0], n - 1),
        }
    }
}

/// The grouping key for a checkpoint tensor: everything but the last two
/// path segments.
///
/// Three transforms in [`crate::weights`] are *intra-group* and so dictate
/// what may be separated:
///
/// | transform | needs together |
/// |---|---|
/// | `fuse_qkv` | `…self_attn.{q,k,v}_proj.weight` |
/// | `fuse_gate_up` | `…mlp.{gate,up}_proj.weight` |
/// | `materialize_weight_norm` | `X.weight_g` + `X.weight_v` |
///
/// Dropping two segments puts each of those sets on one key —
/// `…self_attn`, `…mlp`, and `X`'s parent respectively — because in every
/// case the siblings differ only in their last two segments. Leaf params
/// such as `…input_layernorm.weight` land in harmless singleton groups.
pub fn group_key(name: &str) -> String {
    let segments: Vec<&str> = name.split('.').collect();
    if segments.len() <= 2 {
        return name.to_string();
    }
    segments[..segments.len() - 2].join(".")
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

/// Progress sink for a streamed load.
///
/// `stage` names the phase ("weights", "audiovae", …), `done`/`total` are
/// in bytes, and `detail` is the current batch label.
pub trait ProgressSink {
    /// Report progress. Called once per batch.
    fn report(&self, stage: &str, done: u64, total: u64, detail: &str);
}

/// A [`ProgressSink`] that logs and nothing else.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogProgress;

impl ProgressSink for LogProgress {
    fn report(&self, stage: &str, done: u64, total: u64, detail: &str) {
        let pct = if total == 0 {
            100.0
        } else {
            done as f64 * 100.0 / total as f64
        };
        log::info!("[{stage}] {pct:.1}% ({done}/{total}) {detail}");
    }
}

// ---------------------------------------------------------------------------
// Streaming apply
// ---------------------------------------------------------------------------

/// Stream every tensor of `checkpoint` from `src` into `model`.
///
/// Applies batch by batch, so peak WASM heap tracks `budget` rather than
/// the checkpoint size. Returns the accumulated [`ApplyResult`]; the caller
/// should pass it through [`crate::weights::finalize_apply_result`] once
/// *all* sources have been loaded (each batch legitimately reports every
/// parameter it does not itself carry as "missing").
pub async fn stream_into<B, M, S, P>(
    model: &mut M,
    src: &S,
    checkpoint: &Checkpoint,
    namespace: Namespace,
    budget: u64,
    stage: &str,
    progress: &P,
) -> Result<ApplyResult>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    S: ByteSource,
    P: ProgressSink,
{
    let target_dtype = weights::backend_float_dtype::<B>();
    let batches = checkpoint.plan(budget);
    let total = checkpoint.data_bytes();
    let mut done: u64 = 0;
    let mut acc = weights::empty_apply_result();

    log::info!(
        "{stage}: {} tensors in {} batches, {:.2} MB source -> {:?} on GPU",
        checkpoint.entries.len(),
        batches.len(),
        total as f64 / 1048576.0,
        target_dtype,
    );

    for (bi, batch) in batches.iter().enumerate() {
        let label = batch.label();
        progress.report(stage, done, total, &label);

        let t = crate::compat::Stopwatch::start();
        let result = apply_batch::<B, M, S>(model, src, checkpoint, batch, namespace).await?;
        log::debug!(
            "{stage}: batch {}/{} `{label}` {:.2} MB in {:.0}ms",
            bi + 1,
            batches.len(),
            batch.nbytes as f64 / 1048576.0,
            t.elapsed_ms()
        );
        weights::merge_apply_result(&mut acc, result);

        done += batch.nbytes;
    }

    progress.report(stage, total, total, "done");
    Ok(acc)
}

/// Fetch, convert and apply one batch.
async fn apply_batch<B, M, S>(
    model: &mut M,
    src: &S,
    checkpoint: &Checkpoint,
    batch: &Batch,
    namespace: Namespace,
) -> Result<ApplyResult>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    S: ByteSource,
{
    let tensors = read_batch(src, checkpoint, &batch.indices).await?;
    weights::apply_tensor_group::<B, M>(model, &batch.label(), tensors, namespace)
}

/// Read a batch's tensors, coalescing contiguous entries into single range
/// requests.
///
/// safetensors lays tensors out back to back in header order, so a batch
/// is usually one contiguous span and therefore one `fetch`. Entries are
/// re-sorted and merged rather than assumed contiguous, so an unusual
/// layout costs extra requests instead of returning wrong bytes.
async fn read_batch<S: ByteSource>(
    src: &S,
    checkpoint: &Checkpoint,
    wanted: &[usize],
) -> Result<Vec<RawTensor>> {
    let mut indices = wanted.to_vec();
    indices.sort_by_key(|i| checkpoint.entries[*i].offset);

    let mut out: Vec<RawTensor> = Vec::with_capacity(indices.len());
    let mut pos = 0usize;

    while pos < indices.len() {
        // Extend the run while entries remain exactly adjacent.
        let run_start = pos;
        let mut run_end_byte = {
            let e = &checkpoint.entries[indices[pos]];
            e.offset + e.nbytes
        };
        pos += 1;
        while pos < indices.len() {
            let e = &checkpoint.entries[indices[pos]];
            if e.offset != run_end_byte {
                break;
            }
            run_end_byte = e.offset + e.nbytes;
            pos += 1;
        }

        let base = checkpoint.entries[indices[run_start]].offset;
        let buf = src.read_at(base, run_end_byte - base).await?;

        for &i in &indices[run_start..pos] {
            let e = &checkpoint.entries[i];
            let lo = (e.offset - base) as usize;
            let hi = lo + e.nbytes as usize;
            out.push((e.name.clone(), e.shape.clone(), e.dtype, buf[lo..hi].to_vec()));
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_key_keeps_fusion_siblings_together() {
        let q = group_key("base_lm.layers.3.self_attn.q_proj.weight");
        let k = group_key("base_lm.layers.3.self_attn.k_proj.weight");
        let v = group_key("base_lm.layers.3.self_attn.v_proj.weight");
        assert_eq!(q, "base_lm.layers.3.self_attn");
        assert_eq!(q, k);
        assert_eq!(q, v);

        let g = group_key("base_lm.layers.3.mlp.gate_proj.weight");
        let u = group_key("base_lm.layers.3.mlp.up_proj.weight");
        assert_eq!(g, "base_lm.layers.3.mlp");
        assert_eq!(g, u);

        let wg = group_key("decoder.model.2.block.1.weight_g");
        let wv = group_key("decoder.model.2.block.1.weight_v");
        assert_eq!(wg, "decoder.model.2.block");
        assert_eq!(wg, wv);
    }

    #[test]
    fn group_key_handles_short_names() {
        assert_eq!(group_key("weight"), "weight");
        assert_eq!(group_key("a.weight"), "a.weight");
        assert_eq!(group_key("a.b.weight"), "a");
    }

    #[test]
    fn plan_never_splits_a_fusion_group() {
        // Two attention layers plus a stray norm, laid out back to back.
        let mut entries = Vec::new();
        let mut offset = 0u64;
        let push = |name: &str, nbytes: u64, entries: &mut Vec<TensorEntry>, offset: &mut u64| {
            entries.push(TensorEntry {
                name: name.to_string(),
                dtype: Dtype::BF16,
                shape: vec![(nbytes / 2) as usize],
                offset: *offset,
                nbytes,
            });
            *offset += nbytes;
        };
        for layer in 0..2 {
            for part in ["q_proj", "k_proj", "v_proj"] {
                push(
                    &format!("lm.layers.{layer}.self_attn.{part}.weight"),
                    4 * 1024 * 1024,
                    &mut entries,
                    &mut offset,
                );
            }
            push(
                &format!("lm.layers.{layer}.input_layernorm.weight"),
                1024,
                &mut entries,
                &mut offset,
            );
        }
        let checkpoint = Checkpoint {
            entries,
            data_start: 0,
            file_size: offset,
        };

        // A budget far smaller than one attention group: the group must
        // still arrive intact, or `fuse_qkv` would see only part of it.
        let batches = checkpoint.plan(1024);
        for layer in 0..2 {
            let key = format!("lm.layers.{layer}.self_attn");
            let holding: Vec<&Batch> = batches
                .iter()
                .filter(|b| {
                    b.indices
                        .iter()
                        .any(|i| group_key(&checkpoint.entries[*i].name) == key)
                })
                .collect();
            assert_eq!(holding.len(), 1, "group {key} was split across batches");
            let n = holding[0]
                .indices
                .iter()
                .filter(|i| group_key(&checkpoint.entries[**i].name) == key)
                .count();
            assert_eq!(n, 3, "group {key} lost members");
        }

        // Every tensor appears exactly once overall.
        let mut seen: Vec<usize> = batches.iter().flat_map(|b| b.indices.clone()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..checkpoint.entries.len()).collect::<Vec<_>>());
    }
}
