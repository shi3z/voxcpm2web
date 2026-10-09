//! Exercise the browser's streaming checkpoint loader on native hardware.
//!
//! [`voxcpm_rs::stream`] is the code path the browser uses to get a 4.37 GB
//! checkpoint onto the GPU without ever holding it in memory. It is
//! platform-independent and sits behind
//! [`ByteSource`][voxcpm_rs::stream::ByteSource], so running it here over a
//! [`FileSource`][voxcpm_rs::stream::FileSource] tests the header parsing,
//! the grouping rule, the single-tensor fast path and the incremental
//! burn-store application — everything except `fetch` — with a real
//! debugger and real error messages attached.
//!
//! It is also useful in its own right: the default loader needs roughly
//! 22 GB of RAM to apply this checkpoint at F32, and this needs a few
//! hundred MB.
//!
//! ```sh
//! # same WGSL compiler the browser uses, same precision
//! cargo run --release --example stream_load \
//!     --no-default-features --features wgpu -- \
//!     /path/to/VoxCPM2 "こんにちは。" /tmp/stream.wav
//! ```

#![recursion_limit = "256"]

use std::path::PathBuf;

use voxcpm_rs::stream::{self, Checkpoint, FileSource, LogProgress};
use voxcpm_rs::weights::{self, Namespace};
use voxcpm_rs::{GenerateOptions, TextTokenizer, VoxCPM, VoxCpm2Config};

/// The default element type, matching the browser's `webgpu` build.
#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
type B = burn::backend::Wgpu<f32, i32>;
/// `--f16`: matches the browser's `webgpu-f16` build. Same WGSL compiler
/// path, so this is the closest thing to testing that build without a
/// browser that can reach a GPU.
#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
type BF16 = burn::backend::Wgpu<half::f16, i32>;
#[cfg(all(feature = "wgpu", feature = "vulkan"))]
type B = burn::backend::Vulkan<half::bf16, i32>;
#[cfg(all(feature = "wgpu", feature = "vulkan"))]
type BF16 = burn::backend::Vulkan<half::f16, i32>;
#[cfg(all(not(feature = "wgpu"), not(feature = "vulkan"), feature = "cpu"))]
type B = burn::backend::NdArray<f32>;
#[cfg(all(not(feature = "wgpu"), not(feature = "vulkan"), feature = "cpu"))]
type BF16 = burn::backend::NdArray<half::f16>;

/// A global allocator that tracks live and peak heap bytes.
///
/// This, not RSS, is the number that maps to WASM linear memory. Native
/// `VmHWM` also counts the GPU driver's host-side mappings of device
/// memory — on this checkpoint that is most of the 8.7 GB of F32 weights,
/// which in the browser live in GPU memory and never touch the wasm heap.
mod track {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub static LIVE: AtomicUsize = AtomicUsize::new(0);
    pub static PEAK: AtomicUsize = AtomicUsize::new(0);

    pub struct Tracking;

    impl Tracking {
        fn add(n: usize) {
            let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
    }

    unsafe impl GlobalAlloc for Tracking {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(layout) };
            if !p.is_null() {
                Self::add(layout.size());
            }
            p
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let p = unsafe { System.realloc(ptr, layout, new_size) };
            if !p.is_null() {
                if new_size >= layout.size() {
                    Self::add(new_size - layout.size());
                } else {
                    LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
                }
            }
            p
        }
    }

    /// Peak live heap in MB since process start.
    pub fn peak_mb() -> f64 {
        PEAK.load(Ordering::Relaxed) as f64 / 1048576.0
    }

    /// Currently live heap in MB.
    pub fn live_mb() -> f64 {
        LIVE.load(Ordering::Relaxed) as f64 / 1048576.0
    }

    /// Reset the peak to the current live value.
    pub fn reset_peak() {
        PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: track::Tracking = track::Tracking;

/// Peak resident set size in MB, from `/proc/self/status`.
fn peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        "info,wgpu_hal=error,wgpu_core=error,naga=error,cubecl_wgpu=warn",
    ))
    .init();

    // Flags may appear anywhere; positionals are <dir> [text] [out.wav].
    let mut positional: Vec<String> = Vec::new();
    let mut budget_mb = stream::DEFAULT_BATCH_BUDGET / (1024 * 1024);
    let mut use_async = false;
    let mut use_stream = false;
    let mut use_f16 = false;
    let mut max_len: Option<usize> = None;
    let mut argv = std::env::args().skip(1);
    while let Some(a) = argv.next() {
        match a.as_str() {
            "--budget-mb" => {
                budget_mb = argv.next().and_then(|v| v.parse().ok()).unwrap_or(budget_mb);
            }
            "--max-len" => max_len = argv.next().and_then(|v| v.parse().ok()),
            "--async" => use_async = true,
            "--f16" => use_f16 = true,
            "--stream" => {
                use_stream = true;
                use_async = true;
            }
            "-h" | "--help" => {
                println!(
                    "usage: stream_load <checkpoint-dir> [text] [out.wav] \
                     [--budget-mb N] [--max-len N] [--async] [--stream] [--f16]\n\n\
                     --async   drive generation through `generate_async`, the browser's\n\
                     \x20         code path, instead of the blocking `generate`.\n\
                     --stream  drive it through `GenerateStream::next_chunk_async`,\n\
                     \x20         the browser's streaming path; reports time-to-first-audio.\n\
                     --f16     run with f16 weights, matching the browser's\n\
                     \x20         `webgpu-f16` build instead of the default f32.\n\n\
                     Requires audiovae.safetensors — run the convert_audiovae example first."
                );
                return;
            }
            other => positional.push(other.to_string()),
        }
    }
    let dir = PathBuf::from(positional.first().cloned().unwrap_or_else(|| {
        eprintln!("usage: stream_load <checkpoint-dir> [text] [out.wav] [--budget-mb N] [--max-len N] [--async]");
        std::process::exit(2);
    }));
    let text = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| "こんにちは。これはブラウザだけで生成した音声です。".to_string());
    let out = positional
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/stream_load.wav".to_string());
    let budget = budget_mb * 1024 * 1024;

    if use_f16 {
        run::<BF16>(&dir, &text, &out, budget, use_async, use_stream, max_len, "f16");
    } else {
        run::<B>(&dir, &text, &out, budget, use_async, use_stream, max_len, "f32");
    }
}

/// The whole pipeline, generic over the element type so `--f16` and the
/// default share one implementation.
#[allow(clippy::too_many_arguments)]
fn run<Bk: burn::prelude::Backend>(
    dir: &std::path::Path,
    text: &str,
    out: &str,
    budget: u64,
    use_async: bool,
    use_stream: bool,
    max_len: Option<usize>,
    precision: &str,
) {
    println!("element type: {precision}");

    let device = Default::default();
    let t_total = std::time::Instant::now();

    // --- config + tokenizer, exactly as the browser does ----------------
    let config_bytes = std::fs::read(dir.join("config.json")).expect("read config.json");
    let config: VoxCpm2Config = serde_json::from_slice(&config_bytes).expect("parse config.json");
    let tok_bytes = std::fs::read(dir.join("tokenizer.json")).expect("read tokenizer.json");
    let tokenizer = TextTokenizer::from_bytes(&tok_bytes).expect("parse tokenizer.json");
    drop(tok_bytes);

    println!("allocating module tree...");
    let t = std::time::Instant::now();
    let mut model = voxcpm_rs::voxcpm2::VoxCpm2Model::<Bk>::new(config, &device);
    println!("  {:.1}s", t.elapsed().as_secs_f64());

    let mut acc = weights::empty_apply_result();

    // --- model.safetensors, streamed ------------------------------------
    let model_path = dir.join("model.safetensors");
    println!(
        "streaming {} (budget {} MB/batch)",
        model_path.display(),
        budget / (1024 * 1024)
    );
    let t = std::time::Instant::now();
    let src = FileSource::open(&model_path).expect("open model.safetensors");
    let checkpoint = stream::block_on_ready(Checkpoint::open(&src)).expect("parse header");
    println!(
        "  header: {} tensors, data starts at {}, file {} bytes",
        checkpoint.entries.len(),
        checkpoint.data_start,
        checkpoint.file_size
    );
    // The thing that would silently corrupt a 32-bit build:
    if checkpoint.file_size > u32::MAX as u64 {
        println!(
            "  note: file is {} bytes, past u32::MAX — offsets must be u64",
            checkpoint.file_size
        );
    }
    let batches = checkpoint.plan(budget);
    let biggest = batches.iter().map(|b| b.nbytes).max().unwrap_or(0);
    println!(
        "  plan: {} batches, largest {:.1} MB ({})",
        batches.len(),
        biggest as f64 / 1048576.0,
        batches
            .iter()
            .max_by_key(|b| b.nbytes)
            .map(|b| b.label())
            .unwrap_or_default()
    );

    let r = stream::block_on_ready(stream::stream_into::<Bk, _, _, _>(
        &mut model,
        &src,
        &checkpoint,
        Namespace::Model,
        budget,
        "weights",
        &LogProgress,
    ))
    .expect("stream model weights");
    weights::merge_apply_result(&mut acc, r);
    println!("  {:.1}s", t.elapsed().as_secs_f64());

    // --- audiovae.safetensors -------------------------------------------
    let vae_path = dir.join("audiovae.safetensors");
    if !vae_path.exists() {
        eprintln!(
            "\nmissing {}\nRun:\n  cargo run --release --example convert_audiovae \
             --no-default-features --features cpu -- {}",
            vae_path.display(),
            dir.display()
        );
        std::process::exit(1);
    }
    println!("streaming {}", vae_path.display());
    let t = std::time::Instant::now();
    let vae_src = FileSource::open(&vae_path).expect("open audiovae.safetensors");
    let vae_checkpoint = stream::block_on_ready(Checkpoint::open(&vae_src)).expect("parse vae header");
    let r = stream::block_on_ready(stream::stream_into::<Bk, _, _, _>(
        &mut model,
        &vae_src,
        &vae_checkpoint,
        Namespace::AudioVae,
        budget,
        "audiovae",
        &LogProgress,
    ))
    .expect("stream audiovae weights");
    weights::merge_apply_result(&mut acc, r);
    println!("  {:.1}s", t.elapsed().as_secs_f64());

    weights::finalize_apply_result(&mut acc);
    println!(
        "\nweights: applied={} missing={} unused={} errors={}",
        acc.applied.len(),
        acc.missing.len(),
        acc.unused.len(),
        acc.errors.len()
    );
    for (k, ctx) in acc.missing.iter().take(20) {
        println!("  MISSING {k} [{ctx}]");
    }
    for k in acc.unused.iter().take(20) {
        println!("  UNUSED  {k}");
    }
    for e in acc.errors.iter().take(20) {
        println!("  ERROR   {e:?}");
    }
    if !acc.errors.is_empty() || !acc.missing.is_empty() {
        eprintln!("\nload is not clean — generated audio would be wrong");
        std::process::exit(1);
    }
    println!("load total: {:.1}s", t_total.elapsed().as_secs_f64());
    println!(
        "heap: peak {:.0} MB, live {:.0} MB  <- this is the number that must fit \
         in wasm32's 4 GB address space",
        track::peak_mb(),
        track::live_mb()
    );
    if let Some(mb) = peak_rss_mb() {
        println!("RSS peak: {mb:.0} MB (includes GPU driver host mappings)");
    }
    track::reset_peak();

    // --- generate --------------------------------------------------------
    let voxcpm = VoxCPM::from_parts(model, tokenizer, &device);
    let mut opts = GenerateOptions::default();
    if let Some(n) = max_len {
        opts.max_len = n;
    }
    println!(
        "\ngenerating ({} path): {text}",
        if use_stream {
            "async streaming — browser"
        } else if use_async {
            "async — browser"
        } else {
            "sync — native"
        }
    );
    let t = std::time::Instant::now();
    let pcm = if use_stream {
        // The browser's streaming path. Each chunk is produced by running
        // up to `chunk_patches` AR steps and then decoding — real
        // incremental generation, not a finished waveform sliced up.
        let mut stream = voxcpm.generate_stream(text, opts).expect("generate_stream");
        let mut all: Vec<f32> = Vec::new();
        let mut first: Option<f64> = None;
        let mut chunks = 0usize;
        let mut prev = std::time::Instant::now();
        while let Some(chunk) = stream::block_on_ready(stream.next_chunk_async()) {
            let chunk = chunk.expect("stream chunk");
            let at = t.elapsed().as_secs_f64();
            if first.is_none() {
                first = Some(at);
            }
            println!(
                "  chunk {:>2}: {:>6} samples ({:.2}s audio) at t={:.1}s (+{:.1}s)",
                chunks,
                chunk.len(),
                chunk.len() as f64 / voxcpm.sample_rate() as f64,
                at,
                prev.elapsed().as_secs_f64(),
            );
            prev = std::time::Instant::now();
            all.extend_from_slice(&chunk);
            chunks += 1;
        }
        println!(
            "  {} chunks, time to first audio {:.2}s",
            chunks,
            first.unwrap_or(f64::NAN)
        );
        if chunks == 0 {
            eprintln!("FAIL: streaming produced no chunks");
            std::process::exit(1);
        }
        all
    } else if use_async {
        // `generate_async` is what the browser calls: it awaits the
        // per-step stop-head readback and the final waveform readback
        // instead of blocking on them. Running it here, on real GPU
        // hardware, checks that the async twin of the AR loop produces the
        // same thing as the sync one — the part of the port that a
        // software-rasterizer browser test cannot say anything useful
        // about.
        stream::block_on_ready(voxcpm.generate_async(text, opts)).expect("generate_async")
    } else {
        voxcpm.generate(text, opts).expect("generate")
    };
    let gen_s = t.elapsed().as_secs_f64();

    let sr = voxcpm.sample_rate();
    let audio_s = pcm.len() as f64 / sr as f64;
    let peak = pcm.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let nonfinite = pcm.iter().filter(|v| !v.is_finite()).count();
    let rms = (pcm.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / pcm.len() as f64).sqrt();

    println!(
        "{} samples @ {sr} Hz = {audio_s:.2}s in {gen_s:.1}s (RTF {:.2})",
        pcm.len(),
        gen_s / audio_s
    );
    println!("peak={peak:.4} rms={rms:.4} non-finite={nonfinite}");
    if nonfinite > 0 {
        eprintln!("FAIL: {nonfinite} non-finite samples");
        std::process::exit(1);
    }
    if peak < 1e-3 {
        eprintln!("FAIL: output is effectively silent (peak {peak:.2e})");
        std::process::exit(1);
    }

    voxcpm_rs::audio::write_wav(out, &pcm, sr).expect("write wav");
    println!("wrote {out}");

    // Raw little-endian f32, for comparing against the browser's
    // `Float32Array` without 16-bit WAV quantization in the way.
    if std::env::var("VOXCPM_DUMP_F32").is_ok() {
        let raw = format!("{out}.f32");
        let mut bytes = Vec::with_capacity(pcm.len() * 4);
        for v in &pcm {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&raw, &bytes).expect("write raw f32");
        println!("wrote {raw}");
    }
    println!(
        "generation heap: peak {:.0} MB, live {:.0} MB",
        track::peak_mb(),
        track::live_mb()
    );
}
