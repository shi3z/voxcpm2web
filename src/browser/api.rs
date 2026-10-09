//! The JavaScript-facing API.
//!
//! Everything here is `async`, because every step the browser cares about
//! is: `requestAdapter` is async, `fetch` is async, and GPU readback is
//! `mapAsync` (see [`crate::browser`]).
//!
//! Every entry point returns `Result<_, JsValue>` carrying the real
//! [`crate::Error`] text, so a failure shows up in JS as a readable
//! message rather than `RuntimeError: unreachable`.

use burn::prelude::*;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

use super::fetch::HttpRangeSource;
use crate::stream::{self, Checkpoint, MemorySource, ProgressSink};
use crate::weights::{self, Namespace};
use crate::{GenerateOptions, Prompt, PromptAudio, TextTokenizer, VoxCpm2Config, VoxCPM};

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

/// The browser backend: cubecl's WGSL path over `wgpu`, which selects
/// `wgpu::Backend::BrowserWebGpu` on wasm.
///
/// `fusion` and `autotune` are deliberately absent — burn-wgpu's own docs
/// advise disabling fusion on wasm, and autotune wants to time kernels,
/// which is awkward in a browser. That is why the `webgpu` feature derives
/// from `burn/wgpu` and not from this crate's `wgpu-fast`.
///
/// Precision is a build-time switch (`webgpu-f16`); see `Cargo.toml` for
/// what it costs in VRAM.
#[cfg(not(feature = "webgpu-f16"))]
pub type WebBackend = burn::backend::Wgpu<f32, i32>;
#[cfg(feature = "webgpu-f16")]
pub type WebBackend = burn::backend::Wgpu<half::f16, i32>;

/// Human-readable name of the compiled-in precision.
#[cfg(not(feature = "webgpu-f16"))]
pub const PRECISION: &str = "f32";
#[cfg(feature = "webgpu-f16")]
pub const PRECISION: &str = "f16";

type Device = <WebBackend as Backend>::Device;

fn to_js(e: crate::Error) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn err(msg: impl core::fmt::Display) -> JsValue {
    JsValue::from_str(&msg.to_string())
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

/// Install the panic hook and the `log` → `console` bridge.
///
/// Call this first, before anything else. `level` is one of `error`,
/// `warn`, `info`, `debug`, `trace`; anything else means `info`.
#[wasm_bindgen]
pub fn init(level: Option<String>) {
    let filter = match level.as_deref().unwrap_or("info") {
        "error" => log::LevelFilter::Error,
        "warn" => log::LevelFilter::Warn,
        "debug" => log::LevelFilter::Debug,
        "trace" => log::LevelFilter::Trace,
        _ => log::LevelFilter::Info,
    };
    super::init_logging(filter);
    log::info!(
        "voxcpm-rs browser runtime: precision={PRECISION}, backend=WebGPU (cubecl WGSL)"
    );
}

/// Whether a software-rasterizer adapter (SwiftShader, llvmpipe) may be
/// used. Off by default — see [`allow_software_adapter`].
static ALLOW_SOFTWARE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Permit running on a software WebGPU adapter.
///
/// Off by default, deliberately. Chrome will happily hand out a
/// SwiftShader adapter when it cannot reach the real GPU — on Linux, most
/// often because the browser process has no access to the DRM render node
/// (`/dev/dri/renderD*` is typically `root:render` and granted to the
/// seated user by ACL). WebGPU then "works", and this 2.3 B-parameter
/// model runs on the CPU at a few seconds per diffusion step instead of
/// on the GPU. Silently accepting that would make the demo look broken
/// rather than misconfigured, so [`init_webgpu`] refuses by default and
/// says what to fix.
///
/// Call this with `true` only to exercise the pipeline deliberately — it
/// is useful for verifying correctness in an environment with no GPU
/// access, and useless for measuring anything.
#[wasm_bindgen]
pub fn allow_software_adapter(allow: bool) {
    ALLOW_SOFTWARE.store(allow, core::sync::atomic::Ordering::Relaxed);
    if allow {
        log::warn!(
            "software WebGPU adapters are now permitted — inference will run on the CPU \
             and be orders of magnitude slower. Timings from such a run are meaningless."
        );
    }
}

/// Adapter identity as seen from JavaScript.
///
/// This has to come from JS. On the WebGPU backend `wgpu`'s
/// `Adapter::get_info()` returns all-empty values — `name: ""`,
/// `vendor: 0`, `device_type: Other` — because the WebGPU spec does not
/// expose adapter identity to the page's graphics API the way a native
/// driver does. The information *is* reachable from JS as `adapter.info`,
/// so the page hands it in before [`init_webgpu`] runs. Without this, the
/// software-adapter guard below could never fire and the UI would show a
/// blank adapter name.
#[derive(Debug, Clone, Default)]
struct AdapterHint {
    vendor: String,
    architecture: String,
    device: String,
    description: String,
    software: bool,
}

thread_local! {
    static ADAPTER_HINT: core::cell::RefCell<Option<AdapterHint>> =
        const { core::cell::RefCell::new(None) };
}

/// Hand Rust the adapter identity that only JavaScript can see.
///
/// Call before [`init_webgpu`], from
/// `(await navigator.gpu.requestAdapter()).info`. `software` should be the
/// page's own verdict on whether this is a rasterizer rather than a GPU
/// (`vendor === 'google' && architecture === 'swiftshader'`, `llvmpipe`,
/// and friends).
#[wasm_bindgen]
pub fn set_adapter_hint(
    vendor: String,
    architecture: String,
    device: String,
    description: String,
    software: bool,
) {
    ADAPTER_HINT.with(|h| {
        *h.borrow_mut() = Some(AdapterHint {
            vendor,
            architecture,
            device,
            description,
            software,
        })
    });
}

/// Heuristic fallback for when no hint was supplied: does the name `wgpu`
/// reported look like a software rasterizer? Useful only on backends that
/// actually fill `AdapterInfo` in.
fn looks_like_software(name: &str, driver: &str, device_type: &str) -> bool {
    if device_type == "Cpu" {
        return true;
    }
    let haystack = format!("{name} {driver}").to_lowercase();
    ["swiftshader", "llvmpipe", "softpipe", "software", "lavapipe", "basic render"]
        .iter()
        .any(|needle| haystack.contains(needle))
}

/// Set a debug flag (the browser stand-in for an environment variable).
///
/// * `VOXCPM_Z_ZERO` — replace the diffusion sampler's Gaussian noise with
///   zeros, making generation deterministic. Use it to diff browser output
///   against a native run numerically — **not** to judge quality: the
///   noise is part of the flow-matching sampler, and zeroing it measurably
///   degrades the output (voiced frames ~57% → ~24% on a native reference
///   run, and the stop head stops firing, so generation runs to
///   `max_len`).
/// * `VOXCPM_PROFILE` — log per-step AR timings.
#[wasm_bindgen]
pub fn set_flag(name: String, on: bool) {
    super::set_debug_flag(&name, on);
}

/// Initialize the WebGPU device and report what we got.
///
/// Must be awaited before any tensor exists: `cubecl::wgpu::init_setup`
/// panics on wasm ("Creating a wgpu setup synchronously is unsupported"),
/// so the async variant is the only way in.
///
/// Returns a JSON string with the adapter info and the limits that matter
/// for this model.
#[wasm_bindgen]
pub async fn init_webgpu() -> Result<String, JsValue> {
    use burn::backend::wgpu::{graphics::WebGpu, init_setup_async};

    // Setup runs at most once per page: cubecl registers one compute client
    // per device and `ComputeClient::init` panics with "Can't create a new
    // client on an already registered server" on a second call. Both
    // `webgpu_self_test` and `load_model` get here, and a caller may also
    // retry after being refused below, so the *setup* is cached while the
    // *guards* are re-evaluated on every call.
    if let Some(cached) = REPORT.with(|r| r.borrow().clone()) {
        let report: AdapterReport = serde_json::from_str(&cached)
            .map_err(|e| err(format!("cached adapter report is unreadable: {e}")))?;
        enforce_guards(&report)?;
        return Ok(cached);
    }

    let device = Device::default();
    let setup = init_setup_async::<WebGpu>(&device, Default::default()).await;

    let info = setup.adapter.get_info();
    let limits = setup.device.limits();
    // `wgpu::Features` is not reachable by name: `burn-wgpu` re-exports only
    // selected items from `cubecl::wgpu`, and neither re-exports the `wgpu`
    // crate itself. Adding a direct `wgpu` dependency risks resolving a
    // second, type-incompatible copy, so the feature set is inspected
    // through its `Debug` rendering instead — a bitflags list of names.
    let features = format!("{:?}", setup.adapter.features());
    let has_f16 = features.contains("SHADER_F16");

    let device_type = format!("{:?}", info.device_type);
    let hint = ADAPTER_HINT.with(|h| h.borrow().clone());

    // Prefer the page's view of who the adapter is, since `wgpu` cannot
    // see it on the WebGPU backend.
    let (name, software) = match &hint {
        Some(h) => {
            let label = [h.vendor.as_str(), h.architecture.as_str(), h.device.as_str()]
                .iter()
                .filter(|s| !s.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(" / ");
            let label = if label.is_empty() {
                h.description.clone()
            } else {
                label
            };
            (label, h.software)
        }
        None => (
            info.name.clone(),
            looks_like_software(&info.name, &info.driver, &device_type),
        ),
    };
    let name = if name.is_empty() {
        "(not exposed)".to_string()
    } else {
        name
    };

    let report = AdapterReport {
        backend: format!("{:?}", info.backend),
        name,
        driver: info.driver.clone(),
        driver_info: hint
            .as_ref()
            .map(|h| h.description.clone())
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| info.driver_info.clone()),
        device_type,
        software,
        adapter_info_from_page: hint.is_some(),
        shader_f16: has_f16,
        precision: PRECISION.to_string(),
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
    };

    log::info!("WebGPU adapter: {} ({})", report.name, report.backend);
    log::info!(
        "limits: maxBufferSize={} maxStorageBufferBindingSize={} shader-f16={}",
        report.max_buffer_size,
        report.max_storage_buffer_binding_size,
        report.shader_f16
    );

    if !report.adapter_info_from_page {
        log::warn!(
            "no adapter hint from the page — `wgpu` reports no identity on the WebGPU \
             backend, so a software adapter cannot be detected. Call \
             set_adapter_hint() before init_webgpu()."
        );
    }

    // Cache the report *before* enforcing the guards: the GPU device now
    // exists either way, and a caller who fixes the condition (ticks
    // "allow software") must be able to retry without re-running setup.
    let json = serde_json::to_string(&report)
        .map_err(|e| err(format!("serialize adapter report: {e}")))?;
    REPORT.with(|r| *r.borrow_mut() = Some(json.clone()));

    enforce_guards(&report)?;
    Ok(json)
}

/// Refuse to proceed on an adapter this model cannot usefully run on.
///
/// Evaluated on every [`init_webgpu`] call, not just the first, so the
/// decision tracks the current [`allow_software_adapter`] setting.
fn enforce_guards(report: &AdapterReport) -> Result<(), JsValue> {
    if report.software && !ALLOW_SOFTWARE.load(core::sync::atomic::Ordering::Relaxed) {
        return Err(err(format!(
            "WebGPU returned a SOFTWARE adapter ({}, device_type={}), not a GPU. \
             This model has 2.3 B parameters; running it on a CPU rasterizer would take \
             hours, so it is refused rather than silently run.\n\n\
             On Linux this usually means the browser cannot open the DRM render node. \
             Check that your user is in the `render` and `video` groups:\n    \
             sudo usermod -aG render,video $USER   # then log out and back in\n\
             and that chrome://gpu reports hardware-accelerated WebGPU.\n\n\
             To run on the software adapter anyway (correctness testing only, timings \
             meaningless), call allow_software_adapter(true) first.",
            report.name, report.device_type
        )));
    }

    if PRECISION == "f16" && !report.shader_f16 {
        return Err(err(
            "this build is compiled for f16 weights but the adapter does not report \
             `shader-f16`. Rebuild without the `webgpu-f16` feature, or use a GPU \
             that supports 16-bit shader arithmetic.",
        ));
    }
    Ok(())
}

thread_local! {
    /// The adapter report from the one [`init_webgpu`] that ran setup.
    static REPORT: core::cell::RefCell<Option<String>> =
        const { core::cell::RefCell::new(None) };
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AdapterReport {
    backend: String,
    name: String,
    driver: String,
    driver_info: String,
    device_type: String,
    /// `true` if this looks like a software rasterizer rather than a GPU.
    software: bool,
    /// Whether the identity above came from the page's `adapter.info`
    /// rather than from `wgpu` (which reports nothing on WebGPU).
    adapter_info_from_page: bool,
    shader_f16: bool,
    precision: String,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u32,
    max_compute_workgroup_storage_size: u32,
    max_compute_invocations_per_workgroup: u32,
    max_compute_workgroups_per_dimension: u32,
}

// ---------------------------------------------------------------------------
// Milestone 2: prove real GPU compute before touching the model
// ---------------------------------------------------------------------------

/// Run a tiny matmul through the exact backend VoxCPM2 will use, and verify
/// the result numerically.
///
/// `A` is `[n, n]` of ones, `B` is the `[n, n]` identity scaled by 2, so
/// `A @ B` must be every element `== 2.0` and the sum must be `2 * n * n`.
/// A CPU fallback or a silently-broken pipeline fails this.
///
/// Returns a short human-readable report; errors carry the mismatch.
#[wasm_bindgen]
pub async fn webgpu_self_test() -> Result<String, JsValue> {
    let adapter = init_webgpu().await?;
    let device = Device::default();
    let n = 64usize;

    let a = Tensor::<WebBackend, 2>::ones([n, n], &device);
    let b = Tensor::<WebBackend, 2>::eye(n, &device).mul_scalar(2.0);
    let c = a.matmul(b);

    let data = c
        .into_data_async()
        .await
        .map_err(|e| err(format!("matmul readback failed: {e:?}")))?;
    let values: Vec<f32> = data
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|_| err("unexpected self-test dtype"))?;

    if values.len() != n * n {
        return Err(err(format!(
            "matmul returned {} elements, expected {}",
            values.len(),
            n * n
        )));
    }
    if let Some((i, v)) = values
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - 2.0).abs() > 1e-3)
    {
        return Err(err(format!(
            "matmul test FAILED: element {i} is {v}, expected 2.0"
        )));
    }
    let sum: f64 = values.iter().map(|v| *v as f64).sum();
    let expected = 2.0 * (n * n) as f64;
    if (sum - expected).abs() > expected * 1e-4 {
        return Err(err(format!(
            "matmul test FAILED: sum {sum} != expected {expected}"
        )));
    }

    // Reductions get their own battery, because matmul passing says
    // nothing about them and the model leans on them hard: every RMSNorm
    // is a mean, and the stop head is an argmax. A broken reduce yields
    // audio that is wrong rather than absent.
    //
    // Every probe runs and reports its value even after one fails, so a
    // single run localizes the fault — length-dependent errors point at
    // line-size / vectorization, a correct sum with a wrong mean points at
    // the divisor, and so on.
    let mut report: Vec<String> = Vec::new();
    let mut failures = 0usize;

    async fn scalar_of(t: Tensor<WebBackend, 1>) -> Result<f32, JsValue> {
        let v = t
            .into_data_async()
            .await
            .map_err(|e| err(format!("reduce readback failed: {e:?}")))?
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|_| err("unexpected reduce dtype"))?;
        v.first()
            .copied()
            .ok_or_else(|| err("reduce returned no elements"))
    }

    let mut check = |label: &str, got: f32, want: f32, report: &mut Vec<String>| {
        // f16 has ~3 decimal digits, so scale the tolerance with the value.
        let tol = (want.abs() * 1e-2).max(1e-2);
        if (got - want).abs() <= tol {
            report.push(format!("  {label:<28} {got:>12.4}   ok"));
        } else {
            failures += 1;
            report.push(format!("  {label:<28} {got:>12.4}   FAIL, expected {want}"));
        }
    };

    let t4 = || Tensor::<WebBackend, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0], &device);
    check("sum([1,2,3,4])", scalar_of(t4().sum()).await?, 10.0, &mut report);
    check("mean([1,2,3,4])", scalar_of(t4().mean()).await?, 2.5, &mut report);
    check("max([1,2,3,4])", scalar_of(t4().max()).await?, 4.0, &mut report);
    check("min([1,2,3,4])", scalar_of(t4().min()).await?, 1.0, &mut report);

    // A reduce that reads the same element repeatedly — the classic
    // line-size / stride bug — still gets `ones` exactly right, so these
    // use non-uniform data. `mean([1,2,3,4]) == 1` (the value actually
    // observed on f16 hardware) is what summing x[0] four times and
    // dividing by four produces, so this is the shape of bug to probe
    // for.
    let skew = || Tensor::<WebBackend, 1>::from_floats([5.0f32, 1.0, 1.0, 1.0], &device);
    // Reading x[0] four times would give sum 20 / mean 5 instead.
    check("sum([5,1,1,1])", scalar_of(skew().sum()).await?, 8.0, &mut report);
    check("mean([5,1,1,1])", scalar_of(skew().mean()).await?, 2.0, &mut report);

    // Length sweep over ones: catches an outright broken element count.
    for len in [1usize, 2, 3, 8, 64, 1024] {
        let ones = Tensor::<WebBackend, 1>::ones([len], &device);
        check(
            &format!("sum(ones[{len}])"),
            scalar_of(ones.clone().sum()).await?,
            len as f32,
            &mut report,
        );
        check(
            &format!("mean(ones[{len}])"),
            scalar_of(ones.mean()).await?,
            1.0,
            &mut report,
        );
    }

    // Length sweep over an alternating 0,1 pattern: the sum is
    // index-sensitive, so a stride bug shows up as 0 or as the full
    // length instead of half. Values stay small enough to be exact in
    // f16 at every length here.
    for len in [8usize, 64, 1024] {
        let alt: Vec<f32> = (0..len).map(|i| (i % 2) as f32).collect();
        let t = Tensor::<WebBackend, 1>::from_data(
            burn::tensor::TensorData::new(alt, [len]),
            &device,
        );
        check(
            &format!("sum([0,1,0,1,...][{len}])"),
            scalar_of(t.sum()).await?,
            (len / 2) as f32,
            &mut report,
        );
    }

    // The model never takes a whole-tensor reduce in its hot path: every
    // RMSNorm reduces the *last dimension* of a 2-D activation, which is
    // a different cubek-reduce routine. Probe that separately.
    let rows = Tensor::<WebBackend, 2>::from_data(
        burn::tensor::TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 4],
        ),
        &device,
    );
    let per_row = rows
        .mean_dim(1)
        .into_data_async()
        .await
        .map_err(|e| err(format!("mean_dim readback failed: {e:?}")))?
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|_| err("unexpected mean_dim dtype"))?;
    check(
        "mean_dim([[1..4],[5..8]],1)[0]",
        per_row.first().copied().unwrap_or(f32::NAN),
        2.5,
        &mut report,
    );
    check(
        "mean_dim([[1..4],[5..8]],1)[1]",
        per_row.get(1).copied().unwrap_or(f32::NAN),
        6.5,
        &mut report,
    );

    // argmax is what the stop head uses to decide when to stop talking.
    let t = Tensor::<WebBackend, 1>::from_floats([0.1f32, 0.9, 0.3, 0.2], &device);
    let arg = t
        .argmax(0)
        .into_data_async()
        .await
        .map_err(|e| err(format!("argmax readback failed: {e:?}")))?
        .iter::<i64>()
        .next()
        .unwrap_or(-1);
    if arg == 1 {
        report.push(format!("  {:<28} {arg:>12}   ok", "argmax([.1,.9,.3,.2])"));
    } else {
        failures += 1;
        report.push(format!(
            "  {:<28} {arg:>12}   FAIL, expected 1",
            "argmax([.1,.9,.3,.2])"
        ));
    }

    let body = format!(
        "WebGPU available\nadapter: {adapter}\nprecision: {PRECISION}\n\n\
         matrix test: PASS ({n}x{n} matmul, sum={sum})\n\n\
         reduce tests:\n{}",
        report.join("\n")
    );

    if failures > 0 {
        return Err(err(format!(
            "{body}\n\n{failures} reduce probe(s) FAILED. Reductions back every \
             RMSNorm and the stop head, so this build would produce wrong audio \
             rather than none.\n\n\
             If precision is f16, use f32 instead (append ?precision=f32) — f32 is \
             verified to match a native run exactly. Please report this output along \
             with the Diagnostics panel."
        )));
    }

    Ok(format!("{body}\n\nall reduce probes PASS"))
}

// ---------------------------------------------------------------------------
// Progress plumbing
// ---------------------------------------------------------------------------

/// Forwards [`ProgressSink`] events to a JS callback
/// `(stage, done, total, detail) => void`.
struct JsProgress {
    callback: Option<js_sys::Function>,
}

impl ProgressSink for JsProgress {
    fn report(&self, stage: &str, done: u64, total: u64, detail: &str) {
        let pct = if total == 0 {
            100.0
        } else {
            done as f64 * 100.0 / total as f64
        };
        log::debug!("[{stage}] {pct:.1}% {detail}");
        if let Some(cb) = &self.callback {
            let args = js_sys::Array::new();
            args.push(&JsValue::from_str(stage));
            args.push(&JsValue::from_f64(done as f64));
            args.push(&JsValue::from_f64(total as f64));
            args.push(&JsValue::from_str(detail));
            // A throwing callback must not abort the load.
            if let Err(e) = cb.apply(&JsValue::NULL, &args) {
                log::warn!("progress callback threw: {e:?}");
            }
        }
    }
}

impl JsProgress {
    fn stage(&self, stage: &str, detail: &str) {
        self.report(stage, 0, 0, detail);
    }
}

// ---------------------------------------------------------------------------
// Model session
// ---------------------------------------------------------------------------

/// A loaded VoxCPM2 model, resident on the GPU.
///
/// Weights are uploaded once by [`load_model`] and stay on the GPU;
/// [`VoxCpmSession::generate`] reuses them. Keep the handle alive for the
/// lifetime of the page.
#[wasm_bindgen]
#[derive(Debug)]
pub struct VoxCpmSession {
    inner: VoxCPM<WebBackend>,
    load_ms: f64,
}

/// Where each checkpoint file comes from.
///
/// Every field is an absolute URL or a path relative to the page. Any
/// field left out falls back to `{base}/{filename}`, so the common
/// same-origin case is just `{"base": "/models"}`.
///
/// Mixed sources are the point: a page on GitHub Pages cannot host a
/// 4.37 GB file (GitHub rejects any single file over 100 MB, and a Pages
/// site is capped at 1 GB), but it *can* stream the checkpoint
/// cross-origin from Hugging Face, which serves `Range` requests with
/// `Access-Control-Allow-Origin` and exposes `Content-Range`.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct ModelSources {
    /// Fallback prefix for any field not given.
    base: Option<String>,
    config: Option<String>,
    tokenizer: Option<String>,
    model: Option<String>,
    audiovae: Option<String>,
}

impl ModelSources {
    fn resolve(&self, field: &Option<String>, filename: &str) -> Result<String, JsValue> {
        if let Some(u) = field {
            return Ok(u.clone());
        }
        match &self.base {
            Some(b) => Ok(format!("{}/{filename}", b.trim_end_matches('/'))),
            None => Err(err(format!(
                "no source for `{filename}`: give it explicitly or set `base`"
            ))),
        }
    }
}

/// Download a checkpoint and build a model on the GPU.
///
/// `sources_json` is a JSON object naming where each file lives — see
/// [`ModelSources`]. The simplest form is `{"base":"/models"}`; to stream
/// the weights from Hugging Face instead:
///
/// ```json
/// {
///   "base":  "https://huggingface.co/openbmb/VoxCPM2/resolve/main",
///   "audiovae": "https://huggingface.co/you/your-repo/resolve/main/audiovae.safetensors"
/// }
/// ```
///
/// | file | how it is read |
/// |---|---|
/// | `config.json` | whole |
/// | `tokenizer.json` | whole |
/// | `model.safetensors` | **streamed** via HTTP Range, group at a time |
/// | `audiovae.safetensors` | whole (359 MB), or handed in as `audiovae_bytes` |
///
/// `model.safetensors` is 4.37 GB and `wasm32` has 4 GB of address space,
/// so it is never materialized: only its header is parsed up front, then
/// tensors are fetched, converted and uploaded in batches. Whatever serves
/// it must honour `Range` requests; the loader fails with a clear message
/// if it does not.
///
/// `audiovae_bytes` lets the page supply the AudioVAE directly — from an
/// `<input type="file">`, Cache Storage, or OPFS — which is the way out of
/// a real hosting problem: upstream ships only `audiovae.pth`, a Python
/// pickle this build cannot read, and no safetensors version of it is
/// published anywhere. Convert it once with the `convert_audiovae`
/// example, then either host the result somewhere CORS-friendly or just
/// pick the local file.
///
/// `progress` is an optional `(stage, done, total, detail) => void`.
#[wasm_bindgen]
pub async fn load_model(
    sources_json: String,
    audiovae_bytes: Option<js_sys::Uint8Array>,
    progress: Option<js_sys::Function>,
    batch_budget_bytes: Option<f64>,
) -> Result<VoxCpmSession, JsValue> {
    let t_total = crate::compat::Stopwatch::start();
    let p = JsProgress { callback: progress };
    let sources: ModelSources = serde_json::from_str(&sources_json)
        .map_err(|e| err(format!("bad model sources JSON: {e}")))?;
    let budget = batch_budget_bytes
        .filter(|b| *b >= 1.0)
        .map(|b| b as u64)
        .unwrap_or(stream::DEFAULT_BATCH_BUDGET);

    // The device must exist before any tensor is created.
    p.stage("init", "initializing WebGPU");
    init_webgpu().await?;
    let device = Device::default();

    // --- config.json ------------------------------------------------------
    let config_url = sources.resolve(&sources.config, "config.json")?;
    p.stage("config", &format!("downloading {config_url}"));
    let config_bytes = super::fetch::fetch_bytes(&config_url).await.map_err(to_js)?;
    let config: VoxCpm2Config =
        serde_json::from_slice(&config_bytes).map_err(|e| err(format!("config.json: {e}")))?;

    // --- tokenizer.json ---------------------------------------------------
    let tokenizer_url = sources.resolve(&sources.tokenizer, "tokenizer.json")?;
    p.stage("tokenizer", &format!("downloading {tokenizer_url}"));
    let tok_bytes = super::fetch::fetch_bytes(&tokenizer_url).await.map_err(to_js)?;
    let tokenizer = TextTokenizer::from_bytes(&tok_bytes).map_err(to_js)?;
    drop(tok_bytes);

    // --- build the (randomly-initialized) module tree ---------------------
    p.stage("model", "allocating model on GPU");
    let t_alloc = crate::compat::Stopwatch::start();
    let mut model = crate::voxcpm2::VoxCpm2Model::<WebBackend>::new(config, &device);
    log::info!("module tree allocated in {:.0}ms", t_alloc.elapsed_ms());

    let mut acc = weights::empty_apply_result();

    // --- model.safetensors, streamed --------------------------------------
    let model_url = sources.resolve(&sources.model, "model.safetensors")?;
    p.stage("weights", &format!("reading header of {model_url}"));
    let src = HttpRangeSource::open(&model_url).await.map_err(to_js)?;
    let checkpoint = Checkpoint::open(&src).await.map_err(to_js)?;
    let r = stream::stream_into::<WebBackend, _, _, _>(
        &mut model,
        &src,
        &checkpoint,
        Namespace::Model,
        budget,
        "weights",
        &p,
    )
    .await
    .map_err(to_js)?;
    weights::merge_apply_result(&mut acc, r);

    // --- audiovae.safetensors --------------------------------------------
    // Small enough (359 MB) to hold whole, and its weight_norm pairs want
    // to be grouped anyway.
    let (vae_label, vae_bytes) = match audiovae_bytes {
        Some(arr) => {
            // Supplied by the page — a local file, or a cache hit.
            p.stage("audiovae", "reading AudioVAE supplied by the page");
            let bytes = arr.to_vec();
            if bytes.is_empty() {
                return Err(err("audiovae_bytes is empty"));
            }
            ("audiovae (from page)".to_string(), bytes)
        }
        None => {
            let vae_url = sources.resolve(&sources.audiovae, "audiovae.safetensors")?;
            p.stage("audiovae", &format!("downloading {vae_url}"));
            let bytes = super::fetch::fetch_bytes(&vae_url).await.map_err(|e| {
                err(format!(
                    "{e}\n\nNo `audiovae.safetensors` at that URL. Upstream publishes \
                     only `audiovae.pth`, a Python pickle this build cannot read, and no \
                     safetensors version of it is hosted anywhere public.\n\n\
                     Convert it once:\n    \
                     cargo run --release --example convert_audiovae \\\n    \
                     --no-default-features --features cpu -- <checkpoint-dir>\n\n\
                     Then either host the result somewhere that sends CORS headers (a \
                     Hugging Face repo works; GitHub release assets do not), or just \
                     select the file in the page."
                ))
            })?;
            (vae_url, bytes)
        }
    };
    let vae_src = MemorySource::new(&vae_label, vae_bytes);
    let vae_checkpoint = Checkpoint::open(&vae_src).await.map_err(to_js)?;
    let r = stream::stream_into::<WebBackend, _, _, _>(
        &mut model,
        &vae_src,
        &vae_checkpoint,
        Namespace::AudioVae,
        budget,
        "audiovae",
        &p,
    )
    .await
    .map_err(to_js)?;
    weights::merge_apply_result(&mut acc, r);
    drop(vae_src);

    // --- report -----------------------------------------------------------
    weights::finalize_apply_result(&mut acc);
    crate::voxcpm2::wrapper::report_apply_result(&acc);
    if !acc.errors.is_empty() {
        return Err(err(format!(
            "weight load reported {} error(s); first: {:?}",
            acc.errors.len(),
            acc.errors[0]
        )));
    }
    if !acc.missing.is_empty() {
        // Not fatal — `new()` random-initialized everything, so the model
        // will still run. But it will sound wrong, so say so loudly.
        log::error!(
            "{} model parameter(s) were never supplied by the checkpoint and are \
             still randomly initialized — output will be wrong",
            acc.missing.len()
        );
    }

    let load_ms = t_total.elapsed_ms();
    p.stage("ready", "ready");
    log::info!(
        "model ready in {:.1}s ({} params applied)",
        load_ms / 1000.0,
        acc.applied.len()
    );

    Ok(VoxCpmSession {
        inner: VoxCPM::from_parts(model, tokenizer, &device),
        load_ms,
    })
}

#[wasm_bindgen]
impl VoxCpmSession {
    /// Output sample rate in Hz — feed this to `AudioContext`/`AudioBuffer`.
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// How long [`load_model`] took, in milliseconds.
    #[wasm_bindgen(getter)]
    pub fn load_ms(&self) -> f64 {
        self.load_ms
    }

    /// The float precision the weights are resident in on the GPU —
    /// `"f32"` or `"f16"`, fixed at build time by the `webgpu-f16`
    /// feature.
    #[wasm_bindgen(getter)]
    pub fn precision(&self) -> String {
        PRECISION.to_string()
    }

    /// Synthesize `text` and return mono PCM at [`Self::sample_rate`].
    ///
    /// Returns a `Float32Array` — a plain copy out of WASM memory, ready
    /// for `AudioBuffer.copyToChannel`. No WAV encoding happens in Rust;
    /// JavaScript owns playback.
    ///
    /// `reference_pcm` / `reference_sample_rate`, when given, clone that
    /// voice (Milestone 8). The samples must be mono `f32` in `[-1, 1]`;
    /// `AudioContext.decodeAudioData` + `getChannelData(0)` produces
    /// exactly that, and resampling to the model's rate happens in Rust.
    #[wasm_bindgen]
    pub async fn generate(
        &self,
        text: String,
        cfg_value: Option<f32>,
        timesteps: Option<usize>,
        max_len: Option<usize>,
        reference_pcm: Option<Float32Array>,
        reference_sample_rate: Option<u32>,
    ) -> Result<Float32Array, JsValue> {
        if text.trim().is_empty() {
            return Err(err("generate: text is empty"));
        }

        let mut builder = GenerateOptions::builder();
        if let Some(v) = cfg_value {
            builder = builder.cfg(v);
        }
        if let Some(v) = timesteps {
            builder = builder.timesteps(v);
        }
        if let Some(v) = max_len {
            builder = builder.max_len(v);
        }
        if let Some(pcm) = reference_pcm {
            let sr = reference_sample_rate
                .ok_or_else(|| err("reference_pcm given without reference_sample_rate"))?;
            let samples = pcm.to_vec();
            if samples.is_empty() {
                return Err(err("reference_pcm is empty"));
            }
            builder = builder.prompt(Prompt::Reference {
                audio: PromptAudio::Pcm {
                    samples,
                    sample_rate: sr,
                },
            });
        }

        let t = crate::compat::Stopwatch::start();
        let pcm = self
            .inner
            .generate_async(&text, builder.build())
            .await
            .map_err(to_js)?;
        let gen_ms = t.elapsed_ms();

        if pcm.is_empty() {
            return Err(err("generation produced 0 samples"));
        }
        // Catch a numerically broken run here rather than letting WebAudio
        // play silence or a click.
        let bad = pcm.iter().filter(|v| !v.is_finite()).count();
        if bad > 0 {
            return Err(err(format!(
                "generation produced {bad} non-finite sample(s) out of {} — \
                 NaN/Inf in the pipeline",
                pcm.len()
            )));
        }
        let peak = pcm.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let audio_s = pcm.len() as f64 / self.inner.sample_rate() as f64;
        log::info!(
            "generated {:.2}s of audio in {:.0}ms (RTF {:.3}), peak {:.4}",
            audio_s,
            gen_ms,
            gen_ms / 1000.0 / audio_s,
            peak
        );
        if peak < 1e-4 {
            log::warn!("output peak amplitude is {peak:.2e} — effectively silent");
        }

        Ok(Float32Array::from(pcm.as_slice()))
    }

    /// Streaming variant of [`Self::generate`]: emit PCM chunks as they
    /// are produced, instead of waiting for the whole utterance.
    ///
    /// `on_chunk` is called as `(Float32Array, chunkIndex, totalSamples)`
    /// for every chunk; the full waveform is also returned at the end so
    /// the caller can still save it.
    ///
    /// This is real incremental decoding, not a finished waveform cut into
    /// pieces: each chunk runs up to `chunk_patches` autoregressive steps
    /// and then decodes. Chunk boundaries are seamless because the
    /// AudioVAE decoder is causal, so re-decoding a longer prefix
    /// reproduces the earlier samples exactly.
    ///
    /// The cost of that simplicity is that each chunk re-decodes the whole
    /// accumulated latent, which is `O(N^2)` decode work across a long
    /// utterance. Smaller `chunk_patches` lowers time-to-first-audio and
    /// raises total work; 5 (~400 ms of audio) is a reasonable balance.
    #[wasm_bindgen]
    pub async fn generate_streaming(
        &self,
        text: String,
        on_chunk: js_sys::Function,
        cfg_value: Option<f32>,
        timesteps: Option<usize>,
        max_len: Option<usize>,
        chunk_patches: Option<usize>,
        reference_pcm: Option<Float32Array>,
        reference_sample_rate: Option<u32>,
    ) -> Result<Float32Array, JsValue> {
        if text.trim().is_empty() {
            return Err(err("generate_streaming: text is empty"));
        }

        let mut builder = GenerateOptions::builder();
        if let Some(v) = cfg_value {
            builder = builder.cfg(v);
        }
        if let Some(v) = timesteps {
            builder = builder.timesteps(v);
        }
        if let Some(v) = max_len {
            builder = builder.max_len(v);
        }
        if let Some(v) = chunk_patches {
            builder = builder.chunk_patches(v);
        }
        if let Some(pcm) = reference_pcm {
            let sr = reference_sample_rate
                .ok_or_else(|| err("reference_pcm given without reference_sample_rate"))?;
            let samples = pcm.to_vec();
            if samples.is_empty() {
                return Err(err("reference_pcm is empty"));
            }
            builder = builder.prompt(Prompt::Reference {
                audio: PromptAudio::Pcm {
                    samples,
                    sample_rate: sr,
                },
            });
        }

        let t0 = crate::compat::Stopwatch::start();
        let mut stream = self
            .inner
            .generate_stream(&text, builder.build())
            .map_err(to_js)?;

        let mut all: Vec<f32> = Vec::new();
        let mut index = 0usize;
        let mut first_chunk_ms: Option<f64> = None;

        while let Some(chunk) = stream.next_chunk_async().await {
            let chunk = chunk.map_err(to_js)?;
            if first_chunk_ms.is_none() {
                let ms = t0.elapsed_ms();
                first_chunk_ms = Some(ms);
                log::info!("time to first audio: {ms:.0}ms");
            }
            all.extend_from_slice(&chunk);

            let args = js_sys::Array::new();
            args.push(&Float32Array::from(chunk.as_slice()));
            args.push(&JsValue::from_f64(index as f64));
            args.push(&JsValue::from_f64(all.len() as f64));
            if let Err(e) = on_chunk.apply(&JsValue::NULL, &args) {
                log::warn!("chunk callback threw: {e:?}");
            }
            index += 1;
        }

        if all.is_empty() {
            return Err(err("streaming generation produced 0 samples"));
        }
        let bad = all.iter().filter(|v| !v.is_finite()).count();
        if bad > 0 {
            return Err(err(format!(
                "streaming generation produced {bad} non-finite sample(s) out of {}",
                all.len()
            )));
        }
        log::info!(
            "streamed {} chunks, {:.2}s of audio in {:.0}ms (first audio at {:.0}ms)",
            index,
            all.len() as f64 / self.inner.sample_rate() as f64,
            t0.elapsed_ms(),
            first_chunk_ms.unwrap_or(0.0),
        );
        Ok(Float32Array::from(all.as_slice()))
    }

    /// Encode PCM as a 16-bit WAV, for the "save" button.
    ///
    /// Playback does not need this — JS gets raw PCM from
    /// [`Self::generate`]. This is only for handing the user a file.
    #[wasm_bindgen]
    pub fn encode_wav(&self, pcm: Float32Array) -> Result<Vec<u8>, JsValue> {
        crate::audio::encode_wav(&pcm.to_vec(), self.inner.sample_rate()).map_err(to_js)
    }
}
