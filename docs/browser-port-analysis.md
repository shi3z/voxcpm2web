# VoxCPM2 → WASM + WebGPU: port analysis

**Milestone 0 deliverable.** Written against `voxcpm-rs` @ `006ae18` (v0.5.0),
`burn` 0.20.1 / `cubecl` 0.9.0 / `wgpu` 26.0.1, upstream checkpoint
`openbmb/VoxCPM2`.

Goal of the port: VoxCPM2 generating audible speech entirely in Chrome via
WASM + WebGPU, no inference server. This document is the survey that the
rest of the work is planned from — current architecture, every blocker
found, and what to do about each.

---

## 1. Current architecture

```
text
 │
 ├─ tokenizer.rs ............ HF `tokenizers` (LlamaTokenizerFast), plus
 │                            mask_multichar_chinese_tokens re-splitting
 ▼
voxcpm2/wrapper.rs ........... VoxCPM<B>: public `generate` / `generate_stream`,
 │                            prompt assembly, sentence-parallel batching
 ▼
voxcpm2/model.rs ............. VoxCpm2Model<B>: prefill → AR loop of
 │                            (dit_step → lm_step), stop head, latent stacking
 ├─ minicpm4/ ............... base LM (28 layers) + residual LM (8 layers):
 │                            attention w/ fused QKV, GQA, LongRoPE, static KV
 │                            cache, gated MLP w/ fused gate+up
 ├─ locenc.rs ............... LocEnc: patch encoder (12-layer transformer)
 ├─ locdit/ ................. LocDiT + UnifiedCFM: 12-layer DiT, Euler
 │                            flow-matching sampler, CFG, CFG-zero-star
 ├─ fsq.rs .................. finite scalar quantization helpers
 ▼
audiovae/ .................... encoder (prompt audio → latent) and decoder
 │                            (latent → waveform); causal convs, Snake
 │                            activations, weight_norm, sample-rate
 │                            conditioning (16 kHz in / 48 kHz out)
 ▼
Vec<f32> PCM → audio.rs → WAV
```

Support modules: `config.rs` (serde of `config.json`), `weights.rs`
(checkpoint → module params), `error.rs`, `audio.rs` (symphonia decode,
rubato resample, hound WAV).

### Model shape (from upstream `config.json`)

| Component | Config | Approx params |
|---|---|---|
| Base LM (MiniCPM4) | `hidden 2048`, `ffn 6144`, 28 layers, 16 heads / 2 KV heads, `kv_channels 128`, vocab 73448 | ~1.47 B |
| Residual LM (RALM) | same block shape, 8 layers, no RoPE | ~0.38 B |
| LocEnc | `hidden 1024`, `ffn 4096`, 12 layers | ~0.25 B |
| LocDiT | `hidden 1024`, `ffn 4096`, 12 layers | ~0.25 B |
| AudioVAE | enc dim 128 / dec dim 2048, latent 64, 16 kHz → 48 kHz | ~0.18 B |

| File | Size | Notes |
|---|---|---|
| `model.safetensors` | **4367.91 MB** | BF16 |
| `audiovae.pth` | **359.49 MB** | PyTorch pickle |
| `tokenizer.json` | 3.51 MB | |
| `config.json` | ~4 KB | |

`patch_size 4`, `feat_dim 64`; one AR step ≈ one latent patch ≈ 4 × 1024
samples @ 16 kHz input rate, decoded to 48 kHz output.

### What the port does *not* have to fight

Verified absent from the crate — no work needed:

- **No threads.** No `std::thread`, no `rayon`, no `spawn`, no
  `available_parallelism`, no `Mutex`/`RwLock`/channels anywhere in `src/`.
- **No async runtime.** No `tokio`, no `async`, no `block_on` in crate code.
- **No `unsafe`** outside two spots in `weights.rs` (mmap + the
  `SafetensorsFile` self-reference), both native-only diagnostics.
- **No build scripts** of our own; no C/C++ in the dependency tree once
  `tokenizers/onig` is swapped out (see B4).
- **Byte-oriented entry points already exist** for audio
  (`load_audio_bytes`, `load_audio_bytes_as`, `encode_wav`,
  `write_wav_to`) and for prompts (`PromptAudio::Encoded`,
  `PromptAudio::Pcm`). Voice cloning from a browser `File` or a microphone
  capture needs no new decode path.
- **`burn`'s wgpu backend targets WebGPU by design.** `cubecl-wgpu`
  enables `fragile-send-sync-non-atomic-wasm` unconditionally, has
  `target_family = "wasm"` paths in `runtime.rs`, `device.rs`,
  `graphics.rs`, `compute/{stream,poll}.rs`, and `GraphicsApi::backend()`
  returns `wgpu::Backend::BrowserWebGpu` on wasm. `burn-wgpu` documents
  WebGPU support and advises disabling `fusion` on wasm.

---

## 2. Browser blockers

Ordered by how much they shape the design.

### B1 — Synchronous GPU readback panics on wasm *(critical)*

`cubecl_common::future::block_on` on `target_family = "wasm"` routes to
`reader::read_sync`, which is `embassy_futures::poll_once` plus
`.expect(...)`. It therefore only succeeds **if the future is already
resolved**. Burn's `Tensor::into_data()` goes through the same
`try_read_sync`. A WebGPU readback needs `mapAsync`, whose callback is
delivered on the event loop, so `poll_once` returns `Pending` and the
process panics:

> Failed to read tensor data synchronously. This can happen on platforms
> that don't support blocking futures like WASM.

Blocking the thread does not help, in a worker either: the browser drives
buffer mapping from the event loop, so blocking deadlocks instead.

The hot path has exactly **two** readbacks, both small and both
unavoidable:

| Site | Frequency | Payload |
|---|---|---|
| `model.rs:326` — stop-head `argmax(1).into_data()` inside `dit_step` | **once per AR step** | `B` × i64 (1 element at `B=1`) |
| `wrapper.rs:1107` — `wav.into_data()` in `decode_latent_to_samples` | once per utterance (or per stream chunk) | full waveform |

Plus `model.rs:442` `sync_barrier`, which only runs under
`VOXCPM_PROFILE` and is a no-op in the browser (see B6).

Burn offers `into_data_async()` / `to_data_async()` /
`into_scalar_async()` (`burn-tensor-0.20.1/src/tensor/api/base.rs:1851`),
returning `Result<TensorData, ExecutionError>`. So the fix is mechanical
but it makes the AR loop `async`, which propagates up through
`inference` → `generate` to the `wasm_bindgen` boundary.

**Decision:** add `async` twins rather than converting the existing
functions. The native path keeps its exact current code (zero regression
risk for `cpu` / `cpu-blas` / `wgpu` / `wgpu-fast` / `vulkan`), and the
browser gets `dit_step_async`, `inference_async`, `generate_async`,
`decode_latent_to_samples_async`. The duplicated AR loop is ~70 lines;
unifying it behind a hand-rolled native `block_on` was considered and
rejected, because on native the readback waker is driven by cubecl's own
poll task and a naive executor risks a deadlock in a path that currently
works.

### B2 — 4.37 GB of weights vs. a 4 GB address space *(critical)*

`wasm32-unknown-unknown` has 32-bit pointers: linear memory caps at 4 GB,
and a single `Vec` must fit inside it alongside everything else. The
upstream checkpoint is 4367.91 MB of BF16. So:

- Fetching `model.safetensors` into one `Vec<u8>` **cannot work**. Not
  "slow" — it cannot be allocated.
- The current loader is worse than one copy. `weights.rs` reads the whole
  file, then `materialize_weight_norm` builds a `HashMap` of *converted*
  tensors, then `safetensors::serialize` copies that into a second full
  buffer, then burn-store uploads from it. At F32 that is
  4.4 GB (raw) + 8.7 GB (converted) + 8.7 GB (re-serialized) in flight.

The task brief allows whole-file loading "for the first working version".
At this model size that concession does not apply, so the streaming
loader is in scope for v1 rather than deferred.

**Decision:** stream per tensor group.

```
GET model.safetensors  Range: bytes=0-7          → header length (u64 LE)
GET model.safetensors  Range: bytes=8-(8+N-1)    → JSON header
   → { name: { dtype, shape, data_offsets:[s,e] } }   offsets are
     relative to the end of the header (8 + N)
group tensors by module                        (grouping rule below)
for each group, under a byte budget:
   Range-fetch exactly that group's byte span
   convert BF16 → backend float dtype
   materialize weight_norm, fuse q/k/v and gate/up
   serialize a small in-memory safetensors buffer
   burn-store apply with allow_partial(true)   → uploads to GPU
   drop everything
```

Peak WASM heap becomes a function of the largest group, not of the model.
`SafetensorsStore::from_bytes` + `allow_partial(true)` already exists in
`weights.rs`, so repeated partial application needs no change to
burn-store; it just needs the file-reading half factored out.

**Grouping rule.** Three transforms in `weights.rs` are *intra-group* and
constrain how tensors may be split:

- `fuse_qkv` needs `q_proj` + `k_proj` + `v_proj` together,
- `fuse_gate_up` needs `gate_proj` + `up_proj` together,
- `materialize_weight_norm` needs `weight_g` + `weight_v` together.

Grouping by "the checkpoint key minus its last two path segments"
satisfies all three: `…self_attn.{q,k,v}_proj.weight` → `…self_attn`;
`…mlp.{gate,up}_proj.weight` → `…mlp`; `X.weight_{g,v}` → both land on
the same parent. Leaf params like `…input_layernorm.weight` form
harmless singleton groups.

**Largest single group** is the token embedding:
73448 × 2048 × 2 B = **301 MB** raw, which at F32 becomes 602 MB
converted + 602 MB re-serialized. Bounded, but the worst spike in the
load. Noted as the first target for the single-tensor direct-write fast
path if testing shows it hurts.

### B3 — Precision determines whether this runs at all

`weights.rs` materializes weights in the backend's native float type
because burn-store does not auto-cast. So GPU weight residency is:

| `B::FloatElem` | Weights on GPU | Viable where |
|---|---|---|
| `f32` | **8.74 GB** | ≥ 12 GB VRAM only |
| `f16` | **4.37 GB** | ≥ 8 GB VRAM |
| INT8 (future) | ~2.2 GB | — |
| INT4 (future) | ~1.2 GB | — |

The brief says "start with F32 if necessary, correctness first". F32 is
retained as the first target **only because the development box is a
24 GB RTX 4090**, where it fits and exactly matches the known-good native
`Wgpu<f32, i32>` configuration — so a wrong output can be attributed to
the port rather than to a precision change. F16 is the real shipping
target and is wired as a build-time switch from the start.

F16 feasibility is already confirmed in the dependency tree:
`burn-cubecl`'s `element.rs` implements `FloatElement` and
`MatmulElement` for `half::f16`, and `cubecl-wgpu`'s
`backend/wgsl.rs::register_types` registers `FloatKind::F16` for all
scalar usage whenever the adapter reports `wgpu::Features::SHADER_F16`.

### B4 — `tokenizers` pulls in C

`tokenizers = { default-features = false, features = ["onig"] }` links
oniguruma, a C library, which does not build for
`wasm32-unknown-unknown`. The crate ships `unstable_wasm`
(`fancy-regex` + `getrandom/js`) as the pure-Rust substitute.

Feature unification means this cannot be toggled by *our* feature flag —
it has to be a per-target dependency table, so that the wasm build and
the native build request different `tokenizers` features.

### B5 — `Instant::now()` panics on wasm

`std::time::Instant::now()` is compiled but panics
("time not implemented on this platform"). Live call sites:
`weights.rs` ×4 (load timing) and `model.rs:455/462/487` (the
`VOXCPM_PROFILE` block). `performance.now()` is the browser equivalent,
reachable from both `Window` and `WorkerGlobalScope`.

### B6 — `std::env::var` is inert

`model.rs:435` (`VOXCPM_PROFILE`) and `locdit/unified_cfm.rs:48`
(`VOXCPM_Z_ZERO`, which zeroes the diffusion noise — genuinely useful for
determinism when diffing browser output against native). Both compile and
silently return `Err` in the browser, so the knobs vanish. They need a
browser-settable backing store.

### B7 — `getrandom` has no backend on `wasm32-unknown-unknown`

The sampler calls `Tensor::random(..., Distribution::Normal)` in
`locdit/unified_cfm.rs:51`, `audiovae/layers.rs:177` and
`locenc.rs:21`. Both `getrandom` 0.2.17 and 0.3.4 are in the lockfile.
0.3 refuses to pick a backend for this target unless **both** the
`wasm_js` Cargo feature *and* `--cfg getrandom_backend="wasm_js"` are
set; 0.2 needs its `js` feature. Without this the build fails with
getrandom's "unsupported target" error.

### B8 — `memmap2`

Used in `weights.rs` (`load_single`, `SafetensorsFile`). It *compiles*
for wasm — 0.9 ships `src/stub.rs` for non-unix/non-windows — but every
call fails at runtime. Also unavoidable transitively: `burn-store`'s
`std` feature hard-enables `dep:memmap2`. So nothing to remove, just
code paths to `cfg` out so the browser can't reach them.

### B9 — Filesystem loading

`VoxCPM::from_local` (`wrapper.rs:317`) does
`std::fs::read_to_string(config.json)`, `TextTokenizer::from_local`
(`tokenizer.rs:26`) does `Tokenizer::from_file`, and
`weights::load_pretrained` does `Path::exists` + `File::open`. All
path-based, all unreachable in the browser.

`audio.rs` needs the same treatment for `write_wav`, `load_audio`,
`load_audio_as`, and `PromptAudio::File` becomes unserviceable (its
`Encoded` / `Pcm` siblings cover the browser cases).

### B10 — `audiovae.pth` is a Python pickle, read by path

Upstream ships the AudioVAE only as `.pth`.
`weights.rs::load_single_pth` uses `burn_store::pytorch::PytorchReader::new(path)`
— path-based, and a pickle interpreter is a lot of wasm code for no
reason. At 359 MB the AudioVAE is small enough to load whole, so the
browser path wants `audiovae.safetensors`, produced by a one-time
offline conversion. That also lets the browser loader use a single
code path (safetensors) for both files.

### B11 — WebGPU device creation is async

`cubecl::wgpu::init_setup` explicitly panics on wasm
("Creating a wgpu setup synchronously is unsupported on wasm. Use
init_async instead"). The browser must call
`init_setup_async::<graphics::WebGpu>(&device, options)` and `await` it
before any tensor exists.

Good news on limits: `cubecl-wgpu`'s `backend/wgsl.rs::request_device`
requests `required_limits: adapter.limits()` and
`required_features: adapter.features() - MAPPABLE_PRIMARY_BUFFERS`,
i.e. the maximum the adapter will give. That is what makes the 301 MB
embedding buffer legal — WebGPU's *default* `maxBufferSize` (256 MB) and
`maxStorageBufferBindingSize` (128 MB) would both reject it. It also
means `SHADER_F16` is enabled automatically when present.

### B12 — `fusion` on wasm

`burn-wgpu`'s own docs: "You can disable the `fusion` feature flag to
remove that functionality, which might be necessary on `wasm` for now."
The existing `wgpu-fast` feature enables `burn/fusion` + `burn/autotune`;
autotune additionally wants to time kernels, which is awkward in a
browser. The browser feature must therefore derive from plain `wgpu`, not
from `wgpu-fast`.

### B13 — Vulkan-only machinery must stay out of the wasm graph

`Cargo.toml` carries `[patch.crates-io]` entries for vendored
`burn-cubecl` and `cubecl-spirv` (bf16 fixes for the radv SPIR-V path).
`cubecl-spirv` is an optional dependency gated behind `cubecl-wgpu`'s
`spirv` feature, so as long as the browser feature does not enable
`burn/vulkan` it never enters the graph. The vendored `burn-cubecl` does
(it is a plain `[patch]`), and its change — accumulating conv sums in f32
instead of the element type — is desirable for f16 too.

### B14 — Crate type and wasm glue

The crate has no `[lib]` section, so it builds as `rlib` only;
`wasm-bindgen` needs `cdylib`. `wasm-bindgen` / `wasm-bindgen-futures` /
`js-sys` / `web-sys` / `console_error_panic_hook` must be wasm-only
dependencies so native builds and `cargo publish` consumers never see
them.

### B15 — Error surfacing

`#![recursion_limit = "256"]` is already set and is needed for the
wgpu/naga generic chain. Separately, a Rust `panic!` in wasm surfaces to
JS as a bare `RuntimeError: unreachable` unless a panic hook is
installed — the brief explicitly calls this out, so
`console_error_panic_hook` plus a `log` → `console` bridge is part of the
deliverable, and every `wasm_bindgen` entry point must return
`Result<_, JsValue>` carrying the real `crate::Error` text.

---

## 3. Proposed modifications

### Build / packaging

- `[lib] crate-type = ["cdylib", "rlib"]`.
- New isolated feature `webgpu = ["burn/wgpu"]` — deliberately *not*
  derived from `wgpu-fast` (B12) and not enabling `burn/vulkan` (B13).
  `cpu`, `cpu-blas`, `wgpu`, `wgpu-fast`, `vulkan` untouched.
- Optional `webgpu-f16` to flip `B::FloatElem` from `f32` to `half::f16`
  (B3).
- Per-target dependency tables (B4, B8, B14): native gets
  `tokenizers/onig` + `memmap2`; wasm gets `tokenizers/unstable_wasm` +
  the wasm-bindgen stack.
- `.cargo/config.toml` adding `--cfg getrandom_backend="wasm_js"` for
  `wasm32-unknown-unknown` (B7).

### New `src/compat.rs`

`now_ms()` / `Stopwatch` over `Instant` natively and `performance.now()`
in the browser (B5); `env_flag()` over `std::env` natively and a
browser-settable registry in the browser (B6).

### New `src/browser/` (wasm only)

- `fetch.rs` — a `ByteSource` abstraction with `len()` + `read_at(offset, len)`,
  implemented by an HTTP-Range source and an in-memory source. This is the
  brief's `ModelSource` idea, shaped around *ranges* rather than
  whole-file `load(name)` because of B2.
- `loader.rs` — safetensors header parse, the grouping rule from B2,
  group-at-a-time streaming apply, progress events.
- `api.rs` — `wasm_bindgen` surface: panic hook + logger init, WebGPU
  self-test, model load with progress callback, `generate`.

### `src/weights.rs`

Factor the shared tail (`repack_and_apply`) into a public
`apply_tensor_group(model, label, tensors, namespace)` taking
`Vec<RawTensor>` instead of a `&Path`, so the browser loader reuses the
weight_norm / fusion / dtype pipeline verbatim. Add
`empty_apply_result` / `merge_apply_result` / `finalize_apply_result` for
accumulating across groups, and `backend_float_dtype::<B>()` so the UI
can report real precision. `cfg` out the path-based readers, the pickle
reader and `SafetensorsFile` (B8, B9, B10).

### `src/model.rs`, `src/voxcpm2/wrapper.rs`

Async twins for the two readback sites (B1). `VoxCPM::from_parts` so the
browser can build the model first and stream weights into it.
`report_apply_result` extracted so both loaders log identically.

### `src/audio.rs`, `src/tokenizer.rs`

`TextTokenizer::from_bytes` (B9). Path-based audio functions gated, with
browser stubs that return a `crate::Error::Unsupported` explaining the
`Encoded` / `Pcm` alternative rather than failing opaquely (B15).

### `web/`

`index.html` + `main.js` + `style.css`, no framework. WebGPU init, model
load with progress, generate, WebAudio playback, and the diagnostics
panel the brief asks for (adapter info, `shader-f16`, `maxBufferSize`,
`maxStorageBufferBindingSize`, workgroup limits).

### Tooling

`examples/convert_audiovae.rs` transcodes `audiovae.pth` →
`audiovae.safetensors` (B10), reusing the crate's own pickle reader so no
Python/torch install is needed. `scripts/serve.py` is a static file
server that supports HTTP Range (B2), and `scripts/build-web.sh` drives
the wasm build. `examples/stream_load.rs` runs the browser's loader and
inference paths natively, over a file instead of HTTP, so they can be
debugged without a browser.

---

## 4. Files likely requiring changes

| File | Change | Blockers |
|---|---|---|
| `Cargo.toml` | `[lib] cdylib`, `webgpu` feature, per-target deps | B3 B4 B8 B12 B13 B14 |
| `.cargo/config.toml` | *new* — getrandom cfg | B7 |
| `src/lib.rs` | register `browser`, `compat` | — |
| `src/compat.rs` | *new* — time + env shims | B5 B6 |
| `src/browser/mod.rs` | *new* — init, logging, debug flags | B6 B15 |
| `src/browser/fetch.rs` | *new* — `ByteSource`, HTTP Range | B2 |
| `src/browser/loader.rs` | *new* — streaming safetensors loader | B2 B3 B9 B10 |
| `src/browser/api.rs` | *new* — `wasm_bindgen` surface | B1 B11 B15 |
| `src/weights.rs` | public `apply_tensor_group`, gate fs/mmap/pickle | B2 B8 B9 B10 |
| `src/voxcpm2/model.rs` | `*_async` twins, compat shims | B1 B5 B6 |
| `src/voxcpm2/wrapper.rs` | `*_async` twins, `from_parts`, gate `from_local` | B1 B9 |
| `src/tokenizer.rs` | `from_bytes`, gate `from_local` | B4 B9 |
| `src/audio.rs` | gate path fns + browser stubs | B9 B15 |
| `src/locdit/unified_cfm.rs` | `env_flag` | B6 |
| `web/*` | *new* — UI | — |
| `scripts/*` | *new* — Range-capable server, web build, PCM compare | B2 |
| `examples/convert_audiovae.rs` | *new* — pth → safetensors | B10 |
| `examples/stream_load.rs` | *new* — native driver for the browser paths | B1 B2 |

Untouched: `src/config.rs`, `src/error.rs`, `src/fsq.rs`,
`src/locenc.rs`, `src/minicpm4/*`, `src/audiovae/*` (except via
`compat`), `src/locdit/local_dit.rs`, and every `examples/*.rs`.

---

## 4b. Blockers found while implementing

The survey above is what reading the code predicted. These four only
appeared once the port actually ran, and three of them are hard failures
that no amount of reading would have surfaced.

### B16 — `usize` is 32-bit and the checkpoint is not *(critical)*

`model.safetensors` is **4,580,080,592 bytes**. `usize::MAX` on `wasm32`
is 4,294,967,295. Any offset past the 4 GB mark — which is most of the
file — is unrepresentable.

Two consequences:

1. **Header parsing.** `safetensors::tensor::TensorInfo` types
   `data_offsets` as `(usize, usize)`, so it cannot describe this file on
   a 32-bit target. `src/stream.rs` parses the header into its own
   `TensorEntry` with `u64` offsets instead. (`SafeTensors::read_metadata`
   is unusable regardless — it asserts the buffer holds the whole file.)
2. **Tensor size validation.** `safetensors` **0.7** — the version
   `burn-store` depends on, distinct from the 0.4 this crate uses
   directly — validates a tensor by computing
   `n_elements × dtype.bitsize()` in `usize`. In *bits*. That caps a
   single tensor at 512 MiB on `wasm32`, and VoxCPM2's token embedding is
   `73448 × 2048` = 602 MiB at F32. The browser load died with:

   ```
   safetensors store: SafeTensors error: overflow computing buffer size
   from shape and/or element type
   ```

   Fixed by not going through safetensors at all on the apply side — see
   B17.

### B17 — the safetensors round-trip was both wasteful and 32-bit-limited

The original loader converted each group into a tensor map, re-serialized
it with `safetensors::serialize`, and handed the bytes to
`SafetensorsStore::from_bytes` for burn-store to parse back. That is a
full extra copy of every tensor, and it is what ran into B16's 512 MiB
ceiling.

`burn_store::ModuleSnapshot::apply` takes `Vec<TensorSnapshot>` directly,
and `TensorSnapshot::from_data` builds one from a `TensorData`. Switching
to it removes the copy *and* the size limit. The transposition behaviour
is preserved: `Applier` substitutes the **module's** container stack
before invoking `PyTorchToBurnAdapter`, so snapshots need no container
information of their own.

This also simplified `src/stream.rs` — an oversized-tensor "direct write"
fast path existed purely to dodge the re-serialize copy, and became
dead weight once the copy was gone.

### B18 — `cubek-reduce` requests illegal workgroups on WebGPU *(critical)*

Every RMSNorm and every `argmax` in the model goes through a reduce.
`cubek-reduce` 0.1.1 builds its workgroup by hand:

```rust
let cube_dim = CubeDim::new_2d(plane_size, plane_count);
```

skipping the `max_units_per_cube` clamp that cubecl's own `CubeDim::new`
applies — whose comment is, pointedly, *"Make sure it respects the max
units per cube (especially on wasm)"*.

WebGPU exposes no subgroup size, so `cubecl-wgpu` substitutes
`plane_size_max = 128`; with `plane_count = 4` that asks for a
512-invocation workgroup. WebGPU's **baseline**
`maxComputeInvocationsPerWorkgroup` is 256, so adapters at the baseline
reject it:

```
The total number of workgroup invocations (512) exceeds the maximum allowed (256).
[Invalid ComputePipeline "reduce_kernel"] is invalid due to a previous error.
```

Discrete GPUs report 1024 and never see it, which is exactly why it is
easy to ship broken. Fixed by vendoring `cubek-reduce` with the clamp —
same mechanism the repo already uses for its bf16 fixes. See
`patches/cubek-reduce-workgroup-clamp.patch`.

### B19 — `wgpu` cannot see the adapter's identity on WebGPU

`Adapter::get_info()` returns `name: ""`, `vendor: 0`,
`device_type: Other` on the WebGPU backend — the spec does not hand
adapter identity to the graphics API the way a native driver does. So a
Rust-side check for "is this SwiftShader?" can never fire, and the UI
would show a blank adapter name.

The information *is* available to JavaScript as `adapter.info`. The page
passes it in via `set_adapter_hint()` before `init_webgpu()`, which is
also what makes the refuse-to-run-on-CPU guard possible at all.

### B20 — `init_webgpu` must be idempotent

`ComputeClient::init` panics with *"Can't create a new client on an
already registered server"* on a second call for the same device. Both
`webgpu_self_test` and `load_model` need the device, and the user may
retry after a refusal, so setup is cached while the guards are
re-evaluated on every call.

---

## 4c. Verification

What was actually run, and on what.

### In Chrome (headless Chrome 154, WASM + WebGPU)

| milestone | result |
|---|---|
| 1 — compiles for `wasm32-unknown-unknown` | yes, and all five native features still build |
| 2 — WebGPU compute self-test | **PASS** — 64×64 matmul (sum 8192 = 2·64²) and a reduce (mean 2.5) through `burn`/cubecl |
| 3 — browser model loading | **PASS** — 4.37 GB streamed by HTTP Range |
| 4 — instantiate VoxCPM2 | **PASS** — `applied=632, missing=0, unused=0, errors=0`, in 35.7 s |
| 5 — speech generation | **PASS** — PCM out, finite, correct length |
| 6 — browser UI | built; drives all of the above |
| 7 — Japanese | the test texts are Japanese throughout |

### The decisive check: browser vs. native, numerically

With `VOXCPM_Z_ZERO` set on both sides (the diffusion sampler's Gaussian
noise zeroed, so the pipeline is deterministic), the same text at the
same `max_len`:

```
A native  (Vulkan, RTX 4090)  samples=23040 peak=0.01270 rms=0.001536
B browser (WebGPU, Chrome)    samples=23040 peak=0.01269 rms=0.001536

correlation : 1.000000
max |A-B|   : 0.000005   (0.04% of peak)
rms(A-B)    : 0.000001   (0.05% of rms)
```

Identical to f32 rounding. Tokenizer, Range-streamed weight loading,
every kernel in the LM / LocEnc / LocDiT / AudioVAE, the async AR loop and
the PCM output all agree with the native implementation.

### Audio quality, measured natively

`VOXCPM_Z_ZERO` degrades the output, so quality was judged with the
sampler intact, on the *same async and streaming code paths the browser
uses* (`examples/stream_load.rs --async` / `--stream`):

| run | duration | voiced | median F0 | syllables/s |
|---|---|---|---|---|
| `--async` | 6.08 s | 57.0 % | 186.0 Hz | 2.8 |
| `--stream` | 7.52 s | 69.0 % | 166.7 Hz | 3.9 |
| `--async` (other text) | 5.28 s | 49.6 % | 119.4 Hz | 3.4 |
| `VOXCPM_Z_ZERO` reference | 29.92 s | 24.1 % | 275.9 Hz | 17.0 |

Autocorrelation pitch in the human range with a plausible syllable rate
on every normal run; the `Z_ZERO` row is the control, and the detector
correctly flags it as degenerate. No NaN or Inf in any run, no clipping.

### What could NOT be verified here, and why

**The hardware GPU path in a browser.** This machine's X session belongs
to `gdm` and the user is not in the `render`/`video` groups, so
`/dev/dri/renderD128` (`root:render`, mode 660, ACL granting only `gdm`)
is unreachable from Chrome's GPU process. Chrome therefore falls back to
SwiftShader, its CPU rasterizer. Native Vulkan is unaffected because the
NVIDIA driver goes through `/dev/nvidia*`, which is world-accessible —
which is why every native measurement here is on the real 4090 while the
browser ones are not.

All browser numbers above are from SwiftShader. Correctness is fully
exercised by it — it is a conforming WebGPU implementation, and the
numeric match above was produced on it. **Performance is not**: the
browser run took 1087 s for 0.48 s of audio (RTF ~2264), which says
nothing about a real GPU. The equivalent native run is RTF ~13–20 at F32
without fusion or autotune.

To get hardware WebGPU on a machine like this:

```sh
sudo usermod -aG render,video "$USER"   # then log out and back in
```

and confirm `chrome://gpu` reports hardware-accelerated WebGPU. The demo
refuses to run on a software adapter by default precisely so this
misconfiguration is visible rather than silent.

---

## 5. Known limitations to carry forward

- **Whole-file loading is not an option** at 4.37 GB (B2); the streaming
  loader is required for v1, not an optimization.
- **F32 needs ≥ 12 GB VRAM** (B3). The first working version targets it
  only to match native bit-for-bit on a 24 GB dev GPU; F16 is what makes
  this reach normal hardware.
- **`audiovae.pth` must be converted offline** to `.safetensors` (B10).
- **`fancy-regex` replaces `onig`** in the browser tokenizer (B4).
  Equivalent for this BPE configuration, but it is a different regex
  engine, so tokenization is worth diffing against native.
- **No `fusion` / `autotune`** in the browser (B12), so steady-state RTF
  will trail a native `wgpu-fast` build.
- **Streaming (Milestone 9) is only partly free.** `GenerateStream`
  re-decodes the whole accumulated latent per chunk and relies on AudioVAE
  causality for seamless output — `O(N²)` decode work across an utterance.
  Fine for correctness, but real streaming wants the stateful decoder.
