# voxcpm2web

**VoxCPM2 speech synthesis running entirely in a web browser — WebAssembly + WebGPU, with no inference server.**

Open a page, wait for the model to load, type text, hear speech. Every
neural-network kernel runs on the local GPU through WebGPU. There is no
Python backend, no inference API, no WebSocket, and no CPU fallback.

```
text ──► WASM (Rust / Burn) ──► WebGPU compute ──► Vec<f32> PCM
                                                        │
                                       Float32Array ────┴──► AudioBuffer ──► speakers
```

A small static HTTP server is involved, but only to hand over HTML,
JavaScript, the `.wasm` bundle and the model weights.

This is a port of [**voxcpm-rs**](https://github.com/mii-nipah/voxcpm-rs)
— pure-Rust [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) inference on
the [Burn](https://burn.dev) framework — to `wasm32-unknown-unknown` plus
WebGPU. The model code is reused as-is; the work is in everything around
it. Upstream's own documentation — the native backends, the Rust API tour,
the architecture walkthrough — is preserved at
[`docs/voxcpm-rs.md`](docs/voxcpm-rs.md).

---

## Status

| | |
| --- | --- |
| Compiles for `wasm32-unknown-unknown` | yes, with all native backends still building |
| WebGPU compute in-browser | verified — 64×64 matmul and a reduce through Burn/cubecl |
| Checkpoint loading in-browser | verified — 4.37 GB streamed by HTTP Range, `applied=632, missing=0, errors=0`, 35.7 s |
| Speech generation in-browser | verified — finite PCM at the correct length |
| Agreement with native | **correlation 1.000000**, max difference 0.04 % of peak |
| Browser UI | works: self-test, load, generate, play, save, voice cloning, streaming |
| Japanese | the test texts are Japanese throughout |

### The decisive check

With the diffusion sampler made deterministic on both sides
(`VOXCPM_Z_ZERO`, which zeroes its Gaussian noise), the same text at the
same length:

```
A  native  (Vulkan, RTX 4090)   samples=23040  peak=0.01270  rms=0.001536
B  browser (WebGPU, Chrome)     samples=23040  peak=0.01269  rms=0.001536

correlation : 1.000000
max |A-B|   : 0.000005   (0.04 % of peak)
rms(A-B)    : 0.000001   (0.05 % of rms)
```

Identical to f32 rounding. The tokenizer, the Range-streamed weight
loading, every kernel in the LM / LocEnc / LocDiT / AudioVAE, the
autoregressive loop and the PCM output all agree with the native
implementation.

### Honest caveat about performance

The browser runs above were executed on Chrome's **SwiftShader** (CPU)
adapter, because the development machine's browser could not reach its
GPU — its X session belonged to `gdm` and the user was not in the
`render`/`video` groups, so `/dev/dri/renderD128` was unreachable from
Chrome's GPU process. Native Vulkan was unaffected (the NVIDIA driver
goes through `/dev/nvidia*`).

That fully exercises **correctness** — SwiftShader is a conforming WebGPU
implementation, and the numeric match above was produced on it. It says
**nothing about speed**: that run took 1087 s for 0.48 s of audio. Every
speed figure quoted in this README is from a native run on the same code
paths. Browser performance on real GPU hardware is not yet measured.

---

## Requirements

- **Chrome / Chromium 113+** (Edge works too). Safari is untested.
- **A GPU with enough VRAM for the weights:**

  | build | weights on GPU | needs |
  | --- | --- | --- |
  | F32 (default) | 8.74 GB | ≥ 12 GB VRAM |
  | F16 (`--f16`) | 4.37 GB | ≥ 8 GB VRAM, adapter reports `shader-f16` |

- **~4.8 GB of disk** for the checkpoint, and the patience to transfer it
  to the browser once per page load (there is no local cache yet).
- A Rust toolchain with the `wasm32-unknown-unknown` target, to build.

The demo **refuses to run on a software adapter** by default. Chrome
silently hands out SwiftShader when it cannot reach the GPU, and for a
2.3 B-parameter model that means hours per utterance — so it reports the
misconfiguration instead of looking broken. There is an explicit override
for correctness testing.

---

## Quick start

```bash
# 1. toolchain
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version \
  "$(awk '/^name = "wasm-bindgen"$/{f=1;next} f&&/^version/{gsub(/[":]/,"");print $3;exit}' Cargo.lock)"

# 2. get the checkpoint
huggingface-cli download openbmb/VoxCPM2 --local-dir ./VoxCPM2

# 3. one-time: transcode the AudioVAE
#    Upstream ships it only as a PyTorch pickle, which the browser build
#    cannot read (it would need `zip` -> `zstd-sys`, i.e. C). This reuses
#    the crate's own pickle reader, so no Python or torch is required.
cargo run --release --example convert_audiovae \
    --no-default-features --features cpu -- ./VoxCPM2

# 4. build the browser bundle
scripts/build-web.sh            # F32  (≥12 GB VRAM)
scripts/build-web.sh --f16      # F16  (≥8 GB VRAM)

# 5. serve it
python3 scripts/serve.py --model ./VoxCPM2
# open http://localhost:8080
```

Then in the page: **Self-test WebGPU** → **Load model** → **Generate**.

### The server must support HTTP Range

`model.safetensors` is 4.37 GB and `wasm32` has a 4 GB address space, so
the checkpoint is never materialized in memory. Only its header is parsed
up front; tensors are then fetched, converted and uploaded to the GPU one
module at a time.

`scripts/serve.py` honours `Range`. `python3 -m http.server` does **not**,
and the loader reports that explicitly rather than trying to allocate
4.4 GB.

---

## Opening it from another machine

**WebGPU requires a secure context.** `http://localhost` qualifies by
special case; `http://192.168.1.5:8080` from another machine does not, so
`navigator.gpu` is simply `undefined` there. Binding the server to
`0.0.0.0` gets you a page that loads and then reports no WebGPU — no
server setting can change that. It has to be HTTPS.

On a [Tailscale](https://tailscale.com) tailnet that is one flag:

```bash
python3 scripts/serve.py --model ./VoxCPM2 --tailscale
#   open this on any tailnet machine:  https://<machine>.<tailnet>.ts.net:8080/
```

That asks `tailscale cert` for a real Let's Encrypt certificate for the
node's MagicDNS name, serves HTTPS with it, and binds **only** that
node's Tailscale v4 and v6 addresses — so the checkpoint is not also
exposed on your LAN or any public interface. No sudo, no browser flags,
no certificate warnings. It needs HTTPS enabled for the tailnet
([docs](https://tailscale.com/kb/1153/enabling-https)).

Other situations:

| | |
| --- | --- |
| Not on a tailnet | `--tls-cert` / `--tls-key` with any certificate the client trusts (e.g. [`mkcert`](https://github.com/FiloSottile/mkcert)) |
| Just want it local | the default — `http://localhost:8080` is already a secure context |
| Cannot do HTTPS at all | on the *client*, launch Chrome with `--unsafely-treat-insecure-origin-as-secure=http://HOST:8080` |

The page diagnoses this itself: if it loads without a secure context it
says so, and how to fix it, rather than blaming your GPU.

---

## What the model is

| component | shape | params |
| --- | --- | --- |
| Base LM (MiniCPM4) | hidden 2048, FFN 6144, 28 layers, 16 heads / 2 KV heads, vocab 73448 | ~1.47 B |
| Residual LM | same block, 8 layers, no RoPE | ~0.38 B |
| LocEnc | hidden 1024, FFN 4096, 12 layers | ~0.25 B |
| LocDiT (diffusion) | hidden 1024, FFN 4096, 12 layers | ~0.25 B |
| AudioVAE | 16 kHz latent in, 48 kHz audio out | ~0.18 B |

**2.29 B parameters total.** `model.safetensors` is 4,580,080,592 bytes of
BF16; `audiovae.pth` is another 359 MB.

Generation is autoregressive over ~80 ms latent patches, each produced by
a 10-step Euler flow-matching solve through the DiT, then decoded to
audio by the causal AudioVAE.

---

## How the browser port works

Three things in the original code cannot exist in a browser tab, and each
drove a design decision.

### 1. The checkpoint is bigger than the address space

`wasm32` has 32-bit pointers: 4 GB of linear memory, total. The
checkpoint is 4.37 GB. The original loader reads the whole file, converts
every tensor into a second full-size map, and re-serializes that into a
third — roughly 22 GB at F32. That is not slow in a browser; it is
unallocatable.

So weights stream in, group by group:

```
read 0..8      → header length (u64 LE)
read 8..8+N    → JSON header { name: {dtype, shape, data_offsets} }
                 ├─ group tensors by module
                 └─ batch groups under a byte budget (64 MiB of source bytes)
for each batch:
    Range-fetch its byte runs
    convert dtype, materialize weight_norm, fuse q/k/v and gate/up
    apply to the module  → uploads to the GPU
    drop everything
```

Peak heap then tracks the batch budget rather than the model: **1285 MB**
measured natively with a tracking allocator, **2.0 GB** as reported by
`WebAssembly.Memory` in Chrome (higher because WASM memory only grows and
is never returned). Generation itself peaks at 36 MB — weights stay
resident on the GPU and are never re-uploaded.

Grouping is not arbitrary. Three transforms need sibling tensors present
together — `q/k/v` fusion, `gate/up` fusion, and `weight_norm`
materialization — so a group is never split.

Two 32-bit landmines showed up here. Offsets past the 4 GB mark do not
fit in `usize`, so the safetensors header is parsed into a local type
with `u64` offsets. And `safetensors` 0.7 validates a tensor by computing
`n_elements × dtype.bitsize()` in `usize` — *in bits* — which caps a
single tensor at 512 MiB on `wasm32`. The F32 token embedding is 602 MiB.
Fixed by handing Burn's store `TensorSnapshot`s directly instead of
round-tripping through serialized safetensors, which also removed a full
copy of every tensor.

### 2. Nothing can block

WebGPU readback is `mapAsync`, delivered on the event loop. cubecl's
`block_on` on wasm is `poll_once` + `expect`, so it panics on anything
that does not resolve immediately — and blocking the thread deadlocks
instead, in a worker too.

The hot path has exactly two readbacks: the stop-head argmax (once per
autoregressive step, one bool per batch element) and the final waveform.
Both now have `async` twins — `dit_step_async`, `inference_async`,
`generate_async`, `next_chunk_async` — while the original blocking
functions are left byte-for-byte alone, so the native backends keep their
proven code path.

### 3. No filesystem, no `Instant`, no env vars

Checkpoint bytes arrive over `fetch` behind a `ByteSource` trait, which
also has a native file implementation — so the whole loader can be driven
and debugged outside a browser. `std::time::Instant::now()` *panics* on
`wasm32`, so profiling routes through `performance.now()`.
`std::env::var` is inert, so the debug switches get a browser-settable
registry.

### An upstream bug worth knowing about

`cubek-reduce` builds its workgroup shape by hand and skips the
`max_units_per_cube` clamp that cubecl's own `CubeDim::new` applies —
whose comment is, pointedly, *"especially on wasm"*. Since WebGPU exposes
no subgroup size, cubecl substitutes `plane_size_max = 128`, and the
routines then ask for a **512-invocation workgroup**. WebGPU's *baseline*
`maxComputeInvocationsPerWorkgroup` is **256**, so adapters at the
baseline reject the pipeline:

```
The total number of workgroup invocations (512) exceeds the maximum allowed (256).
[Invalid ComputePipeline "reduce_kernel"] is invalid due to a previous error.
```

Every RMSNorm and every argmax in the model goes through a reduce, so the
model cannot run at all on such a device. Discrete GPUs report 1024 and
never see it — which is exactly why it is easy to ship broken. A vendored
one-line clamp fixes it.

---

## Features

- **Zero-shot synthesis** from text.
- **Voice cloning** from a WAV file or a microphone recording. Audio never
  leaves the tab; it is decoded with `AudioContext.decodeAudioData`,
  downmixed in JS, and resampled in Rust.
- **Streaming playback** — each chunk is played as it is produced, not
  sliced off a finished waveform. Chunk boundaries are seamless because
  the AudioVAE decoder is causal. The UI reports time-to-first-audio and
  counts buffer underruns.
- **Diagnostics panel** — user agent, origin, `isSecureContext`, adapter
  identity, `shader-f16`, `maxBufferSize`,
  `maxStorageBufferBindingSize`, workgroup limits, and an explicit check
  that the 602 MB embedding will fit.
- **Real error messages.** A Rust `panic!` in wasm surfaces to JS as a
  bare `RuntimeError: unreachable` unless a hook is installed, so one is,
  and every entry point returns the actual error text.

### Measured performance (native, F32, RTX 4090)

Native runs on the same async / streaming code paths the browser uses:

| | |
| --- | --- |
| Checkpoint load (streamed) | 33–46 s |
| RTF, non-streaming | ~13–20 |
| Streaming: time to first audio | 14.5 s |
| Streaming: chunk latency | 10 s → 18 s over a 7.5 s utterance |

RTF below 1 would be faster than realtime; this is not there yet. The
build has no kernel fusion and no autotune (Burn advises disabling fusion
on wasm), runs F32, and is unquantized. Streaming chunk latency grows
because every chunk re-decodes the whole accumulated latent — `O(N²)`
across an utterance.

Output was checked objectively rather than by ear: autocorrelation pitch
lands at 119–186 Hz with 50–69 % voiced frames and a plausible syllable
rate on every normal run, with no NaN, Inf or clipping.

---

## Not yet done

- **Local checkpoint caching.** A page reload re-downloads the model,
  modulo the HTTP cache. Cache Storage or OPFS keyed on a manifest hash
  would fix it; the loader already goes through `ByteSource`, so a
  cache-backed source drops in beside the HTTP one.
- **Quantization.** F16 is wired as a build flag but unmeasured. INT8
  (~2.2 GB) and INT4 (~1.2 GB) weight-only are designed but not
  implemented — the rule that matters is dequantizing *inside* the matmul
  kernel, never into a full F16 copy first.
- **Linear-time streaming decode**, by porting the reference stateful
  streaming VAE decoder.
- **Browser performance on real GPU hardware** — see the caveat above.

---

## Documentation

| | |
| --- | --- |
| [`docs/voxcpm-rs.md`](docs/voxcpm-rs.md) | The full library documentation: native backends (`cpu`, `cpu-blas`, `wgpu`, `wgpu-fast`, `vulkan`), the Rust API, voice cloning, batching, architecture. This is upstream's README, with a browser section added. |
| [`docs/browser-port-analysis.md`](docs/browser-port-analysis.md) | The port survey: architecture, every blocker found and what was done about each, and what was and was not verified. |
| [`docs/webgpu-memory.md`](docs/webgpu-memory.md) | Measured memory, per component, and the F16 / INT8 / INT4 plan. |
| [`patches/README.md`](patches/README.md) | The vendored upstream fixes and why each is needed. |

Worth knowing about the layout:

- `src/stream.rs` — the streaming checkpoint loader. Platform-independent,
  behind a `ByteSource` trait, so it runs over HTTP in a browser and over a
  file natively.
- `src/browser/` — the only wasm-only code: `fetch.rs` (HTTP Range),
  `api.rs` (the `wasm_bindgen` surface).
- `src/compat.rs` — the time and env-var shims.
- `web/` — the UI. No framework; `index.html` + `main.js` + `style.css`.
- `scripts/` — `build-web.sh`, `serve.py` (Range + HTTPS), `compare_pcm.py`.
- `examples/stream_load.rs` — runs the browser's loader and inference paths
  natively, which is far easier to debug than a tab.

The crate is still named `voxcpm-rs` and its `Cargo.toml` metadata still
points at upstream, deliberately: this is that library plus a browser
target, not a rename.

## Credits

- [**voxcpm-rs**](https://github.com/mii-nipah/voxcpm-rs) by
  [nipah~✰!](https://github.com/mii-nipah) — the pure-Rust VoxCPM2
  inference implementation this is a port of. All of the model code is
  theirs.
- [**VoxCPM2**](https://huggingface.co/openbmb/VoxCPM2) by
  [OpenBMB](https://github.com/OpenBMB) — the model.
- [**Burn**](https://burn.dev) and [**CubeCL**](https://github.com/tracel-ai/cubecl)
  by [Tracel AI](https://github.com/tracel-ai) — the framework and the
  GPU compute layer that makes one WGSL codebase run on Vulkan, Metal and
  WebGPU alike.

## License

Apache-2.0, inherited from `voxcpm-rs`. The VoxCPM2 model weights carry
their own license — see the
[model card](https://huggingface.co/openbmb/VoxCPM2).
