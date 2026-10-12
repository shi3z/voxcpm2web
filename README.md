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

## Try it in a browser

**https://shi3z.github.io/voxcpm2web/**

Open it, press **Load model**, press **Generate**. Nothing to install,
nothing to supply. It needs **an 8 GB GPU** and the patience for a
2.45 GiB download.

Where each piece comes from:

| | |
| --- | --- |
| page + wasm | GitHub Pages, over HTTPS (WebGPU needs a secure context) |
| `config.json`, `tokenizer.json`, `model.safetensors` | streamed from [`shi3z/VoxCPM2-q8`](https://huggingface.co/shi3z/VoxCPM2-q8) by HTTP Range |
| the AudioVAE | already inside that `model.safetensors` |

The default source is a **pre-quantized INT8 repack** of
[`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2), published so
the demo fits ordinary hardware. It exists because of two measurements:

- quantizing the full-precision checkpoint *in the browser* ends at the
  same 2.9 GB of resident weights, but it has to download 4.37 GB to get
  there, and peaks at **5.74 GiB** of GPU memory on the way;
- loading the full-precision checkpoint unquantized peaks at
  **9.65 GiB**, which is why this used to want a 12 GB card.

The pre-quantized file peaks at **5.22 GiB**, generation included, and
rolls the AudioVAE in — which also disposes of the one file that had no
public host. Upstream publishes the AudioVAE only as `audiovae.pth`, a
Python pickle the browser build cannot read, and no `.safetensors`
version of it is hosted anywhere public.

VoxCPM2 is Apache-2.0 and not gated, so republishing a numerically
transformed copy is permitted; the licence travels with it, in the
repository's `LICENSE` and model card.

The full-precision source is still selectable in the page (*Weights
source*) if you want to compare, and the deploy workflow still converts
and ships `audiovae.safetensors` alongside the page so that path works
too (`MODEL-LICENSE.txt`, `MODEL-NOTICE.txt`).

### Why the checkpoint is not hosted on GitHub

It cannot be. GitHub rejects any single file over **100 MB**, and a Pages
site is capped at **1 GB**; `model.safetensors` is **4.37 GB**. Git LFS
does not help either — Pages does not resolve LFS pointers.

Hugging Face does work, and this was verified rather than assumed:

```
OPTIONS .../model.safetensors   (Origin: https://shi3z.github.io, Request-Headers: range)
  → 200, access-control-allow-headers: range
         access-control-expose-headers: ..., Accept-Ranges, Content-Range

GET     .../model.safetensors   (Range: bytes=0-99)
  → 206, content-range: bytes 0-99/4580080592
         access-control-allow-origin: *
```

`Range` is not a CORS-safelisted header, so it triggers a preflight —
that preflight passing is the whole reason this works at all.

Measured end to end, with the page on one origin and the checkpoint on
Hugging Face, both sources verified this way:

| source | size | load | result |
| --- | --- | --- | --- |
| `shi3z/VoxCPM2-q8` *(default)* | 2.45 GiB | 186 s | `applied=632, missing=0, unused=0, errors=0`, 2.90 GB on the GPU |
| `openbmb/VoxCPM2` | 4.4 GB | 474 s | `applied=632, missing=0, unused=0, errors=0` |

The first row is the **deployed** page at the URL above, driven headless,
pulling from Hugging Face cross-origin — so the preflight, the deployed
`wasm-opt`ed bundle and the manifest detection are all covered by it.

Those times are bandwidth, not compute: the same default source measured
465 s on a slower link an hour earlier. Read them as "a few minutes on a
home connection", not as a benchmark. There is no local cache yet, so a
reload re-downloads.

The AudioVAE *is* small enough for Pages: ~359 MB alongside two 10.8 MB
wasm bundles sits well inside the 1 GB site limit, and it is still
deployed for the optional full-precision source.

The default source no longer touches it, which is what makes the hosted
demo cheap to run: a default visit pulls **~11 MB** from Pages — one
wasm bundle and the page — and everything else from Hugging Face.
Against Pages' ~100 GB/month soft bandwidth cap that is thousands of
loads rather than the ~250 a month the 359 MB AudioVAE allowed.

### Feeding your own weights instead

The page checks what is actually reachable rather than guessing. If the
origin serving it has a checkpoint at `/models` — which
`scripts/serve.py --model DIR` mounts — it uses that and skips the
download entirely. Useful when you are iterating locally or over a
tailnet.

To point the *hosted* page at weights on your own machine, that server
has to send CORS headers:

```bash
python3 scripts/serve.py --model /path/to/VoxCPM2 --tailscale \
    --cors https://YOU.github.io
```

`--cors` is opt-in and takes an origin; bare `--cors` allows any, which
lets any site you visit read files from it while it runs. Note this is
for your own convenience, not for the demo — a public demo cannot depend
on a machine only you can reach.

---

## Status

| | |
| --- | --- |
| Compiles for `wasm32-unknown-unknown` | yes, with all native backends still building |
| WebGPU compute in-browser | verified — 64×64 matmul and a reduce through Burn/cubecl |
| Checkpoint loading in-browser | verified — both sources streamed by HTTP Range, `applied=632, missing=0, errors=0`: 2.45 GiB pre-quantized in 9.1 s, 4.37 GB full precision in 35.7 s (local server) |
| The deployed demo end to end | verified — self-test and a full load from Hugging Face, on the published bundle |
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
- **A GPU with enough VRAM.** Peak GPU memory, measured with
  `nvidia-smi` across a whole run — load *and* generation, not just
  resident weights:

  | weights source | peak GPU | needs |
  | --- | --- | --- |
  | **pre-quantized Q8** *(demo default)* | **5.22 GiB** | **≥ 8 GB VRAM** |
  | full precision + Q8 on load | 5.74 GiB | ≥ 8 GB VRAM |
  | full precision, unquantized | 9.65 GiB | ≥ 12 GB VRAM |

  The two Q8 rows end at the same ~2.9 GB of resident weights; they
  differ in what has to pass through the GPU to get there, and in
  download size (2.45 GiB against 4.37 GB).

  5.22 GiB leaves ~2.7 GB of an 8 GB card free, which is headroom but
  not a lot of it: a desktop compositor and the browser's own GPU
  process can account for 1–2 GB before the model starts. These numbers
  were measured on an otherwise-idle card, so an 8 GB laptop running a
  full desktop is the case most likely to be tight.

- **~2.5 GB of transfer** per page load for the default source (there is
  no local cache yet), or ~4.8 GB for the full-precision one.
- A Rust toolchain with the `wasm32-unknown-unknown` target, to build.

The demo **refuses to run on a software adapter** by default. Chrome
silently hands out SwiftShader when it cannot reach the GPU, and for a
2.3 B-parameter model that means hours per utterance — so it reports the
misconfiguration instead of looking broken. There is an explicit override
for correctness testing.

---

## Quantization

`Linear` weights are quantized to **8 bits by default**, which is what
makes this runnable on ordinary hardware. It is a runtime choice in the
page (*Weight quantization*), not a build flag.

| | GPU weights | notes |
| --- | --- | --- |
| **Q8** *(default)* | **2.97 GB** | measured; ~2× faster than F32 |
| none (F32) | 9.09 GB | exact; needs a 12 GB card |

Measured on an RTX 4090, same text, `--async`:

```
Q8    2974 MB resident (2139M params quantized + 245M float)   RTF 10.7
F32   9095 MB                                                  RTF 13–20
```

Output stays speech. Against the F32 baseline on identical text, both
pass the pitch and syllable-rate checks in `scripts/`:

| | duration | voiced | median F0 | syllables/s |
| --- | --- | --- | --- | --- |
| F32 | 6.08 s | 57.0 % | 186 Hz | 2.8 |
| Q8 | 6.08 s | 82.8 % | 222 Hz | 5.8 |

Those are different trajectories, not a degradation metric — an
autoregressive sampler diverges from any perturbation. **Nobody has
listened to the Q8 output and compared it by ear**, which is the
judgement that actually matters; the numbers only establish that it is
still speech with healthy amplitude and no NaN.

### How it works, and why during the load

Nothing in the model changes. A burn `Tensor` can hold a quantized
primitive, `matmul` dispatches to `q_matmul`, and that dequantizes inside
the kernel instead of materializing a float copy. So quantization is just
a transform over parameters.

It runs **as each batch of weights lands**, not once at the end. A single
pass afterwards would need the whole model in floating point first,
peaking at 9.1 GB — exactly what quantization is for avoiding.

Left in floating point, deliberately:

- **Embeddings** (73448 × 2048 = 301 MB). `q_gather` is
  `unimplemented!()` on this backend, so a quantized embedding panics at
  inference. This is the floor the approach cannot go below.
- **Convolutions** (the AudioVAE) — no quantized convolution kernel.
- **Norm and scale parameters** — 1-D and negligible.
- **Anything whose last dimension is not a multiple of `32/bits`.**
  `QuantStore::PackedU32` packs that many values per `u32` along the
  innermost axis and panics otherwise. VoxCPM2 has such a layer: the stop
  head projects to 2 classes.

The self-test probes a Q8 matmul, so a device that cannot do it says so
in seconds rather than after a 4.4 GB download. Verified in-browser on
SwiftShader, which reports only 32 KB of workgroup storage — the
tightest limit available here.

### Q4 does not work yet

Q4 would put the checkpoint at ~1.45 GB. The quantization itself
succeeds; inference does not:

```
cubek-matmul auto(): SimpleCyclicCmma -> Unavailable (no cooperative matrix on WebGPU)
                   -> falls back to SimpleUnit
                   -> needs 98304 bytes of shared memory, limit is 49152
                   -> unwraps the error instead of trying a smaller tile
```

The tile selection does not consult `max_shared_memory_size`; it
validates against it and fails. Fixing it means patching
`cubek-matmul`'s strategy selection. `QuantPreset::Q4` exists and is
refused with that explanation rather than silently falling back.

### Pre-quantized checkpoints

Quantizing in the browser cuts VRAM and time but not the download — all
4.37 GB still has to arrive first. Quantizing *offline* cuts both:

```bash
cargo run --release --example quantize_checkpoint \
    --no-default-features --features wgpu -- \
    /path/to/VoxCPM2 /path/to/VoxCPM2-q8 q8
```

Measured, loading the result instead of the upstream checkpoint:

| | upstream | pre-quantized Q8 |
| --- | --- | --- |
| download | 4.27 GiB + 359 MB AudioVAE | **2.45 GiB, AudioVAE included** |
| load time (native) | 30–35 s | **7.4 s** |
| load time (browser) | 35.7 s | **9.1 s** |
| GPU weights | 9.09 GB | 2.90 GB |
| peak GPU, generation included | 9.65 GiB | **5.22 GiB** |
| peak WASM heap | 2.0 GB | 1.4–2.0 GB (see below) |

`applied=632, missing=0, unused=0, errors=0` either way, and the output
passes the same speech checks.

The WASM heap row deserves the hedge. `WebAssembly.Memory` only ever
grows, so what the page reports is a high-water mark including allocator
fragmentation — and that mark is not deterministic. Three loads of this
same pre-quantized checkpoint reserved 1.4 GB, 1.4 GB and **2.0 GB**,
the 2.0 GB being the deployed bundle pulling from Hugging Face. So
pre-quantizing does not reliably reduce the WASM high-water mark; it
reduces the download, the load time and the GPU footprint. The heap
number that matters is that it stays well under the 4 GB address-space
limit, which it does in every run.

The format is plain safetensors, so the Range-streaming loader works
unchanged. Quantized weights are stored as flat `U8` blobs holding
exactly what `q_into_data` produced — packed values with the scales
appended — and a manifest in `__metadata__` records each one's logical
shape and the scheme. `q_from_data` consumes that layout directly, so
loading never materializes a float copy. Everything left in floating
point goes out at 16 bits.

Two consequences worth knowing:

- The file holds **final module paths**, exported from a loaded model, so
  it is already remapped, fused and transposed. It must be applied
  **without** `PyTorchToBurnAdapter`, or every Linear weight gets
  transposed twice. The loader detects this from the manifest.
- The AudioVAE is baked in, so there is no separate `audiovae.safetensors`
  to fetch — which also removes the one file that had no public host.

### Where the pre-quantized checkpoint is hosted

Not on GitHub: at 2.45 GiB it exceeds both the 100 MB per-file push limit
and the 1 GB cap on a Pages site. It lives on Hugging Face instead, at
[**`shi3z/VoxCPM2-q8`**](https://huggingface.co/shi3z/VoxCPM2-q8), and is
the page's default *Weights source*.

That host has to do two things, both verified rather than assumed:

```
OPTIONS .../model.safetensors   (Origin: https://shi3z.github.io, Request-Headers: range)
  → 200, access-control-allow-origin: https://shi3z.github.io
         access-control-allow-headers: range
         access-control-expose-headers: ..., Accept-Ranges, Content-Range

GET     .../model.safetensors   (Range: bytes=0-99)
  → 206, content-range: bytes 0-99/2629105078
         access-control-allow-origin: *
```

A local directory still works and still wins when present: point the
page's *Base URL* at a directory containing a pre-quantized
`model.safetensors` and the loader detects it from the manifest.
`scripts/serve.py --model DIR --tailscale` is enough for your own
machines.

## Switching between F32 and F16

Precision is a **build-time** switch: `B::FloatElem` is a type parameter,
so the two are separate wasm modules.

```bash
scripts/build-web.sh          # F32 -> web/pkg
scripts/build-web.sh --f16    # F16 -> web/pkg-f16
scripts/build-web.sh --both   # both, and the page chooses
```

With `--both` deployed, the page's **Weight precision** selector offers:

| | |
| --- | --- |
| Auto *(default)* | **F32** — F16 is not selected automatically while its reduce bug stands |
| F32 | 8.74 GB of VRAM. Verified to match a native run exactly |
| F16 | 4.37 GB of VRAM. **Reductions are currently wrong** — see below. Refused outright if `shader-f16` is missing |

Changing it reloads the page with `?precision=…`, because the GPU device
is already registered with cubecl by then and swapping modules in place
is not safe.

> ### F16 is currently broken
>
> On the first run on real hardware WebGPU with `shader-f16`, the F16
> build got **matmul right and reductions wrong**:
>
> ```
> self-test FAILED: reduce test FAILED: mean 1 != 2.5
> ```
>
> `mean([1,2,3,4])` returned `1`. Reductions back every RMSNorm and the
> stop head, so this build does not produce quiet or missing audio — it
> produces *wrong* audio. Auto therefore selects **F32**, and F16 is
> opt-in with a warning until this is fixed.
>
> The self-test now runs 17 reduce probes (sum/mean/max/min, a length
> sweep over `ones`, and argmax) and reports every value, so one run
> localizes the fault. If you can run it on an F16-capable GPU, that
> output plus the Diagnostics panel is exactly what is needed.

F16 halves both the VRAM and the peak WASM heap. It was originally the
plan for reaching an 8 GB card; INT8 quantization got there first and
more cheaply (5.22 GiB peak, measured), so F16 is no longer load-bearing
for that — but halving the WASM heap is still worth having, and it would
compose with Q8 rather than compete with it. It builds, and the bundle
loads and runs matmuls, but its reductions are wrong (above).
Before that was known, it could not be verified here at all, for a
separate reason worth recording:

```
$ cargo run --release --example gpu_features --no-default-features --features wgpu
adapter      : NVIDIA GeForce RTX 4090 (Vulkan)
                        adapter  device
  SHADER_F16             no       no
```

`wgpu` 26's Vulkan backend does not advertise `SHADER_F16` on this
driver, so the native WGSL path cannot run f16 at all — it fails in naga
validation with *"Using `f16` values requires the
`naga::valid::Capabilities::FLOAT16` flag"*. And the test machine's
browser only ever got a SwiftShader adapter, which has no `shader-f16`
either. So neither route to verification was available here.

Note that the native failure does **not** imply the browser build is
broken: native `wgpu` validates WGSL through naga, whereas in a browser
the WGSL is compiled by the browser itself (Dawn, in Chrome), and
`wgpu`'s wasm backend hands it straight over. That turned out to be the
right call — the browser build *does* load and compute matmuls on an
f16-capable GPU. Its reduce kernels are what is wrong.

`examples/gpu_features.rs` prints the above for your own hardware, which
is quicker than waiting out a 4.4 GB load to find out.

The thing to watch if you do try it: the AudioVAE's transposed
convolutions sum ~32k products per output. That is the reduction that
collapsed in BF16 and motivated the vendored `burn-cubecl` patch. The
patch accumulates in F32 regardless of element type, so it should cover
F16 too — but should is not the same as does.

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

# 4. build the browser bundle(s)
scripts/build-web.sh --both     # F32 and F16; the page picks

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

A page served this way needs no configuring: it probes
`/models/config.json` on its own origin and, finding a checkpoint there,
uses it instead of downloading anything.

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
  natively, which is far easier to debug than a tab. `--async`, `--stream`
  and `--f16` select the browser's code paths.
- `examples/gpu_features.rs` — prints what your GPU backend actually
  supports (`SHADER_F16`, buffer limits, workgroup limits), which answers
  "will `--f16` work here?" in seconds instead of after a 4.4 GB load.
- `scripts/check-wasm.mjs` — instantiates a built bundle outside the
  browser and performs the startup table growth that the generated glue
  does. `build-web.sh` runs it on every bundle and refuses to finish if it
  fails. It exists because a bad `wasm-opt` once shipped a module that
  validated cleanly and then died on every page load; see the comment at
  the top of the file.

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
