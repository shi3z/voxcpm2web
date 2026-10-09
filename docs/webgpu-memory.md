# VoxCPM2 in the browser: memory

**Milestone 10 deliverable, measurement half.** Numbers below are measured,
not estimated, unless marked otherwise. Model: `openbmb/VoxCPM2`
(`model.safetensors` 4367.91 MB BF16 + `audiovae.pth` 359.49 MB).

Two budgets matter and they are completely separate:

| budget | hard limit | what lives there |
|---|---|---|
| **WASM linear memory** | 4 GB (32-bit pointers) | the checkpoint bytes in flight during load, activations' host copies, PCM |
| **GPU memory** | the adapter's | every weight, the KV cache, activations |

The first is what makes the naive loader impossible; the second is what
makes F32 impractical.

---

## 1. Weights on the GPU

2.29 B parameters, so weight residency is just `params × bytes_per_param`:

| `B::FloatElem` | weights | reachable hardware |
|---|---|---|
| F32 (`--features webgpu`) | **8.74 GB** | ≥ 12 GB VRAM |
| F16 (`--features webgpu-f16`) | **4.37 GB** | ≥ 8 GB VRAM |
| INT8 weight-only (not implemented) | ~2.2 GB | ≥ 4 GB VRAM |
| INT4 weight-only (not implemented) | ~1.2 GB | ≥ 3 GB VRAM |

Per component, from the upstream `config.json`:

| component | shape | params | F32 | F16 |
|---|---|---|---|---|
| Base LM (MiniCPM4) | `hidden 2048`, `ffn 6144`, 28 layers, 16 heads / 2 KV heads | ~1.47 B | 5.88 GB | 2.94 GB |
| Residual LM | same block, 8 layers, no RoPE | ~0.38 B | 1.51 GB | 0.76 GB |
| LocEnc | `hidden 1024`, `ffn 4096`, 12 layers | ~0.25 B | 1.01 GB | 0.50 GB |
| LocDiT | `hidden 1024`, `ffn 4096`, 12 layers | ~0.25 B | 1.01 GB | 0.50 GB |
| AudioVAE | enc 128 / dec 2048, latent 64 | ~0.18 B | 0.72 GB | 0.36 GB |

The single largest tensor is the token embedding, `73448 × 2048`:
**602 MB at F32, 301 MB at F16**. It sets a floor on per-buffer limits —
see §4.

### KV cache

`StaticKvCache` is allocated for `prefill + max_len` positions up front.
Per position, per model: `2 (K+V) × num_kv_heads × kv_channels × layers`.

* Base LM: `2 × 2 × 128 × 28` = 14336 elements/position
* Residual LM: `2 × 2 × 128 × 8` = 4096 elements/position

At the default `max_len = 2000` plus a ~100-token prefill (2100 positions):

| | F32 | F16 |
|---|---|---|
| Base LM KV | 120 MB | 60 MB |
| Residual LM KV | 34 MB | 17 MB |
| **total** | **154 MB** | **77 MB** |

Small next to the weights, and linear in `max_len` — the UI defaults
`max_len` to 600 (≈ 48 s of audio) rather than 2000, which cuts this to
~46 MB at F32.

### Activations

Batch 1, one position per AR step, so activations are tiny: the widest
intermediate is the gated MLP's `2 × 6144` plus the DiT's `4096`, i.e.
tens of KB per layer. The AudioVAE decode is the exception — it expands
the latent to 48 kHz audio, so its transient is proportional to utterance
length (~0.2 MB/s of audio at F32). Not a factor at these scales.

### Measured total

Nothing in WebGPU reports process-wide GPU usage, so the honest statement
is: weights dominate, and the table in §1 is the number to plan around.
`maxBufferSize` / `maxStorageBufferBindingSize` *are* reported and are
surfaced in the demo's Diagnostics panel.

---

## 2. WASM linear memory during load

This is the budget the port was actually designed around.

### What the default loader would need

`weights::load_pretrained` reads the whole file, converts every tensor
into a second full-size map, and (before this port) re-serialized that
into a third full-size safetensors buffer:

```
4.37 GB  raw BF16 file
8.74 GB  converted to F32
8.74 GB  re-serialized
-------
~21.9 GB peak
```

Against a 4 GB address space that is not slow, it is unallocatable.

### What the streaming loader needs

`src/stream.rs` parses only the header, groups tensors by module, and
applies one batch at a time under a byte budget
(`DEFAULT_BATCH_BUDGET = 64 MiB` of *source* bytes). Measured with a
tracking global allocator (`examples/stream_load.rs`, which reports true
live/peak heap rather than RSS):

| phase | peak heap | live after |
|---|---|---|
| full model load, F32 | **1285 MB** | 21 MB |
| generation | **36 MB** | 36 MB |

Measured in Chrome via `WebAssembly.Memory.buffer.byteLength` for the same
F32 load: **2.0 GB**. Higher than the native 1285 MB because WASM memory
only ever grows — it is the high-water mark including allocator
fragmentation, and it is never returned. It is the number that has to fit
under 4 GB, and it does, with F16 roughly halving it.

Native RSS for the same run reads ~10 GB, but that is misleading: it
counts the GPU driver's host-side mappings of the 8.7 GB of device
memory, which in a browser never touch the WASM heap. Hence the tracking
allocator.

### Why the peak is 1285 MB and not 64 MiB

A group is never split, because three transforms in `weights.rs` need
sibling tensors present together (`fuse_qkv`, `fuse_gate_up`,
`materialize_weight_norm`). The token embedding is therefore applied in
one piece:

```
287 MiB  raw BF16 (freed per tensor as it converts)
574 MiB  converted F32
574 MiB  burn/cubecl staging for the GPU upload
```

Everything else is bounded by the batch budget. Lowering
`batch_budget_bytes` (third argument to `load_model`) does not reduce the
peak below the embedding's cost; only F16 or a quantized embedding does.

---

## 3. What this port already does about it

Changes made for memory, with their effect:

1. **Stream per group instead of whole-file.** ~21.9 GB → ~1.3 GB peak.
   This is what makes the load possible at all.
2. **Single-pass dtype conversion.** `convert_float_slice` writes
   BF16 → target directly into the destination buffer, instead of
   `decode_f32` + `encode_float` with a `Vec<f32>` in between. Saves a
   full extra copy of each tensor.
3. **Apply `TensorSnapshot`s directly instead of re-serializing
   safetensors.** Removes one full copy of every converted group *and*
   fixes a hard 32-bit bug: `safetensors` 0.7 validates a tensor by
   computing `n_elements × dtype.bitsize()` in `usize`, which on `wasm32`
   caps any single tensor at **512 MiB**. The F32 embedding is 602 MiB, so
   the browser load failed with "overflow computing buffer size from shape
   and/or element type" until this went in.
4. **Weights stay resident.** `load_model` runs once; `generate` reuses
   the GPU buffers. Generation's own heap peak is 36 MB.

---

## 4. Buffer limits

WebGPU's *default* limits are smaller than this model's largest tensor:

| limit | WebGPU default | needed (F32) | needed (F16) |
|---|---|---|---|
| `maxBufferSize` | 256 MB | 602 MB | 301 MB |
| `maxStorageBufferBindingSize` | 128 MB | 602 MB | 301 MB |

Nothing extra had to be done about this: `cubecl-wgpu`'s
`backend/wgsl.rs::request_device` asks for
`required_limits: adapter.limits()`, i.e. the adapter's maximum rather
than the defaults. Chrome on a discrete GPU typically reports well above
1 GB for both. The demo's Diagnostics panel prints both, plus an explicit
"fits 602 MB embedding (F32)" check, because a device that reports less
will fail mid-load and the reason would otherwise be obscure.

---

## 5. Next steps, in order

Ordered by payoff per unit of risk. Not yet implemented.

**A. F16 end-to-end** — `--features webgpu-f16`. Halves both budgets
(8.74 → 4.37 GB GPU, ~2.0 → ~1.1 GB WASM peak) and is the difference
between "needs a 12 GB card" and "runs on an 8 GB laptop GPU". The
plumbing is in place: `burn-cubecl` implements `FloatElement` for
`half::f16`, and `cubecl-wgpu` registers `FloatKind::F16` whenever the
adapter reports `shader-f16`. The bundle builds, and the page will load
it and refuse cleanly if the adapter lacks the feature.

It is **unverified**, and not for lack of trying: `wgpu` 26's Vulkan
backend reports `SHADER_F16: no` on an RTX 4090 / driver 590.48.01
(see `examples/gpu_features.rs`), so f16 kernels fail naga validation on
the native WGSL path; and the browser used for testing only ever got a
SwiftShader adapter, which has no `shader-f16`. The native failure says
nothing about the browser — there, WGSL is compiled by Dawn, not naga.

The remaining question is numerical, not mechanical — the AudioVAE's transposed convolutions sum ~32k products per
output, which is exactly the reduction that collapsed in BF16 and
motivated the vendored `burn-cubecl` patch in `patches/`. That patch
accumulates conv sums in F32 regardless of element type, so it should
cover F16 too; it needs verifying rather than assuming.

**B. Per-component precision.** The components do not deserve equal
treatment:

| component | candidate | why |
|---|---|---|
| Base LM / Residual LM | Q8 → Q4 | 81 % of the weights; transformer weights quantize well |
| KV cache | F16 | small; not worth the dequant cost |
| LocEnc | F16 | small, feeds the DiT directly |
| LocDiT | F16 | iterated 10x per patch; error compounds across Euler steps |
| AudioVAE | F16 | long reductions, directly audible |

**C. Weight-only INT8, then INT4.** Weights INT8 with F16 activations and
F32 accumulation gets the model to ~2.2 GB. INT4 (two values per `u8`,
per-group scales, group sizes 32/64/128 worth benchmarking) gets to
~1.2 GB. The rule that matters for memory: **dequantize inside the matmul
kernel**, next to the multiply, never into a full F16 copy of the weight
first — a whole-model dequant would reintroduce the F16 footprint and
defeat the point.

**D. Shrink the embedding specifically.** At 602 MB (F32) it is both the
largest single buffer and the sole reason the load peaks at 1.3 GB rather
than a few hundred MB. It is a pure lookup table, so it is the easiest
thing in the model to quantize aggressively without touching any matmul
kernel.

**E. Cache the checkpoint locally.** Orthogonal to memory, but the
dominant cost of a page reload today is re-downloading 4.4 GB. Cache
Storage or OPFS keyed on a manifest hash would remove it. The loader
already goes through [`ByteSource`][crate], so a cache-backed
implementation slots in beside `HttpRangeSource` without touching the
streaming logic.

**F. Linear-time streaming decode.** Not a weight-memory item, but it is
the biggest remaining inefficiency: `GenerateStream` re-decodes the whole
accumulated latent for every chunk, which is `O(N^2)` decode work across
an utterance and shows up directly as chunk latency growing from ~10 s to
~18 s over a 7.5 s utterance (measured, F32, RTX 4090). Porting Python's
stateful `StreamingVAEDecoder` would make it linear and would also cap
the decode transient, which currently grows with utterance length.

[crate]: ../src/stream.rs
