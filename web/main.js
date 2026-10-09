// VoxCPM2 WebGPU demo — UI glue.
//
// All inference happens in WASM on the GPU. JavaScript's jobs are: check
// that WebGPU exists, drive the Rust entry points, and play the PCM that
// comes back through WebAudio. No WAV encoding on the hot path — Rust
// hands back a Float32Array and it goes straight into an AudioBuffer.

// The wasm bundle is loaded dynamically, because weight precision is a
// build-time switch: `B::FloatElem` is a type parameter, so F16 and F32
// are two separate modules (`scripts/build-web.sh --both`). Picking one
// here is what lets the page offer the choice without a rebuild.
let vox = null;

const $ = (id) => document.getElementById(id);

const els = {
  unsupported: $('unsupported'),
  unsupportedWhy: $('unsupported-why'),
  selftest: $('btn-selftest'),
  load: $('btn-load'),
  generate: $('btn-generate'),
  play: $('btn-play'),
  stop: $('btn-stop'),
  save: $('btn-save'),
  record: $('btn-record'),
  clearRef: $('btn-clear-ref'),
  refFile: $('ref-file'),
  refStatus: $('ref-status'),
  modelUrl: $('model-url'),
  precision: $('precision'),
  sourcePreset: $('source-preset'),
  vaeUrl: $('vae-url'),
  vaeFile: $('vae-file'),
  clearVae: $('btn-clear-vae'),
  vaeStatus: $('vae-status'),
  text: $('text'),
  cfg: $('cfg'),
  timesteps: $('timesteps'),
  maxlen: $('maxlen'),
  zzero: $('zzero'),
  stream: $('stream'),
  chunkPatches: $('chunk-patches'),
  chunkField: $('chunk-field'),
  allowSoftware: $('allow-software'),
  selftestOut: $('selftest-out'),
  loadProgress: $('load-progress'),
  barFill: $('bar-fill'),
  progressStage: $('progress-stage'),
  progressPct: $('progress-pct'),
  progressDetail: $('progress-detail'),
  status: $('status'),
  diag: $('diag').querySelector('tbody'),
};

const metric = (id, value) => { $(id).textContent = value; };

let wasm = null;        // the wasm module exports (for memory introspection)
let session = null;     // VoxCpmSession handle
let pcm = null;         // Float32Array of the last generation
let sampleRate = 48000;
let audioCtx = null;
let playing = null;     // AudioBufferSourceNode (whole-utterance playback)
let streamNodes = [];   // scheduled chunk sources (streaming playback)
let nextChunkTime = 0;  // AudioContext time the next chunk should start at
let underruns = 0;      // chunks that arrived after their slot had passed
let reference = null;   // { samples: Float32Array, sampleRate: number }
let recorder = null;
let vaeBytes = null;    // AudioVAE supplied from a local file, if any

// Where the weights come from. The checkpoint is 4.37 GB, which no
// GitHub Pages site can host (100 MB per file, 1 GB per site), but
// Hugging Face serves Range requests with CORS — so the page and the
// weights can live in different places.
const SOURCE_PRESETS = {
  local: {
    base: '/models',
    vae: '/models/audiovae.safetensors',
  },
  hf: {
    base: 'https://huggingface.co/openbmb/VoxCPM2/resolve/main',
    // Deliberately blank: there is no public safetensors AudioVAE to
    // point at, so the user supplies one.
    vae: '',
  },
};

// ---------------------------------------------------------------------------
// Status log
// ---------------------------------------------------------------------------

const lines = [];
function say(msg, kind) {
  const stamp = new Date().toLocaleTimeString();
  lines.push(`${stamp}  ${msg}`);
  if (lines.length > 200) lines.shift();
  els.status.textContent = lines.join('\n');
  els.status.scrollTop = els.status.scrollHeight;
  els.status.className = 'out' + (kind ? ' ' + kind : '');
  if (kind === 'error') console.error(msg); else console.log(msg);
}

function fail(prefix, e) {
  // Rust entry points reject with a plain string carrying the real
  // `crate::Error` text; a genuine JS exception has .message.
  const text = (e && e.message) ? e.message : String(e);
  say(`${prefix}: ${text}`, 'error');
}

const fmtBytes = (n) => {
  if (!Number.isFinite(n)) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
};

function heapBytes() {
  // `memory` is exported by wasm-bindgen's generated glue.
  return wasm && wasm.memory ? wasm.memory.buffer.byteLength : NaN;
}

function refreshHeap() {
  metric('m-heap', fmtBytes(heapBytes()));
}

// ---------------------------------------------------------------------------
// Diagnostics panel
// ---------------------------------------------------------------------------

function diagRow(label, value, good) {
  const tr = document.createElement('tr');
  const th = document.createElement('th');
  th.textContent = label;
  const td = document.createElement('td');
  td.textContent = value;
  if (good === true) td.className = 'yes';
  if (good === false) td.className = 'no';
  tr.append(th, td);
  els.diag.append(tr);
}

// Names that mean "this is a CPU rasterizer pretending to be a GPU".
// WebGPU exposes adapter identity only here, in JS: wgpu's own
// `Adapter::get_info()` returns all-empty values on the WebGPU backend, so
// Rust cannot tell SwiftShader from a 4090 without being told.
const SOFTWARE_NAMES = /swiftshader|llvmpipe|softpipe|lavapipe|software|basic render/i;

function isSoftwareAdapter(info) {
  if (!info) return false;
  const bits = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean).join(' ');
  return SOFTWARE_NAMES.test(bits);
}

async function fillDiagnostics() {
  els.diag.textContent = '';
  diagRow('navigator.userAgent', navigator.userAgent);
  diagRow('navigator.gpu', navigator.gpu ? 'available' : 'MISSING', !!navigator.gpu);
  diagRow('origin', location.origin);
  diagRow('isSecureContext', String(window.isSecureContext), window.isSecureContext);
  diagRow('crossOriginIsolated', String(self.crossOriginIsolated));
  diagRow('hardwareConcurrency', String(navigator.hardwareConcurrency ?? '?'));

  if (!navigator.gpu) return null;
  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    diagRow('requestAdapter', String(e), false);
    return null;
  }
  if (!adapter) {
    diagRow('requestAdapter', 'returned null', false);
    return null;
  }

  // `requestAdapterInfo` was folded into `adapter.info`; support both.
  let info = adapter.info;
  if (!info && adapter.requestAdapterInfo) {
    try { info = await adapter.requestAdapterInfo(); } catch { /* optional */ }
  }
  const software = isSoftwareAdapter(info);
  if (info) {
    const bits = [info.vendor, info.architecture, info.device, info.description]
      .filter(Boolean).join(' / ');
    diagRow('adapter info', bits || '(not exposed)');
  }
  diagRow('hardware accelerated', software ? 'NO — software rasterizer' : 'yes', !software);
  const f16 = adapter.features.has('shader-f16');
  diagRow('shader-f16', f16 ? 'yes' : 'no', f16);
  diagRow('timestamp-query', adapter.features.has('timestamp-query') ? 'yes' : 'no');

  const l = adapter.limits;
  diagRow('maxBufferSize', `${l.maxBufferSize} (${fmtBytes(l.maxBufferSize)})`);
  diagRow('maxStorageBufferBindingSize',
    `${l.maxStorageBufferBindingSize} (${fmtBytes(l.maxStorageBufferBindingSize)})`);
  diagRow('maxComputeWorkgroupStorageSize', String(l.maxComputeWorkgroupStorageSize));
  diagRow('maxComputeInvocationsPerWorkgroup', String(l.maxComputeInvocationsPerWorkgroup));
  diagRow('maxComputeWorkgroupsPerDimension', String(l.maxComputeWorkgroupsPerDimension));

  // The token embedding is 73448 x 2048; at F32 that is one 602 MB buffer.
  // WebGPU's *defaults* (256 MB / 128 MB) would reject it, so this is worth
  // surfacing explicitly — cubecl requests the adapter maximum, which is
  // what makes it legal.
  const embedF32 = 73448 * 2048 * 4;
  const ok = l.maxStorageBufferBindingSize >= embedF32;
  diagRow('fits 602 MB embedding (F32)', ok ? 'yes' : `no — needs ${fmtBytes(embedF32)}`, ok);

  return { info: info || {}, software, shaderF16: f16 };
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

// Load the F16 or F32 bundle. Returns the resolved precision.
//
// `--both` deploys `pkg/` (F32) and `pkg-f16/` (F16); a build of only one
// leaves the other absent, so a missing bundle falls back rather than
// failing.
async function loadWasm(want, shaderF16) {
  const order =
    want === 'f16' ? ['f16']
    : want === 'f32' ? ['f32']
    : shaderF16 ? ['f16', 'f32']
    : ['f32'];

  let lastErr = null;
  for (const p of order) {
    const dir = p === 'f16' ? './pkg-f16' : './pkg';
    try {
      const mod = await import(`${dir}/voxcpm_rs.js`);
      // `init()` resolves to the wasm exports, which is where `memory`
      // lives — that is what the heap readout reports.
      const exports = await mod.default();
      vox = mod;
      return { precision: p, dir, exports };
    } catch (e) {
      lastErr = e;
      say(`${dir} unavailable (${(e && e.message) || e}) — trying the next bundle.`);
    }
  }
  throw lastErr || new Error('no wasm bundle could be loaded');
}

async function boot() {
  const gpu = await fillDiagnostics();

  if (!navigator.gpu) {
    els.unsupported.hidden = false;
    // By far the most common cause when the page is opened from another
    // machine: WebGPU is gated on a secure context, and plain http:// to
    // anything but localhost is not one. No server setting can change
    // that — the page has to be served over HTTPS (or from localhost).
    els.unsupportedWhy.textContent = !window.isSecureContext
      ? `This page was loaded from ${location.origin}, which is not a secure context ` +
        '(window.isSecureContext is false). WebGPU is only exposed to secure ' +
        'contexts, so navigator.gpu is undefined and nothing here can run.\n\n' +
        'Serve the page over HTTPS, or open it on the host itself via ' +
        'http://localhost:8080. On a tailnet:\n\n' +
        '    python3 scripts/serve.py --model /path/to/VoxCPM2 --tailscale\n\n' +
        'which serves HTTPS with a tailscale-issued certificate at ' +
        'https://<machine>.<tailnet>.ts.net:8080/'
      : 'navigator.gpu is undefined, so this browser exposes no WebGPU at all. ' +
        'Use Chrome/Chromium 113+ on a supported GPU. On Linux, Chrome may need ' +
        '--enable-features=Vulkan. The demo will not fall back to CPU inference — ' +
        'that would take hours for this model.';
    els.selftest.disabled = true;
    say(window.isSecureContext
      ? 'WebGPU unavailable — see the message above.'
      : `insecure context (${location.origin}) — WebGPU is unavailable. Serve over HTTPS.`,
      'error');
    return;
  }

  try {
    const want = els.precision.value;
    const shaderF16 = !!(gpu && gpu.shaderF16);
    if (want === 'f16' && !shaderF16) {
      say('F16 was requested but this adapter does not report shader-f16. ' +
          'It will load and then refuse — pick Auto or F32.', 'error');
    }
    const chosen = await loadWasm(want, shaderF16);
    wasm = chosen.exports;
    vox.init('info');
    say(`loaded the ${chosen.precision.toUpperCase()} bundle from ${chosen.dir}.`);
    // Rust cannot see who the adapter is on the WebGPU backend; tell it.
    if (gpu) {
      vox.set_adapter_hint(
        gpu.info.vendor || '', gpu.info.architecture || '',
        gpu.info.device || '', gpu.info.description || '', gpu.software,
      );
      if (gpu.software) {
        els.allowSoftware.closest('.check').style.outline = '2px solid var(--bad)';
        say('this browser is using a SOFTWARE WebGPU adapter, not the GPU. ' +
            'Inference is refused by default — on Linux check that your user is in the ' +
            '`render` and `video` groups and that chrome://gpu shows hardware WebGPU. ' +
            'Tick the checkbox above to run anyway (correctness testing only).', 'error');
      }
    }
    vox.allow_software_adapter(els.allowSoftware.checked);
    say('WASM module loaded. Run the self-test.');
    refreshHeap();
  } catch (e) {
    fail('failed to load the WASM module', e);
    els.selftest.disabled = true;
  }
}

// ---------------------------------------------------------------------------
// Milestone 2: self-test
// ---------------------------------------------------------------------------

function applySourcePreset() {
  const p = SOURCE_PRESETS[els.sourcePreset.value];
  if (!p) return;
  els.modelUrl.value = p.base;
  els.vaeUrl.value = p.vae;
  if (els.sourcePreset.value === 'hf' && !vaeBytes) {
    // Open the section, because this is the one thing the user has to do.
    $('vae-details').open = true;
    say('streaming weights from Hugging Face. The AudioVAE has no public ' +
        'safetensors host — select your converted file below, or give a URL.');
  }
}

els.sourcePreset.addEventListener('change', applySourcePreset);

els.precision.addEventListener('change', () => {
  // The module is already instantiated and the GPU device is registered
  // with cubecl, so swapping bundles in place is not safe. Reload.
  say('precision changed — reloading the page to load the other bundle.');
  const u = new URL(location.href);
  u.searchParams.set('precision', els.precision.value);
  location.assign(u.toString());
});

els.vaeFile.addEventListener('change', async () => {
  const file = els.vaeFile.files?.[0];
  if (!file) return;
  try {
    const buf = await file.arrayBuffer();
    vaeBytes = new Uint8Array(buf);
    els.clearVae.disabled = false;
    els.vaeStatus.textContent =
      `Using ${file.name} (${fmtBytes(vaeBytes.length)}) from this machine. ` +
      'It is read in the tab and never uploaded.';
    say(`AudioVAE loaded from ${file.name} (${fmtBytes(vaeBytes.length)}).`);
  } catch (e) {
    fail('could not read the AudioVAE file', e);
  }
});

els.clearVae.addEventListener('click', () => {
  vaeBytes = null;
  els.vaeFile.value = '';
  els.clearVae.disabled = true;
  els.vaeStatus.textContent = 'Using the URL above.';
  say('AudioVAE file cleared — will fetch from the URL instead.');
});

els.stream.addEventListener('change', () => {
  els.chunkField.hidden = !els.stream.checked;
});

els.allowSoftware.addEventListener('change', () => {
  vox.allow_software_adapter(els.allowSoftware.checked);
  if (els.allowSoftware.checked) {
    say('software WebGPU adapters allowed — inference will run on the CPU. ' +
        'Timings from this run mean nothing.');
  }
});

els.selftest.addEventListener('click', async () => {
  els.selftest.disabled = true;
  els.selftestOut.hidden = false;
  els.selftestOut.className = 'out';
  els.selftestOut.textContent = 'running...';
  say('running WebGPU self-test (matmul + reduce on the real backend)...');
  try {
    const report = await vox.webgpu_self_test();
    els.selftestOut.className = 'out ok';
    els.selftestOut.textContent = report;
    say('self-test PASSED — GPU compute confirmed.');

    // The report's first JSON blob is the adapter report.
    const m = report.match(/\{.*\}/s);
    if (m) {
      const a = JSON.parse(m[0]);
      metric('m-adapter', `${a.name} (${a.backend})${a.software ? ' — SOFTWARE, not a GPU' : ''}`);
      metric('m-precision', `${a.precision}${a.shader_f16 ? ', shader-f16 available' : ''}`);
      if (a.software) {
        say('WARNING: this is a software adapter. Correctness only — no timing here is meaningful.', 'error');
      }
      const bytes = a.precision === 'f16' ? 4367.91 * 1048576 : 8735.82 * 1048576;
      metric('m-gpu', `${fmtBytes(bytes)} of weights at ${a.precision}`);
    }
    els.load.disabled = false;
    refreshHeap();
  } catch (e) {
    els.selftestOut.className = 'out error';
    els.selftestOut.textContent = (e && e.message) ? e.message : String(e);
    fail('self-test FAILED', e);
    els.selftest.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Milestones 3+4: model load
// ---------------------------------------------------------------------------

els.load.addEventListener('click', async () => {
  els.load.disabled = true;
  els.loadProgress.hidden = false;

  const base = els.modelUrl.value.trim() || '/models';
  const vaeUrl = els.vaeUrl.value.trim();
  if (!vaeBytes && !vaeUrl) {
    els.load.disabled = false;
    say('no AudioVAE source: give a URL or select the converted ' +
        'audiovae.safetensors under "AudioVAE source".', 'error');
    return;
  }
  const sources = { base };
  if (vaeUrl) sources.audiovae = vaeUrl;

  say(`loading model from ${base} — 4.4 GB streamed by HTTP Range, never held whole in WASM memory.`);

  const t0 = performance.now();
  let lastHeapCheck = 0;
  const onProgress = (stage, done, total, detail) => {
    els.progressStage.textContent = stage;
    if (total > 0) {
      const pct = (done / total) * 100;
      els.barFill.style.width = `${pct}%`;
      els.progressPct.textContent = `${pct.toFixed(1)}%  (${fmtBytes(done)} / ${fmtBytes(total)})`;
    } else {
      els.progressPct.textContent = '';
    }
    els.progressDetail.textContent = detail;
    const now = performance.now();
    if (now - lastHeapCheck > 500) { lastHeapCheck = now; refreshHeap(); }
  };

  try {
    session = await vox.load_model(
      JSON.stringify(sources),
      vaeBytes || undefined,
      onProgress,
      undefined,
    );
    // The bytes are on the GPU now; let the 359 MB copy go.
    vaeBytes = null;
    els.clearVae.disabled = true;
    sampleRate = session.sample_rate;
    const secs = (performance.now() - t0) / 1000;
    metric('m-load', `${secs.toFixed(1)} s`);
    metric('m-precision', session.precision);
    els.barFill.style.width = '100%';
    els.progressStage.textContent = 'ready';
    els.progressDetail.textContent = '';
    say(`model ready in ${secs.toFixed(1)} s — output sample rate ${sampleRate} Hz.`);
    els.generate.disabled = false;
    refreshHeap();
  } catch (e) {
    fail('model load failed', e);
    els.load.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Reference voice (Milestone 8)
// ---------------------------------------------------------------------------

function ctx() {
  if (!audioCtx) audioCtx = new AudioContext();
  return audioCtx;
}

async function decodeToMono(arrayBuffer) {
  const buf = await ctx().decodeAudioData(arrayBuffer);
  if (buf.numberOfChannels === 1) {
    return { samples: buf.getChannelData(0).slice(), sampleRate: buf.sampleRate };
  }
  // Downmix. Rust resamples to the model's rate; it does not downmix PCM.
  const n = buf.length;
  const out = new Float32Array(n);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += ch[i];
  }
  for (let i = 0; i < n; i++) out[i] /= buf.numberOfChannels;
  return { samples: out, sampleRate: buf.sampleRate };
}

function setReference(ref, label) {
  reference = ref;
  els.clearRef.disabled = !ref;
  els.refStatus.textContent = ref
    ? `${label} — ${(ref.samples.length / ref.sampleRate).toFixed(2)} s @ ${ref.sampleRate} Hz. Stays in this tab.`
    : 'No reference — the model improvises a voice. Audio stays in this tab; nothing is uploaded.';
}

els.refFile.addEventListener('change', async () => {
  const file = els.refFile.files?.[0];
  if (!file) return;
  try {
    setReference(await decodeToMono(await file.arrayBuffer()), file.name);
    say(`reference voice loaded from ${file.name}.`);
  } catch (e) {
    fail('could not decode the reference audio', e);
  }
});

els.clearRef.addEventListener('click', () => {
  els.refFile.value = '';
  setReference(null);
  say('reference voice cleared.');
});

els.record.addEventListener('click', async () => {
  if (recorder) {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      els.record.classList.remove('recording');
      els.record.textContent = '● Record';
      recorder = null;
      try {
        const blob = new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' });
        setReference(await decodeToMono(await blob.arrayBuffer()), 'microphone');
        say('reference voice captured from the microphone.');
      } catch (e) {
        fail('could not decode the recording', e);
      }
    };
    recorder.start();
    els.record.classList.add('recording');
    els.record.textContent = '■ Stop recording';
    say('recording — click again to stop.');
  } catch (e) {
    fail('microphone access failed', e);
    recorder = null;
  }
});

// ---------------------------------------------------------------------------
// Milestone 5: generate
// ---------------------------------------------------------------------------

els.generate.addEventListener('click', async () => {
  if (!session) return;
  const text = els.text.value.trim();
  if (!text) { say('nothing to generate — the text box is empty.', 'error'); return; }

  els.generate.disabled = true;
  els.play.disabled = true;
  els.save.disabled = true;
  vox.set_flag('VOXCPM_Z_ZERO', els.zzero.checked);
  say(`generating: ${text}`);

  const t0 = performance.now();
  try {
    let ttfaMs = null;
    let chunkCount = 0;

    if (els.stream.checked) {
      // Streaming: Rust hands back each chunk as it is produced and we
      // schedule it immediately, back to back, on the AudioContext clock.
      stop();
      const ac = ctx();
      if (ac.state === 'suspended') await ac.resume();
      streamNodes = [];
      nextChunkTime = 0;
      underruns = 0;

      const onChunk = (chunk) => {
        if (ttfaMs === null) {
          ttfaMs = performance.now() - t0;
          metric('m-ttfa', `${(ttfaMs / 1000).toFixed(2)} s`);
        }
        chunkCount += 1;
        scheduleChunk(chunk);
        metric('m-chunks', `${chunkCount} / ${underruns}`);
      };

      pcm = await session.generate_streaming(
        text,
        onChunk,
        parseFloat(els.cfg.value),
        parseInt(els.timesteps.value, 10),
        parseInt(els.maxlen.value, 10),
        parseInt(els.chunkPatches.value, 10),
        reference ? reference.samples : undefined,
        reference ? reference.sampleRate : undefined,
      );
    } else {
      pcm = await session.generate(
        text,
        parseFloat(els.cfg.value),
        parseInt(els.timesteps.value, 10),
        parseInt(els.maxlen.value, 10),
        reference ? reference.samples : undefined,
        reference ? reference.sampleRate : undefined,
      );
    }
    const genMs = performance.now() - t0;
    const audioS = pcm.length / sampleRate;

    let peak = 0;
    for (let i = 0; i < pcm.length; i++) {
      const a = Math.abs(pcm[i]);
      if (a > peak) peak = a;
    }

    metric('m-gen', `${(genMs / 1000).toFixed(2)} s`);
    metric('m-dur', `${audioS.toFixed(2)} s (${pcm.length} samples @ ${sampleRate} Hz)`);
    metric('m-rtf', (genMs / 1000 / audioS).toFixed(3));
    metric('m-peak', peak.toFixed(4));
    refreshHeap();

    say(`done — ${audioS.toFixed(2)} s of audio in ${(genMs / 1000).toFixed(2)} s ` +
        `(RTF ${(genMs / 1000 / audioS).toFixed(3)}), peak ${peak.toFixed(4)}.`);
    // Exposed for automated checks (scripts drive the real UI and then
    // pull the samples out to validate them outside the browser).
    window.__voxPcm = pcm;
    window.__voxSampleRate = sampleRate;
    els.play.disabled = false;
    els.save.disabled = false;
    if (!els.stream.checked) {
      await play();
    }
  } catch (e) {
    fail('generation failed', e);
  } finally {
    els.generate.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Playback: Float32Array -> AudioBuffer -> AudioBufferSourceNode
// ---------------------------------------------------------------------------

async function play() {
  if (!pcm || !pcm.length) return;
  stop();
  const ac = ctx();
  if (ac.state === 'suspended') await ac.resume();
  // The model's rate need not match the AudioContext's; AudioBuffer
  // resamples on playback.
  const buffer = ac.createBuffer(1, pcm.length, sampleRate);
  buffer.copyToChannel(pcm, 0);
  const node = ac.createBufferSource();
  node.buffer = buffer;
  node.connect(ac.destination);
  node.onended = () => {
    if (playing === node) { playing = null; els.stop.disabled = true; }
  };
  node.start();
  playing = node;
  els.stop.disabled = false;
}

// Schedule one streamed chunk to play immediately after the previous one.
//
// Chunks are produced incrementally by the model, so they have to be
// stitched on the AudioContext clock rather than concatenated first. If a
// chunk arrives after its slot has already passed, playback has caught up
// with generation — that is an underrun, and it is counted rather than
// hidden, because it is the number that says whether this is actually
// realtime.
function scheduleChunk(chunk) {
  if (!chunk || !chunk.length) return;
  const ac = ctx();
  const buffer = ac.createBuffer(1, chunk.length, sampleRate);
  buffer.copyToChannel(chunk, 0);
  const node = ac.createBufferSource();
  node.buffer = buffer;
  node.connect(ac.destination);

  const now = ac.currentTime;
  if (nextChunkTime === 0) {
    // Small lead-in so the first chunk is not clipped by scheduling jitter.
    nextChunkTime = now + 0.08;
  } else if (nextChunkTime < now) {
    underruns += 1;
    nextChunkTime = now;
  }
  node.start(nextChunkTime);
  nextChunkTime += buffer.duration;
  streamNodes.push(node);
  els.stop.disabled = false;
}

function stop() {
  if (playing) {
    try { playing.stop(); } catch { /* already stopped */ }
    playing = null;
  }
  for (const n of streamNodes) {
    try { n.stop(); } catch { /* already stopped or not yet started */ }
  }
  streamNodes = [];
  nextChunkTime = 0;
  els.stop.disabled = true;
}

els.play.addEventListener('click', () => { play().catch((e) => fail('playback failed', e)); });
els.stop.addEventListener('click', stop);

els.save.addEventListener('click', () => {
  if (!session || !pcm) return;
  try {
    const bytes = session.encode_wav(pcm);
    const url = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = 'voxcpm2.wav';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    say('WAV saved.');
  } catch (e) {
    fail('WAV encoding failed', e);
  }
});

setReference(null);
// A page served from anywhere but the host itself has no /models to read,
// so default to Hugging Face.
const wantedPrecision = new URL(location.href).searchParams.get('precision');
if (wantedPrecision && ['auto', 'f16', 'f32'].includes(wantedPrecision)) {
  els.precision.value = wantedPrecision;
}
const isLocal = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
els.sourcePreset.value = isLocal ? 'local' : 'hf';
applySourcePreset();
boot();
