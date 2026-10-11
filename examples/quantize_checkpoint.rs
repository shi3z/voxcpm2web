//! Produce a pre-quantized VoxCPM2 checkpoint.
//!
//! Quantizing in the browser shrinks GPU residency but not the download:
//! all 4.37 GB still has to arrive first. Quantizing here and shipping the
//! result shrinks both — to about **2.6 GB** at Q8.
//!
//! ```sh
//! cargo run --release --example quantize_checkpoint \
//!     --no-default-features --features wgpu -- \
//!     /path/to/VoxCPM2 /path/to/out q8
//! ```
//!
//! Writes `<out>/model.safetensors` plus copies of `config.json`,
//! `tokenizer.json` and `audiovae.safetensors`, so the output directory is
//! a drop-in replacement for the input.
//!
//! The weights are loaded the normal way first — remapped, fused and
//! transposed by `PyTorchToBurnAdapter` — then quantized, then exported
//! under their final module paths. That is why the result must be loaded
//! **without** the adapter; the loader detects this from the manifest.

#![recursion_limit = "256"]

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::tensor::DType;
use burn_store::ModuleSnapshot;
use safetensors::Dtype;
use voxcpm_rs::quant::{QuantManifest, QuantPreset, MANIFEST_VERSION};
use voxcpm_rs::stream::{self, Checkpoint, FileSource, LogProgress};
use voxcpm_rs::weights::{self, Namespace};
use voxcpm_rs::VoxCpm2Config;

#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
type B = burn::backend::Wgpu<f32, i32>;
#[cfg(all(feature = "wgpu", feature = "vulkan"))]
type B = burn::backend::Vulkan<f32, i32>;
#[cfg(all(not(feature = "wgpu"), not(feature = "vulkan"), feature = "cpu"))]
type B = burn::backend::NdArray<f32>;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        "info,wgpu_hal=error,wgpu_core=error,naga=error,cubecl_wgpu=warn",
    ))
    .init();

    let mut args = std::env::args().skip(1);
    let src_dir = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!(
            "usage: quantize_checkpoint <in-dir> <out-dir> [q8]\n\n\
             <in-dir> needs config.json, tokenizer.json, model.safetensors and\n\
             audiovae.safetensors (run convert_audiovae first)."
        );
        std::process::exit(2);
    }));
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("missing <out-dir>");
        std::process::exit(2);
    }));
    let spec = args.next().unwrap_or_else(|| "q8".to_string());

    let preset = match QuantPreset::parse(&spec) {
        Some(p) => p,
        None => {
            eprintln!("unknown preset `{spec}` (q8|q4|q4b32|q4b64|q4b128)");
            std::process::exit(2);
        }
    };
    if !preset.is_supported() {
        eprintln!(
            "warning: {preset:?} does not currently run inference on this model; \
             the file will be written but will not load"
        );
    }

    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let device = Default::default();

    // --- load the upstream checkpoint, quantizing as it streams ----------
    let config_bytes = std::fs::read(src_dir.join("config.json")).expect("read config.json");
    let config: VoxCpm2Config = serde_json::from_slice(&config_bytes).expect("parse config.json");
    let mut model = voxcpm_rs::voxcpm2::VoxCpm2Model::<B>::new(config, &device);

    let model_path = src_dir.join("model.safetensors");
    println!("loading {} (quantizing to {spec} as it streams)", model_path.display());
    let src = FileSource::open(&model_path).expect("open model.safetensors");
    let checkpoint = stream::block_on_ready(Checkpoint::open(&src)).expect("parse header");
    let (m, result) = stream::block_on_ready(stream::stream_into::<B, _, _, _>(
        model,
        &src,
        &checkpoint,
        Namespace::Model,
        stream::DEFAULT_BATCH_BUDGET,
        "weights",
        &LogProgress,
        Some(preset),
    ))
    .expect("stream weights");
    model = m;
    let mut acc = result;

    let vae_path = src_dir.join("audiovae.safetensors");
    if !vae_path.exists() {
        eprintln!(
            "missing {}\nRun: cargo run --release --example convert_audiovae \
             --no-default-features --features cpu -- {}",
            vae_path.display(),
            src_dir.display()
        );
        std::process::exit(1);
    }
    let vae_src = FileSource::open(&vae_path).expect("open audiovae");
    let vae_ck = stream::block_on_ready(Checkpoint::open(&vae_src)).expect("parse vae header");
    let (m, r) = stream::block_on_ready(stream::stream_into::<B, _, _, _>(
        model,
        &vae_src,
        &vae_ck,
        Namespace::AudioVae,
        stream::DEFAULT_BATCH_BUDGET,
        "audiovae",
        &LogProgress,
        None,
    ))
    .expect("stream audiovae");
    model = m;
    weights::merge_apply_result(&mut acc, r);
    weights::finalize_apply_result(&mut acc);

    if !acc.errors.is_empty() || !acc.missing.is_empty() {
        eprintln!(
            "refusing to export an incomplete model: {} missing, {} errors",
            acc.missing.len(),
            acc.errors.len()
        );
        for (k, _) in acc.missing.iter().take(10) {
            eprintln!("  missing {k}");
        }
        std::process::exit(1);
    }
    println!("loaded: applied={}", acc.applied.len());

    // --- export ----------------------------------------------------------
    // `collect` yields every parameter under its final module path, which
    // is exactly what the loader will look for.
    println!("\nexporting...");
    let snapshots = model.collect(None, None, false);
    println!("  {} parameters", snapshots.len());

    // Owned buffers, because safetensors views borrow.
    let mut blobs: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> = Vec::with_capacity(snapshots.len());
    let mut quantized: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut scheme = None;
    let (mut q_bytes, mut f_bytes) = (0usize, 0usize);

    for snap in &snapshots {
        let path = snap.full_path();
        let data = snap
            .to_data()
            .unwrap_or_else(|e| panic!("materialize `{path}`: {e:?}"));

        match data.dtype {
            DType::QFloat(s) => {
                // Store verbatim: packed values with the scales appended,
                // exactly what `q_from_data` expects back.
                if let Some(prev) = scheme {
                    assert_eq!(prev, s, "mixed quantization schemes in one checkpoint");
                } else {
                    scheme = Some(s);
                }
                let bytes = data.as_bytes().to_vec();
                q_bytes += bytes.len();
                quantized.insert(path.clone(), data.shape.clone());
                // Flat U8: safetensors has no quantized dtype, so the real
                // shape lives in the manifest.
                let len = bytes.len();
                blobs.push((path, vec![len], Dtype::U8, bytes));
            }
            other => {
                // Everything left in floating point goes out at 16 bits:
                // these are embeddings, convolutions and norms, and
                // halving them saves ~490 MB of download for no quality
                // cost beyond what the F32-to-F16 round trip already is.
                let shape = data.shape.clone();
                let (dtype, bytes) = match other {
                    DType::F32 | DType::F16 | DType::BF16 => {
                        (Dtype::F16, data.convert::<half::f16>().as_bytes().to_vec())
                    }
                    _ => (
                        burn_dtype_to_safetensors(other)
                            .unwrap_or_else(|| panic!("no safetensors dtype for {other:?}")),
                        data.as_bytes().to_vec(),
                    ),
                };
                f_bytes += bytes.len();
                blobs.push((path, shape, dtype, bytes));
            }
        }
    }

    let scheme = scheme.expect("nothing was quantized");
    let manifest = QuantManifest {
        version: MANIFEST_VERSION,
        scheme,
        quantized,
    };
    println!(
        "  {} quantized ({:.0} MB) + {} float ({:.0} MB) = {:.2} GB",
        manifest.quantized.len(),
        q_bytes as f64 / 1048576.0,
        blobs.len() - manifest.quantized.len(),
        f_bytes as f64 / 1048576.0,
        (q_bytes + f_bytes) as f64 / 1073741824.0,
    );

    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        voxcpm_rs::quant::MANIFEST_KEY.to_string(),
        manifest.to_json().expect("serialize manifest"),
    );

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = blobs
        .iter()
        .map(|(name, shape, dtype, bytes)| {
            let v = safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes)
                .unwrap_or_else(|e| panic!("view `{name}`: {e}"));
            (name.clone(), v)
        })
        .collect();

    let out_model = out_dir.join("model.safetensors");
    println!("\nwriting {}", out_model.display());
    let serialized = safetensors::serialize(
        views.iter().map(|(k, v)| (k.clone(), v)),
        &Some(metadata),
    )
    .expect("serialize safetensors");
    std::fs::write(&out_model, &serialized).expect("write model");
    println!("  {:.2} GB", serialized.len() as f64 / 1073741824.0);

    // --- the small files come along so the directory is self-sufficient ---
    for name in ["config.json", "tokenizer.json"] {
        let from = src_dir.join(name);
        if from.exists() {
            std::fs::copy(&from, out_dir.join(name)).expect("copy");
            println!("  copied {name}");
        }
    }
    // The AudioVAE is already in the exported model, but the loader still
    // wants the file for an unquantized run, and it is only 359 MB.
    println!(
        "\ndone. The AudioVAE is baked into model.safetensors, so this directory\n\
         needs no audiovae.safetensors."
    );
}

fn burn_dtype_to_safetensors(dt: DType) -> Option<Dtype> {
    Some(match dt {
        DType::F64 => Dtype::F64,
        DType::F32 | DType::Flex32 => Dtype::F32,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
        DType::I64 => Dtype::I64,
        DType::I32 => Dtype::I32,
        DType::I16 => Dtype::I16,
        DType::I8 => Dtype::I8,
        DType::U64 => Dtype::U64,
        DType::U32 => Dtype::U32,
        DType::U16 => Dtype::U16,
        DType::U8 => Dtype::U8,
        DType::Bool => Dtype::BOOL,
        _ => return None,
    })
}
