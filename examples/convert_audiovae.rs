//! Convert `audiovae.pth` to `audiovae.safetensors`.
//!
//! Upstream `openbmb/VoxCPM2` ships the AudioVAE only as a PyTorch pickle.
//! The browser build cannot read it: `burn_store`'s `PytorchReader` is
//! path-based, and the `pytorch` feature pulls `zip`, whose default
//! compression backends include `zstd` — and `zstd-sys` is C, so it does
//! not build for `wasm32-unknown-unknown`. See
//! `docs/browser-port-analysis.md` (B10).
//!
//! Rather than require a Python/torch install just to transcode a file,
//! this reuses the pickle reader the crate already has on native.
//!
//! Shapes and dtypes are copied through **unchanged** — in particular the
//! `weight_g` / `weight_v` pairs stay un-materialized, because the loader
//! does that itself (and must, since the effective weight depends on the
//! target dtype).
//!
//! Keys get one transform: the top-level container prefix that HF-published
//! pickles wrap their weights in (`state_dict.` here) is stripped, exactly
//! as `weights::load_single_pth` does on the native path. Without it the
//! AudioVAE key remapper sees `state_dict.decoder.model.…` instead of
//! `decoder.model.…` and silently matches nothing.
//!
//! ```sh
//! cargo run --release --example convert_audiovae \
//!     --no-default-features --features cpu -- /path/to/VoxCPM2
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use burn_store::pytorch::PytorchReader;
use safetensors::{Dtype, serialize};

/// Strip the top-level container prefix HF-published pickles use, so
/// downstream key remapping sees bare parameter paths. Mirrors the private
/// `strip_pth_top_level` in `voxcpm_rs::weights`.
fn strip_top_level(name: &str) -> &str {
    for prefix in ["state_dict.", "model_state_dict.", "module."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            return rest;
        }
    }
    name
}

fn burn_dtype_to_safetensors(dt: burn::tensor::DType) -> Option<Dtype> {
    use burn::tensor::DType as D;
    Some(match dt {
        D::F64 => Dtype::F64,
        D::F32 | D::Flex32 => Dtype::F32,
        D::F16 => Dtype::F16,
        D::BF16 => Dtype::BF16,
        D::I64 => Dtype::I64,
        D::I32 => Dtype::I32,
        D::I16 => Dtype::I16,
        D::I8 => Dtype::I8,
        D::U64 => Dtype::U64,
        D::U32 => Dtype::U32,
        D::U16 => Dtype::U16,
        D::U8 => Dtype::U8,
        D::Bool => Dtype::BOOL,
        _ => return None,
    })
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let dir: PathBuf = match std::env::args().nth(1) {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!(
                "usage: convert_audiovae <VoxCPM2-checkpoint-dir>\n\n\
                 Reads  <dir>/audiovae.pth\n\
                 Writes <dir>/audiovae.safetensors"
            );
            std::process::exit(2);
        }
    };

    let src = dir.join("audiovae.pth");
    let dst = dir.join("audiovae.safetensors");
    if !src.exists() {
        eprintln!("not found: {}", src.display());
        std::process::exit(1);
    }

    println!("reading {}", src.display());
    let reader = match PytorchReader::new(&src) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to read {}: {e}", src.display());
            std::process::exit(1);
        }
    };

    // `(dtype, shape, bytes)` per tensor, keyed by the original name.
    let mut owned: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut stripped = 0usize;

    for (name, snapshot) in reader.tensors() {
        let data = match snapshot.to_data() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  skip `{name}`: cannot materialize ({e:?})");
                skipped.push(name.clone());
                continue;
            }
        };
        let Some(dtype) = burn_dtype_to_safetensors(data.dtype) else {
            eprintln!("  skip `{name}`: dtype {:?} has no safetensors form", data.dtype);
            skipped.push(name.clone());
            continue;
        };
        let bare = strip_top_level(name).to_string();
        if bare != *name {
            stripped += 1;
        }
        owned.insert(bare, (dtype, data.shape.clone(), data.as_bytes().to_vec()));
    }
    if stripped > 0 {
        println!("  stripped a top-level container prefix from {stripped} key(s)");
    }

    if owned.is_empty() {
        eprintln!("no tensors could be converted");
        std::process::exit(1);
    }

    let total: usize = owned.values().map(|(_, _, b)| b.len()).sum();
    println!(
        "converting {} tensors ({:.1} MB){}",
        owned.len(),
        total as f64 / 1048576.0,
        if skipped.is_empty() {
            String::new()
        } else {
            format!(", {} skipped", skipped.len())
        }
    );

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = owned
        .iter()
        .map(|(k, (dtype, shape, data))| {
            let view = safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                .unwrap_or_else(|e| panic!("compose view `{k}`: {e}"));
            (k.clone(), view)
        })
        .collect();

    let bytes = match serialize(views.iter().map(|(k, v)| (k.clone(), v)), &None) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("serialize failed: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::write(&dst, &bytes) {
        eprintln!("write {} failed: {e}", dst.display());
        std::process::exit(1);
    }
    println!(
        "wrote {} ({:.1} MB)",
        dst.display(),
        bytes.len() as f64 / 1048576.0
    );
}
