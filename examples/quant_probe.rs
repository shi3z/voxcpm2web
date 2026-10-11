//! Does quantized matmul actually work on the WGSL backend?
//!
//! This decides whether a Q4/Q8 build is possible at all, so it is worth
//! answering in a minute rather than after a day of plumbing.
//!
//! VoxCPM2's weights are 4.37 GB of BF16. Dropping to 4-bit would take
//! the download and the VRAM to roughly 1.3 GB — but only if the weights
//! can stay quantized and be dequantized *inside* the matmul. Burn
//! exposes that (`q_matmul`), and `QuantStore::PackedU32` is the storage
//! WGSL needs since WGSL has no 8-bit type. Whether the kernels actually
//! compile and compute correctly through naga/WGSL is the open question.
//!
//! ```sh
//! cargo run --release --example quant_probe --no-default-features --features wgpu
//! ```

#![recursion_limit = "256"]

use burn::prelude::*;
use burn::tensor::quantization::{QuantLevel, QuantScheme, QuantStore, QuantValue};

#[cfg(all(feature = "wgpu", not(feature = "vulkan")))]
type B = burn::backend::Wgpu<f32, i32>;
#[cfg(all(feature = "wgpu", feature = "vulkan"))]
type B = burn::backend::Vulkan<f32, i32>;
#[cfg(all(not(feature = "wgpu"), not(feature = "vulkan"), feature = "cpu"))]
type B = burn::backend::NdArray<f32>;

/// Deterministic pseudo-random values in [-1, 1], so runs are comparable.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ((x >> 8) as f32 / 8388608.0) - 1.0
        })
        .collect()
}

fn try_scheme(label: &str, scheme: QuantScheme, device: &<B as Backend>::Device) {
    // Shapes in the ballpark of a real projection: [1, 2048] x [2048, 2048].
    let (m, k, n) = (1usize, 2048usize, 2048usize);

    let a: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(values(m * k, 1), [m, k]), device);
    let w: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(values(k * n, 2), [k, n]), device);

    let reference = a.clone().matmul(w.clone());

    // `catch_unwind` because an unsupported kernel surfaces as a panic from
    // deep inside cubecl, and the point here is to report that rather than
    // die on it.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let wq = w.clone().quantize_dynamic(&scheme);
        let out = a.clone().matmul(wq);
        out.into_data()
    }));

    match result {
        Err(_) => println!("  {label:<34} UNSUPPORTED (kernel panicked)"),
        Ok(data) => {
            let got: Vec<f32> = match data.convert::<f32>().into_vec::<f32>() {
                Ok(v) => v,
                Err(_) => {
                    println!("  {label:<34} FAILED (unexpected output dtype)");
                    return;
                }
            };
            let want: Vec<f32> = reference
                .into_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("reference dtype");

            if got.len() != want.len() {
                println!(
                    "  {label:<34} FAILED (got {} elements, want {})",
                    got.len(),
                    want.len()
                );
                return;
            }
            let nonfinite = got.iter().filter(|v| !v.is_finite()).count();
            // Relative L2: the honest measure for quantization, since
            // individual elements can be off by a lot while the result is
            // still usable.
            let num: f64 = got
                .iter()
                .zip(&want)
                .map(|(g, w)| ((*g - *w) as f64).powi(2))
                .sum();
            let den: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum();
            let rel = (num / den.max(1e-30)).sqrt();
            let verdict = if nonfinite > 0 {
                "BROKEN (non-finite)"
            } else if rel < 0.02 {
                "ok"
            } else if rel < 0.15 {
                "ok-ish (lossy, as expected for 4-bit)"
            } else {
                "SUSPECT (error too large)"
            };
            println!("  {label:<34} rel.L2 {rel:>8.5}  {verdict}");
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        "error,wgpu_hal=off,wgpu_core=off,naga=off,cubecl_wgpu=off",
    ))
    .init();
    // Keep cubecl's own panic noise out of the report; we catch and label it.
    std::panic::set_hook(Box::new(|_| {}));

    let device = Default::default();
    println!("quantized matmul on this backend, [1,2048] x [2048,2048]:\n");

    let base = QuantScheme::default().with_store(QuantStore::PackedU32(0));

    for (label, scheme) in [
        ("Q8S per-tensor", base.with_value(QuantValue::Q8S)),
        ("Q8F per-tensor", base.with_value(QuantValue::Q8F)),
        (
            "Q8S per-block(32)",
            base.with_value(QuantValue::Q8S).with_level(QuantLevel::block([32])),
        ),
        ("Q4S per-tensor", base.with_value(QuantValue::Q4S)),
        ("Q4F per-tensor", base.with_value(QuantValue::Q4F)),
        (
            "Q4S per-block(32)",
            base.with_value(QuantValue::Q4S).with_level(QuantLevel::block([32])),
        ),
        (
            "Q4S per-block(64)",
            base.with_value(QuantValue::Q4S).with_level(QuantLevel::block([64])),
        ),
        (
            "Q4S per-block(128)",
            base.with_value(QuantValue::Q4S).with_level(QuantLevel::block([128])),
        ),
    ] {
        try_scheme(label, scheme, &device);
    }

    let _ = std::panic::take_hook();

    // Is an already-quantized tensor detectable? The streaming loader
    // quantizes after every batch and must skip what it already did, so
    // this predicate is load-bearing.
    println!("\nidempotency check:");
    let t: Tensor<B, 2> = Tensor::from_data(TensorData::new(values(64 * 64, 3), [64, 64]), &device);
    println!("  float tensor dtype      : {:?}", t.dtype());
    let q = t.clone().quantize_dynamic(&QuantScheme::default()
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S));
    println!("  quantized tensor dtype  : {:?}", q.dtype());
    println!(
        "  matches DType::QFloat(_)? {}",
        matches!(q.dtype(), burn::tensor::DType::QFloat(_))
    );

    println!(
        "\nrel.L2 is relative L2 error against the float matmul. 4-bit is\n\
         lossy by construction, so 'ok-ish' is the expected good outcome;\n\
         UNSUPPORTED means the kernel does not exist on this backend."
    );
}
