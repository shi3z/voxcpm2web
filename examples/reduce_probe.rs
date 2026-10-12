//! Are whole-tensor reductions correct, and does it depend on the element type?
//!
//! A user running the F16 browser bundle on an Apple Metal-3 GPU reported
//! every scalar reduction coming back as exactly `1.0` — `sum([1,2,3,4])`,
//! `mean([1,2,3,4])`, `max([1,2,3,4])`, `sum([0,1,0,1,...][1024])`, all
//! 1.0 — while `mean_dim` along the last axis of a 2-D tensor and `argmax`
//! were both correct. The same shape of failure was seen earlier on an
//! RTX 3090. Reductions back every RMSNorm and the stop head, so an F16
//! build produces wrong audio rather than no audio.
//!
//! This runs the browser self-test's probe set natively, for both element
//! types, to answer one question: is the bug in burn/cubecl's reduce for
//! `f16` generally, or is it specific to the WebGPU backend?
//!
//! ```sh
//! cargo run --release --example reduce_probe --no-default-features --features wgpu
//! ```
//!
//! SwiftShader cannot answer it — it does not report `shader-f16`, so the
//! F16 bundle refuses to run there.

#![recursion_limit = "256"]

use burn::prelude::*;

// The plain `wgpu` backend does not enable naga's FLOAT16 capability, so
// an f16 kernel there fails to compile rather than running — which is
// itself informative, but useless for testing f16 arithmetic. `vulkan`
// is the native configuration that actually runs it.
#[cfg(not(feature = "vulkan"))]
type Bk<F> = burn::backend::wgpu::Wgpu<F, i32>;
#[cfg(feature = "vulkan")]
type Bk<F> = burn::backend::Vulkan<F, i32>;

#[cfg(not(feature = "vulkan"))]
const BACKEND: &str = "wgpu (WGSL)";
#[cfg(feature = "vulkan")]
const BACKEND: &str = "vulkan (SPIR-V)";

struct Probe {
    failures: usize,
    total: usize,
}

impl Probe {
    fn check(&mut self, label: &str, got: f32, want: f32) {
        self.total += 1;
        // f16 carries ~3 decimal digits, so scale the tolerance.
        let tol = (want.abs() * 1e-2).max(1e-2);
        if (got - want).abs() <= tol {
            println!("  {label:<30} {got:>12.4}   ok");
        } else {
            self.failures += 1;
            println!("  {label:<30} {got:>12.4}   FAIL, expected {want}");
        }
    }
}

fn scalar<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map(|v| v.first().copied().unwrap_or(f32::NAN))
        .unwrap_or(f32::NAN)
}

fn run<F>(label: &str) -> usize
where
    F: burn::backend::wgpu::FloatElement,
{
    let device = Default::default();
    println!("\n=== {label} ===");
    let mut p = Probe { failures: 0, total: 0 };

    let t4 = || Tensor::<Bk<F>, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0], &device);
    p.check("sum([1,2,3,4])", scalar(t4().sum()), 10.0);
    p.check("mean([1,2,3,4])", scalar(t4().mean()), 2.5);
    p.check("max([1,2,3,4])", scalar(t4().max()), 4.0);
    p.check("min([1,2,3,4])", scalar(t4().min()), 1.0);

    let skew = || Tensor::<Bk<F>, 1>::from_floats([5.0f32, 1.0, 1.0, 1.0], &device);
    p.check("sum([5,1,1,1])", scalar(skew().sum()), 8.0);
    p.check("mean([5,1,1,1])", scalar(skew().mean()), 2.0);

    for len in [1usize, 2, 3, 8, 64, 1024] {
        let ones = Tensor::<Bk<F>, 1>::ones([len], &device);
        p.check(&format!("sum(ones[{len}])"), scalar(ones.clone().sum()), len as f32);
        p.check(&format!("mean(ones[{len}])"), scalar(ones.mean()), 1.0);
    }

    for len in [8usize, 64, 1024] {
        let alt: Vec<f32> = (0..len).map(|i| (i % 2) as f32).collect();
        let t = Tensor::<Bk<F>, 1>::from_data(
            burn::tensor::TensorData::new(alt, [len]),
            &device,
        );
        p.check(
            &format!("sum([0,1,0,1,...][{len}])"),
            scalar(t.sum()),
            (len / 2) as f32,
        );
    }

    // The model's hot path: reduce the last dimension of a 2-D activation,
    // which is a different cubek-reduce routine than reduce-to-scalar.
    let rows = Tensor::<Bk<F>, 2>::from_data(
        burn::tensor::TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 4],
        ),
        &device,
    );
    let per_row = rows
        .mean_dim(1)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap_or_default();
    p.check(
        "mean_dim([[1..4],[5..8]],1)[0]",
        per_row.first().copied().unwrap_or(f32::NAN),
        2.5,
    );
    p.check(
        "mean_dim([[1..4],[5..8]],1)[1]",
        per_row.get(1).copied().unwrap_or(f32::NAN),
        6.5,
    );

    // Same reduction, but wide enough to cross a workgroup boundary —
    // that is where a plane/unit routine would be chosen differently.
    let wide_n = 4096usize;
    let wide = Tensor::<Bk<F>, 2>::ones([2, wide_n], &device);
    let wide_sum = wide
        .sum_dim(1)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap_or_default();
    p.check(
        &format!("sum_dim(ones[2,{wide_n}],1)[0]"),
        wide_sum.first().copied().unwrap_or(f32::NAN),
        wide_n as f32,
    );

    println!("  -> {}/{} probes failed", p.failures, p.total);
    p.failures
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("warn,wgpu_hal=error,wgpu_core=error,naga=error"),
    )
    .init();

    println!("backend: {BACKEND}");

    let f32_failures = run::<f32>("f32");
    let f16_failures = run::<burn::tensor::f16>("f16 (half::f16)");

    println!("\nsummary: f32 {f32_failures} failure(s), f16 {f16_failures} failure(s)");
    if f16_failures > 0 && f32_failures == 0 {
        println!(
            "\nThe bug reproduces natively and only for f16, so it is not specific\n\
             to WebGPU or to a particular driver — it is in the f16 reduce path."
        );
    } else if f16_failures == 0 && f32_failures == 0 {
        println!(
            "\nBoth element types are correct here, so the browser failure is\n\
             specific to the WebGPU backend or to the reporting device."
        );
    }
}
