//! Print what the GPU backend actually supports.
//!
//! Useful for answering "will `--f16` work on my machine?" before waiting
//! out a 4.4 GB load, and for diagnosing why a kernel failed to compile.
//! The browser equivalent is the demo page's Diagnostics panel.
//!
//! ```sh
//! cargo run --release --example gpu_features --no-default-features --features wgpu
//! ```

#![recursion_limit = "256"]

#[cfg(any(feature = "wgpu", feature = "vulkan"))]
fn main() {
    use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup};

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        "warn,wgpu_hal=error,wgpu_core=error,naga=error",
    ))
    .init();

    let device = Default::default();
    let setup = init_setup::<AutoGraphicsApi>(&device, Default::default());

    let info = setup.adapter.get_info();
    println!("adapter      : {} ({:?})", info.name, info.backend);
    println!("device type  : {:?}", info.device_type);
    println!("driver       : {} {}", info.driver, info.driver_info);

    // `Features` is a bitflags type that `burn-wgpu` does not re-export,
    // so it is inspected through its Debug rendering.
    let adapter_features = format!("{:?}", setup.adapter.features());
    let device_features = format!("{:?}", setup.device.features());

    let has = |hay: &str, name: &str| if hay.contains(name) { "yes" } else { "no " };
    println!();
    println!("                        adapter  device");
    for f in [
        "SHADER_F16",
        "SHADER_F64",
        "SHADER_INT64",
        "SUBGROUP",
        "TIMESTAMP_QUERY",
        "PIPELINE_CACHE",
    ] {
        println!(
            "  {f:<22} {}      {}",
            has(&adapter_features, f),
            has(&device_features, f)
        );
    }

    let limits = setup.device.limits();
    println!();
    println!("maxBufferSize                   : {}", limits.max_buffer_size);
    println!(
        "maxStorageBufferBindingSize     : {}",
        limits.max_storage_buffer_binding_size
    );
    println!(
        "maxComputeInvocationsPerWorkgroup: {}",
        limits.max_compute_invocations_per_workgroup
    );
    println!(
        "maxComputeWorkgroupStorageSize  : {}",
        limits.max_compute_workgroup_storage_size
    );

    println!();
    if device_features.contains("SHADER_F16") {
        println!("f16: the device has SHADER_F16 — `--f16` should work.");
    } else if adapter_features.contains("SHADER_F16") {
        println!(
            "f16: the ADAPTER advertises SHADER_F16 but the DEVICE does not have it \
             enabled — f16 kernels will fail to compile."
        );
    } else {
        println!(
            "f16: SHADER_F16 is not advertised by this adapter, so `--f16` will fail \
             with\n     \"Using `f16` values requires the naga::valid::Capabilities::FLOAT16 flag\".\n\
             \n     Note this is the *native* WGSL path, which validates through naga.\n\
             In a browser, WGSL is compiled by the browser itself (Dawn in Chrome),\n\
             so a browser whose adapter reports `shader-f16` is a separate question."
        );
    }

    // The 602 MB (F32) / 301 MB (F16) token embedding is the binding that
    // decides whether this model can be loaded at all.
    let embed_f32: u64 = 73448 * 2048 * 4;
    println!();
    println!(
        "token embedding at F32 is {} bytes: {}",
        embed_f32,
        if limits.max_storage_buffer_binding_size as u64 >= embed_f32 {
            "fits"
        } else {
            "DOES NOT FIT in maxStorageBufferBindingSize"
        }
    );
}

#[cfg(not(any(feature = "wgpu", feature = "vulkan")))]
fn main() {
    eprintln!("build with --features wgpu (or vulkan)");
}
