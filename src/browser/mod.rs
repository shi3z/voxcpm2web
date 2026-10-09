//! Browser runtime: WASM32 + WebGPU.
//!
//! Compiled only for `target_arch = "wasm32"`. Everything here exists to
//! bridge three gaps between the native crate and a browser tab:
//!
//! * **No filesystem.** Checkpoint bytes arrive over `fetch`. See
//!   [`fetch`] for the HTTP byte source; the streaming safetensors reader
//!   that keeps a 4.4 GB checkpoint from ever existing in WASM linear
//!   memory at once is platform-independent and lives in
//!   [`crate::stream`].
//! * **No blocking.** WebGPU readback is `mapAsync`, so
//!   `Tensor::into_data()` panics here (cubecl's `block_on` is
//!   `poll_once` + `expect` on wasm). The async inference twins in
//!   [`crate::voxcpm2`] exist for this reason, and every entry point in
//!   [`api`] is `async`.
//! * **No env vars, no `Instant`.** See [`crate::compat`].
//!
//! The public JS surface lives in [`api`].

pub mod api;
pub mod fetch;

use std::cell::RefCell;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Debug flags — the browser stand-in for `std::env::var`
// ---------------------------------------------------------------------------

thread_local! {
    static DEBUG_FLAGS: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

/// Whether a debug flag is set. Backs [`crate::compat::env_flag`], which is
/// what `VOXCPM_PROFILE` / `VOXCPM_Z_ZERO` read.
pub fn debug_flag(name: &str) -> bool {
    DEBUG_FLAGS.with(|f| f.borrow().contains(name))
}

/// Set or clear a debug flag.
///
/// `VOXCPM_Z_ZERO` is the useful one: it replaces the diffusion sampler's
/// Gaussian noise with zeros, making generation deterministic so browser
/// output can be diffed against a native run. `VOXCPM_PROFILE` turns on
/// the per-step AR timing log.
pub fn set_debug_flag(name: &str, on: bool) {
    DEBUG_FLAGS.with(|f| {
        let mut f = f.borrow_mut();
        if on {
            f.insert(name.to_string());
        } else {
            f.remove(name);
        }
    });
}

// ---------------------------------------------------------------------------
// log -> console bridge
// ---------------------------------------------------------------------------

struct ConsoleLogger;

impl log::Log for ConsoleLogger {
    fn enabled(&self, _m: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        let msg = format!("[{}] {}", record.target(), record.args());
        let js = wasm_bindgen::JsValue::from_str(&msg);
        match record.level() {
            log::Level::Error => web_sys::console::error_1(&js),
            log::Level::Warn => web_sys::console::warn_1(&js),
            log::Level::Info => web_sys::console::info_1(&js),
            log::Level::Debug | log::Level::Trace => web_sys::console::debug_1(&js),
        }
    }

    fn flush(&self) {}
}

static LOGGER: ConsoleLogger = ConsoleLogger;

/// Install the panic hook and the `log` → `console` bridge.
///
/// Without the panic hook a Rust `panic!` reaches JS as a bare
/// `RuntimeError: unreachable` with no message and no stack, which makes
/// every failure in here indistinguishable. Idempotent.
pub fn init_logging(level: log::LevelFilter) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        console_error_panic_hook::set_once();
        let _ = log::set_logger(&LOGGER);
    });
    // Always honour the latest requested level, even on repeat calls.
    log::set_max_level(level);
}
