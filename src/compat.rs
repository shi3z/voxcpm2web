//! Small platform-compat shims so the model/weight code can be shared
//! verbatim between the native build and the `wasm32-unknown-unknown`
//! browser build.
//!
//! Two things in `std` are compiled but non-functional on
//! `wasm32-unknown-unknown`:
//!
//! * [`std::time::Instant::now`] **panics** ("time not implemented on this
//!   platform"). Any profiling timer therefore has to route through
//!   `performance.now()` instead.
//! * [`std::env::var`] always returns `Err`, which is harmless but means the
//!   debug env-var switches silently do nothing. We make that explicit.

/// Monotonic wall-clock in fractional milliseconds.
///
/// Native: `std::time::Instant` against a process-lifetime origin.
/// Browser: `performance.now()` from whichever global scope we are in
/// (`Window` on the main thread, `WorkerGlobalScope` inside a worker).
#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let origin = ORIGIN.get_or_init(Instant::now);
    origin.elapsed().as_secs_f64() * 1000.0
}

#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    performance_now().unwrap_or(0.0)
}

#[cfg(target_arch = "wasm32")]
fn performance_now() -> Option<f64> {
    use wasm_bindgen::JsCast;
    let global = js_sys::global();
    if let Some(w) = global.dyn_ref::<web_sys::Window>() {
        return w.performance().map(|p| p.now());
    }
    if let Some(s) = global.dyn_ref::<web_sys::WorkerGlobalScope>() {
        return s.performance().map(|p| p.now());
    }
    None
}

/// A started stopwatch. Replaces the `Instant::now()` / `elapsed()` pairs
/// that would panic in the browser.
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch(f64);

impl Stopwatch {
    /// Start timing now.
    pub fn start() -> Self {
        Self(now_ms())
    }

    /// Milliseconds since [`Stopwatch::start`].
    pub fn elapsed_ms(&self) -> f64 {
        now_ms() - self.0
    }

    /// Milliseconds between two start marks — the analogue of
    /// `Instant::duration_since`, for code that takes a sequence of marks
    /// and differences them afterwards.
    pub fn since(&self, earlier: &Stopwatch) -> f64 {
        self.0 - earlier.0
    }
}

/// Whether a debug environment variable is set.
///
/// Always `false` in the browser — `std::env` has no backing store there, so
/// the knob is exposed through `browser::set_debug_flag` instead (that
/// module only exists on `wasm32`, so this is not a link).
#[cfg(not(target_arch = "wasm32"))]
pub fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}

#[cfg(target_arch = "wasm32")]
pub fn env_flag(name: &str) -> bool {
    crate::browser::debug_flag(name)
}
