//! HTTP byte sources for the browser.
//!
//! The brief sketches a `ModelSource { async fn load(&self, name) -> Vec<u8> }`
//! abstraction. That shape does not survive contact with this checkpoint:
//! `model.safetensors` is 4.37 GB and `wasm32` has a 4 GB address space, so
//! "give me the whole file" is not an operation that can succeed. The
//! abstraction here is therefore over *byte ranges* instead of whole files:
//!
//! ```text
//! HTTP ──Range──► fetch ──► ArrayBuffer ──► Uint8Array ──► Vec<u8>
//!                                                            │
//!                         dtype convert, fuse, weight_norm ──┘
//!                                                            ▼
//!                                TensorSnapshot ──► burn-store ──► WebGPU
//! ```
//!
//! [`HttpRangeSource`] implements [`ByteSource`], so
//! [`crate::stream`] drives it with exactly the same code it uses for a
//! native file. It needs a server that honours `Range` — see
//! `scripts/serve.py`.
//!
//! For the small files (`config.json`, `tokenizer.json`,
//! `audiovae.safetensors`) and for bytes handed in from JS (an
//! `<input type="file">`, a Cache Storage or OPFS hit), use
//! [`fetch_bytes`] with [`crate::stream::MemorySource`] instead.

use js_sys::Uint8Array;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Headers, Request, RequestInit, Response};

use crate::stream::ByteSource;
use crate::{Error, Result};

fn js_err(ctx: &str, e: JsValue) -> Error {
    Error::Other(format!("{ctx}: {}", describe(&e)))
}

/// Best-effort human-readable form of a `JsValue`. `JsValue`'s `Debug` is
/// often just `JsValue(Object)`, which tells the user nothing — pull the
/// `message` out when it is an `Error`.
fn describe(e: &JsValue) -> String {
    if let Some(s) = e.as_string() {
        return s;
    }
    if let Some(err) = e.dyn_ref::<js_sys::Error>() {
        return String::from(err.message());
    }
    if let Ok(msg) = js_sys::Reflect::get(e, &JsValue::from_str("message")) {
        if let Some(s) = msg.as_string() {
            return s;
        }
    }
    format!("{e:?}")
}

/// Call `fetch` from whichever global scope we are running in — `Window` on
/// the main thread, `WorkerGlobalScope` inside a worker.
fn fetch(request: &Request) -> Result<js_sys::Promise> {
    let global = js_sys::global();
    if let Some(w) = global.dyn_ref::<web_sys::Window>() {
        return Ok(w.fetch_with_request(request));
    }
    if let Some(s) = global.dyn_ref::<web_sys::WorkerGlobalScope>() {
        return Ok(s.fetch_with_request(request));
    }
    Err(Error::Other(
        "no `fetch` in this global scope (expected Window or WorkerGlobalScope)".into(),
    ))
}

async fn send(request: Request, ctx: &str) -> Result<Response> {
    let promise = fetch(&request)?;
    let value = JsFuture::from(promise)
        .await
        .map_err(|e| js_err(&format!("{ctx}: network error"), e))?;
    let response: Response = value
        .dyn_into()
        .map_err(|e| js_err(&format!("{ctx}: not a Response"), e))?;
    Ok(response)
}

async fn response_bytes(response: Response, ctx: &str) -> Result<Vec<u8>> {
    let buf = JsFuture::from(
        response
            .array_buffer()
            .map_err(|e| js_err(&format!("{ctx}: arrayBuffer()"), e))?,
    )
    .await
    .map_err(|e| js_err(&format!("{ctx}: reading body"), e))?;
    Ok(Uint8Array::new(&buf).to_vec())
}

fn build_request(url: &str, range: Option<(u64, u64)>) -> Result<Request> {
    let init = RequestInit::new();
    init.set_method("GET");
    // `default` lets the HTTP cache serve repeat range requests, which is
    // what makes a page reload cheap before a real OPFS/Cache Storage layer
    // exists.
    init.set_cache(web_sys::RequestCache::Default);

    if let Some((start, end_inclusive)) = range {
        let headers = Headers::new().map_err(|e| js_err("Headers::new", e))?;
        headers
            .set("Range", &format!("bytes={start}-{end_inclusive}"))
            .map_err(|e| js_err("set Range header", e))?;
        init.set_headers(&headers);
    }

    Request::new_with_str_and_init(url, &init)
        .map_err(|e| js_err(&format!("Request::new(`{url}`)"), e))
}

/// `GET` a whole URL into memory.
///
/// Only for files that comfortably fit — `config.json`, `tokenizer.json`,
/// `audiovae.safetensors`. Never for `model.safetensors`.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let ctx = format!("GET {url}");
    let response = send(build_request(url, None)?, &ctx).await?;
    if !response.ok() {
        return Err(Error::Other(format!(
            "{ctx}: HTTP {} {}",
            response.status(),
            response.status_text()
        )));
    }
    response_bytes(response, &ctx).await
}

/// `GET` `len` bytes starting at `offset`, using a `Range` header.
///
/// Verifies the server actually honoured the range: a server that ignores
/// `Range` replies `200` with the entire body, which at 4.37 GB would
/// blow up the heap. That case is reported as a clear error instead.
pub async fn fetch_range(url: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let end = offset + len - 1;
    let ctx = format!("GET {url} bytes={offset}-{end}");
    let response = send(build_request(url, Some((offset, end)))?, &ctx).await?;

    match response.status() {
        206 => {}
        200 => {
            return Err(Error::Other(format!(
                "{ctx}: server ignored the Range header (replied 200, not 206). \
                 Serving a {len}-byte slice of a multi-gigabyte checkpoint needs a \
                 Range-capable static server — see scripts/serve.py"
            )));
        }
        s => {
            return Err(Error::Other(format!(
                "{ctx}: HTTP {s} {}",
                response.status_text()
            )));
        }
    }

    let bytes = response_bytes(response, &ctx).await?;
    if bytes.len() as u64 != len {
        return Err(Error::Other(format!(
            "{ctx}: expected {len} bytes, got {}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

/// Total size of a URL in bytes, from a `Range`-probe's `Content-Range`.
///
/// A `HEAD` would be the obvious way, but plenty of static servers (and
/// some CDNs) answer `HEAD` without `Content-Length`. Asking for one byte
/// and reading `Content-Range: bytes 0-0/TOTAL` also proves up front that
/// the server supports ranges at all, which is the thing we actually need
/// to know before starting a 4 GB streamed load.
pub async fn content_length(url: &str) -> Result<u64> {
    let ctx = format!("probe {url}");
    let response = send(build_request(url, Some((0, 0)))?, &ctx).await?;

    if response.status() != 206 {
        return Err(Error::Other(format!(
            "{ctx}: HTTP {} — expected 206 Partial Content. The server must \
             support HTTP Range requests to stream the checkpoint.",
            response.status()
        )));
    }

    let header = response
        .headers()
        .get("Content-Range")
        .map_err(|e| js_err(&format!("{ctx}: read Content-Range"), e))?
        .ok_or_else(|| {
            Error::Other(format!("{ctx}: 206 response without a Content-Range header"))
        })?;

    // `bytes 0-0/4578476032`
    header
        .rsplit('/')
        .next()
        .and_then(|t| t.trim().parse::<u64>().ok())
        .ok_or_else(|| Error::Other(format!("{ctx}: malformed Content-Range `{header}`")))
}

// ---------------------------------------------------------------------------
// HttpRangeSource
// ---------------------------------------------------------------------------

/// A [`ByteSource`] backed by HTTP `Range` requests — one `fetch` per read.
///
/// This is the implementation the browser needs and the reason
/// [`crate::stream`] is range-oriented at all: `model.safetensors` is
/// 4.37 GB, which does not fit in a `wasm32` address space, so there is no
/// "load the file" operation to implement.
#[derive(Debug, Clone)]
pub struct HttpRangeSource {
    url: String,
    len: u64,
}

impl HttpRangeSource {
    /// Probe `url` for its length and for `Range` support.
    pub async fn open(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        let len = content_length(&url).await?;
        log::info!("HttpRangeSource: {url} ({len} bytes, Range OK)");
        Ok(Self { url, len })
    }

    /// The URL being read.
    pub fn url(&self) -> &str {
        &self.url
    }
}

impl ByteSource for HttpRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    async fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        if offset + len > self.len {
            return Err(Error::Other(format!(
                "{}: read {len} bytes at {offset} is past the end ({})",
                self.url, self.len
            )));
        }
        fetch_range(&self.url, offset, len).await
    }

    fn label(&self) -> &str {
        &self.url
    }
}
