#!/usr/bin/env python3
"""Static file server with HTTP Range support, for the VoxCPM2 WebGPU demo.

Python's `http.server` does not implement `Range`, and this demo cannot work
without it: `model.safetensors` is 4.37 GB and wasm32 has a 4 GB address
space, so the loader fetches the checkpoint one tensor group at a time
(`src/stream.rs`). A server that answers 200-with-the-whole-body instead of
206 makes the browser try to allocate the entire file.

It also sets the COOP/COEP headers, so `SharedArrayBuffer` is available if
an AudioWorklet ring buffer ever wants one, and serves .wasm with the right
MIME type so Chrome can use the streaming compiler.

    # local only (default)
    python3 scripts/serve.py --model /path/to/VoxCPM2

    # reachable from every machine on your tailnet, over real HTTPS
    python3 scripts/serve.py --model /path/to/VoxCPM2 --tailscale

`--model DIR` mounts a checkpoint directory at /models, so the page can
fetch /models/model.safetensors without copying 4.4 GB into web/.

## Why `--tailscale` and not just `--bind 0.0.0.0`

**WebGPU requires a secure context.** `http://localhost` counts as one by
special case, but `http://100.x.y.z:8080` from another machine does not —
`navigator.gpu` is simply `undefined` there, and no flag on this server
can change that. Serving the page over plain HTTP to a peer gets you a
page that loads and then reports no WebGPU.

`--tailscale` solves it properly rather than working around it:

* asks `tailscale cert` for a real Let's Encrypt certificate for this
  node's MagicDNS name (needs HTTPS enabled for the tailnet — it already
  is if `tailscale status --json` lists `CertDomains`),
* serves HTTPS with it, so peers get a genuine secure context with no
  certificate warnings and no browser flags,
* binds **only** this node's Tailscale addresses (v4 and v6), so the
  checkpoint is not also exposed on your LAN or any public interface.

`tailscale serve --bg 8080` is an alternative that gets you port 443 (no
`:8080` in the URL). It is not done automatically here because it changes
persistent tailscaled state; this script only reads a cert.
"""

import argparse
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

EXTRA_TYPES = {
    ".wasm": "application/wasm",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".safetensors": "application/octet-stream",
}


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus single-range `bytes=` support."""

    model_dir = None
    protocol_version = "HTTP/1.1"

    def translate_path(self, path):
        # Mount the checkpoint directory at /models so a multi-gigabyte
        # model does not have to be copied into the web root.
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if self.model_dir and (clean == "/models" or clean.startswith("/models/")):
            rel = clean[len("/models"):].lstrip("/")
            # Resolve and confine to model_dir.
            full = os.path.realpath(os.path.join(self.model_dir, rel))
            root = os.path.realpath(self.model_dir)
            if full == root or full.startswith(root + os.sep):
                return full
            return root  # path traversal attempt -> deny by pointing at the dir
        return super().translate_path(path)

    def guess_type(self, path):
        ext = os.path.splitext(str(path))[1].lower()
        if ext in EXTRA_TYPES:
            return EXTRA_TYPES[ext]
        return super().guess_type(path)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        # Needed if an AudioWorklet ring buffer wants SharedArrayBuffer.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        # The checkpoint is immutable and huge; let the HTTP cache keep it.
        if self.path.startswith("/models/"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        header = self.headers.get("Range")
        if not header:
            return super().do_GET()

        match = RANGE_RE.match(header.strip())
        if not match:
            # Multi-range and other units are not supported; a correct
            # client falls back, and ours only ever sends a single range.
            self.send_error(416, "Only single `bytes=` ranges are supported")
            return None

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().do_GET()
        try:
            size = os.path.getsize(path)
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        with f:
            first, last = match.group(1), match.group(2)
            if first == "":
                # `bytes=-N`: the final N bytes.
                if last == "":
                    self.send_error(416, "Malformed Range")
                    return None
                length = min(int(last), size)
                start = size - length
                end = size - 1
            else:
                start = int(first)
                end = int(last) if last else size - 1
                end = min(end, size - 1)

            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None

            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()

            f.seek(start)
            remaining = length
            chunk = 1024 * 1024
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    return None
                remaining -= len(data)
        return None

    def log_message(self, fmt, *args):
        # One line per request, but skip the range spam from a streamed load.
        msg = fmt % args
        if " 206 " in msg:
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), msg))


def tailscale_self():
    """This node's MagicDNS name and Tailscale addresses, or `None`.

    Returns `(dns_name, [addr, ...], cert_ok)`. `cert_ok` reflects whether
    the tailnet has HTTPS certificates enabled — without it `tailscale
    cert` cannot issue anything and there is no way to get a secure
    context.
    """
    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "status", "--json"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None

    me = data.get("Self") or {}
    dns = (me.get("DNSName") or "").rstrip(".")
    addrs = list(me.get("TailscaleIPs") or [])
    cert_domains = data.get("CertDomains") or []
    if not dns or not addrs:
        return None
    return dns, addrs, dns in cert_domains


def ensure_cert(domain, cert_dir):
    """Fetch (or refresh) a Tailscale-issued cert for `domain`.

    `tailscale cert` is cheap to re-run: tailscaled caches the certificate
    and only talks to Let's Encrypt when it is near expiry. It does not
    need root — issuing is done by the daemon on our behalf.
    """
    os.makedirs(cert_dir, mode=0o700, exist_ok=True)
    crt = os.path.join(cert_dir, f"{domain}.crt")
    key = os.path.join(cert_dir, f"{domain}.key")
    exe = shutil.which("tailscale")
    if not exe:
        sys.exit("tailscale not found on PATH")

    print(f"requesting certificate for {domain} ...")
    out = subprocess.run(
        [exe, "cert", "--cert-file", crt, "--key-file", key, domain],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if out.returncode != 0:
        msg = (out.stderr or out.stdout).strip()
        sys.exit(
            f"`tailscale cert` failed: {msg}\n\n"
            "HTTPS certificates must be enabled for the tailnet — see\n"
            "  https://tailscale.com/kb/1153/enabling-https\n"
            "Without a cert there is no secure context, and WebGPU will not\n"
            "be available to other machines. Alternatively pass your own\n"
            "--tls-cert/--tls-key."
        )
    return crt, key


def make_server(addr, port, handler, ssl_ctx):
    """One bound (optionally TLS-wrapped) server for a single address."""
    family = socket.AF_INET6 if ":" in addr else socket.AF_INET

    class Server(ThreadingHTTPServer):
        address_family = family
        daemon_threads = True
        allow_reuse_address = True

        def server_bind(self):
            # Keep a v6 listener v6-only so it cannot collide with the
            # separate v4 listener on the same port.
            if family == socket.AF_INET6:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            super().server_bind()

    httpd = Server((addr, port), handler)
    if ssl_ctx is not None:
        httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
    return httpd


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--root", default="web", help="directory to serve at /")
    ap.add_argument("--model", default=None, help="checkpoint directory to mount at /models")
    ap.add_argument("--bind", default=None,
                    help="address to bind (default 127.0.0.1; ignored with --tailscale)")
    ap.add_argument("--tailscale", action="store_true",
                    help="serve HTTPS on this node's Tailscale addresses, using a "
                         "tailscale-issued certificate, so other tailnet machines get "
                         "a secure context (required for WebGPU)")
    ap.add_argument("--tls-cert", default=None, help="PEM certificate (implies HTTPS)")
    ap.add_argument("--tls-key", default=None, help="PEM private key")
    ap.add_argument("--cert-dir",
                    default=os.path.join(
                        os.environ.get("XDG_CACHE_HOME",
                                       os.path.expanduser("~/.cache")),
                        "voxcpm-rs", "certs"),
                    help="where --tailscale caches its certificate")
    args = ap.parse_args()

    # Line-buffer stdout so the URL shows up immediately even when this is
    # launched with its output redirected to a file.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    root = os.path.realpath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"--root {root} is not a directory")
    model = os.path.realpath(args.model) if args.model else None
    if model and not os.path.isdir(model):
        sys.exit(f"--model {model} is not a directory")
    if bool(args.tls_cert) != bool(args.tls_key):
        sys.exit("--tls-cert and --tls-key must be given together")

    # --- decide where to listen and whether to use TLS -------------------
    hostname = None
    if args.tailscale:
        info = tailscale_self()
        if info is None:
            sys.exit(
                "could not read Tailscale state. Is tailscaled running and this "
                "node logged in? (`tailscale status`)"
            )
        hostname, addrs, cert_ok = info
        if not cert_ok:
            print(
                f"warning: {hostname} is not in this tailnet's CertDomains — "
                "`tailscale cert` will probably fail.\n"
                "         Enable HTTPS: https://tailscale.com/kb/1153/enabling-https",
                file=sys.stderr,
            )
        if args.tls_cert:
            cert, key = args.tls_cert, args.tls_key
        else:
            cert, key = ensure_cert(hostname, args.cert_dir)
        if args.bind:
            addrs = [args.bind]
    else:
        addrs = [args.bind or "127.0.0.1"]
        cert, key = args.tls_cert, args.tls_key

    ssl_ctx = None
    if cert:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=cert, keyfile=key)
        # HTTP/1.1 only; this server does not speak h2.
        ssl_ctx.set_alpn_protocols(["http/1.1"])

    scheme = "https" if ssl_ctx else "http"
    RangeHandler.model_dir = model

    class Handler(RangeHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

    servers = []
    for addr in addrs:
        try:
            servers.append(make_server(addr, args.port, Handler, ssl_ctx))
        except OSError as e:
            print(f"warning: cannot bind {addr}:{args.port} — {e}", file=sys.stderr)
    if not servers:
        sys.exit("no address could be bound")

    # --- report -----------------------------------------------------------
    print(f"serving {root}")
    for s in servers:
        host, port = s.server_address[0], s.server_address[1]
        shown = f"[{host}]" if ":" in host else host
        print(f"  {scheme}://{shown}:{port}/")
    if hostname:
        print()
        print(f"  open this on any tailnet machine:  {scheme}://{hostname}:{args.port}/")
    if model:
        print(f"\n  /models -> {model}")
        for name in ("config.json", "tokenizer.json", "model.safetensors",
                     "audiovae.safetensors"):
            p = os.path.join(model, name)
            mark = "ok  " if os.path.exists(p) else "MISSING"
            size = f"{os.path.getsize(p)/1048576:.1f} MB" if os.path.exists(p) else ""
            print(f"    [{mark}] {name} {size}")
    print("\nRange requests: enabled (required)")
    if ssl_ctx is None and addrs != ["127.0.0.1"]:
        print(
            "\nWARNING: serving plain HTTP on a non-loopback address. WebGPU needs a\n"
            "         secure context, so `navigator.gpu` will be undefined for any\n"
            "         client that is not on localhost. Use --tailscale (or supply\n"
            "         --tls-cert/--tls-key) to serve HTTPS instead.",
            file=sys.stderr,
        )

    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nbye")
        for s in servers:
            s.shutdown()


if __name__ == "__main__":
    main()
