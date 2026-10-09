#!/usr/bin/env bash
# Build the WASM + WebGPU bundle into web/pkg/.
#
#   scripts/build-web.sh            # F32 weights (8.7 GB VRAM) -> web/pkg
#   scripts/build-web.sh --f16      # F16 weights (4.4 GB VRAM) -> web/pkg-f16
#   scripts/build-web.sh --both     # both, so the page can choose at runtime
#   scripts/build-web.sh --debug    # debug build (much larger, much slower)
#
# Precision is a build-time switch because `B::FloatElem` is a type
# parameter. Shipping both bundles is how the page offers the choice
# without a rebuild: it loads pkg-f16 when the adapter reports
# `shader-f16`, and pkg otherwise.
#
# Needs `wasm-bindgen` whose version matches the `wasm-bindgen` crate in
# Cargo.lock; the script checks and tells you what to install.
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE=release
CARGO_PROFILE_FLAG=--release
TARGETS=("webgpu:web/pkg")

for arg in "$@"; do
  case "$arg" in
    --f16)   TARGETS=("webgpu-f16:web/pkg-f16") ;;
    --both)  TARGETS=("webgpu:web/pkg" "webgpu-f16:web/pkg-f16") ;;
    --debug) PROFILE=debug; CARGO_PROFILE_FLAG= ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# --- version check -----------------------------------------------------------
want=$(awk '
  /^name = "wasm-bindgen"$/ { found=1; next }
  found && /^version = / { gsub(/[":]/,""); print $3; exit }
' Cargo.lock)

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "error: wasm-bindgen not found on PATH." >&2
  echo "       cargo install wasm-bindgen-cli --version $want" >&2
  exit 1
fi
have=$(wasm-bindgen --version | awk '{print $2}')
if [ "$have" != "$want" ]; then
  echo "error: wasm-bindgen CLI is $have but the crate is $want." >&2
  echo "       Mismatched versions produce broken glue." >&2
  echo "       cargo install wasm-bindgen-cli --version $want --force" >&2
  exit 1
fi

WASM="target/wasm32-unknown-unknown/$PROFILE/voxcpm_rs.wasm"

for spec in "${TARGETS[@]}"; do
  FEATURES="${spec%%:*}"
  OUT="${spec##*:}"

  # --- compile ---------------------------------------------------------------
  echo "==> cargo build --target wasm32-unknown-unknown --features $FEATURES ($PROFILE)"
  cargo build --target wasm32-unknown-unknown $CARGO_PROFILE_FLAG \
    --no-default-features --features "$FEATURES"
  [ -f "$WASM" ] || { echo "error: $WASM not produced" >&2; exit 1; }
  echo "    $(du -h "$WASM" | cut -f1) raw wasm"

  # --- bindgen ---------------------------------------------------------------
  echo "==> wasm-bindgen -> $OUT"
  rm -rf "$OUT"
  wasm-bindgen "$WASM" --out-dir "$OUT" --target web --no-typescript

  # --- optional size pass ----------------------------------------------------
  if command -v wasm-opt >/dev/null 2>&1 && [ "$PROFILE" = release ]; then
    echo "==> wasm-opt -O2"
    wasm-opt -O2 --enable-bulk-memory --enable-nontrapping-float-to-int \
      "$OUT/voxcpm_rs_bg.wasm" -o "$OUT/voxcpm_rs_bg.opt.wasm" \
      && mv "$OUT/voxcpm_rs_bg.opt.wasm" "$OUT/voxcpm_rs_bg.wasm"
  else
    echo "==> skipping wasm-opt (not installed; optional)"
  fi

  echo
  echo "built $OUT ($FEATURES, $PROFILE):"
  ls -la "$OUT"
  echo
done

echo "Serve it:"
echo "  python3 scripts/serve.py --model /path/to/VoxCPM2"
echo "  open http://localhost:8080"
