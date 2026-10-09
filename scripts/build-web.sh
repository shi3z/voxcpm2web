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

# The smoke test runs the bundle under Node. Node is not needed to *build*,
# so its absence is a loud warning rather than a hard error.
if command -v node >/dev/null 2>&1; then CAN_CHECK=true; else CAN_CHECK=false; fi

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
  #
  # wasm-opt is optional and *verified*, not trusted. Binaryen versions
  # exist that reorder this module's two tables and rebind the
  # `__wbindgen_externrefs` export to the funcref table, which is declared
  # min == max and so cannot grow. The result validates, disassembles
  # identically, and then fails on every page load with
  # "WebAssembly.Table.grow(): failed to grow table by 4". So the
  # optimized output is smoke-tested, and discarded if it regressed.
  if command -v wasm-opt >/dev/null 2>&1 && [ "$PROFILE" = release ]; then
    echo "==> wasm-opt -O2 ($(wasm-opt --version))"
    if wasm-opt -O2 --enable-bulk-memory --enable-nontrapping-float-to-int \
         --enable-reference-types \
         "$OUT/voxcpm_rs_bg.wasm" -o "$OUT/voxcpm_rs_bg.opt.wasm"; then
      if ! $CAN_CHECK; then
        echo "!!! node not found, so the optimized module cannot be verified." >&2
        echo "!!! Keeping the unoptimized one, which is slower to load but known good." >&2
        rm -f "$OUT/voxcpm_rs_bg.opt.wasm"
      elif node scripts/check-wasm.mjs "$OUT/voxcpm_rs_bg.opt.wasm"; then
        mv "$OUT/voxcpm_rs_bg.opt.wasm" "$OUT/voxcpm_rs_bg.wasm"
      else
        echo "!!! wasm-opt produced a broken module — keeping the unoptimized one." >&2
        echo "!!! Upgrade binaryen (>= 123 is known good)." >&2
        rm -f "$OUT/voxcpm_rs_bg.opt.wasm"
      fi
    else
      echo "!!! wasm-opt failed — keeping the unoptimized module." >&2
      rm -f "$OUT/voxcpm_rs_bg.opt.wasm"
    fi
  else
    echo "==> skipping wasm-opt (not installed; optional)"
  fi

  # --- verify what we are about to ship ---------------------------------------
  if $CAN_CHECK; then
    echo "==> smoke test"
    node scripts/check-wasm.mjs "$OUT/voxcpm_rs_bg.wasm" || {
      echo "error: $OUT will not start in a browser; refusing to continue" >&2
      exit 1
    }
  else
    echo "==> NOT smoke-tested: node is not installed." >&2
    echo "    The bundle may compile and still fail on every page load —" >&2
    echo "    that has happened. Install Node to enable scripts/check-wasm.mjs." >&2
  fi

  echo
  echo "built $OUT ($FEATURES, $PROFILE):"
  ls -la "$OUT"
  echo
done

echo "Serve it:"
echo "  python3 scripts/serve.py --model /path/to/VoxCPM2"
echo "  open http://localhost:8080"
