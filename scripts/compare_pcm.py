#!/usr/bin/env python3
"""Compare two mono PCM waveforms — browser output against a native run.

Accepts 16-bit WAV or raw little-endian f32 (`.f32`).

    python3 scripts/compare_pcm.py native.wav.f32 browser.f32

Browser and native output are *not* expected to be bit-identical: the
flow-matching sampler draws Gaussian noise, and WGSL reduction order
differs between implementations. There are two useful modes:

* **Deterministic** — set `VOXCPM_Z_ZERO` on both sides (the UI checkbox,
  or the env var natively). The noise is zeroed, so the waveforms should
  match closely; correlation near 1.0 means the whole pipeline agrees.
  Note that zeroing the noise degrades the audio, so judge *agreement*
  here, not quality.
* **Stochastic** — leave the noise on. Waveforms will differ sample by
  sample, so compare the summary statistics instead: duration, peak, RMS,
  voiced fraction and median F0 should all land in the same region.
"""

import array
import math
import sys
import wave


def load(path):
    if path.endswith(".f32"):
        a = array.array("f")
        with open(path, "rb") as f:
            a.frombytes(f.read())
        return list(a), None
    w = wave.open(path)
    if w.getsampwidth() != 2:
        sys.exit(f"{path}: only 16-bit WAV is supported")
    a = array.array("h")
    a.frombytes(w.readframes(w.getnframes()))
    ch = w.getnchannels()
    x = [v / 32768.0 for v in a]
    if ch > 1:
        x = [sum(x[i : i + ch]) / ch for i in range(0, len(x), ch)]
    return x, w.getframerate()


def stats(x):
    n = len(x)
    if n == 0:
        return dict(n=0, peak=0.0, rms=0.0, dc=0.0)
    return dict(
        n=n,
        peak=max(abs(v) for v in x),
        rms=math.sqrt(sum(v * v for v in x) / n),
        dc=sum(x) / n,
    )


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ap, bp = sys.argv[1], sys.argv[2]
    a, sra = load(ap)
    b, srb = load(bp)

    sa, sb = stats(a), stats(b)
    print(f"A {ap}")
    print(f"  samples={sa['n']} peak={sa['peak']:.5f} rms={sa['rms']:.6f} dc={sa['dc']:+.6f}")
    print(f"B {bp}")
    print(f"  samples={sb['n']} peak={sb['peak']:.5f} rms={sb['rms']:.6f} dc={sb['dc']:+.6f}")
    if sra and srb and sra != srb:
        print(f"  NOTE: sample rates differ ({sra} vs {srb})")

    if sa["n"] != sb["n"]:
        print(f"\nlength differs by {abs(sa['n'] - sb['n'])} samples "
              f"({abs(sa['n'] - sb['n']) / max(1, max(sa['n'], sb['n'])):.1%}) "
              "— comparing the overlapping prefix")
    n = min(sa["n"], sb["n"])
    if n == 0:
        sys.exit("nothing to compare")
    a, b = a[:n], b[:n]

    # Pearson correlation, plus error relative to A's scale.
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")

    diff = [a[i] - b[i] for i in range(n)]
    max_abs = max(abs(d) for d in diff)
    rms_err = math.sqrt(sum(d * d for d in diff) / n)
    denom = max(sa["rms"], 1e-12)

    print(f"\ncorrelation      : {corr:.6f}")
    print(f"max |A-B|        : {max_abs:.6f}  ({max_abs / max(sa['peak'], 1e-12):.2%} of A's peak)")
    print(f"rms(A-B)         : {rms_err:.6f}  ({rms_err / denom:.2%} of A's rms)")

    if corr > 0.999 and rms_err / denom < 0.05:
        print("\nverdict: MATCH — same pipeline, differences at numerical-noise level")
    elif corr > 0.95:
        print("\nverdict: CLOSE — same structure, some numerical divergence")
    elif corr > 0.3:
        print("\nverdict: RELATED — correlated but materially different")
    else:
        print("\nverdict: DIFFERENT — uncorrelated. Expected if the sampler's noise "
              "was left on (compare the summary stats instead); a real problem if "
              "VOXCPM_Z_ZERO was set on both sides.")


if __name__ == "__main__":
    main()
