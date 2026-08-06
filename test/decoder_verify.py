#!/usr/bin/env python3
"""Correctness gate for milestone 13.4 — VITS decoder.

Runs the wasm decoder driver and diffs its float32 waveform output
against the native TVM oracle produced by
`test/compile_decoder_wasm.py`. Fails non-zero if the per-tensor
`max_abs_diff` exceeds the tolerance (default 1e-4).

Usage:
    python3 test/decoder_verify.py [--wasmtime PATH] [--wasm PATH] [--data DIR]

Defaults expect the repo layout: `build/decoder_test.wasm` and
`test/decoder_out/{z_p,y_mask,oracle_waveform}.bin`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

OUTPUTS = ("waveform",)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--wasmtime", default="wasmtime",
                        help="wasmtime executable (default: PATH lookup)")
    parser.add_argument("--wasm", default=str(repo / "build" / "decoder_test.wasm"),
                        help="path to the built decoder_test.wasm module")
    parser.add_argument("--data", default=str(repo / "test" / "decoder_out"),
                        help="dir with z_p.bin, y_mask.bin, oracle_waveform.bin")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="f32 absolute tolerance per-tensor")
    args = parser.parse_args()

    data = Path(args.data)
    wasm = Path(args.wasm)
    if not wasm.exists():
        print(f"[decoder_verify] missing wasm: {wasm}", file=sys.stderr)
        return 2
    for name in OUTPUTS:
        if not (data / f"oracle_{name}.bin").exists():
            print(f"[decoder_verify] missing oracle: oracle_{name}.bin — run "
                  "test/compile_decoder_wasm.py first",
                  file=sys.stderr)
            return 2

    cmd = [args.wasmtime, "run", f"--dir={data}::/data", str(wasm), "/data"]
    print("[decoder_verify] $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[decoder_verify] wasmtime exited {result.returncode}",
              file=sys.stderr)
        return result.returncode

    all_pass = True
    for name in OUTPUTS:
        got = np.fromfile(data / f"wasm_{name}.bin", dtype=np.float32)
        oracle = np.fromfile(data / f"oracle_{name}.bin", dtype=np.float32)
        if got.shape != oracle.shape:
            print(f"[decoder_verify] {name}: SHAPE MISMATCH "
                  f"got={got.shape} oracle={oracle.shape}",
                  file=sys.stderr)
            all_pass = False
            continue
        diff = float(np.max(np.abs(got - oracle)))
        ok = diff <= args.atol
        marker = "PASS" if ok else "FAIL"
        print(f"[decoder_verify] {name}: max_abs_diff = {diff:.6e} "
              f"(atol={args.atol}) {marker}")
        if not ok:
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
