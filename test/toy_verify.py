#!/usr/bin/env python3
"""Correctness gate for milestone 13.2/13.2b.

Runs the wasm toy driver and diffs its output against the native TVM
oracle produced by `test/toy_relax.py`. Fails non-zero on any diff
outside f32 tolerance.

Usage:
    python3 test/toy_verify.py [--wasmtime PATH] [--wasm PATH] [--data DIR]

Defaults expect the repo layout: `build/toy_relax_test.wasm` and
`test/toy_out/{x,w,b,oracle}.bin`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--wasmtime", default="wasmtime",
                        help="wasmtime executable (default: PATH lookup)")
    parser.add_argument("--wasm", default=str(repo / "build" / "toy_relax_test.wasm"),
                        help="path to the built toy_relax_test.wasm module")
    parser.add_argument("--data", default=str(repo / "test" / "toy_out"),
                        help="dir with x.bin, w.bin, b.bin, oracle.bin")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="f32 absolute tolerance for the diff")
    args = parser.parse_args()

    data = Path(args.data)
    wasm = Path(args.wasm)
    oracle_path = data / "oracle.bin"
    output_path = data / "output.bin"
    if not wasm.exists():
        print(f"[toy_verify] missing wasm: {wasm}", file=sys.stderr)
        return 2
    if not oracle_path.exists():
        print(f"[toy_verify] missing oracle: {oracle_path} — run test/toy_relax.py first",
              file=sys.stderr)
        return 2

    cmd = [args.wasmtime, "run", f"--dir={data}::/data", str(wasm), "/data"]
    print("[toy_verify] $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[toy_verify] wasmtime exited {result.returncode}", file=sys.stderr)
        return result.returncode

    got = np.fromfile(output_path, dtype=np.float32)
    oracle = np.fromfile(oracle_path, dtype=np.float32)
    diff = np.max(np.abs(got - oracle)) if got.shape == oracle.shape else float("inf")
    matches = got.shape == oracle.shape and np.allclose(got, oracle, atol=args.atol)
    print(f"[toy_verify] wasm    = {got}")
    print(f"[toy_verify] oracle  = {oracle}")
    print(f"[toy_verify] max_abs_diff = {diff}")
    print(f"[toy_verify] match (atol={args.atol}): {matches}")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
