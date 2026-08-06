#!/usr/bin/env python3
"""Compile the Inflect nano VITS encoder for 13.3 correctness gate.

Loads cognition's `encoder.patched.onnx`, drives it through TVM 0.25's
Relax ONNX frontend the same way `crates/inflect-tvm-kernel/scripts/
tvm_compile.py` does — but produces two compiled artifacts:

1. **Native oracle**  — `llvm` target, run in-process via
   `relax.VirtualMachine`. Its outputs are the ground-truth tensors
   the wasm driver's outputs are compared against.
2. **wasm32-wasi bundle** — `.tar` at `test/encoder_out/encoder.tar`.
   Same shape as the toy program's `test/toy_out/toy_relax.tar`:
   contains `devc.o` (blob-registrar constructor) + `lib0.o` (TIR
   kernels + weak `__tvm_ffi__library_ctx`). The CMake pipeline
   extracts + repacks it into `libencoder_gen.a`.

Shape pinning: `token_count=256`, `frame_count=1024` — the cognition-
side compile defaults. Fixed input: token IDs `[42, 55, 17, 3, 91,
...]` deterministic (seeded RNG), `lengths=[16]`, `length_scale=1.0`.
Same input feeds native oracle and (via bin files) the wasm driver.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxsim
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

# Cognition's compile defaults (crates/inflect-tvm-kernel/scripts/tvm_compile.py).
TOKEN_COUNT = 256
FRAME_COUNT = 1024

# Fixed deterministic input. `lengths=16` means the encoder computes
# up to `frame_count` internally but only the first ~16 tokens carry
# signal — matches how cognition would invoke it for a short utterance.
INPUT_LENGTHS = 16


def build_inputs() -> dict[str, np.ndarray]:
    """Deterministic input tensors — every wasm driver run + oracle
    run consumes the SAME bytes so any divergence is runtime, not input."""
    rng = np.random.default_rng(seed=13_003)
    tokens = np.zeros((1, TOKEN_COUNT), dtype=np.int64)
    # Draw ids in [1, 100) — nano's phoneme vocabulary comfortably covers it.
    tokens[0, :INPUT_LENGTHS] = rng.integers(1, 100, size=INPUT_LENGTHS)
    lengths = np.array([INPUT_LENGTHS], dtype=np.int64)
    length_scale = np.array(1.0, dtype=np.float32)  # rank-0 scalar
    return {"tokens": tokens, "lengths": lengths, "length_scale": length_scale}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--onnx",
        default="/Users/zacharywhitley/git/inflect-tts/inference-nodejs/models/nano-tvm/encoder.patched.onnx",
        help="Path to the Range-limit-rewritten encoder ONNX (produced by cognition's patch_encoder.py).",
    )
    ap.add_argument("--out-dir", default=None, help="Defaults to test/encoder_out/ next to this script.")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"missing encoder ONNX: {onnx_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "encoder_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load + pin shapes ----------------
    print(f"[encoder] loading {onnx_path.name}")
    model = onnx.load(str(onnx_path))
    pins = {"tokens": [1, TOKEN_COUNT], "lengths": [1], "length_scale": []}
    print(f"[encoder] onnxsim pin shapes {pins}")
    model, ok = onnxsim.simplify(model, overwrite_input_shapes=pins)
    if not ok:
        print("[encoder] onnxsim check failed", file=sys.stderr)
        return 3
    print(f"[encoder] simplified nodes: {len(model.graph.node)}")

    output_names = [o.name for o in model.graph.output]
    print(f"[encoder] outputs: {output_names}")

    print("[encoder] onnx -> relax")
    mod = from_onnx(model, keep_params_in_input=False)

    # ---------------- Native oracle ----------------
    inputs = build_inputs()
    print("[native] compile llvm")
    ex_native = tvm.compile(mod, tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex_native, tvm.runtime.cpu())

    # `main` positional order comes from the frontend — matches the ONNX
    # graph's input order (tokens, lengths, length_scale for this graph).
    positional = [
        tvm.runtime.tensor(inputs["tokens"]),
        tvm.runtime.tensor(inputs["lengths"]),
        tvm.runtime.tensor(inputs["length_scale"]),
    ]
    print("[native] vm.main(...)")
    result = vm["main"](*positional)
    try:
        oracle_arrs = [r.numpy() for r in result]
    except TypeError:
        oracle_arrs = [result.numpy()]
    if len(oracle_arrs) != len(output_names):
        print(
            f"[native] output count mismatch: {len(oracle_arrs)} vs {output_names}",
            file=sys.stderr,
        )
        return 4

    # ---------------- Write inputs + oracle ----------------
    inputs["tokens"].tofile(out_dir / "tokens.bin")
    inputs["lengths"].tofile(out_dir / "lengths.bin")
    inputs["length_scale"].tofile(out_dir / "length_scale.bin")

    oracle_meta = {}
    for name, arr in zip(output_names, oracle_arrs):
        arr = np.ascontiguousarray(arr)
        arr.tofile(out_dir / f"oracle_{name}.bin")
        oracle_meta[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        print(f"[native] oracle {name} shape={arr.shape} dtype={arr.dtype}")

    meta = {
        "output_order": output_names,
        "outputs": oracle_meta,
        "inputs": {
            "tokens": {"shape": list(inputs["tokens"].shape), "dtype": "int64"},
            "lengths": {"shape": list(inputs["lengths"].shape), "dtype": "int64"},
            "length_scale": {"shape": list(inputs["length_scale"].shape), "dtype": "float32"},
        },
        "shape_pinning": pins,
    }
    (out_dir / "encoder-io.json").write_text(json.dumps(meta, indent=2))
    print(f"[native] wrote encoder-io.json")

    # ---------------- wasm system-lib bundle ----------------
    host = {
        "kind": "llvm",
        "mtriple": "wasm32-wasi",
        "mattr": ["+simd128", "+bulk-memory"],
    }
    target_wasm = tvm.target.Target(host, host=host)
    print(f"[wasm] compile target={target_wasm}")
    ex_wasm = relax.build(mod, target_wasm, system_lib=True)

    tar_path = out_dir / "encoder.tar"
    ex_wasm.export_library(str(tar_path))
    print(f"[wasm] wrote {tar_path} ({tar_path.stat().st_size} B)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
