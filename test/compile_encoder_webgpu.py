#!/usr/bin/env python3
"""Compile the Inflect nano VITS encoder for 14.6 WebGPU correctness gate.

Sibling of `test/compile_encoder_wasm.py` — same encoder ONNX, same
shape pinning, same deterministic inputs, same native LLVM oracle. The
divergence is the wasm bundle target: `Target("webgpu", host=wasm32-wasi)`
instead of the plain LLVM wasm host. Output:

  - `test/encoder_webgpu_out/oracle_*.bin` — from the CPU LLVM Relax VM
    (byte-identical to the wasm-target sibling's oracles).
  - `test/encoder_webgpu_out/encoder.tar` — devc.o + lib0.o with the
    WebGPU-host dispatch stubs + WGSL kernel blob the WebGPU module
    loader parses at load time.

Preserves the `system_lib_prefix="enc_"` trick M13.7 established so
future composition can co-link with the decoder.

## 14.6b resolution — unblocked via ONNX graph rewrite

M14.6 stopped at `CodeGenWebGPU: do not support i64` — nine PrimFuncs
downstream of the encoder's int64 `tokens` / `lengths` inputs tripped
TVM 0.25's WGSL codegen. M14.6b resolves it entirely on the ONNX side:
cognition's `crates/inflect-tvm-kernel/scripts/patch_encoder.py`
gained a second pass that flips `tokens` -> UINT32, `lengths` ->
INT32, all int64 initializers/Constants/ConstantOfShape values to
int32 (with a 2**20 sentinel clamp for INT64_MIN/MAX markers used by
ONNX Slice), and Cast(to=INT64) -> Cast(to=INT32). It also reroutes
the sole Cast(FLOAT->INT32) through INT16 to skirt TVM's frontend
wraparound chain that keeps 32-bit int casts on an int64 intermediate,
and prunes no-op Cast(int32->int32) nodes so onnxsim can subsequently
fold Pad's `pads` inputs to initializers.

With that rewrite in place `relax.build(mod, Target("webgpu",
host=wasm32-wasi))` produces a devc.o + lib0.o tar identical in shape
to the CPU-wasm target's output — same `system_lib_prefix="enc_"`
trick M13.7 established.
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

# Cognition's compile defaults (mirrors compile_encoder_wasm.py).
TOKEN_COUNT = 256
FRAME_COUNT = 1024
INPUT_LENGTHS = 16


def build_inputs() -> dict[str, np.ndarray]:
    # M14.6b: encoder.patched.onnx now uses uint32 tokens (unsigned skips
    # TVM's Gather negative-index-correction chain in the ONNX frontend,
    # which otherwise emits int64 intermediates) and int32 lengths. WebGPU
    # WGSL codegen has no i64 support — the rewrite in cognition's
    # crates/inflect-tvm-kernel/scripts/patch_encoder.py flips both inputs.
    # Same seed + values as the CPU-wasm baseline so the oracle bins
    # remain bit-comparable modulo dtype.
    rng = np.random.default_rng(seed=13_003)
    tokens = np.zeros((1, TOKEN_COUNT), dtype=np.uint32)
    tokens[0, :INPUT_LENGTHS] = rng.integers(1, 100, size=INPUT_LENGTHS)
    lengths = np.array([INPUT_LENGTHS], dtype=np.int32)
    length_scale = np.array(1.0, dtype=np.float32)
    return {"tokens": tokens, "lengths": lengths, "length_scale": length_scale}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--onnx",
        default="/Users/zacharywhitley/git/inflect-tts/inference-nodejs/models/nano-tvm/encoder.patched.onnx",
        help="Path to the Range-limit-rewritten encoder ONNX.",
    )
    ap.add_argument("--out-dir", default=None,
                    help="Defaults to test/encoder_webgpu_out/ next to this script.")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"missing encoder ONNX: {onnx_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "encoder_webgpu_out"
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
            "tokens": {"shape": list(inputs["tokens"].shape), "dtype": "uint32"},
            "lengths": {"shape": list(inputs["lengths"].shape), "dtype": "int32"},
            "length_scale": {"shape": list(inputs["length_scale"].shape), "dtype": "float32"},
        },
        "shape_pinning": pins,
    }
    (out_dir / "encoder-io.json").write_text(json.dumps(meta, indent=2))
    print(f"[native] wrote encoder-io.json")

    # ---------------- WebGPU + wasm32-wasi bundle ----------------
    # M13.7 system_lib_prefix trick — patched _auto_attach_system_lib_prefix
    # to skip if attr already present. Same shape as compile_encoder_wasm.py.
    from tvm.relax import vm_build
    _orig = vm_build._auto_attach_system_lib_prefix

    def _patched(tir_mod, target=None, system_lib=None):
        attrs = dict(tir_mod.attrs) if tir_mod.attrs else {}
        if "system_lib_prefix" in attrs:
            return tir_mod
        return _orig(tir_mod, target, system_lib)

    vm_build._auto_attach_system_lib_prefix = _patched

    mod = mod.with_attr("system_lib_prefix", "enc_")

    host = {
        "kind": "llvm",
        "mtriple": "wasm32-wasi",
        "mattr": ["+simd128", "+bulk-memory"],
    }
    target_webgpu = tvm.target.Target("webgpu", host=host)
    print(f"[wasm] compile target={target_webgpu} (system_lib_prefix='enc_')")
    ex_wasm = relax.build(mod, target_webgpu, system_lib=True)

    tar_path = out_dir / "encoder.tar"
    ex_wasm.export_library(str(tar_path))
    print(f"[wasm] wrote {tar_path} ({tar_path.stat().st_size} B)")

    import tarfile
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
    print(f"[wasm] tar contents ({len(names)}):")
    for n in names:
        print(f"  {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
