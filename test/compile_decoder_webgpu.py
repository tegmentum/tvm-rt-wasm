#!/usr/bin/env python3
"""Compile the Inflect nano VITS decoder for 14.7 WebGPU correctness gate.

Sibling of `test/compile_decoder_wasm.py` — same decoder ONNX, same
shape pinning, same deterministic inputs, same native LLVM oracle. The
divergence is the wasm bundle target: `Target("webgpu", host=wasm32-wasi)`
instead of the plain LLVM wasm host. Output:

  - `test/decoder_webgpu_out/oracle_*.bin` — from the CPU LLVM Relax VM
    (byte-identical to the wasm-target sibling's oracles, and to the
    M13.4 CPU-wasm decoder gate at `test/decoder_out/oracle_waveform.bin`
    since inputs + seed are identical).
  - `test/decoder_webgpu_out/decoder.tar` — devc.o + lib0.o with the
    WebGPU-host dispatch stubs + WGSL kernel blob the WebGPU module
    loader parses at load time.

Preserves the `system_lib_prefix="dec_"` trick M13.7 established so
future composition can co-link with the encoder.

## 14.7 hypothesis

Unlike 14.6, the decoder's inputs (`z_p`, `y_mask`) are both float32.
The i64-in-WebGPU-codegen block that stopped 14.6 requires i64 buffer
dtypes to propagate into a PrimFunc; with only float32 inputs the
decoder's TIR should stay in fp32/i32 territory. ConvTranspose is the
plan's flagged risk (§4 14.7): WGSL has no first-class transposed-conv
semantics; historically TVM's WGSL codegen for ConvTranspose lowers to
reshape+matmul+reshape. Correctness fine expected; perf deferred.
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
FRAME_COUNT = 1024
Z_CHANNELS = 128

# Fixed deterministic input. Same seed as compile_decoder_wasm.py so
# the oracle produced here is byte-identical to M13.4's oracle at
# `test/decoder_out/oracle_waveform.bin`.
INPUT_LENGTHS = 16
SEED = 13_004


def build_inputs() -> dict[str, np.ndarray]:
    """Deterministic input tensors — identical to compile_decoder_wasm.py."""
    rng = np.random.default_rng(seed=SEED)
    z_p = rng.standard_normal((1, Z_CHANNELS, FRAME_COUNT), dtype=np.float32)
    y_mask = np.zeros((1, 1, FRAME_COUNT), dtype=np.float32)
    y_mask[:, :, :INPUT_LENGTHS] = 1.0
    return {"z_p": z_p, "y_mask": y_mask}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--onnx",
        default="/Users/zacharywhitley/git/inflect-tts/inference-nodejs/models/nano/decoder.onnx",
        help="Path to the nano decoder ONNX.",
    )
    ap.add_argument("--out-dir", default=None,
                    help="Defaults to test/decoder_webgpu_out/ next to this script.")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"missing decoder ONNX: {onnx_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "decoder_webgpu_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load + pin shapes ----------------
    print(f"[decoder] loading {onnx_path.name}")
    model = onnx.load(str(onnx_path))
    pins = {"z_p": [1, Z_CHANNELS, FRAME_COUNT], "y_mask": [1, 1, FRAME_COUNT]}
    print(f"[decoder] onnxsim pin shapes {pins}")
    model, ok = onnxsim.simplify(model, overwrite_input_shapes=pins)
    if not ok:
        print("[decoder] onnxsim check failed", file=sys.stderr)
        return 3
    print(f"[decoder] simplified nodes: {len(model.graph.node)}")

    output_names = [o.name for o in model.graph.output]
    print(f"[decoder] outputs: {output_names}")

    print("[decoder] onnx -> relax")
    mod = from_onnx(model, keep_params_in_input=False)

    # ---------------- Native oracle ----------------
    inputs = build_inputs()
    print("[native] compile llvm")
    ex_native = tvm.compile(mod, tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex_native, tvm.runtime.cpu())

    positional = [
        tvm.runtime.tensor(inputs["z_p"]),
        tvm.runtime.tensor(inputs["y_mask"]),
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
    inputs["z_p"].tofile(out_dir / "z_p.bin")
    inputs["y_mask"].tofile(out_dir / "y_mask.bin")

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
            "z_p": {"shape": list(inputs["z_p"].shape), "dtype": "float32"},
            "y_mask": {"shape": list(inputs["y_mask"].shape), "dtype": "float32"},
        },
        "shape_pinning": pins,
        "seed": SEED,
        "input_lengths": INPUT_LENGTHS,
    }
    (out_dir / "decoder-io.json").write_text(json.dumps(meta, indent=2))
    print("[native] wrote decoder-io.json")

    # ---------------- WebGPU + wasm32-wasi bundle ----------------
    # M13.7 system_lib_prefix trick — patched _auto_attach_system_lib_prefix
    # to skip if attr already present. Same shape as compile_decoder_wasm.py
    # and compile_encoder_webgpu.py.
    from tvm.relax import vm_build
    _orig = vm_build._auto_attach_system_lib_prefix

    def _patched(tir_mod, target=None, system_lib=None):
        attrs = dict(tir_mod.attrs) if tir_mod.attrs else {}
        if "system_lib_prefix" in attrs:
            return tir_mod
        return _orig(tir_mod, target, system_lib)

    vm_build._auto_attach_system_lib_prefix = _patched

    mod = mod.with_attr("system_lib_prefix", "dec_")

    host = {
        "kind": "llvm",
        "mtriple": "wasm32-wasi",
        "mattr": ["+simd128", "+bulk-memory"],
    }
    target_webgpu = tvm.target.Target("webgpu", host=host)
    print(f"[wasm] compile target={target_webgpu} (system_lib_prefix='dec_')")
    ex_wasm = relax.build(mod, target_webgpu, system_lib=True)

    tar_path = out_dir / "decoder.tar"
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
