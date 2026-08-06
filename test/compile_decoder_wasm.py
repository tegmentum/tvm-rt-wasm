#!/usr/bin/env python3
"""Compile the Inflect nano VITS decoder for 13.4 correctness gate.

Sibling of `test/compile_encoder_wasm.py` — same flow, but for the
decoder half of the VITS pipeline. Loads cognition's `decoder.onnx`,
drives it through TVM 0.25's Relax ONNX frontend the same way
`crates/inflect-tvm-kernel/scripts/tvm_compile.py` does, and produces:

1. **Native oracle**  — `llvm` target, run in-process via
   `relax.VirtualMachine`. Its output is the ground-truth waveform the
   wasm driver's output is compared against.
2. **wasm32-wasi bundle** — `.tar` at `test/decoder_out/decoder.tar`.
   Same shape as `test/encoder_out/encoder.tar`: `devc.o` + `lib0.o`
   consumed by the fork's system-lib loader. The CMake pipeline
   extracts + repacks it into `libdecoder_gen.a`.

Shape pinning: `frame_count=1024` — matches the encoder gate. Decoder
input is:
  * `z_p`     f32 [1, 128, 1024]  — sampled deterministically from a
                                    fixed-seed RNG so oracle + wasm
                                    driver consume identical bytes.
  * `y_mask`  f32 [1, 1, 1024]    — ones for the first `INPUT_LENGTHS`
                                    frames, zeros after. Matches how
                                    the encoder emits its own y_mask
                                    (length-16 utterance analogue).

Note: unlike the encoder, the decoder ONNX needs no Range-limit
rewrite (the bug is encoder-side only). We load `decoder.onnx`
directly.
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

# Fixed deterministic input. Same seed-namespace as the encoder gate
# but a different seed so drift shows up as drift and doesn't get
# masked by identical PRNG streams.
INPUT_LENGTHS = 16
SEED = 13_004


def build_inputs() -> dict[str, np.ndarray]:
    """Deterministic input tensors — every wasm driver run + oracle
    run consumes the SAME bytes so any divergence is runtime, not input."""
    rng = np.random.default_rng(seed=SEED)
    # z_p ~ N(0,1). VITS' actual z_p is m_p + std * randn but the decoder
    # is a deterministic function of z_p regardless of how it was sampled,
    # so we can seed straight from a standard normal for the gate.
    z_p = rng.standard_normal((1, Z_CHANNELS, FRAME_COUNT), dtype=np.float32)
    # y_mask is 1s over the first INPUT_LENGTHS frames, 0s after — the
    # shape the encoder emits for a short utterance. The decoder uses
    # it to zero out padded region before upsample.
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
    ap.add_argument("--out-dir", default=None, help="Defaults to test/decoder_out/ next to this script.")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"missing decoder ONNX: {onnx_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "decoder_out"
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

    # `main` positional order comes from the frontend — matches the ONNX
    # graph's input order (z_p, y_mask for this graph).
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
    print(f"[native] wrote decoder-io.json")

    # ---------------- wasm system-lib bundle ----------------
    host = {
        "kind": "llvm",
        "mtriple": "wasm32-wasi",
        "mattr": ["+simd128", "+bulk-memory"],
    }
    target_wasm = tvm.target.Target(host, host=host)
    print(f"[wasm] compile target={target_wasm}")
    ex_wasm = relax.build(mod, target_wasm, system_lib=True)

    tar_path = out_dir / "decoder.tar"
    ex_wasm.export_library(str(tar_path))
    print(f"[wasm] wrote {tar_path} ({tar_path.stat().st_size} B)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
