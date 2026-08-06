#!/usr/bin/env python3
"""Toy Relax module for 14.1 WebGPU lowering probe.

Mirrors ``test/toy_relax.py``'s shape but targets ``webgpu`` instead of
``llvm -mtriple=wasm32-wasi``. This is the M14 §4.1 binary go/no-go on
whether TVM 0.25's Relax pipeline lowers cleanly to WGSL for the ops the
CPU toy uses. No runtime execution — the deliverable is the artifact
(if it produces at all) and its contents.

Toy program: ``f(x, w, b) = matmul(x, w) + b`` — same as ``toy_relax.py``
so we can compare envelope shape against the CPU catalog side-by-side.

Outcomes:

- YES — ``tvm.compile(Toy, Target("webgpu"))`` returns; ``export_library``
  writes a ``.tar`` that unpacks to WGSL shader sources plus a devc/lib0
  pair. This means the M14 plan §2.2 unknown is resolved in the
  favourable direction — Relax → WGSL works for at least matmul+add and
  M14 can proceed to 14.2 (WIT design).
- NO — the compile pipeline bails. The exception message is the load-
  bearing artifact; captured to stderr with a traceback for the M14.1
  report.

The compile requires a host target for the driver-side dispatch code
(TVM 0.25's WebGPU codegen splits into host-side stub + device WGSL,
same shape as CUDA/OpenCL). Under wasi-sdk the host would eventually be
``llvm -mtriple=wasm32-wasi``, but 14.1 is a viability probe so we
default to bare ``llvm`` (native host) which matches how
``cognition/crates/inflect-tvm-kernel/scripts/tvm_compile.py`` compiles
today — one axis at a time.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class Toy:
    @R.function
    def main(
        x: R.Tensor((2, 3), "float32"),
        w: R.Tensor((3, 4), "float32"),
        b: R.Tensor((4,), "float32"),
    ) -> R.Tensor((2, 4), "float32"):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.add(y, b)
            R.output(z)
        return z


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "toy_webgpu_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Host = bare llvm for the driver-side dispatch code. WebGPU codegen
    # splits into a host stub (dispatch, arg-marshaling) + device WGSL,
    # so a host target is mandatory. `Target(kind, host=...)` is the
    # 0.25 shape (same convention toy_relax.py uses for the wasm run).
    device = tvm.target.Target("webgpu")
    host = tvm.target.Target("llvm")
    target = tvm.target.Target(device, host=host)
    print(f"[webgpu] target={target}")

    try:
        ex = tvm.compile(Toy, target)
        print(f"[webgpu] compile OK: {type(ex).__name__}")
    except Exception as exc:  # noqa: BLE001
        print(f"[webgpu] compile FAILED: {exc!r}", file=sys.stderr)
        traceback.print_exc(limit=6)
        return 1

    tar_path = out_dir / "toy_webgpu.tar"
    try:
        ex.export_library(str(tar_path))
        print(f"[webgpu] wrote {tar_path} ({tar_path.stat().st_size} B)")
    except Exception as exc:  # noqa: BLE001
        print(f"[webgpu] export_library FAILED: {exc!r}", file=sys.stderr)
        traceback.print_exc(limit=6)
        return 2

    # Peek at envelope contents — the deliverable of 14.1.
    import tarfile

    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
    print(f"[webgpu] tar contents ({len(names)}):")
    for n in names:
        print(f"  {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
