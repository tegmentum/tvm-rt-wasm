#!/usr/bin/env python3
"""Toy Relax module for 14.5 WebGPU end-to-end correctness gate.

Compiles ``f(x, w, b) = matmul(x, w) + b`` twice:

1. **Native LLVM** — run in-process through ``tvm.relax.VirtualMachine``
   to produce ``oracle.bin``. Same numeric-reference discipline
   ``test/toy_relax.py`` uses for the CPU-wasm path.
2. **WebGPU device + wasm32-wasi host** — writes ``toy_webgpu.tar``
   containing:
      - ``lib0.o`` — wasm32-wasi ELF object with the host-side stub
        (compute-pipeline dispatch scaffolding, arg marshaling into
        WGPU_MemoryCopyHtoD/DtoH, WGPU_FunctionRun) plus the sys-lib
        constructor blob TVM 0.25 emits under ``system_lib=True``.
      - ``devc.o`` — the second wasm32-wasi object; also carries the
        blob header the sys-lib loader consumes on ctor.

Both objects are linked into ``build/toy_webgpu_test.wasm`` alongside
libtvm-rt-{core,backend-relax-vm,accelerator-webgpu}.a — same recipe
``test/toy_relax.c`` uses for the CPU sibling.

Inputs + oracle are serialized into ``toy_webgpu_out/`` for the
downstream Node driver (cognition's ``web/test/toy-webgpu.test.mjs``)
to read via WASI preopen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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

    # Fixed inputs — same seeds as toy_relax.py so the oracle bytes are
    # comparable between the CPU and WebGPU paths (invariant that the
    # matmul+add op sequence is deterministic on both).
    x = np.arange(6, dtype="float32").reshape(2, 3)
    w = np.arange(12, dtype="float32").reshape(3, 4) * 0.1
    b = np.arange(4, dtype="float32") - 1.5

    # ------------------------- Native oracle -------------------------
    print("[native] compile llvm")
    ex_native = tvm.compile(Toy, tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex_native, tvm.runtime.cpu())
    out = vm["main"](
        tvm.runtime.tensor(x), tvm.runtime.tensor(w), tvm.runtime.tensor(b)
    )
    oracle = out.numpy()
    print("[native] oracle output:")
    print(oracle)

    x.tofile(out_dir / "x.bin")
    w.tofile(out_dir / "w.bin")
    b.tofile(out_dir / "b.bin")
    oracle.tofile(out_dir / "oracle.bin")
    np.save(out_dir / "oracle.npy", oracle)
    print(f"[native] wrote inputs + oracle under {out_dir}")

    # ------------------------- WebGPU + wasm32-wasi ------------------
    # Host = wasm32-wasi (matches the CPU port's build). Device = webgpu.
    # TVM 0.25 needs `system_lib=True` so the emitted constructors call
    # TVMFFIEnvModRegisterSystemLibSymbol — the fork's system_library.c
    # picks these up under wasi-sdk where dlopen is gated off.
    host = {
        "kind": "llvm",
        "mtriple": "wasm32-wasi",
        "mattr": ["+simd128", "+bulk-memory"],
    }
    target_wasm = tvm.target.Target(host, host=host)
    target_webgpu = tvm.target.Target("webgpu", host=host)
    print(f"[webgpu] target={target_webgpu}")

    ex_wgpu = relax.build(Toy, target_webgpu, system_lib=True)
    tar_path = out_dir / "toy_webgpu.tar"
    ex_wgpu.export_library(str(tar_path))
    print(f"[webgpu] wrote {tar_path} ({tar_path.stat().st_size} B)")

    import tarfile

    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
    print(f"[webgpu] tar contents ({len(names)}):")
    for n in names:
        print(f"  {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
