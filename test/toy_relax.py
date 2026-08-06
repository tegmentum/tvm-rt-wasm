#!/usr/bin/env python3
"""Toy Relax module for 13.2 correctness gate.

Compiles a trivial `main(x, w, b) = matmul(x, w) + b` Relax function twice:

1. Native LLVM (`tvm.target.Target("llvm")`) — run via Python
   `tvm.relax.VirtualMachine` to produce the oracle output tensor.
2. wasm32-wasi system-lib (`llvm -mtriple=wasm32-wasi -mattr=+simd128,+bulk-memory --system-lib=1`)
   — export_library() writes a `.tar` bundle. The tar contains the
   compiled `.o` (TIR kernels linked as ELF for the wasm target) and
   `devc.o` (device blob wrapper) that the driver will `ar` into a
   static lib and link with the runtime.

Straight sequential Relax: matmul + add. No closures, no dynamic-shape
TIR — dodges the fork's known-stub built-ins
(`MakeClosure`/`InvokeClosure`/`CallTIRDyn`).
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
    out_dir = Path(__file__).resolve().parent / "toy_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed inputs for oracle + wasm driver.
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

    # Serialize inputs and oracle output for the wasm driver.
    x.tofile(out_dir / "x.bin")
    w.tofile(out_dir / "w.bin")
    b.tofile(out_dir / "b.bin")
    oracle.tofile(out_dir / "oracle.bin")
    np.save(out_dir / "oracle.npy", oracle)
    print(f"[native] wrote inputs + oracle under {out_dir}")

    # ------------------------- wasm system-lib -----------------------
    # TVM 0.25 dropped the CLI target-string form; use JSON dict.
    # `system-lib=1` makes the code generator emit
    # `TVMBackendRegisterSystemLibSymbol` constructors — the fork's
    # system_library.c consumes those, which is the only module-load
    # path available under wasi-sdk (dlopen is gated off).
    host = {
        "kind": "llvm",
        "mtriple": "wasm32-wasi",
        "mattr": ["+simd128", "+bulk-memory"],
    }
    target_wasm = tvm.target.Target(host, host=host)
    print(f"[wasm] compile target={target_wasm}")
    # `system_lib=True` makes relax.build emit
    # `TVMFFIEnvModRegisterSystemLibSymbol` constructors so the
    # fork's system_library.c can pick up the compiled TIR kernels
    # under wasi-sdk (dlopen is gated off there — see
    # src/core/module/shared_library.c:USE_WASI_SDK).
    ex_wasm = relax.build(Toy, target_wasm, system_lib=True)

    tar_path = out_dir / "toy_relax.tar"
    ex_wasm.export_library(str(tar_path))
    print(f"[wasm] wrote {tar_path} ({tar_path.stat().st_size} B)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
