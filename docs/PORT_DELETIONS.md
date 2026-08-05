# Port Deletions

This fork of `yanghaku/tvm-rt-wasm` has been narrowed to a single
purpose: driving a CPU Relax VM inside a WASI Preview 1 wasm
component for the cognition project. Everything that does not serve
that purpose has been deleted so the ported surface — and, going
forward, the TVM-ABI-drift maintenance treadmill — stays small.

Full context and rationale live in the cognition milestone plan under
`docs/milestone-13-plan.md` §3.7 ("Deferred wholesale"), which this
document mirrors. Anything called out below can be re-introduced in a
future milestone by cherry-picking from upstream
`yanghaku/tvm-rt-wasm` at commit `5f65881` (2023-08-03) — the
snapshot this fork started from.

## Removed backends

| Path | Reason |
| --- | --- |
| `src/backends/graph/` | Legacy Graph Executor. Cognition emits Relax VM executables from `tvm_compile.py`; Graph Executor format is not produced by that pipeline. |
| `src/backends/relay_vm/` | Pre-Relax IR. Superseded by Relax VM upstream in TVM Unity (~v0.14). |
| `src/backends/aot/` | AoT executor is not on the cognition roadmap. Relax VM covers M13's shape. |
| `include/graph_executor.h`, `include/aot_executor.h`, `include/relay_vm.h` | Public headers for the dropped backends. |

Kept: `src/backends/relax_vm/` (six files — the VM cognition targets).

## Removed accelerators

| Path | Reason |
| --- | --- |
| `src/accelerators/cuda/` | CPU-only port. CUDA is out of the browser story. |
| `src/accelerators/webgpu/` | Deferred pending a compute-WIT design (cognition `docs/architecture.md` Part II §6/§7/§20). |
| `3rdparty/cuda/` (vendored CUDA header snapshot) | Not needed once CUDA is dropped. |
| `3rdparty/webgpu-headers/` (submodule) | Not needed once WebGPU is dropped. Removed from `.gitmodules`. |

Preserved as a reference: `docs/reference-patterns/cuda_driver_stub.c`
— a copy of the upstream `src/accelerators/cuda/wasi_sdk/cuda_driver_stub.c`
kept because it documents the "compile CUDA-referencing code under
wasi-sdk by stubbing the driver" pattern. A future WebGPU or CUDA
milestone that adds accelerator code back should follow this shape.

## Removed adjacent code

| Path | Reason |
| --- | --- |
| `js/` | The cognition consumer drives the runtime via wasm-component imports, not tvm-rt-wasm's TypeScript API. |
| `examples/` | Documentation-only. If a minimal cognition-side example helps future contributors it will be re-authored to match the trimmed surface. |

## Kept

- `src/core/` — runtime primitives, module loader, device API. Required substrate.
- `src/backends/relax_vm/` — the VM. Highest expected drift point vs. TVM 0.25 per the M13 plan.
- `include/relax_vm.h` — public header for the kept backend.
- `3rdparty/tvm/include/` — vendored TVM v0.12 headers, kept for now.
  These are the drift target the port has to reconcile with TVM 0.25's
  headers; expect this directory to be either replaced or updated as
  patches land.
- `3rdparty/dlpack/` — DLPack v0.7 submodule. TVM 0.25 uses DLPack
  ~v1.0; expect this to be bumped once ABI-drift patches force it.
- `.clang-format`, `CMakeLists.txt`, `LICENSE`, `README.md` — unchanged
  (CMakeLists.txt will be pruned to match the kept targets in a
  follow-up commit).
