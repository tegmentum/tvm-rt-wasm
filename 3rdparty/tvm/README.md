TVM header exposure for the port
================================

The TVM headers under `include/tvm/` are copied from two upstream repos:

- `include/tvm/runtime/` — copied from
  [`apache/tvm`](https://github.com/apache/tvm) tag `v0.25.0`
  (commit `c7ba0735a4f346c67b761e1fde38a68a60be8adb`).
- `include/tvm/support/` — copied from the same `apache/tvm` tag.
- `include/tvm/ffi/` — copied from
  [`apache/tvm-ffi`](https://github.com/apache/tvm-ffi) at commit
  `59da4c0b82af0d499dae34bd89ef010f64d3ff45` (the SHA
  `apache/tvm@v0.25.0` pins its `3rdparty/tvm-ffi` submodule to).

**Path (b) — copy, not submodule**, chosen because:
- `apache/tvm` as a submodule would drag in ~1 GB of Python/C++/CMake we
  do not consume (the fork only needs the C-facing header surface).
- `apache/tvm-ffi` transitively pins its own `dlpack` and `libbacktrace`
  submodules; our build only needs DLPack, which is already a submodule
  at `3rdparty/dlpack/`.
- The fork's pre-existing convention was already "copy TVM headers here";
  path (b) preserves that shape.

DLPack (`3rdparty/dlpack/`, git submodule) is pinned to v1.3
(commit `84d107bf416c6bab9ae68ad285876600d230490d`) — the version
`apache/tvm-ffi@59da4c0` requires.

To refresh headers to a newer TVM tag, re-run the sparse-checkout copy
described in `docs/PORT_TVM_0_25_ERROR_CATALOG.md`.

Prior version: v0.12.0 (2023-07-14).
