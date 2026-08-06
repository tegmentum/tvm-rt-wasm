# TVM 0.25 compile-error catalog

Deliverable of cognition sub-task **13.1b**. This document enumerates
the compile-time breakage that surfaces when the fork's C sources are
built against TVM 0.25 headers (via `apache/tvm@v0.25.0` +
`apache/tvm-ffi@59da4c0b`). It exists to scope the follow-on ABI-bridge
work for sub-task **13.1c**; it is **not** a fix.

Toolchain: wasi-sdk 33 (`~/wasi-sdk-33.0-arm64-macos`, clang 22.1.0),
wasm32-wasip1, CMake 4.4.1, `-Wall -Wextra -Werror` (fork default).

---

## 0. Header exposure path chosen — path (b) copy

Rationale for copy-in over submodule (13.1b considered both):

- `apache/tvm@v0.25.0` as a full submodule drags in ~1 GB of
  Python/C++/CMake we do not consume. The fork only needs the C-facing
  header surface (`include/tvm/runtime/*.h`,
  `include/tvm/support/{io,serializer}.h`). Cost/benefit is poor —
  we are neither building nor linking the upstream tree.
- `apache/tvm-ffi@59da4c0b` transitively pins its own
  `3rdparty/{dlpack,libbacktrace}` submodules; a submodule of a
  submodule of a submodule is ergonomically painful on fresh clones
  (`git submodule update --init --recursive` becomes load-bearing for
  every wrangler build).
- The fork's pre-existing convention was already **copy** — see
  `3rdparty/tvm/README.md` at commit `d660306` ("The files are copied
  from Apache TVM. Now version: v0.12.0"). Path (b) preserves that
  shape verbatim.

**What landed in `3rdparty/tvm/include/tvm/`:**

| Origin | Path | Files |
| --- | --- | --- |
| `apache/tvm@v0.25.0:include/tvm/runtime/` | `tvm/runtime/` | 17 headers (`base.h`, `c_backend_api.h`, `tensor.h`, `data_type.h`, `device_api.h`, `logging.h`, `timer.h`, `disco/*.h`, `memory/memory_manager.h`, `vm/{builtin,bytecode,executable,tensor_cache_support,vm}.h`) |
| `apache/tvm@v0.25.0:include/tvm/support/` | `tvm/support/` | `io.h`, `serializer.h`, `cuda/` |
| `apache/tvm-ffi@59da4c0b:include/tvm/ffi/` | `tvm/ffi/` | 52 headers (`c_api.h`, `any.h`, `object.h`, `function.h`, `container/*`, `extra/*`, `reflection/*`, …) |

**DLPack (`3rdparty/dlpack/`, submodule):** bumped from
`e2bdd3b` (v0.7) → `84d107b` (v1.3), matching the SHA `apache/tvm-ffi`
pins. `DLPACK_MAJOR_VERSION 1, DLPACK_MINOR_VERSION 3`.

**Fresh-checkout smoke tests** confirm the new headers themselves parse
cleanly for **C consumers of only the C portion** of the FFI:

- `#include <tvm/ffi/c_api.h>` — OK.
- `#include <tvm/runtime/c_backend_api.h>` — OK (transitively pulls
  base.h + c_api.h; both are C-safe).
- `#include <tvm/runtime/base.h>` — OK.
- `#include <tvm/runtime/vm/executable.h>` — **C-INCOMPATIBLE**. Pulls
  `tvm/ffi/extra/module.h` → `tvm/ffi/container/array.h` →
  `tvm/ffi/any.h` → `tvm/ffi/string.h` → `tvm/ffi/base_details.h` →
  `tvm/ffi/endian.h` → `<cstddef>`. Every runtime header past
  `c_backend_api.h` is now a C++ header. See §2 gate finding.

---

## 1. The gate finding — TVM 0.25 runtime headers are C++-only

TVM 0.14 exposed a C-consumable runtime header surface via
`tvm/runtime/c_runtime_api.h`. **TVM 0.25 deleted that file entirely.**
The only C-callable entry point that remains is
`tvm/ffi/c_api.h` (1517 lines, tvm-ffi commit `59da4c0b`).

Every other header under `tvm/runtime/` (`vm/executable.h`, `vm/vm.h`,
`vm/builtin.h`, `vm/bytecode.h`, `tensor.h`, `data_type.h`,
`device_api.h`, `memory/memory_manager.h`, `disco/*`) is now a C++
header that includes `<atomic>`, `<memory>`, `<vector>`,
`<unordered_map>`, `<tvm/ffi/container/*>` (all C++), etc.

**Consequence for the port**: the fork's C files can no longer include
any TVM runtime header except `c_backend_api.h` and `base.h`. All
executable/bytecode/vm parsing that the fork currently does by
including TVM headers directly must instead:

- Bridge across the FFI C API (`TVMFFIFunctionGetGlobal("vm_load_executable", …)`
  → runtime returns an opaque `TVMFFIObjectHandle`), letting the C++
  runtime library do the actual parsing; OR
- Reimplement the executable-format parsing in C (keeping the fork's
  current approach), interoperating with the runtime only via the
  narrow C FFI at kernel-call time.

The fork does not link against `libtvm_runtime.so` — it *is* the
runtime. So option 1 is not available. **The port must keep option 2**:
the fork's existing C parser for the Relax executable stays; the
runtime→packed-function invocation path gets rewired through
`TVMFFIFunctionCall` / `TVMFFISafeCallType`.

Load-bearing constant that unblocks this: **the Relax VM bytecode
format is unchanged.** `tvm/runtime/vm/executable.h` at v0.25.0 still
declares `#define VM_VERSION "0.14"` (comment: *"this version should
set to minimum TVM version it support"*). Opcode set (`Call=1, Ret=2,
Goto=3, If=4`) and ArgKind (`Register=0, Immediate=1, ConstIdx=2,
FuncIdx=3`) match the fork's `RelaxInstructionType` / `RelaxInstructionCallArgType`
enums 1:1. The parser body is fine; only the *typecode enums it emits*
(kTVMArgInt → kTVMFFIInt) and the *function-invocation path it drives*
change.

---

## 2. Distinct API changes needed (the shopping list)

Numbers in `[N]` are the distinct error-message classes observed
across the fork after applying an include-shim
(`tvm/runtime/c_runtime_api.h` → `tvm/ffi/c_api.h`). Without the shim
the compiler bails at the missing include; with the shim the deeper
type/identifier breakage surfaces per file (see §3).

### 2.1 Deleted types — need new definitions or aliases

| TVM 0.14 (deleted) | TVM 0.25 replacement | Notes |
| --- | --- | --- |
| `TVMValue` (union: int, float, ptr, DLTensor*, …) | `TVMFFIAny` (tagged struct `{int32_t type_index; uint32_t small_str_len; union {int64_t, double, void*, …}}`) | Union+parallel-typecode → embedded tag. Every callsite that built `TVMValue[]` + `int typecode[]` collapses to a single `TVMFFIAny[]`. |
| `TVMBackendPackedCFunc` = `int(*)(TVMValue*, int*, int, TVMRetValueHandle, void*)` | `TVMFFISafeCallType` = `int(*)(void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result)` | Different arity (5 params → 4), different arg-marshal convention, `self` handle moves to first position. Every built-in in `vm_builtin.c` changes signature. |
| `TVMFunctionHandle` = `void*` | `TVMFFIObjectHandle` = `void*` | Same underlying type; semantic layer now enforces reference-counted `TVMFFIObject` at the handle target. `TVMFFIObjectIncRef`/`TVMFFIObjectDecRef` needed for lifetime. |
| `TVMRetValueHandle` (opaque) | Gone. `TVMFFIAny*` output param on `TVMFFIFunctionCall` / `TVMFFISafeCallType`. | Every `TVMCFuncSetReturn(ret, val, tc, n)` call becomes direct `*result = ...`. |
| `TVMModuleHandle` | `TVMFFIObjectHandle` (module is a TVMFFIObject) | Typecode is `kTVMFFIModule = 68`. |
| `TVMArrayHandle` | `DLTensor*` (unchanged) or `TVMFFIObjectHandle` for the tensor-object wrapper | Bare DLTensor* remains; TVMFFI wraps with a `TVMFFITensor` object (typecode `kTVMFFITensor = 67`). |
| `TVMStreamHandle` | **Deleted with no replacement.** | TVM 0.25 has no exposed stream API in the FFI. The fork's `device_api.h` uses this for CPU device abstraction; define locally as `typedef void* TVMStreamHandle;` in a fork-owned compat header. |
| `TVM_DLL`, `TVM_ATTRIBUTE_UNUSED`, `tvm_index_t` | Deleted macros/typedefs. | `TVM_DLL` → `TVM_RUNTIME_DLL` (base.h) or drop for internal decls. `TVM_ATTRIBUTE_UNUSED` → `__attribute__((unused))`. `tvm_index_t` → `int64_t`. |

### 2.2 Deleted typecode constants — 1:1 rename to TVMFFITypeIndex

The `kTVMArgTypeCode` enum in old c_runtime_api.h had ~15 tags. TVM 0.25
replaces it with `enum TVMFFITypeIndex : int32_t` (c_api.h:86–190,
77+ tags). Mapping needed at the call sites the fork uses:

| Old constant | TVM 0.25 replacement | Notes |
| --- | --- | --- |
| `kTVMArgInt` | `kTVMFFIInt` (= 1) | Compiler suggests this rename. |
| `kTVMArgFloat` | `kTVMFFIFloat` (= 3) | Compiler suggests this rename. |
| `kTVMOpaqueHandle` | `kTVMFFIOpaquePtr` (= 6) | Compiler suggests `kDLOpaqueHandle` — **wrong**; that's a DLPack constant. Correct is `kTVMFFIOpaquePtr`. |
| `kTVMNullptr` | `kTVMFFINone` (= 0) | Semantic shift: "None" not "null-typed pointer". |
| `kTVMDataType` | `kTVMFFIDataType` (= 5) | |
| `kTVMDLTensorHandle` | No direct constant. Use `kTVMFFITensor` (= 67) and wrap DLTensor via `TVMFFITensorCreateUnsafeView` or `TVMFFITensorFromDLPackVersioned`. | Structural change — the "bare DLTensor pointer" typecode no longer exists at the FFI boundary. |
| `kTVMObjectHandle` | `kTVMFFIObject` (= 64) | Semantic split: static objects start at 64, dynamic at 128+. |
| `kTVMModuleHandle` | `kTVMFFIModule` (= 68) | |
| `kTVMPackedFuncHandle` | `kTVMFFIFunction` (= 69) | |
| `kTVMStr` | `kTVMFFIStr` (= 65) | |
| `kTVMBytes` | `kTVMFFIBytes` (= 66) | |
| `kTVMNDArrayHandle` | `kTVMFFITensor` (= 67) | See DLTensor note above. |
| `kDLDevice` | `kTVMFFIDevice` (= 4) | Used in `vm_builtin.c` device-copy builtins. |

### 2.3 Deleted C functions — need FFI-backed reimplementation

| TVM 0.14 (deleted) | TVM 0.25 replacement | Bridging cost |
| --- | --- | --- |
| `TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out)` | `TVMFFIFunctionGetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle* out)` | Every callsite must build a `TVMFFIByteArray{.data=name,.size=strlen(name)}`. Consider fork-owned inline helper. |
| `TVMFuncCall(func, TVMValue* args, int* tc, int n, TVMValue* ret, int* rtc)` | `TVMFFIFunctionCall(handle, TVMFFIAny* args, int32_t n, TVMFFIAny* result)` | Argument marshaling changes fundamentally: parallel arrays collapse into `TVMFFIAny` array. Caller must init `result->type_index = kTVMFFINone`. |
| `TVMFuncFree(handle)` | `TVMFFIObjectDecRef(handle)` | Reference-counted lifetime. |
| `TVMFuncRegisterGlobal(name, f, override)` | `TVMFFIFunctionSetGlobal(TVMFFIByteArray* name, TVMFFIObjectHandle f, int override)` | Same shape modulo name-as-bytearray. |
| `TVMFuncCreateFromCFunc(cfunc, self, fin, out)` | `TVMFFIFunctionCreate(void* self, TVMFFISafeCallType safe_call, void(*deleter)(void*), TVMFFIObjectHandle* out)` | Deleter signature simplifies. |
| `TVMArrayAlloc(shape, ndim, code, bits, lanes, dev_type, dev_id, out)` | `TVMFFITensorFromDLPackVersioned(DLManagedTensorVersioned* from, int32_t align, int32_t rw, TVMFFIObjectHandle* out)` | Caller now allocates DLTensor + wraps via DLManagedTensorVersioned. The fork's `src/core/device/device_api.c` already owns the actual allocation; only the "wrap and hand out" step needs the new API. |
| `TVMArrayFree(handle)` | `TVMFFIObjectDecRef(tensor_obj)` | Ref-counted; DLTensor lifetime tied to owning FFI object. |
| `TVMByteArray` = `{const char* data; size_t size}` | `TVMFFIByteArray` — same field layout. | Trivial `typedef TVMFFIByteArray TVMByteArray;` alias in a compat header would neutralize call sites. |
| `TVMAPISetLastError(msg)` | `TVMFFIErrorSetRaisedFromCStr("kind", msg)` | Requires classifying errors with a kind string; the fork can use `"InternalError"` uniformly at first pass. |
| `TVMCFuncSetReturn(ret, val, tc, n)` / `TVMCbArgToReturn` | Deleted; write to `TVMFFIAny* result` directly. | Every built-in-return path in `vm_builtin.c` collapses to `result->type_index = X; result->v_int64 = …;`. |
| `TVMDeviceCopyDataFromTo`, `TVMDeviceFreeDataSpace` (referenced by `relax_vm.c` / `c_runtime_api.c`) | Not exposed in the FFI C API. | The fork's `src/core/device/device_api.c` already implements CPU copy/free internally; drop the external calls, route through the fork's own DeviceAPI vtable. |

### 2.4 Backend API signature change

`tvm/runtime/c_backend_api.h` still exists, but one signature moved:

- `TVMBackendGetFuncFromEnv(void* mod_node, const char* name, TVMFunctionHandle* out)` →
  `TVMBackendGetFuncFromEnv(void* mod_node, const char* name, TVMFFIObjectHandle* out)`

The out-param typedef change is source-compatible if `TVMFunctionHandle`
is aliased to `TVMFFIObjectHandle`. `TVMBackendAllocWorkspace`,
`TVMBackendFreeWorkspace`, `TVMBackendParallelLaunch`,
`TVMBackendParallelBarrier`, `TVMBackendRunOnce`, `TVMParallelGroupEnv`,
`FTVMParallelLambda` — all unchanged.

---

## 3. Per-file breakdown

Error counts below are from a probe build with an include-shim
(`3rdparty/tvm/include/tvm/runtime/c_runtime_api.h` → `<tvm/ffi/c_api.h>`)
so the compiler can see past the missing-header wall. `-ferror-limit=200`.
Files ordered by dependency (foundation first).

### 3.1 Foundation layer — bridge first

#### `include/relax_vm.h` (121 LoC) — public API surface

- Includes `<tvm/runtime/c_runtime_api.h>` (deleted).
- Signature: `TVM_RT_WASM_RelaxVirtualMachineCreate(..., TVMModuleHandle, ...)` — `TVMModuleHandle` gone.
- Rewrite est: ~40 LoC touched.

#### `src/core/tvm_runtime/c_runtime_api.c` (452 LoC) — 49 errors

Fork's *private facade* over the old TVM C API. Every rest-of-fork
call to `TVMFuncCall` / `TVMFuncGetGlobal` / `TVMArrayAlloc` /
`TVMFuncRegisterGlobal` / `TVMCFuncSetReturn` lands here. The whole
file needs to be re-authored to sit on top of `TVMFFIFunctionCall` /
`TVMFFIFunctionGetGlobal` / `TVMFFITensorFromDLPackVersioned` / etc.
Distinct error kinds seen (post-shim):

```
error: call to undeclared function 'TVMDeviceCopyDataFromTo'
error: call to undeclared function 'TVMDeviceFreeDataSpace'
error: duplicate member 'TVMStreamHandle'  (device_api.h transitively)
error: no member named 'CreateStream' in 'struct DeviceAPI'
error: unknown type name 'TVM_ATTRIBUTE_UNUSED'
error: unknown type name 'tvm_index_t'
error: unknown type name 'TVMArrayHandle'
error: unknown type name 'TVMBackendPackedCFunc'
error: unknown type name 'TVMFunctionHandle'
error: unknown type name 'TVMModuleHandle'
error: unknown type name 'TVMStreamHandle'
error: unknown type name 'TVMValue'
error: use of undeclared identifier 'TVMBackendPackedCFunc'
```

- Rewrite est: **~350 LoC** (near-complete rewrite; keep API shape the
  rest of the fork calls, retarget implementation onto tvm-ffi).

#### `src/core/tvm_runtime/c_backend_api.c` (78 LoC) — 14 errors

Thin passthroughs from fork-internal callers to `TVMFuncGetGlobal`.
Once c_runtime_api.c bridges the FFI, this file is 2 signature swaps.

- Rewrite est: ~20 LoC.

#### `src/core/device/device_api.h` (174 LoC) + `.c` (70 LoC) + `cpu_memory.h` (36 LoC) — 11 errors

`TVMStreamHandle` used pervasively in the DeviceAPI vtable macros
(`CreateStream`, `FreeStream`, `StreamSync`, `SetStream`, `GetStream`,
plus copy-with-stream). TVM 0.25 removed the type. Fork owns its own
device impl so no upstream link is broken — just define the type
locally.

- Rewrite est: ~50 LoC (add `typedef void* TVMStreamHandle;` in a
  compat header; keep macros).

#### `src/core/module/module.h` (89) + `module.c` (228) + `shared_library.c` (165) + `system_library.c` (156) + `function_info.h` (127) + `module_impl.h` (51) — 1+6+2+3 errors

`TVMBackendPackedCFunc` type used for the PackedFunction table entries.
Once retyped to `TVMFFISafeCallType`, the `dlsym`-loaded shared-library
path (`shared_library.c`) still works — same void*-returning ABI. The
function-info assertions (`function_info.h`) using `kTVMArgInt` need
the enum-constant swap from §2.2.

- Rewrite est: ~180 LoC across the six files.

### 3.2 Utility layer — passes with header shim

#### `src/core/utils/tensor_helper.c` (160) — 12 errors

`kTVMNDArrayMagic` constant is fork-defined (not from TVM); the errors
are transitive `TVMStreamHandle` / `duplicate member` bleeding from
device_api.h. Fixes once §3.1 (device_api compat header) lands.

- Rewrite est: ~30 LoC (chase transitive breakage).

#### `src/core/utils/stream_reader.c` (170), `trie.c` (56)

**0 errors** — no direct TVM API dependencies. Untouched.

### 3.3 Relax-VM backend

#### `src/backends/relax_vm/relax_vm_register.h` (169 LoC)

Every `RelaxVMRegType_*` enum member is defined as `= kTVMArgInt`,
`= kTVMArgFloat`, `= kTVMOpaqueHandle`, `= kTVMNullptr`,
`= kTVMDataType`, `= kTVMDLTensorHandle`, `= kTVMObjectHandle`,
`= kTVMModuleHandle`, `= kTVMPackedFuncHandle`, `= kTVMStr`, `= kTVMBytes`,
`= kTVMNDArrayHandle`. Every one needs the §2.2 rename. Also has
`TVMValue value;` field on the register struct — swap to `TVMFFIAny`.

- Rewrite est: ~50 LoC.

#### `src/backends/relax_vm/relax_executable.h` (175 LoC)

`TVMValue` used for pre-allocated argument scratch. Swap to
`TVMFFIAny`. Executable-format struct layout otherwise unchanged
(bytecode format stable — see §1).

- Rewrite est: ~15 LoC.

#### `src/backends/relax_vm/relax_executable_module.c` (417 LoC) — 15 errors

File-format parsing (magic constants, versioned reads, function-table
build). All errors are transitive typecode/type-name breakage from the
headers it includes; the parser body itself is fine (bytecode format is
stable — §1).

- Rewrite est: ~40 LoC (transitive fixes only).

#### `src/backends/relax_vm/relax_vm.h` (80 LoC)

`TVMValue *call_packed_args_value;` field — swap to `TVMFFIAny*`.

- Rewrite est: ~15 LoC.

#### `src/backends/relax_vm/relax_vm.c` (361 LoC) — 28 errors

Callsite for `TVMFuncGetGlobal(name_buffer, &pf)` and the packed-func
tag-checks. All in §2.2 and §2.3 rename/rewrite tables.

- Rewrite est: ~90 LoC.

#### `src/backends/relax_vm/relax_vm_runner.c` (289 LoC) — 39 errors

The VM interpreter loop. Every Call instruction dispatches through a
`TVMBackendPackedCFunc` — every dispatch site needs the
`TVMFFISafeCallType`(handle, args, n, &result) rewrite. This is the
hottest structural change in the backend.

- Rewrite est: ~140 LoC (interpreter loop rewrite for the Call
  opcode path).

#### `src/backends/relax_vm/vm_builtin.c` (341 LoC) — 104 errors

**The largest single file to rewrite.** Every one of the ~20 built-in
packed functions (`vm_builtin_alloc_storage`,
`vm_builtin_alloc_tensor`, `vm_builtin_null_value`,
`vm_builtin_reshape`, `vm_builtin_call_tir_dyn`,
`vm_builtin_copy`, `vm_builtin_check_tensor_info`, `vm_builtin_read_if_cond`,
etc.) is declared with the old `TVMBackendPackedCFunc` signature and
uses the old `TVMValue*`/typecode-array argument-passing convention.
Every one has to be re-authored to `TVMFFISafeCallType` (`{void* self,
const TVMFFIAny* args, int32_t n, TVMFFIAny* result}`) with in-place
`TVMFFIAny` access.

- Rewrite est: **~280 LoC** (near-complete rewrite of all built-ins).

---

## 4. Aggregate

| Layer | Files | LoC in scope | LoC-to-rewrite |
| --- | --- | --- | --- |
| Foundation (compat, FFI facade, module, device) | 8 | 1,354 | ~640 |
| Utility (tensor_helper, stream_reader, trie) | 3 | 386 | ~30 |
| Relax-VM backend | 6 | 1,505 | ~630 |
| Public header | 1 | 121 | ~40 |
| **Total** | **18** | **3,366** | **~1,340** |

Distinct error classes observed with the include-shim in place: **13**
unique kinds (see §2 table cells). Without the shim, every fork C file
fails at `fatal error: 'tvm/runtime/c_runtime_api.h' file not found`
before any deeper issue can surface.

Compile-error counts per file (post-shim, `-ferror-limit=200`):

```
104  src/backends/relax_vm/vm_builtin.c
 49  src/core/tvm_runtime/c_runtime_api.c
 39  src/backends/relax_vm/relax_vm_runner.c
 28  src/backends/relax_vm/relax_vm.c
 15  src/backends/relax_vm/relax_executable_module.c
 14  src/core/tvm_runtime/c_backend_api.c
 12  src/core/utils/tensor_helper.c
 11  src/core/device/device_api.c
  6  src/core/module/shared_library.c
  3  src/core/module/system_library.c
  1  src/core/module/module.c
  0  src/core/utils/stream_reader.c
  0  src/core/utils/trie.c
```

---

## 5. Recommendation for 13.1c

**Doable as focused work; recommend splitting into two dependent
subtasks.** The pattern is structurally uniform (rewrite one
packed-function-related callsite to the new FFI convention, replicate
~40 times across ~10 files) so the second half is largely mechanical
once the first defines the abstraction layer. But it is ~1,300 LoC
across 18 files — enough surface area to warrant a checkpoint.

Proposed sub-split:

- **13.1c-foundation** — Author a fork-owned
  `src/core/tvm_compat.h` that provides the trivially-restorable
  identifiers (`TVMStreamHandle`, `TVMValue` alias, `TVM_DLL`,
  `TVM_ATTRIBUTE_UNUSED`, `tvm_index_t`, `TVMByteArray` alias) and
  `#define`-aliases for the 1:1 enum renames (§2.2). Rewrite
  `src/core/tvm_runtime/c_runtime_api.c` (the fork's private FFI
  facade). Fix `src/core/module/*`, `src/core/device/*`,
  `src/core/utils/tensor_helper.c`. **Gate**: `libtvm-rt-core.a` links.
  Estimate: ~640 LoC, 8 files.

- **13.1c-backend** — Rewrite the Relax VM built-ins and interpreter
  once the foundation defines `TVMFFISafeCallType`-shaped abstraction
  points. Fix `include/relax_vm.h`. **Gate**: `libtvm-rt-backend-relax-vm.a`
  links against the foundation library. Estimate: ~670 LoC, 7 files.

**Non-gates** (both parts): the runtime does not have to *run*
anything at the end of 13.1c — only link. Correctness comes at 13.2
(toy Relax) and 13.3 (VITS encoder), as per the M13 plan §4.

---

## 6. Reproducing the probe

```sh
git clone https://github.com/tegmentum/tvm-rt-wasm ~/git/tegmentum/tvm-rt-wasm
cd ~/git/tegmentum/tvm-rt-wasm
git checkout port/tvm-0.25
git submodule update --init --recursive

# Configure & attempt build (dies at fatal 'c_runtime_api.h' not found):
rm -rf build
cmake -DUSE_WASI_SDK=$HOME/wasi-sdk-33.0-arm64-macos -B build
cmake --build build -- -k

# Post-shim per-file probe (to surface deeper errors):
mkdir -p /tmp/probe/tvm/runtime
printf '#include <tvm/ffi/c_api.h>\n' > /tmp/probe/tvm/runtime/c_runtime_api.h
CLANG=$HOME/wasi-sdk-33.0-arm64-macos/bin/clang
$CLANG --target=wasm32-wasip1 \
  --sysroot=$HOME/wasi-sdk-33.0-arm64-macos/share/wasi-sysroot \
  -Wall -Wextra -ferror-limit=200 \
  -isystem /tmp/probe -I 3rdparty/tvm/include -I 3rdparty/dlpack/include \
  -I include -I src/core -I src/backends \
  -c src/backends/relax_vm/vm_builtin.c -o /tmp/probe/vm_builtin.o
```

---

## 6.5 Findings during 13.2 attempt (library-bin + VMExecutable format drift)

The compile-and-link gate 13.1c cleared showed the foundation and
backend static libs build cleanly against real TVM 0.25 headers.
Running an actual compiled artifact through the runtime surfaced a
second wave of format drift that was invisible at compile time:

### 6.5.1 New `tvm_ffi_env_api.h` surface required by generated code

Every Relax module compiled with `system_lib=True` produces a
`devc.o` whose static constructor calls `TVMFFIEnvModRegisterSystemLibSymbol`
to register the library blob, and a `lib0.o` whose TIR kernels call
`TVMFFIErrorSetRaisedFromCStrParts` to raise structured errors.
Neither symbol existed in the old `TVMBackend*` C surface the fork
rehosts. `src/core/tvm_runtime/tvm_ffi_env_api.c` bridges them:
`TVMFFIEnvModRegisterSystemLibSymbol` forwards into
`TVMBackendRegisterSystemLibSymbol` (same trie backend);
`TVMFFIErrorSetRaisedFromCStrParts` joins the parts and stashes into
the fork's `TVMAPISetLastError` buffer.

### 6.5.2 System-lib well-known symbols renamed

Fork tracked upstream's `__tvm_dev_mblob` / `__tvm_module_ctx`
constants. TVM 0.25 renamed these to `__tvm_ffi__library_bin` /
`__tvm_ffi__library_ctx` (see `tvm/ffi/extra/module.h`). Bumped in
`src/core/module/module.h` — devc.o's registrar constructor now
lands the blob under the exact string the loader queries.

### 6.5.3 `LibraryModuleLoadBinaryBlob` envelope format rewrite

The fork inherited a pre-0.20 envelope: `u64 blob_size` + inline
`key_num`-many `(u64 type_key_size, char[] type_key, module body)`
entries with `_lib` / `_import_tree` sentinels intermixed. TVM 0.25
switched to (`tvm_ffi/src/ffi/extra/library_module.cc`,
`ProcessLibraryBin`):

    u64                  nbytes
    vec<u64>             import_tree_indptr        (size = num_modules + 1)
    vec<u64>             import_tree_child_indices
    for i in [0, num_modules):
        str              kind                       (u64 len + bytes)
        if kind != "_lib":
            bytes        module_body                (u64 len + bytes)

Rewritten in `src/core/module/module.c`. Module bodies are now
independently framed — each gets its own `BinaryReader` bounded to
the body length rather than sharing the outer reader.

### 6.5.4 `relax.Executable` → `relax.VMExecutable` module key

Serialized Relax VM executables now carry the type key
`relax.VMExecutable` (18 chars). Added the case in
`ModuleCreateFromReader`.

### 6.5.5 Bytecode magic bump V1 → V2

`kTVMVMBytecodeMagic` bumped from `0xD225DE2F4214151D` (V1) to
`0xD225DE2F4214151E` (V2). V2 gates in a new `MemoryScopeSection`
between Global and Constant sections. Fork's
`RelaxExecutableModuleCreate` accepts V2 magic and drops the deleted
`exec_size` u64 preamble (per-module body framing is now the outer
envelope's job, per 6.5.3).

### 6.5.6 VMExecutable Global-section format drift — **RESOLVED in 13.2b**

The `strm->Read(&func_table)` path in TVM 0.25's `LoadGlobalSection`
consumes a `std::vector<VMFuncInfo>`, and `VMFuncInfo::Load` reads a
6-field header (int32 kind + string name + 4×int64) followed by a
`vec<std::string> param_names`. The pre-Unity loader that the fork
inherited hard-coded a per-kind field list (5 int64s for Packed;
4 int64s + `num_params == num_args` + names for VMFunc) that
coincidentally matched the new format only when
`param_names.size() == num_args` and never for non-empty Packed
param vectors. Beyond that, V2 magic (0xD225DE2F4214151E) gates in a
new MemoryScope section between Global and Constant.

Fix (`1ff46d9`): read the 6 shared header fields plus the
`param_names` vector once regardless of kind and interpret per-kind
after the fact. Add `LoadMemoryScopeSection` (drains scope entries;
the CPU-only path treats every allocation as global-scope). Dispatch
it between Global and Constant, gated on the V2 magic.

Constant-section drift discovered while chasing 6.5.6: TVM 0.25 tags
each entry with a `TVMFFITypeIndex` (`kTVMFFITensor=70`,
`kTVMFFIShape=69`, `kTVMFFIStr=65`, `kTVMFFIInt=1`, `kTVMFFIFloat=3`,
`kTVMFFIDataType=5`) rather than the old
`RelaxConstantType_{DLTensor=0,DLDataType=1,ShapeTuple=2,String=3,
Int=4}` enum. Loader switched onto FFI indices, translating through
to the fork's internal enum for downstream dispatch. Also folded in a
runtime bug outside the loader path: TIR kernels emitted by
relax.build expect tensor args tagged `kTVMFFIDLTensorPtr` (7), not
the fork's internal `RelaxVMRegType_ManagedDLTensor`
(`kTVMFFITensor=70` — reads as a `TVMFFITensor` object with a
different header offset and thus wrong ndim). The runner now
normalises to `kTVMFFIDLTensorPtr` at the arg-marshalling boundary.

**Correctness gate**: `test/toy_verify.py` runs the built
`toy_relax_test.wasm` under wasmtime and diffs the produced
`output.bin` against the native-TVM oracle from `test/toy_relax.py`.
Current result on `port/tvm-0.25`:

    max_abs_diff = 0.0
    match (atol=1e-5) = True

Bit-exact against native TVM for `f(x, w, b) = matmul(x, w) + b`
(shapes 2×3 · 3×4 + 4). Commits: `1ff46d9` (loaders), `37db522`
(runtime arg tag), `af95c90` (verify script).

---

## 7. Provenance

- Milestone plan: `cognition/docs/milestone-13-plan.md` §2.3, §4 13.1, §8.
- Fork commit at start of 13.1b: `b96d365` (13.1a's wasi-sdk patch).
- TVM tag: `v0.25.0` (`c7ba0735a4f346c67b761e1fde38a68a60be8adb`), released 2026-06-19.
- tvm-ffi commit: `59da4c0b82af0d499dae34bd89ef010f64d3ff45` (pinned by TVM 0.25 as its `3rdparty/tvm-ffi`).
- DLPack: v1.3 (`84d107bf416c6bab9ae68ad285876600d230490d`), pinned by tvm-ffi.
- Toolchain: wasi-sdk 33.0 (clang 22.1.0, wasm32-wasip1), CMake 4.4.1.
