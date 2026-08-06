/**
 * @file tvm_compat.h
 * @brief Fork-owned compatibility shim spanning the TVM 0.12 -> 0.25 C ABI
 * cut. TVM 0.25 deleted `tvm/runtime/c_runtime_api.h` and replaced the
 * runtime's C-facing entry points with the `tvm/ffi/c_api.h` surface. The
 * fork previously included the old header; every C file that used
 * `TVMValue`, `TVMFunctionHandle`, `kTVMArgInt`, `TVMBackendPackedCFunc`,
 * `TVMStreamHandle`, `TVMArrayHandle`, `TVM_DLL`, `TVM_ATTRIBUTE_UNUSED`,
 * or `tvm_index_t` reached them through that include.
 *
 * This header restores those identifiers as thin aliases atop the FFI
 * types so the fork's C sources can keep their existing shape while the
 * underlying calling convention moves to `TVMFFISafeCallType` /
 * `TVMFFIAny`.
 *
 * Semantic notes:
 *  - `TVMValue`  -> `TVMFFIAny` (tagged union, embedded `type_index`).
 *  - `TVMBackendPackedCFunc` -> `TVMFFISafeCallType`
 *      `int (*)(void* self, const TVMFFIAny* args, int32_t n, TVMFFIAny* result)`.
 *    The old 6-arg convention is gone. Fork callsites that dispatch via
 *    `PackedFunction::exec(args,tc,n,ret,rtc,handle)` must be rewritten
 *    (that lives in 13.1c-backend). The typedef alias keeps stored
 *    pointers/casts compiling.
 *  - `TVMStreamHandle` was deleted upstream with no replacement; kept as
 *    `void*` because the fork's own `DeviceAPI` vtable still models
 *    stream slots.
 *  - Enum-constant aliases (`kTVMArgInt`, ...) map to the FFI type-index
 *    values. Because the compiler suggestion for `kTVMOpaqueHandle` was
 *    wrong (`kDLOpaqueHandle` is a DLPack constant), the correct mapping
 *    is `kTVMFFIOpaquePtr` -- see catalog §2.2.
 *  - The fork does not link against libtvm_runtime; it *is* the runtime.
 *    The prototypes for `TVMFuncGetGlobal`, `TVMArrayAlloc`, etc. below
 *    are the fork's internal facade only, backed by `c_runtime_api.c`.
 */

#ifndef TVM_RT_WASM_CORE_TVM_COMPAT_H_INCLUDE_
#define TVM_RT_WASM_CORE_TVM_COMPAT_H_INCLUDE_

#include <stdint.h>
#include <stddef.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------- Attribute / DLL macros -------------------------------*/

/**
 * @brief Legacy TVM export macro. TVM 0.25 splits into `TVM_DLL` (compiler
 * lib) and `TVM_RUNTIME_DLL` (runtime lib); for the fork's own public
 * symbols we just want default visibility.
 */
#ifndef TVM_DLL
#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define TVM_DLL EMSCRIPTEN_KEEPALIVE
#elif defined(_MSC_VER)
#define TVM_DLL __declspec(dllexport)
#else
#define TVM_DLL __attribute__((visibility("default")))
#endif
#endif /* TVM_DLL */

#ifndef TVM_ATTRIBUTE_UNUSED
#if defined(__GNUC__) || defined(__clang__)
#define TVM_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define TVM_ATTRIBUTE_UNUSED
#endif
#endif /* TVM_ATTRIBUTE_UNUSED */

/*----------------------- Legacy typedefs --------------------------------------*/

/** @brief Legacy shape/index element type -- was defined in the old header. */
typedef int64_t tvm_index_t;

/**
 * @brief The old on-stack value union. In TVM 0.25 the on-stack Any type
 * is `TVMFFIAny`; keep the alias so existing `.v_int64` / `.v_float64` /
 * `.v_ptr` / `.v_device` / `.v_dtype` accesses continue to compile.
 */
typedef TVMFFIAny TVMValue;

/** @brief Old function handle -- opaque pointer, semantics enforced by FFI. */
typedef void *TVMFunctionHandle;

/** @brief Old module handle -- fork-internal `Module*`. */
typedef void *TVMModuleHandle;

/**
 * @brief Old return-value handle. TVM 0.25 replaces this with an output
 * `TVMFFIAny*`; keep the alias only for legacy struct-field declarations.
 */
typedef void *TVMRetValueHandle;

/**
 * @brief Stream handle. Deleted upstream in TVM 0.25 with no replacement;
 * the fork's own `DeviceAPI` still models it, so we own the definition.
 */
typedef void *TVMStreamHandle;

/** @brief Bare-DLTensor handle -- unchanged shape, keep as alias. */
typedef DLTensor *TVMArrayHandle;

/**
 * @brief Old string+length container. Layout matches `TVMFFIByteArray`
 * one-for-one (`{ const char* data; size_t size; }`).
 */
typedef TVMFFIByteArray TVMByteArray;

/**
 * @brief Old backend "packed function" callback type.
 *
 * TVM 0.25 replaced the 5-arg
 *   `int (*)(TVMValue*, int*, int, TVMRetValueHandle, void*)`
 * with the FFI safe-call:
 *   `int (*)(void* self, const TVMFFIAny* args, int32_t n, TVMFFIAny* result)`.
 *
 * The alias below lets legacy declarations continue to compile. Callsites
 * that *invoke* packed functions (rest of the fork lives in 13.1c-backend)
 * must switch to the new arg convention -- pointer casts alone are not
 * enough to bridge the calling ABI.
 */
typedef TVMFFISafeCallType TVMBackendPackedCFunc;

/*----------------------- Legacy union-member aliases --------------------------*/

/*
 * The TVM 0.14 `TVMValue` union spelled its pointer / dtype slots as
 * `v_handle` / `v_type`; the TVM 0.25 `TVMFFIAny` union uses `v_ptr` /
 * `v_dtype`. Alias via preprocessor so fork sources keep the old member
 * names without a mass rename. Safe: neither identifier is used as a
 * struct member anywhere else in the fork nor in the vendored TVM 0.25
 * headers (verified via grep across `src/`, `include/`, `3rdparty/`).
 */
#ifndef v_handle
#define v_handle v_ptr
#endif
#ifndef v_type
#define v_type v_dtype
#endif

/*----------------------- Legacy typecode enum ---------------------------------*/

/**
 * @brief Aliases for the ~10 old typecode constants that were in
 * `TVMArgTypeCode`. Point at the corresponding `TVMFFITypeIndex` values;
 * these names must not collide with anything in the FFI c_api.h enum.
 *
 * Note the corrections vs the compiler's suggested renames:
 *  - `kTVMOpaqueHandle` -> `kTVMFFIOpaquePtr` (compiler suggested
 *    `kDLOpaqueHandle`, which is a DLPack constant with a different value
 *    -- see catalog §2.2).
 *  - `kTVMNullptr` semantic shift: FFI's `kTVMFFINone = 0` is
 *    conceptually "None", not "null-typed pointer" -- the fork's uses of
 *    the constant treat it as "absent typed slot", which matches.
 */
enum {
    kTVMArgInt          = (int)kTVMFFIInt,
    kTVMArgBool         = (int)kTVMFFIBool,
    kTVMArgFloat        = (int)kTVMFFIFloat,
    kTVMOpaqueHandle    = (int)kTVMFFIOpaquePtr,
    kTVMNullptr         = (int)kTVMFFINone,
    kTVMDataType        = (int)kTVMFFIDataType,
    kTVMDLDevice        = (int)kTVMFFIDevice,
    /* Fork sources also spell this without the `kTVM` prefix -- see
     * relax_vm_register.h `RelaxVMRegType_DLDevice = kDLDevice`. Keep both
     * spellings pointing at the same FFI index. */
    kDLDevice           = (int)kTVMFFIDevice,
    kTVMDLTensorHandle  = (int)kTVMFFIDLTensorPtr,
    kTVMObjectHandle    = (int)kTVMFFIObject,
    kTVMModuleHandle    = (int)kTVMFFIModule,
    kTVMPackedFuncHandle = (int)kTVMFFIFunction,
    kTVMStr             = (int)kTVMFFIStr,
    kTVMBytes           = (int)kTVMFFIBytes,
    kTVMNDArrayHandle   = (int)kTVMFFITensor,
};

/*----------------------- Fork-internal facade ---------------------------------*/
/*
 * These are the fork's own C entry points -- they retain the old names
 * because the fork's own C sources still call them internally. Their
 * bodies live in src/core/tvm_runtime/c_runtime_api.c and are
 * reimplemented on top of the fork's own state (Trie of global functions,
 * DeviceAPI vtable, CPU allocator). The fork does not link against
 * libtvm_runtime -- these names would collide with upstream but do not,
 * because upstream's C shim was deleted in v0.25.
 */

/** @brief Store an error message in the fork's TLS-shaped global buffer. */
void TVMAPISetLastError(const char *msg);

/** @brief Read the last error message from the fork's global buffer. */
const char *TVMGetLastError(void);

/* Module lifecycle. */
int TVMModLoadFromFile(const char *file_name, const char *format, TVMModuleHandle *out);
int TVMModImport(TVMModuleHandle mod, TVMModuleHandle dep);
int TVMModGetFunction(TVMModuleHandle mod, const char *func_name, int query_imports,
                      TVMFunctionHandle *out);
int TVMModFree(TVMModuleHandle mod);

/* Packed-function registry. */
int TVMFuncFree(TVMFunctionHandle func);
int TVMFuncRegisterGlobal(const char *name, TVMFunctionHandle f, int override);
int TVMFuncGetGlobal(const char *name, TVMFunctionHandle *out);
int TVMFuncListGlobalNames(int *out_size, const char ***out_array);
int TVMFuncRemoveGlobal(const char *name);

/* Tensor allocation / transfer -- fork-internal names, DeviceAPI backed. */
int TVMArrayAlloc(const tvm_index_t *shape, int ndim, int dtype_code, int dtype_bits,
                  int dtype_lanes, int device_type, int device_id, TVMArrayHandle *out);
int TVMArrayFree(TVMArrayHandle handle);
int TVMArrayCopyFromBytes(TVMArrayHandle handle, void *data, size_t nbytes);
int TVMArrayCopyToBytes(TVMArrayHandle handle, void *data, size_t nbytes);
int TVMArrayCopyFromTo(TVMArrayHandle from, TVMArrayHandle to, TVMStreamHandle stream);
int TVMArrayFromDLPack(DLManagedTensor *from, TVMArrayHandle *out);
int TVMArrayToDLPack(TVMArrayHandle from, DLManagedTensor **out);
void TVMDLManagedTensorCallDeleter(DLManagedTensor *dltensor);

/* Stream. On CPU-only wasm builds these degenerate to no-ops. */
int TVMStreamCreate(int device_type, int device_id, TVMStreamHandle *out);
int TVMStreamFree(int device_type, int device_id, TVMStreamHandle stream);
int TVMSetStream(int device_type, int device_id, TVMStreamHandle handle);
int TVMSynchronize(int device_type, int device_id, TVMStreamHandle stream);
int TVMStreamStreamSynchronize(int device_type, int device_id, TVMStreamHandle src,
                               TVMStreamHandle dst);

/* Device allocation -- CPU path stays local, GPU paths route through the
 * fork's DeviceAPI. */
int TVMDeviceAllocDataSpace(DLDevice dev, size_t nbytes, size_t alignment, DLDataType type_hint,
                            void **out_data);
int TVMDeviceAllocDataSpaceWithScope(DLDevice dev, int ndim, const int64_t *shape,
                                     DLDataType dtype, const char *mem_scope, void **out_data);
int TVMDeviceFreeDataSpace(DLDevice dev, void *ptr);
int TVMDeviceCopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream);

/**
 * @brief Fork-internal helper to install the built-in "__tvm_set_device"
 * packed function in the global registry. Signature matches
 * `TVMFFISafeCallType` -- called through PackedFunction dispatch.
 */
int TVM_RT_WASM_SetDevice(void *self, const TVMFFIAny *args, int32_t num_args, TVMFFIAny *result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TVM_RT_WASM_CORE_TVM_COMPAT_H_INCLUDE_ */
