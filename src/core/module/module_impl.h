/**
 * @file module/module_impl.h
 * @brief Private module interface, only for module implementation files.
 */

#ifndef TVM_RT_WASM_CORE_MODULE_MODULE_IMPL_H_INCLUDE_
#define TVM_RT_WASM_CORE_MODULE_MODULE_IMPL_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <module/module.h>
#include <utils/common.h>

#define MODULE_BASE_MEMBER_FREE(_mod)                                                              \
    do {                                                                                           \
        if ((_mod)->imports) {                                                                     \
            for (uint32_t i = 0; i < (_mod)->num_imports; ++i) {                                   \
                if ((_mod)->imports[i]) {                                                          \
                    (_mod)->imports[i]->Release((_mod)->imports[i]);                               \
                }                                                                                  \
            }                                                                                      \
            TVM_RT_WASM_HeapMemoryFree((_mod)->imports);                                           \
        }                                                                                          \
        if ((_mod)->module_funcs_map) {                                                            \
            TVM_RT_WASM_TrieRelease((_mod)->module_funcs_map);                                     \
        }                                                                                          \
        if ((_mod)->env_funcs_map) {                                                               \
            TVM_RT_WASM_TrieRelease((_mod)->env_funcs_map);                                        \
        }                                                                                          \
    } while (0)

/** @brief Default implementation for module get function. */
int TVM_RT_WASM_DefaultModuleGetFunction(Module *mod, const char *func_name, int query_imports,
                                         PackedFunction **out);

/**
 * @brief Allocate a per-load proxy Module that borrows @p base's
 * `module_funcs_map` (shared, read-only) but carries independent
 * `imports`, `env_funcs_map`, and lifecycle.
 *
 * Used by `TVM_RT_WASM_LibraryModuleLoadBinaryBlob` to materialise each
 * `_lib` slot in the library binary's module tree. Aliasing the raw
 * base singleton would let a subsequent load stomp the earlier root's
 * import edges (concretely, the multi-model VITS full-WebGPU variant
 * where every load's `_lib` node imports a WebGPU submodule — encoder's
 * WebGPU edge lost when decoder's load overwrote it).
 *
 * The proxy's Release func frees the shell + private state only. The
 * borrowed `module_funcs_map` remains owned by @p base (typically the
 * base system-lib module released at process exit).
 *
 * @param base Source module whose `module_funcs_map` gets shared.
 * @param out  Pointer to store the newly allocated proxy.
 */
void TVM_RT_WASM_LibraryLoaderProxyCreate(Module *base, Module **out);

/**
 * @brief Load modules tree from binary blob.
 * @param blob the dev_blob binary.
 * @param lib_module The root library module handle.
 * @return 0 if successful
 * @note It can only be used in library module.
 */
int TVM_RT_WASM_LibraryModuleLoadBinaryBlob(const char *blob, Module **lib_module);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_MODULE_MODULE_IMPL_H_INCLUDE_
