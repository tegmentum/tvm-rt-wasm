/**
 * @file module/module.h
 * @brief Define the module base interface and member.
 */

#ifndef TVM_RT_WASM_CORE_MODULE_MODULE_H_INCLUDE_
#define TVM_RT_WASM_CORE_MODULE_MODULE_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm_compat.h>
#include <utils/trie.h>

typedef struct Module Module;

typedef struct {
    /** @brief The function pointer to execute. */
    TVMBackendPackedCFunc exec;
} PackedFunction;

/** @brief the base interface in module */
#define MODULE_BASE_INTERFACE                                                                      \
    /**                                                                                            \
     * @brief Release the resource for this module.                                                \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*Release)(Module * self);                                                                 \
    /**                                                                                            \
     * @brief Find function from module.                                                           \
     * @param mod The module handle.                                                               \
     * @param func_name The name of the function.                                                  \
     * @param query_imports Whether to query imported modules.                                     \
     * @param out The pointer to save result packed function.                                      \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*GetFunction)(Module * mod, const char *func_name, int query_imports,                     \
                       PackedFunction **out);

/** @brief The base member in module. */
#define MODULE_BASE_MEMBER                                                                         \
    /** @brief the base interfaces. */                                                             \
    MODULE_BASE_INTERFACE                                                                          \
    /** @brief the depend modules array. */                                                        \
    Module **imports;                                                                              \
    /** @brief the cached map <string, PackedFunction*>.                                           \
     *  For "GetFuncFromEnv", save imports + global function.                                      \
     */                                                                                            \
    Trie *env_funcs_map;                                                                           \
    /** @brief the module functions, map <string, PackedFunction*>. */                             \
    Trie *module_funcs_map;                                                                        \
    /** @brief the number of imports. */                                                           \
    size_t num_imports;

/** @brief The base Module. */
struct Module {
    MODULE_BASE_MEMBER
};

/** @brief symbols */
/*
 * TVM 0.25 renamed the well-known library-module symbols under the
 * `__tvm_ffi_` prefix (see tvm/ffi/extra/module.h). The fork's
 * system-library loader queries the registry by these exact strings,
 * so bumping them here means generated `devc.o` constructors — whose
 * `TVMFFIEnvModRegisterSystemLibSymbol` calls emit the blob under
 * `__tvm_ffi__library_bin` — can be picked up unchanged. The context
 * variable is now a weak global in the generated `lib0.o` named
 * `__tvm_ffi__library_ctx`.
 */
#define TVM_MODULE_CTX "__tvm_ffi__library_ctx"
#define TVM_DEV_MODULE_BLOB "__tvm_ffi__library_bin"
#define TVM_SET_DEVICE_FUNCTION "__tvm_set_device"
#define TVM_MODULE_MAIN "__tvm_main__"
#define TVM_GET_METADATA_FUNC get_c_metadata
#define TVM_GET_METADATA_FUNC_NAME TOSTRING(TVM_GET_METADATA_FUNC)

/**
 * @brief Create a system library module. (It will be a single instance).
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 */
int TVM_RT_WASM_SystemLibraryModuleCreate(Module **out);

/**
 * @brief Create a system library root module keyed by TVM system_lib_prefix.
 *
 * Multi-model wasm variants (M13.7) compile each Relax IRModule with a
 * distinct `system_lib_prefix` attribute so the exported symbols don't
 * collide when linked into the same binary. The devc.o constructors
 * still route into the fork's single sys_lib trie, but the
 * `<prefix>__tvm_ffi__library_bin` / `<prefix>__tvm_ffi__library_ctx`
 * pair is per-model. This entry point looks up the pair for a specific
 * prefix and returns the Relax-executable root that imports the shared
 * kernels trie.
 *
 * The shared "wrapped" SystemLibraryModule (packed_functions bound to
 * the trie's kernel entries) is built once on the first call for any
 * prefix; subsequent calls reuse it. Root modules are cached by prefix.
 *
 * Passing prefix=NULL or prefix="" is equivalent to
 * TVM_RT_WASM_SystemLibraryModuleCreate — the legacy no-prefix path.
 *
 * @param prefix TVM system_lib_prefix (or "" / NULL for the legacy path).
 * @param out The pointer to save created module instance.
 * @return 0 if successful.
 */
int TVM_RT_WASM_SystemLibraryModuleCreateWithPrefix(const char *prefix, Module **out);

/**
 * @brief Create a library module from the dynamic shared library.
 * @param filename The filename.
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 */
int TVM_RT_WASM_SharedLibraryModuleCreate(const char *filename, Module **out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_MODULE_MODULE_H_INCLUDE_
