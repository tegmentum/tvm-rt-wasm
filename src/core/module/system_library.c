/**
 * @file module/system_library.c
 * @brief Implementation for system library module.
 */

#include <module/module_impl.h>
#include <string.h>
#include <tvm_compat.h>

/** @brief SystemLibraryModule, derive from Module. */
typedef struct SystemLibraryModule {
    MODULE_BASE_MEMBER

    PackedFunction *packed_functions;
} SystemLibraryModule;

/**
 * @brief The symbols for system library.
 * @note The system library module instance will manage this trie after created.
 */
static Trie *sys_lib_symbols = NULL;

/**
 * @brief The shared "kernels" base module, built once on first
 * SystemLibraryModuleCreate* call. Owns the sys_lib_symbols trie after
 * wrap; imported by every per-prefix Relax-executable root.
 */
static SystemLibraryModule *base_sys_lib = NULL;

/**
 * @brief Cache of per-prefix root Modules: prefix string -> Module *.
 * Populated lazily on first CreateWithPrefix(prefix) call and reused on
 * subsequent calls for the same prefix.
 */
static Trie *root_by_prefix = NULL;

/**
 * if sys_lib_symbols is not NULL, the num_sys_lib_symbol is valid.
 * if base_sys_lib is not NULL, the sys_lib_root_mod_release_func is valid.
 */
static union {
    /** @brief The number of system library symbols. */
    size_t num_sys_lib_symbol;
    /** @brief The system library root module release function. */
    int (*sys_lib_root_mod_release_func)(Module *self);
} sys_lib_status = {.num_sys_lib_symbol = 0};

/** @brief Destroy the single instance when exit. */
static TVM_ATTRIBUTE_UNUSED __attribute__((destructor)) void TVM_RT_WASM_Destructor_SysLib() {
    if (base_sys_lib) {
        sys_lib_status.sys_lib_root_mod_release_func((Module *)base_sys_lib);
    } else if (sys_lib_symbols) {
        TVM_RT_WASM_TrieRelease(sys_lib_symbols);
    }
    if (root_by_prefix) {
        TVM_RT_WASM_TrieRelease(root_by_prefix);
        root_by_prefix = NULL;
    }
}

/**
 * @brief Backend function to register system-wide library symbol.
 * @sa tvm/runtime/c_backend_api.h
 */
TVM_DLL TVM_ATTRIBUTE_UNUSED int TVMBackendRegisterSystemLibSymbol(const char *name, void *ptr) {
    if (unlikely(sys_lib_symbols == NULL)) {
        if (unlikely(base_sys_lib != NULL)) {
            TVM_RT_SET_ERROR_RETURN(
                -1, "Cannot register symbol! The system library module has been created!");
        }
        TVM_RT_WASM_TrieCreate(&sys_lib_symbols);
    }
    ++sys_lib_status.num_sys_lib_symbol;
    return TVM_RT_WASM_TrieInsert(sys_lib_symbols, (const uint8_t *)name, ptr);
}

/** @brief The release function for system library module. */
static int TVM_RT_WASM_SysLibModuleReleaseFunc(Module *mod) {
    SystemLibraryModule *sys_lib_module = (SystemLibraryModule *)mod;
    MODULE_BASE_MEMBER_FREE(sys_lib_module);
    if (sys_lib_module->packed_functions) {
        TVM_RT_WASM_HeapMemoryFree(sys_lib_module->packed_functions);
    }
    TVM_RT_WASM_HeapMemoryFree(sys_lib_module);
    return 0;
}

/** @brief Do nothing function, replace the system library root module release function.
 * The system library root module can only be released when exit.
 */
static int TVM_RT_WASM_ReleaseDoNothing(Module *mod) {
    (void)mod;
    return 0;
}

/** @brief Visit function, change the symbol to packed function. */
static void TVM_RT_WASM_TrieVisit_ChangeSymbolToPackedFunc(void **data_ptr, void *source_handle) {
    static int now_functions = 0;
    PackedFunction *pf = (PackedFunction *)source_handle;
    void *data = *data_ptr;
    if (data != NULL) {
        pf[now_functions].exec = data;
        *data_ptr = pf + now_functions;
        ++now_functions;
    }
}

/**
 * @brief Ensure `base_sys_lib` is materialised. Idempotent: only runs
 * the wrap-into-packed-functions dance on the first call, subsequent
 * calls are no-ops. On success, `sys_lib_symbols` has been ownership-
 * transferred into `base_sys_lib->module_funcs_map` and set to NULL.
 * @return 0 on success.
 */
static int EnsureBaseSysLib(void) {
    if (base_sys_lib != NULL) {
        return 0;
    }
    if (unlikely(sys_lib_symbols == NULL)) {
        TVM_RT_SET_ERROR_RETURN(-1, "No symbol in system library!");
    }

    SystemLibraryModule *sl = TVM_RT_WASM_HeapMemoryAlloc(sizeof(SystemLibraryModule));
    memset(sl, 0, sizeof(SystemLibraryModule));

    Module *sl_mod = (Module *)sl;
    sl_mod->Release = TVM_RT_WASM_SysLibModuleReleaseFunc;
    sl_mod->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    TVM_RT_WASM_TrieCreate(&sl_mod->env_funcs_map);

    sl->packed_functions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * sys_lib_status.num_sys_lib_symbol);
    memset(sl->packed_functions, 0,
           sizeof(PackedFunction) * sys_lib_status.num_sys_lib_symbol);
    TVM_RT_WASM_TrieVisit(sys_lib_symbols, TVM_RT_WASM_TrieVisit_ChangeSymbolToPackedFunc,
                          sl->packed_functions);

    sl_mod->module_funcs_map = sys_lib_symbols;
    sys_lib_symbols = NULL;

    base_sys_lib = sl;
    sys_lib_status.sys_lib_root_mod_release_func = sl_mod->Release;
    sl_mod->Release = TVM_RT_WASM_ReleaseDoNothing;
    return 0;
}

int TVM_RT_WASM_SystemLibraryModuleCreateWithPrefix(const char *prefix, Module **out_module) {
    if (prefix == NULL) {
        prefix = "";
    }

    /*
     * Cache hit: return the cached root Module for this prefix.
     */
    if (root_by_prefix != NULL) {
        void *cached = NULL;
        if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery(root_by_prefix, (const uint8_t *)prefix,
                                                  &cached)) {
            *out_module = (Module *)cached;
            return 0;
        }
    }

    /*
     * Build the shared kernel-base module once. After this, the raw
     * sys_lib_symbols trie is owned by base_sys_lib->module_funcs_map,
     * and each entry's value is a PackedFunction* whose `.exec` field
     * holds the original raw pointer (data blob / slot address for the
     * library_bin/library_ctx entries; TIR function pointer for kernels).
     */
    int status = EnsureBaseSysLib();
    if (status != 0) {
        return status;
    }

    if (root_by_prefix == NULL) {
        TVM_RT_WASM_TrieCreate(&root_by_prefix);
    }

    /*
     * Build the trie lookup keys `<prefix><TVM_DEV_MODULE_BLOB>` and
     * `<prefix><TVM_MODULE_CTX>`. TVM's `system_lib_prefix` attribute
     * gets baked into the C-symbol names by the LLVM codegen backend —
     * for prefix="enc_", the symbols land as `enc___tvm_ffi__library_bin`
     * and `enc___tvm_ffi__library_ctx`.
     */
    char bin_key[256];
    char ctx_key[256];
    int written = snprintf(bin_key, sizeof(bin_key), "%s%s", prefix, TVM_DEV_MODULE_BLOB);
    if (written < 0 || (size_t)written >= sizeof(bin_key)) {
        TVM_RT_SET_ERROR_RETURN(-1, "prefix `%s` overflows library_bin key buffer", prefix);
    }
    written = snprintf(ctx_key, sizeof(ctx_key), "%s%s", prefix, TVM_MODULE_CTX);
    if (written < 0 || (size_t)written >= sizeof(ctx_key)) {
        TVM_RT_SET_ERROR_RETURN(-1, "prefix `%s` overflows library_ctx key buffer", prefix);
    }

    /*
     * library_bin: look up the wrapped PackedFunction entry, recover the
     * raw blob pointer from its `.exec` field, hand to the shared
     * LoadBinaryBlob loader. `*root` goes in as base_sys_lib (used to
     * fill any `_lib` sentinel) and comes out as the Relax-executable
     * root module (modules[0] by TVM 0.25 convention).
     */
    PackedFunction *blob_pf = NULL;
    if (TRIE_SUCCESS != TVM_RT_WASM_TrieQuery(((Module *)base_sys_lib)->module_funcs_map,
                                              (const uint8_t *)bin_key, (void **)&blob_pf) ||
        blob_pf == NULL) {
        TVM_RT_SET_ERROR_RETURN(-1, "library_bin `%s` not registered", bin_key);
    }
    const char *blob = (const char *)blob_pf->exec;

    Module *root = (Module *)base_sys_lib;
    status = TVM_RT_WASM_LibraryModuleLoadBinaryBlob(blob, &root);
    if (status != 0) {
        return status;
    }

    /*
     * library_ctx: the weak `void *` slot the generated code reads to
     * find "its" module handle. It MUST be the Relax-executable root
     * loaded from the blob — that root's import list carries the
     * device sub-modules (e.g. the WebGPU module holding the WGSL
     * kernels). Off-host kernel host stubs in lib0.o resolve their
     * kernels via `TVMBackendGetFuncFromEnv(module_ctx, name, &pf)`,
     * which walks `module_ctx->imports` to find the WebGPU submodule's
     * `fused_*_kernel` entry.
     *
     * Previously we wrote `base_sys_lib` here — that works for the
     * inlined-kernel CPU path (lib0.o calls the kernel symbol
     * directly, module_ctx is unused) but breaks the WebGPU path
     * because base_sys_lib has no imports and the kernel lookup
     * misses.
     */
    PackedFunction *ctx_pf = NULL;
    if (TRIE_SUCCESS != TVM_RT_WASM_TrieQuery(((Module *)base_sys_lib)->module_funcs_map,
                                              (const uint8_t *)ctx_key, (void **)&ctx_pf) ||
        ctx_pf == NULL) {
        TVM_RT_SET_ERROR_RETURN(-1, "library_ctx `%s` not registered", ctx_key);
    }
    void **module_context = (void **)ctx_pf->exec;
    *module_context = (void *)root;

    TVM_RT_WASM_TrieInsert(root_by_prefix, (const uint8_t *)prefix, root);
    *out_module = root;
    return 0;
}

int TVM_RT_WASM_SystemLibraryModuleCreate(Module **out_module) {
    return TVM_RT_WASM_SystemLibraryModuleCreateWithPrefix("", out_module);
}
