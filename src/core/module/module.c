/**
 * @file module/module.c
 * @brief Implement functions for module_impl.h
 */

#include <string.h>

#include <device/cpu_memory.h>
#include <module/module_impl.h>
#include <utils/binary_reader.h>

#define MODULE_CREATE_IF_NO_SUPPORT(dev)                                                           \
    _Pragma(TOSTRING(weak TVM_RT_WASM_##dev##ModuleCreate));                                       \
    int TVM_RT_WASM_##dev##ModuleCreate(BinaryReader *reader, Module **out) {                      \
        (void)reader;                                                                              \
        *out = NULL;                                                                               \
        TVM_RT_##dev##_NOT_LINK();                                                                 \
        return -1;                                                                                 \
    }

MODULE_CREATE_IF_NO_SUPPORT(CUDA)
MODULE_CREATE_IF_NO_SUPPORT(WebGPU)

#define TVM_RT_BACKEND_NOT_ON(_mod, _backend)                                                      \
    do {                                                                                           \
        fprintf(stderr,                                                                            \
                "%s module is not supported! You can link with the backend library `%s`.\n",       \
                TOSTRING(_mod), _backend);                                                         \
        exit(-1);                                                                                  \
    } while (0)

#define TVM_RT_RelaxExecutable_NOT_LINK()                                                          \
    TVM_RT_BACKEND_NOT_ON(RelaxExecutable, "tvm-rt-backend-relax-vm")

MODULE_CREATE_IF_NO_SUPPORT(RelaxExecutable)

/** @brief Default function for module get function. */
int TVM_RT_WASM_DefaultModuleGetFunction(Module *mod, const char *func_name, int query_imports,
                                         PackedFunction **out) {
    int status = -1;
    if (mod->module_funcs_map != NULL) {
        status =
            TVM_RT_WASM_TrieQuery(mod->module_funcs_map, (const uint8_t *)func_name, (void **)out);
        if (likely(status != TRIE_NOT_FOUND)) {
            return status;
        }
    }

    if (query_imports) {
        if (mod->env_funcs_map != NULL) {
            status =
                TVM_RT_WASM_TrieQuery(mod->env_funcs_map, (const uint8_t *)func_name, (void **)out);
        }

        if (status && mod->imports) {
            for (size_t i = 0; i < mod->num_imports; ++i) {
                Module *m = mod->imports[i];
                if (m) {
                    status = m->GetFunction(m, func_name, query_imports, out);
                    if (status == 0) {
                        return status;
                    }
                }
            }
        }
    }

    return status;
}

/**
 * @brief Create a module instance from the byte stream.
 * @param type_key The module type key to read.
 * @param type_key_size The module type key string length.
 * @param reader The module binary reader.
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 * @note This function cannot create Library module, such as system library and shared library.
 */
static int TVM_RT_WASM_ModuleCreateFromReader(const char *type_key, size_t type_key_size,
                                              BinaryReader *reader, Module **out) {
    switch (type_key_size) {
    case 4:
        if (!memcmp(type_key, "cuda", 4)) {
            return TVM_RT_WASM_CUDAModuleCreate(reader, out);
        }
        break;
    case 6:
        if (!memcmp(type_key, "webgpu", 6)) {
            return TVM_RT_WASM_WebGPUModuleCreate(reader, out);
        }
        break;
    case 15:
        if (!memcmp(type_key, "metadata_module", 15)) {
            // empty module
            *out = NULL;
            return 0;
        }
        break;
    case 16:
        if (!memcmp(type_key, "relax.Executable", 16)) {
            return TVM_RT_WASM_RelaxExecutableModuleCreate(reader, out);
        }
        break;
    case 18:
        /*
         * TVM 0.25 renamed the serialized Relax executable's module
         * type-key from "relax.Executable" (pre-0.20) to
         * "relax.VMExecutable" (see src/runtime/relax_vm/executable.cc
         * upstream). Bytecode format is stable across the rename;
         * only the key string moved.
         */
        if (!memcmp(type_key, "relax.VMExecutable", 18)) {
            return TVM_RT_WASM_RelaxExecutableModuleCreate(reader, out);
        }
        break;
    default:
        break;
    }
    TVM_RT_SET_ERROR_RETURN(-1, "Unsupported module type key `%.*s` (size %zu)",
                            (int)type_key_size, type_key, type_key_size);
}

int TVM_RT_WASM_LibraryModuleLoadBinaryBlob(const char *blob, Module **lib_module) {
    /*
     * TVM 0.25 library binary layout (see tvm_ffi/src/ffi/extra/library_module.cc,
     * `ProcessLibraryBin`):
     *
     *   u64                       nbytes
     *   vec<u64>                  import_tree_indptr        (size = num_modules + 1)
     *   vec<u64>                  import_tree_child_indices
     *   for i in [0, num_modules):
     *       str                   kind                      (u64 len + bytes)
     *       if kind != "_lib":
     *           bytes             module_body               (u64 len + bytes)
     *
     * The old format the fork inherited from pre-0.20 TVM stored a flat
     * `key_num` and interleaved module bodies straight into the outer
     * reader (no per-module body-size framing). This rewrite matches the
     * new shape so every module's serialized bytes get their own bounded
     * reader — the Relax executable path in RelaxExecutableModuleCreate
     * no longer needs the outer "module size" preamble either.
     */
    size_t blob_size = (size_t) * (uint64_t *)blob;
    blob += sizeof(uint64_t);

    BinaryReader reader_st = TVM_RT_WASM_BinaryReaderCreate(blob, blob_size);
    if (unlikely(reader_st.current_ptr == NULL)) {
        return -1;
    }
    BinaryReader *reader = &reader_st;
    const char *cur_ptr;
    Module **modules = NULL;
    uint64_t *indptr = NULL;
    uint64_t *child_indices = NULL;
    size_t indptr_size = 0;
    size_t child_indices_size = 0;
    size_t num_modules = 0;
    int status = 0;

#define ModuleBinaryCheckReadOrGoto(_ptr, _read_size)                                              \
    TVM_RT_WASM_BinaryCheckReadOrGoto(_ptr, _read_size, parse_binary_return)

    /* import_tree_indptr */
    ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t));
    indptr_size = (size_t) * (uint64_t *)cur_ptr;
    if (unlikely(indptr_size == 0)) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(parse_binary_return,
                                  "Library binary: import_tree_indptr must be non-empty.");
    }
    {
        size_t bytes = sizeof(uint64_t) * indptr_size;
        indptr = TVM_RT_WASM_HeapMemoryAlloc(bytes);
        ModuleBinaryCheckReadOrGoto(cur_ptr, bytes);
        memcpy(indptr, cur_ptr, bytes);
    }

    /* import_tree_child_indices */
    ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t));
    child_indices_size = (size_t) * (uint64_t *)cur_ptr;
    if (child_indices_size > 0) {
        size_t bytes = sizeof(uint64_t) * child_indices_size;
        child_indices = TVM_RT_WASM_HeapMemoryAlloc(bytes);
        ModuleBinaryCheckReadOrGoto(cur_ptr, bytes);
        memcpy(child_indices, cur_ptr, bytes);
    }

    num_modules = indptr_size - 1;
    modules = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module *) * num_modules);
    memset(modules, 0, sizeof(Module *) * num_modules);

    for (size_t i = 0; i < num_modules; ++i) {
        ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t));
        size_t kind_size = (size_t) * (uint64_t *)cur_ptr;
        const char *kind;
        ModuleBinaryCheckReadOrGoto(kind, kind_size);

        if (kind_size == 4 && !memcmp(kind, "_lib", 4)) {
            /* Placeholder for the caller-supplied DSO / system-lib module. */
            modules[i] = *lib_module;
            continue;
        }

        /* Non-_lib entries carry their own serialized body (u64 length + bytes). */
        ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t));
        size_t body_size = (size_t) * (uint64_t *)cur_ptr;
        const char *body;
        ModuleBinaryCheckReadOrGoto(body, body_size);

        BinaryReader body_reader = TVM_RT_WASM_BinaryReaderCreate(body, body_size);
        if (unlikely(body_reader.current_ptr == NULL)) {
            status = -1;
            goto parse_binary_return;
        }
        status = TVM_RT_WASM_ModuleCreateFromReader(kind, kind_size, &body_reader, modules + i);
        if (unlikely(status)) {
            goto parse_binary_return;
        }
    }

    /* Wire imports from the CSR indptr/child_indices structure. */
    for (size_t i = 0; i < num_modules; ++i) {
        if (unlikely(indptr[i] > indptr[i + 1])) {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(parse_binary_return,
                                      "Library binary: import_tree_indptr not monotonic.");
        }
        size_t num_imports = (size_t)(indptr[i + 1] - indptr[i]);
        if (modules[i] == NULL) {
            /* Empty (metadata) module — collapse when it has a single import. */
            if (num_imports == 1) {
                modules[i] = modules[child_indices[indptr[i]]];
            }
            continue;
        }
        modules[i]->num_imports = num_imports;
        if (num_imports == 0) {
            continue;
        }
        modules[i]->imports = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module *) * num_imports);
        memset(modules[i]->imports, 0, sizeof(Module *) * num_imports);
        for (size_t j = indptr[i], x = 0; j < indptr[i + 1]; ++j, ++x) {
            if (unlikely(j >= child_indices_size)) {
                break;
            }
            modules[i]->imports[x] = modules[child_indices[j]];
        }
    }

    /* Module 0 is the root by TVM 0.25 convention (see ProcessLibraryBin). */
    *lib_module = modules[0];
    if ((*lib_module) && (*lib_module)->env_funcs_map == NULL) {
        TVM_RT_WASM_TrieCreate(&(*lib_module)->env_funcs_map);
    }

parse_binary_return:
    if (modules) {
        TVM_RT_WASM_HeapMemoryFree(modules);
    }
    if (indptr) {
        TVM_RT_WASM_HeapMemoryFree(indptr);
    }
    if (child_indices) {
        TVM_RT_WASM_HeapMemoryFree(child_indices);
    }
    return status;
}
