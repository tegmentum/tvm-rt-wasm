/**
 * @file module/webgpu_module.c
 * @brief Implement functions for WebGPU module.
 */

#include <device/device_api.h>
#include <module/function_info.h>
#include <module/module_impl.h>
#include <tvm_compat.h>
#include <utils/binary_reader.h>
#include <webgpu_common.h>

/** @brief WebGPU function information, derive from PackedFunction. */
typedef struct WebGPUFunctionInfo {
    /** @brief The function pointer to execute (TVM 0.25 TVMFFISafeCallType). */
    TVMBackendPackedCFunc exec;

    BASE_FUNCTION_INFO

    WGPU_Function device_func;

    /* TVM 0.25 WebGPU dispatch metadata.
     *
     * Kernel arguments split into two classes:
     *   - handle args (tensor pointers) → storage-buffer bindings 0..N-1
     *   - POD scalars (int/uint/float)   → packed into the PODArgs uniform
     *
     * TVM's WGSL codegen unconditionally emits a uniform binding at index
     * num_handle_args for the PODArgs struct (contains scalar args plus a
     * trailing packGridDimX). All plumbed through to WGPU_FunctionCreate. */
    uint8_t *handle_write_access;   /* size = num_handle_args; 1=rw, 0=ro. */
    DLDataType *pod_arg_dtypes;     /* size = num_pod_args (may be NULL). */
    uint32_t num_handle_args;       /* count of arg_types with code == kDLOpaqueHandle. */
    uint32_t num_pod_args;          /* num_kernel_args - num_handle_args. */
    uint32_t *arg_class;            /* size = num_kernel_args; 1=handle, 0=POD.
                                     * Preserves arg-order so the wrapper can split
                                     * the incoming TVMFFIAny args back into the two
                                     * classes before dispatching. */
} WebGPUFunctionInfo;

/** @brief define the WebGPU module derived from module */
typedef struct WebGPUModule {
    MODULE_BASE_MEMBER

    /** @brief the WebGPU module */
    // todo: multi-GPU support
    WebGPUFunctionInfo *functions;
    size_t num_functions;
} WebGPUModule;

/*
 * TVM 0.25 dispatch convention (TVMFFISafeCallType):
 *   int (*)(void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result)
 *
 * The pre-0.25 6-arg convention (parallel typecode arrays, out-param
 * ret_val) is gone — each arg is a self-tagged TVMFFIAny, and the wrapper
 * function is the callee (self carries the WebGPUFunctionInfo pointer).
 *
 * The function_info.h CHECK_DYN_MEM / CHECK_AND_GET_DIM macros reference
 * the old `type_codes` / `args[i].v_int64` shape; open-code the moral
 * equivalent here against args_value[i].type_index + args_value[i].v_int64.
 * Same round-trip pattern the CPU built-ins in vm_builtin.c apply.
 *
 * Tensor arg tag normalisation: TVM 0.25's Relax runner may pass args
 * tagged `kTVMFFIDLTensorPtr` (7) OR `kTVMFFITensor` (managed) — both
 * carry a DLTensor* semantically; the accelerator wants the raw device
 * pointer, so we always dereference args_value[i].v_handle for the first
 * num_kernel_args slots. Matches §6.5.6 fix in relax_vm_runner.c.
 */
static int TVM_RT_WASM_WebGPUWrappedFunction(void *self, const TVMFFIAny *args_value,
                                             int32_t num_args, TVMFFIAny *ret_value) {
    (void)ret_value;
    WebGPUFunctionInfo *info = (WebGPUFunctionInfo *)self;
    size_t block_dim[] = {1, 1, 1};
    size_t grid_dim[] = {1, 1, 1};
    size_t dyn_shared_mem_size = 0;

    uint32_t num_kernel_args = info->num_kernel_args;

    /* CHECK_DYN_MEM equivalent — TVM 0.25 shape. */
    if (info->use_dyn_mem) {
        if (unlikely(num_kernel_args + info->num_func_arg_map + 1 != (uint32_t)num_args)) {
            TVM_RT_SET_ERROR_RETURN(
                -1, "Params number expect %d, but given %d",
                num_kernel_args + info->num_func_arg_map + 1, num_args);
        }
        if (unlikely((int32_t)args_value[num_args - 1].type_index != (int32_t)kTVMArgInt)) {
            TVM_RT_SET_ERROR_RETURN(-1, "Expect int type for param %d", num_args - 1);
        }
        dyn_shared_mem_size = (size_t)args_value[num_args - 1].v_int64;
    } else {
        if (unlikely(num_kernel_args + info->num_func_arg_map != (uint32_t)num_args)) {
            TVM_RT_SET_ERROR_RETURN(-1, "Params number expect %d, but given %d",
                                    num_kernel_args + info->num_func_arg_map, num_args);
        }
    }

    if (dyn_shared_mem_size != 0) {
        TVM_RT_SET_ERROR_RETURN(-1,
                                "WebGPU cannot support dynamic shared memory, but got size %zu.",
                                dyn_shared_mem_size);
    }

    /* CHECK_AND_GET_DIM equivalent — TVM 0.25 shape. */
    for (uint32_t i = 0; i < info->num_func_arg_map; ++i) {
        if (unlikely((int32_t)args_value[i + num_kernel_args].type_index !=
                     (int32_t)kTVMArgInt)) {
            TVM_RT_SET_ERROR_RETURN(-1, "Expect int type for param %d", i);
        }
        if (info->func_arg_index_map[i] >= 3) {
            block_dim[info->func_arg_index_map[i] - 3] =
                (size_t)args_value[num_kernel_args + i].v_int64;
        } else {
            grid_dim[info->func_arg_index_map[i]] =
                (size_t)args_value[num_kernel_args + i].v_int64;
        }
    }

    /* Split kernel args by class: handles into kernel_arg_storages
     * (front-packed, size = info->num_handle_args), POD scalars into
     * a small stack buffer (size = info->num_pod_args). The order in
     * TVM's arg vector matches the order of arg_types on the module —
     * so info->arg_class[i] tells us where slot i belongs. */
    uint32_t handle_idx = 0;
    uint64_t pod_values[info->num_pod_args + 1u];
    uint32_t pod_idx = 0;
    for (uint32_t i = 0; i < num_kernel_args; ++i) {
        if (info->arg_class && info->arg_class[i] == 0u) {
            /* POD scalar arg: TVMFFIAny v_int64 slot carries the payload
             * (int/uint value in low bits, float bit pattern lifted into
             * the low 4 bytes for f32). */
            pod_values[pod_idx++] = (uint64_t)args_value[i].v_int64;
        } else {
            info->kernel_arg_storages[handle_idx++] = args_value[i].v_handle;
        }
    }

    (void)block_dim;
    int status = WGPU_FunctionRun(info->device_func, (WGPU_Memory *)info->kernel_arg_storages,
                                  info->num_handle_args,
                                  info->num_pod_args > 0 ? pod_values : NULL,
                                  info->num_pod_args, grid_dim[0], grid_dim[1], grid_dim[2]);

    return status;
}

static int TVM_RT_WASM_WebGPUModuleReleaseFunc(Module *self) {
    WebGPUModule *w = (WebGPUModule *)self;
    MODULE_BASE_MEMBER_FREE(w);

    for (size_t i = 0; i < w->num_functions; ++i) {
        if (w->functions[i].func_arg_index_map) {
            TVM_RT_WASM_HeapMemoryFree(w->functions[i].func_arg_index_map);
        }
        if (w->functions[i].kernel_arg_storages) {
            TVM_RT_WASM_HeapMemoryFree(w->functions[i].kernel_arg_storages);
        }
        if (w->functions[i].handle_write_access) {
            TVM_RT_WASM_HeapMemoryFree(w->functions[i].handle_write_access);
        }
        if (w->functions[i].pod_arg_dtypes) {
            TVM_RT_WASM_HeapMemoryFree(w->functions[i].pod_arg_dtypes);
        }
        if (w->functions[i].arg_class) {
            TVM_RT_WASM_HeapMemoryFree(w->functions[i].arg_class);
        }
        if (w->functions[i].device_func) {
            WGPU_FunctionFree(w->functions[i].device_func);
        }
    }
    TVM_RT_WASM_HeapMemoryFree(w->functions);

    // free self
    TVM_RT_WASM_HeapMemoryFree(w);
    return 0;
}

static void TVM_RT_WASM_WebGPUModuleAllocate(WebGPUModule **webgpuModule, size_t num_func) {
    *webgpuModule = TVM_RT_WASM_HeapMemoryAlloc(sizeof(WebGPUModule));
    memset(*webgpuModule, 0, sizeof(WebGPUModule));
    (*webgpuModule)->Release = TVM_RT_WASM_WebGPUModuleReleaseFunc;
    (*webgpuModule)->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    TVM_RT_WASM_TrieCreate(&((*webgpuModule)->module_funcs_map));
    (*webgpuModule)->functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(WebGPUFunctionInfo) * num_func);
    memset((*webgpuModule)->functions, 0, sizeof(WebGPUFunctionInfo) * num_func);
    (*webgpuModule)->num_functions = num_func;
    for (size_t fid = 0; fid < num_func; ++fid) {
        /* Direct assignment — signature now matches TVMFFISafeCallType (see
         * tvm_compat.h). No cast required; the -Wcast-function-type-mismatch
         * error from the pre-0.25 6-arg version is now unreachable. */
        (*webgpuModule)->functions[fid].exec = TVM_RT_WASM_WebGPUWrappedFunction;
    }
}

/*
 * TVM 0.25 WebGPU module envelope (see apache/tvm
 * web/emcc/webgpu_runtime.cc::WebGPUModuleLoadFromBytes and
 * include/tvm/support/serializer.h):
 *
 *   ffi::Map<String, FunctionInfo> fmap  -- fmap first
 *   std::unordered_map<string, string> smap
 *
 * Wire format for Map<K,V>:
 *   u64 count
 *   for i in [0, count):
 *       Serializer<K>::Write(...)      -- String: u64 len + bytes
 *       Serializer<V>::Write(...)      -- FunctionInfo:
 *           String name                    -- u64 len + bytes
 *           Array<DLDataType> arg_types    -- u64 len + count * { u8, u8, u16 }
 *           Array<String> launch_param_tags -- u64 len + count * String
 *           Array<ArgExtraTags> arg_extra_tags -- u64 len + count * i32
 *
 * Wire format for unordered_map<K,V> (serialized as vector<pair<K,V>>):
 *   u64 count
 *   for i in [0, count):
 *       string key    -- u64 len + bytes
 *       string value  -- u64 len + bytes
 *
 * The fork's pre-0.20 envelope shape (u64 func_map_size + PARSE_FUNC_INFO
 * entries with inline func_arg_index_map + u64 source_map_size + pairs)
 * is gone. Same envelope-drift class documented in §6.5.3 for the CPU
 * library-bin; this rewrite handles the WebGPU-module-body variant.
 */
int TVM_RT_WASM_WebGPUModuleCreate(BinaryReader *reader, Module **out) {
    *out = NULL;
    const char *cur_ptr;
    int status = -1;

    /* ---- fmap: Map<String, FunctionInfo> ---- */
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
    size_t func_map_size = (size_t) * (uint64_t *)cur_ptr;

    TVM_RT_WASM_WebGPUModuleAllocate((WebGPUModule **)out, func_map_size);
    WebGPUModule *webgpu_module = *(WebGPUModule **)out;
    WebGPUFunctionInfo *func_info_list = webgpu_module->functions;

    for (size_t fid = 0; fid < func_map_size; ++fid) {
        WebGPUFunctionInfo *info = func_info_list + fid;

        /* Map key: String name. Register the trie entry pointing to
         * this WebGPUFunctionInfo so GetFunction dispatches to it. */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t key_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, key_size, fail_label);
        TVM_RT_WASM_TrieInsertWithLen(webgpu_module->module_funcs_map,
                                      (const uint8_t *)cur_ptr, key_size, info);

        /* FunctionInfo.name (String). Skipped -- same as key by construction. */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t name_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, name_size, fail_label);

        /* FunctionInfo.arg_types (Array<DLDataType>) -- count == num_kernel_args.
         *
         * Split into handle vs POD scalars by dtype code (kDLOpaqueHandle=3
         * → tensor arg → storage-buffer binding; anything else → scalar arg
         * → PODArgs uniform slot). Preserve arg-order in arg_class so the
         * wrapper can route incoming TVMFFIAny slots back to the right
         * class at dispatch time. */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t num_kernel_arg = (size_t) * (uint64_t *)cur_ptr;
        info->num_kernel_args = (uint32_t)num_kernel_arg;
        /* kernel_arg_storages holds handle args only; sized max at num_kernel_arg
         * (we don't know the split yet — resized-in-place after arg_types parse). */
        info->kernel_arg_storages =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(void *) * (num_kernel_arg > 0 ? num_kernel_arg : 1));
        info->arg_class = TVM_RT_WASM_HeapMemoryAlloc(
            sizeof(uint32_t) * (num_kernel_arg > 0 ? num_kernel_arg : 1));
        /* Each DLDataType is 4 bytes on wire (u8 code, u8 bits, u16 lanes). */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(DLDataType) * num_kernel_arg,
                                          fail_label);
        const DLDataType *dtypes_wire = (const DLDataType *)cur_ptr;
        info->num_handle_args = 0;
        info->num_pod_args = 0;
        for (size_t i = 0; i < num_kernel_arg; ++i) {
            if (dtypes_wire[i].code == kDLOpaqueHandle) {
                info->arg_class[i] = 1u;
                ++info->num_handle_args;
            } else {
                info->arg_class[i] = 0u;
                ++info->num_pod_args;
            }
        }
        /* Copy POD dtypes into a compact array (indexed 0..num_pod_args-1
         * in encounter order — the same order pod_values gets packed at
         * dispatch time, and the same order TVM's WGSL PODArgs struct
         * expects the field lanes). */
        if (info->num_pod_args > 0) {
            info->pod_arg_dtypes =
                TVM_RT_WASM_HeapMemoryAlloc(sizeof(DLDataType) * info->num_pod_args);
            uint32_t p = 0;
            for (size_t i = 0; i < num_kernel_arg; ++i) {
                if (info->arg_class[i] == 0u) {
                    info->pod_arg_dtypes[p++] = dtypes_wire[i];
                }
            }
        }

        /* FunctionInfo.launch_param_tags (Array<String>) -- carries block/grid
         * axis assignments in the pre-0.25 fork format's "func_arg_index_map"
         * slot. Same string tags as before ("blockIdx.x", "threadIdx.y",
         * "tir.use_dyn_shared_memory", "paramWriteAccess:..."). */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t mp_size = (size_t) * (uint64_t *)cur_ptr;
        info->num_func_arg_map = (uint32_t)mp_size;
        info->func_arg_index_map = TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint32_t) * mp_size);
        for (size_t i = 0; i < mp_size; ++i) {
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
            size_t tag_size = (size_t) * (uint64_t *)cur_ptr;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, tag_size, fail_label);

            /* After the second BinaryCheckReadOrGoto above, cur_ptr already
             * points AT the tag bytes (the macro sets cur_ptr = old reader
             * position, then advances the reader). The earlier version of
             * these branches subtracted tag_size from cur_ptr — which
             * pointed BEFORE the preceding u64 length field, so every
             * memcmp failed and every tag fell through to the "unknown
             * tag" error even though the bytes were valid. */
            if (tag_size == 25 &&
                memcmp(cur_ptr, "tir.use_dyn_shared_memory", 25) == 0) {
                if (unlikely(i + 1 != mp_size)) {
                    TVM_RT_SET_ERROR_AND_GOTO(
                        fail_label,
                        "WebGPU launch_param_tags: tir.use_dyn_shared_memory must be last.\n");
                }
                --info->num_func_arg_map;
                info->use_dyn_mem = 1;
            } else if (tag_size > 17 &&
                       memcmp(cur_ptr, "paramWriteAccess:", 17) == 0) {
                /* `paramWriteAccess:[1,0,0,0]` — one integer per handle
                 * arg (1 = writable storage buffer, 0 = read-only). Wire
                 * these through WGPU_FunctionCreate so the bind-group
                 * layout matches the WGSL `var<storage, read[_write]>`
                 * declarations. Previously ignored → Dawn's pipeline
                 * validation silently rejected the pipeline (all-storage
                 * layout vs read-only shader decls) and the dispatch
                 * wrote nothing. */
                if (info->handle_write_access) {
                    TVM_RT_SET_ERROR_AND_GOTO(
                        fail_label,
                        "WebGPU launch_param_tags: duplicate paramWriteAccess.\n");
                }
                /* Count commas + 1 to size the array. Bracketed form. */
                const char *body = cur_ptr + 17;
                size_t body_len = tag_size - 17;
                if (body_len < 2 || body[0] != '[' || body[body_len - 1] != ']') {
                    TVM_RT_SET_ERROR_AND_GOTO(
                        fail_label,
                        "WebGPU launch_param_tags: malformed paramWriteAccess `%.*s`.\n",
                        (int)tag_size, cur_ptr);
                }
                size_t n = 1;
                for (size_t j = 1; j + 1 < body_len; ++j) {
                    if (body[j] == ',') {
                        ++n;
                    }
                }
                if (body_len == 2) {
                    n = 0; /* "[]" — empty list. */
                }
                info->handle_write_access = TVM_RT_WASM_HeapMemoryAlloc(n > 0 ? n : 1);
                size_t written = 0;
                for (size_t j = 1; j + 1 <= body_len - 1; ) {
                    /* Skip whitespace. */
                    while (j + 1 < body_len && (body[j] == ' ' || body[j] == '\t')) {
                        ++j;
                    }
                    if (j + 1 > body_len - 1) {
                        break;
                    }
                    if (body[j] < '0' || body[j] > '9') {
                        TVM_RT_SET_ERROR_AND_GOTO(
                            fail_label,
                            "WebGPU launch_param_tags: expected digit in paramWriteAccess.\n");
                    }
                    /* Single-digit 0 or 1 per the codegen convention. */
                    info->handle_write_access[written++] = (uint8_t)(body[j] - '0');
                    ++j;
                    if (j + 1 < body_len && body[j] == ',') {
                        ++j;
                    }
                }
                if (written != n) {
                    TVM_RT_SET_ERROR_AND_GOTO(
                        fail_label,
                        "WebGPU launch_param_tags: paramWriteAccess parse length mismatch.\n");
                }
                if (n != info->num_handle_args) {
                    TVM_RT_SET_ERROR_AND_GOTO(
                        fail_label,
                        "WebGPU launch_param_tags: paramWriteAccess count %zu "
                        "!= num_handle_args %u.\n",
                        n, info->num_handle_args);
                }
                /* No effect on num_func_arg_map — write-access hints are
                 * not passed as runtime dispatch args by the compiled stub;
                 * they consume no slot in func_arg_index_map[]. */
                --info->num_func_arg_map;
                info->func_arg_index_map[i] = 0;
            } else if (tag_size == 10 &&
                       memcmp(cur_ptr, "blockIdx.", 9) == 0) {
                info->func_arg_index_map[i] = (uint8_t)(*(cur_ptr + 9) - 'x');
            } else if (tag_size == 11 &&
                       memcmp(cur_ptr, "threadIdx.", 10) == 0) {
                info->func_arg_index_map[i] = (uint8_t)(*(cur_ptr + 10) - 'x' + 3);
            } else {
                TVM_RT_SET_ERROR_AND_GOTO(
                    fail_label, "WebGPU launch_param_tags: unknown tag `%.*s`\n",
                    (int)tag_size, cur_ptr);
            }
        }

        /* FunctionInfo.arg_extra_tags (Array<ArgExtraTags>) -- TVM 0.25
         * added this field (kNone / kTensorMap per tensor arg). Not used
         * by the WebGPU dispatch path; skip past. Underlying enum is
         * `int` (typically 32-bit). */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t extra_tags_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int32_t) * extra_tags_size,
                                          fail_label);
    }

    /* ---- smap: unordered_map<string, string> (vector<pair<string,string>>) ---- */
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
    size_t source_map_size = (size_t) * (uint64_t *)cur_ptr;
    if (source_map_size != func_map_size) {
        TVM_RT_SET_ERROR_AND_GOTO(fail_label,
                                  "Invalid module: function size (%zu) != source size (%zu)\n",
                                  func_map_size, source_map_size);
    }

    DeviceAPI *webgpu_dev_api = NULL;
    status = TVM_RT_WASM_DeviceAPIGet(kDLWebGPU, &webgpu_dev_api);
    if (unlikely(status)) {
        goto fail_label;
    }
    WGPU_Device gpu_device = (WGPU_Device)webgpu_dev_api->GetStream();

    for (size_t fid = 0; fid < source_map_size; ++fid) {
        /* key: entry-point name. `TVM_RT_WASM_BinaryCheckReadOrGoto` sets
         * cur_ptr to the pre-advance reader position, i.e. AT the payload
         * bytes — never subtract the length back off (same class of bug
         * that the launch_param_tags loop fixed in commit 45e2c1c). */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t name_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, name_size, fail_label);
        const char *entry_name = cur_ptr;

        WebGPUFunctionInfo *matched = NULL;
        int query_status = TVM_RT_WASM_TrieQueryWithLen(
            webgpu_module->module_funcs_map, (const uint8_t *)entry_name, name_size,
            (void **)&matched);
        if (unlikely(query_status != 0 || matched == NULL)) {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(fail_label,
                                      "WebGPU source map key `%.*s` not in fmap.\n",
                                      (int)name_size, entry_name);
        }

        /* value: WGSL source. Same pointer-arithmetic caveat as above. */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t src_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, src_size, fail_label);
        const char *src_bytes = cur_ptr;

        status = WGPU_FunctionCreate(gpu_device, &matched->device_func, src_bytes,
                                     (uint32_t)src_size, entry_name, (uint32_t)name_size,
                                     matched->num_handle_args, matched->handle_write_access,
                                     matched->num_pod_args, matched->pod_arg_dtypes);
        if (unlikely(status)) {
            goto fail_label;
        }
    }

    return 0;
fail_label:
    if (*out) {
        TVM_RT_WASM_WebGPUModuleReleaseFunc(*out);
        *out = NULL;
    }
    return status;
}
