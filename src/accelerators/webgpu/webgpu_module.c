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

    for (uint32_t i = 0; i < num_kernel_args; ++i) {
        info->kernel_arg_storages[i] = args_value[i].v_handle;
    }

    (void)block_dim;
    int status = WGPU_FunctionRun(info->device_func, (WGPU_Memory *)info->kernel_arg_storages,
                                  num_kernel_args, grid_dim[0], grid_dim[1], grid_dim[2]);

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

/**
 * @brief Create a WebGPU module instance from the byte stream.
 * @param reader The module binary reader.
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 */
int TVM_RT_WASM_WebGPUModuleCreate(BinaryReader *reader, Module **out) {
    *out = NULL;
    const char *cur_ptr;
    int status = -1;

    // parse function map: <string, FunctionInfo{name, arg_types, launch_params_tags} >
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
    size_t func_map_size = (size_t) * (uint64_t *)cur_ptr;

    TVM_RT_WASM_WebGPUModuleAllocate((WebGPUModule **)out, func_map_size);
    WebGPUModule *webgpu_module = *(WebGPUModule **)out;

    WebGPUFunctionInfo *func_info_list = webgpu_module->functions;
    for (size_t fid = 0; fid < func_map_size; ++fid) {
        WebGPUFunctionInfo *info = func_info_list + fid;
        PARSE_FUNC_INFO(webgpu_module, cur_ptr, fail_label);
    }

    // parse source map <string, string>
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
    // get the device
    WGPU_Device gpu_device = (WGPU_Device)webgpu_dev_api->GetStream();

    for (size_t fid = 0; fid < source_map_size; ++fid) {
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        // key: name
        size_t name_size = (size_t) * (uint64_t *)cur_ptr;
        // skip name, (equal to function names)
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, name_size, fail_label);

        // key: source
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t src_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, src_size, fail_label);
        status = WGPU_FunctionCreate(gpu_device, &func_info_list[fid].device_func, cur_ptr,
                                     src_size, NULL, 0, func_info_list[fid].num_kernel_args);
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
