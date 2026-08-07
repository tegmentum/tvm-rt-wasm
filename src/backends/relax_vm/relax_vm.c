/**
 * @file relax_vm/relax_vm.c
 * @brief The implementation for relax_vm public api.
 */

#include <module/module.h>
#include <relax_vm/relax_vm.h>
#include <utils/tensor_helper.h>

#define CHECK_RelaxVirtualMachine(vm) CHECK_INPUT_POINTER(vm, -2, "RelaxVirtualMachine")

/**
 * @brief Copy the input tensor to relax VM register.
 *
 * When src_tensor and dst_device live on the same device, the register is
 * set to hold a bare DLTensor handle whose lifetime is owned by the caller.
 *
 * When they differ (e.g. CPU input feeding a WebGPU kernel), a new
 * RelaxVMRegisterManagedDLTensor is created that owns:
 *   - a heap-allocated int64[ndim] shape (should_free_shape = true), and
 *   - a device-side data buffer allocated via TVMDeviceAllocDataSpace on
 *     dst_device (should_free_storage = true).
 * The source contents are then copied into the destination buffer via
 * TVMDeviceCopyDataFromTo. Both allocations are released when the register
 * is next overwritten or when the VM tears down, via the standard
 * TVM_RT_WASM_RelaxVMRegisterFreeManagedDLTensor path.
 *
 * @return 0 if successful.
 */
static int TVM_RT_WASM_RelaxVM_CopyTensorToRegister(const DLTensor *src_tensor,
                                                    RelaxVMRegister *dst_reg, DLDevice dst_device,
                                                    bool deep_copy_shape) {
    (void)deep_copy_shape;

    // The devices are same, just set register as a DLTensor handle.
    if (dst_device.device_type == src_tensor->device.device_type &&
        (dst_device.device_type == kDLCPU ||
         dst_device.device_id == src_tensor->device.device_id)) {
        TVM_RT_WASM_RelaxVMRegisterFreeValue(*dst_reg);
        dst_reg->typecode = RelaxVMRegType_DLTensorHandle;
        dst_reg->value.v_handle = (void *)src_tensor;
        return 0;
    }

    // Cross-device copy: fully materialise a destination tensor on dst_device
    // before invoking TVMDeviceCopyDataFromTo (which requires both sides to
    // have shape/dtype/device populated and destination storage allocated).
    TVM_RT_WASM_RelaxVMRegisterFreeValue(*dst_reg);

    int ndim = src_tensor->ndim;
    if (unlikely(ndim < 0)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid src_tensor ndim: %d", ndim);
    }

    RelaxVMRegisterManagedDLTensor *managed_tensor;
    TVM_RT_WASM_RelaxVMRegisterCreateManagedDLTensor(managed_tensor);
    memset(&managed_tensor->dl_tensor, 0, sizeof(managed_tensor->dl_tensor));
    managed_tensor->shape_obj = NULL;
    managed_tensor->storage_obj = NULL;
    managed_tensor->should_free_shape = false;
    managed_tensor->should_free_storage = false;

    // Owned copy of the source shape.
    size_t shape_slots = (size_t)(ndim > 0 ? ndim : 1);
    int64_t *dst_shape = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * shape_slots);
    if (unlikely(dst_shape == NULL)) {
        TVM_RT_WASM_HeapMemoryFree(managed_tensor);
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot allocate destination shape buffer.");
    }
    if (ndim > 0) {
        memcpy(dst_shape, src_tensor->shape, sizeof(int64_t) * (size_t)ndim);
    }

    managed_tensor->dl_tensor.device = dst_device;
    managed_tensor->dl_tensor.ndim = ndim;
    managed_tensor->dl_tensor.dtype = src_tensor->dtype;
    managed_tensor->dl_tensor.shape = dst_shape;
    managed_tensor->dl_tensor.strides = NULL;
    managed_tensor->dl_tensor.byte_offset = 0;
    managed_tensor->should_free_shape = true;

    // Allocate destination storage on dst_device.
    size_t nbytes = TVM_RT_WASM_DLTensor_GetDataBytes(dst_shape, ndim, src_tensor->dtype);
    void *dst_data = NULL;
    int status = TVMDeviceAllocDataSpace(dst_device, nbytes, (size_t)(1 << DATA_ALIGNMENT_BITS),
                                         src_tensor->dtype, &dst_data);
    if (unlikely(status != 0 || dst_data == NULL)) {
        TVM_RT_WASM_HeapMemoryFree(dst_shape);
        TVM_RT_WASM_HeapMemoryFree(managed_tensor);
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot allocate %zu bytes on device_type=%d.", nbytes,
                                (int)dst_device.device_type);
    }
    managed_tensor->dl_tensor.data = dst_data;
    managed_tensor->should_free_storage = true;

    dst_reg->typecode = RelaxVMRegType_ManagedDLTensor;
    dst_reg->value.v_handle = managed_tensor;

    status = TVMDeviceCopyDataFromTo((DLTensor *)src_tensor, &managed_tensor->dl_tensor, NULL);
    if (unlikely(status != 0)) {
        TVM_RT_WASM_RelaxVMRegisterFreeValue(*dst_reg);
        return status;
    }
    return 0;
}

static TVM_RT_WASM_RelaxVirtualMachine
TVM_RT_WASM_RelaxVirtualMachineCreateImpl(TVMModuleHandle module_handle, const char *sys_lib_prefix,
                                          const DLDevice *devices, uint32_t num_dev) {
    CHECK_INPUT_POINTER(devices, NULL, "Devices");
    if (unlikely(num_dev == 0)) {
        TVM_RT_SET_ERROR_RETURN(
            NULL, "Invalid argument: the number of devices cannot be zero, at least 1.");
    }

    Module *module = (Module *)module_handle;
    if (module == NULL) {
        int status =
            TVM_RT_WASM_SystemLibraryModuleCreateWithPrefix(sys_lib_prefix, &module);
        if (unlikely(status)) {
            return NULL;
        }
    }
    RelaxExecutableModule *exec_module = (RelaxExecutableModule *)module;

    // set up the packed functions
    int status = TVM_RT_WASM_RelaxVMRegisterBuiltinGlobalFunctions();
    if (unlikely(status)) {
        return NULL;
    }
    char *name_buffer = NULL;
    size_t name_buffer_len = 0;
    for (size_t i = 0; i < exec_module->exec.num_relex_functions; ++i) {
        RelaxFunctionInfo *func = exec_module->exec.relax_functions + i;
        if (func->type == RelaxFuncType_Packed) {
            name_buffer_len = MAX(name_buffer_len, func->packed_func.name_size);
        }
    }
    if (name_buffer_len != 0) {
#define SPECIAL_PACKED_FUNC_NAME "vm.builtin.null_value"
        name_buffer = TVM_RT_WASM_WorkplaceMemoryAlloc(name_buffer_len + 1);
        for (size_t i = 0; i < exec_module->exec.num_relex_functions; ++i) {
            RelaxFunctionInfo *func = exec_module->exec.relax_functions + i;
            if (func->type == RelaxFuncType_Packed) {
                // The special empty packed function.
                if (memcmp(func->packed_func.name_ptr, SPECIAL_PACKED_FUNC_NAME,
                           sizeof(SPECIAL_PACKED_FUNC_NAME) - 1) == 0) {
                    func->packed_func.pf = NULL;
                    continue;
                }
                memcpy(name_buffer, func->packed_func.name_ptr, func->packed_func.name_size);
                name_buffer[func->packed_func.name_size] = 0;
                status = exec_module->GetFunction(module, name_buffer, 1, &func->packed_func.pf);
                if (status) {
                    status =
                        TVMFuncGetGlobal(name_buffer, (TVMFunctionHandle)&func->packed_func.pf);
                }
                if (unlikely(status)) {
                    TVM_RT_SET_ERROR("Cannot find packed function `%s` from module.", name_buffer);
                    TVM_RT_WASM_WorkplaceMemoryFree(name_buffer);
                    return NULL;
                }
            }
        }
        TVM_RT_WASM_WorkplaceMemoryFree(name_buffer);
    }
    // set up the constants
    RelaxVMRegister *constants =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMRegister) * exec_module->exec.num_constants);
    RelaxConstant *relax_constants = exec_module->exec.constants;
    for (size_t i = 0; i < exec_module->exec.num_constants; ++i) {
        switch (relax_constants[i].type) {
#if TENSOR_DATA_MUST_ALIGN
        case RelaxConstantType_DLTensorShouldFree:
#endif // TENSOR_DATA_MUST_ALIGN
        case RelaxConstantType_DLTensor: {
            const DLTensor *src_tensor = &relax_constants[i].dl_tensor;
            constants[i].typecode = RelaxVMRegType_Nullptr;
            status = TVM_RT_WASM_RelaxVM_CopyTensorToRegister(src_tensor, constants + i, devices[0],
                                                              false);
            if (unlikely(status)) {
                // free the created constants.
                for (size_t j = 0; j < i; ++j) {
                    TVM_RT_WASM_RelaxVMRegisterFreeValue(constants[i]);
                }
                TVM_RT_WASM_HeapMemoryFree(constants);
                return NULL;
            }
            break;
        }
        case RelaxConstantType_DLDataType:
            constants[i].typecode = RelaxVMRegType_DataType;
            constants[i].value.v_type = relax_constants[i].dl_datatype;
            break;
        case RelaxConstantType_ShapeTuple:
            constants[i].typecode = RelaxVMRegType_VMObjectShapeTuple;
            constants[i].value.v_handle = &relax_constants[i].register_obj;
            break;
        case RelaxConstantType_String:
            constants[i].typecode = RelaxVMRegType_VMObjectString;
            constants[i].value.v_handle = &relax_constants[i].register_obj.ref_num;
            break;
        case RelaxConstantType_Int:
            constants[i].typecode = RelaxVMRegType_Int;
            constants[i].value.v_int64 = relax_constants[i].int_value;
            break;
        }
    }

    TVM_RT_WASM_RelaxVirtualMachine vm =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct TVM_RT_WASM_RelaxVirtualMachine_st));
    memset(vm, 0, sizeof(struct TVM_RT_WASM_RelaxVirtualMachine_st));

    vm->constants = constants;
    vm->call_packed_args_typecode =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(int) * exec_module->exec.max_num_call_args);
    vm->call_packed_args_value =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVMValue) * exec_module->exec.max_num_call_args);
    vm->num_device = num_dev;
    vm->devices = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DLDevice) * vm->num_device);
    memcpy(vm->devices, devices, sizeof(DLDevice) * vm->num_device);
    TVM_RT_WASM_TrieCreate(&vm->func_inputs_output_map);

    vm->exec_module = exec_module;
    return vm;
}

TVM_RT_WASM_RelaxVirtualMachine TVM_RT_WASM_RelaxVirtualMachineCreate(TVMModuleHandle module_handle,
                                                                      const DLDevice *devices,
                                                                      uint32_t num_dev) {
    return TVM_RT_WASM_RelaxVirtualMachineCreateImpl(module_handle, "", devices, num_dev);
}

TVM_RT_WASM_RelaxVirtualMachine
TVM_RT_WASM_RelaxVirtualMachineCreateWithPrefix(const char *prefix, const DLDevice *devices,
                                                uint32_t num_dev) {
    return TVM_RT_WASM_RelaxVirtualMachineCreateImpl(NULL, prefix, devices, num_dev);
}

/** @brief The function to free vm's function inputs/outputs registers. */
static void TVM_RT_WASM_TrieVisit_FreeInputsOutput(void **data_ptr, void *source_handle) {
    (void)source_handle;
    RelaxVMFunctionInputsOutput *p = *data_ptr;
    for (size_t i = 0; i <= p->num_inputs; ++i) {
        TVM_RT_WASM_RelaxVMRegisterFreeValue(p->inputs_output[i]);
    }
    TVM_RT_WASM_HeapMemoryFree(p->inputs_output);
    TVM_RT_WASM_HeapMemoryFree(p);
}

int TVM_RT_WASM_RelaxVirtualMachineFree(TVM_RT_WASM_RelaxVirtualMachine vm) {
    CHECK_RelaxVirtualMachine(vm);
    if (vm->call_packed_args_typecode) {
        TVM_RT_WASM_HeapMemoryFree(vm->call_packed_args_typecode);
    }
    if (vm->call_packed_args_value) {
        TVM_RT_WASM_HeapMemoryFree(vm->call_packed_args_value);
    }
    if (vm->constants) {
        for (size_t i = 0; i < vm->exec_module->exec.num_constants; ++i) {
            // only free the managed tensor.
            if (vm->constants[i].typecode == RelaxVMRegType_ManagedDLTensor) {
                RelaxVMRegisterManagedDLTensor *t = vm->constants[i].value.v_handle;
                TVM_RT_WASM_RelaxVMRegisterFreeManagedDLTensor(t);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(vm->constants);
    }
    if (vm->frames) {
        for (size_t i = 0; i < vm->frame_capacity; ++i) {
            RelaxVMFrame *frame = vm->frames + i;
            if (frame->registers) {
                for (size_t r = 0; r < frame->register_size; ++r) {
                    TVM_RT_WASM_RelaxVMRegisterFreeValue(frame->registers[r]);
                }
                TVM_RT_WASM_HeapMemoryFree(frame->registers);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(vm->frames);
    }
    if (vm->func_inputs_output_map) {
        TVM_RT_WASM_TrieVisit(vm->func_inputs_output_map, TVM_RT_WASM_TrieVisit_FreeInputsOutput,
                              NULL);
        TVM_RT_WASM_TrieRelease(vm->func_inputs_output_map);
    }
    if (vm->exec_module) {
        vm->exec_module->Release((Module *)vm->exec_module);
    }
    if (vm->devices) {
        TVM_RT_WASM_HeapMemoryFree(vm->devices);
    }
    TVM_RT_WASM_HeapMemoryFree(vm);
    return 0;
}

#define TVM_RT_WASM_RelaxVMGetAndCheckFunc(_vm, _func_name)                                        \
    RelaxFunctionInfo *func;                                                                       \
    do {                                                                                           \
        if ((_func_name) == NULL) {                                                                \
            (_func_name) = TVM_RT_WASM_RelaxDefaultFunctionName;                                   \
        }                                                                                          \
        if (unlikely(TVM_RT_WASM_TrieQuery((_vm)->exec_module->exec.relax_vm_functions_map,        \
                                           (const uint8_t *)(_func_name),                          \
                                           (void **)&func) != TRIE_SUCCESS)) {                     \
            TVM_RT_SET_ERROR_RETURN(-1, "Cannot find function `%s`", _func_name);                  \
        }                                                                                          \
    } while (0)

#define TVM_RT_WASM_RelaxVMFuncInputsOutputGetOrCreate(_vm, _func, _func_name)                     \
    RelaxVMFunctionInputsOutput *inputs_output = NULL;                                             \
    do {                                                                                           \
        if (unlikely(TVM_RT_WASM_TrieQuery((_vm)->func_inputs_output_map,                          \
                                           (const uint8_t *)(_func_name),                          \
                                           (void **)&inputs_output))) {                            \
            inputs_output = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMFunctionInputsOutput));      \
            inputs_output->num_inputs = (_func)->vm_func.num_params;                               \
            inputs_output->inputs_output = TVM_RT_WASM_HeapMemoryAlloc(                            \
                sizeof(RelaxVMRegister) * (inputs_output->num_inputs + 1));                        \
            for (size_t i = 0; i <= inputs_output->num_inputs; ++i) {                              \
                inputs_output->inputs_output[i].typecode = RelaxVMRegType_Nullptr;                 \
            }                                                                                      \
            TVM_RT_WASM_TrieInsert((_vm)->func_inputs_output_map, (const uint8_t *)(_func_name),   \
                                   inputs_output);                                                 \
        }                                                                                          \
    } while (0)

int TVM_RT_WASM_RelaxVirtualMachineRun(TVM_RT_WASM_RelaxVirtualMachine vm, const char *func_name) {
    CHECK_RelaxVirtualMachine(vm);
    TVM_RT_WASM_RelaxVMGetAndCheckFunc(vm, func_name);
    TVM_RT_WASM_RelaxVMFuncInputsOutputGetOrCreate(vm, func, func_name);
    for (size_t i = 0; i < inputs_output->num_inputs; ++i) {
        if (inputs_output->inputs_output[i].typecode == RelaxVMRegType_Nullptr) {
            TVM_RT_SET_ERROR_RETURN(-2, "The function `%s` input index %zu has not been set now.",
                                    func_name, i);
        }
    }
    return TVM_RT_WASM_RelaxVMRunFunction(vm, func, inputs_output);
}

int TVM_RT_WASM_RelaxVirtualMachineSetInput(TVM_RT_WASM_RelaxVirtualMachine vm,
                                            const char *func_name, uint32_t index,
                                            const DLTensor *data_in) {
    CHECK_RelaxVirtualMachine(vm);
    CHECK_INPUT_POINTER(data_in, -2, "DLTensor");
    TVM_RT_WASM_RelaxVMGetAndCheckFunc(vm, func_name);
    CHECK_INDEX_RANGE((uint32_t)func->vm_func.num_params, index);
    TVM_RT_WASM_RelaxVMFuncInputsOutputGetOrCreate(vm, func, func_name);
    return TVM_RT_WASM_RelaxVM_CopyTensorToRegister(data_in, inputs_output->inputs_output + index,
                                                    vm->devices[0], true);
}

int TVM_RT_WASM_RelaxVirtualMachineSetInputByName(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                  const char *func_name, const char *name,
                                                  const DLTensor *data_in) {
    CHECK_RelaxVirtualMachine(vm);
    CHECK_INPUT_POINTER(name, -2, "Name");
    CHECK_INPUT_POINTER(data_in, -2, "DLTensor");
    TVM_RT_WASM_RelaxVMGetAndCheckFunc(vm, func_name);
    uintptr_t index;
    int status =
        TVM_RT_WASM_TrieQuery(func->vm_func.params_map, (const uint8_t *)name, (void **)&index);
    if (unlikely(status != TRIE_SUCCESS)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot find function param `%s`", name);
    }
    TVM_RT_WASM_RelaxVMFuncInputsOutputGetOrCreate(vm, func, func_name);
    return TVM_RT_WASM_RelaxVM_CopyTensorToRegister(data_in, inputs_output->inputs_output + index,
                                                    vm->devices[0], true);
}

int TVM_RT_WASM_RelaxVirtualMachineGetOutput(TVM_RT_WASM_RelaxVirtualMachine vm,
                                             const char *func_name, uint32_t index,
                                             DLTensor *data_out) {
    CHECK_RelaxVirtualMachine(vm);
    CHECK_INPUT_POINTER(data_out, -2, "DLTensor");
    TVM_RT_WASM_RelaxVMGetAndCheckFunc(vm, func_name);
    TVM_RT_WASM_RelaxVMFuncInputsOutputGetOrCreate(vm, func, func_name);
    const RelaxVMRegister *output_reg = inputs_output->inputs_output + inputs_output->num_inputs;
    DLTensor *src_tensor;
    if (output_reg->typecode == RelaxVMRegType_ManagedDLTensor ||
        output_reg->typecode == RelaxVMRegType_DLTensorHandle) {
        CHECK_INDEX_RANGE(1, index);
        src_tensor = output_reg->value.v_handle;
    } else if (output_reg->typecode == RelaxVMRegType_VMObjectTuple) {
        RelaxVMRegisterObject *tuple = output_reg->value.v_handle;
        CHECK_INDEX_RANGE((uint32_t)tuple->tuple.size, index);
        RelaxVMRegisterTypeCode typecode = tuple->tuple.ptr[index].typecode;
        if (typecode != RelaxVMRegType_ManagedDLTensor &&
            typecode != RelaxVMRegType_DLTensorHandle) {
            TVM_RT_SET_ERROR_RETURN(-1, "Relax VM output %u is not tensor.", index);
        }
        src_tensor = tuple->tuple.ptr[index].value.v_handle;
    } else {
        TVM_RT_SET_ERROR_RETURN(-1, "Relax VM has no outputs now.");
    }
    return TVMDeviceCopyDataFromTo(src_tensor, data_out, NULL);
}

int TVM_RT_WASM_RelaxVirtualMachineGetOutputShape(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                  const char *func_name, uint32_t index,
                                                  int32_t *out_ndim, DLDataType *out_dtype,
                                                  int64_t *out_shape, int32_t shape_capacity) {
    CHECK_RelaxVirtualMachine(vm);
    CHECK_INPUT_POINTER(out_ndim, -2, "out_ndim");
    TVM_RT_WASM_RelaxVMGetAndCheckFunc(vm, func_name);
    TVM_RT_WASM_RelaxVMFuncInputsOutputGetOrCreate(vm, func, func_name);
    const RelaxVMRegister *output_reg = inputs_output->inputs_output + inputs_output->num_inputs;
    DLTensor *src_tensor;
    if (output_reg->typecode == RelaxVMRegType_ManagedDLTensor ||
        output_reg->typecode == RelaxVMRegType_DLTensorHandle) {
        CHECK_INDEX_RANGE(1, index);
        src_tensor = output_reg->value.v_handle;
    } else if (output_reg->typecode == RelaxVMRegType_VMObjectTuple) {
        RelaxVMRegisterObject *tuple = output_reg->value.v_handle;
        CHECK_INDEX_RANGE((uint32_t)tuple->tuple.size, index);
        RelaxVMRegisterTypeCode typecode = tuple->tuple.ptr[index].typecode;
        if (typecode != RelaxVMRegType_ManagedDLTensor &&
            typecode != RelaxVMRegType_DLTensorHandle) {
            TVM_RT_SET_ERROR_RETURN(-1, "Relax VM output %u is not tensor.", index);
        }
        src_tensor = tuple->tuple.ptr[index].value.v_handle;
    } else {
        TVM_RT_SET_ERROR_RETURN(-1, "Relax VM has no outputs now.");
    }
    *out_ndim = src_tensor->ndim;
    if (out_dtype != NULL) {
        *out_dtype = src_tensor->dtype;
    }
    if (out_shape != NULL) {
        if (shape_capacity < src_tensor->ndim) {
            TVM_RT_SET_ERROR_RETURN(-3, "Output shape capacity %d < ndim %d",
                                    (int)shape_capacity, (int)src_tensor->ndim);
        }
        for (int32_t i = 0; i < src_tensor->ndim; ++i) {
            out_shape[i] = src_tensor->shape[i];
        }
    }
    return 0;
}

/*-----------------Functions to get relax virtual machine information-----------------------------*/

int TVM_RT_WASM_RelaxVirtualMachineGetInputIndex(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                 const char *func_name, const char *name) {
    CHECK_RelaxVirtualMachine(vm);
    CHECK_INPUT_POINTER(name, -2, "Name");
    TVM_RT_WASM_RelaxVMGetAndCheckFunc(vm, func_name);
    uintptr_t index;
    int status = TVM_RT_WASM_TrieQuery(func->vm_func.params_map, (const uint8_t *)func_name,
                                       (void **)&index);
    if (unlikely(status)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot find function param `%s`", name);
    }
    return (int)index;
}

int TVM_RT_WASM_RelaxVirtualMachineGetNumInputs(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                const char *func_name) {
    CHECK_RelaxVirtualMachine(vm);
    TVM_RT_WASM_RelaxVMGetAndCheckFunc(vm, func_name);
    return (int)func->vm_func.num_params;
}

int TVM_RT_WASM_RelaxVirtualMachineGetNumOutputs(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                 const char *func_name) {
    CHECK_RelaxVirtualMachine(vm);
    TVM_RT_WASM_RelaxVMGetAndCheckFunc(vm, func_name);
    TVM_RT_WASM_RelaxVMFuncInputsOutputGetOrCreate(vm, func, func_name);
    const RelaxVMRegister *output_reg = inputs_output->inputs_output + inputs_output->num_inputs;
    if (output_reg->typecode == RelaxVMRegType_ManagedDLTensor ||
        output_reg->typecode == RelaxVMRegType_DLTensorHandle) {
        return 1;
    } else if (output_reg->typecode == RelaxVMRegType_VMObjectTuple) {
        return ((RelaxVMRegisterObject *)(output_reg->value.v_handle))->tuple.size;
    }
    return 0;
}
