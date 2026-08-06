/**
 * @file relay_vm/vm_builtin.c
 * @brief The Relax builtin global functions.
 */

#include <string.h>

#include <device/device_api.h>
#include <module/module.h>
#include <relax_vm/relax_vm.h>
#include <tvm_compat.h>
#include <utils/common.h>
#include <utils/tensor_helper.h>

// Define the relax vm builtin function name prefix.
#define RELAX_VM_BUILTIN_FUNC_NAME(_name_suffix) TVM_RT_WASM_RelaxVM_Builtin_##_name_suffix

/*
 * TVM 0.25 dispatch convention (TVMFFISafeCallType):
 *   int (*)(void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result)
 *
 * The old fork built-ins used the pre-tvm-ffi 6-arg convention with
 * parallel typecode arrays. The tag now lives on each `TVMFFIAny`
 * (`args_value[i].type_index`) instead of a side `args_typecode` array,
 * and the return typecode lives on `ret_value->type_index`.
 *
 * Relax VM built-ins care about the fork-internal `RelaxVMRegisterTypeCode`
 * namespace (which includes tags outside the FFI type-index range, e.g.
 * `RelaxVMRegType_VMObjectStorage = 1 | (1 << 9)`). The runner packs
 * that register-typecode straight into `TVMFFIAny::type_index`; that value
 * fits in `int32_t` and does not collide with any real FFI static type
 * index (<= 68) so it round-trips lossless.
 */
#define RELAX_VM_FUNC(_name_suffix)                                                                \
    static int RELAX_VM_BUILTIN_FUNC_NAME(_name_suffix)(                                           \
        void *source_handle, const TVMFFIAny *args_value, int32_t num_args,                        \
        TVMFFIAny *ret_value)

RELAX_VM_FUNC(AllocaShape) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    return 0;
}

/*
 * Match op-code for MatchShape / MakeShape family, per apache/tvm 0.25
 * relax_vm/executable.h. Kept local; only kAssertEqualToImm is exercised
 * by the encoder (all symbolic dims lower to literals under our static
 * shape-pinning).
 */
enum {
    kMatchShapeAssertEqualToImm = 0,
    kMatchShapeStoreToHeap = 1,
    kMatchShapeNoOp = 2,
    kMatchShapeAssertEqualToLoad = 3,
};

RELAX_VM_FUNC(MatchShape) {
    (void)source_handle;
    (void)ret_value;

    /*
     * Signature (per apache/tvm src/runtime/vm/builtin.cc):
     *   match_shape(input, heap, size, code[0], reg[0], ..., code[N-1], reg[N-1], err_ctx)
     *
     * input can be a Tensor / DLTensor* / Shape.
     * heap is a rank-1 int64 DLTensor* used to store extracted dims when
     * code == kStoreToHeap; may be nullptr (arrives here as kTVMFFINone).
     * Under this fork's static-shape encoder, heap is always nullptr and
     * codes are always kAssertEqualToImm.
     */
    if (unlikely(num_args < 3)) {
        TVM_RT_SET_ERROR_RETURN(-1, "vm.builtin.match_shape: too few args (%d).", num_args);
    }

    /* Recover input shape from the first arg. */
    const int64_t *input_shape;
    int input_ndim;
    int32_t input_tag = args_value[0].type_index;
    if (input_tag == (int32_t)kTVMFFIDLTensorPtr ||
        input_tag == (int32_t)RelaxVMRegType_ManagedDLTensor ||
        input_tag == (int32_t)RelaxVMRegType_DLTensorHandle) {
        DLTensor *t = (DLTensor *)args_value[0].v_handle;
        input_shape = t->shape;
        input_ndim = t->ndim;
    } else if (input_tag == (int32_t)RelaxVMRegType_VMObjectShapeTuple) {
        RelaxVMRegisterObject *s = args_value[0].v_handle;
        input_shape = s->shape_tuple.shape;
        input_ndim = s->shape_tuple.ndim;
    } else {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.match_shape: unsupported input type_index %d.", input_tag);
    }

    int64_t *heap_data = NULL;
    if (args_value[1].type_index == (int32_t)kTVMFFIDLTensorPtr) {
        DLTensor *heap = (DLTensor *)args_value[1].v_handle;
        heap_data = (int64_t *)heap->data;
    }
    /* kTVMFFINone -> no heap; leave heap_data NULL. */

    int64_t size = args_value[2].v_int64;
    if (unlikely(size != input_ndim)) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.match_shape: expected %lld dims, input has %d.",
            (long long)size, input_ndim);
    }
    if (unlikely(num_args < 3 + 2 * size + 1)) {
        TVM_RT_SET_ERROR_RETURN(-1, "vm.builtin.match_shape: arg count %d < 3+2*%lld+1.",
                                num_args, (long long)size);
    }

    for (int64_t i = 0; i < size; ++i) {
        int64_t code = args_value[3 + i * 2].v_int64;
        int64_t reg = args_value[3 + i * 2 + 1].v_int64;
        switch (code) {
        case kMatchShapeAssertEqualToImm:
            if (unlikely(input_shape[i] != reg)) {
                TVM_RT_SET_ERROR_RETURN(
                    -1, "vm.builtin.match_shape: shape[%lld] = %lld != expected %lld.",
                    (long long)i, (long long)input_shape[i], (long long)reg);
            }
            break;
        case kMatchShapeStoreToHeap:
            if (unlikely(!heap_data)) {
                TVM_RT_SET_ERROR_RETURN(
                    -1, "vm.builtin.match_shape: kStoreToHeap requires a heap tensor.");
            }
            heap_data[reg] = input_shape[i];
            break;
        case kMatchShapeNoOp:
            break;
        case kMatchShapeAssertEqualToLoad:
            if (unlikely(!heap_data)) {
                TVM_RT_SET_ERROR_RETURN(
                    -1, "vm.builtin.match_shape: kAssertEqualToLoad requires a heap tensor.");
            }
            if (unlikely(input_shape[i] != heap_data[reg])) {
                TVM_RT_SET_ERROR_RETURN(
                    -1,
                    "vm.builtin.match_shape: shape[%lld] = %lld != heap[%lld] = %lld.",
                    (long long)i, (long long)input_shape[i], (long long)reg,
                    (long long)heap_data[reg]);
            }
            break;
        default:
            TVM_RT_SET_ERROR_RETURN(
                -1, "vm.builtin.match_shape: unknown code %lld at dim %lld.",
                (long long)code, (long long)i);
        }
    }
    return 0;
}

RELAX_VM_FUNC(MakeShape) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(CheckTensorInfo) {
    (void)source_handle;
    (void)ret_value;

    /*
     * Signature (apache/tvm src/runtime/vm/builtin.cc):
     *   check_tensor_info(arg, ndim, [dtype,] err_ctx)
     *
     * ndim == -1  -> any-rank
     * dtype void  -> any-dtype (encoded as {code=0,bits=0,lanes=0})
     * arg must be a Tensor / DLTensor*; for a plain None (untyped
     * slot) we accept — the frontend only emits this for typed slots.
     */
    if (unlikely(num_args < 3 || num_args > 4)) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.check_tensor_info: bad arg count %d.", num_args);
    }

    int32_t arg_tag = args_value[0].type_index;
    if (arg_tag != (int32_t)kTVMFFIDLTensorPtr &&
        arg_tag != (int32_t)RelaxVMRegType_ManagedDLTensor &&
        arg_tag != (int32_t)RelaxVMRegType_DLTensorHandle) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.check_tensor_info: expected Tensor, got type_index %d.",
            arg_tag);
    }

    DLTensor *t = (DLTensor *)args_value[0].v_handle;
    int64_t expect_ndim = args_value[1].v_int64;
    if (expect_ndim != -1 && t->ndim != (int)expect_ndim) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.check_tensor_info: expected ndim %lld, got %d.",
            (long long)expect_ndim, t->ndim);
    }

    if (num_args == 4) {
        DLDataType dt = args_value[2].v_dtype;
        int is_void = (dt.code == 0 && dt.bits == 0 && dt.lanes == 0);
        if (!is_void) {
            if (t->dtype.code != dt.code || t->dtype.bits != dt.bits ||
                t->dtype.lanes != dt.lanes) {
                TVM_RT_SET_ERROR_RETURN(
                    -1,
                    "vm.builtin.check_tensor_info: dtype mismatch "
                    "(expected code=%u bits=%u lanes=%u, got code=%u bits=%u lanes=%u).",
                    dt.code, dt.bits, dt.lanes, t->dtype.code, t->dtype.bits,
                    t->dtype.lanes);
            }
        }
    }
    return 0;
}

RELAX_VM_FUNC(CheckShapeInfo) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(CheckTupleInfo) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(CheckFuncInfo) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(AllocStorage) {
    (void)num_args;
    (void)source_handle;

    TVM_RT_WASM_RelaxVirtualMachine vm = args_value[0].v_handle;
    RelaxVMRegisterObject *shape_tuple = args_value[1].v_handle;
    DLDataType dtype = args_value[3].v_type;
    DLDevice dev;
    void *data;

    // device index
    if (args_value[2].v_int64 == -1) {
        dev.device_type = kDLCPU;
    } else {
        dev = vm->devices[0];
    }
    size_t nbytes = TVM_RT_WASM_DLTensor_GetDataBytes(shape_tuple->shape_tuple.shape,
                                                      shape_tuple->shape_tuple.ndim, dtype);
    if (dev.device_type == kDLCPU) {
        data = TVM_RT_WASM_HeapMemoryAlignedAlloc(nbytes);
    } else {
        DeviceAPI *device_api;
        int status = TVM_RT_WASM_DeviceAPIGet(dev.device_type, &device_api);
        if (unlikely(status)) {
            return status;
        }
        data = device_api->AllocDataSpace(dev.device_id, nbytes);
    }
    if (unlikely(data == NULL)) {
        return -1;
    }
    RelaxVMRegisterObject *storage_obj;
    TVM_RT_WASM_RelaxVMRegisterCreateObject(storage_obj);

    storage_obj->storage.device = dev;
    storage_obj->storage.data = data;
    ret_value->v_handle = storage_obj;
    ret_value->type_index = (int32_t)RelaxVMRegType_VMObjectStorage;
    return 0;
}

RELAX_VM_FUNC(AllocDLTensor) {
    (void)num_args;
    (void)source_handle;

    RelaxVMRegisterObject *storage_obj = args_value[0].v_handle;
    int64_t offset = args_value[1].v_int64;
    RelaxVMRegisterObject *shape_obj = args_value[2].v_handle;
    DLDataType dtype = args_value[3].v_type;

    RelaxVMRegisterManagedDLTensor *dl_tensor;
    // set shape_obj and storage_obj
    TVM_RT_WASM_RelaxVMRegisterCreateManagedDLTensor(dl_tensor);
    dl_tensor->dl_tensor.data = storage_obj->storage.data;
    dl_tensor->dl_tensor.device = storage_obj->storage.device;
    dl_tensor->dl_tensor.ndim = shape_obj->shape_tuple.ndim;
    dl_tensor->dl_tensor.shape = shape_obj->shape_tuple.shape;
    dl_tensor->dl_tensor.dtype = dtype;
    dl_tensor->dl_tensor.strides = NULL;
    dl_tensor->dl_tensor.byte_offset = offset;

    ++storage_obj->ref_num;
    dl_tensor->storage_obj = storage_obj;
    ++shape_obj->ref_num;
    dl_tensor->shape_obj = shape_obj;

    ret_value->v_handle = dl_tensor;
    ret_value->type_index = (int32_t)RelaxVMRegType_ManagedDLTensor;
    return 0;
}

RELAX_VM_FUNC(MakeClosure) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    TVM_RT_NOT_IMPLEMENT(-1);
}

RELAX_VM_FUNC(InvokeClosure) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    TVM_RT_NOT_IMPLEMENT(-1);
}

RELAX_VM_FUNC(CallTIRDyn) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    TVM_RT_NOT_IMPLEMENT(-1);
}

RELAX_VM_FUNC(ShapeOf) {
    (void)num_args;
    (void)source_handle;

    /*
     * Input can be either a Tensor (which in this fork lives as a
     * RelaxVMRegisterManagedDLTensor* whose first field is a DLTensor
     * and reaches us tagged kTVMFFIDLTensorPtr — see the runner's tag
     * normalization in relax_vm_runner.c) or a rank-typed DLTensor
     * handle (kTVMFFIDLTensorPtr with a bare DLTensor* — same shape).
     * Both cases can be treated as (DLTensor*).
     *
     * Output is a new VMObjectShapeTuple that owns a fresh int64[] copy
     * of the tensor's shape — the source tensor's lifetime is not tied
     * to the returned Shape.
     */
    DLTensor *tensor = (DLTensor *)args_value[0].v_handle;
    int ndim = tensor->ndim;

    RelaxVMRegisterObject *shape_obj;
    TVM_RT_WASM_RelaxVMRegisterCreateObject(shape_obj);
    shape_obj->shape_tuple.ndim = ndim;
    shape_obj->shape_tuple.shape =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * (size_t)(ndim > 0 ? ndim : 1));
    if (ndim > 0) {
        memcpy(shape_obj->shape_tuple.shape, tensor->shape, sizeof(int64_t) * (size_t)ndim);
    }

    ret_value->v_handle = shape_obj;
    ret_value->type_index = (int32_t)RelaxVMRegType_VMObjectShapeTuple;
    return 0;
}

RELAX_VM_FUNC(Copy) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(Reshape) {
    (void)num_args;
    (void)source_handle;

    RelaxVMRegisterObject *shape_obj = args_value[1].v_handle;
    RelaxVMRegisterManagedDLTensor *dl_tensor;
    // set shape_obj and storage_obj and should_free_storage (if storage_obj is NULL)
    TVM_RT_WASM_RelaxVMRegisterCreateManagedDLTensor(dl_tensor);

    if ((int32_t)args_value[0].type_index == RelaxVMRegType_ManagedDLTensor) {
        RelaxVMRegisterManagedDLTensor *src_tensor = args_value[0].v_handle;
        dl_tensor->dl_tensor = src_tensor->dl_tensor;
        dl_tensor->storage_obj = src_tensor->storage_obj;
        if (dl_tensor->storage_obj) {
            ++dl_tensor->storage_obj->ref_num;
        } else {
            dl_tensor->storage_obj = NULL;
            dl_tensor->should_free_storage = false;
        }
    } else if ((int32_t)args_value[0].type_index == RelaxVMRegType_DLTensorHandle) {
        dl_tensor->dl_tensor = *((DLTensor *)(args_value[0].v_handle));
        dl_tensor->storage_obj = NULL;
        dl_tensor->should_free_storage = false;
    }

    dl_tensor->dl_tensor.shape = shape_obj->shape_tuple.shape;
    dl_tensor->dl_tensor.ndim = shape_obj->shape_tuple.ndim;
    ++shape_obj->ref_num;
    dl_tensor->shape_obj = shape_obj;

    ret_value->v_handle = dl_tensor;
    ret_value->type_index = (int32_t)RelaxVMRegType_ManagedDLTensor;
    return 0;
}

RELAX_VM_FUNC(ReadIfCond) {
    (void)args_value;
    (void)num_args;
    (void)ret_value;
    (void)source_handle;
    return 0;
}

/*
 * Reconstruct the internal register typecode from a TVMFFIAny arg. The
 * runner normalises RelaxVMRegType_ManagedDLTensor to kTVMFFIDLTensorPtr
 * when packing kernel/builtin call args (see relax_vm_runner.c); when
 * that value round-trips into a tuple element we must undo the mapping
 * so refcount / free logic that pivots on RelaxVMRegType_ManagedDLTensor
 * continues to fire.
 */
static RelaxVMRegisterTypeCode TVM_RT_WASM_RelaxVM_FFIArgToRegTypeCode(int32_t ffi_tag) {
    if (ffi_tag == (int32_t)kTVMFFIDLTensorPtr) {
        return RelaxVMRegType_ManagedDLTensor;
    }
    return (RelaxVMRegisterTypeCode)ffi_tag;
}

RELAX_VM_FUNC(TupleGetItem) {
    (void)source_handle;

    /*
     * tuple_getitem(tuple, index) -> element.
     * Element is copied out with the fork's ref-count-aware macro so the
     * caller register owns a real reference.
     */
    if (unlikely(num_args != 2)) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.tuple_getitem: expected 2 args, got %d.", num_args);
    }
    if (unlikely(args_value[0].type_index != (int32_t)RelaxVMRegType_VMObjectTuple)) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.tuple_getitem: first arg is not a tuple (type_index=%d).",
            args_value[0].type_index);
    }
    RelaxVMRegisterObject *tup = args_value[0].v_handle;
    int64_t idx = args_value[1].v_int64;
    if (unlikely(idx < 0 || idx >= tup->tuple.size)) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.tuple_getitem: index %lld out of range [0, %d).",
            (long long)idx, tup->tuple.size);
    }
    RelaxVMRegister src = tup->tuple.ptr[idx];
    /* Manual ref-count bump matching TVM_RT_WASM_RelaxVMRegisterCopy. */
    if (src.typecode == RelaxVMRegType_ManagedDLTensor) {
        ++(((RelaxVMRegisterManagedDLTensor *)(src.value.v_handle))->ref_num);
    } else if ((int)src.typecode & RelaxVMRegType_VMObjectMask) {
        ++(((RelaxVMRegisterObject *)(src.value.v_handle))->ref_num);
    }
    *ret_value = src.value;
    ret_value->type_index = (int32_t)src.typecode;
    return 0;
}

RELAX_VM_FUNC(MakeTuple) {
    (void)source_handle;

    /*
     * make_tuple(x0, x1, ..., xN-1) -> tuple.
     * Owns fresh storage for the element array; each element inherits an
     * added reference from the caller side (the caller's own register
     * ref remains valid — a subsequent null_value on that register drops
     * only its ref, not the tuple's).
     */
    RelaxVMRegisterObject *tup;
    TVM_RT_WASM_RelaxVMRegisterCreateObject(tup);
    tup->tuple.size = num_args;
    tup->tuple.ptr =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMRegister) * (size_t)(num_args > 0 ? num_args : 1));
    for (int32_t i = 0; i < num_args; ++i) {
        RelaxVMRegisterTypeCode tc =
            TVM_RT_WASM_RelaxVM_FFIArgToRegTypeCode(args_value[i].type_index);
        tup->tuple.ptr[i].typecode = tc;
        tup->tuple.ptr[i].value = args_value[i];
        if (tc == RelaxVMRegType_ManagedDLTensor) {
            ++(((RelaxVMRegisterManagedDLTensor *)(args_value[i].v_handle))->ref_num);
        } else if ((int)tc & RelaxVMRegType_VMObjectMask) {
            ++(((RelaxVMRegisterObject *)(args_value[i].v_handle))->ref_num);
        }
    }
    ret_value->v_handle = tup;
    ret_value->type_index = (int32_t)RelaxVMRegType_VMObjectTuple;
    return 0;
}

RELAX_VM_FUNC(TensorToShape) {
    (void)num_args;
    (void)source_handle;

    /*
     * Convert a rank-1 int64 tensor into a Shape (VMObjectShapeTuple).
     * The tensor is the fork's RelaxVMRegisterManagedDLTensor wrapper whose
     * first field is DLTensor, so args_value[0].v_handle is a DLTensor*.
     */
    DLTensor *tensor = (DLTensor *)args_value[0].v_handle;
    if (unlikely(tensor->ndim != 1) || unlikely(tensor->dtype.code != kDLInt) ||
        unlikely(tensor->dtype.bits != 64) || unlikely(tensor->dtype.lanes != 1)) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "vm.builtin.tensor_to_shape expects a rank-1 int64 tensor.");
    }

    int64_t n = tensor->shape[0];
    RelaxVMRegisterObject *shape_obj;
    TVM_RT_WASM_RelaxVMRegisterCreateObject(shape_obj);
    shape_obj->shape_tuple.ndim = (int)n;
    shape_obj->shape_tuple.shape =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
    memcpy(shape_obj->shape_tuple.shape, (const char *)tensor->data + tensor->byte_offset,
           sizeof(int64_t) * (size_t)n);

    ret_value->v_handle = shape_obj;
    ret_value->type_index = (int32_t)RelaxVMRegType_VMObjectShapeTuple;
    return 0;
}

/*
 * Convert a Shape (VMObjectShapeTuple) into a rank-1 int64 tensor.
 * Backs the `relax.run.shape_to_tensor` global — used by shape-consuming
 * ops that the frontend lowers via `relax.op.shape_to_tensor(shape_of(x))`.
 * Owns its shape[1] and data buffers; free path is
 *   RelaxVMRegisterFreeManagedDLTensor -> heap free / TVMDeviceFreeDataSpace.
 */
RELAX_VM_FUNC(ShapeToTensor) {
    (void)num_args;
    (void)source_handle;

    RelaxVMRegisterObject *shape_obj = args_value[0].v_handle;
    int ndim = shape_obj->shape_tuple.ndim;

    RelaxVMRegisterManagedDLTensor *dl_tensor;
    TVM_RT_WASM_RelaxVMRegisterCreateManagedDLTensor(dl_tensor);

    int64_t *tensor_shape = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t));
    tensor_shape[0] = (int64_t)ndim;

    size_t nbytes = sizeof(int64_t) * (size_t)(ndim > 0 ? ndim : 1);
    void *data = TVM_RT_WASM_HeapMemoryAlignedAlloc(nbytes);
    if (ndim > 0) {
        memcpy(data, shape_obj->shape_tuple.shape, sizeof(int64_t) * (size_t)ndim);
    }

    dl_tensor->dl_tensor.data = data;
    dl_tensor->dl_tensor.device = (DLDevice){kDLCPU, 0};
    dl_tensor->dl_tensor.ndim = 1;
    dl_tensor->dl_tensor.dtype = (DLDataType){kDLInt, 64, 1};
    dl_tensor->dl_tensor.shape = tensor_shape;
    dl_tensor->dl_tensor.strides = NULL;
    dl_tensor->dl_tensor.byte_offset = 0;

    dl_tensor->shape_obj = NULL;
    dl_tensor->should_free_shape = true;
    dl_tensor->storage_obj = NULL;
    dl_tensor->should_free_storage = true;

    ret_value->v_handle = dl_tensor;
    ret_value->type_index = (int32_t)RelaxVMRegType_ManagedDLTensor;
    return 0;
}

int TVM_RT_WASM_RelaxVMRegisterBuiltinGlobalFunctions() {
    static PackedFunction pf[20];
    static int has_registered = 0;

    if (likely(has_registered)) {
        return 0;
    }

    PackedFunction *current_pf = pf;
    int status;

#define REG_FUNC(_func_name_literal, _func_symbol_name_suffix)                                     \
    do {                                                                                           \
        current_pf->exec =                                                                         \
            (TVMBackendPackedCFunc)RELAX_VM_BUILTIN_FUNC_NAME(_func_symbol_name_suffix);           \
        status = TVMFuncRegisterGlobal("vm.builtin."_func_name_literal, (current_pf++), 1);        \
        if (unlikely(status)) {                                                                    \
            TVM_RT_SET_ERROR_RETURN(status, "Cannot register global function vm.builtin.`%s`.",    \
                                    _func_name_literal);                                           \
        }                                                                                          \
    } while (0)

    REG_FUNC("alloc_shape_heap", AllocaShape);
    REG_FUNC("match_shape", MatchShape);
    REG_FUNC("make_shape", MakeShape);

    REG_FUNC("check_tensor_info", CheckTensorInfo);
    REG_FUNC("check_shape_info", CheckShapeInfo);
    REG_FUNC("check_tuple_info", CheckTupleInfo);
    REG_FUNC("check_func_info", CheckFuncInfo);

    REG_FUNC("alloc_storage", AllocStorage);
    REG_FUNC("alloc_tensor", AllocDLTensor);

    REG_FUNC("make_closure", MakeClosure);
    REG_FUNC("invoke_closure", InvokeClosure);
    REG_FUNC("call_tir_dyn", CallTIRDyn);

    REG_FUNC("shape_of", ShapeOf);
    REG_FUNC("copy", Copy);
    REG_FUNC("reshape", Reshape);

    REG_FUNC("read_if_cond", ReadIfCond);
    REG_FUNC("tuple_getitem", TupleGetItem);
    REG_FUNC("make_tuple", MakeTuple);
    REG_FUNC("tensor_to_shape", TensorToShape);

    /*
     * `relax.run.shape_to_tensor` — registered under the `relax.run.` prefix,
     * not `vm.builtin.`, because the Relax compiler emits calls to it under
     * that name (see tvm/relax/op/base.py). The frontend inserts these calls
     * as part of ONNX -> Relax lowering wherever a shape needs to become a
     * runtime tensor (e.g. Reshape/Expand paths that pass through
     * `shape_to_tensor(shape_of(x))`).
     */
    current_pf->exec =
        (TVMBackendPackedCFunc)RELAX_VM_BUILTIN_FUNC_NAME(ShapeToTensor);
    status = TVMFuncRegisterGlobal("relax.run.shape_to_tensor", (current_pf++), 1);
    if (unlikely(status)) {
        TVM_RT_SET_ERROR_RETURN(
            status, "Cannot register global function relax.run.shape_to_tensor.");
    }

    has_registered = 1;
    return 0;
}
