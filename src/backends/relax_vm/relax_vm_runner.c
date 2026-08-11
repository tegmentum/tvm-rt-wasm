/**
 * @file relax_vm/relax_vm_runner.c
 * @brief Run the relax vm functions.
 */

#include <device/device_api.h>
#include <dlpack/dlpack.h>
#include <relax_vm/relax_vm.h>
#include <stdlib.h>
#include <utils/common.h>

#define TVM_RT_WASM_RelaxVMDefaultFrameCapacity 8

/* ---------------------------------------------------------------------
 * Optional per-run kernel-dispatch batching hooks.
 *
 * The WebGPU accelerator's `WGPU_DeviceGet` calls
 * `TVM_RT_WASM_RegisterKernelBatchHooks(begin, end)` at device-creation
 * time (see src/accelerators/webgpu/c_api/webgpu_js_impl.c) to install
 * WGPU_BeginKernelBatch / WGPU_EndKernelBatch as the per-run
 * open/close pair. When the WebGPU accelerator is not linked
 * (CPU-only tests: `toy_relax_test`, `encoder_test`, `decoder_test`),
 * the hooks stay NULL and `TVM_RT_WASM_RelaxVMRunFunction` skips
 * batching entirely.
 *
 * Batch-per-run is Approach C from
 * docs/tvm-boundary-overhead-investigation.md §4.A (cognition). Opens a
 * batch at RelaxVMRunFunction entry, closes at exit — the compute-pass
 * dispatch loop inside interpret sees `WGPU_FunctionRun` accumulating
 * into a shared encoder/pass. Cuts per-inference boundary crossings
 * from ~13N to ~5N + fixed. See webgpu_c_api.h for the API contract. */
static int (*g_kernel_batch_begin)(void *device_stream) = NULL;
static int (*g_kernel_batch_end)(void *device_stream) = NULL;

void TVM_RT_WASM_RegisterKernelBatchHooks(int (*begin)(void *),
                                          int (*end)(void *)) {
    g_kernel_batch_begin = begin;
    g_kernel_batch_end = end;
}

/**
 * Create a new frame and push to VM frame stack, return the pointer to new frame.
 * @param vm The Relax VM instance.
 * @param num_registers The number of registers in the new frame.
 * @return The pointer to created frame.
 */
INLINE RelaxVMFrame *TVM_RT_WASM_RelaxVMFramePush(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                  size_t num_registers) {
    if (unlikely(vm->frame_size == vm->frame_capacity)) {
        if (vm->frame_capacity == 0) {
            // alloc
            vm->frame_capacity = TVM_RT_WASM_RelaxVMDefaultFrameCapacity;
            vm->frames = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMFrame) * vm->frame_capacity);
            memset(vm->frames, 0, sizeof(RelaxVMFrame) * vm->frame_capacity);
        } else {
            // realloc and copy
            vm->frame_capacity <<= 1;
            RelaxVMFrame *frames =
                TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMFrame) * vm->frame_capacity);
            // vm.frame_size * 2 == vm.frame_capacity.
            memcpy(frames, vm->frames, sizeof(RelaxVMFrame) * vm->frame_size);
            memset(frames + vm->frame_size, 0, sizeof(RelaxVMFrame) * vm->frame_size);
            TVM_RT_WASM_HeapMemoryFree(vm->frames);
            vm->frames = frames;
        }
    }
    RelaxVMFrame *frame = vm->frames + (vm->frame_size++);
    if (frame->register_capacity < num_registers) { // find a buffer or alloc.
        for (size_t i = vm->frame_size; i < vm->frame_capacity; ++i) {
            // vm->frames[i].register_size is 0
            if (vm->frames[i].register_capacity >= num_registers) {
                // swap
                RelaxVMRegister *_tmp_reg = vm->frames[i].registers;
                vm->frames[i].registers = frame->registers;
                frame->registers = _tmp_reg;
                size_t _tmp_cap = vm->frames[i].register_capacity;
                vm->frames[i].register_capacity = frame->register_capacity;
                frame->register_capacity = _tmp_cap;
                break;
            }
        }
        if (frame->register_capacity < num_registers) {
            if (frame->registers) {
                TVM_RT_WASM_HeapMemoryFree(frame->registers);
            }
            frame->registers = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMRegister) * num_registers);
            memset(frame->registers, 0, sizeof(RelaxVMRegister) * num_registers);
            frame->register_capacity = num_registers;
        }
    }
    frame->register_size = num_registers;
    frame->return_pc = vm->pc;
    return frame;
}

/**
 * Pop the frame from VM frame stack.
 * Just free all register values, Do not free register buffers.
 * @param vm The Relax VM instance.
 */
#define TVM_RT_WASM_RelaxVMFramePop(_vm)                                                           \
    do {                                                                                           \
        RelaxVMFrame *_frame = (_vm)->frames + (--((_vm)->frame_size));                            \
        (_vm)->pc = _frame->return_pc;                                                             \
        for (size_t _reg_i = 0; _reg_i < _frame->register_size; ++_reg_i) {                        \
            TVM_RT_WASM_RelaxVMRegisterFreeValue(_frame->registers[_reg_i]);                       \
        }                                                                                          \
        _frame->register_size = 0;                                                                 \
    } while (0)

/**
 * @brief Copy the register from the frame.
 * @note The dst register must be a empty register.
 */
#define TVM_RT_WASM_RelaxRegisterCopyFromFrame(_dst, _src_reg_array, _src_reg_name)                \
    do {                                                                                           \
        if ((_src_reg_name) < RelaxVM_RegName_Special) {                                           \
            TVM_RT_WASM_RelaxVMRegisterCopy(_dst, (_src_reg_array)[_src_reg_name]);                \
        } else if ((_src_reg_name) == RelaxVM_RegName_Void) {                                      \
            (_dst).typecode = RelaxVMRegType_Nullptr;                                              \
        } else if ((_src_reg_name) == RelaxVM_RegName_VM) {                                        \
            (_dst).value.v_handle = vm;                                                            \
            (_dst).typecode = RelaxVMRegType_ObjectHandle;                                         \
        } else {                                                                                   \
            unreachable();                                                                         \
        }                                                                                          \
    } while (0)

static int TVM_RT_WASM_RelaxVMInterpretInstructions(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                    RelaxVMRegister *return_register) {
    RelaxVMFrame *current_frame = vm->frames + (vm->frame_size - 1);
    RelaxVMRegister *registers = current_frame->registers;
    while (1) {
        const RelaxInstruction *instr = &vm->exec_module->exec.instructions[vm->pc];
        switch (instr->type) {
        case RelaxInstructionType_Call: {
            ++vm->pc;
            RelaxFunctionInfo *func =
                vm->exec_module->exec.relax_functions + instr->op_call.func_id;
            switch (func->type) {
            case RelaxFuncType_Packed: {
                /*
                 * TVM 0.25 dispatch: the parallel `args_typecode` array is
                 * gone; each TVMFFIAny carries its tag in `type_index`. Pack
                 * the fork-internal RelaxVMRegisterTypeCode straight into
                 * the tag slot. Built-ins agree on that convention (see
                 * vm_builtin.c). The special-register cases (Void, VM) use
                 * the aliased TVMFFITypeIndex values.
                 *
                 * Tensor-arg tagging is split by callee kind. TIR kernels
                 * emitted by relax.build expect tensor args tagged with
                 * `kTVMFFIDLTensorPtr` (= 7) — normalise the fork's
                 * internal `RelaxVMRegType_ManagedDLTensor` (= 70) to that.
                 * Builtins (`vm.builtin.*`) MUST see the original 70 tag
                 * because e.g. `vm.builtin.reshape` refcounts the source
                 * storage differently on ManagedDLTensor vs bare
                 * DLTensor* — dropping to 7 makes it treat every reshape
                 * source as unmanaged, skipping the refcount, and lets
                 * subsequent alloc_storage malloc into the still-live
                 * tensor data. M13.3c encoder-gate bug root cause. Detect
                 * the callee by the "vm.builtin." name prefix.
                 */
                int is_builtin = (func->packed_func.name_size >= 11 &&
                                  memcmp(func->packed_func.name_ptr, "vm.builtin.", 11) == 0);
                int32_t num_args = (int32_t)instr->op_call.num_args;
                for (int32_t i = 0; i < num_args; ++i) {
                    const struct RelaxInstructionCallArg *arg = instr->op_call.args + i;
                    TVMFFIAny *slot = &vm->call_packed_args_value[i];
                    switch (arg->arg_type) {
                    case RelaxInstructionCallArgType_ConstIdx: {
                        *slot = vm->constants[arg->const_idx].value;
                        RelaxVMRegisterTypeCode ctc =
                            vm->constants[arg->const_idx].typecode;
                        /* Mirror the Register-arm downgrade: TIR kernels
                         * emitted by relax.build expect tensor args tagged
                         * with kTVMFFIDLTensorPtr (=7). Constants live under
                         * ManagedDLTensor (=70 = kTVMFFITensor) because they
                         * were staged into device memory via
                         * RelaxVM_CopyTensorToRegister, so without this
                         * downgrade the WebGPU host stubs' TIR arg-check
                         * reads .ndim through the NDArray-shaped struct
                         * offset and trips "Mismatched param_0.ndim ...
                         * expected 3" on the decoder's first host stub
                         * (fused_conv1d_add_multiply). Builtins keep the
                         * original 70 tag — same invariant the Register
                         * path enforces to preserve reshape refcount
                         * behaviour flagged in M13.3c. */
                        if (ctc == RelaxVMRegType_ManagedDLTensor && !is_builtin) {
                            slot->type_index = (int32_t)kTVMFFIDLTensorPtr;
                        } else {
                            slot->type_index = (int32_t)ctc;
                        }
                        break;
                    }
                    case RelaxInstructionCallArgType_Immediate:
                        slot->v_int64 = arg->immediate_val;
                        slot->type_index = (int32_t)kTVMArgInt;
                        break;
                    case RelaxInstructionCallArgType_Register: {
                        RelaxVMRegisterName reg_name = arg->arg_register;
                        if (reg_name < RelaxVM_RegName_Special) {
                            *slot = registers[reg_name].value;
                            RelaxVMRegisterTypeCode tc = registers[reg_name].typecode;
                            if (tc == RelaxVMRegType_ManagedDLTensor && !is_builtin) {
                                slot->type_index = (int32_t)kTVMFFIDLTensorPtr;
                            } else {
                                slot->type_index = (int32_t)tc;
                            }
                        } else if (reg_name == RelaxVM_RegName_Void) {
                            slot->v_handle = NULL;
                            slot->type_index = (int32_t)kTVMNullptr;
                        } else if (reg_name == RelaxVM_RegName_VM) {
                            slot->v_handle = vm;
                            slot->type_index = (int32_t)kTVMObjectHandle;
                        } else {
                            unreachable();
                        }
                        break;
                    }
                    case RelaxInstructionCallArgType_FuncIdx:
                    default:
                        unreachable();
                    }
                }
                if (func->packed_func.pf) {
                    TVMFFIAny ret_value;
                    /* TVMFFISafeCallType contract: caller must initialize
                     * result->type_index to kTVMFFINone before the call. */
                    ret_value.type_index = (int32_t)kTVMFFINone;
                    ret_value.zero_padding = 0;
                    ret_value.v_int64 = 0;
                    int status = func->packed_func.pf->exec(NULL, vm->call_packed_args_value,
                                                            num_args, &ret_value);

                    if (unlikely(status)) {
                        return status;
                    }
                    if (instr->op_call.reg_dst < current_frame->register_size) {
                        RelaxVMRegister *reg = registers + instr->op_call.reg_dst;
                        TVM_RT_WASM_RelaxVMRegisterFreeValue(*reg);
                        reg->typecode = (RelaxVMRegisterTypeCode)ret_value.type_index;
                        reg->value = ret_value;
                    }
                } else { // The null value function, clear the dst register.
                    RelaxVMRegister *reg = registers + instr->op_call.reg_dst;
                    TVM_RT_WASM_RelaxVMRegisterFreeValue(*reg);
                }
                break;
            }
            case RelaxFuncType_VMFunc: {
                TVM_RT_WASM_RelaxVMFramePush(vm, func->vm_func.register_file_size);
                vm->pc = func->vm_func.start_instr;
                RelaxVMFrame *callee_frame = vm->frames + (vm->frame_size - 1);
                RelaxVMRegister *callee_registers = callee_frame->registers;
                callee_frame->reg_caller_return = instr->op_call.reg_dst;
                for (size_t i = 0; i < func->vm_func.num_params; ++i) {
                    const struct RelaxInstructionCallArg *arg = instr->op_call.args + i;
                    switch (arg->arg_type) {
                    case RelaxInstructionCallArgType_ConstIdx:
                        TVM_RT_WASM_RelaxVMRegisterCopy(callee_registers[i],
                                                        vm->constants[arg->const_idx]);
                        break;
                    case RelaxInstructionCallArgType_Immediate:
                        callee_registers[i].typecode = RelaxVMRegType_Int;
                        callee_registers[i].value.v_int64 = arg->immediate_val;
                        break;
                    case RelaxInstructionCallArgType_Register: {
                        TVM_RT_WASM_RelaxRegisterCopyFromFrame(callee_registers[i], registers,
                                                               arg->arg_register);
                        break;
                    }
                    case RelaxInstructionCallArgType_FuncIdx:
                    default:
                        unreachable();
                    }
                }
                registers = callee_registers;
                current_frame = callee_frame;
                break;
            }
            case RelaxFuncType_VMTIRFunc:
            default:
                unreachable();
            }
            break;
        }
        case RelaxInstructionType_Ret: {
            if (vm->frame_size <= 1) {
                // free the ret register old value.
                TVM_RT_WASM_RelaxVMRegisterFreeValue(*return_register);
                TVM_RT_WASM_RelaxRegisterCopyFromFrame(*return_register, registers,
                                                       instr->op_ret.reg_result);
                TVM_RT_WASM_RelaxVMFramePop(vm);
                return 0;
            } else {
                RelaxVMFrame *caller_frame = vm->frames + (vm->frame_size - 2);
                RelaxVMRegister *caller_registers = caller_frame->registers;
                RelaxVMRegisterName reg_rt = current_frame->reg_caller_return;
                if (likely(reg_rt < caller_frame->register_size)) {
                    // free the ret register old value.
                    TVM_RT_WASM_RelaxVMRegisterFreeValue(caller_registers[reg_rt]);
                    // copy to ret register
                    TVM_RT_WASM_RelaxRegisterCopyFromFrame(caller_registers[reg_rt], registers,
                                                           instr->op_ret.reg_result);
                }
                TVM_RT_WASM_RelaxVMFramePop(vm);
                current_frame = caller_frame;
                registers = caller_registers;
            }
            break;
        }
        case RelaxInstructionType_Goto:
            vm->pc += instr->op_goto.pc_offset;
            break;
        case RelaxInstructionType_If: {
            // assert(instr->op_if.reg_cond < RelaxVM_RegName_Special);
            RelaxVMRegister cond = registers[instr->op_if.reg_cond];
            // assert(cond.typecode == RelaxVMRegType_Int);
            if (cond.value.v_int64 != 0) {
                ++vm->pc;
            } else {
                vm->pc += instr->op_if.false_offset;
            }
            break;
        }
        default:
            unreachable();
        }
    }
}

int TVM_RT_WASM_RelaxVMRunFunction(TVM_RT_WASM_RelaxVirtualMachine vm, RelaxFunctionInfo *func,
                                   RelaxVMFunctionInputsOutput *inputs_output) {
    RelaxVMFrame *current_frame =
        TVM_RT_WASM_RelaxVMFramePush(vm, func->vm_func.register_file_size);
    vm->pc = func->vm_func.start_instr;
    // set input
    for (size_t i = 0; i < inputs_output->num_inputs; ++i) {
        TVM_RT_WASM_RelaxVMRegisterCopy(current_frame->registers[i],
                                        inputs_output->inputs_output[i]);
    }

    /* BATCH-TVM: open a per-run kernel-dispatch batch on the WebGPU
     * device (if any) so kernel invocations during interpretation
     * share encoder + pass + submit.
     *
     * DEFAULT-ON as of 2026-08-11 (promoted from opt-in). Correctness
     * gate verified post-`89e1606` (defer WGPU_MemoryFree during open
     * batch): cosine >= 0.99999947, max_abs_delta <= 5.03e-3, snr_db
     * >= 59.8 dB across 3-5 iterations at load-avg 5-100. Measured
     * wall reduction: -44% at low load (~5-7), -15% at high load
     * (~100). See cognition docs/perf-mitigation-plan.md BATCH-TVM
     * row + docs/perf-prediction-patterns.md §1.2 for the audit trail.
     *
     * Escape hatch: `BATCH_TVM_ENABLE=0` disables the runner hook and
     * routes back through the pre-BATCH per-kernel encoder/submit
     * path — retained for regression debugging on any downstream
     * decoder graph that trips a fresh Dawn buffer-lifetime invariant.
     * A missing / unset env var leaves the fast path armed. */
    void *wgpu_stream = NULL;
    const char *enable = getenv("BATCH_TVM_ENABLE");
    int batch_enabled = 1;
    if (enable != NULL &&
        (enable[0] == '0' || enable[0] == 'f' || enable[0] == 'F' ||
         enable[0] == 'n' || enable[0] == 'N')) {
        batch_enabled = 0;
    }
    if (batch_enabled && g_kernel_batch_begin != NULL &&
        vm->num_device > 0 && vm->devices[0].device_type == kDLWebGPU) {
        DeviceAPI *webgpu_api = NULL;
        if (TVM_RT_WASM_DeviceAPIGet(kDLWebGPU, &webgpu_api) == 0 && webgpu_api != NULL) {
            wgpu_stream = webgpu_api->GetStream();
            if (wgpu_stream != NULL) {
                (void)g_kernel_batch_begin(wgpu_stream);
            }
        }
    }

    int status = TVM_RT_WASM_RelaxVMInterpretInstructions(
        vm, &inputs_output->inputs_output[inputs_output->num_inputs]);

    /* Close the batch — flushes any pending compute (finish + submit)
     * and drops retained bind-groups. The output tensor readback then
     * sees a fully-submitted GPU state. */
    if (wgpu_stream != NULL && g_kernel_batch_end != NULL) {
        (void)g_kernel_batch_end(wgpu_stream);
    }
    return status;
}

/*---------------------Functions for Relax VM register -------------------------------------------*/

void TVM_RT_WASM_RelaxVMRegisterFreeObject(RelaxVMRegisterObject *obj, int typecode) {
    if ((--obj->ref_num) == 0) {
        switch (typecode) {
        case RelaxVMRegType_VMObjectStorage: /* free the Storage */
            TVMDeviceFreeDataSpace(obj->storage.device, obj->storage.data);
            break;
        case RelaxVMRegType_VMObjectShapeTuple: /* free the Shape Tuple */
            TVM_RT_WASM_HeapMemoryFree(obj->shape_tuple.shape);
            break;
        case RelaxVMRegType_VMObjectString: /* free the String */
            TVM_RT_WASM_HeapMemoryFree(obj->string.ptr);
            break;
        case RelaxVMRegType_VMObjectTuple: /* free the Tuple Object. */
            for (int i = 0; i < obj->tuple.size; ++i) {
                RelaxVMRegister reg = obj->tuple.ptr[i];
                TVM_RT_WASM_RelaxVMRegisterFreeValue(reg);
            }
            break;
        default:
            break;
        }
        TVM_RT_WASM_HeapMemoryFree(obj);
    }
}
