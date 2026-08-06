/**
 * @file relax_vm/relax_executable_module.c
 * @brief The implementation for relax executable module.
 */

#include <string.h>

#include <module/module_impl.h>
#include <relax_vm/relax_executable.h>
#include <utils/stream_reader.h>
#include <utils/tensor_helper.h>

/**
 * @brief Magic numbers for the Relax VM executable bytecode.
 *
 * TVM 0.25 (`apache/tvm@v0.25.0`, `src/runtime/vm/executable.cc`):
 *
 *   V1 (0xD225DE2F4214151D): header → Global → Constant → Code
 *   V2 (0xD225DE2F4214151E): header → Global → MemoryScope → Constant → Code
 *
 * The runtime accepts either magic; only V2 executables carry the
 * MemoryScope section. cognition's tvm_compile.py pipeline emits V2
 * unconditionally, but keep V1 read-support so older cached artifacts
 * still load.
 */
#define kTVMVMBytecodeMagicV1 (UINT64_C(0xD225DE2F4214151D))
#define kTVMVMBytecodeMagicV2 (UINT64_C(0xD225DE2F4214151E))

/**
 * TVM special register name.
 * In this implementation, size_t may 32bit, so we must change the special register < 32bit.
 */
#define kTVM_kBeginSpecialReg ((int64_t)(INT64_C(1) << 54))
#define kTVM_kVoidRegister ((int64_t)(kTVM_kBeginSpecialReg + INT64_C(0)))
#define kTVM_kVMRegister ((int64_t)(kTVM_kBeginSpecialReg + INT64_C(1)))

// Definitions for relax call instruction argument data.
#define RelaxInstructionCallArg_TypeEnumBits 8
#define RelaxInstructionCallArg_ValueBits                                                          \
    (sizeof(int64_t) * 8 - RelaxInstructionCallArg_TypeEnumBits)
#define RelaxInstructionCallArg_ValueMask ((INT64_C(1) << RelaxInstructionCallArg_ValueBits) - 1)
// Split data to type and value.
#define RelaxInstructionCallArgDataGetType(_data)                                                  \
    (enum RelaxInstructionCallArgType)(((_data) >> RelaxInstructionCallArg_ValueBits) & 0xFF)
#define RelaxInstructionCallArgDataGetValue(_data)                                                 \
    ((((_data)&RelaxInstructionCallArg_ValueMask) << RelaxInstructionCallArg_TypeEnumBits) >>      \
     RelaxInstructionCallArg_TypeEnumBits)

/*
 * TVM 0.25's VMFuncInfo wire layout (apache/tvm@v0.25.0, src/runtime/vm/executable.cc,
 * VMFuncInfo::Save/Load):
 *
 *   int32_t              kind        (0=Packed, 1=VMFunc, 2=VMTIRFunc)
 *   string               name        (u64 len + bytes)
 *   int64_t              start_instr
 *   int64_t              end_instr
 *   int64_t              num_args
 *   int64_t              register_file_size
 *   vec<string>          param_names (u64 count + [u64 len + bytes]*)
 *
 * All three kinds emit the same 6 header fields + a param_names vector.
 * For Packed the numeric fields are always {0, 0, -2, 0} and the vector
 * is empty. For VMFunc num_args generally equals param_names.size().
 * For VMTIRFunc num_args is the input-count and param_names.size() may
 * be 0 (the parameter names live in the underlying TIR PrimFunc).
 *
 * The old pre-Unity layout the fork inherited omitted the trailing
 * param_names vector (VMFunc emitted a `num_params` int64 that had to
 * equal num_args, followed by names). That coincidentally matches the
 * new format for VMFuncs where param_names.size() == num_args, but
 * breaks for any function where the invariant does not hold, and for
 * Packed's terminating u64 count when it happens to be non-zero.
 */
static int TVM_RT_WASM_RelaxExecutableLoadGlobalSection(RelaxExecutable *exec,
                                                        BinaryReader *reader) {
    int status = 0;
    const char *cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
    size_t func_size = (size_t) * (uint64_t *)cur_ptr;
    TVM_RT_WASM_TrieCreate(&exec->relax_vm_functions_map);
    exec->relax_functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxFunctionInfo) * func_size);
    exec->num_relex_functions = func_size;
    memset(exec->relax_functions, 0, sizeof(RelaxFunctionInfo) * func_size);
    for (size_t index = 0; index < func_size; ++index) {
        RelaxFunctionInfo *info = exec->relax_functions + index;

        /* kind (int32_t on wire; sign is irrelevant for the 3 enum values). */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int32_t), load_global_fail);
        info->type = (enum RelaxFunctionType) * (const int32_t *)cur_ptr;

        /* name (u64 length + bytes) */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
        size_t name_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, name_size, load_global_fail);
        const char *name = cur_ptr;

        /* Read the 4 shared int64 fields regardless of kind. */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t) * 4, load_global_fail);
        const int64_t start_instr = ((const int64_t *)cur_ptr)[0];
        const int64_t end_instr = ((const int64_t *)cur_ptr)[1];
        const int64_t num_args = ((const int64_t *)cur_ptr)[2];
        const int64_t register_file_size = ((const int64_t *)cur_ptr)[3];

        /* param_names vector length (uint64) — always emitted. */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
        const uint64_t num_param_names = *(const uint64_t *)cur_ptr;

        switch (info->type) {
        case RelaxFuncType_Packed:
            /*
             * Packed entries share the header for compatibility but the
             * numeric fields are ignored at runtime. Some compiled programs
             * still emit a non-zero param_names vector for packed entries
             * (see relax.builder), so drain it rather than asserting size 0.
             */
            (void)start_instr;
            (void)end_instr;
            (void)num_args;
            (void)register_file_size;
            for (uint64_t p = 0; p < num_param_names; ++p) {
                TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
                size_t sz = (size_t) * (const uint64_t *)cur_ptr;
                TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sz, load_global_fail);
            }
            info->packed_func.name_ptr = name;
            info->packed_func.name_size = name_size;
            break;
        case RelaxFuncType_VMFunc: {
            /* insert relax VM function to relax function maps */
            TVM_RT_WASM_TrieInsertWithLen(exec->relax_vm_functions_map, (const uint8_t *)name,
                                          name_size, (void *)info);
            info->vm_func.start_instr = (RelaxVMIndex)start_instr;
            info->vm_func.end_instr = (RelaxVMIndex)end_instr;
            if (unlikely((uint64_t)register_file_size >= (uint64_t)RelaxVM_RegName_Special)) {
                TVM_RT_SET_ERROR_AND_GOTO(load_global_fail,
                                          "Relax VM Register Name is too big: %" PRIi64 " >= %zu",
                                          register_file_size, RelaxVM_RegName_Special);
            }
            info->vm_func.register_file_size = (size_t)register_file_size;
            info->vm_func.num_params = (size_t)num_args;

            TVM_RT_WASM_TrieCreate(&info->vm_func.params_map);
            /*
             * Use the param_names vector to populate the params_map. It
             * generally has one entry per positional argument, but in the
             * post-Unity format the two counts are independent — trust
             * num_param_names here.
             */
            for (uint64_t p = 0; p < num_param_names; ++p) {
                TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
                size_t sz = (size_t) * (const uint64_t *)cur_ptr;
                TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sz, load_global_fail);
                status = TVM_RT_WASM_TrieInsertWithLen(info->vm_func.params_map,
                                                       (const uint8_t *)cur_ptr, sz,
                                                       (void *)(uintptr_t)p);
            }
            break;
        }
        case RelaxFuncType_VMTIRFunc:
            /*
             * VMTIRFunc entries are direct-lowered TIR calls (no bytecode).
             * The fork does not execute them today — the toy program does
             * not emit any — but we still drain the param_names bytes so
             * downstream sections keep their frame alignment. Fail loudly
             * only when the interpreter actually tries to dispatch one.
             */
            for (uint64_t p = 0; p < num_param_names; ++p) {
                TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
                size_t sz = (size_t) * (const uint64_t *)cur_ptr;
                TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sz, load_global_fail);
            }
            break;
        default:
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_global_fail, "Unsupported relax function type %u.",
                                      info->type);
        }
    }
load_global_fail:
    return status;
}

/*
 * TVM 0.25 (kTVMVMBytecodeMagicV2) inserts a MemoryScope section between
 * Global and Constant (see apache/tvm@v0.25.0 src/runtime/vm/executable.cc,
 * VMExecutable::SaveMemoryScopeSection / LoadMemoryScopeSection):
 *
 *   uint64_t             num_scopes
 *   for i in [0, num_scopes):
 *       int64_t          const_idx
 *       string           scope_name    (u64 len + bytes)
 *
 * Cognition's browser lane is CPU-only; memory scopes end up as either
 * empty or `"global"` for the whole constant pool. Drain the bytes so
 * the reader lands on the Constant section, but do not surface the
 * scope map to the interpreter — the CPU path treats every allocation
 * as global-scope already.
 */
static int TVM_RT_WASM_RelaxExecutableLoadMemoryScopeSection(RelaxExecutable *exec,
                                                             BinaryReader *reader) {
    (void)exec;
    int status = 0;
    const char *cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_memory_scope_fail);
    const uint64_t num_scopes = *(const uint64_t *)cur_ptr;

    for (uint64_t i = 0; i < num_scopes; ++i) {
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_memory_scope_fail);
        /* const_idx — presently unused. */
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_memory_scope_fail);
        size_t scope_size = (size_t) * (const uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, scope_size, load_memory_scope_fail);
    }

load_memory_scope_fail:
    return status;
}

/*
 * TVM 0.25's Constant section (apache/tvm@v0.25.0,
 * src/runtime/vm/executable.cc, VMExecutable::LoadConstantSection):
 *
 *   uint64_t             num_constants
 *   for i in [0, num_constants):
 *     int32_t            type_index   (TVMFFITypeIndex — see tvm/ffi/c_api.h)
 *     switch (type_index):
 *       kTVMFFITensor    (70) → SaveDLTensor blob (magic + reserved + dev + ndim + dtype + shape + u64 nbytes + data)
 *       kTVMFFIShape     (69) → u64 size + int64[size]
 *       kTVMFFIStr       (65) → u64 size + uint8[size]
 *       kTVMFFIInt       (1)  → int64
 *       kTVMFFIFloat     (3)  → double
 *       kTVMFFIDataType  (5)  → DLDataType (4 bytes: code, bits, lanes)
 *
 * The fork's pre-Unity enum (`RelaxConstantType_{DLTensor=0,DLDataType=1,
 * ShapeTuple=2,String=3,Int=4}`) no longer matches; the executable now
 * tags constants with FFI type indices. Translate on load and stash into
 * the fork's own enum for downstream dispatch.
 */
#define kTVMFFIInt_wire      1
#define kTVMFFIFloat_wire    3
#define kTVMFFIDataType_wire 5
#define kTVMFFIStr_wire      65
#define kTVMFFIShape_wire    69
#define kTVMFFITensor_wire   70

static int TVM_RT_WASM_RelaxExecutableLoadConstantSection(RelaxExecutable *exec,
                                                          BinaryReader *reader) {
    int status = 0;
    const char *cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_constant_fail);
    const size_t num_constants = (const size_t) * (uint64_t *)cur_ptr;
    exec->num_constants = num_constants;
    exec->constants = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxConstant) * num_constants);
    memset(exec->constants, 0, sizeof(RelaxConstant) * num_constants);
    for (size_t c_id = 0; c_id < num_constants; ++c_id) {
        RelaxConstant *constant = exec->constants + c_id;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int32_t), load_constant_fail);
        int32_t type_index = *(const int32_t *)cur_ptr;
        switch (type_index) {
        case kTVMFFITensor_wire: {
            constant->type = RelaxConstantType_DLTensor;
            status = TVM_RT_WASM_DLTensor_LoadFromBinary(&constant->dl_tensor, reader);
            if (unlikely(status)) {
                goto load_constant_fail;
            }
#if TENSOR_DATA_MUST_ALIGN
            void *data = constant->dl_tensor.data;
            if (((uintptr_t)data) & ((1 << DATA_ALIGNMENT_BITS) - 1)) { // not aligned
                constant->type = RelaxConstantType_DLTensorShouldFree;
                size_t bytes = TVM_RT_WASM_DLTensor_GetDataBytes(
                    constant->dl_tensor.shape, constant->dl_tensor.ndim, constant->dl_tensor.dtype);
                constant->dl_tensor.data = TVM_RT_WASM_HeapMemoryAlignedAlloc(bytes);
                memcpy(constant->dl_tensor.data, data, bytes);
            }
#endif // TENSOR_DATA_MUST_ALIGN
            break;
        }
        case kTVMFFIDataType_wire:
            constant->type = RelaxConstantType_DLDataType;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(DLDataType), load_constant_fail);
            constant->dl_datatype = *(DLDataType *)cur_ptr;
            break;
        case kTVMFFIShape_wire: {
            /*
             * TVM 0.25 emits size as u64 then a raw int64 array of that
             * length via WriteArray — no length-prefix per element.
             * The reader keeps a pointer into the (immutable) blob so
             * downstream dispatch can read the shape without copying.
             */
            constant->type = RelaxConstantType_ShapeTuple;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_constant_fail);
            uint64_t shape_size = *(const uint64_t *)cur_ptr;
            constant->register_obj.shape_tuple.ndim = (int)shape_size;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, shape_size * sizeof(int64_t),
                                              load_constant_fail);
            constant->register_obj.shape_tuple.shape = (int64_t *)cur_ptr;
            constant->register_obj.ref_num = 1;
            break;
        }
        case kTVMFFIStr_wire:
            constant->type = RelaxConstantType_String;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_constant_fail);
            constant->register_obj.string.size = (size_t) * (const uint64_t *)cur_ptr;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, constant->register_obj.string.size,
                                              load_constant_fail);
            constant->register_obj.string.ptr = (char *)cur_ptr;
            constant->register_obj.ref_num = 1;
            break;
        case kTVMFFIInt_wire:
            constant->type = RelaxConstantType_Int;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_constant_fail);
            constant->int_value = *(const int64_t *)cur_ptr;
            break;
        case kTVMFFIFloat_wire:
            /*
             * The fork does not yet carry a Float constant kind. Toy
             * program does not emit these; log-and-fail so any surprise
             * lands loud rather than as silent data corruption.
             */
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_constant_fail,
                                      "Unsupported relax constant TVMFFIFloat (index %" PRId32 ")",
                                      type_index);
        default:
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_constant_fail,
                                      "Unsupported relax constant type index %" PRId32,
                                      type_index);
        }
    }
load_constant_fail:
    return status;
}

static int TVM_RT_WASM_RelaxExecutableLoadCodeSection(RelaxExecutable *exec, BinaryReader *reader) {
    int status = 0;
    const char *cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_code_fail);
    const size_t instr_offset_size = (size_t) * (int64_t *)cur_ptr;
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t) * instr_offset_size, load_code_fail);
    const int64_t *instr_offsets = (const int64_t *)cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_code_fail);
    const size_t instr_data_size = (size_t) * (int64_t *)cur_ptr;
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t) * instr_data_size, load_code_fail);
    const int64_t *instr_data = (const int64_t *)cur_ptr;

    exec->num_instructions = instr_offset_size;
    exec->instructions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxInstruction) * exec->num_instructions);
    memset(exec->instructions, 0, sizeof(RelaxInstruction) * exec->num_instructions);
    for (size_t i = 0; i < instr_offset_size; ++i) {
        RelaxInstruction *instr = exec->instructions + i;
        size_t offset = (size_t)instr_offsets[i];

#define CHECK_CONVERT_REG_OR_FAIL(_dst, _val)                                                      \
    do {                                                                                           \
        int64_t _tmp = (_val);                                                                     \
        if (_tmp < kTVM_kBeginSpecialReg) {                                                        \
            if (unlikely(_tmp >= (int64_t)RelaxVM_RegName_Special)) { /* overflow */               \
                TVM_RT_SET_ERROR_RETURN(-1,                                                        \
                                        "Relax VM Register Name is too big: %" PRIi64 " >= %zu",   \
                                        _tmp, RelaxVM_RegName_Special);                            \
            }                                                                                      \
            (_dst) = (RelaxVMRegisterName)_tmp;                                                    \
        } else if (_tmp == kTVM_kVoidRegister) {                                                   \
            (_dst) = RelaxVM_RegName_Void;                                                         \
        } else if (_tmp == kTVM_kVMRegister) {                                                     \
            (_dst) = RelaxVM_RegName_VM;                                                           \
        } else {                                                                                   \
            TVM_RT_SET_ERROR_RETURN(-1, "Unsupported special relax VM register %" PRIi64, _tmp);   \
        }                                                                                          \
    } while (0)

        instr->type = (enum RelaxInstructionType)instr_data[offset];
        switch (instr->type) {
        case RelaxInstructionType_Call:
            CHECK_CONVERT_REG_OR_FAIL(instr->op_call.reg_dst, instr_data[offset + 1]);
            instr->op_call.func_id = (RelaxVMIndex)instr_data[offset + 2];
            instr->op_call.num_args = (RelaxVMIndex)instr_data[offset + 3];
            const int64_t *args_data = &instr_data[offset + 4];
            size_t max_arg_offset = offset + 4 + instr->op_call.num_args;
            if (unlikely(max_arg_offset > instr_data_size)) {
                TVM_RT_SET_ERROR_RETURN(-1, "Instruction Call num args exceed: %zu > %zu",
                                        max_arg_offset, instr_data_size);
            }
            // check and parse args
            size_t num_args = instr->op_call.num_args;
            exec->max_num_call_args = MAX(num_args, exec->max_num_call_args);
            struct RelaxInstructionCallArg *args =
                TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct RelaxInstructionCallArg) * num_args);
            instr->op_call.args = args;
            for (size_t arg_i = 0; arg_i < num_args; ++arg_i) {
                int64_t arg_data = args_data[arg_i];
                args[arg_i].arg_type = RelaxInstructionCallArgDataGetType(arg_data);
                int64_t arg_value = RelaxInstructionCallArgDataGetValue(arg_data);

                switch (args[arg_i].arg_type) {
                case RelaxInstructionCallArgType_Register: {
                    CHECK_CONVERT_REG_OR_FAIL(args[arg_i].arg_register, arg_value);
                    break;
                }
                case RelaxInstructionCallArgType_Immediate:
                    args[arg_i].immediate_val = arg_value;
                    break;
                case RelaxInstructionCallArgType_ConstIdx: {
                    size_t const_id = (size_t)arg_value;
                    if (const_id >= exec->num_constants) {
                        TVM_RT_SET_ERROR_RETURN(-1, "Constants index exceed: %zu >= %zu.", const_id,
                                                exec->num_constants);
                    }
                    args[arg_i].const_idx = (RelaxVMIndex)const_id;
                    break;
                }
                case RelaxInstructionCallArgType_FuncIdx: {
                    size_t func_id = (size_t)arg_value;
                    if (func_id >= exec->num_relex_functions) {
                        TVM_RT_SET_ERROR_RETURN(-1, "Functions index exceed: %zu >= %zu.", func_id,
                                                exec->num_relex_functions);
                    }
                    args[arg_i].func_idx = (RelaxVMIndex)func_id;
                    // todo
                    TVM_RT_SET_ERROR_RETURN(-1, "cannot pass functions in args now.");
                }
                default:
                    TVM_RT_SET_ERROR_RETURN(-1, "Unsupported instruction call arg type %u.",
                                            args[arg_i].arg_type);
                }
            }
            break;
        case RelaxInstructionType_Ret:
            CHECK_CONVERT_REG_OR_FAIL(instr->op_ret.reg_result, instr_data[offset + 1]);
            break;
        case RelaxInstructionType_Goto:
            instr->op_goto.pc_offset = (RelaxVMIndex)instr_data[offset + 1];
            break;
        case RelaxInstructionType_If:
            CHECK_CONVERT_REG_OR_FAIL(instr->op_if.reg_cond, instr_data[offset + 1]);
            instr->op_if.false_offset = (RelaxVMIndex)instr_data[offset + 2];
            break;
        default:
            TVM_RT_SET_ERROR_RETURN(-1, "Unsupported relax instruction type %u", instr->type);
        }
    }
#undef CHECK_CONVERT_REG_OR_FAIL
load_code_fail:
    return status;
}

static int TVM_RT_WASM_RelaxExecutableModuleReleaseFunc(Module *self) {
    RelaxExecutableModule *mod = (RelaxExecutableModule *)self;
    MODULE_BASE_MEMBER_FREE(mod);

    if (mod->exec.constants) {
#if TENSOR_DATA_MUST_ALIGN
        for (size_t i = 0; i < mod->exec.num_constants; ++i) {
            RelaxConstant *constant = mod->exec.constants + i;
            if (constant->type == RelaxConstantType_DLTensorShouldFree) {
                TVM_RT_WASM_HeapMemoryFree(constant->dl_tensor.data);
            }
        }
#endif // TENSOR_DATA_MUST_ALIGN
        TVM_RT_WASM_HeapMemoryFree(mod->exec.constants);
    }
    if (mod->exec.relax_vm_functions_map) {
        TVM_RT_WASM_TrieRelease(mod->exec.relax_vm_functions_map);
    }
    if (mod->exec.relax_functions) {
        RelaxFunctionInfo *funcs = mod->exec.relax_functions;
        for (size_t i = 0; i < mod->exec.num_relex_functions; ++i) {
            RelaxFunctionInfo *func = funcs + i;
            switch (func->type) {
            case RelaxFuncType_VMFunc:
                if (func->vm_func.params_map) {
                    TVM_RT_WASM_TrieRelease(func->vm_func.params_map);
                }
                break;
            case RelaxFuncType_Packed:
            case RelaxFuncType_VMTIRFunc:
                break;
            default:
                unreachable();
            }
        }
        TVM_RT_WASM_HeapMemoryFree(funcs);
    }
    if (mod->exec.instructions) {
        RelaxInstruction *instructions = mod->exec.instructions;
        for (size_t i = 0; i < mod->exec.num_instructions; ++i) {
            RelaxInstruction *instr = instructions + i;
            if (instr->type == RelaxInstructionType_Call) {
                if (instr->op_call.args) {
                    TVM_RT_WASM_HeapMemoryFree(instr->op_call.args);
                }
            }
        }
        TVM_RT_WASM_HeapMemoryFree(instructions);
    }

    TVM_RT_WASM_HeapMemoryFree(mod);
    return 0;
}

int TVM_RT_WASM_RelaxExecutableModuleCreate(BinaryReader *reader, Module **out) {
    *out = NULL;
    int status;
    const char *cur_ptr;

    /*
     * TVM 0.25 dropped the redundant "exec module size" u64 preamble —
     * per-module body framing is now the responsibility of the outer
     * library-bin envelope (see module.c:TVM_RT_WASM_LibraryModuleLoadBinaryBlob).
     * The first field the body carries is the magic number.
     */
    // check magic and version
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_fail);
    uint64_t header_magic = *(uint64_t *)cur_ptr;
    if (unlikely(header_magic != kTVMVMBytecodeMagicV1 && header_magic != kTVMVMBytecodeMagicV2)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid bytecode magic %" PRIu64, header_magic);
    }
    const int has_memory_scope_section = (header_magic == kTVMVMBytecodeMagicV2);
    // version (std::string)
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_fail);
    size_t version_str_len = (size_t) * (uint64_t *)cur_ptr;
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, version_str_len, load_fail);

    // Allocate memory for module instance
    RelaxExecutableModule *exec_mod = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxExecutableModule));
    memset(exec_mod, 0, sizeof(RelaxExecutableModule));
    *out = (Module *)exec_mod;

    exec_mod->Release = TVM_RT_WASM_RelaxExecutableModuleReleaseFunc;
    exec_mod->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;

#define RelaxLoadSection(_section_name)                                                            \
    do {                                                                                           \
        status = TVM_RT_WASM_RelaxExecutableLoad##_section_name##Section(&exec_mod->exec, reader); \
        if (unlikely(status)) {                                                                    \
            DBG("Relax Executable Load " TOSTRING(_section_name) " Section fail.");                \
            return status;                                                                         \
        }                                                                                          \
    } while (0)

    // load relax executable global section
    RelaxLoadSection(Global);
    // load relax executable memory scope section (V2 only)
    if (has_memory_scope_section) {
        RelaxLoadSection(MemoryScope);
    }
    // load relax executable constant section
    RelaxLoadSection(Constant);
    // load relax executable code section
    RelaxLoadSection(Code);

#undef RelaxLoadSection
    return 0;

load_fail:
    if (*out) {
        TVM_RT_WASM_RelaxExecutableModuleReleaseFunc(*out);
        *out = NULL;
    }
    return status;
}
