/**
 * @file relax_vm.h
 * @brief Interfaces for relax virtual machine.
 */

#ifndef TVM_RT_WASM_RELAX_VM_H_INCLUDE_
#define TVM_RT_WASM_RELAX_VM_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <dlpack/dlpack.h>
#include <tvm/runtime/base.h>

/*
 * TVM 0.25 deleted `tvm/runtime/c_runtime_api.h`. This public header only
 * needs `TVM_DLL`, `DLDevice`, `DLTensor`, and an opaque module handle
 * from the old surface; declare the handle inline rather than dragging in
 * the fork's internal compat shim.
 */
typedef void *TVMModuleHandle;

#define TVM_RT_WASM_RelaxDefaultFunctionName "main"

typedef struct TVM_RT_WASM_RelaxVirtualMachine_st *TVM_RT_WASM_RelaxVirtualMachine;

/**
 * @brief Allocate a new TVM_RT_WASM_RelaxVirtualMachine and initialize it.
 * @param module_handle TVM relax executable library module. If NULL, use the system library.
 * @param devices runtime execution device.
 * @param num_dev the number of devices.
 * @note The function will get ownership of this module_handle if create successfully.
 *       **DO NOT** free this module_handle if create successfully!
 * @return Pointer of TVM_RT_WASM_RelaxVirtualMachine instance if successful, NULL if fail.
 */
TVM_DLL TVM_RT_WASM_RelaxVirtualMachine TVM_RT_WASM_RelaxVirtualMachineCreate(
    TVMModuleHandle module_handle, const DLDevice *devices, uint32_t num_dev);

/**
 * @brief Allocate a new TVM_RT_WASM_RelaxVirtualMachine bound to the
 * `<prefix>` sub-namespace of the process-global system library.
 *
 * When multiple Relax IRModules are compiled into the same wasm binary
 * with distinct `system_lib_prefix` attributes (M13.7 multi-model wasm
 * variants), each model's `__tvm_ffi__library_bin` / `library_ctx`
 * symbols land under a distinct prefix. This constructor picks the
 * matching pair and builds a Relax-executable root that imports the
 * shared kernels trie.
 *
 * Passing prefix=NULL or prefix="" is equivalent to
 * TVM_RT_WASM_RelaxVirtualMachineCreate(NULL, ...) — the legacy
 * no-prefix path.
 *
 * @param prefix TVM system_lib_prefix (or "" / NULL for no-prefix).
 * @param devices runtime execution device.
 * @param num_dev the number of devices.
 * @return Pointer of TVM_RT_WASM_RelaxVirtualMachine instance if successful, NULL if fail.
 */
TVM_DLL TVM_RT_WASM_RelaxVirtualMachine
TVM_RT_WASM_RelaxVirtualMachineCreateWithPrefix(const char *prefix, const DLDevice *devices,
                                                uint32_t num_dev);

/**
 * @brief Free the instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineFree(TVM_RT_WASM_RelaxVirtualMachine vm);

/**
 * @brief Execute the VM function.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineRun(TVM_RT_WASM_RelaxVirtualMachine vm,
                                               const char *func_name);

/**
 * @brief Set input to the vm based on index.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param index the index of inputs.
 * @param data_in The input data.
 * @note If the device is same, the function will zero copy.
 *       else, the function will copy `data_in` to vm input tensor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineSetInput(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                    const char *func_name, uint32_t index,
                                                    const DLTensor *data_in);

/**
 * @brief Set input to the vm based on name.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param name the name string for node.
 * @param data_in The input data.
 * @note If the device is same, the function will zero copy.
 *       else, the function will copy `data_in` to vm input tensor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineSetInputByName(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                          const char *func_name, const char *name,
                                                          const DLTensor *data_in);

/**
 * @brief Get output data for given output index.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param index The output index.
 * @param data_out The point to DLTensor. The function will copy vm output tensor to `data_out`.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetOutput(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                     const char *func_name, uint32_t index,
                                                     DLTensor *data_out);

/**
 * @brief Query the shape and dtype of an output tensor without copying it.
 *
 * `GetOutput` requires the caller to pass a pre-allocated `DLTensor` whose
 * shape and dtype match the VM's output. When the caller only knows the
 * function signature at load-time (not the concrete output extents — for
 * example a component-model host bridging into a WIT interface), this
 * helper exposes the metadata needed to size that allocation.
 *
 * Usage is two-pass: first call with `out_shape == NULL` to learn `ndim`
 * (and optionally `dtype`), allocate a shape buffer, then call again with
 * `out_shape` pointing at it. `shape_capacity` bounds how many dimensions
 * will be written; if it is smaller than `*out_ndim`, `-3` is returned and
 * `*out_ndim` / `*out_dtype` are still populated.
 *
 * @param vm             The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name      The function name. NULL uses the default "main".
 * @param index          The output index.
 * @param out_ndim       Non-NULL. Receives the number of dimensions.
 * @param out_dtype      Optional. If non-NULL, receives the tensor dtype.
 * @param out_shape      Optional. If non-NULL, receives the shape (int64).
 * @param shape_capacity Number of int64 slots in `out_shape`. Ignored when
 *                       `out_shape` is NULL.
 * @return 0 on success. -1 on VM lookup errors. -2 on NULL required args.
 *         -3 if `out_shape` was non-NULL but `shape_capacity` < ndim
 *         (ndim/dtype are still filled so the caller can retry).
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetOutputShape(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                          const char *func_name, uint32_t index,
                                                          int32_t *out_ndim,
                                                          DLDataType *out_dtype,
                                                          int64_t *out_shape,
                                                          int32_t shape_capacity);

/*-----------------Functions to get relax virtual machine information-----------------------------*/

/**
 * @brief Get the input index given the name of input.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param name The name of the input.
 * @return The index of input. If cannot find name or error, return -1.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetInputIndex(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                         const char *func_name, const char *name);

/**
 * @brief Get number of input tensors allocated.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @return integer number of input tensors.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetNumInputs(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                        const char *func_name);

/**
 * @brief Get number of output of current relax VM.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @return integer number of output tensors.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetNumOutputs(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                         const char *func_name);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_RELAX_VM_H_INCLUDE_
