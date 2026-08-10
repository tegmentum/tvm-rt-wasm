/**
 * @file webgpu/c_api/webgpu_c_api.h
 * @brief WebGPU sync api wrapper for native and js.
 */

#ifndef TVM_RT_WASM_WEBGPU_WEBGPU_C_API_H
#define TVM_RT_WASM_WEBGPU_WEBGPU_C_API_H

#include <stddef.h>
#include <stdint.h>

#include <dlpack/dlpack.h>

typedef struct WGPU_Device_st *WGPU_Device;

typedef struct WGPU_Memory_st *WGPU_Memory;

typedef struct WGPU_Function_st *WGPU_Function;

/**
 * @brief Get the device.
 * @param device_ptr The pointer to receive device.
 * @return 0 if success.
 */
int WGPU_DeviceGet(WGPU_Device *device_ptr);

/**
 * @brief Free the device instance.
 * @param device The device instance.
 * @return 0 if success.
 */
int WGPU_DeviceFree(WGPU_Device device);

/**
 * @brief Alloc device memory.
 * @param device The device where memory in.
 * @param memory_ptr The pointer to receive allocated device memory.
 * @param nbytes The number of bytes to alloc.
 * @return 0 if success.
 */
int WGPU_MemoryAlloc(WGPU_Device device, WGPU_Memory *memory_ptr, size_t nbytes);

/**
 * @brief Free the device memory.
 * @param memory The device memory to free.
 * @return 0 if success.
 */
int WGPU_MemoryFree(WGPU_Memory memory);

/**
 * @brief Copy memory from host to device.
 * @param dst device memory.
 * @param dst_byte_offset offset in bytes to the beginning pointer to dst data.
 * @param src source host memory.
 * @param src_byte_offset offset in bytes to the beginning pointer to src data.
 * @param nbytes the number of bytes to copy.
 * @return 0 if success.
 */
int WGPU_MemoryCopyHtoD(WGPU_Memory dst, size_t dst_byte_offset, const void *src,
                        size_t src_byte_offset, size_t nbytes);

/**
 * @brief Copy memory from device to host.
 * @param dst host memory.
 * @param dst_byte_offset offset in bytes to the beginning pointer to dst data.
 * @param src source device memory.
 * @param src_byte_offset offset in bytes to the beginning pointer to src data.
 * @param nbytes the number of bytes to copy.
 * @return 0 if success.
 */
int WGPU_MemoryCopyDtoH(void *dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset,
                        size_t nbytes);

/**
 * @brief Copy memory from device to device.
 * @param dst device memory.
 * @param dst_byte_offset offset in bytes to the beginning pointer to dst data.
 * @param src source device memory.
 * @param src_byte_offset offset in bytes to the beginning pointer to src data.
 * @param nbytes the number of bytes to copy.
 * @return 0 if success.
 */
int WGPU_MemoryCopyDtoD(WGPU_Memory dst, size_t dst_byte_offset, WGPU_Memory src,
                        size_t src_byte_offset, size_t nbytes);

/**
 * @brief Create device function.
 * @param device The device where function in.
 * @param func_ptr The pointer to receive created function instance.
 * @param source The text device source code.
 * @param source_len The length of text source code.
 * @param entry_name The name of entry point.
 * @param entry_name_len The length of entry point name.
 * @param num_handle_args The number of handle (tensor / storage-buffer) kernel arguments.
 * @param handle_write_access Per-handle write-access hints parsed from the compiled
 *        module's `paramWriteAccess:[...]` launch-param tag. 1 = writable
 *        (`var<storage, read_write>` in WGSL, `storage-buffer` binding kind); 0 = read-only
 *        (`var<storage, read>` in WGSL, `read-only-storage-buffer` binding kind). Size must
 *        equal @p num_handle_args. Pass NULL to default all handles to writable.
 * @param num_pod_args The number of scalar POD kernel arguments (int / uint / float).
 * @param pod_arg_dtypes The dtype (int/uint/float) of each POD arg — needed at dispatch
 *        time to pack values into the shader's PODArgs uniform buffer. Size must equal
 *        @p num_pod_args (may be NULL when 0).
 * @return 0 if success.
 *
 * @note TVM 0.25 WebGPU codegen always emits an extra uniform buffer at binding index
 *       `num_handle_args` (the `PODArgs` struct in the generated WGSL) — even when
 *       @p num_pod_args is zero, the struct still carries the `packGridDimX` field
 *       used by the codegen's over-65536-workgroup spread guard.
 */
int WGPU_FunctionCreate(WGPU_Device device, WGPU_Function *func_ptr, const char *source,
                        uint32_t source_len, const char *entry_name, uint32_t entry_name_len,
                        uint32_t num_handle_args, const uint8_t *handle_write_access,
                        uint32_t num_pod_args, const DLDataType *pod_arg_dtypes);

/**
 * @brief Submit function to gpu to run.
 * @param function The function instance.
 * @param handle_args The device-memory storage buffer arguments (size num_handle_args).
 * @param num_handle_args The number of handle (storage buffer) arguments.
 * @param pod_arg_values Raw 8-byte payloads of each POD scalar kernel arg (v_int64 slot
 *        from the wrapper's TVMFFIAny args, interpreted per the dtype recorded at
 *        WGPU_FunctionCreate time). Size must equal num_pod_args (may be NULL when 0).
 * @param num_pod_args The number of POD scalar kernel arguments.
 * @param grid_dim_x The x dim of compute work groups.
 * @param grid_dim_y The y dim of compute work groups.
 * @param grid_dim_z The z dim of compute work groups.
 * @return 0 if success.
 */
int WGPU_FunctionRun(WGPU_Function function, const WGPU_Memory *handle_args,
                     uint32_t num_handle_args, const uint64_t *pod_arg_values,
                     uint32_t num_pod_args, size_t grid_dim_x, size_t grid_dim_y,
                     size_t grid_dim_z);

/**
 * @brief Free the device function.
 * @param function The function to free.
 * @return 0 if success.
 */
int WGPU_FunctionFree(WGPU_Function function);

/**
 * @brief Open a kernel-dispatch batch on @p device.
 *
 * Between WGPU_BeginKernelBatch and WGPU_EndKernelBatch (or an implicit
 * flush via WGPU_FlushKernelBatch), WGPU_FunctionRun accumulates its
 * dispatches into a single shared command encoder + compute pass instead
 * of opening/finishing/submitting a fresh encoder per kernel. At flush
 * time the encoder is finished, the resulting command buffer submitted,
 * and per-dispatch bind-groups / shadow buffers dropped in bulk.
 *
 * This is the BATCH-TVM mitigation described in
 * `docs/tvm-boundary-overhead-investigation.md` §4.A: cuts per-kernel WIT
 * host-boundary crossings on the compute path from 13 to ~5 by sharing
 * encoder, pass, and submit across N kernels. For an N=200-kernel VITS
 * decoder inference, `2600 → ~1000` boundary crossings on the kernel
 * portion alone.
 *
 * Nested Begin/End calls are ignored — only the outermost pair transitions
 * the batch state. Calls to WGPU_MemoryCopyDtoH and WGPU_MemoryCopyDtoD
 * implicitly flush before proceeding (readback needs completed GPU state;
 * DtoD copy-buffer-to-buffer must sequence via its own encoder). HtoD
 * queue-write-buffer also flushes so a mid-batch buffer overwrite can't
 * race a still-unsubmitted compute pass that reads the same buffer.
 *
 * Backward-compatible: outside a batch, WGPU_FunctionRun preserves the
 * pre-BATCH-TVM per-kernel encoder/pass/submit path unchanged.
 *
 * @param device The WebGPU device to open a batch on.
 * @return 0 if success.
 */
int WGPU_BeginKernelBatch(WGPU_Device device);

/**
 * @brief Flush any pending batched work: close the current compute pass
 *        (if open), finish the shared encoder, submit the resulting
 *        command buffer, then drop retained bind-groups and destroy
 *        retained shadow buffers. Batch remains active — the next
 *        WGPU_FunctionRun starts a fresh encoder/pass.
 *
 * No-op if called outside a batch or with nothing pending. Called
 * implicitly by WGPU_MemoryCopyDtoH / WGPU_MemoryCopyDtoD /
 * WGPU_MemoryCopyHtoD so callers rarely need to invoke it directly.
 *
 * @param device The WebGPU device whose batch to flush.
 * @return 0 if success.
 */
int WGPU_FlushKernelBatch(WGPU_Device device);

/**
 * @brief End the kernel-dispatch batch: flush any pending work, then
 *        clear the active flag. Subsequent WGPU_FunctionRun calls revert
 *        to the unbatched per-kernel encoder/pass/submit path.
 *
 * Idempotent — calling End without a matching Begin is a no-op.
 *
 * @param device The WebGPU device whose batch to close.
 * @return 0 if success.
 */
int WGPU_EndKernelBatch(WGPU_Device device);

#endif // TVM_RT_WASM_WEBGPU_WEBGPU_C_API_H
