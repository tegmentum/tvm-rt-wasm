/**
 * @file device/cpu_memory.h
 * @brief CPU memory alloc/free interface.
 */

#ifndef TVM_RT_WASM_CORE_DEVICE_CPU_MEMORY_H_INCLUDE_
#define TVM_RT_WASM_CORE_DEVICE_CPU_MEMORY_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Up to the multiple of (1<<_bits) */
#define ALIGN_UP(_a, _bits) (((_a) + (1 << (_bits)) - 1) & (~((1 << (_bits)) - 1)))

/** @brief Default data alignment is 64 (1<<6) */
#ifndef DATA_ALIGNMENT_BITS
#define DATA_ALIGNMENT_BITS 6
#endif // DATA_ALIGNMENT_BITS

#include <stdlib.h>
#include <string.h>

/**
 * Zero-init at aligned allocation time. TIR kernels and VM builtins
 * routinely write to a subset of a destination tensor (e.g. VITS
 * `y_mask` fills only the first ~valid_length positions and expects
 * the tail to be zero). Without zeroing, uninit heap bytes leak into
 * results. Defensive baseline for the wasm CPU port — the mask/expand
 * paths in ONNX-frontend Relax modules depend on it.
 */
static inline void *TVM_RT_WASM_HeapMemoryAlignedAllocZeroed(size_t bytes) {
    size_t aligned = ALIGN_UP(bytes, DATA_ALIGNMENT_BITS);
    void *p = aligned_alloc((1 << DATA_ALIGNMENT_BITS), aligned);
    if (p != NULL) {
        memset(p, 0, aligned);
    }
    return p;
}

#define TVM_RT_WASM_HeapMemoryAlignedAlloc(bytes) TVM_RT_WASM_HeapMemoryAlignedAllocZeroed(bytes)
#define TVM_RT_WASM_HeapMemoryAlloc malloc
#define TVM_RT_WASM_HeapMemoryFree free

#define TVM_RT_WASM_WorkplaceMemoryAlignedAlloc TVM_RT_WASM_HeapMemoryAlignedAlloc
#define TVM_RT_WASM_WorkplaceMemoryAlloc malloc
#define TVM_RT_WASM_WorkplaceMemoryFree free

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_DEVICE_CPU_MEMORY_H_INCLUDE_
