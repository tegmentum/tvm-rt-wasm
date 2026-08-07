/*
 * @file test/toy_webgpu_test.c
 * @brief Milestone 14.5 correctness gate: WebGPU-target sibling of
 * `test/toy_relax.c`.
 *
 * Structural mirror of the CPU toy driver: reads x/w/b bin files from a
 * WASI preopen, builds a Relax VM against the system-lib module the
 * .init_array constructors populated, dispatches `main`, dumps
 * `output.bin` next to the inputs.
 *
 * Divergence from toy_relax.c:
 *
 *   1. Relax bytecode invokes ONE fused kernel `fused_matmul_add`
 *      (TVM 0.25's WebGPU codegen fuses matmul + elementwise-add
 *      into a single WGSL entry `fused_matmul_add_kernel`). The CPU
 *      target keeps them separate; the WebGPU host stub is one
 *      packed-func in lib0.o whose body dispatches into the WGSL
 *      submodule via the runtime's WebGPU device API.
 *
 *   2. VM device list includes kDLWebGPU alongside kDLCPU — the fork's
 *      RelaxVirtualMachine picks up the WebGPU device API (which drives
 *      the WIT-bridged WGPU_* calls) when the compiled module tags a
 *      kernel as WebGPU-target.
 *
 *   3. Runs under Node + JSPI (not wasmtime): wasmtime has no WebGPU
 *      binding, so the accompanying JS driver
 *      (cognition/web/test/toy-webgpu.test.mjs) instantiates this
 *      wasm binary as a component with `host:webgpu` satisfied by
 *      cognition's `web/src/adopt/webgpu-host-impl.js`.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dlpack/dlpack.h>
#include <relax_vm.h>

/* Symbols from the compiled WebGPU-target `lib0.o` — the host-side
 * dispatch stub for the fused matmul+add op + the weak library-ctx
 * slot the sys-lib loader writes the created Module* into. */
extern void *__tvm_ffi__library_ctx;
extern int __tvm_ffi_fused_matmul_add(void *self, const void *args, int32_t num_args, void *result);
extern int __tvm_ffi_main(void *self, const void *args, int32_t num_args, void *result);

extern int TVMBackendRegisterSystemLibSymbol(const char *name, void *ptr);
extern const char *TVMGetLastError(void);

#define RUN(_expr, _msg)                                                                           \
    do {                                                                                           \
        int _status = (_expr);                                                                     \
        if (_status != 0) {                                                                        \
            fprintf(stderr, "[toy_webgpu_test] %s failed status=%d err=%s\n", _msg, _status,       \
                    TVMGetLastError() ? TVMGetLastError() : "(null)");                             \
            return _status;                                                                        \
        }                                                                                          \
    } while (0)

static int read_all(const char *path, void *buf, size_t bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[toy_webgpu_test] cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    size_t got = fread(buf, 1, bytes, fp);
    fclose(fp);
    if (got != bytes) {
        fprintf(stderr, "[toy_webgpu_test] short read %s: %zu/%zu\n", path, got, bytes);
        return -1;
    }
    return 0;
}

static int write_all(const char *path, const void *buf, size_t bytes) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[toy_webgpu_test] cannot open %s for write: %s\n", path, strerror(errno));
        return -1;
    }
    size_t put = fwrite(buf, 1, bytes, fp);
    fclose(fp);
    if (put != bytes) {
        fprintf(stderr, "[toy_webgpu_test] short write %s: %zu/%zu\n", path, put, bytes);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <dir-with-x.bin-w.bin-b.bin>\n", argv[0]);
        fprintf(stderr, "  writes output.bin into the same directory\n");
        return 2;
    }
    const char *dir = argv[1];
    char path[1024];

    static float x_storage[2 * 3];
    static float w_storage[3 * 4];
    static float b_storage[4];
    static float y_storage[2 * 4];

    snprintf(path, sizeof(path), "%s/x.bin", dir);
    RUN(read_all(path, x_storage, sizeof(x_storage)), "read x.bin");
    snprintf(path, sizeof(path), "%s/w.bin", dir);
    RUN(read_all(path, w_storage, sizeof(w_storage)), "read w.bin");
    snprintf(path, sizeof(path), "%s/b.bin", dir);
    RUN(read_all(path, b_storage, sizeof(b_storage)), "read b.bin");

    RUN(TVMBackendRegisterSystemLibSymbol("__tvm_ffi__library_ctx", &__tvm_ffi__library_ctx),
        "register library_ctx");
    RUN(TVMBackendRegisterSystemLibSymbol("fused_matmul_add", (void *)&__tvm_ffi_fused_matmul_add),
        "register fused_matmul_add kernel");
    RUN(TVMBackendRegisterSystemLibSymbol("main", (void *)&__tvm_ffi_main), "register main");

    /* CPU host tensors — the VM copies these into WebGPU device
     * memory as part of SetInput; readback happens during GetOutput
     * via WGPU_MemoryCopyDtoH. */
    DLDevice cpu = {kDLCPU, 0};
    DLDevice gpu = {kDLWebGPU, 0};
    DLDataType f32 = {kDLFloat, 32, 1};
    int64_t x_shape[2] = {2, 3};
    int64_t w_shape[2] = {3, 4};
    int64_t b_shape[1] = {4};
    int64_t y_shape[2] = {2, 4};

    DLTensor x_tensor = {.data = x_storage, .device = cpu, .ndim = 2, .dtype = f32,
                         .shape = x_shape, .strides = NULL, .byte_offset = 0};
    DLTensor w_tensor = {.data = w_storage, .device = cpu, .ndim = 2, .dtype = f32,
                         .shape = w_shape, .strides = NULL, .byte_offset = 0};
    DLTensor b_tensor = {.data = b_storage, .device = cpu, .ndim = 1, .dtype = f32,
                         .shape = b_shape, .strides = NULL, .byte_offset = 0};
    DLTensor y_tensor = {.data = y_storage, .device = cpu, .ndim = 2, .dtype = f32,
                         .shape = y_shape, .strides = NULL, .byte_offset = 0};

    fprintf(stderr, "[toy_webgpu_test] creating VM (system library, WebGPU device)\n");
    /* Two-device list — WebGPU first (index 0, the primary compute device
     * that the compiled bytecode's alloc_storage/set-input paths target
     * via devices[0]), CPU second (index 1) for host-side helpers. TVM's
     * Relax lowering with `Target("webgpu", host=llvm)` bakes device_index=0
     * into the alloc-storage instructions for kernel outputs. */
    DLDevice dev_list[2] = {gpu, cpu};
    TVM_RT_WASM_RelaxVirtualMachine vm =
        TVM_RT_WASM_RelaxVirtualMachineCreate(/* module_handle */ NULL, dev_list, 2);
    if (!vm) {
        fprintf(stderr, "[toy_webgpu_test] VM create failed: %s\n",
                TVMGetLastError() ? TVMGetLastError() : "(null)");
        return 1;
    }

    int n_in = TVM_RT_WASM_RelaxVirtualMachineGetNumInputs(vm, NULL);
    fprintf(stderr, "[toy_webgpu_test] main() num_inputs=%d\n", n_in);

    RUN(TVM_RT_WASM_RelaxVirtualMachineSetInput(vm, NULL, 0, &x_tensor), "set input 0 (x)");
    RUN(TVM_RT_WASM_RelaxVirtualMachineSetInput(vm, NULL, 1, &w_tensor), "set input 1 (w)");
    RUN(TVM_RT_WASM_RelaxVirtualMachineSetInput(vm, NULL, 2, &b_tensor), "set input 2 (b)");
    RUN(TVM_RT_WASM_RelaxVirtualMachineRun(vm, NULL), "run main");
    RUN(TVM_RT_WASM_RelaxVirtualMachineGetOutput(vm, NULL, 0, &y_tensor), "get output 0");

    for (int i = 0; i < 8; ++i) {
        fprintf(stderr, "  y[%d]=%.6f\n", i, y_storage[i]);
    }

    snprintf(path, sizeof(path), "%s/output.bin", dir);
    RUN(write_all(path, y_storage, sizeof(y_storage)), "write output.bin");
    fprintf(stderr, "[toy_webgpu_test] wrote %s (%zu bytes)\n", path, sizeof(y_storage));

    TVM_RT_WASM_RelaxVirtualMachineFree(vm);
    return 0;
}
