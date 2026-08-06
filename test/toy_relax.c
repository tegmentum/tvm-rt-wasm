/*
 * @file test/toy_relax.c
 * @brief Milestone 13.2 correctness gate: wasmtime CLI driver for the
 * toy `main(x, w, b) = matmul(x, w) + b` Relax program compiled by
 * `test/toy_relax.py`.
 *
 * Flow:
 *  1. Relax-side devc.o's C++ constructor (via `__wasm_call_ctors`)
 *     runs BEFORE main and calls `TVMFFIEnvModRegisterSystemLibSymbol
 *     ("__tvm_ffi__library_bin", &__tvm_ffi__library_bin)` — the
 *     fork bridges that into the legacy `TVMBackendRegisterSystemLibSymbol`
 *     path so the sys-lib trie is populated with the executable blob.
 *  2. main() reads `x.bin`, `w.bin`, `b.bin` from a WASI preopen (the
 *     `test/toy_out/` dir the Python side wrote) then registers:
 *      - `__tvm_ffi__library_ctx` -> address of the weak var lib0.o
 *        exports; the sys-lib module init writes the Module* here.
 *      - `matmul` / `add` -> the TIR kernel entry points lib0.o exports
 *        under the `__tvm_ffi_` prefix. relax_vm.c looks up bound
 *        packed funcs by their bare name, so drop the prefix.
 *  3. Create the VM with a NULL module handle so it defaults to the
 *     system library, set inputs positionally, run "main", copy the
 *     output into `output.bin` in the preopen.
 *
 * The host-side Python driver (test/toy_relax.py) re-reads output.bin
 * and diffs it against the native-TVM oracle within f32 tolerance.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dlpack/dlpack.h>
#include <relax_vm.h>

/*
 * These come out of the compiled `toy_relax.tar` (see lib0.o + devc.o
 * symbols in test/toy_out/extracted/). They are provided at link time
 * — the fork does not know about the toy program's specific kernel
 * names; the driver does.
 */
extern void *__tvm_ffi__library_ctx; /* weak var, we set it via the trie */
extern int __tvm_ffi_matmul(void *self, const void *args, int32_t num_args, void *result);
extern int __tvm_ffi_add(void *self, const void *args, int32_t num_args, void *result);

/*
 * The fork's system_library.c owns this registrar. Not declared in any
 * public header; the driver re-declares it inline (same shape as the
 * one tvm_ffi_env_api.c uses).
 */
extern int TVMBackendRegisterSystemLibSymbol(const char *name, void *ptr);
extern const char *TVMGetLastError(void);

#define RUN(_expr, _msg)                                                                           \
    do {                                                                                           \
        int _status = (_expr);                                                                     \
        if (_status != 0) {                                                                        \
            fprintf(stderr, "[toy_relax_test] %s failed status=%d err=%s\n", _msg, _status,        \
                    TVMGetLastError() ? TVMGetLastError() : "(null)");                             \
            return _status;                                                                        \
        }                                                                                          \
    } while (0)

/* Read a whole file into a caller-allocated buffer. */
static int read_all(const char *path, void *buf, size_t expected_bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[toy_relax_test] cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    size_t got = fread(buf, 1, expected_bytes, fp);
    fclose(fp);
    if (got != expected_bytes) {
        fprintf(stderr, "[toy_relax_test] short read %s: %zu/%zu\n", path, got, expected_bytes);
        return -1;
    }
    return 0;
}

static int write_all(const char *path, const void *buf, size_t bytes) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[toy_relax_test] cannot open %s for write: %s\n", path, strerror(errno));
        return -1;
    }
    size_t put = fwrite(buf, 1, bytes, fp);
    fclose(fp);
    if (put != bytes) {
        fprintf(stderr, "[toy_relax_test] short write %s: %zu/%zu\n", path, put, bytes);
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

    /* Fixed shapes matching toy_relax.py: x[2,3] w[3,4] b[4] -> y[2,4]. */
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

    /*
     * Register the pieces the fork's system_library.c looks for by
     * exact string match: the library-ctx variable (whose address it
     * writes the created Module* into) and the TIR kernels the
     * Relax bytecode calls by short name.
     *
     * `__tvm_ffi__library_bin` is registered before main() runs by
     * devc.o's constructor; the fork's compat layer bridged
     * TVMFFIEnvModRegisterSystemLibSymbol -> TVMBackendRegisterSystemLibSymbol
     * so it lands in the same trie the loader queries here.
     */
    RUN(TVMBackendRegisterSystemLibSymbol("__tvm_ffi__library_ctx", &__tvm_ffi__library_ctx),
        "register library_ctx");
    RUN(TVMBackendRegisterSystemLibSymbol("matmul", (void *)&__tvm_ffi_matmul),
        "register matmul kernel");
    RUN(TVMBackendRegisterSystemLibSymbol("add", (void *)&__tvm_ffi_add), "register add kernel");

    DLDevice cpu = {kDLCPU, 0};
    DLDataType f32 = {kDLFloat, 32, 1};
    int64_t x_shape[2] = {2, 3};
    int64_t w_shape[2] = {3, 4};
    int64_t b_shape[1] = {4};
    int64_t y_shape[2] = {2, 4};

    DLTensor x_tensor = {.data = x_storage,
                         .device = cpu,
                         .ndim = 2,
                         .dtype = f32,
                         .shape = x_shape,
                         .strides = NULL,
                         .byte_offset = 0};
    DLTensor w_tensor = {.data = w_storage,
                         .device = cpu,
                         .ndim = 2,
                         .dtype = f32,
                         .shape = w_shape,
                         .strides = NULL,
                         .byte_offset = 0};
    DLTensor b_tensor = {.data = b_storage,
                         .device = cpu,
                         .ndim = 1,
                         .dtype = f32,
                         .shape = b_shape,
                         .strides = NULL,
                         .byte_offset = 0};
    DLTensor y_tensor = {.data = y_storage,
                         .device = cpu,
                         .ndim = 2,
                         .dtype = f32,
                         .shape = y_shape,
                         .strides = NULL,
                         .byte_offset = 0};

    fprintf(stderr, "[toy_relax_test] creating VM (system library)\n");
    TVM_RT_WASM_RelaxVirtualMachine vm =
        TVM_RT_WASM_RelaxVirtualMachineCreate(/* module_handle */ NULL, &cpu, 1);
    if (!vm) {
        fprintf(stderr, "[toy_relax_test] VM create failed: %s\n",
                TVMGetLastError() ? TVMGetLastError() : "(null)");
        return 1;
    }

    int n_in = TVM_RT_WASM_RelaxVirtualMachineGetNumInputs(vm, NULL);
    fprintf(stderr, "[toy_relax_test] main() num_inputs=%d\n", n_in);

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
    fprintf(stderr, "[toy_relax_test] wrote %s (%zu bytes)\n", path, sizeof(y_storage));

    TVM_RT_WASM_RelaxVirtualMachineFree(vm);
    return 0;
}
