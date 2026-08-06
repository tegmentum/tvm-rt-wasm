/**
 * @file tvm_runtime/tvm_ffi_env_api.c
 * @brief Bridges of the TVM 0.25 `tvm/ffi/extra/c_env_api.h` surface into
 * the fork's existing system-lib registration + error propagation paths.
 *
 * Every Relax `.tar` built with `system_lib=True` includes a `devc.o`
 * whose C++ constructor calls `TVMFFIEnvModRegisterSystemLibSymbol` to
 * register the library blob, and TIR kernels in `lib0.o` that call
 * `TVMFFIErrorSetRaisedFromCStrParts` to raise structured errors. Neither
 * symbol existed pre-0.25. The fork's system_library.c still speaks the
 * legacy `TVMBackendRegisterSystemLibSymbol` name — bridge one onto the
 * other so linking generated code against `libtvm-rt-core.a` resolves.
 *
 * The lookup / context registrations are stubs today: the toy driver in
 * `test/toy_relax.c` sidesteps them by pre-registering TIR kernels
 * under short names via `TVMBackendRegisterSystemLibSymbol` directly.
 * VITS integration (13.3+) will likely need real implementations.
 */

#include <stdio.h>
#include <string.h>

#include <tvm_compat.h>
#include <utils/common.h>

/*
 * TVM 0.25 removed the `TVMBackendRegisterSystemLibSymbol` declaration
 * from public headers; the fork still implements it in system_library.c
 * as the sole entry point into `sys_lib_symbols`. Re-declare it here so
 * this TU can forward into it.
 */
TVM_DLL int TVMBackendRegisterSystemLibSymbol(const char *name, void *ptr);

/*
 * `TVMFFIEnvModRegisterSystemLibSymbol` is what modern TVM's LLVM code
 * generator emits inside `devc.o`'s constructor to register the
 * `__tvm_ffi__library_bin` blob (see tvm/ffi/extra/module.h). Forward
 * straight into the fork's legacy registrar — same semantics: stash
 * (name -> ptr) in the sys-lib trie.
 */
TVM_DLL int TVMFFIEnvModRegisterSystemLibSymbol(const char *name, void *symbol) {
    return TVMBackendRegisterSystemLibSymbol(name, symbol);
}

/*
 * `TVMFFIEnvModRegisterContextSymbol` is used to inject weak-linkage
 * runtime hooks the library expects to find (custom allocators,
 * environment lookups). For the fork's static-link CPU path the runtime
 * *is* the environment, so there is nothing to register — accept and
 * discard. If future workloads need real context symbols the stub will
 * need to grow a per-name switch.
 */
TVM_DLL int TVMFFIEnvModRegisterContextSymbol(const char *name, void *symbol) {
    (void)name;
    (void)symbol;
    return 0;
}

/*
 * `TVMFFIEnvModLookupFromImports` is the callee-side helper generated
 * code calls to resolve inter-module PackedFunc lookups (e.g. one .o
 * calling another .o's exported function). The toy driver flattens the
 * module graph — every kernel is registered against the root system-lib
 * — so this lookup is never reached. Return -1 to make an unexpected
 * hit visible instead of silently returning garbage.
 */
TVM_DLL int TVMFFIEnvModLookupFromImports(void *library_ctx, const char *func_name, void **out) {
    (void)library_ctx;
    (void)func_name;
    if (out) {
        *out = NULL;
    }
    fprintf(stderr, "TVMFFIEnvModLookupFromImports: unimplemented lookup for `%s`\n",
            func_name ? func_name : "(null)");
    return -1;
}

/*
 * TIR kernels call `TVMFFIErrorSetRaisedFromCStrParts` to publish an
 * error like `RuntimeError | "matmul: shape mismatch"`. The fork
 * routes runtime errors through `TVMAPISetLastError`; join the parts
 * (skipping NULLs per the c_api.h contract) into the global buffer so
 * `TVMGetLastError` returns a readable string.
 */
TVM_DLL void TVMFFIErrorSetRaisedFromCStrParts(const char *kind, const char **message_parts,
                                               int32_t num_parts) {
    char buf[GLOBAL_BUF_SIZE];
    size_t off = 0;

    if (kind) {
        int written = snprintf(buf + off, sizeof(buf) - off, "[%s] ", kind);
        if (written > 0) {
            off += (size_t)written;
            if (off >= sizeof(buf)) {
                off = sizeof(buf) - 1;
            }
        }
    }
    for (int32_t i = 0; i < num_parts && off < sizeof(buf); ++i) {
        if (message_parts[i] == NULL) {
            continue;
        }
        size_t remain = sizeof(buf) - 1 - off;
        size_t part_len = strnlen(message_parts[i], remain);
        memcpy(buf + off, message_parts[i], part_len);
        off += part_len;
    }
    buf[off] = 0;
    TVMAPISetLastError(buf);
}
