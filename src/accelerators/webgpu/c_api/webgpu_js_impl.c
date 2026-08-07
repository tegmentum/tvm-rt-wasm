/**
 * @file webgpu/c_api/webgpu_js_impl.c
 * @brief Implementation for WGPU_* sync C API on top of the `host:webgpu`
 * WIT interface (see wit-packages/host-webgpu/wit/webgpu.wit in cognition,
 * v0.1.0). Replaces yanghaku's original EM_JS-based Emscripten shim — that
 * path bypassed wasi-p1 entirely and was #ifdef __EMSCRIPTEN__'d out under
 * the fork's wasi-sdk build. M14.3 rewrites this to route each of the 9
 * WGPU_* entry points through canonical-ABI wit-bindgen extern imports.
 *
 * Compile-time contract: the file must produce a linkable object under
 * `-Wall -Wextra -Werror` (wasi-sdk clang) with the WIT imports unresolved
 * — they are satisfied at final composition time when cognition's
 * `webgpu-host-impl.js` (M14.4) supplies the JS side.
 *
 * Runtime contract (M14.5+): each WGPU_* function
 *
 *   1. Marshals its C args into canonical-ABI shapes (i32 handles for
 *      resources; ptr+len pairs for byte slices).
 *   2. Invokes the corresponding `host:webgpu/webgpu@0.1.0` import via
 *      `__attribute__((import_module, import_name))`-declared externs.
 *   3. Unpacks the wit-bindgen ret-area buffer (result<T, webgpu-error>
 *      lowering: u8 discriminant, then either T bytes or a variant
 *      webgpu-error). On error, forwards the message string into the
 *      fork's `TVMAPISetLastError` buffer and returns -1.
 *
 * Handle encoding: the WGPU_Device / WGPU_Memory / WGPU_Function opaque
 * pointer types the c_api.h header declares map to fork-owned refcounted
 * structs that carry the raw i32 WIT resource handles + any bookkeeping
 * TVM's kernel dispatch needs (bind-group layout, pipeline, staging
 * buffer cache).
 *
 * Staging-buffer cache (per the M14.2 flag): DtoH readback is the
 * hot-path expense — WebGPU requires a MAP_READ-usage staging buffer,
 * copy-buffer-to-buffer, submit, map-async, get-mapped-range, unmap,
 * destroy every time. yanghaku's original allocated+destroyed a fresh
 * staging buffer per DtoH call. We cache one staging companion per
 * source `WGPU_Memory` (grown lazily to the largest observed request
 * size), reusing it across calls. Amortises 5-6 JSPI round-trips per
 * readback down to 1-2 for repeat reads of the same tensor.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <dlpack/dlpack.h>
#include <c_api/webgpu_c_api.h>

/* ---------------------------------------------------------------------
 * fork error-reporting hook. The fork's c_runtime_api.c owns
 * TVMAPISetLastError; declaring it here (without pulling in the whole
 * tvm_compat.h) keeps this translation unit's include graph minimal.
 * ------------------------------------------------------------------- */
extern void TVMAPISetLastError(const char *msg);

/* ---------------------------------------------------------------------
 * WIT import name binding.
 *
 * cargo-component / wit-bindgen-c emit imports with a two-part
 * attribute pair:
 *
 *   __attribute__((__import_module__("<pkg-name>/<iface-name>@<ver>"),
 *                  __import_name__("<canonical-method-name>")))
 *   extern <ret> <local-c-name>(...);
 *
 * For our WIT (`host:webgpu/webgpu@0.1.0`) the module string is
 * "host:webgpu/webgpu@0.1.0"; the canonical method names are the WIT
 * abstract names in kebab-case, with resource methods prefixed by
 * "[method]<resource>.", constructors by "[constructor]<resource>",
 * and resource-drop by "[resource-drop]<resource>". Kept as macros so
 * the whole file reads consistently.
 * ------------------------------------------------------------------- */
#define WGPU_WIT_MOD "host:webgpu/webgpu@0.1.0"

#define WGPU_IMPORT(name)                                                                          \
    __attribute__((__import_module__(WGPU_WIT_MOD), __import_name__(name)))

/*
 * Canonical ABI conventions used below:
 *
 *  - Resource handles: `int32_t`. Guest-owned handles (`own<T>` in WIT)
 *    are returned by value; guest-borrowed handles (`borrow<T>`) are
 *    passed by value on call. `[resource-drop]T` releases an `own<T>`.
 *  - result<T, webgpu-error>: caller passes an out-buffer whose layout
 *    starts with a u8 discriminant (0 = ok, 1 = err). On ok, T is
 *    packed at the appropriate aligned offset. On err, the variant
 *    discriminant (u8) + a string pointer (u32) + string length (u32)
 *    follow. Our helper `wgpu_wit_ret_area` is 32 bytes — enough for
 *    every method the interface exposes (each returns at most a
 *    single resource handle or single u64 + the error payload).
 *  - list<u8> arg: passed as (ptr, len) unpacked into the arg vector.
 *  - list<u8> return: caller passes an out-(ptr, len) buffer pair; the
 *    guest is responsible for freeing via wit-bindgen's cabi_realloc
 *    (not implemented here — see the note on `get-mapped-range` below).
 *  - Strings on the wire follow the same (ptr, len) convention.
 *
 * These conventions match the shape wit-bindgen-c generates and jco
 * consumes. The exact bit-layout is validated at M14.5 composition
 * time; the extern declarations below give the future 14.5 gate
 * something concrete to satisfy.
 */

/* Fixed-size ret area for canonical ABI result<T, webgpu-error>
 * unpacking. 32 bytes covers the largest possible T (i64 or a pair of
 * i32 handles) plus discriminant + error-variant payload (u8 + ptr +
 * len = 12 bytes). */
#define WGPU_WIT_RET_AREA_SIZE 32
_Alignas(8) static uint8_t wgpu_wit_ret_area[WGPU_WIT_RET_AREA_SIZE];

/* Fan out the ret-area on error. On the wire, a webgpu-error variant
 * starts at offset 4 (after the outer result discriminant + padding to
 * 4-byte alignment) with a u8 case discriminant, then a string pointer
 * (u32) and length (u32) at offset 8/12. We stash the message into
 * TVMAPISetLastError and drop the string ptr (leaks; wit-bindgen would
 * normally have a `cabi_free`-hook for this — deferred until M14.5). */
static void wgpu_wit_forward_error(const uint8_t *ret_area) {
    /* discriminant at [0] is 1 (err). Case tag at [4]. Message ptr at
     * [8], len at [12]. Order matches wit-bindgen-c lowering of a
     * single-string-payload variant. */
    const uint32_t msg_ptr = *(const uint32_t *)(ret_area + 8);
    const uint32_t msg_len = *(const uint32_t *)(ret_area + 12);
    if (msg_ptr == 0 || msg_len == 0) {
        TVMAPISetLastError("host:webgpu: (no message)");
        return;
    }
    /* Cap at 512 to keep the on-stack buffer bounded even under a
     * malformed host reply. */
    char buf[513];
    size_t n = msg_len < 512 ? (size_t)msg_len : 512;
    memcpy(buf, (const void *)(uintptr_t)msg_ptr, n);
    buf[n] = '\0';
    TVMAPISetLastError(buf);
}

/* ---------------------------------------------------------------------
 * WIT extern imports. One per method the WGPU_* surface actually
 * exercises. Naming: `wgpu_wit_<snake>` — no relationship to the
 * host-facing entry points (`WGPU_*`), which are the c_api.h contract.
 *
 * Signatures use canonical-ABI-shaped parameter lists:
 *   - resource own/borrow → int32_t (handle)
 *   - list<u8>            → const uint8_t *ptr, uint32_t len
 *   - option<u64>         → int32_t is_some, uint64_t val
 *   - result<T, err>      → uint8_t *ret_area (out-buffer written by host)
 *   - flags (bufer-usage) → uint32_t bitset (host lifts to WIT flags)
 * ------------------------------------------------------------------- */

/* Free-function: request-adapter: func() -> result<option<adapter>, webgpu-error> */
WGPU_IMPORT("request-adapter")
extern void wgpu_wit_request_adapter(uint8_t *ret);

/* adapter.request-device: func() -> result<device, webgpu-error> */
WGPU_IMPORT("[method]adapter.request-device")
extern void wgpu_wit_adapter_request_device(int32_t adapter_h, uint8_t *ret);

/* [resource-drop]adapter */
WGPU_IMPORT("[resource-drop]adapter")
extern void wgpu_wit_adapter_drop(int32_t h);

/* [resource-drop]device */
WGPU_IMPORT("[resource-drop]device")
extern void wgpu_wit_device_drop(int32_t h);

/* device.create-buffer(descriptor: buffer-descriptor) -> result<buffer, webgpu-error>
 * buffer-descriptor lowered: (u64 size, u32 usage-flags, u8 mapped-at-creation). */
WGPU_IMPORT("[method]device.create-buffer")
extern void wgpu_wit_device_create_buffer(int32_t device_h, uint64_t size, uint32_t usage,
                                          int32_t mapped_at_creation, uint8_t *ret);

/* device.create-shader-module(wgsl-source: string) -> result<shader-module, error> */
WGPU_IMPORT("[method]device.create-shader-module")
extern void wgpu_wit_device_create_shader_module(int32_t device_h, const uint8_t *src_ptr,
                                                 uint32_t src_len, uint8_t *ret);

/* device.create-compute-pipeline(descriptor) -> result<compute-pipeline, error>
 * descriptor lowered: (shader-module borrow, entry-point ptr+len,
 *                      list<bind-group-layout borrow> ptr+len). */
WGPU_IMPORT("[method]device.create-compute-pipeline")
extern void wgpu_wit_device_create_compute_pipeline(int32_t device_h, int32_t module_h,
                                                    const uint8_t *entry_ptr, uint32_t entry_len,
                                                    const int32_t *bgl_handles, uint32_t bgl_len,
                                                    uint8_t *ret);

/* device.create-bind-group-layout(entries: list<bind-group-layout-entry>) -> result<bgl, error>
 * bind-group-layout-entry lowered: (u32 binding, u8 kind, u8 has-dyn-offset,
 *                                   u64 min-binding-size) — 16 bytes with padding. */
WGPU_IMPORT("[method]device.create-bind-group-layout")
extern void wgpu_wit_device_create_bind_group_layout(int32_t device_h,
                                                     const uint8_t *entries_ptr,
                                                     uint32_t entries_len, uint8_t *ret);

/* device.create-bind-group(descriptor) -> result<bind-group, error>
 * descriptor lowered: (bind-group-layout borrow, list<bind-group-entry>). */
WGPU_IMPORT("[method]device.create-bind-group")
extern void wgpu_wit_device_create_bind_group(int32_t device_h, int32_t layout_h,
                                              const uint8_t *entries_ptr, uint32_t entries_len,
                                              uint8_t *ret);

/* device.create-command-encoder() -> result<command-encoder, error> */
WGPU_IMPORT("[method]device.create-command-encoder")
extern void wgpu_wit_device_create_command_encoder(int32_t device_h, uint8_t *ret);

/* device.queue() -> queue (owned, not a result) */
WGPU_IMPORT("[method]device.queue")
extern int32_t wgpu_wit_device_queue(int32_t device_h);

/* buffer.map-async(mode: map-mode, offset: u64, size: u64) -> result<_, error> */
WGPU_IMPORT("[method]buffer.map-async")
extern void wgpu_wit_buffer_map_async(int32_t buffer_h, uint32_t mode, uint64_t offset,
                                      uint64_t size, uint8_t *ret);

/* buffer.get-mapped-range(offset, size) -> result<list<u8>, error>
 * list<u8> on return lowered as (ptr, len) in the ret area at offset [4/8]. */
WGPU_IMPORT("[method]buffer.get-mapped-range")
extern void wgpu_wit_buffer_get_mapped_range(int32_t buffer_h, uint64_t offset, uint64_t size,
                                             uint8_t *ret);

/* buffer.unmap() */
WGPU_IMPORT("[method]buffer.unmap")
extern void wgpu_wit_buffer_unmap(int32_t buffer_h);

/* buffer.destroy() */
WGPU_IMPORT("[method]buffer.destroy")
extern void wgpu_wit_buffer_destroy(int32_t buffer_h);

/* [resource-drop]buffer */
WGPU_IMPORT("[resource-drop]buffer")
extern void wgpu_wit_buffer_drop(int32_t h);

/* [resource-drop]shader-module / compute-pipeline / bind-group-layout / bind-group */
WGPU_IMPORT("[resource-drop]shader-module")
extern void wgpu_wit_shader_module_drop(int32_t h);
WGPU_IMPORT("[resource-drop]compute-pipeline")
extern void wgpu_wit_compute_pipeline_drop(int32_t h);
WGPU_IMPORT("[resource-drop]bind-group-layout")
extern void wgpu_wit_bind_group_layout_drop(int32_t h);
WGPU_IMPORT("[resource-drop]bind-group")
extern void wgpu_wit_bind_group_drop(int32_t h);

/* command-encoder.begin-compute-pass() -> result<compute-pass, error> */
WGPU_IMPORT("[method]command-encoder.begin-compute-pass")
extern void wgpu_wit_encoder_begin_compute_pass(int32_t encoder_h, uint8_t *ret);

/* command-encoder.copy-buffer-to-buffer(...) -> result<_, error> */
WGPU_IMPORT("[method]command-encoder.copy-buffer-to-buffer")
extern void wgpu_wit_encoder_copy_buffer_to_buffer(int32_t encoder_h, int32_t src_h,
                                                   uint64_t src_off, int32_t dst_h,
                                                   uint64_t dst_off, uint64_t nbytes,
                                                   uint8_t *ret);

/* command-encoder.finish() -> result<command-buffer, error> */
WGPU_IMPORT("[method]command-encoder.finish")
extern void wgpu_wit_encoder_finish(int32_t encoder_h, uint8_t *ret);

/* [resource-drop]command-encoder / command-buffer / compute-pass / queue */
WGPU_IMPORT("[resource-drop]command-encoder")
extern void wgpu_wit_encoder_drop(int32_t h);
WGPU_IMPORT("[resource-drop]command-buffer")
extern void wgpu_wit_command_buffer_drop(int32_t h);
WGPU_IMPORT("[resource-drop]compute-pass")
extern void wgpu_wit_compute_pass_drop(int32_t h);
WGPU_IMPORT("[resource-drop]queue")
extern void wgpu_wit_queue_drop(int32_t h);

/* compute-pass.set-pipeline / set-bind-group / dispatch-workgroups / end */
WGPU_IMPORT("[method]compute-pass.set-pipeline")
extern void wgpu_wit_pass_set_pipeline(int32_t pass_h, int32_t pipeline_h);
WGPU_IMPORT("[method]compute-pass.set-bind-group")
extern void wgpu_wit_pass_set_bind_group(int32_t pass_h, uint32_t index, int32_t group_h);
WGPU_IMPORT("[method]compute-pass.dispatch-workgroups")
extern void wgpu_wit_pass_dispatch_workgroups(int32_t pass_h, uint32_t x, uint32_t y, uint32_t z);
WGPU_IMPORT("[method]compute-pass.end")
extern void wgpu_wit_pass_end(int32_t pass_h);

/* queue.write-buffer(destination borrow, dst-offset u64, data list<u8>)
 *   -> result<_, error>. */
WGPU_IMPORT("[method]queue.write-buffer")
extern void wgpu_wit_queue_write_buffer(int32_t queue_h, int32_t dst_h, uint64_t dst_off,
                                        const uint8_t *data_ptr, uint32_t data_len,
                                        uint8_t *ret);

/* queue.submit(commands: list<command-buffer own>) -> result<_, error>.
 * list<own<T>> lowered as (int32_t *handles_ptr, u32 handles_len).
 * `submit` transfers ownership; the runtime-guest handle-table entries
 * become dangling on the wasm side after this call. */
WGPU_IMPORT("[method]queue.submit")
extern void wgpu_wit_queue_submit(int32_t queue_h, const int32_t *cmds_ptr, uint32_t cmds_len,
                                  uint8_t *ret);

/* ---------------------------------------------------------------------
 * WGPU_Device_st / _Memory_st / _Function_st concrete structs.
 * ------------------------------------------------------------------- */

/* Small per-device rolling registration for the single WIT adapter
 * handle we retain — every device is reachable from `request-adapter`
 * once. Non-thread-safe (fork is single-threaded under wasi-p1). */
struct WGPU_Device_st {
    int32_t adapter_h;
    int32_t device_h;
    int32_t queue_h;
};

/* A GPU-side buffer + its cached DtoH staging companion. The staging
 * buffer is created lazily on first `WGPU_MemoryCopyDtoH` and grown
 * on demand; per M14.2's flag it survives across DtoH calls to
 * amortise the create/copy/submit/map/unmap/destroy dance. */
struct WGPU_Memory_st {
    struct WGPU_Device_st *device;
    int32_t buffer_h;
    size_t size;              /* logical bytes allocated */
    int32_t staging_h;        /* 0 = none cached */
    size_t staging_capacity;  /* bytes; grow-only. */
};

/* A compiled compute pipeline + its bind-group scaffolding. On each
 * `WGPU_FunctionRun` we rebuild the bind-group (bindings change per
 * dispatch) but keep the layout/module/pipeline cached.
 *
 * TVM 0.25's WebGPU codegen emits every kernel with a fixed shape:
 *
 *     @group(0) @binding(0..N-1) var<storage, read[_write]>  handles
 *     @group(0) @binding(N)      var<uniform>                PODArgs
 *
 * `num_handle_args` = N; the uniform is always at binding N. The
 * per-handle read/write kind comes from the module's
 * `paramWriteAccess:[...]` launch-param tag (parsed in
 * webgpu_module.c and threaded here at `WGPU_FunctionCreate` time).
 *
 * The PODArgs uniform carries the kernel's scalar (non-tensor)
 * arguments in order — plus a trailing `packGridDimX` (u32) the
 * codegen uses for its own >65536-workgroup grid-packing guard.
 * `pod_arg_dtypes`/`num_pod_args` describe those scalars; `pod_bytes`
 * is `(num_pod_args + 1) * 4` bytes (i32/u32/f32 lane per pod arg,
 * plus the trailing packGridDimX slot). `pod_buffer_h` is a
 * uniform-usage GPU buffer we allocate once at Create-time and
 * write-through with `queue.write-buffer` each dispatch. */
struct WGPU_Function_st {
    struct WGPU_Device_st *device;
    int32_t shader_module_h;
    int32_t pipeline_h;
    int32_t bind_group_layout_h;
    uint32_t num_handle_args;
    uint32_t num_pod_args;
    DLDataType *pod_arg_dtypes;
    int32_t pod_buffer_h;
    uint32_t pod_bytes;
    /* Per-handle write-access hints, retained from FunctionCreate.
     * Used at WGPU_FunctionRun time to detect TVM 0.25's aliased-in-place
     * dispatches (e.g. `2d_continuous_cumsum`) where a single WGPU_Memory
     * is bound to both a read-only and a read-write slot — Dawn rejects
     * those as "Buffer usage (Storage(read-write)|Storage(read-only))
     * includes writable usage and another usage in the same
     * synchronization scope." Shadow-buffer treatment lives in
     * WGPU_FunctionRun; keeping this here avoids threading it back
     * through the runtime-side WGPU_FunctionRun signature. NULL means
     * all-writable (matches the FunctionCreate contract). */
    uint8_t *handle_write_access;
};

/* ---------------------------------------------------------------------
 * WGPU_* implementations.
 * ------------------------------------------------------------------- */

int WGPU_DeviceGet(WGPU_Device *device_ptr) {
    struct WGPU_Device_st *dev = calloc(1, sizeof(struct WGPU_Device_st));
    if (!dev) {
        TVMAPISetLastError("WGPU_DeviceGet: out of memory");
        return -1;
    }

    /* request-adapter returns result<option<adapter>, webgpu-error>.
     * Ret-area layout: [0]=outer-discr, [4]=inner (option<adapter>
     * for ok, variant tag+payload for err). option<adapter> lowers to
     * u8 is-some + i32 handle. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_request_adapter(wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        free(dev);
        return -1;
    }
    if (wgpu_wit_ret_area[4] == 0) {
        TVMAPISetLastError("WGPU_DeviceGet: no WebGPU adapter available");
        free(dev);
        return -1;
    }
    dev->adapter_h = *(const int32_t *)(wgpu_wit_ret_area + 8);

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_adapter_request_device(dev->adapter_h, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_adapter_drop(dev->adapter_h);
        free(dev);
        return -1;
    }
    dev->device_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    dev->queue_h = wgpu_wit_device_queue(dev->device_h);
    *device_ptr = (WGPU_Device)dev;
    return 0;
}

int WGPU_DeviceFree(WGPU_Device device) {
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;
    if (!dev) {
        return 0;
    }
    if (dev->queue_h) {
        wgpu_wit_queue_drop(dev->queue_h);
    }
    if (dev->device_h) {
        wgpu_wit_device_drop(dev->device_h);
    }
    if (dev->adapter_h) {
        wgpu_wit_adapter_drop(dev->adapter_h);
    }
    free(dev);
    return 0;
}

/* buffer-usage bitset — matches the WIT `flags` declaration order
 * (map-read=1, map-write=2, copy-src=4, copy-dst=8, storage=16,
 * uniform=32). */
enum {
    WGPU_USAGE_MAP_READ = 1u << 0,
    WGPU_USAGE_MAP_WRITE = 1u << 1,
    WGPU_USAGE_COPY_SRC = 1u << 2,
    WGPU_USAGE_COPY_DST = 1u << 3,
    WGPU_USAGE_STORAGE = 1u << 4,
    WGPU_USAGE_UNIFORM = 1u << 5,
};

int WGPU_MemoryAlloc(WGPU_Device device, WGPU_Memory *memory_ptr, size_t nbytes) {
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;
    /* WebGPU requires 4-byte-multiple size for STORAGE|COPY_* buffers. */
    if (nbytes & 3u) {
        nbytes = (nbytes | 3u) + 1u;
    }

    struct WGPU_Memory_st *mem = calloc(1, sizeof(struct WGPU_Memory_st));
    if (!mem) {
        TVMAPISetLastError("WGPU_MemoryAlloc: out of memory");
        return -1;
    }
    mem->device = dev;
    mem->size = nbytes;

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_buffer(dev->device_h, (uint64_t)nbytes,
                                  WGPU_USAGE_STORAGE | WGPU_USAGE_COPY_SRC | WGPU_USAGE_COPY_DST,
                                  0 /* mapped-at-creation */, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        free(mem);
        return -1;
    }
    mem->buffer_h = *(const int32_t *)(wgpu_wit_ret_area + 4);
    *memory_ptr = (WGPU_Memory)mem;
    return 0;
}

int WGPU_MemoryFree(WGPU_Memory memory) {
    struct WGPU_Memory_st *mem = (struct WGPU_Memory_st *)memory;
    if (!mem) {
        return 0;
    }
    if (mem->staging_h) {
        wgpu_wit_buffer_destroy(mem->staging_h);
        wgpu_wit_buffer_drop(mem->staging_h);
    }
    if (mem->buffer_h) {
        wgpu_wit_buffer_destroy(mem->buffer_h);
        wgpu_wit_buffer_drop(mem->buffer_h);
    }
    free(mem);
    return 0;
}

int WGPU_MemoryCopyHtoD(WGPU_Memory dst, size_t dst_byte_offset, const void *src,
                        size_t src_byte_offset, size_t nbytes) {
    struct WGPU_Memory_st *mem = (struct WGPU_Memory_st *)dst;
    const uint8_t *data_ptr = (const uint8_t *)src + src_byte_offset;

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_queue_write_buffer(mem->device->queue_h, mem->buffer_h, (uint64_t)dst_byte_offset,
                                data_ptr, (uint32_t)nbytes, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }
    return 0;
}

/* ---- staging-buffer companion (per-source-buffer cache) ---- */

/* Ensure `mem->staging_h` exists and has capacity >= need_bytes.
 * Returns 0 on success. Caller aligns need_bytes to 4 first. */
static int wgpu_ensure_staging(struct WGPU_Memory_st *mem, size_t need_bytes) {
    if (mem->staging_h != 0 && mem->staging_capacity >= need_bytes) {
        return 0;
    }
    /* Grow: destroy any prior smaller staging companion; allocate fresh
     * with MAP_READ | COPY_DST usage. Grow-in-powers-of-2 to amortise
     * repeated growth on progressively larger reads. */
    if (mem->staging_h != 0) {
        wgpu_wit_buffer_destroy(mem->staging_h);
        wgpu_wit_buffer_drop(mem->staging_h);
        mem->staging_h = 0;
        mem->staging_capacity = 0;
    }
    size_t cap = 4;
    while (cap < need_bytes) {
        cap <<= 1;
    }
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_buffer(mem->device->device_h, (uint64_t)cap,
                                  WGPU_USAGE_MAP_READ | WGPU_USAGE_COPY_DST,
                                  0 /* mapped-at-creation */, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }
    mem->staging_h = *(const int32_t *)(wgpu_wit_ret_area + 4);
    mem->staging_capacity = cap;
    return 0;
}

int WGPU_MemoryCopyDtoH(void *dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset,
                        size_t nbytes) {
    struct WGPU_Memory_st *mem = (struct WGPU_Memory_st *)src;
    /* Rounded-up copy size (WebGPU mapAsync requires 4-multiple). */
    size_t map_size = (nbytes & 3u) ? ((nbytes | 3u) + 1u) : nbytes;
    if (wgpu_ensure_staging(mem, map_size) != 0) {
        return -1;
    }

    /* One encoder + copy + submit round trip. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_command_encoder(mem->device->device_h, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }
    int32_t encoder_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_encoder_copy_buffer_to_buffer(encoder_h, mem->buffer_h, (uint64_t)src_byte_offset,
                                           mem->staging_h, 0, (uint64_t)nbytes,
                                           wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_encoder_drop(encoder_h);
        return -1;
    }

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_encoder_finish(encoder_h, wgpu_wit_ret_area);
    /* encoder_h is spent after finish() regardless of success; drop it. */
    wgpu_wit_encoder_drop(encoder_h);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }
    int32_t cmd_buf_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Submit the copy — single JSPI round-trip amortising the batch. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_queue_submit(mem->device->queue_h, &cmd_buf_h, 1u, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }

    /* Map, read, unmap. mapAsync suspends via JSPI. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_buffer_map_async(mem->staging_h, 1u /* MAP_MODE_READ */, 0, (uint64_t)map_size,
                              wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }

    /* get-mapped-range returns list<u8> on the ok side. Ret-area layout:
     * [0]=outer-discr, [4]=ptr (u32), [8]=len (u32). The host allocates
     * the payload via cabi_realloc — we memcpy out then leak the ptr
     * for now (deferred cabi_free hookup). */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_buffer_get_mapped_range(mem->staging_h, 0, (uint64_t)nbytes, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_buffer_unmap(mem->staging_h);
        return -1;
    }
    uint32_t data_ptr = *(const uint32_t *)(wgpu_wit_ret_area + 4);
    uint32_t data_len = *(const uint32_t *)(wgpu_wit_ret_area + 8);
    if (data_len < nbytes) {
        TVMAPISetLastError("WGPU_MemoryCopyDtoH: host returned short buffer");
        wgpu_wit_buffer_unmap(mem->staging_h);
        return -1;
    }
    memcpy((uint8_t *)dst + dst_byte_offset, (const void *)(uintptr_t)data_ptr, nbytes);

    wgpu_wit_buffer_unmap(mem->staging_h);
    return 0;
}

int WGPU_MemoryCopyDtoD(WGPU_Memory dst, size_t dst_byte_offset, WGPU_Memory src,
                        size_t src_byte_offset, size_t nbytes) {
    struct WGPU_Memory_st *src_mem = (struct WGPU_Memory_st *)src;
    struct WGPU_Memory_st *dst_mem = (struct WGPU_Memory_st *)dst;

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_command_encoder(dst_mem->device->device_h, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }
    int32_t encoder_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_encoder_copy_buffer_to_buffer(encoder_h, src_mem->buffer_h,
                                           (uint64_t)src_byte_offset, dst_mem->buffer_h,
                                           (uint64_t)dst_byte_offset, (uint64_t)nbytes,
                                           wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_encoder_drop(encoder_h);
        return -1;
    }

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_encoder_finish(encoder_h, wgpu_wit_ret_area);
    wgpu_wit_encoder_drop(encoder_h);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }
    int32_t cmd_buf_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_queue_submit(dst_mem->device->queue_h, &cmd_buf_h, 1u, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }
    return 0;
}

/* bind-group-layout-entry lowered as a fixed-size 16-byte record:
 * offset 0: u32 binding
 * offset 4: u8  kind    (0=storage, 1=read-only-storage, 2=uniform)
 * offset 5: u8  has-dynamic-offset
 * offset 8: u64 min-binding-size
 * padding matches wit-bindgen-c record layout. */
struct wgpu_bgl_entry_wire {
    uint32_t binding;
    uint8_t kind;
    uint8_t has_dynamic_offset;
    uint8_t _pad0[2];
    uint64_t min_binding_size;
};

/* bind-group-entry lowered as a fixed-size 32-byte record (validated
 * against wit-bindgen-c 0.60.0 output — see scratchpad probe run for
 * offsets):
 *   offset 0:  u32 binding
 *   offset 4:  i32 buffer (borrow<buffer> = i32 handle, 4-byte align)
 *   offset 8:  u64 offset (8-byte align)
 *   offset 16: u8  size.is_some  (option<u64>::is_some — inline record)
 *   offset 24: u64 size.val      (aligned to 8; 7 bytes tail padding
 *                                  after is_some)
 *   total: 32 bytes.
 *
 * Prior version placed a 4-byte pad before buffer_h and mispositioned
 * every field beyond it — 40-byte struct with buffer at offset 8 —
 * which crashed the host provider's descriptor unpacking with all-zero
 * offsets/sizes past the first entry. */
struct wgpu_bg_entry_wire {
    uint32_t binding;
    int32_t buffer_h;
    uint64_t offset;
    uint8_t size_is_some;
    uint8_t _pad0[7];
    uint64_t size;
};

int WGPU_FunctionCreate(WGPU_Device device, WGPU_Function *func_ptr, const char *source,
                        uint32_t source_len, const char *entry_name, uint32_t entry_name_len,
                        uint32_t num_handle_args, const uint8_t *handle_write_access,
                        uint32_t num_pod_args, const DLDataType *pod_arg_dtypes) {
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;
    struct WGPU_Function_st *fn = calloc(1, sizeof(struct WGPU_Function_st));
    if (!fn) {
        TVMAPISetLastError("WGPU_FunctionCreate: out of memory");
        return -1;
    }
    fn->device = dev;
    fn->num_handle_args = num_handle_args;
    fn->num_pod_args = num_pod_args;
    /* Retain paramWriteAccess for alias-detection at dispatch time. */
    if (num_handle_args > 0 && handle_write_access) {
        fn->handle_write_access = calloc(num_handle_args, sizeof(uint8_t));
        if (!fn->handle_write_access) {
            free(fn);
            TVMAPISetLastError("WGPU_FunctionCreate: out of memory (write-access)");
            return -1;
        }
        memcpy(fn->handle_write_access, handle_write_access,
               sizeof(uint8_t) * num_handle_args);
    }
    if (num_pod_args > 0 && pod_arg_dtypes) {
        fn->pod_arg_dtypes = calloc(num_pod_args, sizeof(DLDataType));
        if (!fn->pod_arg_dtypes) {
            free(fn->handle_write_access);
            free(fn);
            TVMAPISetLastError("WGPU_FunctionCreate: out of memory (pod dtypes)");
            return -1;
        }
        memcpy(fn->pod_arg_dtypes, pod_arg_dtypes, sizeof(DLDataType) * num_pod_args);
    }

    /* PODArgs uniform buffer size = (num_pod_args + 1) * 4 bytes.
     * The trailing 4 bytes hold `packGridDimX` (see JS reference in
     * tvm/web/src/webgpu.ts::createShadeInternal). Even when the
     * kernel has zero POD scalars, the WGSL still declares a
     * uniform PODArgs binding — always allocate 4 bytes. WebGPU also
     * requires a >=16-byte binding size; round up. */
    fn->pod_bytes = (num_pod_args + 1u) * 4u;
    if (fn->pod_bytes < 16u) {
        fn->pod_bytes = 16u;
    }

    /* Default entry-point matches yanghaku's hard-coded "main" when
     * the caller passes NULL/0. TVM's WGSL codegen emits per-function
     * entry names via the `@compute` block in the shader; passing the
     * WIT method a non-empty name lets the compute-pipeline descriptor
     * dispatch to it. */
    const char *entry_ptr = entry_name;
    uint32_t entry_len = entry_name_len;
    if (entry_ptr == NULL || entry_len == 0) {
        entry_ptr = "main";
        entry_len = 4;
    }

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_shader_module(dev->device_h, (const uint8_t *)source, source_len,
                                         wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        return -1;
    }
    fn->shader_module_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Build a bind-group layout: N handle bindings (kind per
     * paramWriteAccess) + 1 trailing uniform binding for PODArgs.
     * Matches TVM 0.25's WGSL codegen convention: bindings 0..N-1 are
     * storage buffers (writable or read-only per the hint), binding N
     * is a uniform buffer carrying the PODArgs struct.
     *
     * The previous single-kind (all storage-rw) layout tripped Dawn's
     * pipeline validation for kernels TVM 0.25 emits — the shader
     * declares `var<storage, read>` for read-only bindings, so a
     * "storage" layout entry mismatches; and the shader's `var<uniform>`
     * binding wasn't in the layout at all, so pipeline creation
     * silently failed and the dispatch wrote nothing. */
    uint32_t bgl_count = num_handle_args + 1u;
    struct wgpu_bgl_entry_wire *bgl_entries =
        calloc(bgl_count, sizeof(struct wgpu_bgl_entry_wire));
    if (!bgl_entries) {
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn);
        TVMAPISetLastError("WGPU_FunctionCreate: out of memory (bgl entries)");
        return -1;
    }
    for (uint32_t i = 0; i < num_handle_args; ++i) {
        bgl_entries[i].binding = i;
        /* binding-kind variant: 0 = storage-buffer (rw), 1 =
         * read-only-storage-buffer. paramWriteAccess=1 → writable,
         * 0 → read-only. */
        uint8_t wa = handle_write_access ? handle_write_access[i] : 1u;
        bgl_entries[i].kind = wa ? 0u : 1u;
        bgl_entries[i].has_dynamic_offset = 0;
        bgl_entries[i].min_binding_size = 0;
    }
    /* Trailing uniform binding for PODArgs. */
    bgl_entries[num_handle_args].binding = num_handle_args;
    bgl_entries[num_handle_args].kind = 2u; /* uniform-buffer */
    bgl_entries[num_handle_args].has_dynamic_offset = 0;
    bgl_entries[num_handle_args].min_binding_size = 0;

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_bind_group_layout(dev->device_h, (const uint8_t *)bgl_entries, bgl_count,
                                             wgpu_wit_ret_area);
    free(bgl_entries);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        return -1;
    }
    fn->bind_group_layout_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Create the compute pipeline: module + entry + [bind-group-layout]. */
    int32_t bgl_list[1] = {fn->bind_group_layout_h};
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_compute_pipeline(dev->device_h, fn->shader_module_h,
                                            (const uint8_t *)entry_ptr, entry_len, bgl_list, 1u,
                                            wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_bind_group_layout_drop(fn->bind_group_layout_h);
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        return -1;
    }
    fn->pipeline_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Allocate the PODArgs uniform buffer up-front and reuse across
     * dispatches — same amortisation trade as the DtoH staging buffer.
     * queue.writeBuffer() rewrites contents each dispatch. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_buffer(dev->device_h, (uint64_t)fn->pod_bytes,
                                  WGPU_USAGE_UNIFORM | WGPU_USAGE_COPY_DST,
                                  0 /* mapped-at-creation */, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_compute_pipeline_drop(fn->pipeline_h);
        wgpu_wit_bind_group_layout_drop(fn->bind_group_layout_h);
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        return -1;
    }
    fn->pod_buffer_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    *func_ptr = (WGPU_Function)fn;
    return 0;
}

/*
 * WebGPU spec §21.3 pins `maxComputeWorkgroupsPerDimension` at 65535 for
 * the default limit tier. TVM's WGSL codegen anticipates this: every
 * emitted kernel guards on `blockIdx.z * gridDim.x + blockIdx.x >
 * podArgs.packGridDimX` and reads its logical X index as that same
 * expression (see the WGSL blob in dec_devc.o). That lets the host split
 * a >65535 X-dimension launch across Z: keep the logical extent in the
 * uniform-buffer slot and launch as (min(x, MAX), y, z * ceil(x / MAX)).
 * Only the X axis is repacked — TVM's WGSL codegen assumes original Z=1
 * when it emits the pack-guard, which holds for the ops it lowers to
 * this shape.
 *
 * Without this split, decoder-scale dispatches (262144 workgroups per
 * dim for the final upsample) trip WebGPU's validation and drop the
 * command buffer — exactly the failure mode flagged in the M14 plan's
 * contingency #3.
 */
#define WGPU_MAX_WORKGROUPS_PER_DIM 65535u

int WGPU_FunctionRun(WGPU_Function function, const WGPU_Memory *handle_args,
                     uint32_t num_handle_args, const uint64_t *pod_arg_values,
                     uint32_t num_pod_args, size_t grid_dim_x, size_t grid_dim_y,
                     size_t grid_dim_z) {
    struct WGPU_Function_st *fn = (struct WGPU_Function_st *)function;

    if (num_handle_args != fn->num_handle_args || num_pod_args != fn->num_pod_args) {
        TVMAPISetLastError("WGPU_FunctionRun: arg-count mismatch vs FunctionCreate");
        return -1;
    }

    /* ---- packGridDimX split ---- */
    uint32_t pack_dim_x = (uint32_t)grid_dim_x;
    uint32_t launch_x = pack_dim_x;
    uint32_t launch_y = (uint32_t)grid_dim_y;
    uint32_t launch_z = (uint32_t)grid_dim_z;
    if (launch_x > WGPU_MAX_WORKGROUPS_PER_DIM) {
        uint32_t chunks = (launch_x + WGPU_MAX_WORKGROUPS_PER_DIM - 1u) /
                          WGPU_MAX_WORKGROUPS_PER_DIM;
        launch_x = WGPU_MAX_WORKGROUPS_PER_DIM;
        launch_z = launch_z * chunks;
    }

    /* ---- Pack + upload PODArgs uniform. ---- */
    /* Layout: (num_pod_args i32/u32/f32 slots) || packGridDimX (u32).
     * Interpretation of each POD slot is per-dtype so int / uint / float
     * arg values reach the shader with the right bit pattern. Value bytes
     * come from the wrapper's TVMFFIAny.v_int64 slot; for float args the
     * codegen packs the float bit-pattern into the low 4 bytes of v_int64
     * (matches TVM's PackedArg contract). */
    if (fn->pod_bytes > 0) {
        uint8_t pod_bytes[fn->pod_bytes];
        memset(pod_bytes, 0, fn->pod_bytes);
        for (uint32_t i = 0; i < num_pod_args; ++i) {
            uint32_t slot = 0;
            DLDataType dt = fn->pod_arg_dtypes ? fn->pod_arg_dtypes[i]
                                               : (DLDataType){0, 32, 1};
            uint64_t raw = pod_arg_values ? pod_arg_values[i] : 0u;
            if (dt.code == kDLFloat) {
                /* v_float64 stored via v_int64 slot; the compiled stub
                 * packs the float bit pattern into the low 4 bytes. */
                float f = (float)*(const double *)&raw;
                memcpy(&slot, &f, sizeof(slot));
            } else {
                /* int / uint: low 32 bits. */
                slot = (uint32_t)raw;
            }
            memcpy(pod_bytes + i * 4u, &slot, 4u);
        }
        memcpy(pod_bytes + num_pod_args * 4u, &pack_dim_x, 4u);

        memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
        wgpu_wit_queue_write_buffer(fn->device->queue_h, fn->pod_buffer_h, 0u, pod_bytes,
                                    fn->pod_bytes, wgpu_wit_ret_area);
        if (wgpu_wit_ret_area[0] != 0) {
            wgpu_wit_forward_error(wgpu_wit_ret_area);
            return -1;
        }
    }

    /* ---- Alias detection + shadow-buffer treatment ----
     *
     * TVM 0.25's memory planner may bind the same WGPU_Memory to both a
     * read-only and a read-write kernel slot (in-place ops — VITS's
     * enc_gpu_2d_continuous_cumsum_kernel is the canonical case). Dawn
     * rejects such dispatches:
     *
     *   "Buffer usage (Storage(read-write)|Storage(read-only)) includes
     *    writable usage and another usage in the same synchronization
     *    scope."
     *
     * For each read-only slot i whose source aliases a read-write slot j
     * (handle_args[i] == handle_args[j], write_access[i]=0,
     * write_access[j]=1), allocate a fresh storage buffer ("shadow") of
     * the same size, then before begin-compute-pass issue a
     * copy-buffer-to-buffer(source → shadow) on the same encoder that
     * will dispatch the kernel. This snapshots the pre-dispatch contents;
     * the read-only binding then sees the pre-scan input while the
     * read-write binding writes the post-scan output — exactly the
     * intended in-place semantics.
     *
     * Shadows are allocated per dispatch and destroyed after
     * queue.submit. VITS runs cumsum a bounded number of times per
     * synthesis, so per-dispatch alloc/free is acceptable; a pool can
     * be layered on later if profiling calls for it. */
    int32_t *bind_buffer_h = NULL;
    int32_t *shadow_h = NULL;
    uint64_t *shadow_size = NULL;
    if (num_handle_args > 0) {
        bind_buffer_h = calloc(num_handle_args, sizeof(int32_t));
        shadow_h = calloc(num_handle_args, sizeof(int32_t));
        shadow_size = calloc(num_handle_args, sizeof(uint64_t));
        if (!bind_buffer_h || !shadow_h || !shadow_size) {
            free(bind_buffer_h);
            free(shadow_h);
            free(shadow_size);
            TVMAPISetLastError("WGPU_FunctionRun: out of memory (alias scratch)");
            return -1;
        }
    }
    for (uint32_t i = 0; i < num_handle_args; ++i) {
        struct WGPU_Memory_st *m = (struct WGPU_Memory_st *)handle_args[i];
        bind_buffer_h[i] = m->buffer_h;
    }
    if (fn->handle_write_access) {
        for (uint32_t i = 0; i < num_handle_args; ++i) {
            if (fn->handle_write_access[i] != 0u) {
                continue; /* only shadow read-only slots. */
            }
            for (uint32_t j = 0; j < num_handle_args; ++j) {
                if (j == i) {
                    continue;
                }
                if (handle_args[i] != handle_args[j]) {
                    continue;
                }
                if (fn->handle_write_access[j] != 1u) {
                    continue; /* two read-only aliases are fine. */
                }
                /* Alias with mismatched access → shadow slot i. */
                struct WGPU_Memory_st *m = (struct WGPU_Memory_st *)handle_args[i];
                memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
                wgpu_wit_device_create_buffer(fn->device->device_h, (uint64_t)m->size,
                                              WGPU_USAGE_STORAGE | WGPU_USAGE_COPY_DST,
                                              0 /* mapped-at-creation */, wgpu_wit_ret_area);
                if (wgpu_wit_ret_area[0] != 0) {
                    wgpu_wit_forward_error(wgpu_wit_ret_area);
                    for (uint32_t k = 0; k < i; ++k) {
                        if (shadow_h[k]) {
                            wgpu_wit_buffer_destroy(shadow_h[k]);
                            wgpu_wit_buffer_drop(shadow_h[k]);
                        }
                    }
                    free(bind_buffer_h);
                    free(shadow_h);
                    free(shadow_size);
                    return -1;
                }
                shadow_h[i] = *(const int32_t *)(wgpu_wit_ret_area + 4);
                shadow_size[i] = (uint64_t)m->size;
                bind_buffer_h[i] = shadow_h[i];
                break;
            }
        }
    }

/* Shadow-cleanup helper: destroy every allocated shadow buffer and free
 * the per-dispatch scratch arrays. Reached from every early-return path
 * below and from the success path. */
#define WGPU_RUN_FREE_SHADOWS()                                                                    \
    do {                                                                                           \
        for (uint32_t _si = 0; _si < num_handle_args; ++_si) {                                     \
            if (shadow_h && shadow_h[_si]) {                                                       \
                wgpu_wit_buffer_destroy(shadow_h[_si]);                                            \
                wgpu_wit_buffer_drop(shadow_h[_si]);                                               \
            }                                                                                      \
        }                                                                                          \
        free(bind_buffer_h);                                                                       \
        free(shadow_h);                                                                            \
        free(shadow_size);                                                                         \
    } while (0)

    /* ---- Build bind-group entries: N storage + 1 uniform. ---- */
    uint32_t bg_count = num_handle_args + 1u;
    struct wgpu_bg_entry_wire *entries =
        calloc(bg_count, sizeof(struct wgpu_bg_entry_wire));
    if (!entries) {
        TVMAPISetLastError("WGPU_FunctionRun: out of memory (bg entries)");
        WGPU_RUN_FREE_SHADOWS();
        return -1;
    }
    for (uint32_t i = 0; i < num_handle_args; ++i) {
        entries[i].binding = i;
        /* Route to the shadow when one was allocated for this slot; the
         * source buffer_h stays available for the pre-dispatch snapshot
         * copy below (encoder_copy_buffer_to_buffer reads from the raw
         * WGPU_Memory struct, not from bind_buffer_h). */
        entries[i].buffer_h = bind_buffer_h[i];
        entries[i].offset = 0;
        entries[i].size_is_some = 0;
        entries[i].size = 0;
    }
    /* Uniform binding at index num_handle_args, size = fn->pod_bytes. */
    entries[num_handle_args].binding = num_handle_args;
    entries[num_handle_args].buffer_h = fn->pod_buffer_h;
    entries[num_handle_args].offset = 0;
    entries[num_handle_args].size_is_some = 1u;
    entries[num_handle_args].size = fn->pod_bytes;

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_bind_group(fn->device->device_h, fn->bind_group_layout_h,
                                      (const uint8_t *)entries, bg_count, wgpu_wit_ret_area);
    free(entries);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        WGPU_RUN_FREE_SHADOWS();
        return -1;
    }
    int32_t bind_group_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Encoder → shadow-snapshot copies → compute pass → set pipeline +
     * bind-group → dispatch → end → finish → submit. All commands
     * batch inside one queue.submit, i.e. one JSPI round-trip for the
     * whole dispatch. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_command_encoder(fn->device->device_h, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_bind_group_drop(bind_group_h);
        WGPU_RUN_FREE_SHADOWS();
        return -1;
    }
    int32_t encoder_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Snapshot each aliased source into its shadow BEFORE begin-compute-pass
     * so the read-only binding sees the pre-dispatch bytes. The compute
     * pass reads the shadow (via bind_buffer_h[i]) and writes the source
     * (via its own read-write binding) — same synchronization scope as
     * intended by the in-place op, but no two-mode alias on any single
     * buffer. */
    for (uint32_t i = 0; i < num_handle_args; ++i) {
        if (!shadow_h || shadow_h[i] == 0) {
            continue;
        }
        struct WGPU_Memory_st *m = (struct WGPU_Memory_st *)handle_args[i];
        memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
        wgpu_wit_encoder_copy_buffer_to_buffer(encoder_h, m->buffer_h, 0, shadow_h[i], 0,
                                               shadow_size[i], wgpu_wit_ret_area);
        if (wgpu_wit_ret_area[0] != 0) {
            wgpu_wit_forward_error(wgpu_wit_ret_area);
            wgpu_wit_encoder_drop(encoder_h);
            wgpu_wit_bind_group_drop(bind_group_h);
            WGPU_RUN_FREE_SHADOWS();
            return -1;
        }
    }

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_encoder_begin_compute_pass(encoder_h, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_encoder_drop(encoder_h);
        wgpu_wit_bind_group_drop(bind_group_h);
        WGPU_RUN_FREE_SHADOWS();
        return -1;
    }
    int32_t pass_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    wgpu_wit_pass_set_pipeline(pass_h, fn->pipeline_h);
    wgpu_wit_pass_set_bind_group(pass_h, 0u, bind_group_h);
    wgpu_wit_pass_dispatch_workgroups(pass_h, launch_x, launch_y, launch_z);
    wgpu_wit_pass_end(pass_h);
    wgpu_wit_compute_pass_drop(pass_h);

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_encoder_finish(encoder_h, wgpu_wit_ret_area);
    wgpu_wit_encoder_drop(encoder_h);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_bind_group_drop(bind_group_h);
        WGPU_RUN_FREE_SHADOWS();
        return -1;
    }
    int32_t cmd_buf_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_queue_submit(fn->device->queue_h, &cmd_buf_h, 1u, wgpu_wit_ret_area);
    /* bind_group ownership is retained across the submit — the compute
     * pass borrowed it. Drop after the queue accepts the batch. */
    wgpu_wit_bind_group_drop(bind_group_h);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        WGPU_RUN_FREE_SHADOWS();
        return -1;
    }
    /* Success — destroy shadows (WebGPU allows buffer.destroy() while
     * prior in-flight submissions complete; the submitted commands
     * hold internal refs on the buffer until the queue drains). */
    WGPU_RUN_FREE_SHADOWS();
#undef WGPU_RUN_FREE_SHADOWS
    return 0;
}

int WGPU_FunctionFree(WGPU_Function function) {
    struct WGPU_Function_st *fn = (struct WGPU_Function_st *)function;
    if (!fn) {
        return 0;
    }
    if (fn->pod_buffer_h) {
        wgpu_wit_buffer_destroy(fn->pod_buffer_h);
        wgpu_wit_buffer_drop(fn->pod_buffer_h);
    }
    if (fn->pipeline_h) {
        wgpu_wit_compute_pipeline_drop(fn->pipeline_h);
    }
    if (fn->bind_group_layout_h) {
        wgpu_wit_bind_group_layout_drop(fn->bind_group_layout_h);
    }
    if (fn->shader_module_h) {
        wgpu_wit_shader_module_drop(fn->shader_module_h);
    }
    if (fn->pod_arg_dtypes) {
        free(fn->pod_arg_dtypes);
    }
    if (fn->handle_write_access) {
        free(fn->handle_write_access);
    }
    free(fn);
    return 0;
}
