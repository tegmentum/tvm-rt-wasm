/**
 * @file webgpu/c_api/webgpu_js_impl.c
 * @brief Implementation for WGPU_* sync C API on top of the `browser:webgpu`
 * WIT interface (see wit-packages/browser-webgpu/wit/webgpu.wit in cognition,
 * v0.8.0). M15.3 rebridge — replaces the M14.3 bindings against the retiring
 * `host:webgpu@0.1.0`. Each of the WGPU_* entry points still routes through
 * canonical-ABI wit-bindgen-shaped extern imports; the extern set is
 * regenerated against browser:webgpu's `gpu` interface (compute slice —
 * render/canvas/texture/sampler methods live in the imported world but are
 * never invoked from the cognition path).
 *
 * Structural deltas from the M14.3 shape (see docs/milestone-15-api-mapping.md
 * §1, §6 for the full list):
 *   - Package/interface: `host:webgpu/webgpu@0.1.0` → `browser:webgpu/gpu@0.8.0`.
 *   - `resource queue` is dissolved: `queue.write-buffer` becomes
 *     `device.queue-write-buffer`, `queue.submit` becomes `device.submit`.
 *     WGPU_Device_st no longer caches a queue handle.
 *   - `buffer.get-mapped-range` returns a `mapped-range` resource; readback
 *     is now a 2-step call sequence (`get-mapped-range` → `mapped-range.read`
 *     → drop). Net +2 WIT calls per DtoH copy.
 *   - `pipeline-layout` is now a first-class resource: WGPU_FunctionCreate
 *     allocates one before create-compute-pipeline via a new
 *     `device.create-pipeline-layout` call.
 *   - `bind-group-layout-entry` restructured: adds required `visibility:
 *     shader-stage` field; the flat `binding-kind` variant becomes a nested
 *     `binding-type::buffer(buffer-binding-layout{type, has-dynamic-offset,
 *     min-binding-size: option<u64>})` shape. Entry size grows 16 → 40 bytes.
 *   - `bind-group-entry` restructured: the inlined (buffer, offset, size)
 *     tuple becomes `binding-resource::buffer(buffer-binding{...})`. Entry
 *     size grows 32 → 48 bytes.
 *   - `compute-pass` resource renamed `compute-pass-encoder` (linker-name
 *     `[resource-drop]compute-pass-encoder` and `[method]compute-pass-encoder.*`).
 *   - `compute-pass-encoder.set-bind-group` grows a required `dynamic-offsets:
 *     list<u32>` parameter (we always pass empty).
 *   - Several methods shed their `result<_, error>` wrappers:
 *     `device.create-command-encoder`, `command-encoder.copy-buffer-to-buffer`,
 *     `command-encoder.finish`, `command-encoder.begin-compute-pass`,
 *     `device.queue-write-buffer`, `device.submit`. Ret-area check blocks
 *     after these calls disappear; validation errors surface at submit time.
 *   - `request-adapter` grows a required `adapter-options` parameter (we pass
 *     `{power-preference: some(high-performance), force-fallback-adapter: false}`)
 *     AND drops its `result` wrapper on the option — non-availability is now
 *     silent `option::none`.
 *   - `adapter.request-device` grows a required `device-descriptor` parameter
 *     (we pass `{required-features: [], required-limits: none}` — the JS
 *     satisfier still echoes adapter.limits back per M14 behavioral parity).
 *   - `buffer-usage` flag bit positions shift: browser:webgpu inserts
 *     `index, vertex` between `copy-dst` and `uniform, storage`, so
 *     `storage` moves from bit 4 to bit 7 and `uniform` from bit 5 to bit 6.
 *   - `webgpu-error` → `gpu-error`: arm-count identical, arm names mostly
 *     line up (`not-supported → other`, `internal-error → internal`). The
 *     ret-area layout is the same and the fork only reads the message
 *     string, so error routing needs no arm-specific change.
 *
 * Compile-time contract: the file must produce a linkable object under
 * `-Wall -Wextra -Werror` (wasi-sdk clang) with the WIT imports unresolved
 * — they are satisfied at final composition time when cognition's
 * `webgpu-host-impl.js` (M15.4 rewrite) supplies the JS side.
 *
 * Runtime contract: each WGPU_* function
 *
 *   1. Marshals its C args into canonical-ABI shapes (i32 handles for
 *      resources; ptr+len pairs for byte slices; nested variant/record
 *      layouts for the descriptor structs).
 *   2. Invokes the corresponding `browser:webgpu/gpu@0.8.0` import via
 *      `__attribute__((import_module, import_name))`-declared externs.
 *   3. On methods that still carry a `result<_, gpu-error>` wrapper,
 *      unpacks the wit-bindgen ret-area buffer and forwards error strings
 *      into `TVMAPISetLastError`. On methods that shed the wrapper,
 *      unpacks the raw return directly (or returns void).
 *
 * Handle encoding, staging-buffer cache, and shadow-buffer alias treatment
 * are unchanged from the M14 shape.
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
 * For our WIT (`browser:webgpu/gpu@0.8.0`) the module string is
 * "browser:webgpu/gpu@0.8.0"; canonical method names are the WIT
 * abstract names in kebab-case, with resource methods prefixed by
 * "[method]<resource>.", constructors by "[constructor]<resource>",
 * and resource-drop by "[resource-drop]<resource>". WIT `%`-escaped
 * keywords (e.g. `%end`) are unescaped in the ABI string (`end`).
 * ------------------------------------------------------------------- */
#define WGPU_WIT_MOD "browser:webgpu/gpu@0.8.0"

#define WGPU_IMPORT(name)                                                                          \
    __attribute__((__import_module__(WGPU_WIT_MOD), __import_name__(name)))

/*
 * Canonical ABI conventions used below:
 *
 *  - Resource handles: `int32_t`. Guest-owned handles (`own<T>` in WIT)
 *    are returned by value; guest-borrowed handles (`borrow<T>`) are
 *    passed by value on call. `[resource-drop]T` releases an `own<T>`.
 *  - result<T, gpu-error>: caller passes an out-buffer whose layout
 *    starts with a u8 discriminant (0 = ok, 1 = err). On ok, T is
 *    packed at the appropriate aligned offset. On err, the variant
 *    discriminant (u8) + a string pointer (u32) + string length (u32)
 *    follow. Our helper `wgpu_wit_ret_area` is 32 bytes — enough for
 *    every method the interface exposes (each returns at most a
 *    single resource handle or single u64 + the error payload).
 *  - list<u8> arg: passed as (ptr, len) unpacked into the arg vector.
 *  - list<u8> return: caller passes an out-(ptr, len) buffer pair; the
 *    guest is responsible for freeing via wit-bindgen's cabi_realloc
 *    (not implemented here — the mapped-range readback leaks the
 *    per-call buffer for now, mirroring the M14.3 shape).
 *  - Strings on the wire follow the same (ptr, len) convention.
 *  - Records/variants that flatten to more than MAX_FLAT_PARAMS (16)
 *    scalars are passed as a pointer to a caller-allocated struct.
 *    `device-descriptor` is the only such case in the compute path.
 *  - Flags: packed into the smallest u8/u16/u32/u64 that holds the
 *    bit set. `shader-stage` (3 flags) → u8; `buffer-usage` (10
 *    flags) → u16 packed into u32 param at the ABI boundary.
 *
 * These conventions match the shape wit-bindgen-c generates and jco
 * consumes.
 */

/* Fixed-size ret area for canonical ABI result<T, gpu-error>
 * unpacking. 32 bytes covers the largest possible T (i64 or a pair of
 * i32 handles) plus discriminant + error-variant payload (u8 + ptr +
 * len = 12 bytes). */
#define WGPU_WIT_RET_AREA_SIZE 32
_Alignas(8) static uint8_t wgpu_wit_ret_area[WGPU_WIT_RET_AREA_SIZE];

/* Fan out the ret-area on error. On the wire, a gpu-error variant
 * starts at offset 4 (after the outer result discriminant + padding to
 * 4-byte alignment) with a u8 case discriminant, then a string pointer
 * (u32) and length (u32) at offset 8/12. We stash the message into
 * TVMAPISetLastError and drop the string ptr (leaks; wit-bindgen would
 * normally have a `cabi_free`-hook for this — deferred). */
static void wgpu_wit_forward_error(const uint8_t *ret_area) {
    /* discriminant at [0] is 1 (err). Case tag at [4]. Message ptr at
     * [8], len at [12]. Order matches wit-bindgen-c lowering of a
     * single-string-payload variant (gpu-error's five arms all carry
     * a single `string` payload, so the layout is arm-independent). */
    const uint32_t msg_ptr = *(const uint32_t *)(ret_area + 8);
    const uint32_t msg_len = *(const uint32_t *)(ret_area + 12);
    if (msg_ptr == 0 || msg_len == 0) {
        TVMAPISetLastError("browser:webgpu: (no message)");
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
 *   - option<u64>         → uint8_t is_some, uint64_t val   (in structs)
 *                         → i32 is_some, i64 val            (flattened)
 *   - result<T, err>      → uint8_t *ret_area (out-buffer written by host)
 *   - flags (buffer-usage → u16-fits-in-u32) → uint32_t bitset
 *   - flags (shader-stage → u8)              → uint8_t bitset  (in structs)
 * ------------------------------------------------------------------- */

/* Free-function: request-adapter: func(options: adapter-options) -> option<adapter>
 *
 * adapter-options flattens to 3 scalars (≤ MAX_FLAT_PARAMS): option<power-preference>
 * = (i32 is_some, i32 case), plus i32 force-fallback-adapter (bool). The
 * return type `option<adapter>` (no result wrapper) lowers to a 2-word ret
 * area: [0]=is_some (u8 padded to 4), [4]=handle (i32). */
WGPU_IMPORT("request-adapter")
extern void wgpu_wit_request_adapter(int32_t power_pref_is_some, int32_t power_pref_val,
                                     int32_t force_fallback, uint8_t *ret);

/* adapter.request-device: func(desc: device-descriptor) -> result<device, gpu-error>
 *
 * device-descriptor flattens to 16 scalars (list<string> = 2 for ptr+len,
 * plus option<limits> = 1 discriminant + 13 flat limits fields = 14, total
 * 16). Combined with the resource `self` handle (1 flat scalar), the
 * method's params total 17 flat — exceeds MAX_FLAT_PARAMS (16), so per
 * the canonical ABI *all* params (including self) collapse into a single
 * caller-supplied pointer. The result `result<own<device>, gpu-error>`
 * also exceeds MAX_FLAT_RESULTS (1), so a ret_ptr is added.
 *
 * Wire signature is therefore (params_ptr, ret_ptr) — 2 pointer params.
 * Confirmed against wit-bindgen 0.60.0 output for the same WIT
 * (see scratchpad/wit-bindgen-test/out/importer.c). Caller layout for
 * the params buffer (88 bytes = 80 + 2*sizeof(void*), align 8) is
 * documented at WGPU_DeviceGet below. */
WGPU_IMPORT("[method]adapter.request-device")
extern void wgpu_wit_adapter_request_device(const uint8_t *params_ptr, uint8_t *ret);

/* [resource-drop]adapter */
WGPU_IMPORT("[resource-drop]adapter")
extern void wgpu_wit_adapter_drop(int32_t h);

/* [resource-drop]device */
WGPU_IMPORT("[resource-drop]device")
extern void wgpu_wit_device_drop(int32_t h);

/* device.create-buffer(desc: buffer-descriptor) -> result<buffer, gpu-error>
 * buffer-descriptor flattens to 3 scalars (u64 size = 1, u32 usage-flags = 1,
 * u8 mapped-at-creation = 1). Same lowering shape as under host:webgpu@0.1.0
 * — only the buffer-usage bit positions changed (see WGPU_USAGE_* below). */
WGPU_IMPORT("[method]device.create-buffer")
extern void wgpu_wit_device_create_buffer(int32_t device_h, uint64_t size, uint32_t usage,
                                          int32_t mapped_at_creation, uint8_t *ret);

/* device.create-shader-module(wgsl: string) -> result<shader-module, gpu-error> */
WGPU_IMPORT("[method]device.create-shader-module")
extern void wgpu_wit_device_create_shader_module(int32_t device_h, const uint8_t *src_ptr,
                                                 uint32_t src_len, uint8_t *ret);

/* device.create-compute-pipeline(desc: compute-pipeline-descriptor)
 *   -> result<compute-pipeline, gpu-error>
 *
 * Descriptor lowered to 5 flat scalars (≤ MAX_FLAT_PARAMS):
 *   compute-state.module (i32), compute-state.entry-point (i32 ptr, i32 len),
 *   pipeline-layout-option discriminant (i32; 0=auto, 1=explicit),
 *   pipeline-layout-option payload (i32; borrow<pipeline-layout> handle,
 *     ignored when discriminant=0).
 *
 * Every fork call uses `explicit(...)` with a pre-created pipeline-layout
 * (see WGPU_FunctionCreate for the sequencing). */
WGPU_IMPORT("[method]device.create-compute-pipeline")
extern void wgpu_wit_device_create_compute_pipeline(int32_t device_h, int32_t module_h,
                                                    const uint8_t *entry_ptr, uint32_t entry_len,
                                                    int32_t layout_variant,
                                                    int32_t pipeline_layout_h, uint8_t *ret);

/* device.create-compute-pipelines-async(descs: list<compute-pipeline-descriptor>)
 *   -> list<result<compute-pipeline, gpu-error>>
 *
 * v0.8 batched-async pipeline creation. Collapses N per-kernel
 * create-compute-pipeline crossings into ONE. Host drives
 * `createComputePipelineAsync` + `Promise.all(...)` under the covers so
 * WGSL compile runs in parallel. Every element of the returned list
 * carries its own `result<>` — partial-batch success is representable.
 *
 * ABI shape:
 *   * `descs` collapses to (ptr, len) — a packed array of
 *     `struct wgpu_compute_pipeline_descriptor_wire` (see below).
 *   * Return is `list<result<compute-pipeline, gpu-error>>` — the
 *     canonical ABI writes (ptr, len) into `ret[0..8]` where ptr points
 *     at a host-allocated buffer of `len` `struct
 *     wgpu_compute_pipeline_result_wire` elements (see below).
 *   * The `borrow<>` semantics inside each descriptor scope every
 *     shader-module / pipeline-layout handle for the duration of the
 *     one call — same as calling create-compute-pipeline N times
 *     back-to-back, just batched.
 *
 * See WGPU_FunctionCreateBatch below for the caller that packages the
 * per-kernel setup into a single batched pipeline dispatch. Motivating
 * workload: TVM's decoder VMCreate compiles ~75 kernels — see
 * `cognition/docs/guest-wasm-bottleneck-investigation.md`. */
WGPU_IMPORT("[method]device.create-compute-pipelines-async")
extern void wgpu_wit_device_create_compute_pipelines_async(int32_t device_h,
                                                           const uint8_t *descs_ptr,
                                                           uint32_t descs_len, uint8_t *ret);

/* device.create-bind-group-layout(desc: bind-group-layout-descriptor)
 *   -> result<bind-group-layout, gpu-error>
 *
 * bind-group-layout-descriptor flattens to 2 scalars (entries: list<...> =
 * (ptr, len)) — same as under host:webgpu. Only the entry element shape
 * changed: see struct wgpu_bgl_entry_wire below. */
WGPU_IMPORT("[method]device.create-bind-group-layout")
extern void wgpu_wit_device_create_bind_group_layout(int32_t device_h,
                                                     const uint8_t *entries_ptr,
                                                     uint32_t entries_len, uint8_t *ret);

/* device.create-bind-group(desc: bind-group-descriptor)
 *   -> result<bind-group, gpu-error>
 *
 * bind-group-descriptor flattens to 3 scalars (layout borrow (i32),
 * entries list (ptr, len)) — same as under host:webgpu. Entry element
 * shape changed: see struct wgpu_bg_entry_wire below. */
WGPU_IMPORT("[method]device.create-bind-group")
extern void wgpu_wit_device_create_bind_group(int32_t device_h, int32_t layout_h,
                                              const uint8_t *entries_ptr, uint32_t entries_len,
                                              uint8_t *ret);

/* device.create-pipeline-layout(layouts: list<borrow<bind-group-layout>>)
 *   -> result<pipeline-layout, gpu-error>
 *
 * New in browser:webgpu (v0.3). The fork calls this once per WGPU_Function
 * during Create to wrap the bind-group-layout in a first-class pipeline
 * layout resource, which then goes into `create-compute-pipeline`. */
WGPU_IMPORT("[method]device.create-pipeline-layout")
extern void wgpu_wit_device_create_pipeline_layout(int32_t device_h, const int32_t *bgl_handles,
                                                   uint32_t bgl_len, uint8_t *ret);

/* [resource-drop]pipeline-layout (new resource in browser:webgpu) */
WGPU_IMPORT("[resource-drop]pipeline-layout")
extern void wgpu_wit_pipeline_layout_drop(int32_t h);

/* device.create-command-encoder() -> command-encoder
 *
 * No result wrapper in browser:webgpu — this call cannot fail at the
 * spec level. Return the handle directly. */
WGPU_IMPORT("[method]device.create-command-encoder")
extern int32_t wgpu_wit_device_create_command_encoder(int32_t device_h);

/* buffer.map-async(mode: map-mode, offset: u64, size: u64)
 *   -> result<_, gpu-error> */
WGPU_IMPORT("[method]buffer.map-async")
extern void wgpu_wit_buffer_map_async(int32_t buffer_h, uint32_t mode, uint64_t offset,
                                      uint64_t size, uint8_t *ret);

/* buffer.get-mapped-range(offset: u64, size: u64)
 *   -> result<mapped-range, gpu-error>
 *
 * Now returns a `mapped-range` resource handle rather than bytes-inline —
 * caller follows with `mapped-range.read(...)` to pull the actual bytes,
 * then drops the mapped-range resource, then unmaps the buffer. See
 * WGPU_MemoryCopyDtoH for the new 5-call readback sequence.
 *
 * Ret area layout: [0]=outer discr, [4]=mapped-range handle (ok) or
 * inner variant discr + msg ptr/len (err). */
WGPU_IMPORT("[method]buffer.get-mapped-range")
extern void wgpu_wit_buffer_get_mapped_range(int32_t buffer_h, uint64_t offset, uint64_t size,
                                             uint8_t *ret);

/* mapped-range.read(offset: u64, size: u64) -> list<u8>
 *
 * Returns raw bytes (no result wrapper). Ret area layout: [0]=data ptr (u32),
 * [4]=data len (u32). Guest-owned memory (host allocated via cabi_realloc);
 * fork memcpy's out and currently leaks the returned buffer — same
 * deferred-cabi_free debt as the M14 `get-mapped-range` bytes path. */
WGPU_IMPORT("[method]mapped-range.read")
extern void wgpu_wit_mapped_range_read(int32_t mr_h, uint64_t offset, uint64_t size,
                                       uint8_t *ret);

/* [resource-drop]mapped-range */
WGPU_IMPORT("[resource-drop]mapped-range")
extern void wgpu_wit_mapped_range_drop(int32_t h);

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

/* command-encoder.begin-compute-pass() -> compute-pass-encoder
 *
 * No result wrapper in browser:webgpu. Returns the handle directly.
 * Note the resource-type rename: `compute-pass` (host:webgpu) →
 * `compute-pass-encoder` (browser:webgpu). Method import strings and
 * the corresponding [resource-drop] slot pick up the new name. */
WGPU_IMPORT("[method]command-encoder.begin-compute-pass")
extern int32_t wgpu_wit_encoder_begin_compute_pass(int32_t encoder_h);

/* command-encoder.copy-buffer-to-buffer(src, src-off, dst, dst-off, size)
 *
 * No result wrapper in browser:webgpu — fire-and-forget. Validation errors
 * from bad offsets surface later at `device.submit` time. */
WGPU_IMPORT("[method]command-encoder.copy-buffer-to-buffer")
extern void wgpu_wit_encoder_copy_buffer_to_buffer(int32_t encoder_h, int32_t src_h,
                                                   uint64_t src_off, int32_t dst_h,
                                                   uint64_t dst_off, uint64_t nbytes);

/* command-encoder.finish() -> command-buffer
 *
 * No result wrapper. Returns the handle directly. */
WGPU_IMPORT("[method]command-encoder.finish")
extern int32_t wgpu_wit_encoder_finish(int32_t encoder_h);

/* [resource-drop]command-encoder / command-buffer / compute-pass-encoder
 *
 * `compute-pass-encoder` replaces host:webgpu's `compute-pass` at the drop
 * slot — same wire semantics, different linker string. `queue` drop slot
 * disappears entirely (resource dissolved into `device`). */
WGPU_IMPORT("[resource-drop]command-encoder")
extern void wgpu_wit_encoder_drop(int32_t h);
WGPU_IMPORT("[resource-drop]command-buffer")
extern void wgpu_wit_command_buffer_drop(int32_t h);
WGPU_IMPORT("[resource-drop]compute-pass-encoder")
extern void wgpu_wit_compute_pass_encoder_drop(int32_t h);

/* compute-pass-encoder.set-pipeline / set-bind-group / dispatch-workgroups / end
 *
 * `set-bind-group` grows a required `dynamic-offsets: list<u32>` parameter
 * per browser:webgpu v0.3; the fork passes an empty list at every call
 * (TVM's dispatch path never binds dynamic-offset entries). */
WGPU_IMPORT("[method]compute-pass-encoder.set-pipeline")
extern void wgpu_wit_pass_set_pipeline(int32_t pass_h, int32_t pipeline_h);
WGPU_IMPORT("[method]compute-pass-encoder.set-bind-group")
extern void wgpu_wit_pass_set_bind_group(int32_t pass_h, uint32_t index, int32_t group_h,
                                         const uint32_t *dyn_off_ptr, uint32_t dyn_off_len);
WGPU_IMPORT("[method]compute-pass-encoder.dispatch-workgroups")
extern void wgpu_wit_pass_dispatch_workgroups(int32_t pass_h, uint32_t x, uint32_t y, uint32_t z);
/* WIT declares this as `%end` (keyword-escape); the canonical ABI import
 * string uses the un-escaped `end`. */
WGPU_IMPORT("[method]compute-pass-encoder.end")
extern void wgpu_wit_pass_end(int32_t pass_h);

/* device.queue-write-buffer(buf: borrow<buffer>, buffer-offset: u64, data: list<u8>)
 *
 * Migrated from `queue.write-buffer` (queue resource dissolved). No result
 * wrapper on browser side — fire-and-forget; validation errors surface at
 * next submit. */
WGPU_IMPORT("[method]device.queue-write-buffer")
extern void wgpu_wit_device_queue_write_buffer(int32_t device_h, int32_t dst_h, uint64_t dst_off,
                                               const uint8_t *data_ptr, uint32_t data_len);

/* device.submit(buffers: list<command-buffer>)
 *
 * Migrated from `queue.submit`. No result wrapper. `list<own<command-buffer>>`
 * lowered as (int32_t *handles_ptr, u32 handles_len); submit consumes each
 * command-buffer handle. */
WGPU_IMPORT("[method]device.submit")
extern void wgpu_wit_device_submit(int32_t device_h, const int32_t *cmds_ptr, uint32_t cmds_len);

/* ---------------------------------------------------------------------
 * WGPU_Device_st / _Memory_st / _Function_st concrete structs.
 * ------------------------------------------------------------------- */

/* BATCH-TVM kernel-dispatch batch state (see WGPU_BeginKernelBatch in
 * webgpu_c_api.h and docs/tvm-boundary-overhead-investigation.md §4.A).
 *
 * A batch shares one command encoder and one compute pass across N
 * kernel invocations, deferring pass.end() / encoder.finish() /
 * device.submit() / bind-group.drop() / shadow-buffer.destroy() until
 * WGPU_FlushKernelBatch runs. Encoder/pass are opened lazily on the
 * first WGPU_FunctionRun after Begin, and re-opened after each flush
 * inside the same active batch.
 *
 * Retained arrays are grow-only across the whole batch lifetime; they
 * survive Begin/End cycles (only freed at WGPU_DeviceFree) to amortise
 * the alloc cost across VITS's ~200-dispatch-per-inference workload. */
struct WGPU_Batch_st {
    /* 1 between Begin and End; 0 outside — WGPU_FunctionRun keys off
     * this to pick the batched vs unbatched dispatch path. */
    int active;
    /* 1 when encoder_h is a live [own]command-encoder handle. */
    int has_encoder;
    /* 1 when pass_h is a live [own]compute-pass-encoder handle
     * (implies has_encoder — passes are always created against an
     * open encoder). */
    int has_pass;
    int32_t encoder_h;
    int32_t pass_h;
    /* Bind-groups the batched pass has attached via set-bind-group.
     * The pass borrows each until end() runs; safest to drop them all
     * after submit accepts the command buffer, mirroring the unbatched
     * path's post-submit drop. */
    int32_t *pending_bind_groups;
    size_t pending_bg_count;
    size_t pending_bg_capacity;
    /* Shadow buffers allocated by aliased-in-place dispatches inside
     * the batch (see the WGPU_RUN_FREE_SHADOWS block below). Destroyed
     * + dropped after submit — WebGPU accepts buffer.destroy() while
     * in-flight submits still hold internal refs. */
    int32_t *pending_shadows;
    size_t pending_shadow_count;
    size_t pending_shadow_capacity;
};

/* Small per-device rolling registration. The `queue_h` field the M14 shape
 * carried is gone — browser:webgpu dissolved the queue resource into
 * `device`. Every write-buffer / submit call now targets `device_h` directly. */
struct WGPU_Device_st {
    int32_t adapter_h;
    int32_t device_h;
    /* Zero-initialised via calloc in WGPU_DeviceGet — batch starts
     * inactive with no encoder/pass and empty retained arrays. */
    struct WGPU_Batch_st batch;
};

/* A GPU-side buffer + its cached DtoH staging companion. The staging
 * buffer is created lazily on first `WGPU_MemoryCopyDtoH` and grown
 * on demand; per M14.2's flag it survives across DtoH calls to
 * amortise the create/copy/submit/map/unmap/destroy dance.
 *
 * When @ref parent_buffer_h is non-zero this record is an ALIAS: it
 * borrows the parent's underlying buffer, with @ref parent_offset
 * pointing at the alias's origin inside the parent. Aliases are
 * produced by @ref WGPU_ConstantsBulkUpload — the Relax VM's constants
 * pool packs N tensors into one giant parent buffer to collapse
 * VMCreate's per-tensor `create-buffer` + `queue-write-buffer`
 * crossings (see cognition's tvm-vmcreate-parse-investigation.md).
 *
 * Alias contract:
 *   - `buffer_h` mirrors the parent's handle for existing consumers
 *     (bind-group entries, queue-write, copy-buffer-to-buffer). Callers
 *     that need the alias's slice add `parent_offset` to the buffer
 *     offset they hand the wire call.
 *   - `size` is the logical size the caller reserved (not the parent's
 *     total). Used to set an explicit bind-group entry `size` so a
 *     shader can't read outside its slice.
 *   - `staging_h` / `staging_capacity` are per-alias — if a caller
 *     ever DtoH's from an alias (constants never do), the staging
 *     buffer lives on the alias record, not the parent.
 *   - @ref WGPU_MemoryFree on an alias frees the record only; the
 *     parent buffer must be freed exactly once by its owner. */
struct WGPU_Memory_st {
    struct WGPU_Device_st *device;
    int32_t buffer_h;
    size_t size;              /* logical bytes allocated */
    int32_t staging_h;        /* 0 = none cached */
    size_t staging_capacity;  /* bytes; grow-only. */
    /* Non-zero when this record aliases a slice of another buffer.
     * Set to `buffer_h` at alias creation (informational marker). */
    int32_t parent_buffer_h;
    /* Byte offset into the parent buffer where this alias starts.
     * Added to caller-supplied dst/src offsets in the copy paths and
     * used as the binding `offset` in bind-group entries. Zero for
     * standalone buffers. */
    uint64_t parent_offset;
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
 * write-through with `device.queue-write-buffer` each dispatch.
 *
 * `pipeline_layout_h`: new resource in browser:webgpu — a first-class
 * pipeline-layout handle wrapping `bind_group_layout_h`, threaded into
 * `create-compute-pipeline`'s descriptor. Owned; dropped at Free. */
struct WGPU_Function_st {
    struct WGPU_Device_st *device;
    int32_t shader_module_h;
    int32_t pipeline_h;
    int32_t bind_group_layout_h;
    int32_t pipeline_layout_h;
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
 * Params-buffer layout for adapter.request-device.
 *
 * The method's params exceed MAX_FLAT_PARAMS (see the WGPU_IMPORT above),
 * so the canonical ABI collapses all params — including the resource
 * `self` handle — into a single caller-supplied pointer. Layout
 * (canonical ABI on wasm32, align 8, size 88 = 80 + 2*sizeof(void*),
 * matches wit-bindgen 0.60.0's `80+2*sizeof(void*)` ret_area
 * accounting):
 *
 *   offset  0-3:   self.__handle                   (i32, adapter handle)
 *   offset  4-7:   padding to align 8
 *   offset  8-11:  required_features.ptr           (u32, list<string> ptr)
 *   offset 12-15:  required_features.len           (u32, list<string> len)
 *   offset 16:     required_limits.is_some         (u8)
 *   offset 17-23:  padding to align 8 for the limits body
 *   offset 24-27:  max_texture_dimension_d1        (i32)
 *   offset 28-31:  max_texture_dimension_d2        (i32)
 *   offset 32-35:  max_texture_dimension_d3        (i32)
 *   offset 36-39:  max_texture_array_layers        (i32)
 *   offset 40-43:  max_bind_groups                 (i32)
 *   offset 44-47:  padding to align 8 for the i64 pair
 *   offset 48-55:  max_uniform_buffer_binding_size (i64)
 *   offset 56-63:  max_storage_buffer_binding_size (i64)
 *   offset 64-67:  max_vertex_buffers              (i32)
 *   offset 68-71:  max_vertex_attributes           (i32)
 *   offset 72-75:  max_compute_workgroup_size_x    (i32)
 *   offset 76-79:  max_compute_workgroup_size_y    (i32)
 *   offset 80-83:  max_compute_workgroup_size_z    (i32)
 *   offset 84-87:  max_compute_workgroups_per_dim  (i32)
 *
 * The fork always passes `{required-features: [], required-limits: none}`
 * — bytes 4-87 stay fully zero; only self at bytes 0-3 varies per call.
 * The JS satisfier (M15.4) restores the M14 max-out-limits echo when
 * required-limits decodes to `none`.
 * ------------------------------------------------------------------- */
#define WGPU_REQUEST_DEVICE_PARAMS_SIZE 88

/* ---------------------------------------------------------------------
 * WGPU_* implementations.
 * ------------------------------------------------------------------- */

/* Registration function owned by the Relax VM runner
 * (src/backends/relax_vm/relax_vm_runner.c). Called at
 * WGPU_DeviceGet-time to hook WGPU_BeginKernelBatch / WGPU_EndKernelBatch
 * as the per-run open/close pair. Requires tvm-rt-backend-relax-vm to
 * be linked; every in-tree WebGPU test target already links both libs. */
extern void TVM_RT_WASM_RegisterKernelBatchHooks(int (*begin)(void *),
                                                 int (*end)(void *));

/* Registration function owned by the Relax VM (relax_vm.c). Called at
 * WGPU_DeviceGet-time to hook the bulk-upload primitive into
 * RelaxVirtualMachineCreateImpl's constants loop, so the VM can
 * collapse its N per-tensor create-buffer + queue-write-buffer
 * crossings into 2. See docs/tvm-vmcreate-parse-investigation.md.
 *
 * When the WebGPU accelerator is not linked (CPU-only tests), the
 * hooks stay NULL and the VM falls back to the per-constant path. */
extern void TVM_RT_WASM_RegisterConstantsBulkUploadHooks(
    int (*upload)(void *device_stream, uint32_t n,
                  const void *const *srcs, const size_t *sizes,
                  void **aliases_out, void **parent_out),
    int (*free_parent)(void *parent));

/* Trampoline that adapts WGPU_BeginKernelBatch's WGPU_Device signature
 * to the runner-visible void*. WGPU_Device is `struct WGPU_Device_st *`
 * — same width as void*, so the cast is safe. */
static int wgpu_batch_begin_trampoline(void *device_stream) {
    return WGPU_BeginKernelBatch((WGPU_Device)device_stream);
}
static int wgpu_batch_end_trampoline(void *device_stream) {
    return WGPU_EndKernelBatch((WGPU_Device)device_stream);
}

/* Trampolines for the constants bulk-upload hook. The VM passes
 * `void *device_stream` opaquely; it's a WGPU_Device the runner
 * fetched from the same `webgpu_api->GetStream()` slot the batch
 * hooks use. WGPU_Memory and void* are pointer-width equivalent so
 * the cast round-trips cleanly. */
static int wgpu_constants_bulk_upload_trampoline(void *device_stream, uint32_t n,
                                                 const void *const *srcs,
                                                 const size_t *sizes,
                                                 void **aliases_out,
                                                 void **parent_out) {
    return WGPU_ConstantsBulkUpload((WGPU_Device)device_stream, n, srcs, sizes,
                                    (WGPU_Memory *)aliases_out,
                                    (WGPU_Memory *)parent_out);
}
static int wgpu_constants_bulk_free_parent_trampoline(void *parent) {
    return WGPU_MemoryFree((WGPU_Memory)parent);
}

int WGPU_DeviceGet(WGPU_Device *device_ptr) {
    struct WGPU_Device_st *dev = calloc(1, sizeof(struct WGPU_Device_st));
    if (!dev) {
        TVMAPISetLastError("WGPU_DeviceGet: out of memory");
        return -1;
    }

    /* Install the Relax VM's batching hooks so
     * TVM_RT_WASM_RelaxVMRunFunction opens/closes a batch around each
     * VM run. Idempotent; safe on repeated device creation. */
    TVM_RT_WASM_RegisterKernelBatchHooks(wgpu_batch_begin_trampoline,
                                         wgpu_batch_end_trampoline);

    /* Install the Relax VM's constants bulk-upload hooks so
     * RelaxVirtualMachineCreateImpl can collapse its per-tensor
     * CPU→GPU upload crossings into one create-buffer +
     * queue-write-buffer pair. Same idempotence semantics as the
     * batch hooks. */
    TVM_RT_WASM_RegisterConstantsBulkUploadHooks(
        wgpu_constants_bulk_upload_trampoline,
        wgpu_constants_bulk_free_parent_trampoline);

    /* request-adapter returns plain option<adapter> (browser:webgpu dropped
     * the outer result wrapper — non-availability is silent none, JS-side
     * errors become traps at the component boundary). Ret-area layout:
     * [0]=is_some (u8 padded to 4), [4]=adapter handle (i32). Pass
     * `{power-preference: some(high-performance), force-fallback-adapter:
     * false}` for behavioral parity with M14 (which was fixed at
     * high-performance internally). */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_request_adapter(/* power_pref_is_some */ 1,
                             /* power_pref_val (high-performance) */ 1,
                             /* force_fallback */ 0, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] == 0) {
        TVMAPISetLastError("WGPU_DeviceGet: no WebGPU adapter available");
        free(dev);
        return -1;
    }
    dev->adapter_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* request-device grew a required device-descriptor parameter. The
     * descriptor + self exceed MAX_FLAT_PARAMS, so the canonical ABI
     * collapses both into a single indirect params buffer (self at
     * offset 0, zero-shaped descriptor at offsets 8-87 — layout
     * documented at WGPU_REQUEST_DEVICE_PARAMS_SIZE above). Pass an
     * empty-features / none-limits descriptor — the JS satisfier
     * max-outs limits from adapter.limits on the none-branch to
     * preserve M14 behavioral parity. */
    _Alignas(8) uint8_t rd_params[WGPU_REQUEST_DEVICE_PARAMS_SIZE];
    memset(rd_params, 0, sizeof(rd_params));
    *(int32_t *)(rd_params + 0) = dev->adapter_h;
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_adapter_request_device(rd_params, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_adapter_drop(dev->adapter_h);
        free(dev);
        return -1;
    }
    dev->device_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    *device_ptr = (WGPU_Device)dev;
    return 0;
}

int WGPU_DeviceFree(WGPU_Device device) {
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;
    if (!dev) {
        return 0;
    }
    /* Flush + tear down any active batch before releasing the device.
     * WGPU_EndKernelBatch first flushes (which drops retained bind-groups
     * and destroys retained shadow buffers) then clears `active`. Then
     * we free the retained arrays themselves. */
    if (dev->batch.active) {
        WGPU_EndKernelBatch(device);
    }
    free(dev->batch.pending_bind_groups);
    free(dev->batch.pending_shadows);
    if (dev->device_h) {
        wgpu_wit_device_drop(dev->device_h);
    }
    if (dev->adapter_h) {
        wgpu_wit_adapter_drop(dev->adapter_h);
    }
    free(dev);
    return 0;
}

/* buffer-usage bitset — matches the browser:webgpu@0.8.0 `flags` declaration
 * order (map-read=1, map-write=2, copy-src=4, copy-dst=8, index=16, vertex=32,
 * uniform=64, storage=128, indirect=256, query-resolve=512).
 *
 * Delta from host:webgpu@0.1.0's 6-flag set: browser inserts index/vertex
 * between copy-dst and uniform, and adds indirect/query-resolve at the tail.
 * The bit positions of `storage` and `uniform` shift accordingly —
 * cognition uses only the 4 copy/map flags + storage + uniform on the
 * compute path, so the new positions matter only for the two shifted
 * flags. */
enum {
    WGPU_USAGE_MAP_READ = 1u << 0,
    WGPU_USAGE_MAP_WRITE = 1u << 1,
    WGPU_USAGE_COPY_SRC = 1u << 2,
    WGPU_USAGE_COPY_DST = 1u << 3,
    WGPU_USAGE_INDEX = 1u << 4,
    WGPU_USAGE_VERTEX = 1u << 5,
    WGPU_USAGE_UNIFORM = 1u << 6,
    WGPU_USAGE_STORAGE = 1u << 7,
    WGPU_USAGE_INDIRECT = 1u << 8,
    WGPU_USAGE_QUERY_RESOLVE = 1u << 9,
};

/* shader-stage flags (3 members → u8 in canonical ABI). Cognition binds
 * every entry as compute-only. */
enum {
    WGPU_SHADER_STAGE_VERTEX = 1u << 0,
    WGPU_SHADER_STAGE_FRAGMENT = 1u << 1,
    WGPU_SHADER_STAGE_COMPUTE = 1u << 2,
};

/* buffer-binding-type enum arm values (declaration order in browser:webgpu):
 *   0 = uniform, 1 = storage, 2 = read-only-storage.
 *
 * Delta from host:webgpu's binding-kind (which used 0=storage-buffer,
 * 1=read-only-storage-buffer, 2=uniform-buffer): the read-only vs writable
 * storage discriminator flips values, and uniform moves from 2 to 0.
 * FunctionCreate below picks the arm per per-handle write-access. */
enum {
    WGPU_BUFFER_BINDING_TYPE_UNIFORM = 0,
    WGPU_BUFFER_BINDING_TYPE_STORAGE = 1,
    WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE = 2,
};

/* binding-type variant discriminator (declaration order): 0=buffer,
 * 1=sampler, 2=texture, 3=storage-texture. Cognition only emits buffer. */
#define WGPU_BINDING_TYPE_DISCR_BUFFER 0

/* binding-resource variant discriminator (declaration order): 0=buffer,
 * 1=sampler, 2=texture-view. Cognition only emits buffer. */
#define WGPU_BINDING_RESOURCE_DISCR_BUFFER 0

/* pipeline-layout-option variant discriminator: 0=auto, 1=explicit. */
#define WGPU_PIPELINE_LAYOUT_OPTION_AUTO 0
#define WGPU_PIPELINE_LAYOUT_OPTION_EXPLICIT 1

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
    /* Aliases (parent_buffer_h != 0) never own their underlying buffer
     * — only the parent handle destroys it. The parent's owner is
     * responsible for calling WGPU_MemoryFree once on the parent AFTER
     * every alias has been released. See WGPU_ConstantsBulkUpload for
     * the producer side. Staging buffers, if any, are still per-record
     * and get released here. */
    if (mem->staging_h) {
        wgpu_wit_buffer_destroy(mem->staging_h);
        wgpu_wit_buffer_drop(mem->staging_h);
    }
    if (mem->buffer_h && mem->parent_buffer_h == 0) {
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

    /* Flush any pending batched compute so a mid-batch overwrite can't
     * race a still-unsubmitted pass that reads @p mem — the batch's
     * encoder hasn't reached the queue yet, but queue-write-buffer
     * takes effect on the queue timeline immediately. */
    (void)WGPU_FlushKernelBatch((WGPU_Device)mem->device);

    /* device.queue-write-buffer sheds its result wrapper in browser:webgpu —
     * fire-and-forget. Validation errors from bad offsets surface at the
     * next device.submit. When @p mem is an alias the underlying buffer
     * offset is (parent_offset + dst_byte_offset). Parent_offset is 0
     * for standalone buffers so this stays a no-op there. */
    wgpu_wit_device_queue_write_buffer(mem->device->device_h, mem->buffer_h,
                                       mem->parent_offset + (uint64_t)dst_byte_offset,
                                       data_ptr, (uint32_t)nbytes);
    return 0;
}

/* ---------------------------------------------------------------------
 * GUEST-CONSTANTS-BULK-UPLOAD — collapse N per-tensor CPU→GPU uploads
 * into 1 `create-buffer` + 1 `queue-write-buffer` crossing.
 *
 * WebGPU's `minStorageBufferOffsetAlignment` defaults to 256 bytes:
 * every alias's offset into the parent buffer must be 256-mult so it
 * can be bound as a storage buffer (Dawn rejects unaligned bindings).
 * Between-tensor gaps therefore round up to 256; total padding for
 * N=235 tensors ≤ ~60 KB, negligible vs the ~12 MB total.
 *
 * The parent buffer's usage matches the standalone alloc path
 * (STORAGE | COPY_SRC | COPY_DST) so subsequent DtoD/DtoH from any
 * alias slice works transparently. `queue-write-buffer`'s data_len is
 * u32 on the WIT wire — 12 MB fits comfortably; a check would fire
 * only if TVM ever emits a >4 GB constants blob for a single VM.
 * ------------------------------------------------------------------- */
#define WGPU_BULK_UPLOAD_OFFSET_ALIGN ((size_t)256)

/* Round @p n up to a multiple of @p align. @p align must be a
 * power of two. */
static inline size_t wgpu_align_up(size_t n, size_t align) {
    return (n + (align - 1)) & ~(align - 1);
}

int WGPU_ConstantsBulkUpload(WGPU_Device device, uint32_t n,
                             const void *const *srcs, const size_t *sizes,
                             WGPU_Memory *aliases_out, WGPU_Memory *parent_out) {
    if (n == 0) {
        if (parent_out) {
            *parent_out = NULL;
        }
        return 0;
    }
    if (!device || !srcs || !sizes || !aliases_out || !parent_out) {
        TVMAPISetLastError("WGPU_ConstantsBulkUpload: null argument");
        return -1;
    }
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;

    /* Pre-zero the alias slots so failure paths can safely free
     * whatever has been populated so far without touching junk. */
    for (uint32_t i = 0; i < n; ++i) {
        aliases_out[i] = NULL;
    }
    *parent_out = NULL;

    /* Pass 1: compute 256-byte-aligned offsets + total. Alignment
     * padding sits BETWEEN slots — each slot's byte-count itself is
     * kept at its logical size, and the next slot's offset rounds up
     * from (offset + size). */
    size_t *offsets = calloc(n, sizeof(size_t));
    if (!offsets) {
        TVMAPISetLastError("WGPU_ConstantsBulkUpload: OOM (offsets)");
        return -1;
    }
    size_t cursor = 0;
    for (uint32_t i = 0; i < n; ++i) {
        offsets[i] = cursor;
        /* Advance past this slot; the next slot's offset is rounded
         * up to WGPU_BULK_UPLOAD_OFFSET_ALIGN so it stays bindable. */
        cursor = wgpu_align_up(cursor + sizes[i], WGPU_BULK_UPLOAD_OFFSET_ALIGN);
    }
    /* Parent buffer size must be a 4-mult (WGPU_MemoryAlloc rounds too,
     * but do it here for clarity). */
    size_t parent_size = wgpu_align_up(cursor, 4);
    if (parent_size == 0) {
        /* Every input was zero-length. Nothing to allocate; make
         * every alias record a size-0 non-alias so the caller's
         * subsequent WGPU_MemoryFree is a no-op. */
        for (uint32_t i = 0; i < n; ++i) {
            struct WGPU_Memory_st *a = calloc(1, sizeof(struct WGPU_Memory_st));
            if (!a) {
                for (uint32_t j = 0; j < i; ++j) {
                    free(aliases_out[j]);
                    aliases_out[j] = NULL;
                }
                free(offsets);
                TVMAPISetLastError("WGPU_ConstantsBulkUpload: OOM (empty alias)");
                return -1;
            }
            a->device = dev;
            aliases_out[i] = (WGPU_Memory)a;
        }
        free(offsets);
        return 0;
    }

    /* Pass 2: allocate the parent buffer — one WebGPU crossing. Uses
     * WGPU_MemoryAlloc so the parent gets STORAGE|COPY_SRC|COPY_DST
     * usage matching every standalone AllocDataSpace buffer, and
     * inherits the 4-mult size rounding. */
    struct WGPU_Memory_st *parent = NULL;
    if (WGPU_MemoryAlloc(device, (WGPU_Memory *)&parent, parent_size) != 0) {
        free(offsets);
        return -1;
    }

    /* Pass 3: aggregate the caller blobs into one guest-side buffer.
     * Padding bytes (between slots + trailing to 4-mult) are left
     * zero — never read by any binding because size_is_some on each
     * alias binding clips to the alias's logical size. */
    uint8_t *agg = malloc(parent_size);
    if (!agg) {
        WGPU_MemoryFree((WGPU_Memory)parent);
        free(offsets);
        TVMAPISetLastError("WGPU_ConstantsBulkUpload: OOM (aggregate)");
        return -1;
    }
    memset(agg, 0, parent_size);
    for (uint32_t i = 0; i < n; ++i) {
        if (sizes[i] > 0 && srcs[i] != NULL) {
            memcpy(agg + offsets[i], srcs[i], sizes[i]);
        }
    }

    /* Pass 4: one queue-write-buffer for the whole aggregate — the
     * second (and last) WebGPU crossing this bulk primitive makes.
     * Any pending batched compute is flushed first for the same
     * "no mid-batch overwrite" invariant as WGPU_MemoryCopyHtoD. */
    (void)WGPU_FlushKernelBatch(device);
    wgpu_wit_device_queue_write_buffer(dev->device_h, parent->buffer_h,
                                       0 /* dst_off */, agg,
                                       (uint32_t)parent_size);
    free(agg);

    /* Pass 5: populate alias records — no WebGPU crossings. Each
     * alias mirrors the parent's buffer_h with its own offset/size,
     * and marks parent_buffer_h so WGPU_MemoryFree skips the destroy
     * path when the caller releases the alias. */
    for (uint32_t i = 0; i < n; ++i) {
        struct WGPU_Memory_st *a = calloc(1, sizeof(struct WGPU_Memory_st));
        if (!a) {
            for (uint32_t j = 0; j < i; ++j) {
                free(aliases_out[j]);
                aliases_out[j] = NULL;
            }
            WGPU_MemoryFree((WGPU_Memory)parent);
            free(offsets);
            TVMAPISetLastError("WGPU_ConstantsBulkUpload: OOM (alias)");
            return -1;
        }
        a->device = dev;
        a->buffer_h = parent->buffer_h;
        a->size = sizes[i];
        a->parent_buffer_h = parent->buffer_h;
        a->parent_offset = (uint64_t)offsets[i];
        aliases_out[i] = (WGPU_Memory)a;
    }

    *parent_out = (WGPU_Memory)parent;
    free(offsets);
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
    /* Readback needs completed GPU state — any pending batched compute
     * that writes @p mem must submit before this DtoH copy issues.
     * Batched dispatches inside the same encoder have unresolved
     * writes; without this flush the map-async below would read stale
     * data. No-op outside an active batch. */
    (void)WGPU_FlushKernelBatch((WGPU_Device)mem->device);

    /* Rounded-up copy size (WebGPU mapAsync requires 4-multiple). */
    size_t map_size = (nbytes & 3u) ? ((nbytes | 3u) + 1u) : nbytes;
    if (wgpu_ensure_staging(mem, map_size) != 0) {
        return -1;
    }

    /* One encoder + copy + submit round trip. create-command-encoder,
     * copy-buffer-to-buffer, and finish all shed their result wrappers
     * in browser:webgpu — validation errors surface at submit time. */
    int32_t encoder_h = wgpu_wit_device_create_command_encoder(mem->device->device_h);

    /* Alias-aware source offset: for an alias mem, the underlying buffer
     * offset is (parent_offset + src_byte_offset). Zero parent_offset
     * for standalone buffers preserves the pre-alias behavior. */
    wgpu_wit_encoder_copy_buffer_to_buffer(encoder_h, mem->buffer_h,
                                           mem->parent_offset + (uint64_t)src_byte_offset,
                                           mem->staging_h, 0, (uint64_t)nbytes);

    int32_t cmd_buf_h = wgpu_wit_encoder_finish(encoder_h);
    /* encoder_h is spent after finish(); drop it. */
    wgpu_wit_encoder_drop(encoder_h);

    /* Submit the copy — single JSPI round-trip amortising the batch.
     * No result wrapper on browser side. */
    wgpu_wit_device_submit(mem->device->device_h, &cmd_buf_h, 1u);

    /* Map, read via mapped-range, unmap. mapAsync suspends via JSPI. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_buffer_map_async(mem->staging_h, 1u /* MAP_MODE_READ */, 0, (uint64_t)map_size,
                              wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        return -1;
    }

    /* get-mapped-range now returns a `mapped-range` resource handle instead
     * of bytes-inline. Ret-area layout: [0]=outer discr, [4]=mapped-range
     * handle (ok) or inner error variant (err). */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_buffer_get_mapped_range(mem->staging_h, 0, (uint64_t)nbytes, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_buffer_unmap(mem->staging_h);
        return -1;
    }
    int32_t mapped_range_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* mapped-range.read returns list<u8> (no result wrapper). Ret-area
     * layout: [0]=data ptr (u32), [4]=data len (u32). Host allocates
     * the payload via cabi_realloc; we memcpy out then leak the ptr
     * (same deferred-cabi_free hookup as M14). */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_mapped_range_read(mapped_range_h, 0, (uint64_t)nbytes, wgpu_wit_ret_area);
    uint32_t data_ptr = *(const uint32_t *)(wgpu_wit_ret_area + 0);
    uint32_t data_len = *(const uint32_t *)(wgpu_wit_ret_area + 4);
    if (data_len < nbytes) {
        TVMAPISetLastError("WGPU_MemoryCopyDtoH: host returned short buffer");
        wgpu_wit_mapped_range_drop(mapped_range_h);
        wgpu_wit_buffer_unmap(mem->staging_h);
        return -1;
    }
    memcpy((uint8_t *)dst + dst_byte_offset, (const void *)(uintptr_t)data_ptr, nbytes);

    wgpu_wit_mapped_range_drop(mapped_range_h);
    wgpu_wit_buffer_unmap(mem->staging_h);
    return 0;
}

int WGPU_MemoryCopyDtoD(WGPU_Memory dst, size_t dst_byte_offset, WGPU_Memory src,
                        size_t src_byte_offset, size_t nbytes) {
    struct WGPU_Memory_st *src_mem = (struct WGPU_Memory_st *)src;
    struct WGPU_Memory_st *dst_mem = (struct WGPU_Memory_st *)dst;

    /* DtoD copy uses its own encoder + immediate submit — flush any
     * pending batched compute first so the copy sees the post-batch
     * state and doesn't share an encoder with a still-open compute
     * pass (WebGPU forbids mixing copy commands into an open pass). */
    (void)WGPU_FlushKernelBatch((WGPU_Device)dst_mem->device);

    int32_t encoder_h = wgpu_wit_device_create_command_encoder(dst_mem->device->device_h);

    /* Alias-aware offsets: fold each side's parent_offset into the
     * underlying buffer offset. Both parent_offsets are 0 for
     * standalone buffers, preserving the pre-alias behavior. */
    wgpu_wit_encoder_copy_buffer_to_buffer(
        encoder_h, src_mem->buffer_h,
        src_mem->parent_offset + (uint64_t)src_byte_offset, dst_mem->buffer_h,
        dst_mem->parent_offset + (uint64_t)dst_byte_offset, (uint64_t)nbytes);

    int32_t cmd_buf_h = wgpu_wit_encoder_finish(encoder_h);
    wgpu_wit_encoder_drop(encoder_h);

    wgpu_wit_device_submit(dst_mem->device->device_h, &cmd_buf_h, 1u);
    return 0;
}

/* buffer-binding-layout — nested inside binding-type::buffer variant arm.
 * Canonical ABI layout (24 bytes, align 8):
 *   offset 0:    %type (u8)                 — buffer-binding-type enum
 *   offset 1:    has-dynamic-offset (u8)
 *   offset 2-7:  padding
 *   offset 8:    min-binding-size.is_some (u8)
 *   offset 9-15: padding
 *   offset 16:   min-binding-size.val (u64)
 */
struct wgpu_buffer_binding_layout_wire {
    uint8_t type;
    uint8_t has_dynamic_offset;
    uint8_t _pad0[6];
    uint8_t min_binding_size_is_some;
    uint8_t _pad1[7];
    uint64_t min_binding_size;
};

/* bind-group-layout-entry restructured for browser:webgpu (40 bytes, align 8).
 * Canonical ABI layout:
 *   offset 0-3:   binding (u32)
 *   offset 4:     visibility (u8)             — shader-stage flags
 *   offset 5-7:   padding
 *   offset 8:     ty discriminator (u8)       — 0=buffer, 1=sampler, 2=texture, 3=storage-texture
 *   offset 9-15:  padding to align(u64)
 *   offset 16-39: variant payload (buffer-binding-layout when discriminator=0)
 *
 * Delta from host:webgpu (16 bytes: binding + kind + has_dyn + min_binding_size):
 *   - New required `visibility` field (compute-only per cognition).
 *   - Flat `binding-kind` variant wraps into `binding-type::buffer(...)` inline.
 *   - `min-binding-size` becomes `option<u64>` (cognition passes `none`).
 *   - Enum arm values shift: writable storage=1 (was 0), read-only=2 (was 1),
 *     uniform=0 (was 2). See WGPU_BUFFER_BINDING_TYPE_* below.
 */
struct wgpu_bgl_entry_wire {
    uint32_t binding;
    uint8_t visibility;
    uint8_t _pad0[3];
    uint8_t ty_discriminant;
    uint8_t _pad1[7];
    struct wgpu_buffer_binding_layout_wire buffer;
};

/* buffer-binding — nested inside binding-resource::buffer variant arm.
 * Canonical ABI layout (32 bytes, align 8):
 *   offset 0-3:   buffer (i32 borrow handle)
 *   offset 4-7:   padding to align(u64)
 *   offset 8-15:  offset (u64)
 *   offset 16:    size.is_some (u8)
 *   offset 17-23: padding
 *   offset 24-31: size.val (u64)
 */
struct wgpu_buffer_binding_wire {
    int32_t buffer_h;
    uint8_t _pad0[4];
    uint64_t offset;
    uint8_t size_is_some;
    uint8_t _pad1[7];
    uint64_t size;
};

/* bind-group-entry restructured for browser:webgpu (48 bytes, align 8).
 * Canonical ABI layout:
 *   offset 0-3:   binding (u32)
 *   offset 4-7:   padding to align(u64)
 *   offset 8:     resource discriminator (u8) — 0=buffer, 1=sampler, 2=texture-view
 *   offset 9-15:  padding
 *   offset 16-47: variant payload (buffer-binding when discriminator=0)
 *
 * Delta from host:webgpu (32 bytes: binding + buffer_h + offset + size option):
 *   - Buffer/offset/size wrap into `binding-resource::buffer(buffer-binding{...})`.
 *   - Nesting adds 8 bytes of discriminator+padding front-matter.
 */
struct wgpu_bg_entry_wire {
    uint32_t binding;
    uint8_t _pad0[4];
    uint8_t resource_discriminant;
    uint8_t _pad1[7];
    struct wgpu_buffer_binding_wire buffer;
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
     * Under browser:webgpu, each entry carries a required `visibility`
     * (shader-stage flags — cognition binds every entry as compute-only)
     * and a nested `binding-type::buffer(buffer-binding-layout{...})`
     * variant. Enum arm values re-derived per browser:webgpu's
     * declaration order (see WGPU_BUFFER_BINDING_TYPE_* above). */
    uint32_t bgl_count = num_handle_args + 1u;
    struct wgpu_bgl_entry_wire *bgl_entries =
        calloc(bgl_count, sizeof(struct wgpu_bgl_entry_wire));
    if (!bgl_entries) {
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        TVMAPISetLastError("WGPU_FunctionCreate: out of memory (bgl entries)");
        return -1;
    }
    for (uint32_t i = 0; i < num_handle_args; ++i) {
        bgl_entries[i].binding = i;
        bgl_entries[i].visibility = WGPU_SHADER_STAGE_COMPUTE;
        bgl_entries[i].ty_discriminant = WGPU_BINDING_TYPE_DISCR_BUFFER;
        /* paramWriteAccess=1 → writable storage; 0 → read-only storage. */
        uint8_t wa = handle_write_access ? handle_write_access[i] : 1u;
        bgl_entries[i].buffer.type = wa ? WGPU_BUFFER_BINDING_TYPE_STORAGE
                                        : WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE;
        bgl_entries[i].buffer.has_dynamic_offset = 0;
        bgl_entries[i].buffer.min_binding_size_is_some = 0;
        bgl_entries[i].buffer.min_binding_size = 0;
    }
    /* Trailing uniform binding for PODArgs. */
    bgl_entries[num_handle_args].binding = num_handle_args;
    bgl_entries[num_handle_args].visibility = WGPU_SHADER_STAGE_COMPUTE;
    bgl_entries[num_handle_args].ty_discriminant = WGPU_BINDING_TYPE_DISCR_BUFFER;
    bgl_entries[num_handle_args].buffer.type = WGPU_BUFFER_BINDING_TYPE_UNIFORM;
    bgl_entries[num_handle_args].buffer.has_dynamic_offset = 0;
    bgl_entries[num_handle_args].buffer.min_binding_size_is_some = 0;
    bgl_entries[num_handle_args].buffer.min_binding_size = 0;

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

    /* browser:webgpu addition: allocate a pipeline-layout resource wrapping
     * the bind-group-layout before create-compute-pipeline. Cognition's
     * kernels use exactly one bind-group, so the layouts list has one
     * element. */
    int32_t bgl_list[1] = {fn->bind_group_layout_h};
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_pipeline_layout(dev->device_h, bgl_list, 1u, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_bind_group_layout_drop(fn->bind_group_layout_h);
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        return -1;
    }
    fn->pipeline_layout_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Create the compute pipeline. Descriptor: compute-state {module,
     * entry-point} + pipeline-layout-option {explicit(pipeline_layout_h)}. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_compute_pipeline(dev->device_h, fn->shader_module_h,
                                            (const uint8_t *)entry_ptr, entry_len,
                                            WGPU_PIPELINE_LAYOUT_OPTION_EXPLICIT,
                                            fn->pipeline_layout_h, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_pipeline_layout_drop(fn->pipeline_layout_h);
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
     * queue-write-buffer rewrites contents each dispatch. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_buffer(dev->device_h, (uint64_t)fn->pod_bytes,
                                  WGPU_USAGE_UNIFORM | WGPU_USAGE_COPY_DST,
                                  0 /* mapped-at-creation */, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_compute_pipeline_drop(fn->pipeline_h);
        wgpu_wit_pipeline_layout_drop(fn->pipeline_layout_h);
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

/* ---------------------------------------------------------------------
 * GUEST-VMCREATE-DECODER-PIPELINE-BATCH — batched compute-pipeline
 * creation.
 *
 * Wire layout for the batched call's arg list. `list<compute-pipeline-
 * descriptor>` collapses to (ptr, len); each element is packed at
 * `sizeof(struct wgpu_compute_pipeline_descriptor_wire)` stride.
 *
 * compute-pipeline-descriptor canonical ABI (20 bytes, align 4):
 *   offset  0-3:  compute.module (i32 borrow handle)
 *   offset  4-7:  compute.entry-point.ptr (u32)
 *   offset  8-11: compute.entry-point.len (u32)
 *   offset 12:    pipeline-layout-option discriminant (u8)  0=auto, 1=explicit
 *   offset 13-15: padding to align 4
 *   offset 16-19: pipeline-layout-option.explicit payload (i32 borrow handle;
 *                 ignored when discriminant=0)
 *
 * The pipeline-layout-option variant's arms are: `auto` (empty) and
 * `explicit(borrow<pipeline-layout>)` — payload size 4, align 4. Its
 * canonical ABI layout is discriminant (u8) + padding + payload (4)
 * → 8 bytes at align 4. Nested inside the outer record, it starts at
 * offset 12 (aligned) and occupies 12..20.
 * ------------------------------------------------------------------- */
struct wgpu_compute_pipeline_descriptor_wire {
    int32_t shader_module_h;
    uint32_t entry_ptr;
    uint32_t entry_len;
    uint8_t layout_discriminant;
    uint8_t _pad[3];
    int32_t pipeline_layout_h;
};

/* Return-element wire layout for list<result<compute-pipeline, gpu-error>>.
 *
 * result<T, E> canonical ABI (16 bytes, align 4) where:
 *   T = compute-pipeline handle (i32, size 4, align 4)
 *   E = gpu-error variant (5 arms each carrying `string`):
 *     - variant discriminant (u8) + padding to align 4
 *     - string.ptr (u32) + string.len (u32)
 *     Total: 12 bytes, align 4
 *
 * Element layout:
 *   offset 0:    outer tag (u8)  — 0=ok, 1=err
 *   offset 1-3:  padding to align 4
 *   offset 4-7:  ok payload: pipeline handle (i32)                     — arm=0
 *   offset 4:    err payload: gpu-error variant tag (u8)               — arm=1
 *   offset 5-7:  err padding
 *   offset 8-11: err string.ptr (u32)
 *   offset 12-15:err string.len (u32)
 *
 * The union'd payload region spans [4..16]; the ok arm only touches
 * [4..8], and the err arm touches [4..16]. Padding bytes are
 * indeterminate — do not compare element-wise.
 */
struct wgpu_compute_pipeline_result_wire {
    uint8_t outer_tag;
    uint8_t _pad0[3];
    union {
        struct {
            int32_t pipeline_h;
        } ok;
        struct {
            uint8_t err_tag;
            uint8_t _pad1[3];
            uint32_t msg_ptr;
            uint32_t msg_len;
        } err;
    } payload;
};

/* Compile-time asserts so the wire structs match the canonical ABI. If
 * a compiler pads unexpectedly the batched call would silently misread
 * host output — fail the build instead. */
_Static_assert(sizeof(struct wgpu_compute_pipeline_descriptor_wire) == 20,
               "compute-pipeline-descriptor wire layout must be 20 bytes");
_Static_assert(sizeof(struct wgpu_compute_pipeline_result_wire) == 16,
               "result<compute-pipeline, gpu-error> wire layout must be 16 bytes");

/* Internal helper: create the shader-module + bind-group-layout +
 * pipeline-layout + POD-uniform buffer for one kernel, deferring the
 * compute-pipeline creation to the batched dispatch. Populates every
 * field of @p fn EXCEPT pipeline_h (left zero-initialised so the
 * caller's teardown-on-error path can skip the compute-pipeline drop).
 *
 * Returns 0 on success. On failure, releases every host resource this
 * call allocated, frees @p fn, and sets *out_fn to NULL. */
static int wgpu_kernel_setup_pre_pipeline(WGPU_Device device,
                                          const WGPU_KernelBatchInfo *info,
                                          struct WGPU_Function_st **out_fn) {
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;
    struct WGPU_Function_st *fn = calloc(1, sizeof(struct WGPU_Function_st));
    if (!fn) {
        TVMAPISetLastError("WGPU_FunctionCreateBatch: out of memory");
        return -1;
    }
    fn->device = dev;
    fn->num_handle_args = info->num_handle_args;
    fn->num_pod_args = info->num_pod_args;
    if (info->num_handle_args > 0 && info->handle_write_access) {
        fn->handle_write_access = calloc(info->num_handle_args, sizeof(uint8_t));
        if (!fn->handle_write_access) {
            free(fn);
            TVMAPISetLastError("WGPU_FunctionCreateBatch: out of memory (write-access)");
            return -1;
        }
        memcpy(fn->handle_write_access, info->handle_write_access,
               sizeof(uint8_t) * info->num_handle_args);
    }
    if (info->num_pod_args > 0 && info->pod_arg_dtypes) {
        fn->pod_arg_dtypes = calloc(info->num_pod_args, sizeof(DLDataType));
        if (!fn->pod_arg_dtypes) {
            free(fn->handle_write_access);
            free(fn);
            TVMAPISetLastError("WGPU_FunctionCreateBatch: out of memory (pod dtypes)");
            return -1;
        }
        memcpy(fn->pod_arg_dtypes, info->pod_arg_dtypes,
               sizeof(DLDataType) * info->num_pod_args);
    }
    fn->pod_bytes = (info->num_pod_args + 1u) * 4u;
    if (fn->pod_bytes < 16u) {
        fn->pod_bytes = 16u;
    }

    /* Create shader-module. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_shader_module(dev->device_h, (const uint8_t *)info->source,
                                         info->source_len, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        return -1;
    }
    fn->shader_module_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Build bind-group-layout entries — same shape as the single-kernel
     * path (N storage bindings + 1 trailing uniform for PODArgs). */
    uint32_t bgl_count = info->num_handle_args + 1u;
    struct wgpu_bgl_entry_wire *bgl_entries =
        calloc(bgl_count, sizeof(struct wgpu_bgl_entry_wire));
    if (!bgl_entries) {
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        TVMAPISetLastError("WGPU_FunctionCreateBatch: out of memory (bgl entries)");
        return -1;
    }
    for (uint32_t i = 0; i < info->num_handle_args; ++i) {
        bgl_entries[i].binding = i;
        bgl_entries[i].visibility = WGPU_SHADER_STAGE_COMPUTE;
        bgl_entries[i].ty_discriminant = WGPU_BINDING_TYPE_DISCR_BUFFER;
        uint8_t wa = info->handle_write_access ? info->handle_write_access[i] : 1u;
        bgl_entries[i].buffer.type = wa ? WGPU_BUFFER_BINDING_TYPE_STORAGE
                                        : WGPU_BUFFER_BINDING_TYPE_READ_ONLY_STORAGE;
        bgl_entries[i].buffer.has_dynamic_offset = 0;
        bgl_entries[i].buffer.min_binding_size_is_some = 0;
        bgl_entries[i].buffer.min_binding_size = 0;
    }
    bgl_entries[info->num_handle_args].binding = info->num_handle_args;
    bgl_entries[info->num_handle_args].visibility = WGPU_SHADER_STAGE_COMPUTE;
    bgl_entries[info->num_handle_args].ty_discriminant = WGPU_BINDING_TYPE_DISCR_BUFFER;
    bgl_entries[info->num_handle_args].buffer.type = WGPU_BUFFER_BINDING_TYPE_UNIFORM;

    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_bind_group_layout(dev->device_h, (const uint8_t *)bgl_entries,
                                             bgl_count, wgpu_wit_ret_area);
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

    /* Wrap bind-group-layout in a pipeline-layout resource. */
    int32_t bgl_list[1] = {fn->bind_group_layout_h};
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_pipeline_layout(dev->device_h, bgl_list, 1u, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_bind_group_layout_drop(fn->bind_group_layout_h);
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        return -1;
    }
    fn->pipeline_layout_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    /* Allocate the PODArgs uniform buffer. Reused across dispatches. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_buffer(dev->device_h, (uint64_t)fn->pod_bytes,
                                  WGPU_USAGE_UNIFORM | WGPU_USAGE_COPY_DST,
                                  0 /* mapped-at-creation */, wgpu_wit_ret_area);
    if (wgpu_wit_ret_area[0] != 0) {
        wgpu_wit_forward_error(wgpu_wit_ret_area);
        wgpu_wit_pipeline_layout_drop(fn->pipeline_layout_h);
        wgpu_wit_bind_group_layout_drop(fn->bind_group_layout_h);
        wgpu_wit_shader_module_drop(fn->shader_module_h);
        free(fn->pod_arg_dtypes);
        free(fn->handle_write_access);
        free(fn);
        return -1;
    }
    fn->pod_buffer_h = *(const int32_t *)(wgpu_wit_ret_area + 4);

    *out_fn = fn;
    return 0;
}

/* Internal helper: drop every host resource associated with a
 * pre-pipeline-populated fn and free it. Matches the teardown paths in
 * wgpu_kernel_setup_pre_pipeline; safe to call on a fn that also has
 * `pipeline_h` populated. */
static void wgpu_kernel_teardown(struct WGPU_Function_st *fn) {
    if (!fn) {
        return;
    }
    if (fn->pipeline_h) {
        wgpu_wit_compute_pipeline_drop(fn->pipeline_h);
    }
    if (fn->pod_buffer_h) {
        wgpu_wit_buffer_drop(fn->pod_buffer_h);
    }
    if (fn->pipeline_layout_h) {
        wgpu_wit_pipeline_layout_drop(fn->pipeline_layout_h);
    }
    if (fn->bind_group_layout_h) {
        wgpu_wit_bind_group_layout_drop(fn->bind_group_layout_h);
    }
    if (fn->shader_module_h) {
        wgpu_wit_shader_module_drop(fn->shader_module_h);
    }
    free(fn->pod_arg_dtypes);
    free(fn->handle_write_access);
    free(fn);
}

int WGPU_FunctionCreateBatch(WGPU_Device device, uint32_t n,
                             const WGPU_KernelBatchInfo *infos,
                             WGPU_Function *functions_out) {
    if (n == 0) {
        return 0;
    }
    if (!infos || !functions_out) {
        TVMAPISetLastError("WGPU_FunctionCreateBatch: null infos/functions_out");
        return -1;
    }
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;

    /* Zero the output slots up front so failure paths can safely
     * wgpu_kernel_teardown any entry populated so far. */
    for (uint32_t i = 0; i < n; ++i) {
        functions_out[i] = NULL;
    }
    struct WGPU_Function_st **fns = calloc(n, sizeof(struct WGPU_Function_st *));
    struct wgpu_compute_pipeline_descriptor_wire *descs =
        calloc(n, sizeof(struct wgpu_compute_pipeline_descriptor_wire));
    if (!fns || !descs) {
        free(fns);
        free(descs);
        TVMAPISetLastError("WGPU_FunctionCreateBatch: out of memory (staging)");
        return -1;
    }

    /* Stage 1: per-kernel shader-module + BGL + PL + POD buffer.
     * Each iteration is 4 WIT crossings — same as the single-kernel
     * path minus the pipeline. On any failure, unwind everything
     * populated so far. */
    for (uint32_t i = 0; i < n; ++i) {
        struct WGPU_Function_st *fn = NULL;
        if (wgpu_kernel_setup_pre_pipeline(device, &infos[i], &fn) != 0) {
            for (uint32_t j = 0; j < i; ++j) {
                wgpu_kernel_teardown(fns[j]);
            }
            free(fns);
            free(descs);
            return -1;
        }
        fns[i] = fn;

        /* Build the descriptor entry for this slot. Default the entry
         * name to "main" when the caller passed NULL/0, matching
         * WGPU_FunctionCreate's behaviour. */
        const char *entry_ptr = infos[i].entry_name;
        uint32_t entry_len = infos[i].entry_name_len;
        if (entry_ptr == NULL || entry_len == 0) {
            entry_ptr = "main";
            entry_len = 4;
        }
        descs[i].shader_module_h = fn->shader_module_h;
        descs[i].entry_ptr = (uint32_t)(uintptr_t)entry_ptr;
        descs[i].entry_len = entry_len;
        descs[i].layout_discriminant = WGPU_PIPELINE_LAYOUT_OPTION_EXPLICIT;
        descs[i].pipeline_layout_h = fn->pipeline_layout_h;
    }

    /* Stage 2: ONE batched pipeline dispatch. Host runs
     * `createComputePipelineAsync` per descriptor and awaits them all
     * via `Promise.all(...)` so WGSL compiles run in parallel. */
    memset(wgpu_wit_ret_area, 0, WGPU_WIT_RET_AREA_SIZE);
    wgpu_wit_device_create_compute_pipelines_async(dev->device_h, (const uint8_t *)descs, n,
                                                   wgpu_wit_ret_area);
    uint32_t results_ptr = *(const uint32_t *)(wgpu_wit_ret_area + 0);
    uint32_t results_len = *(const uint32_t *)(wgpu_wit_ret_area + 4);
    if (results_len != n) {
        for (uint32_t i = 0; i < n; ++i) {
            wgpu_kernel_teardown(fns[i]);
        }
        free(fns);
        free(descs);
        TVMAPISetLastError("WGPU_FunctionCreateBatch: host returned wrong result count");
        return -1;
    }

    /* Stage 3: attach the returned pipeline handle to each fn. If ANY
     * slot failed compilation, tear the whole batch down atomically —
     * partial-success semantics live in the WIT return type but the
     * caller (WebGPUModuleCreate) can't cope with a partial module, so
     * fail the whole batch here. Report the first error message. */
    const struct wgpu_compute_pipeline_result_wire *results =
        (const struct wgpu_compute_pipeline_result_wire *)(uintptr_t)results_ptr;
    int any_err = 0;
    char first_err[513];
    first_err[0] = '\0';
    for (uint32_t i = 0; i < n; ++i) {
        if (results[i].outer_tag == 0) {
            fns[i]->pipeline_h = results[i].payload.ok.pipeline_h;
        } else {
            if (!any_err) {
                uint32_t msg_ptr = results[i].payload.err.msg_ptr;
                uint32_t msg_len = results[i].payload.err.msg_len;
                if (msg_ptr && msg_len) {
                    size_t cap = msg_len < 512 ? (size_t)msg_len : 512;
                    memcpy(first_err, (const void *)(uintptr_t)msg_ptr, cap);
                    first_err[cap] = '\0';
                }
                any_err = 1;
            }
        }
    }
    if (any_err) {
        for (uint32_t i = 0; i < n; ++i) {
            wgpu_kernel_teardown(fns[i]);
        }
        free(fns);
        free(descs);
        if (first_err[0]) {
            TVMAPISetLastError(first_err);
        } else {
            TVMAPISetLastError("WGPU_FunctionCreateBatch: pipeline compile failed");
        }
        return -1;
    }

    /* All good — hand ownership to the caller. */
    for (uint32_t i = 0; i < n; ++i) {
        functions_out[i] = (WGPU_Function)fns[i];
    }
    free(fns);
    free(descs);
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

/* ---------------------------------------------------------------------
 * BATCH-TVM kernel-dispatch batching (private helpers).
 *
 * The public entry points WGPU_BeginKernelBatch / WGPU_FlushKernelBatch /
 * WGPU_EndKernelBatch live below. These statics implement the on-demand
 * encoder/pass management and the pending-bind-group / pending-shadow
 * grow-only queues that the batched WGPU_FunctionRun fast-path feeds.
 * ------------------------------------------------------------------- */

/* Grow @p arr from *cap to at least min_cap slots (doubling policy).
 * Returns 0 on success; -1 on OOM. On failure the existing array is
 * left intact for the caller to keep using (partial-fill drops the new
 * entry; batch remains flushable). */
static int wgpu_batch_grow_i32(int32_t **arr, size_t *cap, size_t min_cap) {
    if (*cap >= min_cap) {
        return 0;
    }
    size_t new_cap = *cap ? *cap : 16u;
    while (new_cap < min_cap) {
        new_cap <<= 1u;
    }
    int32_t *fresh = realloc(*arr, new_cap * sizeof(int32_t));
    if (!fresh) {
        return -1;
    }
    *arr = fresh;
    *cap = new_cap;
    return 0;
}

/* Push @p h onto @p batch's pending_bind_groups. Returns 0 / -1(OOM). */
static int wgpu_batch_push_bg(struct WGPU_Batch_st *b, int32_t h) {
    if (wgpu_batch_grow_i32(&b->pending_bind_groups, &b->pending_bg_capacity,
                            b->pending_bg_count + 1u) != 0) {
        return -1;
    }
    b->pending_bind_groups[b->pending_bg_count++] = h;
    return 0;
}

/* Push a shadow buffer handle onto @p batch's pending_shadows. Returns 0 / -1. */
static int wgpu_batch_push_shadow(struct WGPU_Batch_st *b, int32_t h) {
    if (wgpu_batch_grow_i32(&b->pending_shadows, &b->pending_shadow_capacity,
                            b->pending_shadow_count + 1u) != 0) {
        return -1;
    }
    b->pending_shadows[b->pending_shadow_count++] = h;
    return 0;
}

/* Close the batch's current compute pass (if any). No-op otherwise.
 * Encoder stays open — a subsequent dispatch or shadow-copy can still
 * emit onto it before flush. */
static void wgpu_batch_close_pass(struct WGPU_Batch_st *b) {
    if (!b->has_pass) {
        return;
    }
    wgpu_wit_pass_end(b->pass_h);
    wgpu_wit_compute_pass_encoder_drop(b->pass_h);
    b->has_pass = 0;
    b->pass_h = 0;
}

/* Ensure an encoder is open on @p device's batch. Callers hold the
 * device handle; created encoders are owned by the batch and released
 * at flush. No-op if one is already open. */
static void wgpu_batch_ensure_encoder(struct WGPU_Device_st *dev) {
    if (dev->batch.has_encoder) {
        return;
    }
    dev->batch.encoder_h = wgpu_wit_device_create_command_encoder(dev->device_h);
    dev->batch.has_encoder = 1;
}

/* Ensure a compute pass is open on @p device's batch. Opens an encoder
 * first if needed. Passes are re-created after each flush and after each
 * shadow-copy pass-break within a batch. */
static void wgpu_batch_ensure_pass(struct WGPU_Device_st *dev) {
    if (dev->batch.has_pass) {
        return;
    }
    wgpu_batch_ensure_encoder(dev);
    dev->batch.pass_h = wgpu_wit_encoder_begin_compute_pass(dev->batch.encoder_h);
    dev->batch.has_pass = 1;
}

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

        /* device.queue-write-buffer is fire-and-forget (no result wrapper). */
        wgpu_wit_device_queue_write_buffer(fn->device->device_h, fn->pod_buffer_h, 0u, pod_bytes,
                                           fn->pod_bytes);
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
     * device.submit. VITS runs cumsum a bounded number of times per
     * synthesis, so per-dispatch alloc/free is acceptable; a pool can
     * be layered on later if profiling calls for it. */
    int32_t *bind_buffer_h = NULL;
    int32_t *shadow_h = NULL;
    uint64_t *shadow_size = NULL;
    /* Non-zero when at least one shadow was allocated for this dispatch —
     * used below to pass-break the batched encoder before emitting the
     * pre-dispatch copy-buffer-to-buffer commands. */
    uint32_t num_shadows = 0;
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
                ++num_shadows;
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
        struct WGPU_Memory_st *m_i = (struct WGPU_Memory_st *)handle_args[i];
        entries[i].binding = i;
        entries[i].resource_discriminant = WGPU_BINDING_RESOURCE_DISCR_BUFFER;
        /* Route to the shadow when one was allocated for this slot; the
         * source buffer_h stays available for the pre-dispatch snapshot
         * copy below (encoder_copy_buffer_to_buffer reads from the raw
         * WGPU_Memory struct, not from bind_buffer_h). */
        entries[i].buffer.buffer_h = bind_buffer_h[i];
        /* When @p m_i is an alias into a bulk-uploaded parent buffer,
         * the binding must window in on the alias's slice. Standalone
         * buffers keep offset=0 / size_is_some=0 (WebGPU's default
         * "cover the whole buffer" behavior) so this stays free for
         * the pre-alias workload. Shadowed slots preserve their
         * shadow-buffer-scoped binding (parent_offset is 0 on the
         * freshly-allocated shadow). */
        if (m_i->parent_buffer_h != 0 && shadow_h[i] == 0) {
            entries[i].buffer.offset = m_i->parent_offset;
            entries[i].buffer.size_is_some = 1u;
            entries[i].buffer.size = (uint64_t)m_i->size;
        } else {
            entries[i].buffer.offset = 0;
            entries[i].buffer.size_is_some = 0;
            entries[i].buffer.size = 0;
        }
    }
    /* Uniform binding at index num_handle_args, size = fn->pod_bytes. */
    entries[num_handle_args].binding = num_handle_args;
    entries[num_handle_args].resource_discriminant = WGPU_BINDING_RESOURCE_DISCR_BUFFER;
    entries[num_handle_args].buffer.buffer_h = fn->pod_buffer_h;
    entries[num_handle_args].buffer.offset = 0;
    entries[num_handle_args].buffer.size_is_some = 1u;
    entries[num_handle_args].buffer.size = fn->pod_bytes;

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

    /* ---- BATCH-TVM fast path ----
     *
     * When a batch is active on the device, defer end-pass / finish-encoder
     * / submit / bind-group-drop / shadow-destroy to WGPU_FlushKernelBatch.
     * Instead the dispatch just:
     *   - Retains bind_group_h + any shadows for deferred cleanup.
     *   - Pass-breaks the current compute pass if shadows need pre-dispatch
     *     copy-buffer-to-buffer commands (encoder-scope, not pass-scope).
     *   - Emits set-pipeline + set-bind-group + dispatch-workgroups onto
     *     the shared pass.
     *
     * Per-kernel WIT crossings drop from 13 (queue-write + create-bg +
     * create-encoder + begin-pass + set-pipeline + set-bg + dispatch +
     * end + drop-pass + finish + drop-encoder + submit + drop-bg) to 5
     * (queue-write + create-bg + set-pipeline + set-bg + dispatch), plus
     * per-shadow copy-buffer-to-buffer and rare pass-break amortising
     * across the batch. See docs/tvm-boundary-overhead-investigation.md
     * §4.A. */
    struct WGPU_Device_st *dev = fn->device;
    /* Shadow-buffer dispatches (aliased-in-place cumsum on the decoder)
     * fall through to the unbatched path — the shadow-destroy-after-
     * submit ordering has been proven correct there and the aliased
     * kernels are rare enough (a few per inference) that unbatching just
     * those does not materially hurt the crossing-count reduction.
     * Flush any pending batched compute first so this dispatch sees
     * the post-batch state and its own submit can go through cleanly. */
    if (dev->batch.active && num_shadows == 0) {
        /* Push retention before doing any dispatch work so partial-failure
         * paths still free correctly at flush. */
        if (wgpu_batch_push_bg(&dev->batch, bind_group_h) != 0) {
            wgpu_wit_bind_group_drop(bind_group_h);
            WGPU_RUN_FREE_SHADOWS();
            TVMAPISetLastError("WGPU_FunctionRun: batch OOM (bind-group queue)");
            return -1;
        }

        wgpu_batch_ensure_encoder(dev);
        wgpu_batch_ensure_pass(dev);

        wgpu_wit_pass_set_pipeline(dev->batch.pass_h, fn->pipeline_h);
        wgpu_wit_pass_set_bind_group(dev->batch.pass_h, 0u, bind_group_h, NULL, 0u);
        wgpu_wit_pass_dispatch_workgroups(dev->batch.pass_h, launch_x, launch_y, launch_z);

        /* All of pass-end / drop-pass / finish / drop-encoder / submit /
         * drop-bg stay deferred to flush time — that is the whole point
         * of batching. Just release the scratch arrays. */
        free(bind_buffer_h);
        free(shadow_h);
        free(shadow_size);
        return 0;
    }
    if (dev->batch.active && num_shadows > 0) {
        /* Falling through: force any pending batched work to submit before
         * the unbatched shadow-copy sequence emits its own encoder. */
        (void)WGPU_FlushKernelBatch((WGPU_Device)dev);
    }

    /* Encoder → shadow-snapshot copies → compute pass → set pipeline +
     * bind-group → dispatch → end → finish → submit. All commands
     * batch inside one device.submit, i.e. one JSPI round-trip for the
     * whole dispatch. create-command-encoder, copy-buffer-to-buffer,
     * begin-compute-pass, and finish all shed their result wrappers in
     * browser:webgpu — validation errors surface at submit time. */
    int32_t encoder_h = wgpu_wit_device_create_command_encoder(fn->device->device_h);

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
        /* Alias-aware source offset: for a bulk-uploaded constant
         * fed into an aliased-in-place kernel the source is a slice
         * of the parent buffer at m->parent_offset. Standalone
         * buffers keep offset 0. */
        wgpu_wit_encoder_copy_buffer_to_buffer(encoder_h, m->buffer_h,
                                               m->parent_offset, shadow_h[i], 0,
                                               shadow_size[i]);
    }

    int32_t pass_h = wgpu_wit_encoder_begin_compute_pass(encoder_h);

    wgpu_wit_pass_set_pipeline(pass_h, fn->pipeline_h);
    /* compute-pass-encoder.set-bind-group grew a required dynamic-offsets
     * list<u32> parameter. Cognition binds no dynamic-offset entries — pass
     * empty. */
    wgpu_wit_pass_set_bind_group(pass_h, 0u, bind_group_h, NULL, 0u);
    wgpu_wit_pass_dispatch_workgroups(pass_h, launch_x, launch_y, launch_z);
    wgpu_wit_pass_end(pass_h);
    wgpu_wit_compute_pass_encoder_drop(pass_h);

    int32_t cmd_buf_h = wgpu_wit_encoder_finish(encoder_h);
    wgpu_wit_encoder_drop(encoder_h);

    wgpu_wit_device_submit(fn->device->device_h, &cmd_buf_h, 1u);
    /* bind_group ownership is retained across the submit — the compute
     * pass borrowed it. Drop after the queue accepts the batch. */
    wgpu_wit_bind_group_drop(bind_group_h);

    /* Success — destroy shadows (WebGPU allows buffer.destroy() while
     * prior in-flight submissions complete; the submitted commands
     * hold internal refs on the buffer until the queue drains). */
    WGPU_RUN_FREE_SHADOWS();
#undef WGPU_RUN_FREE_SHADOWS
    return 0;
}

int WGPU_BeginKernelBatch(WGPU_Device device) {
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;
    if (!dev) {
        return 0;
    }
    /* Nested Begin is a no-op — outermost pair rules. This lets composed
     * callers (e.g. a Relax VM entry-point wrapping another wrapper that
     * also opens a batch) stack without leaking encoder state. */
    /* Nested Begin is a no-op — outermost pair rules. This lets composed
     * callers (e.g. a Relax VM entry-point wrapping another wrapper that
     * also opens a batch) stack without leaking encoder state. */
    if (dev->batch.active) {
        return 0;
    }
    /* Encoder/pass stay lazy — created on the first dispatch inside this
     * batch. Retained arrays keep their capacity across Begin/End cycles
     * (freed only at DeviceFree) to amortise the ~200-dispatch VITS
     * decoder workload. */
    dev->batch.active = 1;
    dev->batch.has_encoder = 0;
    dev->batch.has_pass = 0;
    dev->batch.encoder_h = 0;
    dev->batch.pass_h = 0;
    dev->batch.pending_bg_count = 0;
    dev->batch.pending_shadow_count = 0;
    return 0;
}

int WGPU_FlushKernelBatch(WGPU_Device device) {
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;
    if (!dev || !dev->batch.active) {
        return 0;
    }
    /* Nothing pending — flush is idempotent. */
    if (!dev->batch.has_encoder) {
        dev->batch.pending_bg_count = 0;
        dev->batch.pending_shadow_count = 0;
        return 0;
    }
    /* Close the compute pass so encoder.finish can materialise the
     * command buffer. If a shadow-copy already closed it earlier in the
     * batch and no dispatch reopened it, has_pass is false — skipped. */
    wgpu_batch_close_pass(&dev->batch);

    int32_t cmd_buf_h = wgpu_wit_encoder_finish(dev->batch.encoder_h);
    wgpu_wit_encoder_drop(dev->batch.encoder_h);
    dev->batch.encoder_h = 0;
    dev->batch.has_encoder = 0;

    /* Single submit for the whole batch — the primary boundary-cost
     * win of BATCH-TVM. Command buffer is consumed by submit; no
     * separate command-buffer drop needed. */
    wgpu_wit_device_submit(dev->device_h, &cmd_buf_h, 1u);

    /* Post-submit cleanup: drop retained bind-groups (the ended pass
     * has released its borrows) and destroy retained shadow buffers
     * (WebGPU accepts destroy() while in-flight submits still hold
     * internal refs — matches the unbatched path's post-submit shadow
     * treatment). */
    for (size_t i = 0; i < dev->batch.pending_bg_count; ++i) {
        wgpu_wit_bind_group_drop(dev->batch.pending_bind_groups[i]);
    }
    dev->batch.pending_bg_count = 0;
    for (size_t i = 0; i < dev->batch.pending_shadow_count; ++i) {
        wgpu_wit_buffer_destroy(dev->batch.pending_shadows[i]);
        wgpu_wit_buffer_drop(dev->batch.pending_shadows[i]);
    }
    dev->batch.pending_shadow_count = 0;
    return 0;
}

int WGPU_EndKernelBatch(WGPU_Device device) {
    struct WGPU_Device_st *dev = (struct WGPU_Device_st *)device;
    if (!dev || !dev->batch.active) {
        return 0;
    }
    int rc = WGPU_FlushKernelBatch(device);
    dev->batch.active = 0;
    return rc;
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
    if (fn->pipeline_layout_h) {
        wgpu_wit_pipeline_layout_drop(fn->pipeline_layout_h);
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
