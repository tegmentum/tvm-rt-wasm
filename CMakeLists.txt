cmake_minimum_required(VERSION 3.19)

##---------------------- Project options and default variables ---------------------------------------------------------
## Project options:
## * USE_WASI_SDK: If use the wasi-sdk, set `/path/to/wasi-sdk` or set `ON`/`AUTO`/`TRUE` to use default `/opt/wasi-sdk`.
##
## The fork is CPU-only and Relax-VM-only. WebGPU, CUDA, Graph Executor,
## Relay VM, and AoT paths have been removed — see docs/PORT_DELETIONS.md.
## The emsdk build path has also been removed; cognition targets
## wasm32-wasip1 via wasi-sdk exclusively.

# Default build type: Release
# Default install path: /opt/tvm-rt-wasm/
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX /opt/tvm-rt-wasm/ CACHE PATH "" FORCE)
endif ()

# Used for the developers.
option(BUILD_EXAMPLES "Build the examples." OFF)
option(BUILD_TESTS "Build the tests." OFF)
option(BUILD_MEMORY_CHECK "Build with the memory sanitize." OFF)

#------------------------- Check options and setup toolchain -----------------------------------------------------------
macro(_is_bool_value _str _ret)
    string(TOUPPER ${_str} _s)
    if (${_s} STREQUAL "ON" OR ${_s} STREQUAL "TRUE" OR ${_s} STREQUAL "YES" OR ${_s} STREQUAL "AUTO" OR ${_s} STREQUAL "1")
        set(${_ret} ON)
    else ()
        set(${_ret} OFF)
    endif ()
endmacro()

if (DEFINED USE_WASI_SDK)
    _is_bool_value("${USE_WASI_SDK}" _use_default)
    if (${_use_default})
        set(WASI_SDK_PREFIX /opt/wasi-sdk)
        message(STATUS "Use the default wasi-sdk path: ${WASI_SDK_PREFIX}")
    else ()
        set(WASI_SDK_PREFIX ${USE_WASI_SDK})
        message(STATUS "Use the wasi-sdk path: ${WASI_SDK_PREFIX}")
    endif ()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128 -mbulk-memory")
    # Propagate the wasi-sdk selection into the preprocessor so
    # `#if USE_WASI_SDK` gates in the fork's C sources (e.g. the
    # SharedLibrary path in module/shared_library.c) actually fire.
    add_definitions(-DUSE_WASI_SDK=1)
    # Cognition targets wasm32-wasip1 (WASI Preview 1). Use the -p1 toolchain
    # instead of the legacy wasi-sdk.cmake, which passes the deprecated
    # --target=wasm32-wasi and trips -Werror,-Wdeprecated on clang 22+.
    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk-p1.cmake)
endif ()

#------------------------- Project -------------------------------------------------------------------------------------
project(tvm-rt-wasm LANGUAGES C VERSION 0.2.0)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_definitions(-DTVM_EXPORTS)
include_directories(include 3rdparty/dlpack/include 3rdparty/tvm/include src/core)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fno-exceptions -ffunction-sections -fdata-sections -fvisibility=hidden")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-error=unused-function")

## Core library target
file(GLOB_RECURSE _LIB_CORE_SRC src/core/*.c)
add_library(tvm-rt-core STATIC ${_LIB_CORE_SRC})
target_include_directories(tvm-rt-core INTERFACE $<INSTALL_INTERFACE:include>)

## Backend library targets.
function(add_backend_library _target_name _backend_folder)
    file(GLOB_RECURSE _backend_sources ${_backend_folder}/*.c)
    add_library(${_target_name} STATIC ${_backend_sources})
    target_include_directories(${_target_name} PRIVATE src/backends)
endfunction()
add_backend_library(tvm-rt-backend-relax-vm src/backends/relax_vm)

## Accelerator library targets (M14 — opt-in via USE_WEBGPU).
## When USE_WEBGPU=ON the fork compiles the WebGPU accelerator restored
## from yanghaku's baseline. Under wasi-sdk the JS/native impls of the
## WGPU_* c_api surface are empty (native_impl needs Dawn headers, and
## js_impl is #ifdef __EMSCRIPTEN__ only), so the resulting static lib
## exposes unresolved WGPU_* symbols at link time — M14.3 wires them to
## host:webgpu WIT imports.
if (USE_WEBGPU)
    file(GLOB_RECURSE _WEBGPU_SRC
        src/accelerators/webgpu/webgpu_device_api.c
        src/accelerators/webgpu/webgpu_module.c
        src/accelerators/webgpu/c_api/webgpu_js_impl.c)
    add_library(tvm-rt-accelerator-webgpu STATIC ${_WEBGPU_SRC})
    target_include_directories(tvm-rt-accelerator-webgpu PRIVATE
        src/accelerators/webgpu)
endif ()

# install targets
set(CMAKE_INSTALL_CMAKE_DIR lib/cmake/tvm-rt-wasm)

function(install_library _target_name)
    install(TARGETS ${_target_name} EXPORT tvm-rt-wasm-targets)
endfunction()
install_library(tvm-rt-core)
install_library(tvm-rt-backend-relax-vm)

# install targets.cmake
install(EXPORT tvm-rt-wasm-targets FILE tvm-rt-wasm-targets.cmake
        CONFIGURATIONS Release
        DESTINATION ${CMAKE_INSTALL_CMAKE_DIR}
        NAMESPACE tvm-rt-wasm::)
# install config-version.cmake and config.cmake
include(CMakePackageConfigHelpers)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cmake/tvm-rt-wasm-config.cmake
        "include(\${CMAKE_CURRENT_LIST_DIR}/tvm-rt-wasm-targets.cmake)\n")
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/tvm-rt-wasm-config-version.cmake
        VERSION ${CMAKE_PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake/tvm-rt-wasm-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/tvm-rt-wasm-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_CMAKE_DIR})

# install public header files
# TVM 0.25 deleted `tvm/runtime/c_runtime_api.h`; the public relax_vm.h
# only needs the FFI c_api and base headers to compile downstream.
install(FILES 3rdparty/tvm/include/tvm/ffi/c_api.h DESTINATION include/tvm/ffi/)
install(FILES 3rdparty/tvm/include/tvm/runtime/base.h DESTINATION include/tvm/runtime/)
install(FILES 3rdparty/dlpack/include/dlpack/dlpack.h DESTINATION include/dlpack/)
install(FILES include/relax_vm.h DESTINATION include)

# install LICENSE file
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_PREFIX})

if (${BUILD_MEMORY_CHECK})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
endif ()

#------------------------- Correctness gate -----------------------------------
# Milestone 13.2: link the fork's static libs against a toy Relax program's
# compiled TIR + devc blob and produce a wasm32-wasip1 CLI executable that
# wasmtime can drive end-to-end. Enabled only when wasi-sdk is on and the
# Python-side compile (test/toy_relax.py) has produced test/toy_out/*.o.
#
# Contract:
#   1. `python3 test/toy_relax.py` produces
#      `test/toy_out/toy_relax.tar` containing `devc.o` (blob-registrar
#      constructor) and `lib0.o` (TIR kernels + weak ctx var).
#   2. The tar gets extracted to `test/toy_out/extracted/` and re-packed
#      by llvm-ar into `libtoy_relax_gen.a`.
#   3. `toy_relax_test.wasm` links the driver + gen lib + fork libs.
#      `--whole-archive` around the gen lib forces the .init_array
#      constructor in devc.o to be pulled in (it has no direct symbol
#      reference from the driver; without whole-archive the linker
#      would drop it and the blob would never register).
if (DEFINED USE_WASI_SDK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/toy_relax.c)
    set(_TOY_TAR ${CMAKE_CURRENT_SOURCE_DIR}/test/toy_out/toy_relax.tar)
    set(_TOY_EXTRACT ${CMAKE_CURRENT_BINARY_DIR}/toy_extract)
    set(_TOY_GEN_LIB ${CMAKE_CURRENT_BINARY_DIR}/libtoy_relax_gen.a)
    add_custom_command(
        OUTPUT ${_TOY_GEN_LIB}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${_TOY_EXTRACT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_TOY_EXTRACT}
        COMMAND tar -xf ${_TOY_TAR} -C ${_TOY_EXTRACT}
        COMMAND ${WASI_SDK_PREFIX}/bin/llvm-ar rcs ${_TOY_GEN_LIB}
                ${_TOY_EXTRACT}/devc.o ${_TOY_EXTRACT}/lib0.o
        DEPENDS ${_TOY_TAR}
        COMMENT "Repacking ${_TOY_TAR} into libtoy_relax_gen.a"
        VERBATIM)
    add_custom_target(toy_relax_gen_lib DEPENDS ${_TOY_GEN_LIB})

    add_executable(toy_relax_test test/toy_relax.c)
    set_target_properties(toy_relax_test PROPERTIES SUFFIX ".wasm")
    add_dependencies(toy_relax_test toy_relax_gen_lib)
    # The fork's `MODULE_CREATE_IF_NO_SUPPORT(RelaxExecutable)` shim
    # (src/core/module/module.c) provides a `weak`-marked
    # `TVM_RT_WASM_RelaxExecutableModuleCreate` alongside its
    # `ModuleCreateFromReader` caller. Because the reference and the weak
    # definition live in the same TU, the linker satisfies the reference
    # from the weak stub the moment module.c.obj is pulled — the strong
    # definition in libtvm-rt-backend-relax-vm.a never gets a chance.
    # `--whole-archive` around the backend lib forces every backend TU in
    # regardless, so the strong symbol wins.
    target_link_libraries(toy_relax_test PRIVATE
        -Wl,--whole-archive ${_TOY_GEN_LIB} tvm-rt-backend-relax-vm
        -Wl,--no-whole-archive
        tvm-rt-core)
endif ()

#------------------------- Milestone 13.3 encoder gate ------------------------
# Same shape as the toy gate above, but the tar comes from
# `test/compile_encoder_wasm.py` (cognition's nano encoder ported to
# wasm32-wasi) and the kernel-registration stub is autogenerated from
# lib0.o's symbol table because the encoder has ~100 kernels — too many
# to hand-list the way the toy program does.
if (DEFINED USE_WASI_SDK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/encoder_test.c)
    set(_ENC_TAR ${CMAKE_CURRENT_SOURCE_DIR}/test/encoder_out/encoder.tar)
    set(_ENC_EXTRACT ${CMAKE_CURRENT_BINARY_DIR}/encoder_extract)
    set(_ENC_GEN_LIB ${CMAKE_CURRENT_BINARY_DIR}/libencoder_gen.a)
    set(_ENC_STUB_C ${CMAKE_CURRENT_BINARY_DIR}/encoder_kernels_stub.c)

    add_custom_command(
        OUTPUT ${_ENC_GEN_LIB} ${_ENC_STUB_C}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${_ENC_EXTRACT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_ENC_EXTRACT}
        COMMAND tar -xf ${_ENC_TAR} -C ${_ENC_EXTRACT}
        COMMAND ${CMAKE_COMMAND} -E rename ${_ENC_EXTRACT}/enc_devc.o ${_ENC_EXTRACT}/devc.o
        COMMAND ${WASI_SDK_PREFIX}/bin/llvm-ar rcs ${_ENC_GEN_LIB}
                ${_ENC_EXTRACT}/devc.o ${_ENC_EXTRACT}/lib0.o
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/gen_kernel_stub.py
                ${_ENC_EXTRACT}/lib0.o ${_ENC_STUB_C}
                ${WASI_SDK_PREFIX}/bin/llvm-nm
                --sys-lib-prefix=enc_ --register-fn=register_all_enc_kernels
        DEPENDS ${_ENC_TAR} ${CMAKE_CURRENT_SOURCE_DIR}/test/gen_kernel_stub.py
        COMMENT "Repacking ${_ENC_TAR} + generating encoder_kernels_stub.c"
        VERBATIM)
    add_custom_target(encoder_gen_lib DEPENDS ${_ENC_GEN_LIB} ${_ENC_STUB_C})

    add_executable(encoder_test test/encoder_test.c ${_ENC_STUB_C})
    set_target_properties(encoder_test PROPERTIES SUFFIX ".wasm")
    add_dependencies(encoder_test encoder_gen_lib)
    target_link_libraries(encoder_test PRIVATE
        -Wl,--whole-archive ${_ENC_GEN_LIB} tvm-rt-backend-relax-vm
        -Wl,--no-whole-archive
        tvm-rt-core)
endif ()

#------------------------- Milestone 13.4 decoder gate ------------------------
# Same shape as the encoder gate above, but the tar comes from
# `test/compile_decoder_wasm.py` and the driver is
# `test/decoder_test.c`. The kernel-registration stub reuses the
# `register_all_encoder_kernels` name from gen_kernel_stub.py — that
# name is historical, not encoder-specific, and works verbatim for
# the decoder's TIR kernels (both stubs are model-independent
# extractions of `__tvm_ffi_*` symbols from lib0.o).
#
# Waveform output is 1MB (f32 [1, 1, 262144]) — link with
# `-Wl,-z,stack-size=8388608` to give the driver room for the two
# static buffers on the wasm data segment plus a healthy stack for
# the VM's recursive interpret path.
#------------------------- Milestone 14.5 toy WebGPU gate ---------------------
# Same shape as toy_relax gate above, but the tar comes from
# `test/toy_webgpu.py` which compiles with `target=webgpu` +
# `host=wasm32-wasi`. The compiled tar has the same devc.o+lib0.o
# envelope (relax_vm executable + system-lib blob) — the difference is
# lib0.o contains a WebGPU-host-dispatch stub for `fused_matmul_add`
# instead of separate host-side matmul/add TIR kernels, and the tar
# also carries a WGSL kernel embedded in the module blob that the
# WebGPU-module loader parses out at load time (see
# src/accelerators/webgpu/webgpu_module.c).
#
# Runs under Node + `webgpu` npm package + JSPI, not wasmtime — see
# cognition's `web/test/toy-webgpu.test.mjs`. The build target here
# just produces the wasm binary; the JS driver satisfies host:webgpu
# via cognition's `web/src/adopt/webgpu-host-impl.js`.
if (DEFINED USE_WASI_SDK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/toy_webgpu_test.c)
    set(_TOY_WGPU_TAR ${CMAKE_CURRENT_SOURCE_DIR}/test/toy_webgpu_out/toy_webgpu.tar)
    set(_TOY_WGPU_EXTRACT ${CMAKE_CURRENT_BINARY_DIR}/toy_webgpu_extract)
    set(_TOY_WGPU_GEN_LIB ${CMAKE_CURRENT_BINARY_DIR}/libtoy_webgpu_gen.a)
    add_custom_command(
        OUTPUT ${_TOY_WGPU_GEN_LIB}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${_TOY_WGPU_EXTRACT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_TOY_WGPU_EXTRACT}
        COMMAND tar -xf ${_TOY_WGPU_TAR} -C ${_TOY_WGPU_EXTRACT}
        COMMAND ${WASI_SDK_PREFIX}/bin/llvm-ar rcs ${_TOY_WGPU_GEN_LIB}
                ${_TOY_WGPU_EXTRACT}/devc.o ${_TOY_WGPU_EXTRACT}/lib0.o
        DEPENDS ${_TOY_WGPU_TAR}
        COMMENT "Repacking ${_TOY_WGPU_TAR} into libtoy_webgpu_gen.a"
        VERBATIM)
    add_custom_target(toy_webgpu_gen_lib DEPENDS ${_TOY_WGPU_GEN_LIB})

    add_executable(toy_webgpu_test test/toy_webgpu_test.c)
    set_target_properties(toy_webgpu_test PROPERTIES SUFFIX ".wasm")
    add_dependencies(toy_webgpu_test toy_webgpu_gen_lib)
    target_link_libraries(toy_webgpu_test PRIVATE
        -Wl,--whole-archive ${_TOY_WGPU_GEN_LIB}
        tvm-rt-backend-relax-vm
        tvm-rt-accelerator-webgpu
        -Wl,--no-whole-archive
        tvm-rt-core)
endif ()

if (DEFINED USE_WASI_SDK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/decoder_test.c)
    set(_DEC_TAR ${CMAKE_CURRENT_SOURCE_DIR}/test/decoder_out/decoder.tar)
    set(_DEC_EXTRACT ${CMAKE_CURRENT_BINARY_DIR}/decoder_extract)
    set(_DEC_GEN_LIB ${CMAKE_CURRENT_BINARY_DIR}/libdecoder_gen.a)
    set(_DEC_STUB_C ${CMAKE_CURRENT_BINARY_DIR}/decoder_kernels_stub.c)

    add_custom_command(
        OUTPUT ${_DEC_GEN_LIB} ${_DEC_STUB_C}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DEC_EXTRACT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_DEC_EXTRACT}
        COMMAND tar -xf ${_DEC_TAR} -C ${_DEC_EXTRACT}
        COMMAND ${CMAKE_COMMAND} -E rename ${_DEC_EXTRACT}/dec_devc.o ${_DEC_EXTRACT}/devc.o
        COMMAND ${WASI_SDK_PREFIX}/bin/llvm-ar rcs ${_DEC_GEN_LIB}
                ${_DEC_EXTRACT}/devc.o ${_DEC_EXTRACT}/lib0.o
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/gen_kernel_stub.py
                ${_DEC_EXTRACT}/lib0.o ${_DEC_STUB_C}
                ${WASI_SDK_PREFIX}/bin/llvm-nm
                --sys-lib-prefix=dec_ --register-fn=register_all_dec_kernels
        DEPENDS ${_DEC_TAR} ${CMAKE_CURRENT_SOURCE_DIR}/test/gen_kernel_stub.py
        COMMENT "Repacking ${_DEC_TAR} + generating decoder_kernels_stub.c"
        VERBATIM)
    add_custom_target(decoder_gen_lib DEPENDS ${_DEC_GEN_LIB} ${_DEC_STUB_C})

    add_executable(decoder_test test/decoder_test.c ${_DEC_STUB_C})
    set_target_properties(decoder_test PROPERTIES SUFFIX ".wasm")
    add_dependencies(decoder_test decoder_gen_lib)
    target_link_libraries(decoder_test PRIVATE
        -Wl,--whole-archive ${_DEC_GEN_LIB} tvm-rt-backend-relax-vm
        -Wl,--no-whole-archive
        tvm-rt-core
)
endif ()

#------------------------- Milestone 14.7 decoder WebGPU gate -----------------
# Combines the 13.4 decoder shape (~200 auto-generated kernel registrations
# from gen_kernel_stub.py) with the 14.5 toy-webgpu link recipe
# (accelerator-webgpu accessible + no whole-archive for the WebGPU device
# API). tar comes from `test/compile_decoder_webgpu.py` with target
# `Target("webgpu", host=llvm-wasm32-wasi)` and `system_lib_prefix="dec_"`.
# Runs under Node + JSPI via cognition's `web/test/decoder-webgpu.test.mjs`;
# wasmtime is not usable (no WebGPU binding).
if (DEFINED USE_WASI_SDK AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/decoder_webgpu_test.c)
    set(_DEC_WGPU_TAR ${CMAKE_CURRENT_SOURCE_DIR}/test/decoder_webgpu_out/decoder.tar)
    set(_DEC_WGPU_EXTRACT ${CMAKE_CURRENT_BINARY_DIR}/decoder_webgpu_extract)
    set(_DEC_WGPU_GEN_LIB ${CMAKE_CURRENT_BINARY_DIR}/libdecoder_webgpu_gen.a)
    set(_DEC_WGPU_STUB_C ${CMAKE_CURRENT_BINARY_DIR}/decoder_webgpu_kernels_stub.c)

    add_custom_command(
        OUTPUT ${_DEC_WGPU_GEN_LIB} ${_DEC_WGPU_STUB_C}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DEC_WGPU_EXTRACT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_DEC_WGPU_EXTRACT}
        COMMAND tar -xf ${_DEC_WGPU_TAR} -C ${_DEC_WGPU_EXTRACT}
        COMMAND ${CMAKE_COMMAND} -E rename ${_DEC_WGPU_EXTRACT}/dec_devc.o
                ${_DEC_WGPU_EXTRACT}/devc.o
        COMMAND ${WASI_SDK_PREFIX}/bin/llvm-ar rcs ${_DEC_WGPU_GEN_LIB}
                ${_DEC_WGPU_EXTRACT}/devc.o ${_DEC_WGPU_EXTRACT}/lib0.o
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/gen_kernel_stub.py
                ${_DEC_WGPU_EXTRACT}/lib0.o ${_DEC_WGPU_STUB_C}
                ${WASI_SDK_PREFIX}/bin/llvm-nm
                --sys-lib-prefix=dec_ --register-fn=register_all_dec_kernels
        DEPENDS ${_DEC_WGPU_TAR} ${CMAKE_CURRENT_SOURCE_DIR}/test/gen_kernel_stub.py
        COMMENT "Repacking ${_DEC_WGPU_TAR} + generating decoder_webgpu_kernels_stub.c"
        VERBATIM)
    add_custom_target(decoder_webgpu_gen_lib DEPENDS ${_DEC_WGPU_GEN_LIB} ${_DEC_WGPU_STUB_C})

    add_executable(decoder_webgpu_test test/decoder_webgpu_test.c ${_DEC_WGPU_STUB_C})
    set_target_properties(decoder_webgpu_test PROPERTIES SUFFIX ".wasm")
    add_dependencies(decoder_webgpu_test decoder_webgpu_gen_lib)
    target_link_libraries(decoder_webgpu_test PRIVATE
        -Wl,--whole-archive ${_DEC_WGPU_GEN_LIB}
        tvm-rt-backend-relax-vm
        tvm-rt-accelerator-webgpu
        -Wl,--no-whole-archive
        tvm-rt-core)
endif ()
