################################################################################
# ONNX Runtime - OpenHarmony (OHOS) RISC-V 64 Toolchain File (Minimal MVP)
#
# Usage (example):
#   python3 tools/ci_build/build.py \
#     --ohos --ohos_arch riscv64 --ohos_ndk_root /absolute/path/to/tools/ohos_ndk \
#     --build_dir build/ohos_riscv64 --config Release --update --build --skip_tests \
#     --cmake_extra_defines CMAKE_TOOLCHAIN_FILE=${PWD}/cmake/ohos_riscv64.toolchain.cmake onnxruntime_BUILD_SHARED_LIB=ON
#
# Notes:
# - We deliberately set CMAKE_SYSTEM_NAME to Linux so existing __linux__ gated
#   code paths are reused without immediate source modifications.
# - We still define __OHOS__ so that future fine‑grained adjustments are possible.
# - The NDK root is supplied via -DOHOS_NDK_ROOT (injected from build.py) or can
#   be set manually when invoking CMake directly.
################################################################################

if(NOT DEFINED OHOS_NDK_ROOT)
  # Attempt auto-detection relative to repository layout (toolchain file is in <repo>/cmake)
  get_filename_component(_ORT_TC_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  set(OHOS_NDK_ROOT "${_ORT_TC_DIR}/../tools/ohos_ndk" CACHE PATH "Auto-detected OHOS NDK root")
endif()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Layout expectations (already present in the repository under tools/ohos_ndk/):
#   ${OHOS_NDK_ROOT}/llvm/bin/clang
#   ${OHOS_NDK_ROOT}/sysroot/ (usr/include, usr/lib ...)

set(OHOS_LLVM_ROOT "${OHOS_NDK_ROOT}/llvm")
set(CMAKE_SYSROOT   "${OHOS_NDK_ROOT}/sysroot")

if(NOT EXISTS "${OHOS_LLVM_ROOT}/bin/clang")
  message(FATAL_ERROR "OHOS toolchain: clang not found under ${OHOS_LLVM_ROOT}/bin. Set -DOHOS_NDK_ROOT=/abs/path/to/tools/ohos_ndk")
endif()
if(NOT IS_DIRECTORY "${CMAKE_SYSROOT}")
  message(FATAL_ERROR "OHOS toolchain: sysroot not found under ${CMAKE_SYSROOT}")
endif()

# Some OHOS sysroots ship arch headers under asm-riscv without providing asm symlink.
# Create a local symlink (non-fatal if it already exists) so that headers like <asm/types.h> resolve.
if(EXISTS "${CMAKE_SYSROOT}/usr/include/asm-riscv/asm" AND NOT EXISTS "${CMAKE_SYSROOT}/usr/include/asm")
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink asm-riscv/asm "${CMAKE_SYSROOT}/usr/include/asm" COMMAND_ERROR_IS_FATAL ANY)
  message(STATUS "[OHOS] Created symlink: usr/include/asm -> asm-riscv/asm")
endif()

set(_ORT_OHOS_TARGET_TRIPLE "riscv64-linux-ohos")

# Allow user override of march flags (RVV etc.)
set(OHOS_RISCV_MARCH_FLAGS "-march=rv64gc" CACHE STRING "RISC-V march flags for OHOS build")

set(CMAKE_C_COMPILER   "${OHOS_LLVM_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${OHOS_LLVM_ROOT}/bin/clang++")
set(CMAKE_AR           "${OHOS_LLVM_ROOT}/bin/llvm-ar")
set(CMAKE_RANLIB       "${OHOS_LLVM_ROOT}/bin/llvm-ranlib")
set(CMAKE_STRIP        "${OHOS_LLVM_ROOT}/bin/llvm-strip")
set(CMAKE_LINKER       "${OHOS_LLVM_ROOT}/bin/ld.lld")

# Target triple
set(CMAKE_C_COMPILER_TARGET   "${_ORT_OHOS_TARGET_TRIPLE}")
set(CMAKE_CXX_COMPILER_TARGET "${_ORT_OHOS_TARGET_TRIPLE}")

# Base flags
set(_ORT_OHOS_COMMON_FLAGS "${OHOS_RISCV_MARCH_FLAGS} -fPIC")
# Add asm-riscv to include path (it contains directory 'asm/' with types.h)
set(_ORT_OHOS_ASM_INCLUDE "-I${CMAKE_SYSROOT}/usr/include/asm-riscv")

set(CMAKE_C_FLAGS_INIT   "${_ORT_OHOS_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_ORT_OHOS_COMMON_FLAGS}")

# Also append to current flags (in case project resets INIT values)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${_ORT_OHOS_ASM_INCLUDE}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_ORT_OHOS_ASM_INCLUDE}")

# Define OHOS specific macro while preserving Linux code paths
add_compile_definitions(__OHOS__ ONNX_RUNTIME_OHOS)

# Define ORT_RISCV_VECTOR_ENABLED based on march containing 'v'.
set(_OHOS_RISCV_HAS_V FALSE)
if(OHOS_RISCV_MARCH_FLAGS MATCHES "(^|[_,])v([_,]|$)")
  set(_OHOS_RISCV_HAS_V TRUE)
endif()
if(_OHOS_RISCV_HAS_V)
  add_compile_definitions(ORT_RISCV_VECTOR_ENABLED=1)
else()
  add_compile_definitions(ORT_RISCV_VECTOR_ENABLED=0)
endif()

# Ensure the sysroot is used
set(CMAKE_SYSROOT "${CMAKE_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH
    "${CMAKE_SYSROOT}"
    "${OHOS_NDK_ROOT}"
)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

message(STATUS "[OHOS] Target triple: ${_ORT_OHOS_TARGET_TRIPLE}")
message(STATUS "[OHOS] NDK root: ${OHOS_NDK_ROOT}")
message(STATUS "[OHOS] Sysroot : ${CMAKE_SYSROOT}")
message(STATUS "[OHOS] MARCH   : ${OHOS_RISCV_MARCH_FLAGS}")
