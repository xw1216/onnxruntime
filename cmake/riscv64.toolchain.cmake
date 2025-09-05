# Copyright (c) 2024 SiFive, Inc. All rights reserved.
# Copyright (c) 2024, Phoebe Chen <phoebe.chen@sifive.com>
# Licensed under the MIT License.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_ROOT)

if(NOT RISCV_TOOLCHAIN_ROOT)
  message(FATAL_ERROR "RISCV_TOOLCHAIN_ROOT is not defined. Please set the RISCV_TOOLCHAIN_ROOT variable.")
endif()

set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-g++")

# ---------------------------------------------------------------------------
# User‑tunable ISA flags (Phase 1 enhancement)
# These allow enabling RVV 1.0 or custom extensions without editing build scripts.
# Examples:
#   -DRISCV_MARCH_FLAGS="-march=rv64gcv_zba_zbb"
# Vector (RVV 1.0) typical: rv64gcv  (toolchain must support the 'v' extension)
# ---------------------------------------------------------------------------
set(RISCV_MARCH_FLAGS "-march=rv64gc" CACHE STRING "RISC-V -march flags (override to enable extensions like rv64gcv)")

# If the build explicitly enables RVV (onnxruntime_ENABLE_RISCV_V) and current march lacks 'v', append it.
if(onnxruntime_ENABLE_RISCV_V)
  if(NOT RISCV_MARCH_FLAGS MATCHES "v")
    string(REGEX REPLACE "^-march=" "" _march_body "${RISCV_MARCH_FLAGS}")
    set(_augmented_march "-march=${_march_body}v")
    message(STATUS "[riscv64.toolchain] Enabling RVV: ${RISCV_MARCH_FLAGS} -> ${_augmented_march}")
    set(RISCV_MARCH_FLAGS "${_augmented_march}" CACHE STRING "RISC-V -march flags (RVV auto-appended)" FORCE)
  else()
    message(STATUS "[riscv64.toolchain] RVV already present in ${RISCV_MARCH_FLAGS}")
  endif()
  add_compile_definitions(ORT_RISCV_VECTOR_ENABLED=1)
else()
  add_compile_definitions(ORT_RISCV_VECTOR_ENABLED=0)
endif()

# ABI is fixed to lp64d for this baseline; adjust here if a future need arises.
set(_RISCV_COMMON_FLAGS "${RISCV_MARCH_FLAGS} -mabi=lp64d -fPIC")

# Initialise and also append (in case project logic overrides INIT later)
set(CMAKE_C_FLAGS_INIT   "${_RISCV_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_RISCV_COMMON_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${RISCV_MARCH_FLAGS} -mabi=lp64d")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${_RISCV_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_RISCV_COMMON_FLAGS}")

set(CMAKE_FIND_ROOT_PATH ${RISCV_TOOLCHAIN_ROOT})
set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")
set(CMAKE_INCLUDE_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/include/")
set(CMAKE_LIBRARY_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/lib/")
set(CMAKE_PROGRAM_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/bin/")

if(RISCV_QEMU_PATH)
  message(STATUS "RISCV_QEMU_PATH=${RISCV_QEMU_PATH} is defined during compilation.")
  set(CMAKE_CROSSCOMPILING_EMULATOR "${RISCV_QEMU_PATH};-L;${CMAKE_SYSROOT}")
endif()

set(CMAKE_CROSSCOMPILING TRUE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
