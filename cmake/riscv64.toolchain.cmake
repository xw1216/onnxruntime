# Copyright (c) 2024 SiFive, Inc. All rights reserved.
# Copyright (c) 2024, Phoebe Chen <phoebe.chen@sifive.com>
# Licensed under the MIT License.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_ROOT)

if(NOT RISCV_TOOLCHAIN_ROOT)
  message(FATAL_ERROR "RISCV_TOOLCHAIN_ROOT is not defined. Please set the RISCV_TOOLCHAIN_ROOT variable.")
endif()

# Select compiler suite: prefer clang if requested/available, else fall back to GCC
set(_RISCV_TRIPLE "riscv64-unknown-linux-gnu")

option(RISCV_USE_CLANG "Use clang/clang++ from RISCV_TOOLCHAIN_ROOT for cross-compiling" ON)

if(RISCV_USE_CLANG AND EXISTS "${RISCV_TOOLCHAIN_ROOT}/bin/clang" AND EXISTS "${RISCV_TOOLCHAIN_ROOT}/bin/clang++")
  message(STATUS "[riscv64.toolchain] Using Clang from ${RISCV_TOOLCHAIN_ROOT}/bin")
  set(CMAKE_C_COMPILER   "${RISCV_TOOLCHAIN_ROOT}/bin/clang")
  set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang++")
  set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang")
  # Let CMake pass --target automatically
  set(CMAKE_C_COMPILER_TARGET   "${_RISCV_TRIPLE}")
  set(CMAKE_CXX_COMPILER_TARGET "${_RISCV_TRIPLE}")
  set(CMAKE_ASM_COMPILER_TARGET "${_RISCV_TRIPLE}")
else()
  message(STATUS "[riscv64.toolchain] Using GCC from ${RISCV_TOOLCHAIN_ROOT}/bin")
  set(CMAKE_C_COMPILER   "${RISCV_TOOLCHAIN_ROOT}/bin/${_RISCV_TRIPLE}-gcc")
  set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/${_RISCV_TRIPLE}-gcc")
  set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/${_RISCV_TRIPLE}-g++")
endif()

# ---------------------------------------------------------------------------
# User‑tunable ISA flags (Phase 1 enhancement)
# These allow enabling RVV 1.0 or custom extensions without editing build scripts.
# Examples:
#   -DRISCV_MARCH_FLAGS="-march=rv64gcv_zba_zbb"
# Vector (RVV 1.0) typical: rv64gcv  (toolchain must support the 'v' extension)
# ---------------------------------------------------------------------------
set(RISCV_MARCH_FLAGS "-march=rv64gc" CACHE STRING "RISC-V -march flags (override to enable extensions like rv64gcv)")

# Detect whether RVV is present in march flags
set(_RISCV_HAS_V FALSE)
if(RISCV_MARCH_FLAGS MATCHES "(^|[_,])v([_,]|$)")
  set(_RISCV_HAS_V TRUE)
endif()

# If explicitly enabling RVV and current march lacks 'v', append it.
if(onnxruntime_ENABLE_RISCV_V AND NOT _RISCV_HAS_V)
  string(REGEX REPLACE "^-march=" "" _march_body "${RISCV_MARCH_FLAGS}")
  set(_augmented_march "-march=${_march_body}v")
  message(STATUS "[riscv64.toolchain] Enabling RVV: ${RISCV_MARCH_FLAGS} -> ${_augmented_march}")
  set(RISCV_MARCH_FLAGS "${_augmented_march}" CACHE STRING "RISC-V -march flags (RVV auto-appended)" FORCE)
  set(_RISCV_HAS_V TRUE)
elseif(onnxruntime_ENABLE_RISCV_V AND _RISCV_HAS_V)
  message(STATUS "[riscv64.toolchain] RVV already present in ${RISCV_MARCH_FLAGS}")
endif()

# Expose a simple macro for C/C++ sources indicating whether vector ext is enabled by march
if(_RISCV_HAS_V)
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
## Do not append to CMAKE_C_FLAGS/CMAKE_CXX_FLAGS here to avoid duplicate
## occurrences when CMake seeds flags from INIT and later logic appends again.

set(CMAKE_FIND_ROOT_PATH ${RISCV_TOOLCHAIN_ROOT})
set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")
set(CMAKE_INCLUDE_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/include/")
set(CMAKE_LIBRARY_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/lib/")
set(CMAKE_PROGRAM_PATH "${RISCV_TOOLCHAIN_ROOT}/sysroot/usr/bin/")

# Help CMake propagate sysroot correctly for all tools
set(CMAKE_SYSROOT_COMPILE "${CMAKE_SYSROOT}")
set(CMAKE_SYSROOT_LINK    "${CMAKE_SYSROOT}")

if(RISCV_QEMU_PATH)
  message(STATUS "RISCV_QEMU_PATH=${RISCV_QEMU_PATH} is defined during compilation.")
  set(CMAKE_CROSSCOMPILING_EMULATOR "${RISCV_QEMU_PATH};-L;${CMAKE_SYSROOT}")
endif()

set(CMAKE_CROSSCOMPILING TRUE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
