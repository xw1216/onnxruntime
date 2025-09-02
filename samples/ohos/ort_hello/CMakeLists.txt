cmake_minimum_required(VERSION 3.16)
project(ort_hello C)

# Expect user to build inside existing cross environment with toolchain already set.
# Usage (example):
#   cmake -S samples/ohos/ort_hello -B build/ohos_hello \
#     -DCMAKE_TOOLCHAIN_FILE=${CMAKE_SOURCE_DIR}/cmake/ohos_riscv64.toolchain.cmake \
#     -DORT_ROOT=${CMAKE_SOURCE_DIR}
#   cmake --build build/ohos_hello -j

if(NOT DEFINED ORT_ROOT)
  message(FATAL_ERROR "Please pass -DORT_ROOT pointing to onnxruntime repo root")
endif()

set(ORT_INCLUDE_DIR ${ORT_ROOT}/include)
set(ORT_INCLUDE_SESSION_DIR ${ORT_ROOT}/include/onnxruntime/core/session)
# Built artifacts (user may adjust build dir)
if(NOT DEFINED ORT_BUILD_DIR)
  set(ORT_BUILD_DIR ${ORT_ROOT}/build/ohos_riscv64/Release)
endif()

set(ORT_LIB ${ORT_BUILD_DIR}/libonnxruntime.so)

add_executable(ort_hello main.c)

target_include_directories(ort_hello PRIVATE ${ORT_INCLUDE_DIR} ${ORT_INCLUDE_SESSION_DIR})

target_link_libraries(ort_hello PRIVATE ${ORT_LIB} dl pthread m)

# Rpath so the binary can find libonnxruntime.so when deployed together.
set_target_properties(ort_hello PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "\$ORIGIN"
)

install(TARGETS ort_hello RUNTIME DESTINATION bin)
install(FILES ${ORT_LIB} DESTINATION bin)
