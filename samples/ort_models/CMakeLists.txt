cmake_minimum_required(VERSION 3.16)
project(ort_models CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ORT_ROOT)
  message(FATAL_ERROR "Please pass -DORT_ROOT pointing to onnxruntime repo root")
endif()

set(ORT_INCLUDE_DIR ${ORT_ROOT}/include)
set(ORT_INCLUDE_SESSION_DIR ${ORT_ROOT}/include/onnxruntime/core/session)

# Require explicit ORT_BUILD_DIR. Keep this simple per user request (auto-detection removed).
if(NOT DEFINED ORT_BUILD_DIR)
  message(FATAL_ERROR "ORT_BUILD_DIR is required. Example: -DORT_BUILD_DIR=\n  (linux) ${ORT_ROOT}/build/bianbu_riscv64/Debug OR installed/ort_bianbu_rv64_debug\n  (ohos)  ${ORT_ROOT}/build/ohos_riscv64/Debug OR installed/ort_ohos_riscv64_debug")
else()
  message(STATUS "Using ORT_BUILD_DIR=${ORT_BUILD_DIR}")
endif()

# Resolve libonnxruntime.so potentially located directly or under lib/
unset(ORT_LIB)
set(_ORT_LIB_CANDIDATES
  ${ORT_BUILD_DIR}/libonnxruntime.so
  ${ORT_BUILD_DIR}/lib/libonnxruntime.so
)
foreach(_libcand IN LISTS _ORT_LIB_CANDIDATES)
  if(EXISTS ${_libcand})
    set(ORT_LIB ${_libcand})
    break()
  endif()
endforeach()
if(NOT ORT_LIB)
  message(FATAL_ERROR "libonnxruntime.so not found in ORT_BUILD_DIR='${ORT_BUILD_DIR}'. Tried: ${_ORT_LIB_CANDIDATES}")
else()
  message(STATUS "Using ORT_LIB=${ORT_LIB}")
endif()

add_executable(ort_models main.cpp)

target_include_directories(ort_models PRIVATE ${ORT_INCLUDE_DIR} ${ORT_INCLUDE_SESSION_DIR})

target_link_libraries(ort_models PRIVATE ${ORT_LIB} dl pthread m)

set_target_properties(ort_models PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  # Ensure the executable searches its own dir and ../lib first to avoid picking system copies
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
  BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib"
)

install(TARGETS ort_models RUNTIME DESTINATION bin)
# Do not copy libonnxruntime.so again; runtime finds it via LD_LIBRARY_PATH

# Put assets under the install prefix root (assets/) so scripts can package and keep binaries separate
set(ORT_MODELS_ASSET_DIR assets)

# Models, labels, sample images
file(GLOB MODEL_FILES "${CMAKE_SOURCE_DIR}/models/*.onnx")
if(MODEL_FILES)
  install(FILES ${MODEL_FILES} DESTINATION ${ORT_MODELS_ASSET_DIR}/models)
endif()
file(GLOB LABEL_FILES "${CMAKE_SOURCE_DIR}/labels/*.txt")
if(LABEL_FILES)
  install(FILES ${LABEL_FILES} DESTINATION ${ORT_MODELS_ASSET_DIR}/labels)
endif()
file(GLOB IMAGE_FILES "${CMAKE_SOURCE_DIR}/images/*")
if(IMAGE_FILES)
  install(FILES ${IMAGE_FILES} DESTINATION ${ORT_MODELS_ASSET_DIR}/images)
endif()
