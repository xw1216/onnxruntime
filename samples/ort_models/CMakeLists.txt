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
  # 让可执行优先搜索本目录以及上级 lib 目录，避免命中系统同名旧版本
  INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
  BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib"
)

install(TARGETS ort_models RUNTIME DESTINATION bin)
# 不再重复拷贝 libonnxruntime.so, 运行时通过 LD_LIBRARY_PATH 指向 runtime 安装目录的 lib

# 统一资源放入安装前缀根目录下的 assets/ 子目录，以便部署脚本简单打包且与可执行分离
set(ORT_MODELS_ASSET_DIR assets)

# 模型、标签、示例图片
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
