cmake_minimum_required(VERSION 3.16)
project(ort_models CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ORT_ROOT)
  message(FATAL_ERROR "Please pass -DORT_ROOT pointing to onnxruntime repo root")
endif()

set(ORT_INCLUDE_DIR ${ORT_ROOT}/include)
set(ORT_INCLUDE_SESSION_DIR ${ORT_ROOT}/include/onnxruntime/core/session)
if(NOT DEFINED ORT_BUILD_DIR)
  set(ORT_BUILD_DIR ${ORT_ROOT}/build/ohos_riscv64/Debug)
endif()
set(ORT_LIB ${ORT_BUILD_DIR}/libonnxruntime.so)

add_executable(ort_models main.cpp)

target_include_directories(ort_models PRIVATE ${ORT_INCLUDE_DIR} ${ORT_INCLUDE_SESSION_DIR})

target_link_libraries(ort_models PRIVATE ${ORT_LIB} dl pthread m)

set_target_properties(ort_models PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "\$ORIGIN"
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
