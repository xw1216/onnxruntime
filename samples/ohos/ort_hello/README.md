# OHOS RISC-V onnxruntime Demo

简单示例：初始化 onnxruntime 环境，尝试加载本地 `model.onnx`（若不存在则提示并结束）。

## 先决条件
- 已按之前步骤完成 OHOS riscv64 交叉构建，得到 `build/ohos_riscv64/Release/libonnxruntime.so`
- Toolchain: `cmake/ohos_riscv64.toolchain.cmake`

## 构建
```bash
# 在仓库根目录执行
cmake -S samples/ohos/ort_hello -B build/ohos_hello \
  -DCMAKE_TOOLCHAIN_FILE=$(pwd)/cmake/ohos_riscv64.toolchain.cmake \
  -DORT_ROOT=$(pwd)
cmake --build build/ohos_hello -j
```
产物：`build/ohos_hello/ort_hello`

## 部署到设备
将以下文件拷贝到设备同一目录（例如 /data/ort_demo/）：
- ort_hello
- libonnxruntime.so (以及对应 .1 真实文件若非直接复制符号链接)
- model.onnx (选配，可用一个最小常量图)

设置运行库路径并执行：
```bash
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
./ort_hello
```

如果未提供 model.onnx，会看到加载失败提示（预期）。

## 添加推理逻辑
可在 `main.c` 里补充：创建输入张量、运行 `Run`、读取输出。后续可加入量化/优化设置。
