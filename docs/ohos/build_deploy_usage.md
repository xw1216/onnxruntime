# ONNX runtime OHOS (riscv64) 构建 / 部署 / 使用指南

适用对象: 希望在 OHOS riscv64 设备上构建、部署并调试 ONNX Runtime 与示例应用 (ResNet18 + YOLO) 的开发者。

---
## 1. 先决条件
- 主机: Linux (已安装 `python3`,`cmake`, `ninja`/`make`, `build-essential`, `gdb-multiarch`)
- OHOS NDK: 放置于 `tools/ohos_ndk` (脚本使用该路径)
- 设备工具: `hdc` 已在 PATH 中并可连接设备 (`hdc shell` 可用)
- 源码分支: 已切换到含移植补丁的分支

---
## 2. 一步构建与安装
推荐使用整合脚本 (包含 runtime + 示例 + 测试复制):
```
./tools/scripts/ohos/build_install_with_tests.sh
```
常用参数:
```
--config Release                # 使用 Release (默认 Debug)
--prefix installed/ort_custom   # 自定义安装前缀
--skip-tests                    # 不复制测试
--only-tests                    # 仅复制测试 (前提: 已安装 runtime + models)
--only-models                   # 仅构建示例 (跳过 runtime)
--jobs 12                       # 指定并行度
--clean-models                  # 清理示例 build 目录后重建
--force-reconfigure             # 强制重新配置 CMake
--build-script-extra "--cmake_extra_defines FOO=ON"
```
手动等价分步 (若不使用脚本):
1. 构建 + 安装 runtime:
   ```shell
   python3 tools/ci_build/build.py --ohos --ohos_arch riscv64 --ohos_ndk_root tools/ohos_ndk --build_dir build/ohos_riscv64 --config Debug --update --build --parallel 8 --cmake_extra_defines onnxruntime_BUILD_SHARED_LIB=ON

   cmake --install build/ohos_riscv64/Debug --prefix installed/ort_ohos_riscv64_debug
   ```
2. 构建 + 安装示例:
   ```shell
   cmake -S samples/ohos/ort_models -B build/ohos_models -DCMAKE_TOOLCHAIN_FILE=cmake/ohos_riscv64.toolchain.cmake -DORT_ROOT=$PWD -DORT_BUILD_DIR=$PWD/build/ohos_riscv64/Debug
   d
   cmake --build build/ohos_models -j 8

   cmake --install build/ohos_models --prefix installed/ort_ohos_riscv64_debug
   ```
3. 复制测试:
   `bash tools/scripts/ohos/install_tests_into_prefix.sh`

---
## 3. 安装前缀结构 (关键目录)
```
installed/ort_ohos_riscv64_debug/
  bin/        # ort_models, 各测试二进制
  lib/        # libonnxruntime*.so, libc++_shared.so
  include/    # 头文件
  testdata/   # 测试所需 onnx / 数据
  assets/     # 示例资源
    models/   # resnet-18.onnx, yolo10s.onnx
    labels/   # imagenet_classes.txt, coco_classes.txt
    images/   # 示例图片
```

---
## 4. 部署到设备
使用部署脚本:
```shell
bash tools/scripts/ohos/deploy_installed_all.sh
```
可选参数:
```
--only-bin-lib      # 仅 bin + lib (额外补上 libc++_shared.so)
--skip-testdata     # 不打包 testdata
--skip-models       # 不打包 models (已置于 assets/models 中通常无需)
--skip-samples      # 跳过 samples 目录 (当前脚本可能无此目录或无需)
--overlay           # 叠加覆盖 (不清空远端目录，加快增量迭代)
```
远端默认根: `/data/local/tmp/ort`
推出打包: `build/ort_sync_stage.tar.gz` (自动生成与传输)

部署后远端结构示例:
```
/data/local/tmp/ort/
  bin/ort_models
  lib/libonnxruntime.so.1.22.2 (及链接)
  assets/{models,labels,images}
  testdata/
```

---
## 5. 运行示例
进入远端 (或使用 gdbserver 前台):
```shell
hdc shell "cd /data/local/tmp/ort/bin && LD_LIBRARY_PATH=/data/local/tmp/ort/lib ./ort_models --model resnet --image ../assets/images/EdlNaRhg9ik.jpg"
```

```shell
hdc shell "cd /data/local/tmp/ort/bin && LD_LIBRARY_PATH=/data/local/tmp/ort/lib ./ort_models --model yolo   --image ../assets/images/EdlNaRhg9ik.jpg --yolo-thresh 0.25 --yolo-nms 0.45"
```
参数说明:
- `--model <resnet|yolo|all>`
- `--image <path>` 指向 assets/images 内文件
- `--labels / --yolo-labels` 可覆盖默认标签路径
- `--asset-dir <dir>` 覆盖自动推断的 `../assets`
- `--yolo-thresh / --yolo-nms` 控制检测过滤

---
## 6. VSCode 任务与调试
主要任务 (见 `.vscode/tasks.json`):
- `ort-ohos-build-runtime` / `ort-ohos-install-runtime-debug`
- `ort-ohos-build-models` / `ort-ohos-install-models-debug`
- `ort-ohos-install-tests-into-prefix`
- `ort-ohos-deploy-installed-all`
- 快速调试：
  - `ort-ohos-quick-start-models-resnet`
  - `ort-ohos-quick-start-models-yolo`
  - `ort-ohos-quick-start-test-all`

调试流程 (ResNet 例):
1. 选择调试配置: `OHOS Ort Model Quick Debug (ResNet)`
2. VSCode 将自动执行：端口转发 → gdbserver 启动 → GDB 附加
3. 远端命令实际等价:
   `hdc shell 'cd /data/local/tmp/ort/bin && LD_LIBRARY_PATH=/data/local/tmp/ort/lib /data/local/tmp/gdbserver :41216 ./ort_models --model resnet --image ../assets/images/EdlNaRhg9ik.jpg'`

测试调试:
- 使用 `OHOS Ort Test All Quick Debug` 启动 `onnxruntime_test_all`
- 若需查看特定失败用例 (示例 RNG): `OHOS Ort Test Fail Quick Debug`
  (内部附加 `--gtest_filter=Random.MultinomialDefaultDType` 与 `ORT_DUMP_ACTUAL=1`)

---
## 7. 常见问题 (FAQ)
| 问题 | 解决方案 |
|------|----------|
| 找不到 `libonnxruntime.so` | 确认部署脚本已运行；运行前设置 `LD_LIBRARY_PATH=/data/local/tmp/ort/lib` |
| YOLO 无检测输出 | 降低 `--yolo-thresh` 或确认输入图片存在且尺寸支持 |
| 资源未找到 | 检查 `assets/` 是否随部署；或使用 `--asset-dir` 手动指定 |
| GDB 断点未命中源码 | 确认 `sourceFileMap` 与 `setupCommands` 中 `directory` 设置正确 |
| 测试随机不一致 | 属平台 RNG 差异；当前通过 `__OHOS__` expected 分支稳定，计划后续引入确定性 RNG |

---
## 8. 性能与优化提示
- Release 构建: `--config Release` 可显著提升推理性能。
- 叠加部署: 使用 `--overlay` 加速增量迭代。
- 精简打包: `--only-bin-lib` 适合仅验证推理二进制的快速循环。
- YOLO 前处理: 当前使用简单拉伸，可引入 letterbox 以改善检测精度 (需额外实现)。

---
## 9. 清理
```
# 清除构建与安装
rm -rf build/ohos_riscv64 build/ohos_models installed/ort_ohos_riscv64_debug build/ort_sync_stage*
```
或使用任务: `ort-ohos-clean`

---
## 10. 后续扩展建议
- 增加量化 / 加速器 Provider (若 OHOS 提供相应后端)
- 引入统一 RNG 抽象消除平台分支
- 增加性能基准聚合输出 (perf test)

---
## 11. 快速参考速查
| 操作 | 命令 |
|------|------|
| 一步构建安装 | `./tools/scripts/ohos/build_install_with_tests.sh` |
| 部署 | `bash tools/scripts/ohos/deploy_installed_all.sh` |
| 运行 ResNet | `hdc shell 'cd /data/local/tmp/ort/bin && LD_LIBRARY_PATH=/data/local/tmp/ort/lib ./ort_models --model resnet --image ../assets/images/EdlNaRhg9ik.jpg'` |
| 运行 YOLO | `hdc shell 'cd /data/local/tmp/ort/bin && LD_LIBRARY_PATH=/data/local/tmp/ort/lib ./ort_models --model yolo --image ../assets/images/EdlNaRhg9ik.jpg'` |
| 启动调试 (VSCode) | 选择 `OHOS Ort Model Quick Debug (YOLO)` |
| 清理 | `ort-ohos-clean` 任务或手动 rm |

---
**完成** — 已可在 OHOS riscv64 设备上构建、部署、调试与运行示例与测试。

> 文档更新时间: 2025-09-04
