# RISC-V64 原生开发板基线构建与冒烟验证 (Phase 2)

> 目标：在 Ubuntu 25.10 riscv64 开发板（含 RVV1.0 硬件，可选）或使用 x86_64 主机交叉编译 + 板端运行，获得功能正确的 baseline，用于后续 RVV 内核优化对比。

---
## 1. 交叉工具链假设
假设已经有官方提供的 x86_64 -> riscv64 交叉工具链，形如：
```
<TC_ROOT>/bin/riscv64-unknown-linux-gnu-gcc
<TC_ROOT>/sysroot/...
```
设定：`RISCV_TOOLCHAIN_ROOT=/abs/path/to/toolchain`。

若需开启 RVV：确认编译器支持 `-march=rv64gcv`，可执行：
```
$RISCV_TOOLCHAIN_ROOT/bin/riscv64-unknown-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -E - < /dev/null >/dev/null && echo OK
```

---
## 2. Phase 1 新增能力回顾（已精简）
- `cmake/riscv64.toolchain.cmake` 暴露 `RISCV_MARCH_FLAGS` (默认 `-march=rv64gc`)，ABI 固定为 `lp64d` 不再单独可配。
- build.py CLI：
  - `--riscv_march`  （不含 `-march=` 前缀，例: `rv64gcv_zba_zbb`）
  - 通过 `-DRISCV_MARCH_FLAGS` 注入编译；`-mabi=lp64d` 由工具链文件固定添加。

---
## 3. 交叉编译步骤 (主机 x86_64)
```
export RISCV_TOOLCHAIN_ROOT=/abs/path/to/toolchain
export RISCV_QEMU_PATH=$RISCV_TOOLCHAIN_ROOT/bin/qemu-riscv64   # 若需要在主机上跑测试

python3 tools/ci_build/build.py \
  --rv64 \
  --riscv_toolchain_root $RISCV_TOOLCHAIN_ROOT \
  --riscv_qemu_path $RISCV_QEMU_PATH \
  --build_dir build/riscv64 \
  --config RelWithDebInfo \
  --update --build \
  --cmake_generator Ninja \
  --parallel 8 \
  --build_shared_lib \
  --riscv_march rv64gc   # 或 rv64gcv 开启向量（当前仅影响编译标志，尚无 RVV 专用内核）
```

生成产物位于：`build/riscv64/RelWithDebInfo`。

---
## 4. 开发板原生构建（如果工具链已在板上）
在板上直接：
```
mkdir -p build/native && cd build/native
cmake ../../cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build . -j$(nproc)
```
（可选）开启 RVV：
```
export CFLAGS="-march=rv64gcv -mabi=lp64d"
export CXXFLAGS="$CFLAGS"
# 或在第一次 cmake 时追加 -DCMAKE_C_FLAGS=... -DCMAKE_CXX_FLAGS=...
```

---
## 5. 冒烟测试（交叉 + qemu 或 板上）
只挑选部分算子路径：
```
cd build/riscv64/RelWithDebInfo
./onnxruntime_test_all --gtest_filter=Gemm*:*MatMul*:*Conv* --gtest_repeat=1 --gtest_color=yes
```
若使用 qemu：
```
$RISCV_QEMU_PATH -L $RISCV_TOOLCHAIN_ROOT/sysroot ./onnxruntime_test_all --gtest_filter=Gemm*:*MatMul*:*Conv*
```

---
## 6. 模型推理快速验证
```
./onnx_test_runner ../models/test_squeezenet  # 需准备一个模型目录
```
或运行你已有的 OHOS 样例改写版（适配到 Linux 路径后）。

---
## 7. 远程部署到开发板 (示例)
```
# 主机上
rsync -avz build/riscv64/RelWithDebInfo/ user@board:/opt/ort/bin
# 板上运行
ssh user@board 'cd /opt/ort/bin && ./onnxruntime_test_all --gtest_filter=Gemm*.Basic*'
```

---
## 8. 后续 (Phase 3+) 提前占位
- `docs/riscv/rvv_plan.md`：将记录 RVV 内核列表与基准对比。
- CMake 将新增 `onnxruntime_ENABLE_RISCV_V` 控制向量化实现编译。

---
## 9. 常见问题 (FAQ)
| 问题 | 处理 |
|------|------|
| 编译器不识别 `rv64gcv` | 工具链版本过旧，需升级支持 RVV1.0 的 GCC/Clang |
| 链接阶段找不到 libstdc++ | 确认 `RISCV_TOOLCHAIN_ROOT/sysroot/usr/lib` 在链接器默认搜索路径或手动加 `-L` |
| qemu 运行段错误 | 确认 qemu 版本支持所用的 ISA 扩展；若使用 RVV 标志，需 qemu 新版本 |
| 性能很慢 | 目前仍是 scalar 实现，等待 RVV 内核 (Phase 4+) |

---
**Baseline 完成判定**：构建成功 + 关键冒烟测试通过 + 至少一个小模型推理成功。
