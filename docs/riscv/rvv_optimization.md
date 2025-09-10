# ONNX Runtime RISC-V RVV 算子优化总结与使用指南

> 适用对象：在 RISC-V64 平台（支持 RVV 1.0）上构建、部署、验证 ONNX Runtime（ORT）与 MLAS 内核的开发者与性能调优工程师。

本文汇总本分支完成的 RVV 向量化工作：如何启用/关闭 RVV、如何运行回归与微基准、内核设计与实现要点、数值与性能验证方法、已知限制与后续方向。

- 相关参考文档：
  - `docs/riscv/baseline_riscv64_native.md`（原生/交叉构建基线）
  - `docs/riscv/rvv_toggle.md`（编译期 RVV 总开关说明）
  - OHOS 相关：`docs/ohos/build_deploy_usage.md`、`docs/ohos/port_ohos_riscv64.md`

---

## 1. 启用方式与构建开关

- 有效启用条件（满足其一）：
  - `-march` 中包含 `v`（如 `rv64gcv`、`rv64gcv_zba_zbb_zbs_zbc`）。
  - 通过 CMake/脚本开关显式启用 RVV，且将 `v` 追加到 `-march`。

- 相关变量与宏：
  - CMake 变量：`RISCV_MARCH_FLAGS`（Linux）、`OHOS_RISCV_MARCH_FLAGS`（OHOS）。
  - 显式开关：`onnxruntime_ENABLE_RISCV_V`（见 `docs/riscv/rvv_toggle.md`）。
  - 预定义宏：`ORT_RISCV_VECTOR_ENABLED=1|0`（用于条件编译）。

- CMake 集成要点（精简版）：
  - `cmake/riscv64.toolchain.cmake`、`cmake/ohos_riscv64.toolchain.cmake` 负责识别并设置 `-march`，并据此定义 `ORT_RISCV_VECTOR_ENABLED`。
  - `cmake/onnxruntime_mlas.cmake` 仅在检测到 `-march` 含 `v` 时编译 RVV 相关 bench（避免 RVV 关闭时产生无用告警/错误）。

---

## 2. 程序与基准使用

### 2.1 MLAS RVV 基准（bench）

已提供 3 个面向 RVV 的回归与微基准工具：
- Softmax/LogSoftmax：`mlas_rvv_softmax_bench`
- SGEMM（fp32）：`mlas_rvv_sgemm_bench`
- 激活函数（Exp/Logistic/Tanh，fp32）：`mlas_rvv_activations_bench`

特性：
- 若 RVV 未启用，bench 会提示“RVV not enabled / (rvv n/a)”并回退到标准实现，仍可完成正确性检查。
- 激活基准支持 `--size`、`--iters`、`--seed`，并新增 `--topk` 控制差异打印数量（也支持位置参数与 `--topk 8` 形式）。

示例（仅示意，参数可按需调整）：
```bash
# Softmax/LogSoftmax 回归 + 微基准
./mlas_rvv_softmax_bench --sizes=128,1024,4096 --iters=200 --seed=12138

# SGEMM 回归 + 微基准
./mlas_rvv_sgemm_bench --shapes=64x64x64,128x256x64,255x127x63 --iters=8 --seed=55533

# 激活函数回归 + 微基准（打印 top-8 最大差异）
./mlas_rvv_activations_bench --size=1048576 --iters=200 --seed=42 --topk=8
```

输出包含：
- 正确性：最大绝对误差（max abs diff），以及 top-K 差异条目（索引、标准值、RVV 值、绝对差）。
- 吞吐：标准实现（mlas）与 RVV 实现（rvv）的平均耗时与加速比。

### 2.2 示例程序（samples/ort_models）

- 已默认开启推理计时，支持 `--warmup` 与 `--repeat` 控制热身与计次；ResNet18 与 YOLO 样例会打印最小/平均/最大延迟。
- 标准用法（示例）：
```bash
./ort_models --model resnet --image ../assets/images/EdlNaRhg9ik.jpg
./ort_models --model yolo   --image ../assets/images/EdlNaRhg9ik.jpg --yolo-thresh 0.25 --yolo-nms 0.45
```

---

## 3. RVV 算子实现详解

RVV 内核均位于 `onnxruntime/core/mlas/lib/riscv64/`，通过 `GetMlasPlatform()` 钩入 MLAS 平台分发（platform dispatch）或在核心路径显式选择。

### 3.1 激活函数（Exp / Logistic / Tanh，fp32）
- 文件：`activations_rvv.cpp`
- 导出符号：
  - `MlasComputeExpF32KernelRvv`
  - `MlasLogisticKernelRvv`
  - `MlasTanhKernelRvv`
- 平台接入：`platform.cpp` 在检测到 RVV 后将 `ComputeExpF32Kernel`、`LogisticKernelRoutine`、`TanhKernelRoutine` 指向上述 RVV 版本。
- 向量化要点：
  - 使用 RVV intrinsics（`vsetvl` 动态向量长度）循环处理，剩余元素通过尾掩码（mask）安全收尾。
  - `exp/logistic/tanh` 的标量近似/多项式/分段实现被矢量化，确保在大范围输入上保持与 MLAS 标准路径一致的数值稳定性。
  - 正确性工具：`mlas_rvv_activations_bench` 对比 `mlas` 与 `rvv` 输出并打印 top-K 差异，便于定位极值输入下的偏差。

计算思路（概述）：
- Exp：
  - 对输入向量 x 做分段/范围裁剪，避免溢出/欠流；
  - 通过常规的 exp 近似（与标量路径一致的多项式/表驱动策略）计算 e^x；
  - 使用 RVV 向量 FMA/加法进行批量计算，尾部用掩码收尾。
- Logistic：sigmoid(x) = 1 / (1 + exp(-x))
  - 先向量化计算 -x，再调用向量化 exp；
  - 计算 1/(1+e^{-x})，以 FMA 组合减少中间误差；
  - 对极大正/负 x 做数值钳位，保持与标量实现一致的稳定性。
- Tanh：tanh(x) = 2*sigmoid(2x) - 1 或基于多项式/有理逼近
  - 以与标量实现一致的公式与分段策略进行；
  - 保持偶/奇函数性质与饱和区间的精度一致；
  - 全流程向量化，掩码收尾。

### 3.2 Softmax/LogSoftmax（fp32）辅助内核
- 文件：`softmax_kernel_rvv.cpp`
- 导出符号：
  - `MlasReduceMaximumF32KernelRvv`（逐行最大值）
  - `MlasComputeSumExpF32KernelRvv`（减去行最大值后的 `exp` 与求和，带可选输出缓存）
  - `MlasComputeSoftmaxOutputF32KernelRvv`
  - `MlasComputeLogSoftmaxOutputF32KernelRvv`
- 平台接入：`platform.cpp` 在 RVV 有效时优先选用上述 helper；上层 softmax 计算流程与 x86/ARM 路径保持一致：
  1) ReduceMax 得到 `max`；2) 计算 `neg_max = -max`，将输入平移至稳定区间；3) `sum = sum(exp(x + neg_max))`；4) `output = exp(x + neg_max) / sum` 或 `log(output)`。
- 数值稳定性：通过“减去最大值 + `logsumexp`”管线保持与标量实现等价的稳定性。

计算思路（概述）：
- 典型一行 D 维 softmax：
  1) reduce_max：max = max(x)
  2) 平移：y = x - max（逐元素）
  3) sumexp：s = sum(exp(y))（可以同时缓存 exp(y) 以重复使用）
  4) softmax：out = exp(y) / s
- LogSoftmax：
  - 先做 reduce_max 与 sumexp，得到 logsumexp = log(s) + max；
  - 输出 out_log = x - logsumexp；
- RVV 化：reduce、exp、sum、归一化/取对数均以向量循环实现，Kahan/顺序对齐于标量路径的既有策略，保证数值一致性；尾部掩码安全处理。

### 3.3 SGEMM（fp32）
- 文件：`SgemmKernelRvv.cpp`（微内核与块处理），在 `sgemm.cpp` 中选择：
  - 当 RVV 有效时，`sgemm.cpp` 显式调用 `MlasSgemmKernelZeroRvv`/`MlasSgemmKernelAddRvv`（不经由平台函数指针），以契合 MLAS 现有接口差异。
- 接口与分工：
  - `MlasSgemmKernelZeroRvv`：`C = alpha * A * B`（写零模式，等价标量 `ZeroMode` 分支）。
  - `MlasSgemmKernelAddRvv` ：`C += alpha * A * B`（累加模式，等价标量 `AddMode` 分支）。
- 向量化与分块策略：
  - 采用 `vsetvl` 自适应 VL，行（M）× 列（N）分块，内层沿 K 聚合累积。
  - 对齐与尾部：使用掩码处理不足一向量的尾元素，保证无越界读写。
  - 累加通道与缩放：支持 `alpha` 缩放，Zero/Add 两个入口减少分支开销。

计算思路（概述）：
- 目标：计算 C ← α·A·B（或 C ← C + α·A·B）。
- 外层分块：对 N 方向做向量友好的列分块，对 M 方向做若干行块；
- 内层主循环（沿 K）：
  - 按 VL 加载 A 的一段和 B 的对应列段，执行向量 FMA 累加；
  - 通过 `vsetvl` 处理最后一段不足 VL 的尾部，使用掩码保证存取安全；
- 写回：
  - Zero 路径直接写 C；Add 路径在写前先加载 C 做累加；
- 缓存友好：
  - 尽量保持 A 连续访问、B 以列块复用；
  - 在 VL、块尺寸与平台 VLEN/缓存大小之间折中，避免频繁 cache miss；
- 数值与一致性：遵循 MLAS 既有缩放与累加顺序，保持与标量路径的数值可比性。

---

## 4. 正确性与回退策略

- 编译期回退：未检测到 `v` 时不编译 RVV bench，运行基准时会标注“RVV not enabled / (rvv n/a)”，并统一使用 MLAS 标准实现。
- 运行期分发：
  - 激活/Softmax：通过 `GetMlasPlatform()` 获取函数指针；若 RVV 不可用则保持默认标量路径。
  - SGEMM：在 `sgemm.cpp` 中使用 RVV 专用入口；无 RVV 时走原标量/已有后端。
- 回归工具：三类 bench 均提供“标准 vs RVV”的输出对比与耗时数据，便于快速发现偏差。

---

## 5. 性能测量与建议

- 构建类型：建议 `RelWithDebInfo` 或 `Release` 进行性能评测；Debug 仅用于功能验证与调试。
- 预热与重复：
  - `samples/ort_models`：默认启用计时，支持 `--warmup` 与 `--repeat`；
  - bench：均支持 `--iters`，在小问题规模时建议增大迭代数以平滑噪声。
- 绑定与频率：必要时固定 CPU 频率、关闭后台负载，避免测量抖动。
- 结果记录：建议按尺寸/形状维度聚合输出，记录均值与分位点，并保存随机种子以便复现。

> 注：本文不附带具体分数，以免在不同硬件/编译器版本上造成误导。请使用上述基准在目标板上自行复现。

---

## 6. 已知限制与后续方向

- 覆盖范围：当前提供 fp32 的激活、softmax 辅助、SGEMM RVV 内核；后续可扩展至更多算子（Conv/Depthwise、LayerNorm、GEMV/QGEMM 等）。
- 精度与格式：可进一步探索 fp16/bf16、更高阶近似或查表混合方案，以提升吞吐同时保证误差上界。
- 调度优化：
  - SGEMM 更丰富的分块/预取策略与 Cache 友好性；
  - 多线程分片（结合 MLAS 线程池）与 NUMA 感知；
  - 针对不同 VLEN 的自适应调参与内核变体选择。
- 运行期探测：未来可考虑在运行期探测 V 扩展并多版本分发（ABI 兼容前提下）。

---

## 7. 代码索引（便于二次开发）

- RVV 内核：
  - 激活：`onnxruntime/core/mlas/lib/riscv64/activations_rvv.cpp`
  - Softmax helper：`onnxruntime/core/mlas/lib/riscv64/softmax_kernel_rvv.cpp`
  - SGEMM：`onnxruntime/core/mlas/lib/riscv64/SgemmKernelRvv.cpp`
- 平台分发与选择：
  - `onnxruntime/core/mlas/lib/platform.cpp`（RVV 有效时挂接激活/softmax helper）
  - `onnxruntime/core/mlas/lib/sgemm.cpp`（RVV 分支显式调用 RVV SGEMM 入口）
  - 接口声明：`onnxruntime/core/mlas/lib/mlasi.h`
- CMake 构建：
  - `cmake/riscv64.toolchain.cmake`、`cmake/ohos_riscv64.toolchain.cmake`
  - `cmake/onnxruntime_mlas.cmake`（仅在 RVV 有效时编译 bench/源）
- 基准工具：
  - `onnxruntime/core/mlas/tools/rvv_softmax_bench.cpp`
  - `onnxruntime/core/mlas/tools/rvv_sgemm_bench.cpp`
  - `onnxruntime/core/mlas/tools/rvv_activations_bench.cpp`（支持 `--topk`）

---
## 8. 常见问题 (FAQ)
| 问题 | 处理 |
|------|------|
| 编译器不识别 `rv64gcv` | 工具链版本过旧，需升级支持 RVV1.0 的 GCC/Clang |
| 链接阶段找不到 libstdc++ | 确认 `RISCV_TOOLCHAIN_ROOT/sysroot/usr/lib` 在链接器默认搜索路径或手动加 `-L` |
| qemu 运行段错误 | 确认 qemu 版本支持所用的 ISA 扩展；若使用 RVV 标志，需 qemu 新版本 |
| 开启了 RVV 但 bench 没有生成 | CMake 在未检测到 `v` 时会跳过 RVV bench；检查 `--riscv_march`/`--enable_riscv_v` 是否生效，或查看构建日志中的 RVV 检测信息 |
| bench 显示回退 | 表示当前构建/运行环境未启用 RVV 或平台不支持，将自动回退标量路径 |
| 性能不达预期 | 仅部分算子已有 RVV 优化，且对尺寸较小的输入，向量化收益可能不明显；请参考基准工具参数与“计算思路”说明 |

---

## 9. 展望

如需扩展内核或接入新的 RVV 算子，建议沿用：
- 标量参考实现 + RVV 内核并行开发；
- 基于 bench 的“正确性（max diff + top-K）+ 吞吐（avg ms）”双验证；
- CMake 精准门控与平台分发清晰解耦；
- 以可读性、可维护性为先的内核结构（自适应 VL、统一尾部处理、清晰的 Zero/Add 入口）。

> 文档更新时间: 2025-09-10
