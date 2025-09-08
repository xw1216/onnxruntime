/*++

Minimal RVV v1.0 softmax helpers for fp32 with intrinsic vectorization.

Implements:
  - MlasReduceMaximumF32KernelRvv
  - MlasComputeSumExpF32KernelRvv
  - MlasComputeSoftmaxOutputF32KernelRvv
  - MlasComputeLogSoftmaxOutputF32KernelRvv

These are wired via MLAS_PLATFORM on riscv64 so generic softmax flow in compute.cpp uses them.

--*/

#if defined(__riscv)
#include <riscv_vector.h>
#endif
#include "mlasi.h"
#include <cmath>
#include <limits>
#include <memory>
#include <cstdint>
#include <cstring>

#if defined(MLAS_TARGET_RISCV64) && defined(__riscv_vector)

// Use canonical __riscv_* intrinsics as provided by Clang's riscv_vector.h

namespace {
// Constants copied to match compute.cpp's MlasExpConstants for alignment
constexpr float kLowerRangeSumExp = -88.3762626647949f;
constexpr float kRoundingBias = 12582912.0f; // MLAS_ROUNDING_BIAS_MAGIC
constexpr float kLog2Reciprocal = 1.44269504088896341f;
constexpr float kLog2High = -6.93145752e-1f;
constexpr float kLog2Low = -1.42860677e-6f;
constexpr float kPoly0 = 0x1.694000p-10f;
constexpr float kPoly1 = 0x1.125edcp-7f;
constexpr float kPoly2 = 0x1.555b5ap-5f;
constexpr float kPoly3 = 0x1.555450p-3f;
constexpr float kPoly4 = 0x1.fffff6p-2f;
constexpr float kPoly56 = 0x1.000000p+0f;
constexpr int32_t kMaximumExponentBits = 0x3F800000; // 2^0 as float bits
}

float
MLASCALL
MlasReduceMaximumF32KernelRvv(
    const float* Input,
    size_t N
)
{
    if (N == 0) return -std::numeric_limits<float>::infinity();

    const float neg_inf = -std::numeric_limits<float>::infinity();
    float max_scalar = neg_inf;

    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(Input + i, vl);

        // 把 NaN 映射为 -inf，这样参与 max 归约时相当于被忽略。
        vbool32_t m_not_nan = __riscv_vmfeq_vv_f32m1_b32(v, v, vl); // x==x 仅对非 NaN 为真
        vfloat32m1_t vneg_inf = __riscv_vfmv_v_f_f32m1(neg_inf, vl);
        vfloat32m1_t vclean = __riscv_vmerge_vvm_f32m1(vneg_inf, v, m_not_nan, vl);

        // 用 1-lane 累加器做向量归约为标量最大值
    size_t vl1 = __riscv_vsetvl_e32m1(1);
        vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(max_scalar, vl1);
        vfloat32m1_t vmax1 = __riscv_vfredmax_vs_f32m1_f32m1(vacc, vclean, vl);
        max_scalar = __riscv_vfmv_f_s_f32m1_f32(vmax1);

        i += vl;
    }
    // 若全为 NaN，max_scalar 保持为 -inf，和通用实现一致。
    return max_scalar;
}

float
MLASCALL
MlasComputeSumExpF32KernelRvv(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
)
{
    // Vectorized polynomial approximation aligned with compute.cpp's SumExp path.
    const float neg_max = *NegativeMaximum;
    float acc = 0.0f;

    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    // Temp buffers to assist with bit-level exponent reconstruction without relying on non-portable reinterprets.
    std::unique_ptr<float[]> tmpf(new float[vlmax]);     // store vbiased
    std::unique_ptr<float[]> tmpscale(new float[vlmax]); // reconstructed 2^m

    const float lower = kLowerRangeSumExp;

    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);

        // x' = max(x + neg_max, LowerRangeSumExp)
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(Input + i, vl);
        vx = __riscv_vfadd_vf_f32m1(vx, neg_max, vl);
        vx = __riscv_vfmax_vf_f32m1(vx, lower, vl);

        // biased = x' * log2(e) + rounding_bias
        vfloat32m1_t vbiased = __riscv_vfmv_v_f_f32m1(kRoundingBias, vl);
        vbiased = __riscv_vfmacc_vf_f32m1(vbiased, kLog2Reciprocal, vx, vl);

        // m = biased - rounding_bias
        vfloat32m1_t vm = __riscv_vfsub_vf_f32m1(vbiased, kRoundingBias, vl);

        // reduced = ((m * Log2High) + x') + (m * Log2Low)
        vfloat32m1_t vred = __riscv_vfmacc_vf_f32m1(vx, kLog2High, vm, vl);
        vred = __riscv_vfmacc_vf_f32m1(vred, kLog2Low, vm, vl);

        // Horner polynomial for exp(reduced). Note the two poly_56 steps per compute.cpp SumExp.
        vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(kPoly0, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly1, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly2, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly3, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly4, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly56, vl);
        // N.B. SumExp path in compute.cpp applies an additional Horner step with poly_56.
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly56, vl);

        // Reconstruct scale factor 2^m using vector integer ops:
        // normal = (ReinterpretAsInt32(biased) << 23) + MaximumExponentBits; scale = ReinterpretAsFloat(normal)
        __riscv_vse32_v_f32m1(tmpf.get(), vbiased, vl);
        vint32m1_t vbits = __riscv_vle32_v_i32m1(reinterpret_cast<const int32_t*>(tmpf.get()), vl);
        vint32m1_t vshift = __riscv_vsll_vx_i32m1(vbits, 23, vl);
        vint32m1_t vmaxexp = __riscv_vadd_vx_i32m1(vshift, (int32_t)kMaximumExponentBits, vl);
        __riscv_vse32_v_i32m1(reinterpret_cast<int32_t*>(tmpscale.get()), vmaxexp, vl);
        vfloat32m1_t vscale = __riscv_vle32_v_f32m1(tmpscale.get(), vl);

        // exp(x') = exp(reduced) * 2^m
        vfloat32m1_t vy = __riscv_vfmul_vv_f32m1(p, vscale, vl);

        // Optionally store outputs.
        if (Output) {
            __riscv_vse32_v_f32m1(Output + i, vy, vl);
        }

        // Vector reduction: create 1-lane accumulator and reduce unordered sum.
    size_t vl1 = __riscv_vsetvl_e32m1(1);
        vfloat32m1_t vacc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl1);
        vfloat32m1_t vsum1 = __riscv_vfredusum_vs_f32m1_f32m1(vacc1, vy, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(vsum1);

        i += vl;
    }
    return acc;
}

void
MLASCALL
MlasComputeSoftmaxOutputF32KernelRvv(
    float* Output,
    size_t N,
    const float* Parameters
)
{
    const float scale = Parameters[0]; // 1/sum
    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);
        vfloat32m1_t vout = __riscv_vle32_v_f32m1(Output + i, vl);
        vout = __riscv_vfmul_vf_f32m1(vout, scale, vl);
        __riscv_vse32_v_f32m1(Output + i, vout, vl);
        i += vl;
    }
}

void
MLASCALL
MlasComputeLogSoftmaxOutputF32KernelRvv(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
)
{
    const float neg_max = Parameters[0];
    const float log_sum = Parameters[1];
    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);
        vfloat32m1_t vin = __riscv_vle32_v_f32m1(Input + i, vl);
        // (Input + neg_max) - log_sum
        vin = __riscv_vfadd_vf_f32m1(vin, neg_max, vl);
        vin = __riscv_vfsub_vf_f32m1(vin, log_sum, vl);
        __riscv_vse32_v_f32m1(Output + i, vin, vl);
        i += vl;
    }
}

#endif // MLAS_TARGET_RISCV64 && __riscv_vector
