/*++

RVV v1.0 activations for fp32 using intrinsics:
    - MlasComputeExpF32KernelRvv
    - MlasLogisticKernelRvv
    - MlasTanhKernelRvv

Mirrors algorithms/constants from compute.cpp, logistic.cpp, tanh.cpp.

--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_RISCV64) && defined(__riscv_vector)

#include <riscv_vector.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {
    // Constants copied from compute.cpp (MlasExpConstants)
    constexpr float kLowerRange = -103.9720840454f;
    constexpr float kUpperRange = 88.7762626647950f;
    constexpr float kRoundingBias = MLAS_ROUNDING_BIAS_MAGIC; // 12582912.0f
    constexpr float kLog2Reciprocal = 1.44269504088896341f;
    constexpr float kLog2High = -6.93145752e-1f;
    constexpr float kLog2Low = -1.42860677e-6f;

    constexpr float kPoly0 = 0x1.694000p-10f;
    constexpr float kPoly1 = 0x1.125edcp-7f;
    constexpr float kPoly2 = 0x1.555b5ap-5f;
    constexpr float kPoly3 = 0x1.555450p-3f;
    constexpr float kPoly4 = 0x1.fffff6p-2f;
    constexpr float kPoly56 = 0x1.000000p+0f;
    constexpr int32_t kMinExpBits = int32_t(0xC1000000);
    constexpr int32_t kMaxExpBits = int32_t(0x3F800000);

    MLAS_FORCEINLINE vfloat32m1_t clamp_non_nan(vfloat32m1_t v, float lo, float hi, size_t vl) {
        // Keep NaNs intact: compute clamped and merge back original where x!=x
        vfloat32m1_t vlo = __riscv_vfmv_v_f_f32m1(lo, vl);
        vfloat32m1_t vhi = __riscv_vfmv_v_f_f32m1(hi, vl);
        vfloat32m1_t vcl = __riscv_vfmin_vv_f32m1(__riscv_vfmax_vv_f32m1(v, vlo, vl), vhi, vl);
        vbool32_t m_not_nan = __riscv_vmfeq_vv_f32m1_b32(v, v, vl);
        // If not NaN (mask true) choose clamped value; if NaN (mask false) keep original
        return __riscv_vmerge_vvm_f32m1(vcl, v, m_not_nan, vl);
    }
} // namespace

void MLASCALL MlasComputeExpF32KernelRvv(const float* Input, float* Output, size_t N) {
    if (N == 0) return;

    // Reuse temporary buffers per call to avoid allocations inside the main loop.
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    std::vector<float> tmpf(vlmax);
    std::vector<int32_t> tmpi(vlmax);

    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);

        // Load and clamp
        vfloat32m1_t x = __riscv_vle32_v_f32m1(Input + i, vl);
        x = clamp_non_nan(x, kLowerRange, kUpperRange, vl);

        // Range reduction: x = (2^m) * exp(reduced)
        vfloat32m1_t vbiased = __riscv_vfmv_v_f_f32m1(kRoundingBias, vl);
        vbiased = __riscv_vfmacc_vf_f32m1(vbiased, kLog2Reciprocal, x, vl);
        vfloat32m1_t vm = __riscv_vfsub_vf_f32m1(vbiased, kRoundingBias, vl);

        vfloat32m1_t vred = __riscv_vfmacc_vf_f32m1(x, kLog2High, vm, vl);
        vred = __riscv_vfmacc_vf_f32m1(vred, kLog2Low, vm, vl);

        // Exponent reconstruction terms
        __riscv_vse32_v_f32m1(tmpf.data(), vbiased, vl);
        vint32m1_t vbits = __riscv_vle32_v_i32m1(reinterpret_cast<const int32_t*>(tmpf.data()), vl);
        vint32m1_t vover = __riscv_vsll_vx_i32m1(vbits, 23, vl);
        vint32m1_t vminexp = __riscv_vmv_v_x_i32m1(kMinExpBits, vl);
        vint32m1_t vmaxexp = __riscv_vmv_v_x_i32m1(kMaxExpBits, vl);
        vint32m1_t vnorm = __riscv_vmin_vv_i32m1(__riscv_vmax_vv_i32m1(vover, vminexp, vl), vmaxexp, vl);
        vover = __riscv_vsub_vv_i32m1(vover, vnorm, vl);
        vover = __riscv_vadd_vv_i32m1(vover, vmaxexp, vl);
        vnorm = __riscv_vadd_vv_i32m1(vnorm, vmaxexp, vl);

        // reinterpret back to float via temporary buffers
        __riscv_vse32_v_i32m1(tmpi.data(), vover, vl);
        vfloat32m1_t vover_f = __riscv_vle32_v_f32m1(reinterpret_cast<const float*>(tmpi.data()), vl);
        __riscv_vse32_v_i32m1(tmpi.data(), vnorm, vl);
        vfloat32m1_t vnorm_f = __riscv_vle32_v_f32m1(reinterpret_cast<const float*>(tmpi.data()), vl);

        // Polynomial approximation for exp(reduced); Horner form
        vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(kPoly0, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly1, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly2, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly3, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly4, vl);
        p = __riscv_vfmul_vv_f32m1(p, vred, vl); p = __riscv_vfadd_vf_f32m1(p, kPoly56, vl);

        // Merge final term with overflow factor: p = p * (vred * vover_f) + vover_f; then scale by vnorm_f
        vfloat32m1_t vec = __riscv_vfmul_vv_f32m1(vred, vover_f, vl);
        p = __riscv_vfmul_vv_f32m1(p, vec, vl);
        p = __riscv_vfadd_vv_f32m1(p, vover_f, vl);
        p = __riscv_vfmul_vv_f32m1(p, vnorm_f, vl);

        __riscv_vse32_v_f32m1(Output + i, p, vl);
        i += vl;
    }
}

void MLASCALL MlasLogisticKernelRvv(const float* Input, float* Output, size_t N) {
    // Constants from logistic.cpp
    constexpr float LowerRange = -18.0f;
    constexpr float UpperRange = 18.0f;
    constexpr float alpha_1 = 2.48287947061529e-01f;
    constexpr float alpha_3 = 8.51377133304701e-03f;
    constexpr float alpha_5 = 6.08574864600143e-05f;
    constexpr float alpha_7 = 1.15627324459942e-07f;
    constexpr float alpha_9 = 4.37031012579801e-11f;

    constexpr float beta_0 = 9.93151921023180e-01f;
    constexpr float beta_2 = 1.16817656904453e-01f;
    constexpr float beta_4 = 1.70198817374094e-03f;
    constexpr float beta_6 = 6.29106785017040e-06f;
    constexpr float beta_8 = 5.76102136993427e-09f;
    constexpr float beta_10 = 6.10247389755681e-13f;

    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(Input + i, vl);
        v = clamp_non_nan(v, LowerRange, UpperRange, vl);
        vfloat32m1_t v2 = __riscv_vfmul_vv_f32m1(v, v, vl);

        // Horner form for numerator p(v) and denominator q(v)
        vfloat32m1_t p = __riscv_vfmul_vv_f32m1(v2, __riscv_vfmv_v_f_f32m1(alpha_9, vl), vl);
        p = __riscv_vfadd_vf_f32m1(p, alpha_7, vl);
        p = __riscv_vfmul_vv_f32m1(p, v2, vl); p = __riscv_vfadd_vf_f32m1(p, alpha_5, vl);
        p = __riscv_vfmul_vv_f32m1(p, v2, vl); p = __riscv_vfadd_vf_f32m1(p, alpha_3, vl);
        p = __riscv_vfmul_vv_f32m1(p, v2, vl); p = __riscv_vfadd_vf_f32m1(p, alpha_1, vl);
        p = __riscv_vfmul_vv_f32m1(p, v, vl);

        vfloat32m1_t q = __riscv_vfmul_vv_f32m1(v2, __riscv_vfmv_v_f_f32m1(beta_10, vl), vl);
        q = __riscv_vfadd_vf_f32m1(q, beta_8, vl);
        q = __riscv_vfmul_vv_f32m1(q, v2, vl); q = __riscv_vfadd_vf_f32m1(q, beta_6, vl);
        q = __riscv_vfmul_vv_f32m1(q, v2, vl); q = __riscv_vfadd_vf_f32m1(q, beta_4, vl);
        q = __riscv_vfmul_vv_f32m1(q, v2, vl); q = __riscv_vfadd_vf_f32m1(q, beta_2, vl);
        q = __riscv_vfmul_vv_f32m1(q, v2, vl); q = __riscv_vfadd_vf_f32m1(q, beta_0, vl);

        vfloat32m1_t y = __riscv_vfdiv_vv_f32m1(p, q, vl);
        y = __riscv_vfadd_vf_f32m1(y, 0.5f, vl);
        __riscv_vse32_v_f32m1(Output + i, y, vl);
        i += vl;
    }
}

void MLASCALL MlasTanhKernelRvv(const float* Input, float* Output, size_t N) {
    // Constants from tanh.cpp
    constexpr float LowerRange = -9.0f;
    constexpr float UpperRange = 9.0f;

    constexpr float a1  = 4.89352455891786e-03f;
    constexpr float a3  = 6.37261928875436e-04f;
    constexpr float a5  = 1.48572235717979e-05f;
    constexpr float a7  = 5.12229709037114e-08f;
    constexpr float a9  = -8.60467152213735e-11f;
    constexpr float a11 = 2.00018790482477e-13f;
    constexpr float a13 = -2.76076847742355e-16f;

    constexpr float b0  = 4.89352518554385e-03f;
    constexpr float b2  = 2.26843463243900e-03f;
    constexpr float b4  = 1.18534705686654e-04f;
    constexpr float b6  = 1.19825839466702e-06f;

    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(Input + i, vl);
        v = clamp_non_nan(v, LowerRange, UpperRange, vl);
        vfloat32m1_t v2 = __riscv_vfmul_vv_f32m1(v, v, vl);

    // For float32, when |x| >= ~8.664339, tanh(x) rounds to ±1.0f.
    // Early saturation improves accuracy and avoids unnecessary computation.
        constexpr float kTanhSat = 8.664339f; // ~atanh(1 - 0.5*2^-23)
        vfloat32m1_t vabs = __riscv_vfabs_v_f32m1(v, vl);
        vbool32_t m_sat = __riscv_vmfgt_vf_f32m1_b32(vabs, kTanhSat, vl);

        // Horner form for tanh numerator p(v) and denominator q(v)
        vfloat32m1_t p = __riscv_vfmul_vv_f32m1(v2, __riscv_vfmv_v_f_f32m1(a13, vl), vl);
        p = __riscv_vfadd_vf_f32m1(p, a11, vl);
        p = __riscv_vfmul_vv_f32m1(p, v2, vl); p = __riscv_vfadd_vf_f32m1(p, a9, vl);
        p = __riscv_vfmul_vv_f32m1(p, v2, vl); p = __riscv_vfadd_vf_f32m1(p, a7, vl);
        p = __riscv_vfmul_vv_f32m1(p, v2, vl); p = __riscv_vfadd_vf_f32m1(p, a5, vl);
        p = __riscv_vfmul_vv_f32m1(p, v2, vl); p = __riscv_vfadd_vf_f32m1(p, a3, vl);
        p = __riscv_vfmul_vv_f32m1(p, v2, vl); p = __riscv_vfadd_vf_f32m1(p, a1, vl);
        p = __riscv_vfmul_vv_f32m1(p, v, vl);

        vfloat32m1_t q = __riscv_vfmul_vv_f32m1(v2, __riscv_vfmv_v_f_f32m1(b6, vl), vl);
        q = __riscv_vfadd_vf_f32m1(q, b4, vl);
        q = __riscv_vfmul_vv_f32m1(q, v2, vl); q = __riscv_vfadd_vf_f32m1(q, b2, vl);
        q = __riscv_vfmul_vv_f32m1(q, v2, vl); q = __riscv_vfadd_vf_f32m1(q, b0, vl);

        vfloat32m1_t y = __riscv_vfdiv_vv_f32m1(p, q, vl);
    // Saturation branch: sign(v) * 1.0f
        vfloat32m1_t vone = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        vbool32_t m_neg = __riscv_vmflt_vf_f32m1_b32(v, 0.0f, vl);
        vfloat32m1_t vsign = __riscv_vmerge_vvm_f32m1(vone, __riscv_vfneg_v_f32m1(vone, vl), m_neg, vl);
        y = __riscv_vmerge_vvm_f32m1(y, vsign, m_sat, vl);
        __riscv_vse32_v_f32m1(Output + i, y, vl);
        i += vl;
    }
}

#endif // MLAS_TARGET_RISCV64 && __riscv_vector
