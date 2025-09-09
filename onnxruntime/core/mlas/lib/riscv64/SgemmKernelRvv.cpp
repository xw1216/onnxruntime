/*++

RVV v1.0 SGEMM micro-kernel for fp32.

- Implements both ZeroMode (overwrite C) and Add-mode (accumulate into C),
    using the generic MLAS_GEMM_FLOAT_KERNEL signature with a ZeroMode flag.
- Vectorizes across 4 packed-N columns per block to match the generic packB
    layout used by scalar kernels:
        * For each K, B has 4 consecutive elements for current N-block.
        * When unrolled by 2 over K, the next 4 elements are at +16.

--*/

#include "mlasi.h"
#if defined(MLAS_TARGET_RISCV64) && defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>

// Process up to 2 rows of A and 4 columns of B (packed) per call; return rows handled (1 or 2).
template<bool ZeroMode, bool ProcessTwoRows>
static inline size_t MlasSgemmKernelRvvBlock(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
){
    // We iterate N in blocks of 4 (packed width). For tail N<4, fall back to scalar store path.
    size_t n = CountN;
    const float* a = A;
    float* c_row0 = C;
    float* c_row1 = C + ldc;

    // Accumulators per panel chunk (up to 16 columns to match packB).
    size_t panel_cols = 0; // progress within current 16-wide packed panel
    while (n > 0) {
        // Process up to remaining columns in current 16-wide packed panel.
    size_t maxChunk = (n < (16 - panel_cols)) ? n : (16 - panel_cols);
    // Set VL to the actual hardware-supported length for up to maxChunk elements.
    // Using vsetvl ensures pointer arithmetic matches the true processed lanes.
    size_t vl = __riscv_vsetvl_e32m1(maxChunk);
        vfloat32m1_t acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        const float* b = B; // packed B for these 4 columns
        size_t k = CountK;
        const float* a_it = a;

        // Unroll over K by 2 to match pack stride (+32 for two ks)
        while (k >= 2) {
            float a0_0 = a_it[0];
            float a0_1 = a_it[1];
            vfloat32m1_t vb0 = __riscv_vle32_v_f32m1(b + 0, vl);   // b[0..vl-1]
            acc0 = __riscv_vfmacc_vf_f32m1(acc0, a0_0, vb0, vl);
            if constexpr (ProcessTwoRows) {
                float a1_0 = a_it[lda + 0];
                acc1 = __riscv_vfmacc_vf_f32m1(acc1, a1_0, vb0, vl);
            }
            vfloat32m1_t vb1 = __riscv_vle32_v_f32m1(b + 16, vl); // b[16..16+vl-1]
            acc0 = __riscv_vfmacc_vf_f32m1(acc0, a0_1, vb1, vl);
            if constexpr (ProcessTwoRows) {
                float a1_1 = a_it[lda + 1];
                acc1 = __riscv_vfmacc_vf_f32m1(acc1, a1_1, vb1, vl);
            }
            a_it += 2;
            b += 32;
            k -= 2;
        }

        if (k > 0) {
            float a0 = a_it[0];
            vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + 0, vl);
            acc0 = __riscv_vfmacc_vf_f32m1(acc0, a0, vb, vl);
            if constexpr (ProcessTwoRows) {
                float a1 = a_it[lda + 0];
                acc1 = __riscv_vfmacc_vf_f32m1(acc1, a1, vb, vl);
            }
        }

        // Store: apply alpha and combine with C depending on ZeroMode.
        if constexpr (ZeroMode) {
            // Overwrite: scale accumulators by alpha then store.
            acc0 = __riscv_vfmul_vf_f32m1(acc0, alpha, vl);
            __riscv_vse32_v_f32m1(c_row0, acc0, vl);
            if constexpr (ProcessTwoRows) {
                acc1 = __riscv_vfmul_vf_f32m1(acc1, alpha, vl);
                __riscv_vse32_v_f32m1(c_row1, acc1, vl);
            }
        } else {
            // Add-mode: use fused multiply-add C := C + alpha * acc to reduce rounding error.
            vfloat32m1_t c0 = __riscv_vle32_v_f32m1(c_row0, vl);
            c0 = __riscv_vfmacc_vf_f32m1(c0, alpha, acc0, vl);
            __riscv_vse32_v_f32m1(c_row0, c0, vl);
            if constexpr (ProcessTwoRows) {
                vfloat32m1_t c1 = __riscv_vle32_v_f32m1(c_row1, vl);
                c1 = __riscv_vfmacc_vf_f32m1(c1, alpha, acc1, vl);
                __riscv_vse32_v_f32m1(c_row1, c1, vl);
            }
        }

        // advance to next VL columns of C and packed B panel
        c_row0 += vl;
        if constexpr (ProcessTwoRows) c_row1 += vl;
        B += vl; // next packed group starts +vl
        panel_cols += vl;
        if (panel_cols == 16) {
            B += CountK * 16 - 16; // move to next K slab in packed layout
            panel_cols = 0;
        }
        n -= vl;
    }

    // No scalar tail: vector loop handles any remainder (VL <= 16).

    return ProcessTwoRows ? 2 : 1;
}

// Zero-mode: overwrite C
extern "C" size_t MLASCALL
MlasSgemmKernelZeroRvv(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
)
{
    size_t rows = 0;
    while (CountM > 0) {
        size_t handled;
        if (CountM >= 2) {
            handled = MlasSgemmKernelRvvBlock<true, true>(A, B, C, CountK, CountN, lda, ldc, alpha);
        } else {
            handled = MlasSgemmKernelRvvBlock<true, false>(A, B, C, CountK, CountN, lda, ldc, alpha);
        }
        A += lda * handled;
        C += ldc * handled;
        CountM -= handled;
        rows += handled;
    }
    return rows;
}

// Add-mode: accumulate into C
extern "C" size_t MLASCALL
MlasSgemmKernelAddRvv(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
)
{
    size_t rows = 0;
    while (CountM > 0) {
        size_t handled;
        if (CountM >= 2) {
            handled = MlasSgemmKernelRvvBlock<false, true>(A, B, C, CountK, CountN, lda, ldc, alpha);
        } else {
            handled = MlasSgemmKernelRvvBlock<false, false>(A, B, C, CountK, CountN, lda, ldc, alpha);
        }
        A += lda * handled;
        C += ldc * handled;
        CountM -= handled;
        rows += handled;
    }
    return rows;
}

#endif // MLAS_TARGET_RISCV64 && __riscv_vector
