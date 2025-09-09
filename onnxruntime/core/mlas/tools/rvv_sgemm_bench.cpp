// Minimal regression and micro-benchmark tool for fp32 SGEMM against MLAS.
// It compares MLAS single-threaded SGEMM (which uses the platform-dispatched
// kernel, e.g., RVV on RISCV64) with a naive reference implementation, and
// benchmarks the MLAS path.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "mlasi.h"

// Contract
// Inputs: A[MxK], B[KxN], C[MxN]; NoTrans*NoTrans; alpha, beta.
// Output: C := alpha * A*B + beta*C

struct Metrics {
  float max_abs_err{0};
  float max_rel_err{0};
};

static Metrics compare_arrays(const float* a, const float* b, size_t n) {
  Metrics m;
  for (size_t i = 0; i < n; ++i) {
    float va = a[i], vb = b[i];
    float ae = std::fabs(va - vb);
    float denom = std::max(std::fabs(va), std::fabs(vb));
    float re = denom > 1e-7f ? (ae / denom) : ae;
    if (ae > m.max_abs_err) m.max_abs_err = ae;
    if (re > m.max_rel_err) m.max_rel_err = re;
  }
  return m;
}

static void fill_matrix(std::vector<float>& m, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& v : m) v = dist(rng);
}

static void naive_gemm(size_t M, size_t N, size_t K,
                       float alpha, const float* A, size_t lda,
                       const float* B, size_t ldb,
                       float beta, float* C, size_t ldc) {
  // Row-major, NoTrans * NoTrans
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      const float* arow = A + i * lda;
      const float* bcol = B + j; // step by ldb per row in K loop
      for (size_t k = 0; k < K; ++k) {
        acc += arow[k] * bcol[k * ldb];
      }
      C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
    }
  }
}

struct Options {
  // shapes as triples MxNxK
  std::vector<std::tuple<size_t,size_t,size_t>> shapes{{64,64,64},{128,128,128},{256,256,256},{128,256,64},{255,127,63}};
  int iters{100};
  uint32_t seed{20240908u};
  float alpha{1.0f};
  float beta{0.0f};
  // simplified behavior: always do reference check; always dump when all dims <= 16; always time naive & MLAS
};

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
  if (std::strncmp(argv[i], "--iters=", 8) == 0) opt.iters = std::max(1, std::atoi(argv[i] + 8));
    else if (std::strncmp(argv[i], "--seed=", 7) == 0) opt.seed = static_cast<uint32_t>(std::strtoul(argv[i] + 7, nullptr, 10));
    else if (std::strncmp(argv[i], "--alpha=", 8) == 0) opt.alpha = std::strtof(argv[i] + 8, nullptr);
    else if (std::strncmp(argv[i], "--beta=", 7) == 0) opt.beta = std::strtof(argv[i] + 7, nullptr);
    else if (std::strncmp(argv[i], "--shapes=", 9) == 0) {
      opt.shapes.clear();
      const char* p = argv[i] + 9;
      while (*p) {
        size_t M = std::strtoul(p, nullptr, 10);
        const char* x1 = std::strchr(p, 'x'); if (!x1) break; p = x1 + 1;
        size_t N = std::strtoul(p, nullptr, 10);
        const char* x2 = std::strchr(p, 'x'); if (!x2) break; p = x2 + 1;
        size_t K = std::strtoul(p, nullptr, 10);
        opt.shapes.emplace_back(M,N,K);
        const char* c = std::strchr(p, ',');
        if (!c) break; else p = c + 1;
      }
    }
  }
  return opt;
}

static void print_matrix_pretty(const char* name, const float* data, size_t rows, size_t cols) {
  std::printf("%s (%zux%zu)\n", name, rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    std::printf("row %3zu:", i);
    for (size_t j = 0; j < cols; ++j) {
      std::printf(" % .9e", data[i * cols + j]);
    }
    std::printf("\n");
  }
}

static void print_diffs_topk(const float* ref, const float* got, size_t rows, size_t cols, int k) {
  struct Diff { size_t i, j; float r, g, d; };
  std::vector<Diff> diffs;
  diffs.reserve(rows * cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      float r = ref[i * cols + j];
      float g = got[i * cols + j];
      float d = std::fabs(r - g);
      if (d > 0.0f) diffs.push_back({i, j, r, g, d});
    }
  }
  std::sort(diffs.begin(), diffs.end(), [](const Diff& a, const Diff& b){ return a.d > b.d; });
  int count = std::min<int>(k, static_cast<int>(diffs.size()));
  if (count > 0) {
    std::printf("top-%d diffs (i,j, ref, got, absdiff):\n", count);
    for (int t = 0; t < count; ++t) {
      auto& e = diffs[t];
      std::printf("(%zu,%zu) %.9e %.9e %.9e\n", e.i, e.j, e.r, e.g, e.d);
    }
  } else {
    std::printf("no element-wise diffs.\n");
  }
}

static void run_case(size_t M, size_t N, size_t K, const Options& opt) {
  std::vector<float> A(M*K), B(K*N), C_ref(M*N), C_mlas(M*N);
  fill_matrix(A, opt.seed + 17u, -3.0f, 3.0f);
  fill_matrix(B, opt.seed + 23u, -3.0f, 3.0f);
  fill_matrix(C_ref, opt.seed + 29u, -1.0f, 1.0f);
  std::vector<float> C_init = C_ref; // preserve initial C0 for beta handling in benchmarks

  // Reference
  naive_gemm(M, N, K, opt.alpha, A.data(), K, B.data(), N, opt.beta, C_ref.data(), N);

  // MLAS single-threaded GEMM
  C_mlas = C_init; // start from same initial C0
  MlasSgemmOperation(CblasNoTrans, CblasNoTrans, M, N, K,
                     opt.alpha, A.data(), K, B.data(), N,
                     opt.beta, C_mlas.data(), N);

  {
    Metrics m = compare_arrays(C_ref.data(), C_mlas.data(), M*N);
  std::printf("[zero-mode] M=%zu N=%zu K=%zu  max_abs_err=%.9e  max_rel_err=%.9e\n",
                M, N, K, m.max_abs_err, m.max_rel_err);

    // Always dump for small shapes: only if all dims <= 16
    const bool small_enough = (M <= 16 && N <= 16 && K <= 16);
    if (small_enough) {
      print_matrix_pretty("A", A.data(), M, K);
      print_matrix_pretty("B", B.data(), K, N);
      // Rename groups to align with softmax bench: ref->mlas, mlas->rvv
      print_matrix_pretty("C_mlas", C_ref.data(), M, N);
      print_matrix_pretty("C_rvv", C_mlas.data(), M, N);
      // show top-16 diffs
      print_diffs_topk(C_ref.data(), C_mlas.data(), M, N, 16);
      std::printf("\n");
    }
  }

  // Always run an extra regression with a non-zero beta to exercise add-mode
  {
    const float alpha2 = opt.alpha * 0.5f + 0.25f;
    const float beta2 = (opt.beta == 0.0f ? 0.75f : opt.beta * 1.1f + 0.05f);
    std::vector<float> C_ref2 = C_init;
    std::vector<float> C_mlas2 = C_init;
    naive_gemm(M, N, K, alpha2, A.data(), K, B.data(), N, beta2, C_ref2.data(), N);
    MlasSgemmOperation(CblasNoTrans, CblasNoTrans, M, N, K,
                       alpha2, A.data(), K, B.data(), N,
                       beta2, C_mlas2.data(), N);
    Metrics m2 = compare_arrays(C_ref2.data(), C_mlas2.data(), M*N);
    std::printf("[add-mode] M=%zu N=%zu K=%zu  max_abs_err=%.9e  max_rel_err=%.9e  (alpha=%.3f beta=%.3f)\n",
                M, N, K, m2.max_abs_err, m2.max_rel_err, alpha2, beta2);

    // For small shapes, also dump add-mode results for visual inspection
    const bool small_enough2 = (M <= 16 && N <= 16 && K <= 16);
    if (small_enough2) {
      // A/B are identical to the zero-mode section; only dump C variants here
      print_matrix_pretty("C_mlas_add", C_ref2.data(), M, N);
      print_matrix_pretty("C_rvv_add", C_mlas2.data(), M, N);
      print_diffs_topk(C_ref2.data(), C_mlas2.data(), M, N, 16);
      std::printf("\n");
    }
  }

  // Micro-benchmark MLAS path
  auto bench_mlas = [&]() {
    using clk = std::chrono::steady_clock;
    // warmup
    for (int w = 0; w < 3; ++w) {
      // restore initial C0 before each call when beta != 0
      std::memcpy(C_mlas.data(), C_init.data(), M*N*sizeof(float));
      MlasSgemmOperation(CblasNoTrans, CblasNoTrans, M, N, K,
                         opt.alpha, A.data(), K, B.data(), N,
                         opt.beta, C_mlas.data(), N);
    }
    auto t0 = clk::now();
    for (int it = 0; it < opt.iters; ++it) {
      std::memcpy(C_mlas.data(), C_init.data(), M*N*sizeof(float));
      MlasSgemmOperation(CblasNoTrans, CblasNoTrans, M, N, K,
                         opt.alpha, A.data(), K, B.data(), N,
                         opt.beta, C_mlas.data(), N);
    }
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms;
  };

  // Micro-benchmark naive reference
  auto bench_naive = [&]() {
    using clk = std::chrono::steady_clock;
    std::vector<float> C_tmp(M*N);
    auto t0 = clk::now();
    for (int it = 0; it < opt.iters; ++it) {
      // reset C to initial values C0 for each iteration to keep work consistent with beta
      std::memcpy(C_tmp.data(), C_init.data(), M*N*sizeof(float));
      naive_gemm(M, N, K, opt.alpha, A.data(), K, B.data(), N, opt.beta, C_tmp.data(), N);
    }
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms;
  };

  double t_rvv = bench_mlas();
  double t_mlas = bench_naive();
  double speedup = t_mlas / std::max(1e-9, t_rvv);
  std::printf("bench M=%zu N=%zu K=%zu  iters=%d  mlas=%.3f ms  rvv=%.3f ms  speedup=%.2fx\n\n",
              M, N, K, opt.iters, t_mlas, t_rvv, speedup);
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  std::printf("MLAS SGEMM Regression & Micro-benchmark (NoTrans x NoTrans)\n");
  std::printf("shapes=");
  for (size_t i = 0; i < opt.shapes.size(); ++i) {
    auto [M,N,K] = opt.shapes[i];
    std::printf("%zux%zux%zu%s", M, N, K, (i + 1 == opt.shapes.size()) ? "" : ",");
  }
  std::printf("  iters=%d  alpha=%.3f  beta=%.3f  seed=%u\n\n",
              opt.iters, opt.alpha, opt.beta, opt.seed);

  for (const auto& shp : opt.shapes) {
    auto [M,N,K] = shp;
    run_case(M, N, K, opt);
  }
  return 0;
}
