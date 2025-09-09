/*
Simple regression + micro-bench for RVV activations (Exp/Logistic/Tanh) vs. generic.
Builds inside MLAS and links to onnxruntime_mlas.
*/

#include <chrono>
#include <random>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>

#include "mlasi.h"

using Clock = std::chrono::high_resolution_clock;

static void fill(std::vector<float>& x, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-20.0f, 20.0f);
  for (auto& v : x) v = dist(rng);
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

struct DiffEntry { size_t idx; float mlas; float rvv; float adiff; };
static void print_topk_diffs(const char* tag,
                             const std::vector<float>& mlas,
                             const std::vector<float>& rvv,
                             int k) {
  std::vector<DiffEntry> diffs;
  diffs.reserve(mlas.size());
  for (size_t i = 0; i < mlas.size(); ++i) {
    float d = std::fabs(mlas[i] - rvv[i]);
    if (d > 0.0f) diffs.push_back(DiffEntry{ i, mlas[i], rvv[i], d });
  }
  if (diffs.empty()) {
    std::printf("%s top-%d diffs: none (outputs identical)\n", tag, k);
    return;
  }
  std::partial_sort(diffs.begin(), diffs.begin() + std::min<int>(k, (int)diffs.size()), diffs.end(),
                    [](const DiffEntry& a, const DiffEntry& b){ return a.adiff > b.adiff; });
  int count = std::min<int>(k, static_cast<int>(diffs.size()));
  std::printf("%s top-%d diffs (idx, mlas, rvv, absdiff):\n", tag, count);
  for (int t = 0; t < count; ++t) {
    const auto& e = diffs[t];
    std::printf("[%zu] % .9e  % .9e  % .9e\n", e.idx, e.mlas, e.rvv, e.adiff);
  }
}

static double bench_kernel_ms(void(*ker)(const float*, float*, size_t),
                              const std::vector<float>& in,
                              std::vector<float>& out,
                              int iters) {
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) ker(in.data(), out.data(), in.size());
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

struct Options {
  size_t size{static_cast<size_t>(1) << 20}; // default: 1M elements
  int iters{200};
  uint32_t seed{42};
  int topk{16};
};

static Options parse_args(int argc, char** argv) {
  Options opt;
  int pos = 0; // legacy positional fallback: N, iters, seed
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strncmp(arg, "--size=", 7) == 0) {
      opt.size = std::strtoull(arg + 7, nullptr, 10);
    } else if (std::strncmp(arg, "--iters=", 8) == 0) {
      opt.iters = std::max(1, std::atoi(arg + 8));
    } else if (std::strncmp(arg, "--seed=", 7) == 0) {
      opt.seed = static_cast<uint32_t>(std::strtoul(arg + 7, nullptr, 10));
    } else if (std::strncmp(arg, "--topk=", 7) == 0) {
      opt.topk = std::max(1, std::atoi(arg + 7));
    } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      std::puts("Usage: mlas_rvv_activations_bench [--size=N] [--iters=I] [--seed=S]\n"
                "                     [--topk=K]\n"
                "       Positional fallback: N I S K");
    } else {
      // Positional fallback for compatibility (numbers only). Unknown options (starting with '-') are ignored.
      if (arg[0] == '-') {
        continue;
      }
      if (pos == 0) opt.size = std::strtoull(arg, nullptr, 10);
      else if (pos == 1) opt.iters = std::max(1, std::atoi(arg));
      else if (pos == 2) opt.seed = static_cast<uint32_t>(std::strtoul(arg, nullptr, 10));
      else if (pos == 3) opt.topk = std::max(1, std::atoi(arg));
      ++pos;
    }
  }
  return opt;
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  const size_t N = opt.size;
  const int iters = opt.iters;
  const uint32_t seed = opt.seed;
  const int topk = opt.topk;

  std::printf("MLAS RVV Activations Regression & Micro-benchmark\n");
  std::printf("size=%zu  iters=%d  seed=%u  topk=%d\n\n", N, iters, seed, topk);
  if (N == 0) {
    std::fprintf(stderr, "[WARN] size=0. Please check arguments. Expected --size=N (or positional N).\n");
  }

  std::vector<float> x(N), y1(N), y2(N);
  fill(x, seed);

  auto& P = GetMlasPlatform();
  // Resolve function pointers: generic fallbacks and current platform
  auto exp_ref = MlasComputeExpF32Kernel; // generic reference
  auto exp_opt = P.ComputeExpF32Kernel ? P.ComputeExpF32Kernel : MlasComputeExpF32Kernel;

  auto log_ref = MlasLogisticKernel;
  auto log_opt = P.LogisticKernelRoutine ? P.LogisticKernelRoutine : MlasLogisticKernel;

  auto tanh_ref = MlasTanhKernel;
  auto tanh_opt = P.TanhKernelRoutine ? P.TanhKernelRoutine : MlasTanhKernel;

  std::puts("== Correctness (max abs diff) ==");
  exp_ref(x.data(), y1.data(), N); exp_opt(x.data(), y2.data(), N);
  std::printf("exp (mlas vs rvv): %.9g\n", max_abs_diff(y1, y2));
  print_topk_diffs("exp", y1, y2, topk);
  log_ref(x.data(), y1.data(), N); log_opt(x.data(), y2.data(), N);
  std::printf("logistic (mlas vs rvv): %.9g\n", max_abs_diff(y1, y2));
  print_topk_diffs("logistic", y1, y2, topk);
  tanh_ref(x.data(), y1.data(), N); tanh_opt(x.data(), y2.data(), N);
  std::printf("tanh (mlas vs rvv): %.9g\n", max_abs_diff(y1, y2));
  print_topk_diffs("tanh", y1, y2, topk);

  std::puts("\n== Throughput (avg ms, lower is better) ==");
  double exp_m = bench_kernel_ms(exp_ref, x, y1, iters);
  double exp_r = bench_kernel_ms(exp_opt, x, y2, iters);
  std::printf("exp-mlas     : %.3f ms\n", exp_m);
  std::printf("exp-rvv      : %.3f ms  speedup=%.2fx\n", exp_r, exp_m / std::max(1e-9, exp_r));

  double log_m = bench_kernel_ms(log_ref, x, y1, iters);
  double log_r = bench_kernel_ms(log_opt, x, y2, iters);
  std::printf("log-mlas     : %.3f ms\n", log_m);
  std::printf("log-rvv      : %.3f ms  speedup=%.2fx\n", log_r, log_m / std::max(1e-9, log_r));

  double tanh_m = bench_kernel_ms(tanh_ref, x, y1, iters);
  double tanh_r = bench_kernel_ms(tanh_opt, x, y2, iters);
  std::printf("tanh-mlas    : %.3f ms\n", tanh_m);
  std::printf("tanh-rvv     : %.3f ms  speedup=%.2fx\n", tanh_r, tanh_m / std::max(1e-9, tanh_r));

  return 0;
}
