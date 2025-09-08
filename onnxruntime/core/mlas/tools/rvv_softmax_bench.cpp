// Minimal RVV vs generic MLAS Softmax/LogSoftmax regression and micro-benchmark tool.
// Builds inside onnxruntime and links against onnxruntime_mlas.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mlasi.h"

// Forward declare kernels we need (generic and RVV). These symbols exist in MLAS.
extern "C" {
// Generic kernels
float MLASCALL MlasReduceMaximumF32Kernel(const float* Input, size_t N);
float MLASCALL MlasComputeSumExpF32Kernel(const float* Input, float* Output, size_t N, const float* NegativeMaximum);
void  MLASCALL MlasComputeSoftmaxOutputF32Kernel(float* Output, size_t N, const float* Parameters);
void  MLASCALL MlasComputeLogSoftmaxOutputF32Kernel(const float* Input, float* Output, size_t N, const float* Parameters);

#if defined(__riscv) && defined(__riscv_vector)
// RVV kernels
float MLASCALL MlasReduceMaximumF32KernelRvv(const float* Input, size_t N);
float MLASCALL MlasComputeSumExpF32KernelRvv(const float* Input, float* Output, size_t N, const float* NegativeMaximum);
void  MLASCALL MlasComputeSoftmaxOutputF32KernelRvv(float* Output, size_t N, const float* Parameters);
void  MLASCALL MlasComputeLogSoftmaxOutputF32KernelRvv(const float* Input, float* Output, size_t N, const float* Parameters);
#endif
}

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
    float re = denom > 1e-8f ? (ae / denom) : ae;
    if (ae > m.max_abs_err) m.max_abs_err = ae;
    if (re > m.max_rel_err) m.max_rel_err = re;
  }
  return m;
}

static void fill_data(std::vector<float>& x, bool with_nan, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  for (auto& v : x) v = dist(rng);
  if (with_nan && !x.empty()) {
    // sprinkle a few NaNs
    for (size_t i = 0; i < x.size(); i += (x.size() / 7 + 1)) {
      x[i] = std::numeric_limits<float>::quiet_NaN();
    }
  }
}

struct Options {
  std::vector<size_t> sizes{16, 64, 128, 256, 1024, 4096};
  int iters{200};
  bool with_nan{false};
  bool do_logsoftmax{false};
  uint32_t seed{12345};
};

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--with-nan") == 0) opt.with_nan = true;
    else if (std::strcmp(argv[i], "--logsoftmax") == 0) opt.do_logsoftmax = true;
    else if (std::strncmp(argv[i], "--iters=", 8) == 0) opt.iters = std::max(1, std::atoi(argv[i] + 8));
    else if (std::strncmp(argv[i], "--seed=", 7) == 0) opt.seed = static_cast<uint32_t>(std::strtoul(argv[i] + 7, nullptr, 10));
    else if (std::strncmp(argv[i], "--sizes=", 8) == 0) {
      opt.sizes.clear();
      const char* p = argv[i] + 8;
      while (*p) {
        size_t v = std::strtoul(p, nullptr, 10);
        opt.sizes.push_back(v);
        const char* c = std::strchr(p, ',');
        if (!c) break; else p = c + 1;
      }
    }
  }
  return opt;
}

static void run_case(size_t D, const Options& opt) {
  std::vector<float> input(D), out_std(D), out_rvv(D);
  const size_t kPrintThreshold = 32; // 当尺寸较小时打印详细数值
  float max_std = 0.0f, sum_std = 0.0f;
  float max_rvv = 0.0f, sum_rvv = 0.0f;
  fill_data(input, opt.with_nan, opt.seed + static_cast<uint32_t>(D));

  // Reference (generic) pipeline: Softmax or LogSoftmax
  {
    float maxv = MlasReduceMaximumF32Kernel(input.data(), D);
    float neg_max = -maxv;
    std::vector<float> tmp(D);
    float sum = MlasComputeSumExpF32Kernel(input.data(), opt.do_logsoftmax ? nullptr : tmp.data(), D, &neg_max);
    max_std = maxv;
    sum_std = sum;
    if (!opt.do_logsoftmax) {
      float scale = 1.0f / sum;
      MlasComputeSoftmaxOutputF32Kernel(tmp.data(), D, &scale);
      out_std = std::move(tmp);
    } else {
      float logs = std::log(sum);
      float params[2] = {neg_max, logs};
      MlasComputeLogSoftmaxOutputF32Kernel(input.data(), out_std.data(), D, params);
    }
  }

#if defined(__riscv) && defined(__riscv_vector)
  // RVV pipeline
  {
    float maxv = MlasReduceMaximumF32KernelRvv(input.data(), D);
    float neg_max = -maxv;
    std::vector<float> tmp(D);
    float sum = MlasComputeSumExpF32KernelRvv(input.data(), opt.do_logsoftmax ? nullptr : tmp.data(), D, &neg_max);
    max_rvv = maxv;
    sum_rvv = sum;
    if (!opt.do_logsoftmax) {
      float scale = 1.0f / sum;
      MlasComputeSoftmaxOutputF32KernelRvv(tmp.data(), D, &scale);
      out_rvv = std::move(tmp);
    } else {
      float logs = std::log(sum);
      float params[2] = {neg_max, logs};
      MlasComputeLogSoftmaxOutputF32KernelRvv(input.data(), out_rvv.data(), D, params);
    }
  }
#else
  (void)out_rvv; // silence unused in non-RVV builds
#endif

  // Compare
#if defined(__riscv) && defined(__riscv_vector)
  Metrics m = compare_arrays(out_std.data(), out_rvv.data(), D);
  std::printf("D=%zu  max_abs_err=%.3e  max_rel_err=%.3e\n", D, m.max_abs_err, m.max_rel_err);
  // 小尺寸时打印详细对比：max/sum 以及逐元素
  if (D <= kPrintThreshold) {
    std::printf("  max(std)=%.9e  max(rvv)=%.9e  delta=%.3e\n", max_std, max_rvv, std::fabs(max_std - max_rvv));
    std::printf("  sum(std)=%.9e  sum(rvv)=%.9e  delta=%.3e\n", sum_std, sum_rvv, std::fabs(sum_std - sum_rvv));
    std::puts("  index:        std                rvv                diff");
    for (size_t i = 0; i < D; ++i) {
      double diff = static_cast<double>(out_std[i]) - static_cast<double>(out_rvv[i]);
      std::printf("  [%4zu]  % .9e  % .9e  % .3e\n", i, (double)out_std[i], (double)out_rvv[i], std::fabs(diff));
    }
  }
#else
  std::printf("D=%zu  (RVV not enabled in this build)\n", D);
#endif

  // Micro-benchmark: run full pipelines multiple iterations
  auto bench = [&](bool rvv) {
    using clk = std::chrono::steady_clock;
    volatile float sink = 0.0f; // prevent optimizing away
  std::vector<float> tmp(D), tmpo(D);
    // warmup
    for (int w = 0; w < 5; ++w) {
      if (!rvv) {
        float maxv = MlasReduceMaximumF32Kernel(input.data(), D);
        float neg_max = -maxv;
    float sum = MlasComputeSumExpF32Kernel(input.data(), tmp.data(), D, &neg_max);
        sink += sum;
      } else {
#if defined(__riscv) && defined(__riscv_vector)
        float maxv = MlasReduceMaximumF32KernelRvv(input.data(), D);
        float neg_max = -maxv;
    float sum = MlasComputeSumExpF32KernelRvv(input.data(), tmp.data(), D, &neg_max);
        sink += sum;
#endif
      }
    }
    auto t0 = clk::now();
    for (int it = 0; it < opt.iters; ++it) {
      if (!rvv) {
        float maxv = MlasReduceMaximumF32Kernel(input.data(), D);
        float neg_max = -maxv;
    float sum = MlasComputeSumExpF32Kernel(input.data(), tmp.data(), D, &neg_max);
        if (!opt.do_logsoftmax) {
          float scale = 1.0f / sum;
          MlasComputeSoftmaxOutputF32Kernel(tmp.data(), D, &scale);
          sink += tmp[0];
        } else {
          float params[2] = {neg_max, std::log(sum)};
          MlasComputeLogSoftmaxOutputF32Kernel(input.data(), tmpo.data(), D, params);
          sink += tmpo[0];
        }
      } else {
#if defined(__riscv) && defined(__riscv_vector)
        float maxv = MlasReduceMaximumF32KernelRvv(input.data(), D);
        float neg_max = -maxv;
    float sum = MlasComputeSumExpF32KernelRvv(input.data(), tmp.data(), D, &neg_max);
        if (!opt.do_logsoftmax) {
          float scale = 1.0f / sum;
          MlasComputeSoftmaxOutputF32KernelRvv(tmp.data(), D, &scale);
          sink += tmp[0];
        } else {
          float params[2] = {neg_max, std::log(sum)};
          MlasComputeLogSoftmaxOutputF32KernelRvv(input.data(), tmpo.data(), D, params);
          sink += tmpo[0];
        }
#endif
      }
    }
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms;
  };

  double t_std = bench(false);
#if defined(__riscv) && defined(__riscv_vector)
  double t_rvv = bench(true);
  std::printf("bench D=%zu iters=%d  std=%.3f ms  rvv=%.3f ms  speedup=%.2fx\n\n",
              D, opt.iters, t_std, t_rvv, t_std / std::max(1e-9, t_rvv));
#else
  std::printf("bench D=%zu iters=%d  std=%.3f ms  (rvv n/a)\n\n", D, opt.iters, t_std);
#endif
}

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  std::printf("RVV Softmax Regression & Micro-benchmark\n");
  std::printf("sizes=");
  for (size_t i = 0; i < opt.sizes.size(); ++i) {
    std::printf("%zu%s", opt.sizes[i], (i + 1 == opt.sizes.size()) ? "" : ",");
  }
  std::printf("  iters=%d  with_nan=%d  logsoftmax=%d  seed=%u\n\n",
              opt.iters, opt.with_nan ? 1 : 0, opt.do_logsoftmax ? 1 : 0, opt.seed);

  for (size_t D : opt.sizes) {
    run_case(D, opt);
  }
  return 0;
}
