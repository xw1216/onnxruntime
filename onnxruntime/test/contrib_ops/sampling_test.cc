// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include <gsl/gsl>
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/cuda_op_test_utils.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__linux__) && !defined(__ANDROID__)
#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(SamplingTest, Gpt2Sampling_GPU) {
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int32_t> max_length{15};
  std::vector<int32_t> min_length{1};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 125, 543, 668,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 776, 213, 697,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 450};

  const int64_t batch_size = 3;
  const int64_t sequence_length = 12;

  std::vector<int64_t> input_ids_shape{batch_size, sequence_length};
  std::vector<int64_t> parameter_shape{1};
  std::vector<int64_t> expected_output_shape{input_ids_shape[0], max_length[0]};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  Ort::SessionOptions session_options;
#ifdef USE_CUDA
  constexpr int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support current architecture";
    return;
  }

  OrtCUDAProviderOptionsV2 cuda_options;
  cuda_options.use_tf32 = false;
  session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#else  // USE_ROCM
  OrtROCMProviderOptions rocm_options;
  // TODO - verify the default settings
  session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

  Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_sampling.onnx"), session_options);

  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);

  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sequences = ort_outputs[0];
  ASSERT_TRUE(sequences.IsTensor());

  auto result_ts = sequences.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

  ASSERT_EQ(expected_output_shape, result_ts.GetShape());
  const auto* result_vals = sequences.GetTensorData<int32_t>();
  auto result_span = gsl::make_span(result_vals, expected_output.size());
  // 始终打印实际序列，方便在新平台采集 expected
  printf("Actual GPT2 sequence (%zu): {", result_span.size());
  for (size_t i = 0; i < result_span.size(); ++i) {
    if (i) printf(", ");
    printf("%d", result_span[i]);
  }
  printf("}\n");
  // 若不相等，打印前若干差异索引
  if (!std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end())) {
  printf("First diffs (index: expected != actual): ");
    int printed = 0;
    for (size_t i = 0; i < result_span.size() && printed < 10; ++i) {
      if (expected_output[i] != result_span[i]) {
    printf("%zu:%d!=%d ", i, expected_output[i], result_span[i]);
        ++printed;
      }
    }
  printf("\n");
  }

  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
}
#endif

TEST(SamplingTest, Gpt2Sampling_CPU) {
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int32_t> max_length{15};
  std::vector<int32_t> min_length{1};
  std::vector<float> repetition_penalty{1.0f};

  // 平台相关期望输出：std::default_random_engine 行为不同
#ifdef __OHOS__
  // 来自实际运行 (OHOS riscv64)
  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 76, 390, 800,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 896, 717, 524,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 182};
#else
  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 125, 669, 28,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 475, 944, 527,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 210};
#endif

  const int64_t batch_size = 3;
  const int64_t sequence_length = 12;
  std::vector<int64_t> input_ids_shape{batch_size, sequence_length};

  std::vector<int64_t> parameter_shape{1};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], max_length[0]};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_sampling.onnx"), session_options);

  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);

  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sequences = ort_outputs[0];
  ASSERT_TRUE(sequences.IsTensor());

  auto result_ts = sequences.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

  ASSERT_EQ(expected_output_shape, result_ts.GetShape());
  const auto* result_vals = sequences.GetTensorData<int32_t>();
  auto result_span = gsl::make_span(result_vals, expected_output.size());

  if (std::getenv("ORT_DUMP_ACTUAL")) {
    printf("Actual GPT2 CPU sequence (%zu): {", result_span.size());
    for (size_t i = 0; i < result_span.size(); ++i) {
      if (i) printf(", ");
      printf("%d", result_span[i]);
    }
    printf("}\n");
    if (!std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end())) {
      printf("First diffs (idx exp!=act): ");
      int printed = 0;
      for (size_t i = 0; i < result_span.size() && printed < 10; ++i) {
        if (expected_output[i] != result_span[i]) { printf("%zu:%d!=%d ", i, expected_output[i], result_span[i]); ++printed; }
      }
      printf("\n");
    }
  }

  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
}
#endif
}  // namespace test
}  // namespace onnxruntime
