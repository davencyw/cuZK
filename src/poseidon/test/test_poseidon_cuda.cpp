#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "poseidon.hpp"
#include "poseidon_cuda.cuh"

using namespace Poseidon;

class PoseidonCUDATest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize CUDA Poseidon
    if (!PoseidonCUDA::CudaPoseidonHash::initialize()) {
      GTEST_SKIP() << "CUDA not available or initialization failed";
    }
  }

  void TearDown() override {
    PoseidonCUDA::CudaPoseidonHash::cleanup();
  }
};

TEST_F(PoseidonCUDATest, InitializationTest) {
  // Test that initialization works
  EXPECT_TRUE(PoseidonCUDA::CudaPoseidonHash::initialize());

  // Test performance info
  PoseidonCUDA::CudaPoseidonHash::print_performance_info();
}

TEST_F(PoseidonCUDATest, SingleHashConsistencyTest) {
  // Generate test inputs
  std::vector<FieldElement> test_inputs = {FieldElement(0),
                                           FieldElement(1),
                                           FieldElement(42),
                                           FieldElement(0x123456789ABCDEFULL),
                                           FieldElement::random(),
                                           FieldElement::random()};

  for (const auto& input : test_inputs) {
    // Compute hash on CPU
    FieldElement cpu_result = PoseidonHash::hash_single(input);

    // Compute hash on GPU
    FieldElement gpu_result;
    ASSERT_TRUE(PoseidonCUDA::CudaPoseidonHash::gpu_hash_single(input, gpu_result));

    // Compare results
    EXPECT_EQ(cpu_result, gpu_result) << "CPU and GPU results differ for input: " << input.to_hex();
  }
}

TEST_F(PoseidonCUDATest, PairHashConsistencyTest) {
  // Generate test input pairs
  std::vector<std::pair<FieldElement, FieldElement>> test_pairs = {
      {FieldElement(0), FieldElement(0)},
      {FieldElement(1), FieldElement(0)},
      {FieldElement(0), FieldElement(1)},
      {FieldElement(42), FieldElement(123)},
      {FieldElement::random(), FieldElement::random()},
      {FieldElement::random(), FieldElement::random()}};

  for (const auto& pair : test_pairs) {
    // Compute hash on CPU
    FieldElement cpu_result = PoseidonHash::hash_pair(pair.first, pair.second);

    // Compute hash on GPU
    FieldElement gpu_result;
    ASSERT_TRUE(PoseidonCUDA::CudaPoseidonHash::gpu_hash_pair(pair.first, pair.second, gpu_result));

    // Compare results
    EXPECT_EQ(cpu_result, gpu_result) << "CPU and GPU results differ for pair: ("
                                      << pair.first.to_hex() << ", " << pair.second.to_hex() << ")";
  }
}

TEST_F(PoseidonCUDATest, BatchSingleHashTest) {
  const size_t batch_size = 100;

  // Generate test inputs
  std::vector<FieldElement> inputs;
  inputs.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    inputs.push_back(FieldElement::random());
  }

  // Compute batch hash on GPU
  std::vector<FieldElement> gpu_results;
  ASSERT_TRUE(PoseidonCUDA::CudaPoseidonHash::batch_hash_single(inputs, gpu_results));
  EXPECT_EQ(gpu_results.size(), batch_size);

  // Verify each result against CPU
  for (size_t i = 0; i < batch_size; ++i) {
    FieldElement cpu_result = PoseidonHash::hash_single(inputs[i]);
    EXPECT_EQ(cpu_result, gpu_results[i]) << "Batch result differs at index " << i;
  }
}

TEST_F(PoseidonCUDATest, BatchPairHashTest) {
  const size_t batch_size = 100;

  // Generate test input pairs
  std::vector<FieldElement> left_inputs, right_inputs;
  left_inputs.reserve(batch_size);
  right_inputs.reserve(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    left_inputs.push_back(FieldElement::random());
    right_inputs.push_back(FieldElement::random());
  }

  // Compute batch hash on GPU
  std::vector<FieldElement> gpu_results;
  ASSERT_TRUE(
      PoseidonCUDA::CudaPoseidonHash::batch_hash_pairs(left_inputs, right_inputs, gpu_results));
  EXPECT_EQ(gpu_results.size(), batch_size);

  // Verify each result against CPU
  for (size_t i = 0; i < batch_size; ++i) {
    FieldElement cpu_result = PoseidonHash::hash_pair(left_inputs[i], right_inputs[i]);
    EXPECT_EQ(cpu_result, gpu_results[i]) << "Batch pair result differs at index " << i;
  }
}

TEST_F(PoseidonCUDATest, LargeBatchTest) {
  const size_t large_batch_size = 10000;

  // Generate test inputs
  std::vector<FieldElement> inputs;
  inputs.reserve(large_batch_size);
  for (size_t i = 0; i < large_batch_size; ++i) {
    inputs.push_back(FieldElement(i));
  }

  // Compute batch hash on GPU
  std::vector<FieldElement> gpu_results;
  ASSERT_TRUE(PoseidonCUDA::CudaPoseidonHash::batch_hash_single(inputs, gpu_results));
  EXPECT_EQ(gpu_results.size(), large_batch_size);

  // Spot check a few results against CPU
  std::vector<size_t> check_indices = {0, 100, 1000, 5000, large_batch_size - 1};
  for (size_t idx : check_indices) {
    FieldElement cpu_result = PoseidonHash::hash_single(inputs[idx]);
    EXPECT_EQ(cpu_result, gpu_results[idx]) << "Large batch result differs at index " << idx;
  }
}

TEST_F(PoseidonCUDATest, EmptyBatchTest) {
  std::vector<FieldElement> empty_inputs;
  std::vector<FieldElement> outputs;

  EXPECT_TRUE(PoseidonCUDA::CudaPoseidonHash::batch_hash_single(empty_inputs, outputs));
  EXPECT_TRUE(outputs.empty());
}

TEST_F(PoseidonCUDATest, PerformanceBenchmarkTest) {
  const size_t num_hashes = 1000;
  const size_t batch_size = 256;

  // Run benchmark
  auto stats = PoseidonCUDA::benchmark_cuda_poseidon_single(num_hashes, batch_size);

  // Verify stats are reasonable
  EXPECT_GT(stats.total_time_ms, 0.0);
  EXPECT_GT(stats.avg_time_per_hash_ns, 0.0);
  EXPECT_GT(stats.hashes_per_second, 0);
  EXPECT_EQ(stats.total_hashes, num_hashes);

  std::cout << "CUDA Poseidon Performance:" << std::endl;
  std::cout << "  Total time: " << stats.total_time_ms << " ms" << std::endl;
  std::cout << "  Avg time per hash: " << stats.avg_time_per_hash_ns << " ns" << std::endl;
  std::cout << "  Hashes per second: " << stats.hashes_per_second << std::endl;
  std::cout << "  GPU memory used: " << stats.gpu_memory_used_mb << " MB" << std::endl;
}

TEST_F(PoseidonCUDATest, PerformanceComparisonTest) {
  const size_t num_hashes = 1000;
  const size_t batch_size = 256;

  // Run comparison benchmark
  auto stats = PoseidonCUDA::benchmark_cuda_vs_cpu_poseidon(num_hashes, batch_size);

  // Verify stats are reasonable
  EXPECT_GT(stats.total_time_ms, 0.0);
  EXPECT_GT(stats.avg_time_per_hash_ns, 0.0);
  EXPECT_GT(stats.hashes_per_second, 0);
  EXPECT_EQ(stats.total_hashes, num_hashes);

  if (stats.speedup_vs_cpu > 0) {
    std::cout << "CUDA vs CPU Performance Comparison:" << std::endl;
    std::cout << "  CUDA time per hash: " << stats.avg_time_per_hash_ns << " ns" << std::endl;
    std::cout << "  CUDA hashes per second: " << stats.hashes_per_second << std::endl;
    std::cout << "  Speedup vs CPU: " << stats.speedup_vs_cpu << "x" << std::endl;

    // Expect at least some speedup for large batches
    EXPECT_GT(stats.speedup_vs_cpu, 1.0) << "Expected GPU speedup for large batches";
  }
}

TEST_F(PoseidonCUDATest, ErrorHandlingTest) {
  // Test with mismatched vector sizes
  std::vector<FieldElement> left_inputs = {FieldElement(1), FieldElement(2)};
  std::vector<FieldElement> right_inputs = {FieldElement(3)};  // Different size
  std::vector<FieldElement> outputs;

  EXPECT_FALSE(
      PoseidonCUDA::CudaPoseidonHash::batch_hash_pairs(left_inputs, right_inputs, outputs));
}

// Test for deterministic results
TEST_F(PoseidonCUDATest, DeterministicTest) {
  FieldElement test_input(0x123456789ABCDEFULL);

  // Run the same hash multiple times
  std::vector<FieldElement> results;
  for (int i = 0; i < 5; ++i) {
    FieldElement result;
    ASSERT_TRUE(PoseidonCUDA::CudaPoseidonHash::gpu_hash_single(test_input, result));
    results.push_back(result);
  }

  // All results should be identical
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0], results[i])
        << "Results should be deterministic, but iteration " << i << " differs";
  }
}