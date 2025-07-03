#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <memory>

#include "poseidon.hpp"
#include "../cuda/poseidon_cuda.cuh"
#include "../cuda/poseidon_cuda_benchmarks.hpp"

using namespace Poseidon;

class PoseidonCUDATest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create CUDA Poseidon instance
    hasher_ = std::make_unique<PoseidonCUDA::CudaPoseidonHash>();
    if (!hasher_->is_initialized()) {
      GTEST_SKIP() << "CUDA not available or initialization failed";
    }
  }

  void TearDown() override { 
    // Destructor will handle cleanup automatically
    hasher_.reset();
  }

protected:
  std::unique_ptr<PoseidonCUDA::CudaPoseidonHash> hasher_;
};

TEST_F(PoseidonCUDATest, InitializationTest) {
  // Test that initialization works (already tested in SetUp)
  EXPECT_TRUE(hasher_->is_initialized());
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
  ASSERT_TRUE(
      hasher_->batch_hash_single(inputs, gpu_results));
  EXPECT_EQ(gpu_results.size(), batch_size);

  // Verify each result against CPU
  for (size_t i = 0; i < batch_size; ++i) {
    FieldElement cpu_result = PoseidonHash::hash_single(inputs[i]);
    EXPECT_EQ(cpu_result, gpu_results[i])
        << "Batch result differs at index " << i;
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
  ASSERT_TRUE(hasher_->batch_hash_pairs(
      left_inputs, right_inputs, gpu_results));
  EXPECT_EQ(gpu_results.size(), batch_size);

  // Verify each result against CPU
  for (size_t i = 0; i < batch_size; ++i) {
    FieldElement cpu_result =
        PoseidonHash::hash_pair(left_inputs[i], right_inputs[i]);
    EXPECT_EQ(cpu_result, gpu_results[i])
        << "Batch pair result differs at index " << i;
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
  ASSERT_TRUE(
      hasher_->batch_hash_single(inputs, gpu_results));
  EXPECT_EQ(gpu_results.size(), large_batch_size);

  // Spot check a few results against CPU
  std::vector<size_t> check_indices = {0, 100, 1000, 5000,
                                       large_batch_size - 1};
  for (size_t idx : check_indices) {
    FieldElement cpu_result = PoseidonHash::hash_single(inputs[idx]);
    EXPECT_EQ(cpu_result, gpu_results[idx])
        << "Large batch result differs at index " << idx;
  }
}

TEST_F(PoseidonCUDATest, EmptyBatchTest) {
  std::vector<FieldElement> empty_inputs;
  std::vector<FieldElement> outputs;

  EXPECT_TRUE(
      hasher_->batch_hash_single(empty_inputs, outputs));
  EXPECT_TRUE(outputs.empty());
}

TEST_F(PoseidonCUDATest, PerformanceComparisonTest) {
  const size_t num_hashes = 1000;
  const size_t batch_size = 256;

  // Run comparison benchmark
  auto stats =
      PoseidonCUDA::benchmark_cuda_vs_cpu_poseidon(*hasher_, num_hashes, batch_size);

  // Verify stats are reasonable
  EXPECT_GT(stats.total_time_ms, 0.0);
  EXPECT_GT(stats.avg_time_per_hash_ns, 0.0);
  EXPECT_GT(stats.hashes_per_second, 0);
  EXPECT_EQ(stats.total_hashes, num_hashes);

  if (stats.speedup_vs_cpu > 0) {
    std::cout << "CUDA vs CPU Performance Comparison:" << std::endl;
    std::cout << "  CUDA time per hash: " << stats.avg_time_per_hash_ns << " ns"
              << std::endl;
    std::cout << "  CUDA hashes per second: " << stats.hashes_per_second
              << std::endl;
    std::cout << "  Speedup vs CPU: " << stats.speedup_vs_cpu << "x"
              << std::endl;

    // Expect at least some speedup for large batches
    EXPECT_GT(stats.speedup_vs_cpu, 1.0)
        << "Expected GPU speedup for large batches";
  }
}

TEST_F(PoseidonCUDATest, ErrorHandlingTest) {
  // Test with mismatched vector sizes
  std::vector<FieldElement> left_inputs = {FieldElement(1), FieldElement(2)};
  std::vector<FieldElement> right_inputs = {FieldElement(3)}; // Different size
  std::vector<FieldElement> outputs;

  EXPECT_FALSE(hasher_->batch_hash_pairs(
      left_inputs, right_inputs, outputs));
}

// Test for deterministic results
TEST_F(PoseidonCUDATest, DeterministicTest) {
  FieldElement test_input(0x123456789ABCDEFULL);

  // Run the same hash multiple times
  std::vector<FieldElement> results;
  for (int i = 0; i < 5; ++i) {
    std::vector<FieldElement> single_input = {test_input};
    std::vector<FieldElement> single_result;
    ASSERT_TRUE(
        hasher_->batch_hash_single(single_input, single_result));
    ASSERT_EQ(single_result.size(), 1);
    results.push_back(single_result[0]);
  }

  // All results should be identical
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0], results[i])
        << "Results should be deterministic, but iteration " << i << " differs";
  }
}