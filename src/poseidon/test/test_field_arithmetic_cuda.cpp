#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "field_arithmetic.hpp"
#include "field_arithmetic_cuda.cuh"

using namespace Poseidon;
using CudaFieldArithmetic = Poseidon::CudaFieldOps::CudaFieldArithmetic;

class CudaFieldArithmeticTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FieldConstants::init();

    // Skip CUDA tests if no CUDA devices available
    if (CudaFieldArithmetic::get_device_count() == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    // Initialize CUDA
    if (!CudaFieldArithmetic::initialize()) {
      GTEST_SKIP() << "Failed to initialize CUDA";
    }
  }

  void TearDown() override {
    CudaFieldArithmetic::cleanup();
  }

  // Helper function to create test vectors
  std::vector<FieldElement> createTestVector(size_t count) {
    std::vector<FieldElement> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      result.push_back(FieldElement::random());
    }
    return result;
  }

  // Helper function to compare CPU and GPU results
  bool compareResults(const std::vector<FieldElement>& cpu_result,
                      const std::vector<FieldElement>& gpu_result,
                      const std::string& operation_name) {
    if (cpu_result.size() != gpu_result.size()) {
      std::cerr << operation_name << ": Size mismatch" << std::endl;
      return false;
    }

    for (size_t i = 0; i < cpu_result.size(); ++i) {
      if (cpu_result[i] != gpu_result[i]) {
        std::cerr << operation_name << ": Mismatch at index " << i << std::endl;
        std::cerr << "  CPU: " << cpu_result[i].to_hex() << std::endl;
        std::cerr << "  GPU: " << gpu_result[i].to_hex() << std::endl;
        return false;
      }
    }
    return true;
  }
};

TEST_F(CudaFieldArithmeticTest, BatchAddition) {
  const size_t test_size = 1000;

  // Create test data
  auto a = createTestVector(test_size);
  auto b = createTestVector(test_size);

  // CPU computation
  std::vector<FieldElement> cpu_result;
  cpu_result.reserve(test_size);
  for (size_t i = 0; i < test_size; ++i) {
    cpu_result.push_back(a[i] + b[i]);
  }

  // GPU computation
  std::vector<FieldElement> gpu_result;
  ASSERT_TRUE(CudaFieldArithmetic::batch_add(a, b, gpu_result));

  // Compare results
  EXPECT_TRUE(compareResults(cpu_result, gpu_result, "Batch Addition"));
}

TEST_F(CudaFieldArithmeticTest, BatchSubtraction) {
  const size_t test_size = 1000;

  // Create test data (make sure a > b to avoid modular arithmetic complexity)
  auto a = createTestVector(test_size);
  auto b = createTestVector(test_size);

  // Modify b to be smaller than a
  for (size_t i = 0; i < test_size; ++i) {
    b[i] = FieldElement(i + 1);        // b[i] = i+1
    a[i] = FieldElement(2 * (i + 1));  // a[i] = 2*(i+1)
  }

  // CPU computation
  std::vector<FieldElement> cpu_result;
  cpu_result.reserve(test_size);
  for (size_t i = 0; i < test_size; ++i) {
    cpu_result.push_back(a[i] - b[i]);
  }

  // GPU computation
  std::vector<FieldElement> gpu_result;
  ASSERT_TRUE(CudaFieldArithmetic::batch_subtract(a, b, gpu_result));

  // Compare results
  EXPECT_TRUE(compareResults(cpu_result, gpu_result, "Batch Subtraction"));
}

TEST_F(CudaFieldArithmeticTest, BatchMultiplication) {
  const size_t test_size = 500;  // Smaller size for multiplication (more expensive)

  // Create test data
  auto a = createTestVector(test_size);
  auto b = createTestVector(test_size);

  // CPU computation
  std::vector<FieldElement> cpu_result;
  cpu_result.reserve(test_size);
  for (size_t i = 0; i < test_size; ++i) {
    cpu_result.push_back(a[i] * b[i]);
  }

  // GPU computation
  std::vector<FieldElement> gpu_result;
  ASSERT_TRUE(CudaFieldArithmetic::batch_multiply(a, b, gpu_result));

  // Compare results
  EXPECT_TRUE(compareResults(cpu_result, gpu_result, "Batch Multiplication"));
}

TEST_F(CudaFieldArithmeticTest, BatchSquare) {
  const size_t test_size = 500;

  // Create test data
  auto input = createTestVector(test_size);

  // CPU computation
  std::vector<FieldElement> cpu_result;
  cpu_result.reserve(test_size);
  for (size_t i = 0; i < test_size; ++i) {
    FieldElement temp;
    FieldArithmetic::square(input[i], temp);
    cpu_result.push_back(temp);
  }

  // GPU computation
  std::vector<FieldElement> gpu_result;
  ASSERT_TRUE(CudaFieldArithmetic::batch_square(input, gpu_result));

  // Compare results
  EXPECT_TRUE(compareResults(cpu_result, gpu_result, "Batch Square"));
}

TEST_F(CudaFieldArithmeticTest, BatchPower5) {
  const size_t test_size = 500;

  // Create test data
  auto input = createTestVector(test_size);

  // CPU computation
  std::vector<FieldElement> cpu_result;
  cpu_result.reserve(test_size);
  for (size_t i = 0; i < test_size; ++i) {
    FieldElement temp;
    FieldArithmetic::power5(input[i], temp);
    cpu_result.push_back(temp);
  }

  // GPU computation
  std::vector<FieldElement> gpu_result;
  ASSERT_TRUE(CudaFieldArithmetic::batch_power5(input, gpu_result));

  // Compare results
  EXPECT_TRUE(compareResults(cpu_result, gpu_result, "Batch Power5"));
}

TEST_F(CudaFieldArithmeticTest, LargeVectorPerformance) {
  const size_t large_size = 10000;

  std::cout << "\n=== Performance Test (Large Vectors) ===" << std::endl;
  std::cout << "Vector size: " << large_size << " elements" << std::endl;

  // Create large test vectors
  std::vector<FieldElement> a, b;
  a.reserve(large_size);
  b.reserve(large_size);

  for (size_t i = 0; i < large_size; ++i) {
    a.push_back(FieldElement::random());
    b.push_back(FieldElement::random());
  }

  // Time GPU multiplication
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<FieldElement> gpu_result;
  bool success = CudaFieldArithmetic::batch_multiply(a, b, gpu_result);
  auto end = std::chrono::high_resolution_clock::now();

  ASSERT_TRUE(success);

  auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "GPU batch multiplication time: " << gpu_duration.count() << " ms" << std::endl;

  // Time CPU multiplication for comparison
  start = std::chrono::high_resolution_clock::now();
  std::vector<FieldElement> cpu_result;
  cpu_result.reserve(large_size);
  for (size_t i = 0; i < large_size; ++i) {
    cpu_result.push_back(a[i] * b[i]);
  }
  end = std::chrono::high_resolution_clock::now();

  auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "CPU batch multiplication time: " << cpu_duration.count() << " ms" << std::endl;

  if (cpu_duration.count() > 0) {
    double speedup = static_cast<double>(cpu_duration.count()) / gpu_duration.count();
    std::cout << "GPU speedup: " << speedup << "x" << std::endl;
  }

  // Verify correctness for a sample
  bool correct = true;
  for (size_t i = 0; i < std::min(static_cast<size_t>(100), large_size); ++i) {
    if (cpu_result[i] != gpu_result[i]) {
      correct = false;
      break;
    }
  }
  EXPECT_TRUE(correct);
}

TEST_F(CudaFieldArithmeticTest, EdgeCases) {
  // Test with empty vectors
  std::vector<FieldElement> empty_a, empty_b, empty_result;
  EXPECT_TRUE(CudaFieldArithmetic::batch_add(empty_a, empty_b, empty_result));
  EXPECT_EQ(empty_result.size(), 0);

  // Test with single element
  std::vector<FieldElement> single_a = {FieldElement(42)};
  std::vector<FieldElement> single_b = {FieldElement(13)};
  std::vector<FieldElement> single_result;

  EXPECT_TRUE(CudaFieldArithmetic::batch_add(single_a, single_b, single_result));
  EXPECT_EQ(single_result.size(), 1);
  EXPECT_EQ(single_result[0], FieldElement(55));

  // Test with mismatched sizes
  std::vector<FieldElement> mismatch_a = {FieldElement(1), FieldElement(2)};
  std::vector<FieldElement> mismatch_b = {FieldElement(1)};
  std::vector<FieldElement> mismatch_result;

  EXPECT_FALSE(CudaFieldArithmetic::batch_add(mismatch_a, mismatch_b, mismatch_result));
}

TEST_F(CudaFieldArithmeticTest, Reduce512Consistency) {
  std::cout << "\n=== Testing 512-bit Reduction Consistency ===" << std::endl;

  // Test vectors that specifically exercise the 512-bit reduction
  std::vector<std::pair<FieldElement, FieldElement>> test_pairs;

  // Test case 1: Small values (high part will be zero)
  test_pairs.push_back({FieldElement(123), FieldElement(456)});

  // Test case 2: Medium values
  test_pairs.push_back({FieldElement(0x123456789ABCDEFULL), FieldElement(0xFEDCBA9876543210ULL)});

  // Test case 3: Large values that will produce significant high parts
  test_pairs.push_back({FieldElement(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0, 0),
                        FieldElement(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0, 0)});

  // Test case 4: Random values
  for (int i = 0; i < 10; ++i) {
    test_pairs.push_back({FieldElement::random(), FieldElement::random()});
  }

  for (const auto& pair : test_pairs) {
    const FieldElement& a = pair.first;
    const FieldElement& b = pair.second;

    // Manually compute 512-bit product
    uint64_t product[8] = {0};
    for (int i = 0; i < 4; ++i) {
      uint64_t carry = 0;
      for (int j = 0; j < 4; ++j) {
        __uint128_t prod = (__uint128_t)a.limbs[i] * b.limbs[j] + product[i + j] + carry;
        product[i + j] = (uint64_t)prod;
        carry = (uint64_t)(prod >> 64);
      }
      product[i + 4] = carry;
    }

    // Test CPU reduction
    FieldElement cpu_result;
    FieldArithmetic::reduce_512(product, cpu_result);

    // Test that CPU reduction matches regular multiplication
    FieldElement cpu_mult_result = a * b;
    EXPECT_EQ(cpu_result, cpu_mult_result)
        << "CPU reduce_512 doesn't match regular multiplication for inputs: " << a.to_hex() << " * "
        << b.to_hex();

    // Test GPU multiplication (which uses CUDA reduce_512 internally)
    std::vector<FieldElement> gpu_a = {a};
    std::vector<FieldElement> gpu_b = {b};
    std::vector<FieldElement> gpu_result;

    ASSERT_TRUE(CudaFieldArithmetic::batch_multiply(gpu_a, gpu_b, gpu_result));
    ASSERT_EQ(gpu_result.size(), 1);

    // Compare CPU and GPU results
    EXPECT_EQ(cpu_result, gpu_result[0])
        << "CPU and GPU 512-bit reduction mismatch for inputs: " << a.to_hex() << " * "
        << b.to_hex() << "\nCPU result: " << cpu_result.to_hex()
        << "\nGPU result: " << gpu_result[0].to_hex();
  }

  std::cout << "Tested " << test_pairs.size()
            << " multiplication pairs for 512-bit reduction consistency" << std::endl;
}
