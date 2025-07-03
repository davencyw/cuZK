#include <gtest/gtest.h>

#include <iostream>

#include "poseidon.hpp"

using namespace Poseidon;

class PoseidonTest : public ::testing::Test {
protected:
  void SetUp() override { PoseidonConstants::init(); }
};

TEST_F(PoseidonTest, ConstantsInitialization) {
  // Test that constants are properly initialized
  EXPECT_TRUE(PoseidonConstants::initialized_);

  // Check that round constants are non-zero
  bool found_nonzero = false;
  for (const auto &constant : PoseidonConstants::ROUND_CONSTANTS) {
    if (!constant.is_zero()) {
      found_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(found_nonzero);

  // Check that MDS matrix has non-zero elements
  found_nonzero = false;
  for (const auto &element : PoseidonConstants::MDS_MATRIX) {
    if (!element.is_zero()) {
      found_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(found_nonzero);
}

TEST_F(PoseidonTest, PermutationConsistency) {
  // Test that the permutation is deterministic
  FieldElement state1[PoseidonParams::STATE_SIZE] = {
      FieldElement(1), FieldElement(2), FieldElement(3)};
  FieldElement state2[PoseidonParams::STATE_SIZE] = {
      FieldElement(1), FieldElement(2), FieldElement(3)};

  PoseidonHash::permutation(state1);
  PoseidonHash::permutation(state2);

  for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
    EXPECT_EQ(state1[i], state2[i]);
  }
}

TEST_F(PoseidonTest, PermutationChangesState) {
  // Test that permutation actually changes the state
  const FieldElement original_state[PoseidonParams::STATE_SIZE] = {
      FieldElement(1), FieldElement(2), FieldElement(3)};
  FieldElement modified_state[PoseidonParams::STATE_SIZE] = {
      FieldElement(1), FieldElement(2), FieldElement(3)};

  PoseidonHash::permutation(modified_state);

  bool state_changed = false;
  for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
    if (original_state[i] != modified_state[i]) {
      state_changed = true;
      break;
    }
  }
  EXPECT_TRUE(state_changed);
}

TEST_F(PoseidonTest, HashSingleConsistency) {
  FieldElement input(42);

  FieldElement hash1 = PoseidonHash::hash_single(input);
  FieldElement hash2 = PoseidonHash::hash_single(input);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(PoseidonTest, HashSingleDifferentInputs) {
  FieldElement input1(42);
  FieldElement input2(43);

  FieldElement hash1 = PoseidonHash::hash_single(input1);
  FieldElement hash2 = PoseidonHash::hash_single(input2);

  EXPECT_NE(hash1, hash2);
}

TEST_F(PoseidonTest, HashPairConsistency) {
  FieldElement left(10);
  FieldElement right(20);

  FieldElement hash1 = PoseidonHash::hash_pair(left, right);
  FieldElement hash2 = PoseidonHash::hash_pair(left, right);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(PoseidonTest, HashPairAsymmetry) {
  FieldElement left(10);
  FieldElement right(20);

  FieldElement hash1 = PoseidonHash::hash_pair(left, right);
  FieldElement hash2 = PoseidonHash::hash_pair(right, left);

  // Hash should be different when inputs are swapped
  EXPECT_NE(hash1, hash2);
}

TEST_F(PoseidonTest, HashPairVsSingle) {
  FieldElement input(42);
  FieldElement zero = FieldConstants::ZERO;

  FieldElement single_hash = PoseidonHash::hash_single(input);
  FieldElement pair_hash = PoseidonHash::hash_pair(input, zero);

  // These should be different as the state setup is different
  EXPECT_NE(single_hash, pair_hash);
}

TEST_F(PoseidonTest, HashMultipleEmpty) {
  std::vector<FieldElement> empty_inputs;
  FieldElement hash = PoseidonHash::hash_multiple(empty_inputs);

  // Should produce a valid hash even with empty input
  EXPECT_TRUE(hash < FieldConstants::MODULUS);
}

TEST_F(PoseidonTest, HashMultipleSingle) {
  std::vector<FieldElement> single_input = {FieldElement(42)};
  FieldElement hash = PoseidonHash::hash_multiple(single_input);

  EXPECT_TRUE(hash < FieldConstants::MODULUS);
  EXPECT_FALSE(hash.is_zero()); // Should not be zero for non-zero input
}

TEST_F(PoseidonTest, HashMultipleConsistency) {
  std::vector<FieldElement> inputs = {FieldElement(1), FieldElement(2),
                                      FieldElement(3), FieldElement(4)};

  FieldElement hash1 = PoseidonHash::hash_multiple(inputs);
  FieldElement hash2 = PoseidonHash::hash_multiple(inputs);

  EXPECT_EQ(hash1, hash2);
}

TEST_F(PoseidonTest, HashMultipleDifferentLengths) {
  std::vector<FieldElement> short_inputs = {FieldElement(1), FieldElement(2)};
  std::vector<FieldElement> long_inputs = {FieldElement(1), FieldElement(2),
                                           FieldElement(3)};

  FieldElement hash1 = PoseidonHash::hash_multiple(short_inputs);
  FieldElement hash2 = PoseidonHash::hash_multiple(long_inputs);

  EXPECT_NE(hash1, hash2);
}

TEST_F(PoseidonTest, SpongeFunction) {
  std::vector<FieldElement> inputs = {FieldElement(10), FieldElement(20),
                                      FieldElement(30)};

  FieldElement sponge_hash = PoseidonHash::sponge(inputs);
  FieldElement multiple_hash = PoseidonHash::hash_multiple(inputs);

  // These should be the same since hash_multiple uses sponge
  EXPECT_EQ(sponge_hash, multiple_hash);
}

TEST_F(PoseidonTest, ZeroInputHandling) {
  FieldElement zero = FieldConstants::ZERO;

  // Test single zero input
  FieldElement hash_zero = PoseidonHash::hash_single(zero);
  EXPECT_TRUE(hash_zero < FieldConstants::MODULUS);

  // Test pair with zeros
  FieldElement hash_pair_zeros = PoseidonHash::hash_pair(zero, zero);
  EXPECT_TRUE(hash_pair_zeros < FieldConstants::MODULUS);

  // Should be different hashes
  EXPECT_NE(hash_zero, hash_pair_zeros);
}

TEST_F(PoseidonTest, LargeInputHandling) {
  // Create a large input close to modulus
  FieldElement large_input = FieldConstants::MODULUS;
  large_input -= FieldConstants::ONE;

  FieldElement hash = PoseidonHash::hash_single(large_input);
  EXPECT_TRUE(hash < FieldConstants::MODULUS);
  EXPECT_FALSE(hash.is_zero());
}

TEST_F(PoseidonTest, PermutationRounds) {
  // Test that we have correct number of rounds
  EXPECT_EQ(PoseidonParams::TOTAL_ROUNDS,
            PoseidonParams::ROUNDS_FULL + PoseidonParams::ROUNDS_PARTIAL);
  EXPECT_EQ(PoseidonParams::ROUNDS_FULL, 8);
  EXPECT_EQ(PoseidonParams::ROUNDS_PARTIAL, 56);
}

TEST_F(PoseidonTest, StateSize) {
  EXPECT_EQ(PoseidonParams::STATE_SIZE, 3);
  EXPECT_EQ(PoseidonParams::CAPACITY, 1);
  EXPECT_EQ(PoseidonParams::RATE, 2);
  EXPECT_EQ(PoseidonParams::RATE,
            PoseidonParams::STATE_SIZE - PoseidonParams::CAPACITY);
}

TEST_F(PoseidonTest, RandomInputsProduceDifferentHashes) {
  // Generate several random inputs and verify they produce different hashes
  std::vector<FieldElement> hashes;
  const size_t num_tests = 100;

  for (size_t i = 0; i < num_tests; ++i) {
    FieldElement random_input = FieldElement::random();
    FieldElement hash = PoseidonHash::hash_single(random_input);
    hashes.push_back(hash);
  }

  // Count unique hashes (should be close to num_tests)
  std::sort(hashes.begin(), hashes.end(),
            [](const FieldElement &a, const FieldElement &b) { return a < b; });

  size_t unique_count = 1;
  for (size_t i = 1; i < hashes.size(); ++i) {
    if (hashes[i] != hashes[i - 1]) {
      unique_count++;
    }
  }

  // Should have very high uniqueness (allow for some tiny possibility of
  // collision)
  EXPECT_GT(unique_count, num_tests * 0.95);
}

TEST_F(PoseidonTest, FieldElementBounds) {
  // Test various inputs to ensure outputs are always in field
  std::vector<FieldElement> test_inputs = {
      FieldConstants::ZERO,   FieldConstants::ONE,    FieldConstants::TWO,
      FieldElement::random(), FieldElement::random(), FieldElement::random()};

  for (const auto &input : test_inputs) {
    FieldElement hash = PoseidonHash::hash_single(input);
    EXPECT_TRUE(hash < FieldConstants::MODULUS);

    // Test pairs
    for (const auto &second_input : test_inputs) {
      FieldElement pair_hash = PoseidonHash::hash_pair(input, second_input);
      EXPECT_TRUE(pair_hash < FieldConstants::MODULUS);
    }
  }
}

// Performance and benchmarking tests
TEST_F(PoseidonTest, BenchmarkSingleHashes) {
  const size_t iterations = 1000;
  HashingStats stats = benchmark_poseidon(iterations);

  EXPECT_EQ(stats.total_hashes, iterations);
  EXPECT_GT(stats.hashes_per_second, 0);
  EXPECT_GT(stats.avg_time_per_hash_ns, 0);
  EXPECT_GT(stats.total_time_ms, 0);

  std::cout << "Single hash benchmark:" << std::endl;
  std::cout << "  Total time: " << stats.total_time_ms << " ms" << std::endl;
  std::cout << "  Avg time per hash: " << stats.avg_time_per_hash_ns << " ns"
            << std::endl;
  std::cout << "  Hashes per second: " << stats.hashes_per_second << std::endl;
}

TEST_F(PoseidonTest, BenchmarkPairHashes) {
  const size_t iterations = 1000;
  HashingStats stats = benchmark_poseidon_pairs(iterations);

  EXPECT_EQ(stats.total_hashes, iterations);
  EXPECT_GT(stats.hashes_per_second, 0);

  std::cout << "Pair hash benchmark:" << std::endl;
  std::cout << "  Total time: " << stats.total_time_ms << " ms" << std::endl;
  std::cout << "  Avg time per hash: " << stats.avg_time_per_hash_ns << " ns"
            << std::endl;
  std::cout << "  Hashes per second: " << stats.hashes_per_second << std::endl;
}