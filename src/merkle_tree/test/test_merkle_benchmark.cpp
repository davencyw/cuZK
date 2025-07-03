#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "merkle_tree.hpp"

using namespace MerkleTree;

class MerkleTreeBenchmarkTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Poseidon constants
    Poseidon::PoseidonConstants::init();
    Poseidon::FieldConstants::init();
  }

  void print_benchmark_header() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "           N-ARY MERKLE TREE PERFORMANCE BENCHMARK\n";
    std::cout << std::string(80, '=') << "\n";
  }

  void print_benchmark_results(const MerkleUtils::TreeBenchmarkResult &result) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Leaves: " << std::setw(8) << result.leaf_count
              << " | Arity: " << std::setw(2) << result.arity
              << " | Height: " << std::setw(2) << result.tree_height
              << " | Build: " << std::setw(8) << result.build_time_ms << "ms"
              << " | Proof Gen: " << std::setw(8)
              << result.proof_generation_time_ms << "ms"
              << " | Proof Ver: " << std::setw(8)
              << result.proof_verification_time_ms << "ms\n";
  }
};

TEST_F(MerkleTreeBenchmarkTest, CompareArities) {
  print_benchmark_header();
  std::cout << "Comparing different tree arities with fixed dataset size\n";
  std::cout << std::string(80, '-') << "\n";

  const size_t leaf_count = 1024;
  const size_t num_proofs = 100;

  for (size_t arity = 2; arity <= 8; ++arity) {
    auto result = MerkleUtils::benchmark_tree(leaf_count, arity, num_proofs);
    print_benchmark_results(result);
  }

  std::cout << std::string(80, '=') << "\n";
}

TEST_F(MerkleTreeBenchmarkTest, ScaleWithDataSize) {
  print_benchmark_header();
  std::cout << "Scaling performance with dataset size (binary trees)\n";
  std::cout << std::string(80, '-') << "\n";

  const size_t arity = 2;
  const size_t num_proofs = 50;

  std::vector<size_t> dataset_sizes = {64, 128, 256, 512, 1024, 2048, 4096};

  for (size_t size : dataset_sizes) {
    auto result = MerkleUtils::benchmark_tree(size, arity, num_proofs);
    print_benchmark_results(result);
  }

  std::cout << std::string(80, '=') << "\n";
}

TEST_F(MerkleTreeBenchmarkTest, OptimalArityAnalysis) {
  print_benchmark_header();
  std::cout << "Finding optimal arity for different dataset sizes\n";
  std::cout << std::string(80, '-') << "\n";

  std::vector<size_t> dataset_sizes = {256, 1024, 4096};
  const size_t num_proofs = 100;

  for (size_t size : dataset_sizes) {
    std::cout << "\nDataset size: " << size << " leaves\n";
    std::cout << std::string(40, '-') << "\n";

    double best_total_time = std::numeric_limits<double>::max();
    size_t best_arity = 2;

    for (size_t arity = 2; arity <= 8; ++arity) {
      auto result = MerkleUtils::benchmark_tree(size, arity, num_proofs);
      double total_time = result.build_time_ms +
                          result.proof_generation_time_ms +
                          result.proof_verification_time_ms;

      if (total_time < best_total_time) {
        best_total_time = total_time;
        best_arity = arity;
      }

      print_benchmark_results(result);
    }

    std::cout << ">> Optimal arity for " << size << " leaves: " << best_arity
              << " (total: " << best_total_time << "ms)\n";
  }

  std::cout << std::string(80, '=') << "\n";
}

TEST_F(MerkleTreeBenchmarkTest, BatchProofPerformance) {
  print_benchmark_header();
  std::cout << "Batch proof generation performance\n";
  std::cout << std::string(80, '-') << "\n";

  const size_t leaf_count = 1024;
  const size_t arity = 4;

  auto test_data = MerkleUtils::generate_test_leaves(leaf_count);
  NaryMerkleTree tree(test_data, MerkleTreeConfig(arity));

  std::vector<size_t> batch_sizes = {1, 5, 10, 25, 50, 100};

  std::cout << "Batch Size | Total Time (ms) | Avg per Proof (ms)\n";
  std::cout << std::string(50, '-') << "\n";

  for (size_t batch_size : batch_sizes) {
    // Generate random indices
    std::vector<size_t> indices;
    for (size_t i = 0; i < batch_size; ++i) {
      indices.push_back(i * (leaf_count / batch_size));
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto proofs = tree.generate_batch_proofs(indices);
    auto end = std::chrono::high_resolution_clock::now();

    double total_time =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time = total_time / batch_size;

    std::cout << std::setw(10) << batch_size << " | " << std::setw(15)
              << std::fixed << std::setprecision(3) << total_time << " | "
              << std::setw(17) << avg_time << "\n";
  }

  std::cout << std::string(80, '=') << "\n";
}

TEST_F(MerkleTreeBenchmarkTest, ProofSizeAnalysis) {
  print_benchmark_header();
  std::cout << "Proof size analysis for different arities\n";
  std::cout << std::string(80, '-') << "\n";

  const size_t leaf_count = 1024;
  auto test_data = MerkleUtils::generate_test_leaves(leaf_count);

  std::cout << "Arity | Tree Height | Proof Levels | Siblings per Level | "
               "Total Proof Size (approx)\n";
  std::cout << std::string(80, '-') << "\n";

  for (size_t arity = 2; arity <= 8; ++arity) {
    NaryMerkleTree tree(test_data, MerkleTreeConfig(arity));
    auto proof = tree.generate_proof(0);

    if (proof) {
      size_t proof_levels = proof->path.size();
      size_t siblings_per_level = arity - 1;
      size_t total_elements = proof_levels * siblings_per_level;
      size_t approx_size = total_elements * sizeof(Poseidon::FieldElement);

      std::cout << std::setw(5) << arity << " | " << std::setw(11)
                << tree.get_tree_height() << " | " << std::setw(12)
                << proof_levels << " | " << std::setw(18) << siblings_per_level
                << " | " << std::setw(20) << approx_size << " bytes\n";
    }
  }

  std::cout << std::string(80, '=') << "\n";
}

TEST_F(MerkleTreeBenchmarkTest, TreeConstructionMethods) {
  print_benchmark_header();
  std::cout << "Comparing different tree construction patterns\n";
  std::cout << std::string(80, '-') << "\n";

  const size_t leaf_count = 2048;
  auto test_data = MerkleUtils::generate_test_leaves(leaf_count);

  // Test 1: Build tree all at once
  auto start = std::chrono::high_resolution_clock::now();
  NaryMerkleTree tree1(test_data);
  auto end = std::chrono::high_resolution_clock::now();
  auto build_time =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "Full construction:     " << std::setw(10) << std::fixed
            << std::setprecision(3) << build_time << "ms\n";

  // Test 2: Build tree incrementally
  start = std::chrono::high_resolution_clock::now();
  NaryMerkleTree tree2;
  for (const auto &leaf : test_data) {
    tree2.insert_leaf(leaf);
  }
  end = std::chrono::high_resolution_clock::now();
  auto incremental_time =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "Incremental construction: " << std::setw(7) << incremental_time
            << "ms";
  std::cout << " (slower by " << std::setprecision(1)
            << (incremental_time / build_time) << "x)\n";

  // Verify both trees have same root
  EXPECT_EQ(tree1.get_root_hash(), tree2.get_root_hash());

  std::cout << std::string(80, '=') << "\n";
}

// Stress test for very large trees
TEST_F(MerkleTreeBenchmarkTest, DISABLED_StressTestLargeTree) {
  print_benchmark_header();
  std::cout << "Stress test with very large dataset (disabled by default)\n";
  std::cout << std::string(80, '-') << "\n";

  const size_t large_size = 65536; // 64K leaves
  const size_t arity = 4;

  std::cout << "Building tree with " << large_size << " leaves...\n";

  auto result = MerkleUtils::benchmark_tree(large_size, arity, 10);
  print_benchmark_results(result);

  std::cout << "Stress test completed successfully!\n";
  std::cout << std::string(80, '=') << "\n";
}