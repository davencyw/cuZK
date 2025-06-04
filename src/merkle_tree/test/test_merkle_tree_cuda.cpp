#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../merkle_tree.hpp"
#include "../merkle_tree_cuda.cuh"

using namespace MerkleTree;
using namespace MerkleTree::MerkleTreeCUDA;

class MerkleTreeCUDATest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize CUDA
    if (!CudaNaryMerkleTree::initialize_cuda()) {
      GTEST_SKIP() << "CUDA not available or initialization failed";
    }

    // Check CUDA compatibility
    if (!CudaMerkleUtils::check_cuda_compatibility()) {
      GTEST_SKIP() << "CUDA device not compatible";
    }
  }

  void TearDown() override {
    CudaNaryMerkleTree::cleanup_cuda();
  }

  // Helper function to generate test leaves
  std::vector<FieldElement> generate_test_leaves(size_t count, uint64_t seed = 12345) {
    std::vector<FieldElement> leaves;
    leaves.reserve(count);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint64_t> dist(1, UINT64_MAX);

    for (size_t i = 0; i < count; ++i) {
      leaves.push_back(FieldElement(dist(gen)));
    }

    return leaves;
  }
};

TEST_F(MerkleTreeCUDATest, InitializationTest) {
  // Test CUDA initialization
  EXPECT_TRUE(CudaNaryMerkleTree::initialize_cuda());

  // Test optimal batch sizes
  size_t optimal_batch = CudaNaryMerkleTree::get_optimal_batch_size();
  size_t max_batch = CudaNaryMerkleTree::get_max_batch_size();

  EXPECT_GT(optimal_batch, 0);
  EXPECT_GT(max_batch, optimal_batch);

  std::cout << "Optimal batch size: " << optimal_batch << std::endl;
  std::cout << "Maximum batch size: " << max_batch << std::endl;
}

TEST_F(MerkleTreeCUDATest, BasicTreeBuildingTest) {
  // Test with small binary tree
  std::vector<FieldElement> leaves = generate_test_leaves(8);

  MerkleTreeConfig config(2);  // Binary tree
  CudaNaryMerkleTree cuda_tree(config);

  ASSERT_TRUE(cuda_tree.build_tree(leaves));

  EXPECT_EQ(cuda_tree.get_leaf_count(), 8);
  EXPECT_EQ(cuda_tree.get_arity(), 2);
  EXPECT_GT(cuda_tree.get_tree_height(), 0);

  FieldElement root = cuda_tree.get_root_hash();
  EXPECT_NE(root, FieldElement(0));

  std::cout << "CUDA Tree built successfully:" << std::endl;
  std::cout << "Root hash: " << root.to_hex() << std::endl;
}

TEST_F(MerkleTreeCUDATest, ConsistencyWithCPUTest) {
  // Test different tree configurations
  std::vector<size_t> arities = {2, 4, 8};
  std::vector<size_t> leaf_counts = {4, 16, 64, 256};

  for (size_t arity : arities) {
    for (size_t leaf_count : leaf_counts) {
      SCOPED_TRACE("Arity: " + std::to_string(arity) + ", Leaves: " + std::to_string(leaf_count));

      std::vector<FieldElement> leaves = generate_test_leaves(leaf_count, 42);
      MerkleTreeConfig config(arity);

      // Build CPU tree
      NaryMerkleTree cpu_tree(leaves, config);

      // Build CUDA tree
      CudaNaryMerkleTree cuda_tree(config);
      ASSERT_TRUE(cuda_tree.build_tree(leaves));

      // Compare root hashes
      EXPECT_EQ(cpu_tree.get_root_hash(), cuda_tree.get_root_hash())
          << "Root hashes differ for arity=" << arity << ", leaves=" << leaf_count;

      // Compare tree properties
      EXPECT_EQ(cpu_tree.get_leaf_count(), cuda_tree.get_leaf_count());
      EXPECT_EQ(cpu_tree.get_arity(), cuda_tree.get_arity());
      EXPECT_EQ(cpu_tree.get_tree_height(), cuda_tree.get_tree_height());

      // Test comparison function
      EXPECT_TRUE(cuda_tree.compare_with_cpu_tree(cpu_tree));
    }
  }
}

TEST_F(MerkleTreeCUDATest, ProofGenerationTest) {
  std::vector<FieldElement> leaves = generate_test_leaves(16);
  MerkleTreeConfig config(2);

  CudaNaryMerkleTree cuda_tree(leaves, config);

  // Generate proofs for various leaf indices
  std::vector<size_t> test_indices = {0, 1, 7, 15};

  for (size_t idx : test_indices) {
    SCOPED_TRACE("Leaf index: " + std::to_string(idx));

    auto proof = cuda_tree.generate_proof(idx);
    ASSERT_TRUE(proof.has_value());

    EXPECT_EQ(proof->leaf_index, idx);
    EXPECT_GT(proof->path.size(), 0);
    EXPECT_EQ(proof->path.size(), proof->indices.size());

    // Verify the proof
    EXPECT_TRUE(cuda_tree.verify_proof(*proof, leaves[idx]));
  }
}

TEST_F(MerkleTreeCUDATest, ProofConsistencyWithCPUTest) {
  std::vector<FieldElement> leaves = generate_test_leaves(32);
  MerkleTreeConfig config(4);  // Quaternary tree

  // Build both trees
  NaryMerkleTree cpu_tree(leaves, config);
  CudaNaryMerkleTree cuda_tree(leaves, config);

  // Test proof consistency
  std::vector<size_t> test_indices = {0, 5, 15, 31};

  for (size_t idx : test_indices) {
    SCOPED_TRACE("Leaf index: " + std::to_string(idx));

    auto cpu_proof = cpu_tree.generate_proof(idx);
    auto cuda_proof = cuda_tree.generate_proof(idx);

    ASSERT_TRUE(cpu_proof.has_value());
    ASSERT_TRUE(cuda_proof.has_value());

    // Verify both proofs work
    EXPECT_TRUE(cpu_tree.verify_proof(*cpu_proof, leaves[idx], cpu_tree.get_root_hash()));
    EXPECT_TRUE(cuda_tree.verify_proof(*cuda_proof, leaves[idx]));

    // Cross-verify (CPU proof should work with CUDA tree root)
    EXPECT_TRUE(cpu_tree.verify_proof(*cpu_proof, leaves[idx], cuda_tree.get_root_hash()));
  }
}

TEST_F(MerkleTreeCUDATest, BatchProofGenerationTest) {
  std::vector<FieldElement> leaves = generate_test_leaves(64);
  MerkleTreeConfig config(2);

  CudaNaryMerkleTree cuda_tree(leaves, config);

  // Generate batch proofs
  std::vector<size_t> indices = {0, 1, 15, 31, 32, 63};
  auto proofs = cuda_tree.generate_batch_proofs(indices);

  EXPECT_EQ(proofs.size(), indices.size());

  // Verify each proof individually
  for (size_t i = 0; i < proofs.size(); ++i) {
    EXPECT_TRUE(cuda_tree.verify_proof(proofs[i], leaves[indices[i]]));
  }

  // Test batch verification
  std::vector<FieldElement> batch_leaves;
  for (size_t idx : indices) {
    batch_leaves.push_back(leaves[idx]);
  }

  EXPECT_TRUE(cuda_tree.verify_batch_proofs(proofs, batch_leaves));
}

TEST_F(MerkleTreeCUDATest, LargeBatchProofVerificationTest) {
  // Test GPU-accelerated batch verification with large batches
  std::vector<FieldElement> leaves = generate_test_leaves(1024);
  MerkleTreeConfig config(4);

  CudaNaryMerkleTree cuda_tree(leaves, config);

  // Generate many proof indices
  std::vector<size_t> indices;
  for (size_t i = 0; i < 100; ++i) {
    indices.push_back(i * 10);  // Every 10th element
  }

  auto proofs = cuda_tree.generate_batch_proofs(indices);
  EXPECT_EQ(proofs.size(), indices.size());

  std::vector<FieldElement> batch_leaves;
  for (size_t idx : indices) {
    batch_leaves.push_back(leaves[idx]);
  }

  // This should use GPU acceleration for batch verification
  EXPECT_TRUE(cuda_tree.verify_batch_proofs(proofs, batch_leaves));
}

TEST_F(MerkleTreeCUDATest, BatchTreeBuildingTest) {
  // Test building multiple trees in batch
  size_t num_trees = 10;
  size_t leaves_per_tree = 32;
  MerkleTreeConfig config(2);

  auto batch_leaves = CudaMerkleUtils::generate_batch_test_leaves(num_trees, leaves_per_tree, 999);

  std::vector<CudaNaryMerkleTree> trees;
  ASSERT_TRUE(CudaNaryMerkleTree::build_batch_trees(batch_leaves, trees, config));

  EXPECT_EQ(trees.size(), num_trees);

  // Verify each tree
  for (size_t i = 0; i < num_trees; ++i) {
    EXPECT_EQ(trees[i].get_leaf_count(), leaves_per_tree);
    EXPECT_NE(trees[i].get_root_hash(), FieldElement(0));

    // Each tree should have a different root (since leaves are different)
    if (i > 0) {
      EXPECT_NE(trees[i].get_root_hash(), trees[i - 1].get_root_hash());
    }
  }
}

TEST_F(MerkleTreeCUDATest, EmptyTreeTest) {
  MerkleTreeConfig config(2);
  CudaNaryMerkleTree cuda_tree(config);

  std::vector<FieldElement> empty_leaves;
  EXPECT_TRUE(cuda_tree.build_tree(empty_leaves));

  EXPECT_EQ(cuda_tree.get_leaf_count(), 0);
  EXPECT_EQ(cuda_tree.get_tree_height(), 0);

  // Empty tree should have same root hash as CPU implementation
  NaryMerkleTree cpu_tree(empty_leaves, config);
  EXPECT_EQ(cuda_tree.get_root_hash(), cpu_tree.get_root_hash());
}

TEST_F(MerkleTreeCUDATest, PerformanceComparisonTest) {
  const size_t leaf_count = 1000;
  const size_t num_proofs = 100;

  std::vector<FieldElement> leaves = generate_test_leaves(leaf_count);
  MerkleTreeConfig config(4);

  // Build both trees and measure time
  auto start_cpu = std::chrono::high_resolution_clock::now();
  NaryMerkleTree cpu_tree(leaves, config);
  auto end_cpu = std::chrono::high_resolution_clock::now();

  auto start_cuda = std::chrono::high_resolution_clock::now();
  CudaNaryMerkleTree cuda_tree(leaves, config);
  auto end_cuda = std::chrono::high_resolution_clock::now();

  auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
  auto cuda_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda);

  std::cout << "Performance Comparison (Tree Building):" << std::endl;
  std::cout << "CPU time: " << cpu_time.count() << " μs" << std::endl;
  std::cout << "CUDA time: " << cuda_time.count() << " μs" << std::endl;

  if (cuda_time.count() > 0) {
    double speedup = static_cast<double>(cpu_time.count()) / cuda_time.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;
  }

  // Compare batch proof verification performance
  std::vector<size_t> indices;
  for (size_t i = 0; i < num_proofs; ++i) {
    indices.push_back(i % leaf_count);
  }

  auto cpu_proofs = cpu_tree.generate_batch_proofs(indices);
  auto cuda_proofs = cuda_tree.generate_batch_proofs(indices);

  std::vector<FieldElement> proof_leaves;
  for (size_t idx : indices) {
    proof_leaves.push_back(leaves[idx]);
  }

  // Time CPU batch verification
  start_cpu = std::chrono::high_resolution_clock::now();
  bool cpu_result =
      cpu_tree.verify_batch_proofs(cpu_proofs, proof_leaves, cpu_tree.get_root_hash());
  end_cpu = std::chrono::high_resolution_clock::now();

  // Time CUDA batch verification
  start_cuda = std::chrono::high_resolution_clock::now();
  bool cuda_result = cuda_tree.verify_batch_proofs(cuda_proofs, proof_leaves);
  end_cuda = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(cpu_result);
  EXPECT_TRUE(cuda_result);

  auto cpu_verify_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
  auto cuda_verify_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda);

  std::cout << "Performance Comparison (Batch Proof Verification):" << std::endl;
  std::cout << "CPU time: " << cpu_verify_time.count() << " μs" << std::endl;
  std::cout << "CUDA time: " << cuda_verify_time.count() << " μs" << std::endl;

  if (cuda_verify_time.count() > 0) {
    double verify_speedup = static_cast<double>(cpu_verify_time.count()) / cuda_verify_time.count();
    std::cout << "Verification speedup: " << verify_speedup << "x" << std::endl;
  }
}

TEST_F(MerkleTreeCUDATest, TreeHeightDebuggingTest) {
  // Debug test to understand tree height calculation differences
  std::vector<size_t> arities = {2, 4, 8};
  std::vector<size_t> leaf_counts = {4, 16, 64, 256};

  std::cout << "\n=== Tree Height Debugging ===" << std::endl;

  for (size_t arity : arities) {
    for (size_t leaf_count : leaf_counts) {
      std::cout << "\nArity: " << arity << ", Leaves: " << leaf_count << std::endl;

      // Method 1: CPU static method
      size_t cpu_static_height = NaryMerkleTree::calculate_tree_height(leaf_count, arity);
      std::cout << "  CPU static method: " << cpu_static_height << std::endl;

      // Method 2: CPU actual tree height
      std::vector<FieldElement> leaves = generate_test_leaves(leaf_count, 42);
      NaryMerkleTree cpu_tree(leaves, MerkleTreeConfig(arity));
      size_t cpu_actual_height = cpu_tree.get_tree_height();
      std::cout << "  CPU actual tree: " << cpu_actual_height << std::endl;

      // Method 3: Manual calculation like CPU proof generation
      size_t padded_size = 1;
      while (padded_size < leaf_count) {
        padded_size *= arity;
      }
      std::cout << "  Padded size: " << padded_size << std::endl;

      size_t proof_gen_height = 0;
      size_t temp = padded_size;
      while (temp > 1) {
        temp = (temp + arity - 1) / arity;  // Ceil division like CPU proof gen
        proof_gen_height++;
      }
      std::cout << "  CPU proof gen style (ceil): " << proof_gen_height << std::endl;

      // Method 4: CUDA style calculation
      size_t cuda_style_height = 0;
      temp = padded_size;
      while (temp > 1) {
        temp = temp / arity;  // Exact division like CUDA
        cuda_style_height++;
      }
      std::cout << "  CUDA style (exact): " << cuda_style_height << std::endl;

      // Test CUDA tree
      CudaNaryMerkleTree cuda_tree(leaves, MerkleTreeConfig(arity));
      std::cout << "  CUDA actual tree: " << cuda_tree.get_tree_height() << std::endl;

      std::cout << "  Root match: "
                << (cpu_tree.get_root_hash() == cuda_tree.get_root_hash() ? "YES" : "NO")
                << std::endl;
    }
  }
}

TEST_F(MerkleTreeCUDATest, BinaryTreeSpecificTest) {
  // Specific test for binary trees where we had height mismatches
  std::cout << "\n=== Binary Tree Specific Test ===" << std::endl;

  std::vector<size_t> leaf_counts = {4, 16, 64};

  for (size_t leaf_count : leaf_counts) {
    std::cout << "\nTesting binary tree with " << leaf_count << " leaves:" << std::endl;

    std::vector<FieldElement> leaves = generate_test_leaves(leaf_count, 42);
    MerkleTreeConfig config(2);  // Binary tree

    // Build CPU tree
    NaryMerkleTree cpu_tree(leaves, config);

    // Build CUDA tree
    CudaNaryMerkleTree cuda_tree(config);
    ASSERT_TRUE(cuda_tree.build_tree(leaves));

    std::cout << "  CPU tree height: " << cpu_tree.get_tree_height() << std::endl;
    std::cout << "  CUDA tree height: " << cuda_tree.get_tree_height() << std::endl;
    std::cout << "  CPU root: " << cpu_tree.get_root_hash().to_hex().substr(0, 16) << "..."
              << std::endl;
    std::cout << "  CUDA root: " << cuda_tree.get_root_hash().to_hex().substr(0, 16) << "..."
              << std::endl;

    // Verify heights match
    EXPECT_EQ(cpu_tree.get_tree_height(), cuda_tree.get_tree_height())
        << "Tree heights differ for " << leaf_count << " leaves";

    // Verify root hashes match
    EXPECT_EQ(cpu_tree.get_root_hash(), cuda_tree.get_root_hash())
        << "Root hashes differ for " << leaf_count << " leaves";

    // Verify tree properties
    EXPECT_EQ(cpu_tree.get_leaf_count(), cuda_tree.get_leaf_count());
    EXPECT_EQ(cpu_tree.get_arity(), cuda_tree.get_arity());

    // Test comparison function
    EXPECT_TRUE(cuda_tree.compare_with_cpu_tree(cpu_tree))
        << "Trees don't match for " << leaf_count << " leaves";

    std::cout << "  Result: "
              << (cuda_tree.get_root_hash() == cpu_tree.get_root_hash() ? "PASS" : "FAIL")
              << std::endl;
  }
}

TEST_F(MerkleTreeCUDATest, ProofGenerationAfterHeightFix) {
  // Test proof generation after tree height fix
  std::cout << "\n=== Proof Generation Test (Post-Fix) ===" << std::endl;

  std::vector<FieldElement> leaves = generate_test_leaves(8, 42);
  MerkleTreeConfig config(2);  // Binary tree

  // Build both trees
  NaryMerkleTree cpu_tree(leaves, config);
  CudaNaryMerkleTree cuda_tree(leaves, config);

  std::cout << "Tree properties:" << std::endl;
  std::cout << "  CPU height: " << cpu_tree.get_tree_height() << std::endl;
  std::cout << "  CUDA height: " << cuda_tree.get_tree_height() << std::endl;
  std::cout << "  Roots match: "
            << (cpu_tree.get_root_hash() == cuda_tree.get_root_hash() ? "YES" : "NO") << std::endl;

  // Test proof generation for a few indices
  std::vector<size_t> test_indices = {0, 3, 7};

  for (size_t idx : test_indices) {
    std::cout << "\nTesting proof for leaf " << idx << ":" << std::endl;

    auto cpu_proof = cpu_tree.generate_proof(idx);
    auto cuda_proof = cuda_tree.generate_proof(idx);

    ASSERT_TRUE(cpu_proof.has_value()) << "CPU proof generation failed for index " << idx;
    ASSERT_TRUE(cuda_proof.has_value()) << "CUDA proof generation failed for index " << idx;

    std::cout << "  CPU proof levels: " << cpu_proof->path.size() << std::endl;
    std::cout << "  CUDA proof levels: " << cuda_proof->path.size() << std::endl;

    // Verify proof structure
    EXPECT_EQ(cpu_proof->path.size(), cuda_proof->path.size())
        << "Proof path sizes differ for index " << idx;
    EXPECT_EQ(cpu_proof->indices.size(), cuda_proof->indices.size())
        << "Proof indices sizes differ for index " << idx;

    // Verify both proofs work
    EXPECT_TRUE(cpu_tree.verify_proof(*cpu_proof, leaves[idx], cpu_tree.get_root_hash()))
        << "CPU proof verification failed for index " << idx;
    EXPECT_TRUE(cuda_tree.verify_proof(*cuda_proof, leaves[idx]))
        << "CUDA proof verification failed for index " << idx;

    // Cross-verify: CPU proof should work with CUDA tree root
    EXPECT_TRUE(cpu_tree.verify_proof(*cpu_proof, leaves[idx], cuda_tree.get_root_hash()))
        << "CPU proof doesn't work with CUDA root for index " << idx;

    std::cout << "  Proof verification: "
              << (cuda_tree.verify_proof(*cuda_proof, leaves[idx]) ? "PASS" : "FAIL") << std::endl;
  }
}

TEST_F(MerkleTreeCUDATest, ComprehensiveValidationTest) {
  // Comprehensive test to validate all fixes for binary tree consistency
  std::cout << "\n=== Comprehensive Validation Test ===" << std::endl;

  struct TestCase {
    size_t arity;
    size_t leaves;
    std::string description;
  };

  std::vector<TestCase> test_cases = {{2, 4, "Binary tree with 4 leaves (edge case)"},
                                      {2, 16, "Binary tree with 16 leaves"},
                                      {2, 64, "Binary tree with 64 leaves"},
                                      {2, 15, "Binary tree with 15 leaves (non-power-of-2)"},
                                      {4, 16, "Quaternary tree (should work)"},
                                      {8, 64, "Octal tree (should work)"}};

  for (const auto& test_case : test_cases) {
    std::cout << "\nTesting: " << test_case.description << std::endl;

    std::vector<FieldElement> leaves = generate_test_leaves(test_case.leaves, 42);
    MerkleTreeConfig config(test_case.arity);

    // Build both trees
    NaryMerkleTree cpu_tree(leaves, config);
    CudaNaryMerkleTree cuda_tree(leaves, config);

    // Log details
    std::cout << "  CPU tree height: " << cpu_tree.get_tree_height() << std::endl;
    std::cout << "  CUDA tree height: " << cuda_tree.get_tree_height() << std::endl;
    std::cout << "  Heights match: "
              << (cpu_tree.get_tree_height() == cuda_tree.get_tree_height() ? "YES" : "NO")
              << std::endl;
    std::cout << "  Roots match: "
              << (cpu_tree.get_root_hash() == cuda_tree.get_root_hash() ? "YES" : "NO")
              << std::endl;

    // Critical validations
    EXPECT_EQ(cpu_tree.get_tree_height(), cuda_tree.get_tree_height())
        << "Tree heights differ for " << test_case.description;
    EXPECT_EQ(cpu_tree.get_root_hash(), cuda_tree.get_root_hash())
        << "Root hashes differ for " << test_case.description;
    EXPECT_EQ(cpu_tree.get_leaf_count(), cuda_tree.get_leaf_count())
        << "Leaf counts differ for " << test_case.description;
    EXPECT_EQ(cpu_tree.get_arity(), cuda_tree.get_arity())
        << "Arities differ for " << test_case.description;

    // Test proof generation and verification
    if (test_case.leaves > 0) {
      size_t test_index = std::min(test_case.leaves - 1, size_t(3));
      auto cpu_proof = cpu_tree.generate_proof(test_index);
      auto cuda_proof = cuda_tree.generate_proof(test_index);

      if (cpu_proof && cuda_proof) {
        std::cout << "  Proof structure: CPU=" << cpu_proof->path.size()
                  << " levels, CUDA=" << cuda_proof->path.size() << " levels" << std::endl;

        EXPECT_EQ(cpu_proof->path.size(), cuda_proof->path.size())
            << "Proof path sizes differ for " << test_case.description;

        bool cpu_verify =
            cpu_tree.verify_proof(*cpu_proof, leaves[test_index], cpu_tree.get_root_hash());
        bool cuda_verify = cuda_tree.verify_proof(*cuda_proof, leaves[test_index]);
        bool cross_verify =
            cpu_tree.verify_proof(*cpu_proof, leaves[test_index], cuda_tree.get_root_hash());

        std::cout << "  Proof verification: CPU=" << (cpu_verify ? "PASS" : "FAIL")
                  << ", CUDA=" << (cuda_verify ? "PASS" : "FAIL")
                  << ", Cross=" << (cross_verify ? "PASS" : "FAIL") << std::endl;

        EXPECT_TRUE(cpu_verify) << "CPU proof verification failed for " << test_case.description;
        EXPECT_TRUE(cuda_verify) << "CUDA proof verification failed for " << test_case.description;
        EXPECT_TRUE(cross_verify) << "Cross proof verification failed for "
                                  << test_case.description;
      }
    }

    // Overall result
    bool overall_success = (cpu_tree.get_tree_height() == cuda_tree.get_tree_height()) &&
                           (cpu_tree.get_root_hash() == cuda_tree.get_root_hash());
    std::cout << "  Overall result: " << (overall_success ? "PASS" : "FAIL") << std::endl;

    EXPECT_TRUE(overall_success) << "Overall validation failed for " << test_case.description;
  }
}

TEST_F(MerkleTreeCUDATest, DetailedLevelDebuggingTest) {
  // Detailed level-by-level debugging for binary tree with 4 leaves
  std::cout << "\n=== Detailed Level Debugging ===" << std::endl;

  std::vector<FieldElement> leaves = generate_test_leaves(4, 42);
  MerkleTreeConfig config(2);  // Binary tree

  std::cout << "Input leaves:" << std::endl;
  for (size_t i = 0; i < leaves.size(); ++i) {
    std::cout << "  Leaf[" << i << "]: " << leaves[i].to_hex().substr(0, 16) << "..." << std::endl;
  }

  // Build CPU tree
  NaryMerkleTree cpu_tree(leaves, config);

  // Build CUDA tree
  CudaNaryMerkleTree cuda_tree(leaves, config);

  std::cout << "\nTree properties:" << std::endl;
  std::cout << "  CPU height: " << cpu_tree.get_tree_height() << std::endl;
  std::cout << "  CUDA height: " << cuda_tree.get_tree_height() << std::endl;

  // Get CUDA tree levels for detailed comparison
  const auto& cuda_levels = cuda_tree.get_tree_levels();

  std::cout << "\nCUDA tree levels (" << cuda_levels.size() << " levels):" << std::endl;
  for (size_t level = 0; level < cuda_levels.size(); ++level) {
    std::cout << "  Level " << level << " (" << cuda_levels[level].size()
              << " elements):" << std::endl;
    for (size_t i = 0; i < cuda_levels[level].size() && i < 8; ++i) {
      std::cout << "    [" << i << "]: " << cuda_levels[level][i].to_hex().substr(0, 16) << "..."
                << std::endl;
    }
  }

  std::cout << "\nManual CPU tree construction simulation:" << std::endl;

  // Simulate CPU tree construction step by step
  std::vector<FieldElement> current_level = leaves;

  // Pad to next power of arity (like CPU does)
  size_t padded_size = 1;
  while (padded_size < leaves.size()) {
    padded_size *= config.arity;
  }

  FieldElement empty_hash = NaryMerkleTree::compute_empty_hash(config.arity);
  std::cout << "  Empty hash: " << empty_hash.to_hex().substr(0, 16) << "..." << std::endl;

  while (current_level.size() < padded_size) {
    current_level.push_back(empty_hash);
  }

  std::cout << "  Level 0 (padded leaves, " << current_level.size() << " elements):" << std::endl;
  for (size_t i = 0; i < current_level.size(); ++i) {
    std::cout << "    [" << i << "]: " << current_level[i].to_hex().substr(0, 16) << "..."
              << std::endl;
  }

  // Compare with CUDA level 0
  bool level0_match = true;
  if (cuda_levels.size() > 0 && cuda_levels[0].size() == current_level.size()) {
    for (size_t i = 0; i < current_level.size(); ++i) {
      if (current_level[i] != cuda_levels[0][i]) {
        level0_match = false;
        std::cout << "    MISMATCH at [" << i
                  << "]: CPU=" << current_level[i].to_hex().substr(0, 16)
                  << "..., CUDA=" << cuda_levels[0][i].to_hex().substr(0, 16) << "..." << std::endl;
      }
    }
  } else {
    level0_match = false;
  }
  std::cout << "  Level 0 match: " << (level0_match ? "YES" : "NO") << std::endl;

  // Build next level manually like CPU does
  size_t level_num = 1;
  while (current_level.size() > 1) {
    std::vector<FieldElement> next_level;

    std::cout << "  Building level " << level_num << " from " << current_level.size()
              << " inputs:" << std::endl;

    for (size_t i = 0; i < current_level.size(); i += config.arity) {
      std::vector<FieldElement> children;

      // Collect children
      for (size_t j = 0; j < config.arity && i + j < current_level.size(); ++j) {
        children.push_back(current_level[i + j]);
      }

      // Pad if needed
      while (children.size() < config.arity) {
        children.push_back(empty_hash);
      }

      std::cout << "    Group " << (i / config.arity) << ": ";
      for (size_t k = 0; k < children.size(); ++k) {
        std::cout << children[k].to_hex().substr(0, 8) << (k < children.size() - 1 ? ", " : "");
      }

      // Hash using the same method as CPU
      FieldElement parent_hash = Poseidon::PoseidonHash::hash_multiple(children);
      next_level.push_back(parent_hash);

      std::cout << " -> " << parent_hash.to_hex().substr(0, 16) << "..." << std::endl;
    }

    // Compare with CUDA level
    bool level_match = true;
    if (cuda_levels.size() > level_num && cuda_levels[level_num].size() == next_level.size()) {
      for (size_t i = 0; i < next_level.size(); ++i) {
        if (next_level[i] != cuda_levels[level_num][i]) {
          level_match = false;
          std::cout << "    LEVEL " << level_num << " MISMATCH at [" << i
                    << "]: CPU=" << next_level[i].to_hex().substr(0, 16)
                    << "..., CUDA=" << cuda_levels[level_num][i].to_hex().substr(0, 16) << "..."
                    << std::endl;
        }
      }
    } else {
      level_match = false;
      std::cout << "    LEVEL " << level_num << " SIZE MISMATCH: CPU=" << next_level.size();
      if (cuda_levels.size() > level_num) {
        std::cout << ", CUDA=" << cuda_levels[level_num].size();
      } else {
        std::cout << ", CUDA=N/A";
      }
      std::cout << std::endl;
    }
    std::cout << "  Level " << level_num << " match: " << (level_match ? "YES" : "NO") << std::endl;

    current_level = next_level;
    level_num++;
  }

  std::cout << "\nFinal comparison:" << std::endl;
  std::cout << "  Manual CPU root: " << current_level[0].to_hex().substr(0, 16) << "..."
            << std::endl;
  std::cout << "  CPU tree root: " << cpu_tree.get_root_hash().to_hex().substr(0, 16) << "..."
            << std::endl;
  std::cout << "  CUDA tree root: " << cuda_tree.get_root_hash().to_hex().substr(0, 16) << "..."
            << std::endl;
  std::cout << "  Manual vs CPU: "
            << (current_level[0] == cpu_tree.get_root_hash() ? "MATCH" : "DIFFER") << std::endl;
  std::cout << "  Manual vs CUDA: "
            << (current_level[0] == cuda_tree.get_root_hash() ? "MATCH" : "DIFFER") << std::endl;
  std::cout << "  CPU vs CUDA: "
            << (cpu_tree.get_root_hash() == cuda_tree.get_root_hash() ? "MATCH" : "DIFFER")
            << std::endl;
}