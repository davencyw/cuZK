#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "merkle_tree.hpp"

using namespace MerkleTree;

class MerkleTreeTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Poseidon constants before running tests
    Poseidon::PoseidonConstants::init();
    Poseidon::FieldConstants::init();
  }

  std::vector<FieldElement> generate_test_data(size_t count,
                                               uint64_t seed = 42) {
    std::vector<FieldElement> data;
    data.reserve(count);

    std::mt19937_64 gen(seed);
    for (size_t i = 0; i < count; ++i) {
      uint64_t value = gen();
      data.emplace_back(value);
    }

    return data;
  }
};

// Basic functionality tests
TEST_F(MerkleTreeTest, BasicConstructionAndOperations) {
  // Test empty tree
  NaryMerkleTree empty_tree;
  EXPECT_EQ(empty_tree.get_leaf_count(), 0);
  EXPECT_EQ(empty_tree.get_arity(), 2); // Default arity

  // Test tree with data
  auto test_data = generate_test_data(8);
  NaryMerkleTree tree(test_data);

  EXPECT_EQ(tree.get_leaf_count(), 8);
  EXPECT_EQ(tree.get_arity(), 2);
  EXPECT_GT(tree.get_tree_height(), 0);

  // Root hash should be deterministic
  auto root1 = tree.get_root_hash();
  NaryMerkleTree tree2(test_data);
  auto root2 = tree2.get_root_hash();
  EXPECT_EQ(root1, root2);
}

TEST_F(MerkleTreeTest, ConfigurationTest) {
  auto test_data = generate_test_data(16);

  // Test different arities
  for (size_t arity = 2; arity <= 8; ++arity) {
    MerkleTreeConfig config(arity);
    NaryMerkleTree tree(test_data, config);

    EXPECT_EQ(tree.get_arity(), arity);
    EXPECT_EQ(tree.get_leaf_count(), 16);
    EXPECT_GT(tree.get_tree_height(), 0);

    // Higher arity should result in shorter trees
    if (arity > 2) {
      NaryMerkleTree binary_tree(test_data, MerkleTreeConfig(2));
      EXPECT_LE(tree.get_tree_height(), binary_tree.get_tree_height());
    }
  }
}

TEST_F(MerkleTreeTest, InvalidConfiguration) {
  // Test invalid arity values
  EXPECT_THROW(MerkleTreeConfig(1), std::invalid_argument);  // Too small
  EXPECT_THROW(MerkleTreeConfig(10), std::invalid_argument); // Too large
}

// Proof generation and verification tests
TEST_F(MerkleTreeTest, ProofGenerationAndVerification) {
  auto test_data = generate_test_data(8);
  NaryMerkleTree tree(test_data);
  auto root_hash = tree.get_root_hash();

  // Test proof generation for each leaf
  for (size_t i = 0; i < test_data.size(); ++i) {
    auto proof = tree.generate_proof(i);
    ASSERT_TRUE(proof.has_value());

    // Verify the proof
    bool valid = tree.verify_proof(*proof, test_data[i], root_hash);
    EXPECT_TRUE(valid);

    // Test with wrong leaf value
    auto wrong_value = Poseidon::FieldElement(999999);
    bool invalid = tree.verify_proof(*proof, wrong_value, root_hash);
    EXPECT_FALSE(invalid);
  }
}

TEST_F(MerkleTreeTest, ProofGenerationOutOfBounds) {
  auto test_data = generate_test_data(5);
  NaryMerkleTree tree(test_data);

  // Test out of bounds index
  auto proof = tree.generate_proof(10);
  EXPECT_FALSE(proof.has_value());
}

TEST_F(MerkleTreeTest, DifferentArityProofs) {
  auto test_data = generate_test_data(27); // 3^3, good for arity 3

  for (size_t arity = 2; arity <= 4; ++arity) {
    MerkleTreeConfig config(arity);
    NaryMerkleTree tree(test_data, config);
    auto root_hash = tree.get_root_hash();

    // Test several random indices
    for (size_t i : {0, 5, 10, 20, 26}) {
      auto proof = tree.generate_proof(i);
      ASSERT_TRUE(proof.has_value());

      bool valid = tree.verify_proof(*proof, test_data[i], root_hash);
      EXPECT_TRUE(valid) << "Failed for arity " << arity << " at index " << i;
    }
  }
}

// Tree modification tests
TEST_F(MerkleTreeTest, LeafInsertion) {
  auto test_data = generate_test_data(4);
  NaryMerkleTree tree(test_data);

  auto original_count = tree.get_leaf_count();
  auto original_root = tree.get_root_hash();

  // Insert new leaf
  auto new_leaf = Poseidon::FieldElement(12345);
  tree.insert_leaf(new_leaf);

  EXPECT_EQ(tree.get_leaf_count(), original_count + 1);
  EXPECT_NE(tree.get_root_hash(), original_root); // Root should change

  // Verify the new leaf can be proven
  auto proof = tree.generate_proof(original_count);
  ASSERT_TRUE(proof.has_value());

  bool valid = tree.verify_proof(*proof, new_leaf, tree.get_root_hash());
  EXPECT_TRUE(valid);
}

TEST_F(MerkleTreeTest, LeafUpdate) {
  auto test_data = generate_test_data(8);
  NaryMerkleTree tree(test_data);

  auto original_root = tree.get_root_hash();
  auto new_value = Poseidon::FieldElement(999999);

  // Update a leaf
  tree.update_leaf(3, new_value);

  EXPECT_NE(tree.get_root_hash(), original_root);

  // Verify the updated leaf
  auto proof = tree.generate_proof(3);
  ASSERT_TRUE(proof.has_value());

  bool valid = tree.verify_proof(*proof, new_value, tree.get_root_hash());
  EXPECT_TRUE(valid);

  // Test out of bounds update
  EXPECT_THROW(tree.update_leaf(100, new_value), std::out_of_range);
}

// Batch operations tests
TEST_F(MerkleTreeTest, BatchProofGeneration) {
  auto test_data = generate_test_data(16);
  NaryMerkleTree tree(test_data);
  auto root_hash = tree.get_root_hash();

  std::vector<size_t> indices = {0, 3, 7, 12, 15};
  auto proofs = tree.generate_batch_proofs(indices);

  EXPECT_EQ(proofs.size(), indices.size());

  // Verify each proof individually
  for (size_t i = 0; i < proofs.size(); ++i) {
    size_t leaf_index = indices[i];
    bool valid = tree.verify_proof(proofs[i], test_data[leaf_index], root_hash);
    EXPECT_TRUE(valid) << "Failed batch proof at index " << leaf_index;
  }

  // Verify batch
  std::vector<FieldElement> leaf_values;
  for (size_t idx : indices) {
    leaf_values.push_back(test_data[idx]);
  }

  bool batch_valid = tree.verify_batch_proofs(proofs, leaf_values, root_hash);
  EXPECT_TRUE(batch_valid);
}

// Utility function tests
TEST_F(MerkleTreeTest, StaticUtilities) {
  // Test tree height calculation
  EXPECT_EQ(NaryMerkleTree::calculate_tree_height(1, 2), 1);
  EXPECT_EQ(NaryMerkleTree::calculate_tree_height(2, 2), 2);
  EXPECT_EQ(NaryMerkleTree::calculate_tree_height(4, 2), 3);
  EXPECT_EQ(NaryMerkleTree::calculate_tree_height(8, 2), 4);

  // Test for ternary tree
  EXPECT_EQ(NaryMerkleTree::calculate_tree_height(9, 3), 3);
  EXPECT_EQ(NaryMerkleTree::calculate_tree_height(27, 3), 4);

  // Test max leaves calculation
  EXPECT_EQ(NaryMerkleTree::calculate_max_leaves(3, 2), 4);  // 2^(3-1) = 4
  EXPECT_EQ(NaryMerkleTree::calculate_max_leaves(4, 3), 27); // 3^(4-1) = 27

  // Test empty hash is deterministic
  auto empty1 = NaryMerkleTree::compute_empty_hash(2);
  auto empty2 = NaryMerkleTree::compute_empty_hash(2);
  EXPECT_EQ(empty1, empty2);

  auto empty3 = NaryMerkleTree::compute_empty_hash(3);
  EXPECT_NE(empty1, empty3); // Different arity should give different hash
}

// MerkleUtils tests
TEST_F(MerkleTreeTest, ProofValidation) {
  auto test_data = generate_test_data(8);
  NaryMerkleTree tree(test_data, MerkleTreeConfig(3)); // Ternary tree

  auto proof = tree.generate_proof(2);
  ASSERT_TRUE(proof.has_value());

  // Test valid proof structure
  bool valid_structure = MerkleUtils::is_valid_proof_structure(*proof, 3);
  EXPECT_TRUE(valid_structure);

  // Test with wrong arity
  bool invalid_structure = MerkleUtils::is_valid_proof_structure(*proof, 2);
  EXPECT_FALSE(invalid_structure);
}

TEST_F(MerkleTreeTest, TreeComparison) {
  auto test_data = generate_test_data(10);

  NaryMerkleTree tree1(test_data);
  NaryMerkleTree tree2(test_data);
  NaryMerkleTree tree3(test_data, MerkleTreeConfig(3)); // Different arity

  EXPECT_TRUE(MerkleUtils::compare_trees(tree1, tree2));
  EXPECT_FALSE(MerkleUtils::compare_trees(tree1, tree3));
}

TEST_F(MerkleTreeTest, TestDataGeneration) {
  auto test_leaves = MerkleUtils::generate_test_leaves(100, 42);
  EXPECT_EQ(test_leaves.size(), 100);

  // Same seed should generate same data
  auto test_leaves2 = MerkleUtils::generate_test_leaves(100, 42);
  EXPECT_EQ(test_leaves, test_leaves2);

  // Different seed should generate different data
  auto test_leaves3 = MerkleUtils::generate_test_leaves(100, 43);
  EXPECT_NE(test_leaves, test_leaves3);
}

// Performance and stress tests
TEST_F(MerkleTreeTest, LargeTreeTest) {
  // Test with larger dataset
  auto test_data = generate_test_data(1000);
  NaryMerkleTree tree(test_data, MerkleTreeConfig(4)); // Quaternary tree

  EXPECT_EQ(tree.get_leaf_count(), 1000);
  EXPECT_EQ(tree.get_arity(), 4);

  // Test random proofs
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 999);

  auto root_hash = tree.get_root_hash();

  for (int i = 0; i < 10; ++i) {
    size_t random_index = dis(gen);
    auto proof = tree.generate_proof(random_index);
    ASSERT_TRUE(proof.has_value());

    bool valid = tree.verify_proof(*proof, test_data[random_index], root_hash);
    EXPECT_TRUE(valid) << "Failed for index " << random_index;
  }
}

TEST_F(MerkleTreeTest, EdgeCases) {
  // Single element tree
  std::vector<FieldElement> single = {Poseidon::FieldElement(42)};
  NaryMerkleTree tree(single);

  EXPECT_EQ(tree.get_leaf_count(), 1);
  EXPECT_EQ(tree.get_tree_height(), 1);

  auto proof = tree.generate_proof(0);
  ASSERT_TRUE(proof.has_value());

  bool valid = tree.verify_proof(*proof, single[0], tree.get_root_hash());
  EXPECT_TRUE(valid);

  // Empty tree
  NaryMerkleTree empty_tree;
  auto empty_proof = empty_tree.generate_proof(0);
  EXPECT_FALSE(empty_proof.has_value());
}

// Test printing functionality (mainly for coverage)
TEST_F(MerkleTreeTest, TreePrinting) {
  auto test_data = generate_test_data(4);
  NaryMerkleTree tree(test_data);

  // This mainly tests that print_tree doesn't crash
  testing::internal::CaptureStdout();
  tree.print_tree();
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(output.empty());
  EXPECT_TRUE(output.find("Merkle Tree") != std::string::npos);
}