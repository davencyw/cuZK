#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../common/error_handling.hpp"
#include "../common/namespace_utils.hpp"
#include "../poseidon/poseidon.hpp"

namespace MerkleTree {

using FieldElement = Poseidon::FieldElement;

// Configuration for the n-ary Merkle tree
struct MerkleTreeConfig {
  static constexpr size_t DEFAULT_ARITY = 2;         // Binary tree by default
  static constexpr size_t MIN_ARITY = 2;             // Minimum branching factor
  static constexpr size_t MAX_ARITY = 8;             // Maximum branching factor
  static constexpr size_t DEFAULT_TREE_HEIGHT = 20;  // Default height for empty tree

  size_t arity;
  size_t tree_height;

  explicit MerkleTreeConfig(size_t arity = DEFAULT_ARITY, size_t tree_height = DEFAULT_TREE_HEIGHT)
      : arity(arity), tree_height(tree_height) {
    cuZK::ErrorHandling::validate_range(arity, MIN_ARITY, MAX_ARITY, "arity");
  }
};

// Merkle tree node structure
struct MerkleNode {
  FieldElement hash;
  std::vector<std::shared_ptr<MerkleNode>> children;
  bool is_leaf;

  explicit MerkleNode(const FieldElement& value, bool leaf = false) : hash(value), is_leaf(leaf) {}
};

// Merkle proof structure
struct MerkleProof {
  std::vector<std::vector<FieldElement>> path;  // Sibling hashes at each level
  std::vector<size_t> indices;                  // Position indices at each level
  size_t leaf_index;                            // Index of the leaf in the tree

  MerkleProof() : leaf_index(0) {}
};

// Main N-ary Merkle Tree class
class NaryMerkleTree {
 private:
  MerkleTreeConfig config_;
  std::shared_ptr<MerkleNode> root_;
  std::vector<FieldElement> leaves_;
  size_t leaf_count_;

  std::shared_ptr<MerkleNode> build_tree_bottom_up(const std::vector<FieldElement>& leaves);

  FieldElement compute_internal_hash(const std::vector<FieldElement>& children_hashes) const;

  bool generate_bottom_up_proof(size_t leaf_index, MerkleProof& proof) const;

 public:
  // Constructors
  explicit NaryMerkleTree(const MerkleTreeConfig& config = MerkleTreeConfig());
  explicit NaryMerkleTree(const std::vector<FieldElement>& leaves,
                          const MerkleTreeConfig& config = MerkleTreeConfig());

  // Tree operations
  void build_tree(const std::vector<FieldElement>& leaves);
  void insert_leaf(const FieldElement& leaf);
  void update_leaf(size_t index, const FieldElement& new_value);

  // Proof operations
  std::optional<MerkleProof> generate_proof(size_t leaf_index) const;
  bool verify_proof(const MerkleProof& proof,
                    const FieldElement& leaf_value,
                    const FieldElement& root_hash) const;

  // Batch operations
  std::vector<MerkleProof> generate_batch_proofs(const std::vector<size_t>& indices) const;
  bool verify_batch_proofs(const std::vector<MerkleProof>& proofs,
                           const std::vector<FieldElement>& leaf_values,
                           const FieldElement& root_hash) const;

  // Getters
  FieldElement get_root_hash() const;
  size_t get_leaf_count() const {
    return leaf_count_;
  }
  size_t get_tree_height() const;
  size_t get_arity() const {
    return config_.arity;
  }
  const std::vector<FieldElement>& get_leaves() const {
    return leaves_;
  }

  // Utility functions
  void print_tree() const;

  // Static utilities
  static FieldElement compute_empty_hash(size_t arity);
  static size_t calculate_tree_height(size_t leaf_count, size_t arity);
  static size_t calculate_max_leaves(size_t height, size_t arity);

 private:
  void print_tree_recursive(const std::shared_ptr<MerkleNode>& node,
                            size_t level,
                            const std::string& prefix) const;
};

// Merkle tree utilities
namespace MerkleUtils {
// Proof validation utilities
bool is_valid_proof_structure(const MerkleProof& proof, size_t arity);

// Tree comparison utilities
bool compare_trees(const NaryMerkleTree& tree1, const NaryMerkleTree& tree2);

// Performance testing
struct TreeBenchmarkResult {
  double build_time_ms;
  double proof_generation_time_ms;
  double proof_verification_time_ms;
  size_t tree_height;
  size_t leaf_count;
  size_t arity;
};

TreeBenchmarkResult benchmark_tree(size_t leaf_count, size_t arity, size_t num_proofs = 100);

// Test data generation
std::vector<FieldElement> generate_test_leaves(size_t count, uint64_t seed = 0);

}  // namespace MerkleUtils

}  // namespace MerkleTree