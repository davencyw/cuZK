#include "merkle_tree.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

USING_CUZK_TYPES()
USING_FIELD_CONSTANTS()

namespace MerkleTree {

// NaryMerkleTree constructors
NaryMerkleTree::NaryMerkleTree(const MerkleTreeConfig &config)
    : config_(config), leaf_count_(0) {
  // Validation is now done in MerkleTreeConfig constructor
}

NaryMerkleTree::NaryMerkleTree(const std::vector<FieldElement> &leaves,
                               const MerkleTreeConfig &config)
    : NaryMerkleTree(config) {
  build_tree(leaves);
}

// Tree building
void NaryMerkleTree::build_tree(const std::vector<FieldElement> &leaves) {
  if (leaves.empty()) {
    root_ = nullptr;
    leaves_.clear();
    leaf_count_ = 0;
    return;
  }

  leaves_ = leaves;
  leaf_count_ = leaves.size();

  // Use bottom-up approach for all arities to ensure all leaves at same level
  root_ = build_tree_bottom_up(leaves);
}

std::shared_ptr<MerkleNode>
NaryMerkleTree::build_tree_bottom_up(const std::vector<FieldElement> &leaves) {
  // Create leaf nodes
  std::vector<std::shared_ptr<MerkleNode>> current_level;

  // Pad to next power of arity
  size_t padded_size = 1;
  while (padded_size < leaves.size()) {
    padded_size *= config_.arity;
  }

  // Create leaf nodes (pad with empty hash if needed)
  current_level.reserve(padded_size);
  for (size_t i = 0; i < leaves.size(); ++i) {
    current_level.push_back(std::make_shared<MerkleNode>(leaves[i], true));
  }
  for (size_t i = leaves.size(); i < padded_size; ++i) {
    auto empty_hash = compute_empty_hash(config_.arity);
    current_level.push_back(std::make_shared<MerkleNode>(empty_hash, true));
  }

  // Build tree bottom-up
  while (current_level.size() > 1) {
    std::vector<std::shared_ptr<MerkleNode>> next_level;
    next_level.reserve((current_level.size() + config_.arity - 1) /
                       config_.arity);

    for (size_t i = 0; i < current_level.size(); i += config_.arity) {
      std::vector<FieldElement> child_hashes;
      std::vector<std::shared_ptr<MerkleNode>> children;

      // Collect up to arity children
      for (size_t j = 0; j < config_.arity && i + j < current_level.size();
           ++j) {
        children.push_back(current_level[i + j]);
        child_hashes.push_back(current_level[i + j]->hash);
      }

      // Pad to arity if needed
      while (children.size() < config_.arity) {
        auto empty_hash = compute_empty_hash(config_.arity);
        children.push_back(std::make_shared<MerkleNode>(empty_hash, true));
        child_hashes.push_back(empty_hash);
      }

      FieldElement internal_hash = compute_internal_hash(child_hashes);
      auto internal_node = std::make_shared<MerkleNode>(internal_hash, false);
      internal_node->children = children;

      next_level.push_back(internal_node);
    }

    current_level = next_level;
  }

  return current_level[0];
}

FieldElement NaryMerkleTree::compute_internal_hash(
    const std::vector<FieldElement> &children_hashes) const {
  if (children_hashes.empty()) {
    return compute_empty_hash(config_.arity);
  }

  // Use Poseidon hash for multiple inputs
  return Hash::hash_multiple(children_hashes);
}

// Proof generation
std::optional<MerkleProof>
NaryMerkleTree::generate_proof(size_t leaf_index) const {
  if (!root_ || leaf_index >= leaf_count_) {
    return std::nullopt;
  }

  MerkleProof proof;
  proof.leaf_index = leaf_index;

  // Use direct indexing for bottom-up structure
  if (generate_bottom_up_proof(leaf_index, proof)) {
    return proof;
  }

  return std::nullopt;
}

bool NaryMerkleTree::generate_bottom_up_proof(size_t leaf_index,
                                              MerkleProof &proof) const {
  // Calculate tree height (number of levels from leaf to root)
  size_t padded_size = 1;
  while (padded_size < leaf_count_) {
    padded_size *= config_.arity;
  }

  size_t tree_height = 0;
  size_t temp = padded_size;
  while (temp > 1) {
    temp = (temp + config_.arity - 1) / config_.arity;
    tree_height++;
  }

  // Build proof by walking from root down to leaf, collecting siblings along
  // the way
  auto current_node = root_;
  size_t current_index = leaf_index;

  // Collect siblings from root to leaf, then reverse for correct order
  std::vector<size_t> temp_indices;
  std::vector<std::vector<FieldElement>> temp_path;

  for (size_t depth = 0; depth < tree_height; ++depth) {
    if (!current_node || current_node->children.size() != config_.arity) {
      return false;
    }

    // Calculate which child to follow at this level
    size_t elements_at_level = padded_size;
    for (size_t i = 0; i < depth; ++i) {
      elements_at_level =
          (elements_at_level + config_.arity - 1) / config_.arity;
    }

    size_t elements_per_child = elements_at_level / config_.arity;
    size_t child_index = current_index / elements_per_child;

    if (child_index >= config_.arity) {
      child_index = config_.arity - 1;
    }

    // Store the position for this level
    temp_indices.push_back(child_index);

    // Get the sibling hashes
    std::vector<FieldElement> siblings;
    for (size_t i = 0; i < config_.arity; ++i) {
      if (i != child_index) {
        if (current_node->children[i]) {
          siblings.push_back(current_node->children[i]->hash);
        } else {
          siblings.push_back(compute_empty_hash(config_.arity));
        }
      }
    }
    temp_path.push_back(siblings);

    // Move to the next level
    current_node = current_node->children[child_index];
    if (!current_node) {
      return false;
    }

    // Update index for next level
    current_index %= elements_per_child;
  }

  // Verify we reached the correct leaf
  if (!current_node->is_leaf) {
    return false;
  }

  // Reverse the order so proof goes from leaf level to root level
  for (int i = temp_indices.size() - 1; i >= 0; --i) {
    proof.indices.push_back(temp_indices[i]);
    proof.path.push_back(temp_path[i]);
  }

  return true;
}

// Proof verification
bool NaryMerkleTree::verify_proof(const MerkleProof &proof,
                                  const FieldElement &leaf_value,
                                  const FieldElement &root_hash) const {
  if (proof.path.size() != proof.indices.size()) {
    return false;
  }

  FieldElement current_hash = leaf_value;

  // Work our way up the tree
  for (size_t level = 0; level < proof.path.size(); ++level) {
    const auto &siblings = proof.path[level];
    size_t position = proof.indices[level];

    if (position >= config_.arity || siblings.size() != config_.arity - 1) {
      return false;
    }

    // Reconstruct the parent hash
    std::vector<FieldElement> children_hashes;
    size_t sibling_idx = 0;

    for (size_t i = 0; i < config_.arity; ++i) {
      if (i == position) {
        children_hashes.push_back(current_hash);
      } else {
        if (sibling_idx < siblings.size()) {
          children_hashes.push_back(siblings[sibling_idx]);
          sibling_idx++;
        } else {
          // Pad with empty hash if we run out of siblings
          children_hashes.push_back(compute_empty_hash(config_.arity));
        }
      }
    }

    current_hash = compute_internal_hash(children_hashes);
  }

  return current_hash == root_hash;
}

// Batch operations
std::vector<MerkleProof> NaryMerkleTree::generate_batch_proofs(
    const std::vector<size_t> &indices) const {
  std::vector<MerkleProof> proofs;
  proofs.reserve(indices.size());

  for (size_t index : indices) {
    auto proof = generate_proof(index);
    if (proof) {
      proofs.push_back(*proof);
    }
  }

  return proofs;
}

bool NaryMerkleTree::verify_batch_proofs(
    const std::vector<MerkleProof> &proofs,
    const std::vector<FieldElement> &leaf_values,
    const FieldElement &root_hash) const {
  if (proofs.size() != leaf_values.size()) {
    return false;
  }

  for (size_t i = 0; i < proofs.size(); ++i) {
    if (!verify_proof(proofs[i], leaf_values[i], root_hash)) {
      return false;
    }
  }

  return true;
}

// Tree operations
void NaryMerkleTree::insert_leaf(const FieldElement &leaf) {
  leaves_.push_back(leaf);
  leaf_count_++;
  build_tree(leaves_);
}

void NaryMerkleTree::update_leaf(size_t index, const FieldElement &new_value) {
  cuZK::ErrorHandling::validate_index(index, leaf_count_, "update_leaf");

  leaves_[index] = new_value;
  build_tree(leaves_);
}

// Getters
FieldElement NaryMerkleTree::get_root_hash() const {
  if (!root_) {
    return compute_empty_hash(config_.arity);
  }
  return root_->hash;
}

size_t NaryMerkleTree::get_tree_height() const {
  if (leaf_count_ == 0) {
    return 0;
  }
  return calculate_tree_height(leaf_count_, config_.arity);
}

// Utility functions
void NaryMerkleTree::print_tree() const {
  if (!root_) {
    std::cout << "Empty tree\n";
    return;
  }

  std::cout << "Merkle Tree (arity=" << config_.arity
            << ", leaves=" << leaf_count_ << "):\n";
  print_tree_recursive(root_, 0, "");
}

void NaryMerkleTree::print_tree_recursive(
    const std::shared_ptr<MerkleNode> &node, size_t level,
    const std::string &prefix) const {
  if (!node) {
    return;
  }

  std::cout << prefix << "Level " << level << ": "
            << node->hash.to_hex().substr(0, 16) << "..."
            << (node->is_leaf ? " (leaf)" : " (internal)") << "\n";

  for (size_t i = 0; i < node->children.size(); ++i) {
    print_tree_recursive(node->children[i], level + 1, prefix + "  ");
  }
}

// Static utilities
FieldElement NaryMerkleTree::compute_empty_hash(size_t arity) {
  // Use a deterministic empty hash based on arity
  static std::map<size_t, FieldElement> empty_hashes;

  if (empty_hashes.find(arity) == empty_hashes.end()) {
    std::vector<FieldElement> zeros(arity, ZERO);
    empty_hashes[arity] = Hash::hash_multiple(zeros);
  }

  return empty_hashes[arity];
}

size_t NaryMerkleTree::calculate_tree_height(size_t leaf_count, size_t arity) {
  if (leaf_count <= 1) {
    return 1;
  }

  return static_cast<size_t>(
             std::ceil(std::log(leaf_count) / std::log(arity))) +
         1;
}

size_t NaryMerkleTree::calculate_max_leaves(size_t height, size_t arity) {
  return static_cast<size_t>(std::pow(arity, height - 1));
}

// MerkleUtils namespace implementation
namespace MerkleUtils {

bool is_valid_proof_structure(const MerkleProof &proof, size_t arity) {
  if (proof.path.size() != proof.indices.size()) {
    return false;
  }

  for (size_t i = 0; i < proof.path.size(); ++i) {
    if (proof.path[i].size() != arity - 1) {
      return false;
    }
    if (proof.indices[i] >= arity) {
      return false;
    }
  }

  return true;
}

bool compare_trees(const NaryMerkleTree &tree1, const NaryMerkleTree &tree2) {
  return tree1.get_root_hash() == tree2.get_root_hash() &&
         tree1.get_leaf_count() == tree2.get_leaf_count() &&
         tree1.get_arity() == tree2.get_arity();
}

TreeBenchmarkResult benchmark_tree(size_t leaf_count, size_t arity,
                                   size_t num_proofs) {
  TreeBenchmarkResult result = {};
  result.leaf_count = leaf_count;
  result.arity = arity;

  // Generate test data
  auto leaves = generate_test_leaves(leaf_count);

  // Benchmark tree building
  auto start = std::chrono::high_resolution_clock::now();
  NaryMerkleTree tree(leaves, MerkleTreeConfig(arity));
  auto end = std::chrono::high_resolution_clock::now();

  result.build_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  result.tree_height = tree.get_tree_height();

  // Benchmark proof generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, leaf_count - 1);

  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_proofs; ++i) {
    size_t index = dis(gen);
    tree.generate_proof(index);
  }
  end = std::chrono::high_resolution_clock::now();

  result.proof_generation_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Benchmark proof verification
  auto proof = tree.generate_proof(0);
  if (proof) {
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_proofs; ++i) {
      tree.verify_proof(*proof, leaves[0], tree.get_root_hash());
    }
    end = std::chrono::high_resolution_clock::now();

    result.proof_verification_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
  }

  return result;
}

std::vector<FieldElement> generate_test_leaves(size_t count, uint64_t seed) {
  std::vector<FieldElement> leaves;
  leaves.reserve(count);

  std::mt19937_64 gen(seed);

  for (size_t i = 0; i < count; ++i) {
    uint64_t value = gen();
    leaves.emplace_back(value);
  }

  return leaves;
}

} // namespace MerkleUtils

} // namespace MerkleTree