#include "merkle_tree_cuda.cuh"
#include "../common/error_handling.hpp"
#include "../common/namespace_utils.hpp"
#include "../poseidon/cuda/poseidon_cuda.cuh"
#include "../poseidon/cuda/cuda_field_element.cuh"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
#include <memory>
#include <map>
#include <random>

namespace MerkleTree {
namespace MerkleTreeCUDA {

// Using statements for easier access to CUDA types
using namespace Poseidon;
using namespace Poseidon::PoseidonCUDA;
using CudaFieldElement = Poseidon::CudaFieldElement;

// Static members for device memory management
static bool cuda_initialized = false;

// Use standardized CUDA error handling - macros are imported via include
using cuZK::ErrorHandling::cuda_sync_check;

USING_CUZK_TYPES()

// Device function to compute empty hash (match CPU implementation)
__device__ CudaFieldElement device_compute_empty_hash(size_t arity) {
    // Create array of zeros and hash them
    CudaFieldElement zeros[8]; // Max arity is 8
    for (size_t i = 0; i < arity; ++i) {
        zeros[i] = CudaFieldElement(0);
    }
    
    return device_hash_n(zeros, arity);
}

// CUDA kernel for building a single level of the merkle tree
__global__ void build_level_kernel(const CudaFieldElement* input_level, CudaFieldElement* output_level, 
                                  size_t input_count, size_t arity) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_count = input_count / arity; // Exact division since input is padded by caller
    
    if (idx >= output_count) return;
    
    // Calculate input range for this output element
    size_t input_start = idx * arity;
    
    // Collect children for this node (exactly arity children since pre-padded)
    CudaFieldElement children[8]; // Max arity is 8
    
    for (size_t i = 0; i < arity; ++i) {
        children[i] = input_level[input_start + i];
    }
    
    // Compute hash of children using device Poseidon hash
    output_level[idx] = device_hash_n(children, arity);
}

// CUDA kernel for batch proof verification
__global__ void batch_verify_proofs_kernel(const CudaFieldElement* leaf_values, 
                                          const CudaFieldElement* proof_data,
                                          const size_t* proof_offsets,
                                          const size_t* proof_indices,
                                          const size_t* proof_level_indices,
                                          const CudaFieldElement* root_hash,
                                          bool* results,
                                          size_t num_proofs,
                                          size_t arity,
                                          size_t tree_height) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_proofs) return;
    
    // Get proof data for this thread
    size_t proof_start = proof_offsets[idx];
    size_t proof_end = proof_offsets[idx + 1]; // proof_offsets has (num_proofs + 1) elements
    
    CudaFieldElement current_hash = leaf_values[idx];
    
    // Verify proof level by level - match CPU logic exactly
    size_t proof_data_idx = proof_start;
    size_t level_indices_start = idx * tree_height; // Each proof has tree_height level indices
    
    for (size_t level = 0; level < tree_height; ++level) {
        size_t position_in_group = proof_level_indices[level_indices_start + level];
        
        // Reconstruct the children hashes group - exactly match CPU logic
        CudaFieldElement children_hashes[8]; // Max arity is 8
        size_t sibling_idx = 0;
        
        for (size_t i = 0; i < arity; ++i) {
            if (i == position_in_group) {
                children_hashes[i] = current_hash;
            } else {
                if (sibling_idx < (arity - 1) && proof_data_idx < proof_end) {
                    children_hashes[i] = proof_data[proof_data_idx++];
                    sibling_idx++;
                } else {
                    // Pad with empty hash (match CPU implementation)
                    children_hashes[i] = device_compute_empty_hash(arity);
                }
            }
        }
        
        // Compute parent hash
        current_hash = device_hash_n(children_hashes, arity);
    }
    
    // Check if computed root matches expected root
    results[idx] = (current_hash == *root_hash);
}

// CudaNaryMerkleTree implementation

CudaNaryMerkleTree::CudaNaryMerkleTree(const MerkleTreeConfig& config)
    : config_(config), leaf_count_(0), tree_height_(0) {
    if (!cuda_initialized) {
        initialize_cuda();
    }
}

CudaNaryMerkleTree::CudaNaryMerkleTree(const std::vector<FieldElement>& leaves, 
                                     const MerkleTreeConfig& config)
    : config_(config), leaf_count_(0), tree_height_(0) {
    if (!cuda_initialized) {
        initialize_cuda();
    }
    build_tree(leaves);
}

CudaNaryMerkleTree::~CudaNaryMerkleTree() {
}

bool CudaNaryMerkleTree::build_tree(const std::vector<FieldElement>& leaves) {
    if (leaves.empty()) {
        leaf_count_ = 0;
        tree_height_ = 0;
        leaves_.clear();
        tree_levels_.clear();
        return true;
    }
    
    leaves_ = leaves;
    leaf_count_ = leaves.size();
    
    // Calculate tree height using the same method as CPU implementation
    tree_height_ = NaryMerkleTree::calculate_tree_height(leaf_count_, config_.arity);
    
    return build_tree_cuda(leaves);
}

bool CudaNaryMerkleTree::build_tree_cuda(const std::vector<FieldElement>& leaves) {
    tree_levels_.clear();
    tree_levels_.resize(tree_height_);
    
    // Initialize with leaf level - pad to next power of arity (match CPU implementation)
    size_t padded_size = 1;
    while (padded_size < leaves.size()) {
        padded_size *= config_.arity;
    }
    
    std::vector<FieldElement> padded_leaves;
    padded_leaves.reserve(padded_size);
    
    // Copy original leaves
    for (size_t i = 0; i < leaves.size(); ++i) {
        padded_leaves.push_back(leaves[i]);
    }
    
    // Pad with empty hash to match CPU implementation exactly
    FieldElement empty_hash = compute_empty_hash(config_.arity);
    for (size_t i = leaves.size(); i < padded_size; ++i) {
        padded_leaves.push_back(empty_hash);
    }
    
    tree_levels_[0] = padded_leaves;
    
    // Build each level using GPU
    // tree_height_ includes the leaf level, so we build tree_height_-1 internal levels
    // Level 0: leaves, Level 1: first internal level, ..., Level tree_height_-1: root
    for (size_t level = 0; level < tree_height_ - 1; ++level) {
        if (!build_level_on_gpu(tree_levels_[level], tree_levels_[level + 1])) {
            return false;
        }
    }
    
    return true;
}

bool CudaNaryMerkleTree::build_level_on_gpu(const std::vector<FieldElement>& input_level, 
                                           std::vector<FieldElement>& output_level) {
    if (input_level.empty()) {
        output_level.clear();
        return true;
    }
    
    size_t input_count = input_level.size();
    size_t output_count = calculate_level_size(input_count);
    output_level.resize(output_count);
    
    // Pad input to be divisible by arity (match CPU behavior)
    size_t padded_input_count = output_count * config_.arity;
    std::vector<FieldElement> padded_input = input_level;
    FieldElement empty_hash = compute_empty_hash(config_.arity);
    
    while (padded_input.size() < padded_input_count) {
        padded_input.push_back(empty_hash);
    }
    
    // Convert input to CudaFieldElement
    std::vector<CudaFieldElement> cuda_input;
    cuda_input.reserve(padded_input_count);
    for (const auto& fe : padded_input) {
        cuda_input.emplace_back(fe);
    }
    
    // Allocate device memory for input and output
    CudaFieldElement* d_input;
    CudaFieldElement* d_output;
    
    CUDA_CHECK_RETURN(cudaMalloc(&d_input, padded_input_count * sizeof(CudaFieldElement)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, output_count * sizeof(CudaFieldElement)));
    
    // Copy input to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_input, cuda_input.data(), 
                         padded_input_count * sizeof(CudaFieldElement), cudaMemcpyHostToDevice));
    
    // Launch kernel
    size_t threads_per_block = 256;
    size_t blocks = (output_count + threads_per_block - 1) / threads_per_block;
    
    build_level_kernel<<<blocks, threads_per_block>>>(d_input, d_output, padded_input_count, config_.arity);
    
    if (!cuda_sync_check()) return false;
    
    // Copy result back to host and convert to FieldElement
    std::vector<CudaFieldElement> cuda_output(output_count);
    CUDA_CHECK_RETURN(cudaMemcpy(cuda_output.data(), d_output, 
                         output_count * sizeof(CudaFieldElement), cudaMemcpyDeviceToHost));
    
    // Convert back to FieldElement
    for (size_t i = 0; i < output_count; ++i) {
        output_level[i] = FieldElement(cuda_output[i].limbs[0], cuda_output[i].limbs[1], 
                                     cuda_output[i].limbs[2], cuda_output[i].limbs[3]);
    }
    
    // Clean up device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return true;
}

std::optional<MerkleProof> CudaNaryMerkleTree::generate_proof(size_t leaf_index) const {
    if (leaf_index >= leaf_count_ || tree_levels_.empty()) {
        return std::nullopt;
    }
    
    MerkleProof proof;
    proof.leaf_index = leaf_index;
    proof.path.resize(tree_height_ - 1);  // Proof goes from leaf to root (excluding leaf level)
    proof.indices.resize(tree_height_ - 1);
    
    size_t current_index = leaf_index;
    
    for (size_t level = 0; level < tree_height_ - 1; ++level) {
        size_t group_start = (current_index / config_.arity) * config_.arity;
        size_t position_in_group = current_index % config_.arity;
        
        // Collect siblings (excluding the current element)
        std::vector<FieldElement> siblings;
        for (size_t i = 0; i < config_.arity; ++i) {
            if (i != position_in_group && group_start + i < tree_levels_[level].size()) {
                siblings.push_back(tree_levels_[level][group_start + i]);
            }
        }
        
        proof.path[level] = siblings;
        proof.indices[level] = position_in_group;
        
        current_index /= config_.arity;
    }
    
    return proof;
}

bool CudaNaryMerkleTree::verify_proof(const MerkleProof& proof, const FieldElement& leaf_value) const {
    if (tree_levels_.empty()) return false;
    
    FieldElement current_hash = leaf_value;
    size_t current_index = proof.leaf_index;
    
    for (size_t level = 0; level < proof.path.size(); ++level) {
        const auto& siblings = proof.path[level];
        size_t position = proof.indices[level];
        
        // Reconstruct the full group
        std::vector<FieldElement> group;
        size_t sibling_idx = 0;
        
        for (size_t i = 0; i < config_.arity; ++i) {
            if (i == position) {
                group.push_back(current_hash);
            } else if (sibling_idx < siblings.size()) {
                group.push_back(siblings[sibling_idx++]);
            } else {
                // Use proper empty hash computation (match CPU implementation)
                group.push_back(compute_empty_hash(config_.arity));
            }
        }
        
        // Compute parent hash
        current_hash = compute_internal_hash(group);
        current_index /= config_.arity;
    }
    
    return current_hash == get_root_hash();
}

std::vector<MerkleProof> CudaNaryMerkleTree::generate_batch_proofs(const std::vector<size_t>& indices) const {
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

bool CudaNaryMerkleTree::verify_batch_proofs(const std::vector<MerkleProof>& proofs, 
                                            const std::vector<FieldElement>& leaf_values) const {
    if (proofs.size() != leaf_values.size() || proofs.empty()) {
        return false;
    }
    
    // For small batches, use CPU verification
    if (proofs.size() < 32) {
        for (size_t i = 0; i < proofs.size(); ++i) {
            if (!verify_proof(proofs[i], leaf_values[i])) {
                return false;
            }
        }
        return true;
    }
    
    // Use GPU for large batches
    size_t num_proofs = proofs.size();
    FieldElement root = get_root_hash();
    
    // Prepare proof data for GPU
    std::vector<FieldElement> proof_data;
    std::vector<size_t> proof_offsets;
    std::vector<size_t> proof_indices;
    std::vector<size_t> proof_level_indices; // Store per-level position indices
    
    proof_offsets.push_back(0);
    
    for (const auto& proof : proofs) {
        proof_indices.push_back(proof.leaf_index);
        
        // Store the per-level position indices
        for (size_t level_idx : proof.indices) {
            proof_level_indices.push_back(level_idx);
        }
        
        for (const auto& level : proof.path) {
            for (const auto& sibling : level) {
                proof_data.push_back(sibling);
            }
        }
        
        proof_offsets.push_back(proof_data.size());
    }
    
    // Convert to CudaFieldElement
    std::vector<CudaFieldElement> cuda_leaf_values;
    std::vector<CudaFieldElement> cuda_proof_data;
    CudaFieldElement cuda_root;
    
    cuda_leaf_values.reserve(num_proofs);
    for (const auto& fe : leaf_values) {
        cuda_leaf_values.emplace_back(fe);
    }
    
    cuda_proof_data.reserve(proof_data.size());
    for (const auto& fe : proof_data) {
        cuda_proof_data.emplace_back(fe);
    }
    
    cuda_root = CudaFieldElement(root);
    
    // Allocate device memory
    CudaFieldElement* d_leaf_values;
    CudaFieldElement* d_proof_data;
    size_t* d_proof_offsets;
    size_t* d_proof_indices;
    size_t* d_proof_level_indices;
    CudaFieldElement* d_root_hash;
    bool* d_results;
    
    CUDA_CHECK_RETURN(cudaMalloc(&d_leaf_values, num_proofs * sizeof(CudaFieldElement)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_proof_data, cuda_proof_data.size() * sizeof(CudaFieldElement)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_proof_offsets, proof_offsets.size() * sizeof(size_t)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_proof_indices, num_proofs * sizeof(size_t)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_proof_level_indices, proof_level_indices.size() * sizeof(size_t)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_root_hash, sizeof(CudaFieldElement)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_results, num_proofs * sizeof(bool)));
    
    // Copy data to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_leaf_values, cuda_leaf_values.data(), 
                         num_proofs * sizeof(CudaFieldElement), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_proof_data, cuda_proof_data.data(), 
                         cuda_proof_data.size() * sizeof(CudaFieldElement), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_proof_offsets, proof_offsets.data(), 
                         proof_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_proof_indices, proof_indices.data(), 
                         num_proofs * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_proof_level_indices, proof_level_indices.data(), 
                         proof_level_indices.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_root_hash, &cuda_root, sizeof(CudaFieldElement), cudaMemcpyHostToDevice));
    
    // Launch kernel
    size_t threads_per_block = 256;
    size_t blocks = (num_proofs + threads_per_block - 1) / threads_per_block;
    
    batch_verify_proofs_kernel<<<blocks, threads_per_block>>>(
        d_leaf_values, d_proof_data, d_proof_offsets, d_proof_indices, d_proof_level_indices,
        d_root_hash, d_results, num_proofs, config_.arity, tree_height_ - 1);
    
    if (!cuda_sync_check()) return false;
    
    // Copy results back (use raw array since std::vector<bool> is specialized)
    std::unique_ptr<bool[]> temp_results(new bool[num_proofs]);
    CUDA_CHECK_RETURN(cudaMemcpy(temp_results.get(), d_results, 
                         num_proofs * sizeof(bool), cudaMemcpyDeviceToHost));
    
    // Convert to vector
    std::vector<bool> results(num_proofs);
    for (size_t i = 0; i < num_proofs; ++i) {
        results[i] = temp_results[i];
    }
    
    // Clean up device memory
    cudaFree(d_leaf_values);
    cudaFree(d_proof_data);
    cudaFree(d_proof_offsets);
    cudaFree(d_proof_indices);
    cudaFree(d_proof_level_indices);
    cudaFree(d_root_hash);
    cudaFree(d_results);
    
    // Check if all proofs are valid
    return std::all_of(results.begin(), results.end(), [](bool valid) { return valid; });
}

bool CudaNaryMerkleTree::build_batch_trees(const std::vector<std::vector<FieldElement>>& batch_leaves,
                                          std::vector<CudaNaryMerkleTree>& trees,
                                          const MerkleTreeConfig& config) {
    trees.clear();
    trees.reserve(batch_leaves.size());
    
    for (const auto& leaves : batch_leaves) {
        CudaNaryMerkleTree tree(config);
        if (!tree.build_tree(leaves)) {
            return false;
        }
        trees.push_back(std::move(tree));
    }
    
    return true;
}

FieldElement CudaNaryMerkleTree::get_root_hash() const {
    if (tree_levels_.empty() || tree_levels_.back().empty()) {
        return compute_empty_hash(config_.arity);
    }
    // Root is at the last level (tree_height_ - 1)
    return tree_levels_[tree_height_ - 1][0];
}



void CudaNaryMerkleTree::print_tree() const {
    std::cout << "CUDA Merkle Tree (arity=" << config_.arity << ", height=" << tree_height_ << "):" << std::endl;
    
    for (size_t level = tree_levels_.size(); level > 0; --level) {
        size_t idx = level - 1;
        std::cout << "Level " << idx << ": ";
        for (size_t i = 0; i < std::min(tree_levels_[idx].size(), size_t(8)); ++i) {
            std::cout << tree_levels_[idx][i].to_hex().substr(0, 8) << "... ";
        }
        if (tree_levels_[idx].size() > 8) {
            std::cout << "(+" << (tree_levels_[idx].size() - 8) << " more)";
        }
        std::cout << std::endl;
    }
}

bool CudaNaryMerkleTree::compare_with_cpu_tree(const NaryMerkleTree& cpu_tree) const {
    if (get_leaf_count() != cpu_tree.get_leaf_count() ||
        get_arity() != cpu_tree.get_arity()) {
        return false;
    }
    
    return get_root_hash() == cpu_tree.get_root_hash();
}



size_t CudaNaryMerkleTree::calculate_level_size(size_t input_size) const {
    // Use ceiling division to match CPU implementation
    return (input_size + config_.arity - 1) / config_.arity;
}



FieldElement CudaNaryMerkleTree::compute_internal_hash(const std::vector<FieldElement>& children) const {
    if (children.empty()) {
        return compute_empty_hash(config_.arity);
    }

    // Use the same hash function as CPU implementation for consistency
    return Poseidon::PoseidonHash::hash_multiple(children);
}

FieldElement CudaNaryMerkleTree::compute_empty_hash(size_t arity) const {
    // Use a deterministic empty hash based on arity (match CPU implementation exactly)
    static std::map<size_t, FieldElement> empty_hashes;
    
    if (empty_hashes.find(arity) == empty_hashes.end()) {
        std::vector<FieldElement> zeros(arity, Poseidon::FieldConstants::ZERO);
        empty_hashes[arity] = Poseidon::PoseidonHash::hash_multiple(zeros);
    }
    
    return empty_hashes[arity];
}

// Static utility functions
bool CudaNaryMerkleTree::initialize_cuda() {
    if (cuda_initialized) return true;
    
    // Initialize CUDA Poseidon
    if (!Poseidon::PoseidonCUDA::CudaPoseidonHash::initialize()) {
        std::cerr << "Failed to initialize CUDA Poseidon" << std::endl;
        return false;
    }
    
    cuda_initialized = true;
    return true;
}

void CudaNaryMerkleTree::cleanup_cuda() {
    if (cuda_initialized) {
        Poseidon::PoseidonCUDA::CudaPoseidonHash::cleanup();
        cuda_initialized = false;
    }
}

size_t CudaNaryMerkleTree::get_optimal_batch_size() {
    return Poseidon::PoseidonCUDA::CudaPoseidonHash::get_optimal_batch_size();
}

size_t CudaNaryMerkleTree::get_max_batch_size() {
    return Poseidon::PoseidonCUDA::CudaPoseidonHash::get_max_batch_size();
}

// Utility namespace implementations
namespace CudaMerkleUtils {

MerkleTreeConfig get_optimal_config_for_gpu(size_t leaf_count) {
    // For GPU efficiency, higher arity can be better due to parallelization
    size_t optimal_arity = 4; // Good balance between parallelism and GPU cores
    
    if (leaf_count < 1000) {
        optimal_arity = 2; // Binary for small trees
    } else if (leaf_count > 100000) {
        optimal_arity = 8; // Higher arity for very large trees
    }
    
    size_t tree_height = NaryMerkleTree::calculate_tree_height(leaf_count, optimal_arity);
    return MerkleTreeConfig(optimal_arity, tree_height);
}

bool check_cuda_compatibility() {
    int deviceCount;
    CUDA_CHECK_CLEANUP(cudaGetDeviceCount(&deviceCount), );
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return false;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK_VOID(cudaGetDeviceProperties(&prop, 0));
    
    if (prop.major < 3) {
        std::cerr << "CUDA compute capability 3.0 or higher required" << std::endl;
        return false;
    }
    
    return true;
}

std::vector<std::vector<FieldElement>> generate_batch_test_leaves(
    size_t num_trees, size_t leaves_per_tree, uint64_t seed) {
    
    std::vector<std::vector<FieldElement>> batch_leaves;
    batch_leaves.reserve(num_trees);
    
    for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        std::vector<FieldElement> leaves;
        leaves.reserve(leaves_per_tree);
        
        for (size_t leaf_idx = 0; leaf_idx < leaves_per_tree; ++leaf_idx) {
            // Generate deterministic but varied test data
            uint64_t value = seed + tree_idx * leaves_per_tree + leaf_idx;
            leaves.push_back(FieldElement(value));
        }
        
        batch_leaves.push_back(std::move(leaves));
    }
    
    return batch_leaves;
}

} // namespace CudaMerkleUtils

// Benchmark function implementations
CudaMerkleTreeStats benchmark_cuda_tree_building(size_t num_trees, size_t leaves_per_tree, 
                                                size_t arity, size_t batch_size) {
    CudaMerkleTreeStats stats = {};
    stats.leaf_count = leaves_per_tree;
    stats.arity = arity;
    stats.total_trees = num_trees;
    
    if (!CudaNaryMerkleTree::initialize_cuda()) {
        std::cerr << "Failed to initialize CUDA for benchmarking" << std::endl;
        return stats;
    }
    
    // Generate test data
    auto batch_leaves = CudaMerkleUtils::generate_batch_test_leaves(num_trees, leaves_per_tree, 12345);
    
    // Measure CUDA tree building
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<CudaNaryMerkleTree> trees;
    MerkleTreeConfig config(arity);
    bool success = CudaNaryMerkleTree::build_batch_trees(batch_leaves, trees, config);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (!success || trees.empty()) {
        std::cerr << "CUDA tree building failed" << std::endl;
        return stats;
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.build_time_ms = duration.count() / 1000.0;
    stats.total_time_ms = stats.build_time_ms;
    stats.tree_height = trees[0].get_tree_height();
    
    if (stats.build_time_ms > 0) {
        stats.trees_per_second = static_cast<size_t>((num_trees * 1000.0) / stats.build_time_ms);
    }
    
    return stats;
}

CudaMerkleTreeStats benchmark_cuda_proof_generation(size_t num_proofs, size_t leaves_per_tree,
                                                   size_t arity, size_t batch_size) {
    CudaMerkleTreeStats stats = {};
    stats.leaf_count = leaves_per_tree;
    stats.arity = arity;
    stats.total_proofs = num_proofs;
    
    if (!CudaNaryMerkleTree::initialize_cuda()) {
        std::cerr << "Failed to initialize CUDA for benchmarking" << std::endl;
        return stats;
    }
    
    // Build a test tree
    auto leaves = CudaMerkleUtils::generate_batch_test_leaves(1, leaves_per_tree, 54321)[0];
    MerkleTreeConfig config(arity);
    CudaNaryMerkleTree tree(leaves, config);
    
    stats.tree_height = tree.get_tree_height();
    
    // Generate random indices for proofs
    std::vector<size_t> indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, leaves_per_tree - 1);
    
    for (size_t i = 0; i < num_proofs; ++i) {
        indices.push_back(dis(gen));
    }
    
    // Measure proof generation time
    auto start = std::chrono::high_resolution_clock::now();
    auto proofs = tree.generate_batch_proofs(indices);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.proof_generation_time_ms = duration.count() / 1000.0;
    stats.total_time_ms = stats.proof_generation_time_ms;
    
    if (stats.proof_generation_time_ms > 0) {
        stats.proofs_per_second = static_cast<size_t>((num_proofs * 1000.0) / stats.proof_generation_time_ms);
    }
    
    return stats;
}

CudaMerkleTreeStats benchmark_cuda_proof_verification(size_t num_proofs, size_t leaves_per_tree,
                                                     size_t arity, size_t batch_size) {
    CudaMerkleTreeStats stats = {};
    stats.leaf_count = leaves_per_tree;
    stats.arity = arity;
    stats.total_proofs = num_proofs;
    
    if (!CudaNaryMerkleTree::initialize_cuda()) {
        std::cerr << "Failed to initialize CUDA for benchmarking" << std::endl;
        return stats;
    }
    
    // Build a test tree and generate proofs
    auto leaves = CudaMerkleUtils::generate_batch_test_leaves(1, leaves_per_tree, 98765)[0];
    MerkleTreeConfig config(arity);
    CudaNaryMerkleTree tree(leaves, config);
    
    stats.tree_height = tree.get_tree_height();
    
    // Generate random indices and corresponding proofs
    std::vector<size_t> indices;
    std::vector<FieldElement> leaf_values;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, leaves_per_tree - 1);
    
    for (size_t i = 0; i < num_proofs; ++i) {
        size_t idx = dis(gen);
        indices.push_back(idx);
        leaf_values.push_back(leaves[idx]);
    }
    
    auto proofs = tree.generate_batch_proofs(indices);
    
    // Measure proof verification time
    auto start = std::chrono::high_resolution_clock::now();
    bool result = tree.verify_batch_proofs(proofs, leaf_values);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.proof_verification_time_ms = duration.count() / 1000.0;
    stats.total_time_ms = stats.proof_verification_time_ms;
    
    if (stats.proof_verification_time_ms > 0) {
        stats.proofs_per_second = static_cast<size_t>((num_proofs * 1000.0) / stats.proof_verification_time_ms);
    }
    
    if (!result) {
        std::cerr << "Warning: Batch proof verification failed in benchmark" << std::endl;
    }
    
    return stats;
}

CudaMerkleTreeStats benchmark_cuda_vs_cpu_merkle(size_t num_trees, size_t leaves_per_tree,
                                                 size_t arity, size_t batch_size) {
    CudaMerkleTreeStats stats = {};
    stats.leaf_count = leaves_per_tree;
    stats.arity = arity;
    stats.total_trees = num_trees;
    
    if (!CudaNaryMerkleTree::initialize_cuda()) {
        std::cerr << "Failed to initialize CUDA for benchmarking" << std::endl;
        return stats;
    }
    
    // Generate test data
    auto batch_leaves = CudaMerkleUtils::generate_batch_test_leaves(num_trees, leaves_per_tree, 11111);
    MerkleTreeConfig config(arity);
    
    // Benchmark CPU implementation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    std::vector<NaryMerkleTree> cpu_trees;
    cpu_trees.reserve(num_trees);
    
    for (const auto& leaves : batch_leaves) {
        cpu_trees.emplace_back(leaves, config);
    }
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    double cpu_time_ms = cpu_duration.count() / 1000.0;
    
    // Benchmark CUDA implementation
    auto cuda_start = std::chrono::high_resolution_clock::now();
    
    std::vector<CudaNaryMerkleTree> cuda_trees;
    bool success = CudaNaryMerkleTree::build_batch_trees(batch_leaves, cuda_trees, config);
    
    auto cuda_end = std::chrono::high_resolution_clock::now();
    auto cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>(cuda_end - cuda_start);
    
    if (!success) {
        std::cerr << "CUDA tree building failed in comparison benchmark" << std::endl;
        return stats;
    }
    
    stats.build_time_ms = cuda_duration.count() / 1000.0;
    stats.total_time_ms = stats.build_time_ms;
    stats.tree_height = cuda_trees[0].get_tree_height();
    
    // Calculate speedup
    if (stats.build_time_ms > 0) {
        stats.speedup_vs_cpu = cpu_time_ms / stats.build_time_ms;
        stats.trees_per_second = static_cast<size_t>((num_trees * 1000.0) / stats.build_time_ms);
    }
    
    // Verify consistency
    bool all_consistent = true;
    for (size_t i = 0; i < std::min(cpu_trees.size(), cuda_trees.size()); ++i) {
        if (!cuda_trees[i].compare_with_cpu_tree(cpu_trees[i])) {
            all_consistent = false;
            break;
        }
    }
    
    if (!all_consistent) {
        std::cerr << "Warning: CPU and CUDA trees are not consistent!" << std::endl;
    }
    
    return stats;
}

} // namespace MerkleTreeCUDA
} // namespace MerkleTree 