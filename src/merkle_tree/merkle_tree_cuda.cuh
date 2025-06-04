#pragma once

#include "merkle_tree.hpp"
#include "../poseidon/poseidon_cuda.cuh"
#include "../poseidon/cuda_field_element.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>
#include <optional>

namespace MerkleTree {

// CUDA-specific Merkle Tree operations
namespace MerkleTreeCUDA {

// Using statements for easier access to CUDA types
using CudaFieldElement = Poseidon::CudaFieldElement;

// Forward declarations
struct CudaMerkleTreeStats;

// CUDA kernel function declarations
__global__ void build_level_kernel(const CudaFieldElement* input_level, CudaFieldElement* output_level, 
                                  size_t input_count, size_t arity);

__global__ void batch_verify_proofs_kernel(const CudaFieldElement* leaf_values, 
                                          const CudaFieldElement* proof_data,
                                          const size_t* proof_offsets,
                                          const size_t* proof_indices,
                                          const size_t* proof_level_indices,
                                          const CudaFieldElement* root_hash,
                                          bool* results,
                                          size_t num_proofs,
                                          size_t arity,
                                          size_t tree_height);

// Main CUDA-accelerated Merkle Tree class
class CudaNaryMerkleTree {
private:
    MerkleTreeConfig config_;
    std::vector<FieldElement> leaves_;
    std::vector<std::vector<FieldElement>> tree_levels_;
    size_t leaf_count_;
    size_t tree_height_;
    

    
    // Internal helper functions
    bool build_tree_cuda(const std::vector<FieldElement>& leaves);
    bool build_level_on_gpu(const std::vector<FieldElement>& input_level, 
                           std::vector<FieldElement>& output_level);
    

    
    // Utility functions
    size_t calculate_level_size(size_t input_size) const;
    FieldElement compute_internal_hash(const std::vector<FieldElement>& children) const;
    FieldElement compute_empty_hash(size_t arity) const;
    
public:
    // Constructors
    explicit CudaNaryMerkleTree(const MerkleTreeConfig& config = MerkleTreeConfig());
    explicit CudaNaryMerkleTree(const std::vector<FieldElement>& leaves, 
                               const MerkleTreeConfig& config = MerkleTreeConfig());
    
    // Destructor
    ~CudaNaryMerkleTree();
    
    // Tree operations
    bool build_tree(const std::vector<FieldElement>& leaves);
    
    // Proof operations
    std::optional<MerkleProof> generate_proof(size_t leaf_index) const;
    bool verify_proof(const MerkleProof& proof, const FieldElement& leaf_value) const;
    
    // Batch operations (CUDA-accelerated)
    std::vector<MerkleProof> generate_batch_proofs(const std::vector<size_t>& indices) const;
    bool verify_batch_proofs(const std::vector<MerkleProof>& proofs, 
                            const std::vector<FieldElement>& leaf_values) const;
    
    // Batch tree building from multiple leaf sets
    static bool build_batch_trees(const std::vector<std::vector<FieldElement>>& batch_leaves,
                                 std::vector<CudaNaryMerkleTree>& trees,
                                 const MerkleTreeConfig& config = MerkleTreeConfig());
    
    // Getters
    FieldElement get_root_hash() const;
    size_t get_leaf_count() const { return leaf_count_; }
    size_t get_tree_height() const { return tree_height_; }
    size_t get_arity() const { return config_.arity; }
    const std::vector<FieldElement>& get_leaves() const { return leaves_; }
    const std::vector<std::vector<FieldElement>>& get_tree_levels() const { return tree_levels_; }
    
    // Utility functions
    void print_tree() const;
    
    // Comparison with CPU implementation
    bool compare_with_cpu_tree(const NaryMerkleTree& cpu_tree) const;
    
    // Static utilities
    static bool initialize_cuda();
    static void cleanup_cuda();
    static size_t get_optimal_batch_size();
    static size_t get_max_batch_size();
};

// Performance benchmarking for CUDA Merkle Tree operations
struct CudaMerkleTreeStats {
    double total_time_ms;
    double build_time_ms;
    double proof_generation_time_ms;
    double proof_verification_time_ms;
    size_t trees_per_second;
    size_t proofs_per_second;
    size_t total_trees;
    size_t total_proofs;
    double gpu_utilization_percent;
    double speedup_vs_cpu;
    size_t leaf_count;
    size_t tree_height;
    size_t arity;
};

// Benchmark functions
CudaMerkleTreeStats benchmark_cuda_tree_building(size_t num_trees, size_t leaves_per_tree, 
                                                size_t arity = 2, size_t batch_size = 32);

CudaMerkleTreeStats benchmark_cuda_proof_generation(size_t num_proofs, size_t leaves_per_tree,
                                                   size_t arity = 2, size_t batch_size = 256);

CudaMerkleTreeStats benchmark_cuda_proof_verification(size_t num_proofs, size_t leaves_per_tree,
                                                     size_t arity = 2, size_t batch_size = 512);

CudaMerkleTreeStats benchmark_cuda_vs_cpu_merkle(size_t num_trees, size_t leaves_per_tree,
                                                 size_t arity = 2, size_t batch_size = 32);

// Utility namespace for CUDA Merkle Tree operations
namespace CudaMerkleUtils {
    
    
    // Optimal configuration
    MerkleTreeConfig get_optimal_config_for_gpu(size_t leaf_count);
    
    // Error checking
    bool check_cuda_compatibility();
    
    // Test data generation for GPU
    std::vector<std::vector<FieldElement>> generate_batch_test_leaves(
        size_t num_trees, size_t leaves_per_tree, uint64_t seed = 0);
    
} // namespace CudaMerkleUtils

} // namespace MerkleTreeCUDA
} // namespace MerkleTree 