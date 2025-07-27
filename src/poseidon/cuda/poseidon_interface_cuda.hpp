#pragma once

#include "../poseidon.hpp"
#include "cuda_field_element.cuh"
#include <vector>
#include <array>

namespace Poseidon {
namespace PoseidonCUDA {

// Forward declarations
using namespace Poseidon::CudaFieldOps;

// Performance benchmarking for CUDA Poseidon operations
struct CudaPoseidonStats {
    double total_time_ms;
    double avg_time_per_hash_ns;
    size_t hashes_per_second;
    size_t total_hashes;
    double speedup_vs_cpu;
};

/**
 * Virtual interface for Poseidon CUDA hash operations
 * This interface defines the minimal contract for CUDA-based Poseidon hashing
 */
class IPoseidonCudaHash {
public:
    virtual ~IPoseidonCudaHash() = default;
    
    // Batch hash operations
    virtual bool batch_hash_single(const std::vector<FieldElement>& inputs, 
                                  std::vector<FieldElement>& outputs) = 0;
    
    virtual bool batch_hash_pairs(const std::vector<FieldElement>& left_inputs,
                                 const std::vector<FieldElement>& right_inputs,
                                 std::vector<FieldElement>& outputs) = 0;
    
    virtual bool batch_permutation(std::vector<std::array<CudaFieldElement, PoseidonParams::STATE_SIZE>>& states) = 0;
    
    // Utility functions
    virtual size_t get_optimal_batch_size() const = 0;
    virtual size_t get_max_batch_size() const = 0;
    
    // Status check
    virtual bool is_initialized() const = 0;
};

} // namespace PoseidonCUDA
} // namespace Poseidon 