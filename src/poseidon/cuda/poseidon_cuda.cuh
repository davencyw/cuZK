#pragma once

#include "../poseidon.hpp"
#include "field_arithmetic_cuda.cuh"
#include "cuda_field_element.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

namespace Poseidon {

// CUDA-specific Poseidon operations
namespace PoseidonCUDA {

// Import CUDA field operations for device functions
using namespace Poseidon::CudaFieldOps;

// Device hash functions for merkle trees - implemented in .cu file
__device__ CudaFieldElement device_hash_n(const CudaFieldElement* children, size_t arity);

// Host interface for CUDA Poseidon operations
class CudaPoseidonHash {
public:
    // Initialize CUDA context and copy Poseidon constants to device
    static bool initialize();
    
    // Cleanup CUDA resources
    static void cleanup();
    
    // Batch hash operations on GPU
    static bool batch_hash_single(const std::vector<FieldElement>& inputs, 
                                 std::vector<FieldElement>& outputs);
    
    static bool batch_hash_pairs(const std::vector<FieldElement>& left_inputs,
                                const std::vector<FieldElement>& right_inputs,
                                std::vector<FieldElement>& outputs);
    
    static bool batch_permutation(std::vector<std::array<CudaFieldElement, PoseidonParams::STATE_SIZE>>& states);
    
    // Utility functions
    static size_t get_optimal_batch_size();
    static size_t get_max_batch_size();
    
private:
    static bool initialized_;
    static size_t optimal_batch_size_;
    static size_t max_batch_size_;
    
    // Device constants
    static FieldElement* d_round_constants_;
    static FieldElement* d_mds_matrix_;
    
    // Device memory management
    static FieldElement* allocate_device_memory(size_t count);
    static void free_device_memory(FieldElement* ptr);
    static bool copy_to_device(const std::vector<FieldElement>& host_data, FieldElement* device_ptr);
    static bool copy_from_device(FieldElement* device_ptr, std::vector<FieldElement>& host_data, size_t count);
    static bool copy_constants_to_device();
    
};

// Performance benchmarking for CUDA Poseidon operations
struct CudaPoseidonStats {
    double total_time_ms;
    double avg_time_per_hash_ns;
    size_t hashes_per_second;
    size_t total_hashes;
    size_t gpu_memory_used_mb;
    double gpu_utilization_percent;
    double speedup_vs_cpu;
};

// Benchmark functions
CudaPoseidonStats benchmark_cuda_poseidon_single(size_t num_hashes, size_t batch_size = 1024);
CudaPoseidonStats benchmark_cuda_poseidon_pairs(size_t num_pairs, size_t batch_size = 1024);
CudaPoseidonStats benchmark_cuda_vs_cpu_poseidon(size_t num_hashes, size_t batch_size = 1024);

} // namespace PoseidonCUDA
} // namespace Poseidon 