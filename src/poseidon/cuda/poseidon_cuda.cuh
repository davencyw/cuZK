#pragma once

#include "../poseidon.hpp"
#include "field_arithmetic_cuda.cuh"
#include "cuda_field_element.cuh"
#include "poseidon_interface_cuda.hpp"
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
class CudaPoseidonHash : public IPoseidonCudaHash {
public:
    // Constructor and destructor handle RAII
    CudaPoseidonHash();
    ~CudaPoseidonHash() override;
    
    // Interface implementation
    bool batch_hash_single(const std::vector<FieldElement>& inputs, 
                          std::vector<FieldElement>& outputs) override;
    
    bool batch_hash_pairs(const std::vector<FieldElement>& left_inputs,
                         const std::vector<FieldElement>& right_inputs,
                         std::vector<FieldElement>& outputs) override;
    
    bool batch_permutation(std::vector<std::array<CudaFieldElement, PoseidonParams::STATE_SIZE>>& states) override;
    
    size_t get_optimal_batch_size() const override;
    size_t get_max_batch_size() const override;
    bool is_initialized() const override;
    
private:
    bool initialized_;
    size_t optimal_batch_size_;
    size_t max_batch_size_;
    
    // Device constants
    FieldElement* d_round_constants_;
    FieldElement* d_mds_matrix_;
    
    // Device memory management
    FieldElement* allocate_device_memory(size_t count);
    void free_device_memory(FieldElement* ptr);
    bool copy_to_device(const std::vector<FieldElement>& host_data, FieldElement* device_ptr);
    bool copy_from_device(FieldElement* device_ptr, std::vector<FieldElement>& host_data, size_t count);
    bool copy_constants_to_device();
    
};

} // namespace PoseidonCUDA
} // namespace Poseidon 