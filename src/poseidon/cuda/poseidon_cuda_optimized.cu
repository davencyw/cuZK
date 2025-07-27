#include "poseidon_cuda_optimized.cuh"
#include "field_arithmetic_cuda.cuh"  // Still needed for class methods like initialize()
#include "cuda_field_element.cuh"
#include "poseidon_interface_cuda.hpp"
#include "poseidon_cuda_benchmarks.hpp"
#include "../../common/error_handling.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <iostream>

using namespace Poseidon::CudaFieldOps;  // For CudaFieldArithmetic class methods and CudaFieldElement arithmetic functions

namespace Poseidon {
namespace PoseidonCUDAOptimized {

// Use standardized CUDA error handling - these are macros, imported via include
using cuZK::ErrorHandling::cuda_sync_check;

// Device constants (initialized at runtime via cudaMemcpyToSymbol) - optimized version
// Use plain uint64_t arrays to avoid dynamic initialization issues
__constant__ uint64_t d_poseidon_round_constants_optimized[64 * 3 * 4]; // 64 rounds * 3 states * 4 limbs
__constant__ uint64_t d_poseidon_mds_matrix_optimized[3 * 3 * 4]; // 3x3 matrix * 4 limbs

// ================================
// Device Functions for Poseidon Operations - Optimized
// ================================

__device__ void cuda_add_round_constants(CudaFieldElement state[PoseidonParams::STATE_SIZE], size_t round) {
    for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
        size_t const_idx = round * PoseidonParams::STATE_SIZE + i;
        size_t limb_offset = const_idx * 4; // 4 limbs per FieldElement
        CudaFieldElement round_const = CudaFieldElement(
            d_poseidon_round_constants_optimized[limb_offset + 0],
            d_poseidon_round_constants_optimized[limb_offset + 1],
            d_poseidon_round_constants_optimized[limb_offset + 2],
            d_poseidon_round_constants_optimized[limb_offset + 3]
        );
        cuda_add(state[i], round_const, state[i]);
    }
}

__device__ void cuda_apply_sbox_optimized(CudaFieldElement state[PoseidonParams::STATE_SIZE]) {
    for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
        cuda_power5(state[i], state[i]);
    }
}

__device__ void cuda_apply_partial_sbox(CudaFieldElement state[PoseidonParams::STATE_SIZE]) {
    // Only apply S-box to the first element in partial rounds
    cuda_power5(state[0], state[0]);
}

__device__ void cuda_apply_mds_matrix(CudaFieldElement state[PoseidonParams::STATE_SIZE]) {
    // Store original state in local memory (registers)
    CudaFieldElement original_state[PoseidonParams::STATE_SIZE];
    for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
        original_state[i] = state[i];
    }
    
    // Compute MDS matrix multiplication directly into state array
    // new_state[i] = sum(MDS[i][j] * original_state[j])
    for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
        // Use local accumulator to avoid repeated memory access
        CudaFieldElement accumulator;
        accumulator.set_zero();
        
        // Accumulate: accumulator += MDS[i][j] * original_state[j]
        for (size_t j = 0; j < PoseidonParams::STATE_SIZE; ++j) {
            size_t matrix_idx = i * PoseidonParams::STATE_SIZE + j;
            size_t limb_offset = matrix_idx * 4; // 4 limbs per FieldElement
            
            // Load MDS matrix element directly from constant memory
            CudaFieldElement mds_element = CudaFieldElement(
                d_poseidon_mds_matrix_optimized[limb_offset + 0],
                d_poseidon_mds_matrix_optimized[limb_offset + 1],
                d_poseidon_mds_matrix_optimized[limb_offset + 2],
                d_poseidon_mds_matrix_optimized[limb_offset + 3]
            );
            
            // Multiply: product = MDS[i][j] * original_state[j]
            CudaFieldElement product;
            cuda_multiply(mds_element, original_state[j], product);
            
            // Add to local accumulator: accumulator += product
            cuda_add(accumulator, product, accumulator);
        }
        
        // Write final result to state[i] only once
        state[i] = accumulator;
    }
}

__device__ void cuda_permutation_optimized(CudaFieldElement state[PoseidonParams::STATE_SIZE]) {
    size_t round = 0;
    
    // First half of full rounds
    for (size_t r = 0; r < PoseidonParams::ROUNDS_FULL / 2; ++r) {
        cuda_add_round_constants(state, round++);
        cuda_apply_sbox_optimized(state);
        cuda_apply_mds_matrix(state);
    }
    
    // Partial rounds
    for (size_t r = 0; r < PoseidonParams::ROUNDS_PARTIAL; ++r) {
        cuda_add_round_constants(state, round++);
        cuda_apply_partial_sbox(state);
        cuda_apply_mds_matrix(state);
    }
    
    // Second half of full rounds
    for (size_t r = 0; r < PoseidonParams::ROUNDS_FULL / 2; ++r) {
        cuda_add_round_constants(state, round++);
        cuda_apply_sbox_optimized(state);
        cuda_apply_mds_matrix(state);
    }
}

// ================================
// Device Hash Functions for Merkle Trees - Optimized
// ================================

__device__ CudaFieldElement device_hash_n_optimized(const CudaFieldElement* children, size_t arity) {
    // Initialize with domain separator for sponge construction
    CudaFieldElement state[PoseidonParams::STATE_SIZE];
    state[0] = CudaFieldElement(3); // Same domain separator as CPU hash_multiple
    state[1] = CudaFieldElement(0); 
    state[2] = CudaFieldElement(0);
    
    // Absorb phase - process children in chunks
    size_t input_idx = 0;
    while (input_idx < arity) {
        // Add up to RATE children to the state (rate = 2, capacity = 1)
        for (size_t i = 0; i < PoseidonParams::RATE && input_idx < arity; ++i) {
            // Add to state[i + CAPACITY] = state[i + 1]
            cuda_add(state[i + PoseidonParams::CAPACITY], children[input_idx], 
                    state[i + PoseidonParams::CAPACITY]);
            input_idx++;
        }
        
        // Apply permutation
        cuda_permutation_optimized(state);
    }
    
    // Squeeze phase - return first rate element (state[CAPACITY] = state[1])
    return state[PoseidonParams::CAPACITY];
}

// ================================
// Kernel Functions - Optimized
// ================================

__global__ void batch_hash_single_kernel_optimized(const CudaFieldElement* inputs, CudaFieldElement* outputs, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        // Initialize Poseidon state: [1, input, 0]
        CudaFieldElement state[PoseidonParams::STATE_SIZE];
        state[0] = CudaFieldElement(1); // Domain separator for single hash
        state[1] = inputs[idx];         // Input
        state[2] = CudaFieldElement(0); // Zero padding
        
        // Apply full Poseidon permutation
        cuda_permutation_optimized(state);
        
        // Store result (first rate element)
        outputs[idx] = state[1];
    }
}

__global__ void batch_hash_pairs_kernel_optimized(const CudaFieldElement* left_inputs, const CudaFieldElement* right_inputs, CudaFieldElement* outputs, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        // Initialize Poseidon state: [2, left, right]
        CudaFieldElement state[PoseidonParams::STATE_SIZE];
        state[0] = CudaFieldElement(2);     // Domain separator for pair hash
        state[1] = left_inputs[idx];        // Left input
        state[2] = right_inputs[idx];       // Right input
        
        // Apply full Poseidon permutation
        cuda_permutation_optimized(state);
        
        // Store result (first rate element)
        outputs[idx] = state[1];
    }
}

__global__ void batch_permutation_kernel_optimized(CudaFieldElement* states, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        // Apply permutation to the state at index idx
        // Each state consists of PoseidonParams::STATE_SIZE (3) CudaFieldElements
        size_t state_offset = idx * PoseidonParams::STATE_SIZE;
        
        // Extract the state for this thread - now using natural assignment
        CudaFieldElement local_state[PoseidonParams::STATE_SIZE];
        for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
            local_state[i] = states[state_offset + i];
        }
        
        // Apply the Poseidon permutation
        cuda_permutation_optimized(local_state);
        
        // Write the result back - natural assignment
        for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
            states[state_offset + i] = local_state[i];
        }
    }
}

// ================================
// Host Interface Implementation - Optimized
// ================================

CudaPoseidonHashOptimized::CudaPoseidonHashOptimized() 
    : initialized_(false), optimal_batch_size_(1024), max_batch_size_(65536) {
    
    // Initialize CUDA field arithmetic first
    if (!CudaFieldArithmetic::initialize()) {
        std::cerr << "Failed to initialize CUDA field arithmetic" << std::endl;
        return;
    }
    
    // Initialize Poseidon constants on host
    PoseidonConstants::init();
    
    // Copy constants to device
    if (!copy_constants_to_device()) {
        std::cerr << "Failed to copy Poseidon constants to device" << std::endl;
        return;
    }
    
    // Determine optimal batch sizes based on device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    optimal_batch_size_ = std::min(static_cast<size_t>(prop.maxThreadsPerBlock), static_cast<size_t>(1024));
    max_batch_size_ = std::min(static_cast<size_t>(prop.maxGridSize[0] * optimal_batch_size_), static_cast<size_t>(1 << 20)); // 1M max
    
    initialized_ = true;
}

CudaPoseidonHashOptimized::~CudaPoseidonHashOptimized() {
    // __constant__ memory is automatically managed by CUDA runtime
    // No need to free constant memory
    
    CudaFieldArithmetic::cleanup();
    initialized_ = false;
}

bool CudaPoseidonHashOptimized::copy_constants_to_device() {
    // Copy round constants directly to constant memory
    // The constants are arrays of FieldElement, but we need to copy them as uint64_t arrays
    size_t round_constants_size = PoseidonParams::TOTAL_ROUNDS * PoseidonParams::STATE_SIZE * 4 * sizeof(uint64_t);
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_poseidon_round_constants_optimized, 
                                        reinterpret_cast<const uint64_t*>(PoseidonConstants::ROUND_CONSTANTS.data()), 
                                        round_constants_size));
    
    // Copy MDS matrix directly to constant memory
    size_t mds_matrix_size = PoseidonParams::STATE_SIZE * PoseidonParams::STATE_SIZE * 4 * sizeof(uint64_t);
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_poseidon_mds_matrix_optimized, 
                                        reinterpret_cast<const uint64_t*>(PoseidonConstants::MDS_MATRIX.data()), 
                                        mds_matrix_size));
    
    return true;
}

bool CudaPoseidonHashOptimized::batch_hash_single(const std::vector<FieldElement>& inputs, std::vector<FieldElement>& outputs) {
    if (!initialized_) {
        std::cerr << "CudaPoseidonHashOptimized not initialized" << std::endl;
        return false;
    }
    
    size_t count = inputs.size();
    outputs.resize(count);
    
    if (count == 0) return true;
    
    // Convert inputs to CudaFieldElement
    std::vector<CudaFieldElement> cuda_inputs;
    cuda_inputs.reserve(count);
    for (const auto& fe : inputs) {
        cuda_inputs.emplace_back(fe);
    }
    
    // Allocate device memory for CudaFieldElement
    CudaFieldElement* d_inputs;
    CudaFieldElement* d_outputs;
    
    CUDA_CHECK_RETURN(cudaMalloc(&d_inputs, count * sizeof(CudaFieldElement)));
    
    CUDA_CHECK_RETURN(cudaMalloc(&d_outputs, count * sizeof(CudaFieldElement)));
    
    // Copy input data to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_inputs, cuda_inputs.data(), count * sizeof(CudaFieldElement), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    // Use smaller block size for Poseidon kernels due to high register usage
    size_t block_size = std::min(size_t(128), std::min(optimal_batch_size_, count));
    size_t grid_size = (count + block_size - 1) / block_size;
    
    // Launch kernel
    batch_hash_single_kernel_optimized<<<grid_size, block_size>>>(d_inputs, d_outputs, count);
    
    CUDA_KERNEL_CHECK();

    // Wait for kernel completion
    cuda_sync_check();
    
    // Copy results back to host and convert to FieldElement
    std::vector<CudaFieldElement> cuda_outputs(count);
    CUDA_CHECK_RETURN(cudaMemcpy(cuda_outputs.data(), d_outputs, count * sizeof(CudaFieldElement), cudaMemcpyDeviceToHost));
    
    // Convert back to FieldElement
    for (size_t i = 0; i < count; ++i) {
        outputs[i] = FieldElement(cuda_outputs[i].limbs[0], cuda_outputs[i].limbs[1], 
                                cuda_outputs[i].limbs[2], cuda_outputs[i].limbs[3]);
    }
    
    // Clean up device memory
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    
    return true;
}

bool CudaPoseidonHashOptimized::batch_hash_pairs(const std::vector<FieldElement>& left_inputs,
                                       const std::vector<FieldElement>& right_inputs,
                                       std::vector<FieldElement>& outputs) {
    if (!initialized_) {
        std::cerr << "CudaPoseidonHashOptimized not initialized" << std::endl;
        return false;
    }
    
    if (left_inputs.size() != right_inputs.size()) {
        std::cerr << "Left and right input vectors must have the same size" << std::endl;
        return false;
    }
    
    size_t count = left_inputs.size();
    outputs.resize(count);
    
    if (count == 0) return true;
    
    // Convert inputs to CudaFieldElement
    std::vector<CudaFieldElement> cuda_left_inputs;
    std::vector<CudaFieldElement> cuda_right_inputs;
    cuda_left_inputs.reserve(count);
    cuda_right_inputs.reserve(count);
    
    for (const auto& fe : left_inputs) {
        cuda_left_inputs.emplace_back(fe);
    }
    for (const auto& fe : right_inputs) {
        cuda_right_inputs.emplace_back(fe);
    }
    
    // Allocate device memory for CudaFieldElement
    CudaFieldElement* d_left_inputs;
    CudaFieldElement* d_right_inputs;
    CudaFieldElement* d_outputs;
    
    CUDA_CHECK_RETURN(cudaMalloc(&d_left_inputs, count * sizeof(CudaFieldElement)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_right_inputs, count * sizeof(CudaFieldElement)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_outputs, count * sizeof(CudaFieldElement)));
    
    // Copy input data to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_left_inputs, cuda_left_inputs.data(), count * sizeof(CudaFieldElement), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_right_inputs, cuda_right_inputs.data(), count * sizeof(CudaFieldElement), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    // Use smaller block size for Poseidon kernels due to high register usage
    size_t block_size = std::min(size_t(128), std::min(optimal_batch_size_, count));
    size_t grid_size = (count + block_size - 1) / block_size;
    
    // Launch kernel
    batch_hash_pairs_kernel_optimized<<<grid_size, block_size>>>(d_left_inputs, d_right_inputs, d_outputs, count);
    
    CUDA_KERNEL_CHECK();
    
    // Wait for kernel completion
    cuda_sync_check();
    
    // Copy results back to host and convert to FieldElement
    std::vector<CudaFieldElement> cuda_outputs(count);
    CUDA_CHECK_RETURN(cudaMemcpy(cuda_outputs.data(), d_outputs, count * sizeof(CudaFieldElement), cudaMemcpyDeviceToHost));
    
    // Convert back to FieldElement
    for (size_t i = 0; i < count; ++i) {
        outputs[i] = FieldElement(cuda_outputs[i].limbs[0], cuda_outputs[i].limbs[1], 
                                cuda_outputs[i].limbs[2], cuda_outputs[i].limbs[3]);
    }
    
    // Clean up device memory
    cudaFree(d_left_inputs);
    cudaFree(d_right_inputs);
    cudaFree(d_outputs);
    
    return true;
}

bool CudaPoseidonHashOptimized::batch_permutation(std::vector<std::array<CudaFieldElement, PoseidonParams::STATE_SIZE>>& states) {
    if (!initialized_) {
        std::cerr << "CudaPoseidonHashOptimized not initialized" << std::endl;
        return false;
    }
    
    size_t count = states.size();
    if (count == 0) return true;
    
    // Flatten the states array for GPU processing
    std::vector<CudaFieldElement> flattened_states;
    flattened_states.reserve(count * PoseidonParams::STATE_SIZE);
    
    for (const auto& state : states) {
        for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
            flattened_states.push_back(state[i]);
        }
    }
    
    // Allocate device memory
    CudaFieldElement* d_states;
    CUDA_CHECK_RETURN(cudaMalloc(&d_states, count * PoseidonParams::STATE_SIZE * sizeof(CudaFieldElement)));
    
    // Copy data to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_states, flattened_states.data(), 
                       count * PoseidonParams::STATE_SIZE * sizeof(CudaFieldElement), 
                       cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    // Use smaller block size for Poseidon kernels due to high register usage
    size_t block_size = std::min(size_t(128), std::min(optimal_batch_size_, count));
    size_t grid_size = (count + block_size - 1) / block_size;
    
    // Launch kernel
    batch_permutation_kernel_optimized<<<grid_size, block_size>>>(d_states, count);
    
    CUDA_KERNEL_CHECK();
    
    // Wait for kernel completion
    cuda_sync_check();
    
    // Copy results back to host
    std::vector<CudaFieldElement> result_flattened(count * PoseidonParams::STATE_SIZE);
    CUDA_CHECK_RETURN(cudaMemcpy(result_flattened.data(), d_states, 
                       count * PoseidonParams::STATE_SIZE * sizeof(CudaFieldElement), 
                       cudaMemcpyDeviceToHost));
    
    // Unflatten the results back into the states array
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < PoseidonParams::STATE_SIZE; ++j) {
            states[i][j] = result_flattened[i * PoseidonParams::STATE_SIZE + j];
        }
    }
    
    // Clean up device memory
    cudaFree(d_states);
    
    return true;
}

// ================================
// Utility Functions
// ================================

size_t CudaPoseidonHashOptimized::get_optimal_batch_size() const {
    return optimal_batch_size_;
}

size_t CudaPoseidonHashOptimized::get_max_batch_size() const {
    return max_batch_size_;
}

bool CudaPoseidonHashOptimized::is_initialized() const {
    return initialized_;
}



// Memory management functions for FieldElement - kept for existing kernels that still use FieldElement
FieldElement* CudaPoseidonHashOptimized::allocate_device_memory(size_t count) {
    FieldElement* ptr;
    CUDA_MALLOC_CHECK(ptr, count * sizeof(FieldElement));
    return ptr;
}

void CudaPoseidonHashOptimized::free_device_memory(FieldElement* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

bool CudaPoseidonHashOptimized::copy_to_device(const std::vector<FieldElement>& host_data, FieldElement* device_ptr) {
    CUDA_CHECK_RETURN(cudaMemcpy(device_ptr, host_data.data(), 
                                 host_data.size() * sizeof(FieldElement), 
                                 cudaMemcpyHostToDevice));
    return true;
}

bool CudaPoseidonHashOptimized::copy_from_device(FieldElement* device_ptr, std::vector<FieldElement>& host_data, size_t count) {
    CUDA_CHECK_RETURN(cudaMemcpy(host_data.data(), device_ptr, 
                                 count * sizeof(FieldElement), 
                                 cudaMemcpyDeviceToHost));
    return true;
}

} // namespace PoseidonCUDAOptimized
} // namespace Poseidon 