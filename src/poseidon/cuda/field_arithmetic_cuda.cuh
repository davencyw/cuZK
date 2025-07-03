#pragma once

#include "field_arithmetic.hpp"
#include "cuda_field_element.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

namespace Poseidon {

// CUDA-specific field element operations
namespace CudaFieldOps {

// All field arithmetic operations now use CudaFieldElement directly
// See cuda_field_element.cuh for the clean CudaFieldElement API

// Kernel function declarations for batch operations
__global__ void batch_add_kernel(const CudaFieldElement* a, const CudaFieldElement* b, CudaFieldElement* result, size_t count);
__global__ void batch_subtract_kernel(const CudaFieldElement* a, const CudaFieldElement* b, CudaFieldElement* result, size_t count);
__global__ void batch_multiply_kernel(const CudaFieldElement* a, const CudaFieldElement* b, CudaFieldElement* result, size_t count);
__global__ void batch_square_kernel(const CudaFieldElement* input, CudaFieldElement* result, size_t count);
__global__ void batch_power5_kernel(const CudaFieldElement* input, CudaFieldElement* result, size_t count);

// Host interface functions
class CudaFieldArithmetic {
public:
    // Initialize CUDA context and copy constants to device
    static bool initialize();
    
    // Cleanup CUDA resources
    static void cleanup();
    
    // Batch operations on GPU
    static bool batch_add(const std::vector<FieldElement>& a, 
                         const std::vector<FieldElement>& b, 
                         std::vector<FieldElement>& result);
    
    static bool batch_subtract(const std::vector<FieldElement>& a, 
                              const std::vector<FieldElement>& b, 
                              std::vector<FieldElement>& result);
    
    static bool batch_multiply(const std::vector<FieldElement>& a, 
                              const std::vector<FieldElement>& b, 
                              std::vector<FieldElement>& result);
    
    static bool batch_square(const std::vector<FieldElement>& input, 
                            std::vector<FieldElement>& result);
    
    static bool batch_power5(const std::vector<FieldElement>& input, 
                            std::vector<FieldElement>& result);
    
    // Single operations on GPU (useful for large computations)
    static bool gpu_add(const FieldElement& a, const FieldElement& b, FieldElement& result);
    static bool gpu_subtract(const FieldElement& a, const FieldElement& b, FieldElement& result);
    static bool gpu_multiply(const FieldElement& a, const FieldElement& b, FieldElement& result);
    static bool gpu_square(const FieldElement& a, FieldElement& result);
    static bool gpu_power5(const FieldElement& a, FieldElement& result);
    
    // Utility functions
    static int get_device_count();
    static void print_device_info();
    static size_t get_optimal_block_size();
    
private:
    static bool initialized_;
    static int device_id_;
    static size_t optimal_block_size_;
    
    // Device memory management
    static FieldElement* allocate_device_memory(size_t count);
    static void free_device_memory(FieldElement* ptr);
    static bool copy_to_device(const std::vector<FieldElement>& host_data, FieldElement* device_ptr);
    static bool copy_from_device(FieldElement* device_ptr, std::vector<FieldElement>& host_data, size_t count);
    
    // CudaFieldElement device memory management
    static CudaFieldElement* allocate_cuda_device_memory(size_t count);
    static void free_device_memory(CudaFieldElement* ptr);
    static bool copy_to_cuda_device(const std::vector<FieldElement>& host_data, CudaFieldElement* device_ptr);
    static bool copy_from_cuda_device(CudaFieldElement* device_ptr, std::vector<FieldElement>& host_data, size_t count);
    
};

// Performance benchmarking for CUDA operations
struct CudaHashingStats {
    double total_time_ms;
    double avg_time_per_operation_ns;
    size_t operations_per_second;
    size_t total_operations;
};

CudaHashingStats benchmark_cuda_field_operations(size_t num_operations, size_t batch_size = 1024);

} // namespace CudaFieldOps
} // namespace Poseidon 