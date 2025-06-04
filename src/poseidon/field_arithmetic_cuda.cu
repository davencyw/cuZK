#include "field_arithmetic_cuda.cuh"
#include "cuda_field_element.cuh"
#include "field_arithmetic.hpp"
#include "../common/error_handling.hpp"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstring>

// Note: Using CudaFieldElement functions from cuda_field_element.cuh directly

namespace Poseidon {
namespace CudaFieldOps {

// Use standardized CUDA error handling - these are macros, imported via include
using cuZK::ErrorHandling::cuda_sync_check;

// Static member initialization
bool CudaFieldArithmetic::initialized_ = false;
int CudaFieldArithmetic::device_id_ = 0;
size_t CudaFieldArithmetic::optimal_block_size_ = 256;

// Device constants - using uint64_t arrays to avoid dynamic initialization
__device__ uint64_t d_MODULUS[4];
__device__ uint64_t d_ZERO[4];
__device__ uint64_t d_ONE[4];
__device__ uint64_t d_TWO[4];

// Helper device functions to work with uint64_t arrays
__device__ void copy_limbs(const uint64_t src[4], uint64_t dst[4]) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
}

__device__ void copy_from_field_element(const FieldElement& src, uint64_t dst[4]) {
    dst[0] = src.limbs[0];
    dst[1] = src.limbs[1];
    dst[2] = src.limbs[2];
    dst[3] = src.limbs[3];
}

__device__ void copy_to_field_element(const uint64_t src[4], FieldElement& dst) {
    dst.limbs[0] = src[0];
    dst.limbs[1] = src[1];
    dst.limbs[2] = src[2];
    dst.limbs[3] = src[3];
}

__device__ bool is_less_than_array(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

__device__ bool is_zero_array(const uint64_t a[4]) {
    return a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0;
}

__device__ void add_arrays(const uint64_t a[4], const uint64_t b[4], uint64_t result[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = a[i] + carry;
        carry = (sum < a[i]) ? 1 : 0;
        
        uint64_t final_sum = sum + b[i];
        if (final_sum < sum) carry = 1;
        
        result[i] = final_sum;
    }
}

__device__ void subtract_arrays(const uint64_t a[4], const uint64_t b[4], uint64_t result[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t temp_a = a[i];
        uint64_t temp_b = b[i] + borrow;
        
        if (temp_a >= temp_b) {
            result[i] = temp_a - temp_b;
            borrow = 0;
        } else {
            result[i] = temp_a + (UINT64_MAX - temp_b) + 1;
            borrow = 1;
        }
    }
}

__device__ void reduce_array(uint64_t a[4]) {
    // Simple reduction by comparing with modulus - match CPU version exactly
    while (!is_less_than_array(a, d_MODULUS) && !is_zero_array(a)) {
        subtract_arrays(a, d_MODULUS, a);
    }
}

__device__ void reduce_512_array(const uint64_t product[8], uint64_t result[4]) {
    // 512-bit to 256-bit reduction
    // N = high * 2^256 + low
    // N mod p = (high * (2^256 mod p) + low) mod p
    
    uint64_t low[4], high[4];
    
    // Copy low and high parts
    for (int i = 0; i < 4; ++i) {
        low[i] = product[i];
        high[i] = product[i + 4];
    }
    
    // Check if high part is zero - if so, just reduce low part
    if (is_zero_array(high)) {
        copy_limbs(low, result);
        reduce_array(result);
        return;
    }
    
    // For BN254 field, 2^256 ≡ 4 (mod p) approximately
    // So we compute: high * 4 + low, then reduce
    uint64_t temp_high[4];
    
    // Multiply high by 4 (left shift by 2)
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t shifted = (high[i] << 2) | carry;
        temp_high[i] = shifted;
        carry = high[i] >> 62;  // Carry the top 2 bits
    }
    
    // Add temp_high + low
    add_arrays(temp_high, low, result);
    
    // Final reduction
    reduce_array(result);
}

__device__ void multiply_arrays(const uint64_t a[4], const uint64_t b[4], uint64_t result[4]) {
    uint64_t product[8] = {0};
    
    // Use proper 128-bit arithmetic like CPU version - mimic __uint128_t behavior
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            // Simple 64x64 -> 128 multiplication using 32-bit parts
            uint64_t a_val = a[i];
            uint64_t b_val = b[j];
            
            uint64_t a_lo = a_val & 0xFFFFFFFFULL;
            uint64_t a_hi = a_val >> 32;
            uint64_t b_lo = b_val & 0xFFFFFFFFULL;
            uint64_t b_hi = b_val >> 32;
            
            uint64_t p0 = a_lo * b_lo;
            uint64_t p1 = a_lo * b_hi;
            uint64_t p2 = a_hi * b_lo;
            uint64_t p3 = a_hi * b_hi;
            
            // Assemble the 128-bit result
            uint64_t middle = p1 + p2 + (p0 >> 32);
            uint64_t prod_low = (middle << 32) | (p0 & 0xFFFFFFFFULL);
            uint64_t prod_high = p3 + (middle >> 32);
            
            // Add to existing result with carry propagation
            uint64_t sum1 = product[i + j] + prod_low + carry;
            uint64_t carry1 = (sum1 < product[i + j]) ? 1 : 0;
            if (carry && sum1 == product[i + j]) carry1 = 1;
            
            product[i + j] = sum1;
            carry = prod_high + carry1;
        }
        if (i + 4 < 8) {
            product[i + 4] = carry;
        }
    }
    
    // Reduce the 512-bit product modulo the field modulus
    reduce_512_array(product, result);
}

__device__ void square_array(const uint64_t a[4], uint64_t result[4]) {
    multiply_arrays(a, a, result);
}

__device__ void power5_array(const uint64_t a[4], uint64_t result[4]) {
    uint64_t a2[4], a4[4];
    square_array(a, a2);
    square_array(a2, a4);
    multiply_arrays(a4, a, result);
}

// Kernel implementations - work directly with CudaFieldElement
__global__ void batch_add_kernel(const CudaFieldElement* a, const CudaFieldElement* b, CudaFieldElement* result, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        CudaFieldOps::cuda_add(a[idx], b[idx], result[idx]);
    }
}

__global__ void batch_subtract_kernel(const CudaFieldElement* a, const CudaFieldElement* b, CudaFieldElement* result, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        CudaFieldOps::cuda_subtract(a[idx], b[idx], result[idx]);
    }
}

__global__ void batch_multiply_kernel(const CudaFieldElement* a, const CudaFieldElement* b, CudaFieldElement* result, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        CudaFieldOps::cuda_multiply(a[idx], b[idx], result[idx]);
    }
}

__global__ void batch_square_kernel(const CudaFieldElement* input, CudaFieldElement* result, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        CudaFieldOps::cuda_square(input[idx], result[idx]);
    }
}

__global__ void batch_power5_kernel(const CudaFieldElement* input, CudaFieldElement* result, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        CudaFieldOps::cuda_power5(input[idx], result[idx]);
    }
}

// Host interface implementations
bool CudaFieldArithmetic::initialize() {
    if (initialized_) return true;
    
        // Check for CUDA devices
    int device_count;
    CUDA_CHECK_CLEANUP(cudaGetDeviceCount(&device_count), );
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }

    // Set device
    CUDA_CHECK_CLEANUP(cudaSetDevice(device_id_), );
    
    // Initialize field constants first
    FieldConstants::init();
    
    // Initialize CUDA field constants
    Poseidon::CudaFieldOps::init_cuda_field_constants();
    
    // Copy constants to device
    CUDA_CHECK_VOID(cudaMemcpyToSymbol(d_MODULUS, FieldConstants::MODULUS.limbs, 4 * sizeof(uint64_t)));
    
    CUDA_CHECK_VOID(cudaMemcpyToSymbol(d_ZERO, FieldConstants::ZERO.limbs, 4 * sizeof(uint64_t)));
    
    CUDA_CHECK_VOID(cudaMemcpyToSymbol(d_ONE, FieldConstants::ONE.limbs, 4 * sizeof(uint64_t)));
    
    CUDA_CHECK_VOID(cudaMemcpyToSymbol(d_TWO, FieldConstants::TWO.limbs, 4 * sizeof(uint64_t)));
    
    // Determine optimal block size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    optimal_block_size_ = std::min(prop.maxThreadsPerBlock, 1024);
    
    initialized_ = true;
    
    return true;
}

void CudaFieldArithmetic::cleanup() {
    if (initialized_) {
        cudaDeviceReset();
        initialized_ = false;
    }
}

bool CudaFieldArithmetic::batch_add(const std::vector<FieldElement>& a, 
                                   const std::vector<FieldElement>& b, 
                                   std::vector<FieldElement>& result) {
    if (!initialized_) {
        std::cerr << "CUDA not initialized. Call initialize() first." << std::endl;
        return false;
    }
    
    if (a.size() != b.size()) {
        std::cerr << "Input vectors must have the same size" << std::endl;
        return false;
    }
    
    size_t count = a.size();
    result.resize(count);
    
    // Handle empty vectors
    if (count == 0) {
        return true;
    }
    
    // Allocate device memory for CudaFieldElement
    CudaFieldElement* d_a = allocate_cuda_device_memory(count);
    CudaFieldElement* d_b = allocate_cuda_device_memory(count);
    CudaFieldElement* d_result = allocate_cuda_device_memory(count);
    
    if (!d_a || !d_b || !d_result) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
        return false;
    }
    
    // Copy data to device with conversion
    if (!copy_to_cuda_device(a, d_a) || !copy_to_cuda_device(b, d_b)) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
        return false;
    }
    
    // Launch kernel
    size_t block_size = optimal_block_size_;
    size_t grid_size = (count + block_size - 1) / block_size;
    
    batch_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_result, count);
    
    CUDA_CHECK_CLEANUP(cudaGetLastError(), {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
    });
    
    // Wait for completion
    if (!cuda_sync_check()) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
        return false;
    }
    
    // Copy result back with conversion
    bool success = copy_from_cuda_device(d_result, result, count);
    
    // Cleanup
    free_device_memory(d_a);
    free_device_memory(d_b);
    free_device_memory(d_result);
    
    return success;
}

bool CudaFieldArithmetic::batch_subtract(const std::vector<FieldElement>& a, 
                                        const std::vector<FieldElement>& b, 
                                        std::vector<FieldElement>& result) {
    if (!initialized_) return false;
    if (a.size() != b.size()) return false;
    
    size_t count = a.size();
    result.resize(count);
    
    // Handle empty vectors
    if (count == 0) {
        return true;
    }
    
    CudaFieldElement* d_a = allocate_cuda_device_memory(count);
    CudaFieldElement* d_b = allocate_cuda_device_memory(count);
    CudaFieldElement* d_result = allocate_cuda_device_memory(count);
    
    if (!d_a || !d_b || !d_result || !copy_to_cuda_device(a, d_a) || !copy_to_cuda_device(b, d_b)) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
        return false;
    }
    
    size_t block_size = optimal_block_size_;
    size_t grid_size = (count + block_size - 1) / block_size;
    
    batch_subtract_kernel<<<grid_size, block_size>>>(d_a, d_b, d_result, count);
    
    CUDA_CHECK_CLEANUP(cudaGetLastError(), {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
    });
    
    if (!cuda_sync_check()) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
        return false;
    }
    
    bool success = copy_from_cuda_device(d_result, result, count);
    
    free_device_memory(d_a);
    free_device_memory(d_b);
    free_device_memory(d_result);
    
    return success;
}

bool CudaFieldArithmetic::batch_multiply(const std::vector<FieldElement>& a, 
                                        const std::vector<FieldElement>& b, 
                                        std::vector<FieldElement>& result) {
    if (!initialized_) return false;
    if (a.size() != b.size()) return false;
    
    size_t count = a.size();
    result.resize(count);
    
    // Handle empty vectors
    if (count == 0) {
        return true;
    }
    
    CudaFieldElement* d_a = allocate_cuda_device_memory(count);
    CudaFieldElement* d_b = allocate_cuda_device_memory(count);
    CudaFieldElement* d_result = allocate_cuda_device_memory(count);
    
    if (!d_a || !d_b || !d_result || !copy_to_cuda_device(a, d_a) || !copy_to_cuda_device(b, d_b)) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
        return false;
    }
    
    size_t block_size = optimal_block_size_;
    size_t grid_size = (count + block_size - 1) / block_size;
    
    batch_multiply_kernel<<<grid_size, block_size>>>(d_a, d_b, d_result, count);
    
    CUDA_CHECK_CLEANUP(cudaGetLastError(), {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
    });
    
    if (!cuda_sync_check()) {
        free_device_memory(d_a);
        free_device_memory(d_b);
        free_device_memory(d_result);
        return false;
    }
    
    bool success = copy_from_cuda_device(d_result, result, count);
    
    free_device_memory(d_a);
    free_device_memory(d_b);
    free_device_memory(d_result);
    
    return success;
}

bool CudaFieldArithmetic::batch_square(const std::vector<FieldElement>& input, 
                                      std::vector<FieldElement>& result) {
    if (!initialized_) return false;
    
    size_t count = input.size();
    result.resize(count);
    
    // Handle empty vectors
    if (count == 0) {
        return true;
    }
    
    CudaFieldElement* d_input = allocate_cuda_device_memory(count);
    CudaFieldElement* d_result = allocate_cuda_device_memory(count);
    
    if (!d_input || !d_result || !copy_to_cuda_device(input, d_input)) {
        free_device_memory(d_input);
        free_device_memory(d_result);
        return false;
    }
    
    size_t block_size = optimal_block_size_;
    size_t grid_size = (count + block_size - 1) / block_size;
    
    batch_square_kernel<<<grid_size, block_size>>>(d_input, d_result, count);
    
    CUDA_CHECK_CLEANUP(cudaGetLastError(), {
        free_device_memory(d_input);
        free_device_memory(d_result);
    });
    
    if (!cuda_sync_check()) {
        free_device_memory(d_input);
        free_device_memory(d_result);
        return false;
    }
    
    bool success = copy_from_cuda_device(d_result, result, count);
    
    free_device_memory(d_input);
    free_device_memory(d_result);
    
    return success;
}

bool CudaFieldArithmetic::batch_power5(const std::vector<FieldElement>& input, 
                                      std::vector<FieldElement>& result) {
    if (!initialized_) return false;
    
    size_t count = input.size();
    result.resize(count);
    
    // Handle empty vectors
    if (count == 0) {
        return true;
    }
    
    CudaFieldElement* d_input = allocate_cuda_device_memory(count);
    CudaFieldElement* d_result = allocate_cuda_device_memory(count);
    
    if (!d_input || !d_result || !copy_to_cuda_device(input, d_input)) {
        free_device_memory(d_input);
        free_device_memory(d_result);
        return false;
    }
    
    size_t block_size = optimal_block_size_;
    size_t grid_size = (count + block_size - 1) / block_size;
    
    batch_power5_kernel<<<grid_size, block_size>>>(d_input, d_result, count);
    
    CUDA_CHECK_CLEANUP(cudaGetLastError(), {
        free_device_memory(d_input);
        free_device_memory(d_result);
    });
    
    if (!cuda_sync_check()) {
        free_device_memory(d_input);
        free_device_memory(d_result);
        return false;
    }
    
    bool success = copy_from_cuda_device(d_result, result, count);
    
    free_device_memory(d_input);
    free_device_memory(d_result);
    
    return success;
}

// Utility function implementations
int CudaFieldArithmetic::get_device_count() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

void CudaFieldArithmetic::print_device_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    
    std::cout << "CUDA Device Info:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Optimal Block Size: " << optimal_block_size_ << std::endl;
}

size_t CudaFieldArithmetic::get_optimal_block_size() {
    return optimal_block_size_;
}

// Private helper functions
FieldElement* CudaFieldArithmetic::allocate_device_memory(size_t count) {
    FieldElement* ptr;
    CUDA_MALLOC_CHECK(ptr, count * sizeof(FieldElement));
    return ptr;
}

void CudaFieldArithmetic::free_device_memory(FieldElement* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

bool CudaFieldArithmetic::copy_to_device(const std::vector<FieldElement>& host_data, FieldElement* device_ptr) {
    CUDA_CHECK_RETURN(cudaMemcpy(device_ptr, host_data.data(), 
                                 host_data.size() * sizeof(FieldElement), 
                                 cudaMemcpyHostToDevice));
    return true;
}

bool CudaFieldArithmetic::copy_from_device(FieldElement* device_ptr, std::vector<FieldElement>& host_data, size_t count) {
    CUDA_CHECK_RETURN(cudaMemcpy(host_data.data(), device_ptr, 
                                 count * sizeof(FieldElement), 
                                 cudaMemcpyDeviceToHost));
    return true;
}

// New helper functions for CudaFieldElement arrays
CudaFieldElement* CudaFieldArithmetic::allocate_cuda_device_memory(size_t count) {
    CudaFieldElement* ptr;
    CUDA_MALLOC_CHECK(ptr, count * sizeof(CudaFieldElement));
    return ptr;
}

void CudaFieldArithmetic::free_device_memory(CudaFieldElement* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

bool CudaFieldArithmetic::copy_to_cuda_device(const std::vector<FieldElement>& host_data, CudaFieldElement* device_ptr) {
    // Convert on host first, then copy to device
    std::vector<CudaFieldElement> cuda_host_data;
    cuda_host_data.reserve(host_data.size());
    
    for (const auto& fe : host_data) {
        cuda_host_data.emplace_back(fe);
    }
    
    CUDA_CHECK_RETURN(cudaMemcpy(device_ptr, cuda_host_data.data(), 
                                 cuda_host_data.size() * sizeof(CudaFieldElement), 
                                 cudaMemcpyHostToDevice));
    return true;
}

bool CudaFieldArithmetic::copy_from_cuda_device(CudaFieldElement* device_ptr, std::vector<FieldElement>& host_data, size_t count) {
    // Copy from device to host, then convert
    std::vector<CudaFieldElement> cuda_host_data(count);
    
    CUDA_CHECK_RETURN(cudaMemcpy(cuda_host_data.data(), device_ptr, 
                                 count * sizeof(CudaFieldElement), 
                                 cudaMemcpyDeviceToHost));
    
    host_data.clear();
    host_data.reserve(count);
    for (const auto& cfe : cuda_host_data) {
        host_data.emplace_back(cfe);
    }
    
    return true;
}



// Benchmarking function
CudaHashingStats benchmark_cuda_field_operations(size_t num_operations, size_t batch_size) {
    CudaHashingStats stats = {};
    
    if (!CudaFieldArithmetic::initialize()) {
        std::cerr << "Failed to initialize CUDA for benchmarking" << std::endl;
        return stats;
    }
    
    // Generate test data
    std::vector<FieldElement> input_a, input_b, result;
    for (size_t i = 0; i < batch_size; ++i) {
        input_a.push_back(FieldElement::random());
        input_b.push_back(FieldElement::random());
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run operations
    size_t completed_ops = 0;
    for (size_t i = 0; i < num_operations; i += batch_size) {
        size_t current_batch = std::min(batch_size, num_operations - i);
        
        // Resize inputs if needed
        if (current_batch != input_a.size()) {
            input_a.resize(current_batch);
            input_b.resize(current_batch);
        }
        
        // Test multiplication (most expensive operation)
        if (CudaFieldArithmetic::batch_multiply(input_a, input_b, result)) {
            completed_ops += current_batch;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    stats.total_time_ms = duration.count() / 1000000.0;
    stats.total_operations = completed_ops;
    if (completed_ops > 0) {
        stats.avg_time_per_operation_ns = static_cast<double>(duration.count()) / completed_ops;
        stats.operations_per_second = static_cast<size_t>(1000000000.0 / stats.avg_time_per_operation_ns);
    }
    
    // Estimate memory usage (using CudaFieldElement size)
    stats.gpu_memory_used_mb = (batch_size * 3 * sizeof(CudaFieldElement)) / (1024 * 1024);
    
    return stats;
}

} // namespace CudaFieldOps
} // namespace Poseidon 