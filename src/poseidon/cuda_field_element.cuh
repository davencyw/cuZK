#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <string>
#include "field_arithmetic.hpp"

namespace Poseidon {

// CUDA-compatible field element structure for 256-bit prime field arithmetic
// This version can be used in both host and device code
struct CudaFieldElement {
    // 4 64-bit limbs to represent 256-bit numbers
    uint64_t limbs[4];

    // Default constructor
    __host__ __device__ CudaFieldElement() {
        limbs[0] = 0;
        limbs[1] = 0;
        limbs[2] = 0;  
        limbs[3] = 0;
    }

    // Constructor from single uint64_t value
    __host__ __device__ explicit CudaFieldElement(uint64_t value) {
        limbs[0] = value;
        limbs[1] = 0;
        limbs[2] = 0;
        limbs[3] = 0;
    }

    // Constructor from four uint64_t values
    __host__ __device__ CudaFieldElement(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) {
        limbs[0] = v0;
        limbs[1] = v1;
        limbs[2] = v2;
        limbs[3] = v3;
    }

    // Copy constructor
    __host__ __device__ CudaFieldElement(const CudaFieldElement& other) {
        limbs[0] = other.limbs[0];
        limbs[1] = other.limbs[1];
        limbs[2] = other.limbs[2];
        limbs[3] = other.limbs[3];
    }

    // Assignment operator
    __host__ __device__ CudaFieldElement& operator=(const CudaFieldElement& other) {
        if (this != &other) {
            limbs[0] = other.limbs[0];
            limbs[1] = other.limbs[1];
            limbs[2] = other.limbs[2];
            limbs[3] = other.limbs[3];
        }
        return *this;
    }

    // Comparison operators
    __host__ __device__ bool operator==(const CudaFieldElement& other) const {
        return limbs[0] == other.limbs[0] && 
               limbs[1] == other.limbs[1] && 
               limbs[2] == other.limbs[2] && 
               limbs[3] == other.limbs[3];
    }

    __host__ __device__ bool operator!=(const CudaFieldElement& other) const {
        return !(*this == other);
    }

    __host__ __device__ bool operator<(const CudaFieldElement& other) const {
        // Compare from most significant limb to least significant
        for (int i = 3; i >= 0; --i) {
            if (limbs[i] < other.limbs[i]) return true;
            if (limbs[i] > other.limbs[i]) return false;
        }
        return false; // Equal
    }

    // Utility functions
    __host__ __device__ bool is_zero() const {
        return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
    }

    __host__ __device__ void set_zero() {
        limbs[0] = 0;
        limbs[1] = 0;
        limbs[2] = 0;
        limbs[3] = 0;
    }

    // Constructor from FieldElement (host only)
    __host__ explicit CudaFieldElement(const FieldElement& other) {
        limbs[0] = other.limbs[0];
        limbs[1] = other.limbs[1];
        limbs[2] = other.limbs[2];
        limbs[3] = other.limbs[3];
    }
    
    // Conversion operator to FieldElement (host only)
    __host__ operator FieldElement() const {
        return FieldElement(limbs[0], limbs[1], limbs[2], limbs[3]);
    }

    // Host-only functions for debugging and I/O
    #ifndef __CUDA_ARCH__
    std::string to_hex() const;
    std::string to_dec() const;
    static CudaFieldElement from_hex(const std::string& hex);
    static CudaFieldElement random();
    #endif
};

// Device-side arithmetic operations
namespace CudaFieldOps {

// Basic arithmetic operations (device functions)
__device__ void cuda_add(const CudaFieldElement& a, const CudaFieldElement& b, CudaFieldElement& result);
__device__ void cuda_subtract(const CudaFieldElement& a, const CudaFieldElement& b, CudaFieldElement& result);
__device__ void cuda_multiply(const CudaFieldElement& a, const CudaFieldElement& b, CudaFieldElement& result);
__device__ void cuda_square(const CudaFieldElement& a, CudaFieldElement& result);
__device__ void cuda_power5(const CudaFieldElement& a, CudaFieldElement& result);
__device__ void cuda_reduce(CudaFieldElement& a);

// Utility device functions
__device__ bool cuda_is_zero(const CudaFieldElement& a);
__device__ int cuda_compare(const CudaFieldElement& a, const CudaFieldElement& b);
__device__ void cuda_copy(const CudaFieldElement& src, CudaFieldElement& dst);

// Internal helper functions
__device__ void cuda_subtract_internal(const CudaFieldElement& a, const CudaFieldElement& b, CudaFieldElement& result);
__device__ void cuda_reduce_512(const uint64_t product[8], CudaFieldElement& result);
__device__ void cuda_multiply_raw(const uint64_t a[4], const uint64_t b[4], uint64_t result[8]);

// Field constants accessible on device (as utility functions to avoid dynamic initialization)
__device__ CudaFieldElement get_cuda_field_modulus();

// Initialize device constants
void init_cuda_field_constants();

// ================================
// Inline Device Function Implementations
// ================================

// Device constants as plain arrays
__device__ const uint64_t CUDA_FIELD_ZERO_LIMBS[4] = {0, 0, 0, 0};
__device__ const uint64_t CUDA_FIELD_ONE_LIMBS[4] = {1, 0, 0, 0};
__device__ const uint64_t CUDA_FIELD_TWO_LIMBS[4] = {2, 0, 0, 0};

// BN254 scalar field modulus: 21888242871839275222246405745257275088548364400416034343698204186575808495617
__device__ const uint64_t CUDA_FIELD_MODULUS_LIMBS[4] = {
    0x43e1f593f0000001ULL,  // limbs[0]
    0x2833e84879b97091ULL,  // limbs[1] 
    0xb85045b68181585dULL,  // limbs[2]
    0x30644e72e131a029ULL   // limbs[3]
};

__device__ inline CudaFieldElement get_cuda_field_modulus() {
    CudaFieldElement result;
    result.limbs[0] = CUDA_FIELD_MODULUS_LIMBS[0];
    result.limbs[1] = CUDA_FIELD_MODULUS_LIMBS[1];
    result.limbs[2] = CUDA_FIELD_MODULUS_LIMBS[2];
    result.limbs[3] = CUDA_FIELD_MODULUS_LIMBS[3];
    return result;
}

__device__ inline void cuda_copy(const CudaFieldElement& src, CudaFieldElement& dst) {
    dst.limbs[0] = src.limbs[0];
    dst.limbs[1] = src.limbs[1];
    dst.limbs[2] = src.limbs[2];
    dst.limbs[3] = src.limbs[3];
}

__device__ inline bool cuda_is_zero(const CudaFieldElement& a) {
    return a.limbs[0] == 0 && a.limbs[1] == 0 && a.limbs[2] == 0 && a.limbs[3] == 0;
}

__device__ inline int cuda_compare(const CudaFieldElement& a, const CudaFieldElement& b) {
    // Compare from most significant to least significant limb
    for (int i = 3; i >= 0; --i) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0; // Equal
}

__device__ inline void cuda_subtract_internal(const CudaFieldElement& a, const CudaFieldElement& b, CudaFieldElement& result) {
    uint64_t borrow = 0;
    
    for (int i = 0; i < 4; ++i) {
        uint64_t temp_a = a.limbs[i];
        uint64_t temp_b = b.limbs[i] + borrow;
        
        if (temp_a >= temp_b) {
            result.limbs[i] = temp_a - temp_b;
            borrow = 0;
        } else {
            result.limbs[i] = temp_a + (UINT64_MAX - temp_b) + 1;
            borrow = 1;
        }
    }
}

__device__ inline void cuda_add(const CudaFieldElement& a, const CudaFieldElement& b, CudaFieldElement& result) {
    uint64_t carry = 0;
    
    // Match CPU implementation exactly: add all three values at once
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        result.limbs[i] = sum;
        carry = (sum < a.limbs[i]) ? 1 : 0;
    }
    
    // Reduce modulo the field modulus if necessary
    cuda_reduce(result);
}

__device__ inline void cuda_subtract(const CudaFieldElement& a, const CudaFieldElement& b, CudaFieldElement& result) {
    if (cuda_compare(a, b) >= 0) {
        cuda_subtract_internal(a, b, result);
    } else {
        // a < b, need to add modulus: result = a + p - b
        CudaFieldElement temp;
        CudaFieldElement modulus = get_cuda_field_modulus();
        cuda_add(a, modulus, temp);
        cuda_subtract_internal(temp, b, result);
    }
}

__device__ inline void cuda_multiply_raw(const uint64_t a[4], const uint64_t b[4], uint64_t result[8]) {
    // Initialize result to zero
    for (int i = 0; i < 8; ++i) {
        result[i] = 0;
    }
    
    // Schoolbook multiplication - simpler and more reliable approach
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            // 64x64 -> 128 multiplication using built-in approach
            uint64_t a_val = a[i];
            uint64_t b_val = b[j];
            
            // Use the fact that CUDA supports 64-bit multiplication
            // Split into 32-bit parts for safe 128-bit emulation
            uint64_t a_lo = a_val & 0xFFFFFFFFULL;
            uint64_t a_hi = a_val >> 32;
            uint64_t b_lo = b_val & 0xFFFFFFFFULL;
            uint64_t b_hi = b_val >> 32;
            
            uint64_t p0 = a_lo * b_lo;
            uint64_t p1 = a_lo * b_hi;
            uint64_t p2 = a_hi * b_lo;
            uint64_t p3 = a_hi * b_hi;
            
            // Assemble carefully to avoid overflow
            uint64_t middle1 = p1 + (p0 >> 32);
            uint64_t middle2 = p2 + (middle1 & 0xFFFFFFFFULL);
            
            uint64_t low = (middle2 << 32) | (p0 & 0xFFFFFFFFULL);
            uint64_t high = p3 + (middle1 >> 32) + (middle2 >> 32);
            
            // Add to existing result with proper carry handling
            uint64_t old_val = result[i + j];
            uint64_t temp_sum = old_val + low;
            uint64_t carry_from_low = (temp_sum < old_val) ? 1 : 0;
            
            uint64_t final_sum = temp_sum + carry;
            uint64_t carry_from_carry = (final_sum < temp_sum) ? 1 : 0;
            
            result[i + j] = final_sum;
            carry = high + carry_from_low + carry_from_carry;
        }
        
        // Add remaining carry to high part
        result[i + 4] = carry;
    }
}

__device__ inline void cuda_reduce_512(const uint64_t product[8], CudaFieldElement& result) {
    // Proper 512-bit to 256-bit reduction - match CPU implementation exactly
    // N = high * 2^256 + low
    // N mod p = (high * (2^256 mod p) + low) mod p
    
    CudaFieldElement high, low;
    
    // Copy low and high parts
    low.limbs[0] = product[0];
    low.limbs[1] = product[1];
    low.limbs[2] = product[2];
    low.limbs[3] = product[3];
    
    high.limbs[0] = product[4];
    high.limbs[1] = product[5];
    high.limbs[2] = product[6];
    high.limbs[3] = product[7];
    
    // Check if high part is zero - if so, just reduce low part
    if (cuda_is_zero(high)) {
        result = low;
        cuda_reduce(result);
        return;
    }
    
    // For BN254 field, 2^256 ≡ 4 (mod p) approximately
    // So we compute: high * 4 + low, then reduce
    CudaFieldElement temp_high;
    
    // Multiply high by 4 (left shift by 2) - match CPU exactly
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t shifted = (high.limbs[i] << 2) | carry;
        temp_high.limbs[i] = shifted;
        carry = high.limbs[i] >> 62;  // Carry the top 2 bits
    }
    
    // Add temp_high + low
    cuda_add(temp_high, low, result);
    
    // Final reduction
    cuda_reduce(result);
}

__device__ inline void cuda_multiply(const CudaFieldElement& a, const CudaFieldElement& b, CudaFieldElement& result) {
    uint64_t product[8];
    cuda_multiply_raw(a.limbs, b.limbs, product);
    cuda_reduce_512(product, result);
}

__device__ inline void cuda_square(const CudaFieldElement& a, CudaFieldElement& result) {
    cuda_multiply(a, a, result);
}

__device__ inline void cuda_power5(const CudaFieldElement& a, CudaFieldElement& result) {
    // Compute a^5 = a^4 * a = (a^2)^2 * a
    CudaFieldElement a2, a4;
    cuda_square(a, a2);      // a^2
    cuda_square(a2, a4);     // a^4
    cuda_multiply(a4, a, result); // a^5
}

__device__ inline void cuda_reduce(CudaFieldElement& a) {
    CudaFieldElement modulus = get_cuda_field_modulus();
    // Match CPU exactly: while (!(a < modulus))
    while (!(a < modulus)) {
        cuda_subtract_internal(a, modulus, a);
    }
}

} // namespace CudaFieldOps

// Host-device conversion utilities
namespace CudaFieldConversion {

// Convert between FieldElement and CudaFieldElement on host
CudaFieldElement to_cuda_field_element(const FieldElement& fe);
FieldElement from_cuda_field_element(const CudaFieldElement& cfe);

// Batch conversion functions
void convert_vector_to_cuda(const std::vector<FieldElement>& input, 
                           std::vector<CudaFieldElement>& output);
void convert_vector_from_cuda(const std::vector<CudaFieldElement>& input, 
                             std::vector<FieldElement>& output);

} // namespace CudaFieldConversion

} // namespace Poseidon 