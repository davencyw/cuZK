#include "cuda_field_element.cuh"
#include "../field_arithmetic.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>

namespace Poseidon {

// ================================
// Host-Only Member Functions
// ================================

#ifndef __CUDA_ARCH__

// Host-only constructor and conversion functions are now inline in the header

std::string CudaFieldElement::to_hex() const {
    std::stringstream ss;
    ss << "0x";
    for (int i = 3; i >= 0; --i) {
        ss << std::hex << std::setw(16) << std::setfill('0') << limbs[i];
    }
    return ss.str();
}

std::string CudaFieldElement::to_dec() const {
    // Convert to FieldElement and use its to_dec method
    FieldElement fe(limbs[0], limbs[1], limbs[2], limbs[3]);
    return fe.to_dec();
}

CudaFieldElement CudaFieldElement::from_hex(const std::string& hex) {
    FieldElement fe = FieldElement::from_hex(hex);
    return CudaFieldElement(fe);
}

CudaFieldElement CudaFieldElement::random() {
    FieldElement fe = FieldElement::random();
    return CudaFieldElement(fe);
}

#endif // __CUDA_ARCH__

// ================================
// Device Constants Initialization
// ================================

namespace CudaFieldOps {

void init_cuda_field_constants() {
    // Constants are already defined as __device__ const in the header, so no additional initialization needed
    // This function is here for consistency with the interface
}

} // namespace CudaFieldOps

// ================================
// Host-Device Conversion Utilities
// ================================

namespace CudaFieldConversion {

CudaFieldElement to_cuda_field_element(const FieldElement& fe) {
    return CudaFieldElement(fe.limbs[0], fe.limbs[1], fe.limbs[2], fe.limbs[3]);
}

FieldElement from_cuda_field_element(const CudaFieldElement& cfe) {
    return FieldElement(cfe.limbs[0], cfe.limbs[1], cfe.limbs[2], cfe.limbs[3]);
}

void convert_vector_to_cuda(const std::vector<FieldElement>& input, 
                           std::vector<CudaFieldElement>& output) {
    output.clear();
    output.reserve(input.size());
    
    for (const auto& fe : input) {
        output.push_back(to_cuda_field_element(fe));
    }
}

void convert_vector_from_cuda(const std::vector<CudaFieldElement>& input, 
                             std::vector<FieldElement>& output) {
    output.clear();
    output.reserve(input.size());
    
    for (const auto& cfe : input) {
        output.push_back(from_cuda_field_element(cfe));
    }
}

} // namespace CudaFieldConversion

} // namespace Poseidon 