#pragma once

#include "poseidon_interface_cuda.hpp"
#include "poseidon_cuda.cuh"

namespace Poseidon {
namespace PoseidonCUDA {

// Interface-based benchmark functions
CudaPoseidonStats benchmark_cuda_poseidon_single(IPoseidonCudaHash& hasher, size_t num_hashes, size_t batch_size = 1024);
CudaPoseidonStats benchmark_cuda_poseidon_pairs(IPoseidonCudaHash& hasher, size_t num_pairs, size_t batch_size = 1024);
CudaPoseidonStats benchmark_cuda_vs_cpu_poseidon(IPoseidonCudaHash& hasher, size_t num_hashes, size_t batch_size = 1024);

// Verification function to ensure implementations return the same results
bool verify_cuda_implementations_match(IPoseidonCudaHash& hasher1, IPoseidonCudaHash& hasher2, 
                                      const std::string& name1, const std::string& name2,
                                      size_t num_tests = 100);

} // namespace PoseidonCUDA
} // namespace Poseidon 