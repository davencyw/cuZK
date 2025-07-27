#include "poseidon_cuda_benchmarks.hpp"
#include "poseidon_interface_cuda.hpp"
#include "../poseidon.hpp"
#include "../field_arithmetic.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>

namespace Poseidon {
namespace PoseidonCUDA {

CudaPoseidonStats benchmark_cuda_poseidon_single(IPoseidonCudaHash& hasher, size_t num_hashes, size_t batch_size) {
    CudaPoseidonStats stats = {};
    
    if (!hasher.is_initialized()) {
        std::cerr << "CUDA Poseidon hasher not initialized for benchmarking" << std::endl;
        return stats;
    }
    
    // Generate test data
    std::vector<FieldElement> inputs;
    inputs.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        inputs.push_back(FieldElement::random());
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run hash operations
    size_t completed_hashes = 0;
    std::vector<FieldElement> outputs;
    
    for (size_t i = 0; i < num_hashes; i += batch_size) {
        size_t current_batch = std::min(batch_size, num_hashes - i);
        
        // Resize inputs if needed for the last batch
        if (current_batch != inputs.size()) {
            inputs.resize(current_batch);
            for (size_t j = 0; j < current_batch; ++j) {
                inputs[j] = FieldElement::random();
            }
        }
        
        // Perform batch hash
        if (hasher.batch_hash_single(inputs, outputs)) {
            completed_hashes += current_batch;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    stats.total_time_ms = duration.count() / 1000000.0;
    stats.total_hashes = completed_hashes;
    if (completed_hashes > 0) {
        stats.avg_time_per_hash_ns = static_cast<double>(duration.count()) / completed_hashes;
        stats.hashes_per_second = static_cast<size_t>(1000000000.0 / stats.avg_time_per_hash_ns);
    }
    
    return stats;
}

CudaPoseidonStats benchmark_cuda_poseidon_pairs(IPoseidonCudaHash& hasher, size_t num_pairs, size_t batch_size) {
    CudaPoseidonStats stats = {};
    
    if (!hasher.is_initialized()) {
        std::cerr << "CUDA Poseidon hasher not initialized for benchmarking" << std::endl;
        return stats;
    }
    
    // Generate test data
    std::vector<FieldElement> left_inputs, right_inputs;
    left_inputs.reserve(batch_size);
    right_inputs.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        left_inputs.push_back(FieldElement::random());
        right_inputs.push_back(FieldElement::random());
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run hash operations
    size_t completed_hashes = 0;
    std::vector<FieldElement> outputs;
    
    for (size_t i = 0; i < num_pairs; i += batch_size) {
        size_t current_batch = std::min(batch_size, num_pairs - i);
        
        // Resize inputs if needed for the last batch
        if (current_batch != left_inputs.size()) {
            left_inputs.resize(current_batch);
            right_inputs.resize(current_batch);
            for (size_t j = 0; j < current_batch; ++j) {
                left_inputs[j] = FieldElement::random();
                right_inputs[j] = FieldElement::random();
            }
        }
        
        // Perform batch hash
        if (hasher.batch_hash_pairs(left_inputs, right_inputs, outputs)) {
            completed_hashes += current_batch;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    stats.total_time_ms = duration.count() / 1000000.0;
    stats.total_hashes = completed_hashes;
    if (completed_hashes > 0) {
        stats.avg_time_per_hash_ns = static_cast<double>(duration.count()) / completed_hashes;
        stats.hashes_per_second = static_cast<size_t>(1000000000.0 / stats.avg_time_per_hash_ns);
    }
    
    return stats;
}

CudaPoseidonStats benchmark_cuda_vs_cpu_poseidon(IPoseidonCudaHash& hasher, size_t num_hashes, size_t batch_size) {
    CudaPoseidonStats stats = {};
    
    // First run CPU benchmark for comparison
    auto cpu_stats = Poseidon::benchmark_poseidon(num_hashes);
    
    // Then run CUDA benchmark
    auto cuda_stats = benchmark_cuda_poseidon_single(hasher, num_hashes, batch_size);
    
    // Copy CUDA stats and add speedup calculation
    stats = cuda_stats;
    if (cpu_stats.avg_time_per_hash_ns > 0 && cuda_stats.avg_time_per_hash_ns > 0) {
        stats.speedup_vs_cpu = cpu_stats.avg_time_per_hash_ns / cuda_stats.avg_time_per_hash_ns;
    }
    
    return stats;
}

bool verify_cuda_implementations_match(IPoseidonCudaHash& hasher1, IPoseidonCudaHash& hasher2, 
                                      const std::string& name1, const std::string& name2,
                                      size_t num_tests) {
    std::cout << "\nVerifying " << name1 << " and " << name2 << " implementations match...\n";
    
    // Check if both hashers are initialized
    if (!hasher1.is_initialized() || !hasher2.is_initialized()) {
        std::cerr << "One or both hashers not initialized for verification" << std::endl;
        return false;
    }
    
    bool all_match = true;
    size_t mismatches = 0;
    
    // Test single hash operations
    {
        std::cout << "Testing single hash operations..." << std::endl;
        
        // Generate deterministic test data for reproducibility
        std::vector<FieldElement> test_inputs;
        test_inputs.reserve(num_tests);
        
        for (size_t i = 0; i < num_tests; ++i) {
            // Use deterministic values for reproducible testing
            test_inputs.push_back(FieldElement(i + 1, i * 2 + 1, i * 3 + 1, i * 4 + 1));
        }
        
        std::vector<FieldElement> outputs1, outputs2;
        
        // Hash with both implementations
        if (!hasher1.batch_hash_single(test_inputs, outputs1)) {
            std::cerr << "Failed to hash with " << name1 << std::endl;
            return false;
        }
        
        if (!hasher2.batch_hash_single(test_inputs, outputs2)) {
            std::cerr << "Failed to hash with " << name2 << std::endl;
            return false;
        }
        
        // Compare outputs
        if (outputs1.size() != outputs2.size()) {
            std::cerr << "Output sizes differ: " << outputs1.size() << " vs " << outputs2.size() << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < outputs1.size(); ++i) {
            if (outputs1[i] != outputs2[i]) {
                std::cerr << "Single hash mismatch at index " << i << ":" << std::endl;
                std::cerr << "  " << name1 << ": " << outputs1[i].to_hex() << std::endl;
                std::cerr << "  " << name2 << ": " << outputs2[i].to_hex() << std::endl;
                all_match = false;
                mismatches++;
            }
        }
        
        if (mismatches == 0) {
            std::cout << "✓ All " << num_tests << " single hashes match" << std::endl;
        } else {
            std::cout << "✗ " << mismatches << " single hash mismatches found" << std::endl;
        }
    }
    
    // Test pair hash operations
    {
        std::cout << "Testing pair hash operations..." << std::endl;
        mismatches = 0;
        
        // Generate deterministic test data for reproducibility
        std::vector<FieldElement> left_inputs, right_inputs;
        left_inputs.reserve(num_tests);
        right_inputs.reserve(num_tests);
        
        for (size_t i = 0; i < num_tests; ++i) {
            // Use deterministic values for reproducible testing
            left_inputs.push_back(FieldElement(i + 1, i * 2 + 1, i * 3 + 1, i * 4 + 1));
            right_inputs.push_back(FieldElement(i * 5 + 1, i * 6 + 1, i * 7 + 1, i * 8 + 1));
        }
        
        std::vector<FieldElement> outputs1, outputs2;
        
        // Hash with both implementations
        if (!hasher1.batch_hash_pairs(left_inputs, right_inputs, outputs1)) {
            std::cerr << "Failed to hash pairs with " << name1 << std::endl;
            return false;
        }
        
        if (!hasher2.batch_hash_pairs(left_inputs, right_inputs, outputs2)) {
            std::cerr << "Failed to hash pairs with " << name2 << std::endl;
            return false;
        }
        
        // Compare outputs
        if (outputs1.size() != outputs2.size()) {
            std::cerr << "Pair output sizes differ: " << outputs1.size() << " vs " << outputs2.size() << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < outputs1.size(); ++i) {
            if (outputs1[i] != outputs2[i]) {
                std::cerr << "Pair hash mismatch at index " << i << ":" << std::endl;
                std::cerr << "  " << name1 << ": " << outputs1[i].to_hex() << std::endl;
                std::cerr << "  " << name2 << ": " << outputs2[i].to_hex() << std::endl;
                all_match = false;
                mismatches++;
            }
        }
        
        if (mismatches == 0) {
            std::cout << "✓ All " << num_tests << " pair hashes match" << std::endl;
        } else {
            std::cout << "✗ " << mismatches << " pair hash mismatches found" << std::endl;
        }
    }
    
    if (all_match) {
        std::cout << "✓ All verification tests passed! " << name1 << " and " << name2 << " produce identical results." << std::endl;
    } else {
        std::cout << "✗ Verification failed! " << name1 << " and " << name2 << " produce different results." << std::endl;
    }
    
    return all_match;
}

} // namespace PoseidonCUDA
} // namespace Poseidon 