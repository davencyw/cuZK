#include "poseidon_cuda_optimized.cuh"
#include "poseidon_interface_cuda.hpp"
#include "../field_arithmetic.hpp"
#include "../../common/error_handling.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace Poseidon;
using namespace Poseidon::PoseidonCUDAOptimized;

struct ProfileConfig {
    size_t batch_size;
    size_t num_iterations;
    std::string description;
};

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void profile_single_hash_kernel(const ProfileConfig& config) {
    std::cout << "Profiling batch_hash_single_kernel_optimized:\n";
    std::cout << "  Batch size: " << config.batch_size << "\n";
    std::cout << "  Iterations: " << config.num_iterations << "\n";
    std::cout << "  " << config.description << "\n\n";
    
    // Initialize CUDA Poseidon hasher
    CudaPoseidonHashOptimized hasher;
    if (!hasher.is_initialized()) {
        std::cerr << "Failed to initialize CUDA Poseidon hasher" << std::endl;
        return;
    }
    
    // Generate test data
    std::vector<FieldElement> inputs;
    inputs.reserve(config.batch_size);
    for (size_t i = 0; i < config.batch_size; ++i) {
        inputs.push_back(FieldElement::random());
    }
    
    std::vector<FieldElement> outputs;
    
    // Warm up - single iteration to prepare GPU
    hasher.batch_hash_single(inputs, outputs);
    
    std::cout << "Starting profiling iterations...\n";
    
    // Main profiling loop
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < config.num_iterations; ++i) {
        if (!hasher.batch_hash_single(inputs, outputs)) {
            std::cerr << "Hash operation failed at iteration " << i << std::endl;
            return;
        }
        
        // Progress indicator (only for larger iteration counts)
        if (config.num_iterations >= 10 && (i + 1) % (config.num_iterations / 10) == 0) {
            std::cout << "  Completed " << (i + 1) << "/" << config.num_iterations 
                      << " iterations (" << std::fixed << std::setprecision(1) 
                      << (100.0 * (i + 1) / config.num_iterations) << "%)\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate statistics
    size_t total_hashes = config.batch_size * config.num_iterations;
    double total_time_ms = duration.count() / 1000.0;
    double avg_time_per_hash_ns = (duration.count() * 1000.0) / total_hashes;
    size_t hashes_per_second = static_cast<size_t>(total_hashes * 1000000.0 / duration.count());
    
    std::cout << "\nResults:\n";
    std::cout << "  Total hashes: " << total_hashes << "\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n";
    std::cout << "  Average time per hash: " << std::fixed << std::setprecision(2) 
              << avg_time_per_hash_ns << " ns\n";
    std::cout << "  Hashes per second: " << hashes_per_second << "\n";
}

void profile_pairs_hash_kernel(const ProfileConfig& config) {
    std::cout << "Profiling batch_hash_pairs_kernel_optimized:\n";
    std::cout << "  Batch size: " << config.batch_size << "\n";
    std::cout << "  Iterations: " << config.num_iterations << "\n";
    std::cout << "  " << config.description << "\n\n";
    
    // Initialize CUDA Poseidon hasher
    CudaPoseidonHashOptimized hasher;
    if (!hasher.is_initialized()) {
        std::cerr << "Failed to initialize CUDA Poseidon hasher" << std::endl;
        return;
    }
    
    // Generate test data
    std::vector<FieldElement> left_inputs, right_inputs;
    left_inputs.reserve(config.batch_size);
    right_inputs.reserve(config.batch_size);
    
    for (size_t i = 0; i < config.batch_size; ++i) {
        left_inputs.push_back(FieldElement::random());
        right_inputs.push_back(FieldElement::random());
    }
    
    std::vector<FieldElement> outputs;
    
    // Warm up
    hasher.batch_hash_pairs(left_inputs, right_inputs, outputs);
    
    std::cout << "Starting profiling iterations...\n";
    
    // Main profiling loop
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < config.num_iterations; ++i) {
        if (!hasher.batch_hash_pairs(left_inputs, right_inputs, outputs)) {
            std::cerr << "Hash operation failed at iteration " << i << std::endl;
            return;
        }
        
        // Progress indicator (only for larger iteration counts)
        if (config.num_iterations >= 10 && (i + 1) % (config.num_iterations / 10) == 0) {
            std::cout << "  Completed " << (i + 1) << "/" << config.num_iterations 
                      << " iterations (" << std::fixed << std::setprecision(1) 
                      << (100.0 * (i + 1) / config.num_iterations) << "%)\n";
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate statistics
    size_t total_hashes = config.batch_size * config.num_iterations;
    double total_time_ms = duration.count() / 1000.0;
    double avg_time_per_hash_ns = (duration.count() * 1000.0) / total_hashes;
    size_t hashes_per_second = static_cast<size_t>(total_hashes * 1000000.0 / duration.count());
    
    std::cout << "\nResults:\n";
    std::cout << "  Total hashes: " << total_hashes << "\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n";
    std::cout << "  Average time per hash: " << std::fixed << std::setprecision(2) 
              << avg_time_per_hash_ns << " ns\n";
    std::cout << "  Hashes per second: " << hashes_per_second << "\n";
}

void run_comprehensive_profile() {
    // Different configurations to test various scenarios
    std::vector<ProfileConfig> configs = {
        {1024,   100, "Small batch, many iterations"},
        {8192,   50,  "Medium batch, moderate iterations"},
        {32768,  20,  "Large batch, fewer iterations"},
        {65536,  10,  "Very large batch, minimal iterations"}
    };
    
    print_header("Poseidon CUDA Optimized Kernel Profiling");
    
    for (const auto& config : configs) {
        print_header("Single Hash Kernel - " + config.description);
        profile_single_hash_kernel(config);
        
        print_header("Pairs Hash Kernel - " + config.description);
        profile_pairs_hash_kernel(config);
        
        std::cout << "\n" << std::string(80, '-') << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Poseidon CUDA Optimized Kernel Profiler\n";
    std::cout << "=========================================\n\n";
    
    // Parse command line arguments for custom configuration
    if (argc == 4) {
        size_t batch_size = std::stoull(argv[1]);
        size_t iterations = std::stoull(argv[2]);
        std::string kernel_type = argv[3];
        
        ProfileConfig config = {batch_size, iterations, "Custom configuration"};
        
        if (kernel_type == "single") {
            print_header("Custom Single Hash Kernel Profile");
            profile_single_hash_kernel(config);
        } else if (kernel_type == "pairs") {
            print_header("Custom Pairs Hash Kernel Profile");
            profile_pairs_hash_kernel(config);
        } else if (kernel_type == "both") {
            print_header("Custom Single Hash Kernel Profile");
            profile_single_hash_kernel(config);
            print_header("Custom Pairs Hash Kernel Profile");
            profile_pairs_hash_kernel(config);
        } else {
            std::cerr << "Invalid kernel type. Use 'single', 'pairs', or 'both'.\n";
            return 1;
        }
    } else if (argc == 1) {
        // Run comprehensive profiling
        run_comprehensive_profile();
    } else {
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << "                          # Run comprehensive profiling\n";
        std::cout << "  " << argv[0] << " <batch_size> <iterations> <kernel_type>\n";
        std::cout << "    kernel_type: 'single', 'pairs', or 'both'\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " 8192 100 single         # Profile single hash kernel\n";
        return 1;
    }
    
    return 0;
} 