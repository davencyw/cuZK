#include "poseidon.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <memory>
#include <map>

#ifdef CUDA_ENABLED
#include "../cuda/poseidon_cuda.cuh"
#include "../cuda/poseidon_cuda_optimized.cuh"
#include "../cuda/poseidon_cuda_benchmarks.hpp"
#endif

namespace PoseidonBenchmark {

struct BenchmarkConfig {
    size_t num_hashes;
    size_t batch_size;
    std::string description;
};

struct BenchmarkResult {
    std::string implementation;
    std::string operation;
    double avg_time_per_hash_ns;
    size_t hashes_per_second;
    double total_time_ms;
    size_t total_hashes;
    bool success;
};

#ifdef CUDA_ENABLED
// Structure to hold CUDA implementations for comparison
struct CudaImplementation {
    std::string name;
    std::unique_ptr<Poseidon::PoseidonCUDA::IPoseidonCudaHash> hasher;
    
    CudaImplementation(const std::string& n, std::unique_ptr<Poseidon::PoseidonCUDA::IPoseidonCudaHash> h) 
        : name(n), hasher(std::move(h)) {}
};
#endif

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(120, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(120, '=') << "\n\n";
}

void print_separator() {
    std::cout << std::string(120, '-') << "\n";
}

BenchmarkResult create_cpu_result(const std::string& operation, const Poseidon::HashingStats& stats) {
    return {
        "CPU",
        operation,
        stats.avg_time_per_hash_ns,
        stats.hashes_per_second,
        stats.total_time_ms,
        stats.total_hashes,
        true
    };
}

#ifdef CUDA_ENABLED
BenchmarkResult create_cuda_result(const std::string& implementation, const std::string& operation, 
                                 const Poseidon::PoseidonCUDA::CudaPoseidonStats& stats) {
    return {
        implementation,
        operation,
        stats.avg_time_per_hash_ns,
        stats.hashes_per_second,
        stats.total_time_ms,
        stats.total_hashes,
        true
    };
}
#endif

void print_results_table(const std::vector<BenchmarkResult>& results) {
    // Group results by operation
    std::map<std::string, std::vector<BenchmarkResult>> by_operation;
    for (const auto& result : results) {
        by_operation[result.operation].push_back(result);
    }
    
    for (const auto& [operation, op_results] : by_operation) {
        std::cout << "\n" << operation << " Hash Results:\n";
        std::cout << std::string(120, '-') << "\n";
        
        // Print header
        std::cout << std::left << std::setw(20) << "Implementation" << " | "
                  << std::right << std::setw(15) << "Time (ns)" << " | "
                  << std::right << std::setw(15) << "Hash/s" << " | "
                  << std::right << std::setw(15) << "Total Time (ms)" << " | "
                  << std::right << std::setw(15) << "Speedup vs CPU" << " |\n";
        
        print_separator();
        
        // Find CPU baseline for speedup calculation
        double cpu_time_ns = 0.0;
        for (const auto& result : op_results) {
            if (result.implementation == "CPU") {
                cpu_time_ns = result.avg_time_per_hash_ns;
                break;
            }
        }
        
        // Print results
        for (const auto& result : op_results) {
            double speedup = (cpu_time_ns > 0) ? cpu_time_ns / result.avg_time_per_hash_ns : 1.0;
            
            std::cout << std::left << std::setw(20) << result.implementation << " | "
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << result.avg_time_per_hash_ns << " | "
                      << std::right << std::setw(15) << result.hashes_per_second << " | "
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << result.total_time_ms << " | "
                      << std::right << std::setw(15) << std::fixed << std::setprecision(1) << speedup << "x |\n";
        }
        
        print_separator();
    }
}

#ifdef CUDA_ENABLED
void run_unified_benchmarks(const std::vector<BenchmarkConfig>& configs) {
    print_header("UNIFIED PERFORMANCE COMPARISON: ALL IMPLEMENTATIONS");
    
    // Create all CUDA implementations
    std::vector<CudaImplementation> cuda_implementations;
    
    auto original_hasher = std::make_unique<Poseidon::PoseidonCUDA::CudaPoseidonHash>();
    auto optimized_hasher = std::make_unique<Poseidon::PoseidonCUDAOptimized::CudaPoseidonHashOptimized>();
    
    // Run verification test before benchmarking
    print_header("IMPLEMENTATION VERIFICATION");
    bool verification_passed = Poseidon::PoseidonCUDA::verify_cuda_implementations_match(
        *original_hasher, *optimized_hasher, "CUDA Original", "CUDA Optimized", 100);
    
    if (!verification_passed) {
        std::cerr << "\n❌ CRITICAL ERROR: Implementations do not produce identical results!\n";
        std::cerr << "Benchmarking aborted to prevent misleading performance comparisons.\n";
        return;
    }
    
    std::cout << "\n✅ Verification complete: All implementations produce identical results\n";
    
    cuda_implementations.emplace_back("CUDA Original", std::move(original_hasher));
    cuda_implementations.emplace_back("CUDA Optimized", std::move(optimized_hasher));
    
    // Run benchmarks for each configuration
    for (const auto& config : configs) {
        std::cout << "\n" << std::string(120, '=') << "\n";
        std::cout << "Configuration: " << config.description 
                  << " (Batch Size: " << config.batch_size 
                  << ", Total Hashes: " << config.num_hashes << ")\n";
        std::cout << std::string(120, '=') << "\n";
        
        std::vector<BenchmarkResult> results;
        
        // Run CPU benchmarks once
        auto cpu_single_stats = Poseidon::benchmark_poseidon(config.num_hashes);
        auto cpu_pair_stats = Poseidon::benchmark_poseidon_pairs(config.num_hashes);
        
        results.push_back(create_cpu_result("Single", cpu_single_stats));
        results.push_back(create_cpu_result("Pair", cpu_pair_stats));
        
        // Run each CUDA implementation once
        for (const auto& cuda_impl : cuda_implementations) {
            // Single hash benchmark
            auto cuda_single_stats = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_single(
                *cuda_impl.hasher, config.num_hashes, config.batch_size);
            results.push_back(create_cuda_result(cuda_impl.name, "Single", cuda_single_stats));
            
            // Pair hash benchmark
            auto cuda_pair_stats = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_pairs(
                *cuda_impl.hasher, config.num_hashes, config.batch_size);
            results.push_back(create_cuda_result(cuda_impl.name, "Pair", cuda_pair_stats));
        }

        // Print results table
        print_results_table(results);
        

        
        // Calculate best performers
        std::map<std::string, std::string> best_performers;
        std::map<std::string, double> best_times;
        
        for (const auto& result : results) {
            if (best_times.find(result.operation) == best_times.end() || 
                result.avg_time_per_hash_ns < best_times[result.operation]) {
                best_times[result.operation] = result.avg_time_per_hash_ns;
                best_performers[result.operation] = result.implementation;
            }
        }
        
        std::cout << "\nBest Performers:\n";
        for (const auto& [operation, implementation] : best_performers) {
            std::cout << "- " << operation << " Hash: " << implementation 
                      << " (" << std::fixed << std::setprecision(2) << best_times[operation] << " ns/hash)\n";
        }
    }
    
    std::cout << "\n" << std::string(120, '=') << "\n";
    std::cout << "All benchmarks completed successfully!\n";
    std::cout << std::string(120, '=') << "\n";
}
#endif

} // namespace PoseidonBenchmark

int main() {
    std::cout << "Poseidon Hash Performance Benchmark Suite\n";
    std::cout << "==========================================\n";
    
    // Initialize Poseidon constants
    Poseidon::PoseidonConstants::init();
    
    // Define benchmark configurations
    std::vector<PoseidonBenchmark::BenchmarkConfig> configs = {
        {10000, 512, "Small Scale"},
        {100000, 1024, "Medium Scale"},
        {1000000, 4096, "Large Scale"}
    };
    
#ifdef CUDA_ENABLED
    PoseidonBenchmark::run_unified_benchmarks(configs);
#else
    std::cout << "Benchmarks not available (CUDA not enabled)\n";
    return 1;
#endif
    
    return 0;
} 