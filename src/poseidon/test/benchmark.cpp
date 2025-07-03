#include "poseidon.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#ifdef CUDA_ENABLED
#include "poseidon_cuda.cuh"
#endif

namespace PoseidonBenchmark {

struct BenchmarkConfig {
    size_t num_hashes;
    size_t batch_size;
    std::string description;
};

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void print_separator() {
    std::cout << std::string(80, '-') << "\n";
}

void print_cpu_stats(const Poseidon::HashingStats& stats, const std::string& operation) {
    std::cout << std::left << std::setw(20) << operation << " | "
              << std::right << std::setw(12) << stats.total_hashes << " | "
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << stats.total_time_ms << " | "
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << stats.avg_time_per_hash_ns << " | "
              << std::right << std::setw(15) << stats.hashes_per_second << " |\n";
}

#ifdef CUDA_ENABLED
void print_cuda_stats(const Poseidon::PoseidonCUDA::CudaPoseidonStats& stats, const std::string& operation) {
    std::cout << std::left << std::setw(20) << operation << " | "
              << std::right << std::setw(12) << stats.total_hashes << " | "
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << stats.total_time_ms << " | "
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << stats.avg_time_per_hash_ns << " | "
              << std::right << std::setw(15) << stats.hashes_per_second << " |\n";
}

void print_comparison_stats(const Poseidon::HashingStats& cpu_stats, 
                           const Poseidon::PoseidonCUDA::CudaPoseidonStats& cuda_stats,
                           const std::string& operation) {
    double speedup = cpu_stats.avg_time_per_hash_ns / cuda_stats.avg_time_per_hash_ns;
    
    std::cout << std::left << std::setw(20) << operation << " | "
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << cpu_stats.avg_time_per_hash_ns << " | "
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << cuda_stats.avg_time_per_hash_ns << " | "
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x | "
              << std::right << std::setw(15) << cpu_stats.hashes_per_second << " | "
              << std::right << std::setw(15) << cuda_stats.hashes_per_second << " |\n";
}
#endif

void run_cpu_benchmarks(const std::vector<BenchmarkConfig>& configs) {
    print_header("CPU POSEIDON HASH BENCHMARKS");
    
    std::cout << std::left << std::setw(20) << "Operation" << " | "
              << std::right << std::setw(12) << "Total Hashes" << " | "
              << std::right << std::setw(12) << "Time (ms)" << " | "
              << std::right << std::setw(15) << "Avg/Hash (ns)" << " | "
              << std::right << std::setw(15) << "Hashes/sec" << " |\n";
    print_separator();
    
    for (const auto& config : configs) {
        // Single hash benchmark
        auto single_stats = Poseidon::benchmark_poseidon(config.num_hashes);
        print_cpu_stats(single_stats, "Single Hash");
        
        // Pair hash benchmark
        auto pair_stats = Poseidon::benchmark_poseidon_pairs(config.num_hashes);
        print_cpu_stats(pair_stats, "Pair Hash");
        
        print_separator();
    }
}

#ifdef CUDA_ENABLED
void run_cuda_benchmarks(const std::vector<BenchmarkConfig>& configs) {
    print_header("CUDA POSEIDON HASH BENCHMARKS");
    
    if (!Poseidon::PoseidonCUDA::CudaPoseidonHash::initialize()) {
        std::cout << "Failed to initialize CUDA Poseidon. Skipping CUDA benchmarks.\n";
        return;
    }
    
    std::cout << std::left << std::setw(20) << "Operation" << " | "
              << std::right << std::setw(12) << "Total Hashes" << " | "
              << std::right << std::setw(12) << "Time (ms)" << " | "
              << std::right << std::setw(15) << "Avg/Hash (ns)" << " | "
              << std::right << std::setw(15) << "Hashes/sec" << " |\n";
    print_separator();
    
    for (const auto& config : configs) {
        // Single hash benchmark
        auto single_stats = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_single(config.num_hashes, config.batch_size);
        print_cuda_stats(single_stats, "Single Hash");
        
        // Pair hash benchmark  
        auto pair_stats = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_pairs(config.num_hashes, config.batch_size);
        print_cuda_stats(pair_stats, "Pair Hash");
        
        print_separator();
    }
    
    Poseidon::PoseidonCUDA::CudaPoseidonHash::cleanup();
}

void run_comparison_benchmarks(const std::vector<BenchmarkConfig>& configs) {
    print_header("CPU vs CUDA PERFORMANCE COMPARISON");
    
    if (!Poseidon::PoseidonCUDA::CudaPoseidonHash::initialize()) {
        std::cout << "Failed to initialize CUDA Poseidon. Skipping comparison benchmarks.\n";
        return;
    }
    
    std::cout << std::left << std::setw(20) << "Operation" << " | "
              << std::right << std::setw(12) << "CPU (ns)" << " | "
              << std::right << std::setw(12) << "CUDA (ns)" << " | "
              << std::right << std::setw(12) << "Speedup" << " | "
              << std::right << std::setw(15) << "CPU Hash/s" << " | "
              << std::right << std::setw(15) << "CUDA Hash/s" << " |\n";
    print_separator();
    
    for (const auto& config : configs) {
        std::cout << "Config: " << config.description << " (Batch: " << config.batch_size << ")\n";
        
        // Single hash comparison
        auto cpu_single_stats = Poseidon::benchmark_poseidon(config.num_hashes);
        auto cuda_single_stats = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_single(config.num_hashes, config.batch_size);
        print_comparison_stats(cpu_single_stats, cuda_single_stats, "Single Hash");
        
        // Pair hash comparison
        auto cpu_pair_stats = Poseidon::benchmark_poseidon_pairs(config.num_hashes);
        auto cuda_pair_stats = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_pairs(config.num_hashes, config.batch_size);
        print_comparison_stats(cpu_pair_stats, cuda_pair_stats, "Pair Hash");
        
        print_separator();
    }
    
    Poseidon::PoseidonCUDA::CudaPoseidonHash::cleanup();
}

void run_scalability_analysis() {
    print_header("SCALABILITY ANALYSIS");
    
    std::vector<size_t> scales = {1000, 10000, 100000, 1000000};
    std::vector<size_t> batch_sizes = {256, 1024, 4096};
    
    std::cout << std::left << std::setw(15) << "Scale" << " | "
              << std::right << std::setw(12) << "Batch Size" << " | "
              << std::right << std::setw(12) << "CPU (ns)" << " | "
              << std::right << std::setw(12) << "CUDA (ns)" << " | "
              << std::right << std::setw(12) << "Speedup" << " |\n";
    print_separator();
    
#ifdef CUDA_ENABLED
    if (!Poseidon::PoseidonCUDA::CudaPoseidonHash::initialize()) {
        std::cout << "Failed to initialize CUDA Poseidon. Skipping scalability analysis.\n";
        return;
    }
    
    for (size_t scale : scales) {
        for (size_t batch_size : batch_sizes) {
            auto cpu_stats = Poseidon::benchmark_poseidon(scale);
            auto cuda_stats = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_single(scale, batch_size);
            
            double speedup = cpu_stats.avg_time_per_hash_ns / cuda_stats.avg_time_per_hash_ns;
            
            std::cout << std::left << std::setw(15) << scale << " | "
                      << std::right << std::setw(12) << batch_size << " | "
                      << std::right << std::setw(12) << std::fixed << std::setprecision(2) << cpu_stats.avg_time_per_hash_ns << " | "
                      << std::right << std::setw(12) << std::fixed << std::setprecision(2) << cuda_stats.avg_time_per_hash_ns << " | "
                      << std::right << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x |\n";
        }
        print_separator();
    }
    
    Poseidon::PoseidonCUDA::CudaPoseidonHash::cleanup();
#else
    std::cout << "CUDA not available. Skipping scalability analysis.\n";
#endif
}
#endif

void print_performance_summary() {
    print_header("PERFORMANCE SUMMARY");
    
    std::cout << "Key Performance Insights:\n\n";
    
    // Run a quick comparison for summary
    const size_t test_size = 100000;
    const size_t batch_size = 1024;
    
    auto cpu_single = Poseidon::benchmark_poseidon(test_size);
    auto cpu_pair = Poseidon::benchmark_poseidon_pairs(test_size);
    
#ifdef CUDA_ENABLED
    if (Poseidon::PoseidonCUDA::CudaPoseidonHash::initialize()) {
        auto cuda_single = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_single(test_size, batch_size);
        auto cuda_pair = Poseidon::PoseidonCUDA::benchmark_cuda_poseidon_pairs(test_size, batch_size);
        
        double single_speedup = cpu_single.avg_time_per_hash_ns / cuda_single.avg_time_per_hash_ns;
        double pair_speedup = cpu_pair.avg_time_per_hash_ns / cuda_pair.avg_time_per_hash_ns;
        
        std::cout << "1. Single Hash Performance:\n";
        std::cout << "   - CPU: " << std::fixed << std::setprecision(2) << cpu_single.avg_time_per_hash_ns << " ns/hash\n";
        std::cout << "   - CUDA: " << std::fixed << std::setprecision(2) << cuda_single.avg_time_per_hash_ns << " ns/hash\n";
        std::cout << "   - Speedup: " << std::fixed << std::setprecision(2) << single_speedup << "x\n\n";
        
        std::cout << "2. Pair Hash Performance:\n";
        std::cout << "   - CPU: " << std::fixed << std::setprecision(2) << cpu_pair.avg_time_per_hash_ns << " ns/hash\n";
        std::cout << "   - CUDA: " << std::fixed << std::setprecision(2) << cuda_pair.avg_time_per_hash_ns << " ns/hash\n";
        std::cout << "   - Speedup: " << std::fixed << std::setprecision(2) << pair_speedup << "x\n\n";
        
        std::cout << "3. Throughput Comparison (hashes/second):\n";
        std::cout << "   - CPU Single: " << cpu_single.hashes_per_second << " h/s\n";
        std::cout << "   - CUDA Single: " << cuda_single.hashes_per_second << " h/s\n";
        std::cout << "   - CPU Pair: " << cpu_pair.hashes_per_second << " h/s\n";
        std::cout << "   - CUDA Pair: " << cuda_pair.hashes_per_second << " h/s\n\n";
        
        std::cout << "4. Recommendations:\n";
        if (single_speedup > 2.0) {
            std::cout << "   - CUDA shows significant performance advantages for single hashes\n";
        } else if (single_speedup > 1.2) {
            std::cout << "   - CUDA shows moderate performance advantages for single hashes\n";
        } else {
            std::cout << "   - CPU and CUDA performance are comparable for single hashes\n";
        }
        
        if (pair_speedup > 2.0) {
            std::cout << "   - CUDA shows significant performance advantages for pair hashes\n";
        } else if (pair_speedup > 1.2) {
            std::cout << "   - CUDA shows moderate performance advantages for pair hashes\n";
        } else {
            std::cout << "   - CPU and CUDA performance are comparable for pair hashes\n";
        }
        
        std::cout << "   - Optimal batch size for CUDA: " << Poseidon::PoseidonCUDA::CudaPoseidonHash::get_optimal_batch_size() << "\n";
        std::cout << "   - Maximum batch size for CUDA: " << Poseidon::PoseidonCUDA::CudaPoseidonHash::get_max_batch_size() << "\n";
        
        Poseidon::PoseidonCUDA::CudaPoseidonHash::cleanup();
    } else {
        std::cout << "CUDA not available for comparison.\n";
    }
#else
    std::cout << "1. CPU-only Performance:\n";
    std::cout << "   - Single Hash: " << std::fixed << std::setprecision(2) << cpu_single.avg_time_per_hash_ns << " ns/hash\n";
    std::cout << "   - Pair Hash: " << std::fixed << std::setprecision(2) << cpu_pair.avg_time_per_hash_ns << " ns/hash\n";
    std::cout << "   - Single Hash Throughput: " << cpu_single.hashes_per_second << " h/s\n";
    std::cout << "   - Pair Hash Throughput: " << cpu_pair.hashes_per_second << " h/s\n\n";
    std::cout << "Note: CUDA benchmarks not available (CUDA not enabled)\n";
#endif
}

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
    
    // Run CPU benchmarks
    PoseidonBenchmark::run_cpu_benchmarks(configs);
    
#ifdef CUDA_ENABLED
    // Run CUDA benchmarks
    PoseidonBenchmark::run_cuda_benchmarks(configs);
    
    // Run comparison benchmarks
    PoseidonBenchmark::run_comparison_benchmarks(configs);
    
    // Run scalability analysis
    PoseidonBenchmark::run_scalability_analysis();
#endif
    
    // Print final summary
    PoseidonBenchmark::print_performance_summary();
    
    std::cout << "\nBenchmark completed successfully!\n";
    return 0;
} 