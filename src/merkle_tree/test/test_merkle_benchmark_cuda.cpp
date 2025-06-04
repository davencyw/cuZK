#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../merkle_tree.hpp"
#include "../merkle_tree_cuda.cuh"

using namespace MerkleTree;
using namespace MerkleTree::MerkleTreeCUDA;

class MerkleTreeCUDABenchmarkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize Poseidon constants
    Poseidon::PoseidonConstants::init();
    Poseidon::FieldConstants::init();

    // Initialize CUDA
    if (!CudaNaryMerkleTree::initialize_cuda()) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA benchmark tests";
    }
  }

  void TearDown() override {
    CudaNaryMerkleTree::cleanup_cuda();
  }

  void print_benchmark_header(const std::string& title) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(100, '=') << "\n";
  }

  void print_comparison_results(const MerkleUtils::TreeBenchmarkResult& cpu_result,
                                const CudaMerkleTreeStats& cuda_result) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Dataset: " << std::setw(6) << cpu_result.leaf_count
              << " leaves | Arity: " << std::setw(2) << cpu_result.arity
              << " | Height: " << std::setw(2) << cpu_result.tree_height << "\n";
    std::cout << std::string(100, '-') << "\n";

    // Tree building comparison
    std::cout << "TREE BUILDING:\n";
    std::cout << "  CPU Time:    " << std::setw(10) << cpu_result.build_time_ms << " ms\n";
    std::cout << "  CUDA Time:   " << std::setw(10) << cuda_result.build_time_ms << " ms\n";
    if (cuda_result.build_time_ms > 0) {
      double speedup = cpu_result.build_time_ms / cuda_result.build_time_ms;
      std::cout << "  Speedup:     " << std::setw(10) << speedup << "x ";
      if (speedup > 1.0) {
        std::cout << "(CUDA FASTER)";
      } else {
        std::cout << "(CPU FASTER)";
      }
      std::cout << "\n";
    }

    // Proof generation comparison
    std::cout << "\nPROOF GENERATION:\n";
    std::cout << "  CPU Time:    " << std::setw(10) << cpu_result.proof_generation_time_ms
              << " ms\n";
    std::cout << "  CUDA Time:   " << std::setw(10) << cuda_result.proof_generation_time_ms
              << " ms\n";
    if (cuda_result.proof_generation_time_ms > 0) {
      double speedup = cpu_result.proof_generation_time_ms / cuda_result.proof_generation_time_ms;
      std::cout << "  Speedup:     " << std::setw(10) << speedup << "x ";
      if (speedup > 1.0) {
        std::cout << "(CUDA FASTER)";
      } else {
        std::cout << "(CPU FASTER)";
      }
      std::cout << "\n";
    }

    // Proof verification comparison
    std::cout << "\nPROOF VERIFICATION:\n";
    std::cout << "  CPU Time:    " << std::setw(10) << cpu_result.proof_verification_time_ms
              << " ms\n";
    std::cout << "  CUDA Time:   " << std::setw(10) << cuda_result.proof_verification_time_ms
              << " ms\n";
    if (cuda_result.proof_verification_time_ms > 0) {
      double speedup =
          cpu_result.proof_verification_time_ms / cuda_result.proof_verification_time_ms;
      std::cout << "  Speedup:     " << std::setw(10) << speedup << "x ";
      if (speedup > 1.0) {
        std::cout << "(CUDA FASTER)";
      } else {
        std::cout << "(CPU FASTER)";
      }
      std::cout << "\n";
    }

    std::cout << "\n";
  }

  void print_cuda_stats(const CudaMerkleTreeStats& stats) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CUDA Performance Stats:\n";
    std::cout << "  Dataset Size:      " << stats.leaf_count << " leaves\n";
    std::cout << "  Tree Arity:        " << stats.arity << "\n";
    std::cout << "  Tree Height:       " << stats.tree_height << "\n";
    std::cout << "  Total Time:        " << stats.total_time_ms << " ms\n";
    std::cout << "  Build Time:        " << stats.build_time_ms << " ms\n";
    std::cout << "  Proof Gen Time:    " << stats.proof_generation_time_ms << " ms\n";
    std::cout << "  Proof Ver Time:    " << stats.proof_verification_time_ms << " ms\n";
    std::cout << "  Trees/Second:      " << stats.trees_per_second << "\n";
    std::cout << "  Proofs/Second:     " << stats.proofs_per_second << "\n";
    std::cout << "  Speedup vs CPU:    " << stats.speedup_vs_cpu << "x\n";
    std::cout << "\n";
  }
};

TEST_F(MerkleTreeCUDABenchmarkTest, ComprehensiveCPUvsGPUComparison) {
  print_benchmark_header("COMPREHENSIVE CPU vs GPU MERKLE TREE BENCHMARK");

  std::vector<size_t> leaf_counts = {64, 256, 1024, 4096};
  std::vector<size_t> arities = {2, 4, 8};
  const size_t num_proofs = 100;

  for (size_t arity : arities) {
    std::cout << "\n>>> TESTING ARITY " << arity << " <<<\n";

    for (size_t leaf_count : leaf_counts) {
      std::cout << "\n" << std::string(50, '-') << "\n";

      // CPU benchmark
      auto cpu_result = MerkleUtils::benchmark_tree(leaf_count, arity, num_proofs);

      // CUDA benchmark - first do individual benchmarks
      auto cuda_tree_stats = benchmark_cuda_tree_building(1, leaf_count, arity);
      auto cuda_proof_gen_stats = benchmark_cuda_proof_generation(num_proofs, leaf_count, arity);
      auto cuda_proof_ver_stats = benchmark_cuda_proof_verification(num_proofs, leaf_count, arity);

      // Combine CUDA stats for comparison
      CudaMerkleTreeStats combined_cuda_stats = {};
      combined_cuda_stats.leaf_count = leaf_count;
      combined_cuda_stats.arity = arity;
      combined_cuda_stats.tree_height = cuda_tree_stats.tree_height;
      combined_cuda_stats.build_time_ms = cuda_tree_stats.build_time_ms;
      combined_cuda_stats.proof_generation_time_ms = cuda_proof_gen_stats.proof_generation_time_ms;
      combined_cuda_stats.proof_verification_time_ms =
          cuda_proof_ver_stats.proof_verification_time_ms;

      print_comparison_results(cpu_result, combined_cuda_stats);
    }
  }
}

TEST_F(MerkleTreeCUDABenchmarkTest, ScalabilityAnalysis) {
  print_benchmark_header("CUDA SCALABILITY ANALYSIS");

  std::vector<size_t> dataset_sizes = {100, 500, 1000, 5000, 10000, 50000};
  const size_t arity = 4;
  const size_t num_trees = 10;

  std::cout << "Dataset Size | Build Time (ms) | Trees/sec | Speedup vs CPU\n";
  std::cout << std::string(65, '-') << "\n";

  for (size_t size : dataset_sizes) {
    auto cuda_stats = benchmark_cuda_vs_cpu_merkle(num_trees, size, arity);

    std::cout << std::setw(12) << size << " | " << std::setw(15) << std::fixed
              << std::setprecision(2) << cuda_stats.build_time_ms << " | " << std::setw(9)
              << cuda_stats.trees_per_second << " | " << std::setw(13) << cuda_stats.speedup_vs_cpu
              << "x\n";
  }

  std::cout << std::string(80, '=') << "\n";
}

TEST_F(MerkleTreeCUDABenchmarkTest, BatchProofPerformanceComparison) {
  print_benchmark_header("BATCH PROOF PERFORMANCE: CPU vs GPU");

  const size_t leaf_count = 1024;
  const size_t arity = 4;
  std::vector<size_t> batch_sizes = {10, 50, 100, 500, 1000, 5000};

  // Generate test data
  auto test_data = MerkleUtils::generate_test_leaves(leaf_count);
  MerkleTreeConfig config(arity);

  // Build both trees
  NaryMerkleTree cpu_tree(test_data, config);
  CudaNaryMerkleTree cuda_tree(test_data, config);

  std::cout << "Batch Size | CPU Proof Gen (ms) | GPU Proof Gen (ms) | CPU Verify (ms) | GPU "
               "Verify (ms) | Speedup\n";
  std::cout << std::string(100, '-') << "\n";

  for (size_t batch_size : batch_sizes) {
    // Generate random indices
    std::vector<size_t> indices;
    std::vector<FieldElement> leaf_values;
    for (size_t i = 0; i < batch_size; ++i) {
      size_t idx = i % leaf_count;
      indices.push_back(idx);
      leaf_values.push_back(test_data[idx]);
    }

    // CPU proof generation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_proofs = cpu_tree.generate_batch_proofs(indices);
    auto cpu_gen_end = std::chrono::high_resolution_clock::now();

    // CPU proof verification
    bool cpu_verify_result =
        cpu_tree.verify_batch_proofs(cpu_proofs, leaf_values, cpu_tree.get_root_hash());
    auto cpu_verify_end = std::chrono::high_resolution_clock::now();

    // CUDA proof generation
    auto cuda_start = std::chrono::high_resolution_clock::now();
    auto cuda_proofs = cuda_tree.generate_batch_proofs(indices);
    auto cuda_gen_end = std::chrono::high_resolution_clock::now();

    // CUDA proof verification
    bool cuda_verify_result = cuda_tree.verify_batch_proofs(cuda_proofs, leaf_values);
    auto cuda_verify_end = std::chrono::high_resolution_clock::now();

    auto cpu_gen_time = std::chrono::duration<double, std::milli>(cpu_gen_end - cpu_start).count();
    auto cpu_verify_time =
        std::chrono::duration<double, std::milli>(cpu_verify_end - cpu_gen_end).count();
    auto cuda_gen_time =
        std::chrono::duration<double, std::milli>(cuda_gen_end - cuda_start).count();
    auto cuda_verify_time =
        std::chrono::duration<double, std::milli>(cuda_verify_end - cuda_gen_end).count();

    double total_cpu_time = cpu_gen_time + cpu_verify_time;
    double total_cuda_time = cuda_gen_time + cuda_verify_time;
    double speedup = total_cuda_time > 0 ? total_cpu_time / total_cuda_time : 0;

    std::cout << std::setw(10) << batch_size << " | " << std::setw(18) << std::fixed
              << std::setprecision(2) << cpu_gen_time << " | " << std::setw(18) << cuda_gen_time
              << " | " << std::setw(15) << cpu_verify_time << " | " << std::setw(15)
              << cuda_verify_time << " | " << std::setw(7) << speedup << "x\n";

    EXPECT_TRUE(cpu_verify_result);
    EXPECT_TRUE(cuda_verify_result);
  }

  std::cout << std::string(100, '=') << "\n";
}

TEST_F(MerkleTreeCUDABenchmarkTest, OptimalConfigurationAnalysis) {
  print_benchmark_header("OPTIMAL CONFIGURATION ANALYSIS FOR GPU");

  std::vector<size_t> dataset_sizes = {1000, 5000, 10000};
  std::vector<size_t> arities = {2, 4, 8};
  const size_t num_trees = 5;

  for (size_t size : dataset_sizes) {
    std::cout << "\n>>> DATASET SIZE: " << size << " leaves <<<\n";
    std::cout << "Arity | Build Time (ms) | Trees/sec | Speedup vs CPU\n";
    std::cout << std::string(55, '-') << "\n";

    double best_perf = 0;
    size_t best_arity = 2;

    for (size_t arity : arities) {
      auto stats = benchmark_cuda_vs_cpu_merkle(num_trees, size, arity);

      double perf_score = stats.trees_per_second;
      if (perf_score > best_perf) {
        best_perf = perf_score;
        best_arity = arity;
      }

      std::cout << std::setw(5) << arity << " | " << std::setw(15) << std::fixed
                << std::setprecision(2) << stats.build_time_ms << " | " << std::setw(9)
                << stats.trees_per_second << " | " << std::setw(13) << stats.speedup_vs_cpu
                << "x\n";
    }

    std::cout << ">> Optimal arity for " << size << " leaves: " << best_arity << " (" << best_perf
              << " trees/sec)\n";
  }

  std::cout << std::string(100, '=') << "\n";
}

TEST_F(MerkleTreeCUDABenchmarkTest, ThroughputComparison) {
  print_benchmark_header("THROUGHPUT COMPARISON: CPU vs GPU");

  const size_t leaf_count = 2048;
  const size_t arity = 4;
  std::vector<size_t> tree_counts = {1, 5, 10, 25, 50, 100};

  std::cout << "Tree Count | CPU Build (ms) | GPU Build (ms) | CPU Throughput (trees/s) | GPU "
               "Throughput (trees/s) | Speedup\n";
  std::cout << std::string(110, '-') << "\n";

  for (size_t tree_count : tree_counts) {
    auto stats = benchmark_cuda_vs_cpu_merkle(tree_count, leaf_count, arity);

    // Calculate CPU throughput (reverse engineer from speedup)
    double cpu_time = stats.build_time_ms * stats.speedup_vs_cpu;
    double cpu_throughput = cpu_time > 0 ? (tree_count * 1000.0) / cpu_time : 0;

    std::cout << std::setw(10) << tree_count << " | " << std::setw(15) << std::fixed
              << std::setprecision(2) << cpu_time << " | " << std::setw(15) << stats.build_time_ms
              << " | " << std::setw(24) << std::fixed << std::setprecision(1) << cpu_throughput
              << " | " << std::setw(24) << static_cast<double>(stats.trees_per_second) << " | "
              << std::setw(7) << std::setprecision(2) << stats.speedup_vs_cpu << "x\n";
  }

  std::cout << std::string(110, '=') << "\n";
}

// Quick benchmark test for CI/development
TEST_F(MerkleTreeCUDABenchmarkTest, QuickPerformanceCheck) {
  print_benchmark_header("QUICK PERFORMANCE CHECK");

  const size_t leaf_count = 512;
  const size_t arity = 4;
  const size_t num_proofs = 50;

  // CPU benchmark
  auto cpu_result = MerkleUtils::benchmark_tree(leaf_count, arity, num_proofs);

  // CUDA benchmark
  auto cuda_build_stats = benchmark_cuda_tree_building(1, leaf_count, arity);
  auto cuda_proof_stats = benchmark_cuda_proof_generation(num_proofs, leaf_count, arity);
  auto cuda_verify_stats = benchmark_cuda_proof_verification(num_proofs, leaf_count, arity);

  // Combine results
  CudaMerkleTreeStats combined_stats = {};
  combined_stats.leaf_count = leaf_count;
  combined_stats.arity = arity;
  combined_stats.tree_height = cuda_build_stats.tree_height;
  combined_stats.build_time_ms = cuda_build_stats.build_time_ms;
  combined_stats.proof_generation_time_ms = cuda_proof_stats.proof_generation_time_ms;
  combined_stats.proof_verification_time_ms = cuda_verify_stats.proof_verification_time_ms;

  print_comparison_results(cpu_result, combined_stats);

  // Basic sanity checks
  EXPECT_GT(cuda_build_stats.build_time_ms, 0);
  EXPECT_GE(cuda_proof_stats.proof_generation_time_ms, 0);  // Allow 0 for very fast operations
  EXPECT_GT(cuda_verify_stats.proof_verification_time_ms, 0);
  EXPECT_EQ(combined_stats.tree_height, cpu_result.tree_height);
}