#include "../matrix_generator.h"
#include "../prover.h"
#include "../verifier.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <type_traits>

using namespace freivald;

struct BenchmarkResult {
    int matrix_size;
    int repetitions;
    double freivald_time_ms;
    double recompute_time_ms;
    double speedup;
    std::string matrix_type;
    bool verification_correct;
};

template<typename Scalar>
class MatrixBenchmark {
private:
    MatrixGenerator generator;
    Verifier verifier;
    
    static std::string getTypeName() {
        if constexpr (std::is_same_v<Scalar, int>) return "int";
        else if constexpr (std::is_same_v<Scalar, float>) return "float";
        else if constexpr (std::is_same_v<Scalar, double>) return "double";
        else return "unknown";
    }
    
public:
    MatrixBenchmark(unsigned int seed = 42) : generator(seed), verifier(seed) {}
    
    // Benchmark Freivald's algorithm verification
    double benchmarkFreivaldVerification(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B, 
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& C,
        int repetitions, int iterations = 100) {
        
        // Use appropriate tolerances for each type
        Scalar tolerance;
        if constexpr (std::is_integral_v<Scalar>) {
            tolerance = Scalar(0);  // Exact comparison for integers
        } else if constexpr (std::is_same_v<Scalar, float>) {
            tolerance = Scalar(0.5);  // Very relaxed tolerance for float (precision limitations)
        } else {
            tolerance = Scalar(1e-8);   // Tighter tolerance for double precision
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            // Each iteration performs ONE complete Freivald verification 
            // (which internally does 'repetitions' probabilistic rounds)
            bool result = verifier.verify(A, B, C, repetitions, tolerance);
            (void)result; // Avoid unused variable warning
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return duration.count() / 1000.0 / iterations; // Return milliseconds per verification
    }
    
    // Benchmark recomputing matrix multiplication for verification
    double benchmarkRecomputeVerification(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& C,
        int iterations = 100) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            // Recompute A×B once per verification (realistic verification approach)
            auto computed_C = Prover::multiply(A, B);
            
            // Compare with expected result
            bool matches;
            if constexpr (std::is_integral_v<Scalar>) {
                matches = (computed_C == C);
            } else {
                matches = (computed_C - C).norm() <= Scalar(1e-10);
            }
            (void)matches; // Avoid unused variable warning
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return duration.count() / 1000.0 / iterations; // Return milliseconds per iteration
    }
    

    
    // Run comprehensive benchmark for given matrix size
    BenchmarkResult runBenchmark(int size, int repetitions) {
        std::cout << "  Benchmarking " << getTypeName() << " matrices: " 
                  << size << "x" << size << " with " << repetitions << " repetitions..." << std::flush;
        
        // Generate test matrices
        Scalar min_val, max_val;
        if constexpr (std::is_integral_v<Scalar>) {
            min_val = -10;
            max_val = 10;
        } else {
            min_val = Scalar(-1.0);
            max_val = Scalar(1.0);
        }
        
        auto A = generator.generateRandomMatrix<Scalar>(size, size, min_val, max_val);
        auto B = generator.generateRandomMatrix<Scalar>(size, size, min_val, max_val);
        auto C = Prover::multiply(A, B);
        
        // Determine number of iterations based on matrix size for reasonable timing
        // Ensure at least 3 iterations for statistical reliability, even for large matrices
        int iterations = std::max(3, 1000 / (size * size / 100));
        
        // Define tolerance for correctness checking
        Scalar tolerance;
        if constexpr (std::is_integral_v<Scalar>) {
            tolerance = Scalar(0);  // Exact comparison for integers
        } else if constexpr (std::is_same_v<Scalar, float>) {
            tolerance = Scalar(0.5);  // Very relaxed tolerance for float (precision limitations)
        } else {
            tolerance = Scalar(1e-8);   // Tighter tolerance for double precision
        }
        
        // Create verifier for correctness checking
        Verifier correctness_verifier(42);
        
        // Benchmark both approaches
        double freivald_time = benchmarkFreivaldVerification(A, B, C, repetitions, iterations);
        double recompute_time = benchmarkRecomputeVerification(A, B, C, iterations);
        
        // Verify correctness of the verification algorithm
        std::cout << " Checking correctness..." << std::flush;
        
        // Test 1: Correct multiplication should pass
        bool correct_result_passes = correctness_verifier.verify(A, B, C, repetitions, tolerance);
        
        // Test 2: Incorrect multiplication should fail
        auto C_wrong = C;
        if (C_wrong.rows() > 0 && C_wrong.cols() > 0) {
            C_wrong(0, 0) += (std::is_integral_v<Scalar> ? Scalar(1) : Scalar(10.0));
        }
        bool incorrect_result_fails = !correctness_verifier.verify(A, B, C_wrong, repetitions, tolerance);
        
        // Test 3: Check consistency across different repetition counts
        bool consistent_results = true;
        if (repetitions > 1) {
            for (int test_reps : {1, repetitions/2, repetitions}) {
                if (test_reps > 0 && !correctness_verifier.verify(A, B, C, test_reps, tolerance)) {
                    consistent_results = false;
                    break;
                }
            }
        }
        
        bool verification_correct = correct_result_passes && incorrect_result_fails && consistent_results;
        
        if (!verification_correct) {
            std::cout << " VERIFICATION ERROR!" << std::endl;
            std::cout << "    Correct result passes: " << (correct_result_passes ? "YES" : "NO") << std::endl;
            std::cout << "    Incorrect result fails: " << (incorrect_result_fails ? "YES" : "NO") << std::endl;
            std::cout << "    Results consistent: " << (consistent_results ? "YES" : "NO") << std::endl;
            std::cout << "    Tolerance used: " << tolerance << std::endl;
        }
        
        double speedup = recompute_time / freivald_time;
        
        std::cout << " Done!" << (verification_correct ? "" : " [VERIFICATION FAILED]") << std::endl;
        
        return {
            size,
            repetitions, 
            freivald_time,
            recompute_time,
            speedup,
            getTypeName(),
            verification_correct
        };
    }
};

void printHeader() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "        MATRIX MULTIPLICATION VERIFICATION BENCHMARK" << std::endl;
    std::cout << "           Freivald's Algorithm vs. Recomputation" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(90, '-') << std::endl;
    std::cout << "| " << std::setw(6) << "Type" 
              << " | " << std::setw(6) << "Size"
              << " | " << std::setw(4) << "Reps" 
              << " | " << std::setw(12) << "Freivald (ms)" 
              << " | " << std::setw(12) << "Recompute (ms)"
              << " | " << std::setw(10) << "Speedup" 
              << " | " << std::setw(6) << "Status" << " |" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << "| " << std::setw(6) << result.matrix_type
                  << " | " << std::setw(6) << result.matrix_size
                  << " | " << std::setw(4) << result.repetitions
                  << " | " << std::setw(12) << std::fixed << std::setprecision(3) << result.freivald_time_ms
                  << " | " << std::setw(12) << std::fixed << std::setprecision(3) << result.recompute_time_ms  
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2) << result.speedup << "x"
                  << " | " << std::setw(6) << (result.verification_correct ? "OK" : "FAIL") << " |" << std::endl;
    }
    std::cout << std::string(90, '-') << std::endl;
    
    // Summary of verification results
    int total_tests = results.size();
    int passed_tests = 0;
    for (const auto& result : results) {
        if (result.verification_correct) passed_tests++;
    }
    
    std::cout << "\nVERIFICATION SUMMARY: " << passed_tests << "/" << total_tests 
              << " tests passed";
    if (passed_tests != total_tests) {
        std::cout << " (" << (total_tests - passed_tests) << " FAILED!)";
    }
    std::cout << std::endl;
}

void printAnalysis(const std::vector<BenchmarkResult>& results) {
    std::cout << "\nPERFORMANCE ANALYSIS:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // Calculate average speedup by type
    std::map<std::string, std::vector<double>> speedups_by_type;
    
    for (const auto& result : results) {
        speedups_by_type[result.matrix_type].push_back(result.speedup);
    }
    
    for (const auto& [type, speedups] : speedups_by_type) {
        double avg_speedup = std::accumulate(speedups.begin(), speedups.end(), 0.0) / speedups.size();
        double max_speedup = *std::max_element(speedups.begin(), speedups.end());
        double min_speedup = *std::min_element(speedups.begin(), speedups.end());
        
        std::cout << "• " << type << " matrices: Average " << std::fixed << std::setprecision(2) 
                  << avg_speedup << "x speedup (range: " << std::setprecision(2) 
                  << min_speedup << "x - " << max_speedup << "x)" << std::endl;
    }
    
}

int main() {
    printHeader();
    
    std::vector<BenchmarkResult> all_results;
    
    // Test configurations: {size, repetitions}
    std::vector<std::pair<int, int>> test_configs = {
        {50, 20},    {100, 20},   // Small matrices
        {200, 20},   {500, 20},   // Medium matrices  
        {1000, 20},  {1500, 20},  // Large matrices
        {2000, 20},   {3000, 20},   // Very large matrices
        {4000, 20},   {5000, 20},  // Extra large matrices
    };
    
    std::cout << "\nRunning benchmarks..." << std::endl;
    
    // Benchmark all scalar types with the same configurations
    std::cout << "\nInteger Matrix Benchmarks:" << std::endl;
    MatrixBenchmark<int> int_benchmark(42);
    for (const auto& [size, reps] : test_configs) {
        auto result = int_benchmark.runBenchmark(size, reps);
        all_results.push_back(result);
    }
    
    // Benchmark double matrices  
    std::cout << "\nDouble Matrix Benchmarks:" << std::endl;
    MatrixBenchmark<double> double_benchmark(42);
    for (const auto& [size, reps] : test_configs) {
        auto result = double_benchmark.runBenchmark(size, reps);
        all_results.push_back(result);
    }
    
    // Benchmark float matrices
    std::cout << "\nFloat Matrix Benchmarks:" << std::endl;
    MatrixBenchmark<float> float_benchmark(42);
    for (const auto& [size, reps] : test_configs) {
        auto result = float_benchmark.runBenchmark(size, reps);
        all_results.push_back(result);
    }
    
    // Print comprehensive results
    printResults(all_results);
    printAnalysis(all_results);
    
    std::cout << "\nBenchmark completed!" << std::endl;
    
    return 0;
} 