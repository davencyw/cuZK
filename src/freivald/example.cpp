#include "matrix_generator.h"
#include "prover.h"
#include "verifier.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace freivald;

int main() {
    std::cout << "=== Matrix Multiplication Verification Example ===" << std::endl;
    std::cout << "Using Freivald's Algorithm with Eigen" << std::endl << std::endl;

    // Create components with a fixed seed for reproducible results
    MatrixGenerator generator(12345);
    Verifier verifier(12345);

    // Example 1: Basic verification with small matrices
    std::cout << "Example 1: Small Matrix Verification" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    auto A1 = generator.generateRandomMatrix<int>(3, 3, -5, 5);
    auto B1 = generator.generateRandomMatrix<int>(3, 3, -5, 5);
    auto C1 = Prover::multiply(A1, B1);
    bool result1 = verifier.verify(A1, B1, C1, 10);
    std::cout << "Matrix dimensions: 3x3 * 3x3, Repetitions: 10" << std::endl;
    std::cout << "Result: " << (result1 ? "VERIFIED" : "FAILED") << std::endl << std::endl;

    // Example 2: Test with error introduction
    std::cout << "Example 2: Error Detection Test" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    auto A2 = generator.generateRandomMatrix<int>(3, 3, -5, 5);
    auto B2 = generator.generateRandomMatrix<int>(3, 3, -5, 5);
    auto C2 = Prover::multiply(A2, B2);
    // Introduce error
    if (C2.rows() > 0 && C2.cols() > 0) {
        C2(0, 0) += 1;
    }
    bool result2 = verifier.verify(A2, B2, C2, 15);
    std::cout << "Matrix dimensions: 3x3 * 3x3, Repetitions: 15 (with error)" << std::endl;
    std::cout << "Result: " << (result2 ? "VERIFIED (unexpected!)" : "FAILED (as expected)") << std::endl << std::endl;

    // Example 3: Test different repetition factors
    std::cout << "Example 3: Configurable Repetition Factor" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    std::vector<int> repetitions = {1, 5, 10, 20};
    for (int reps : repetitions) {
        double error_prob = Verifier::getErrorProbability(reps);
        // Generate and test matrices for each repetition count
        auto A = generator.generateRandomMatrix<int>(4, 4, -10, 10);
        auto B = generator.generateRandomMatrix<int>(4, 4, -10, 10);
        auto C = Prover::multiply(A, B);
        bool result = verifier.verify(A, B, C, reps);
        std::cout << "Repetitions: " << std::setw(2) << reps 
                  << " | Error probability: " << std::scientific << std::setprecision(2) << error_prob
                  << " | Result: " << (result ? "PASS" : "FAIL") << std::endl;
    }
    std::cout << std::endl;

    // Example 4: Error probability analysis
    std::cout << "Example 4: Error Probability Analysis" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    std::cout << "Required repetitions for different error probabilities:" << std::endl;
    
    std::vector<double> desired_error_probs = {0.1, 0.01, 0.001, 0.0001};
    for (double prob : desired_error_probs) {
        int required_reps = Verifier::getRequiredRepetitions(prob);
        std::cout << "Error prob: " << std::scientific << std::setprecision(1) << prob
                  << " => Required repetitions: " << required_reps << std::endl;
    }
    std::cout << std::endl;

    // Example 5: Performance test with larger matrices
    std::cout << "Example 5: Performance Test" << std::endl;
    std::cout << "-------------------------" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto A_large = generator.generateRandomMatrix<int>(50, 40, -10, 10);
    auto B_large = generator.generateRandomMatrix<int>(40, 60, -10, 10);
    auto C_large = Prover::multiply(A_large, B_large);
    bool large_result = verifier.verify(A_large, B_large, C_large, 5);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Matrix size: 50x40 * 40x60, Repetitions: 5" << std::endl;
    std::cout << "Result: " << (large_result ? "VERIFIED" : "FAILED") << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl << std::endl;

    // Example 6: Matrix Display and Verification
    std::cout << "Example 6: Matrix Display and Verification" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Generate matrices with different seed for this example
    MatrixGenerator generator2(54321);
    Eigen::MatrixXi A = generator2.generateRandomMatrix<int>(3, 3, -5, 5);
    Eigen::MatrixXi B = generator2.generateRandomMatrix<int>(3, 3, -5, 5);
    
    std::cout << "Generated Matrix A:" << std::endl << A << std::endl << std::endl;
    std::cout << "Generated Matrix B:" << std::endl << B << std::endl << std::endl;
    
    // Compute multiplication using prover
    Eigen::MatrixXi C = Prover::multiply(A, B);
    std::cout << "Computed C = A * B:" << std::endl << C << std::endl << std::endl;
    
    // Verify the result
    Verifier verifier2(54321);
    bool verification_result = verifier2.verify(A, B, C, 20);
    std::cout << "Verification with 20 repetitions: " << (verification_result ? "PASSED" : "FAILED") << std::endl;

    // Example 7: Using different scalar types
    std::cout << "\nExample 7: Different Scalar Types" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    // Test with double matrices
    auto A_double = generator.generateRandomMatrix<double>(2, 2, -1.0, 1.0);
    auto B_double = generator.generateRandomMatrix<double>(2, 2, -1.0, 1.0);
    auto C_double = Prover::multiply(A_double, B_double);
    bool double_result = verifier.verify(A_double, B_double, C_double, 10, 1e-10);
    std::cout << "Double precision matrices (2x2): " << (double_result ? "VERIFIED" : "FAILED") << std::endl;
    
    // Test with float matrices
    auto A_float = generator.generateRandomMatrix<float>(2, 2, -1.0f, 1.0f);
    auto B_float = generator.generateRandomMatrix<float>(2, 2, -1.0f, 1.0f);
    auto C_float = Prover::multiply(A_float, B_float);
    bool float_result = verifier.verify(A_float, B_float, C_float, 10, 1e-6f);
    std::cout << "Float precision matrices (2x2): " << (float_result ? "VERIFIED" : "FAILED") << std::endl;
    
    return 0;
} 