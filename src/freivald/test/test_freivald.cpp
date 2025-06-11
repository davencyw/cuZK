#include <gtest/gtest.h>
#include "matrix_generator.h"
#include "prover.h"
#include "verifier.h"
#include <Eigen/Dense>

using namespace freivald;

const int kTestingSeed = 42069;

class FreivaldTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use fixed seed for reproducible tests
        generator = std::make_unique<MatrixGenerator>(kTestingSeed);
        verifier = std::make_unique<Verifier>(kTestingSeed);
    }

    std::unique_ptr<MatrixGenerator> generator;
    std::unique_ptr<Verifier> verifier;
};

// Test Matrix Generator
TEST_F(FreivaldTest, MatrixGeneratorBasic) {
    const auto matrix = generator->generateRandomMatrix<int>(3, 4, -10, 10);
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 4);
    
    // Check that values are in range
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            EXPECT_GE(matrix(i, j), -10);
            EXPECT_LE(matrix(i, j), 10);
        }
    }
}

TEST_F(FreivaldTest, MatrixGeneratorDouble) {
    const auto matrix = generator->generateRandomMatrix<double>(2, 3, -5.0, 5.0);
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 3);
    
    // Check that values are in range
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            EXPECT_GE(matrix(i, j), -5.0);
            EXPECT_LE(matrix(i, j), 5.0);
        }
    }
}

TEST_F(FreivaldTest, MatrixGeneratorFloat) {
    const auto matrix = generator->generateRandomMatrix<float>(2, 2, -1.0f, 1.0f);
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 2);
    
    // Check that values are in range
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            EXPECT_GE(matrix(i, j), -1.0f);
            EXPECT_LE(matrix(i, j), 1.0f);
        }
    }
}

TEST_F(FreivaldTest, VectorGeneration) {
    auto vec = generator->generateRandomVector<int>(5, -2, 2);
    EXPECT_EQ(vec.size(), 5);
    
    for (int i = 0; i < vec.size(); ++i) {
        EXPECT_GE(vec(i), -2);
        EXPECT_LE(vec(i), 2);
    }
}

TEST_F(FreivaldTest, BinaryVectorGeneration) {
    const auto vec = generator->generateRandomBinaryVector(10);
    EXPECT_EQ(vec.size(), 10);
    
    for (int i = 0; i < vec.size(); ++i) {
        EXPECT_TRUE(vec(i) == 0 || vec(i) == 1);
    }
}

// Test Prover
TEST_F(FreivaldTest, ProverMatrixMultiplication) {
    Eigen::MatrixXi A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    
    Eigen::MatrixXi B(3, 2);
    B << 7, 8,
         9, 10,
         11, 12;
    
    Eigen::MatrixXi C = Prover::multiply(A, B);
    
    // Expected result: [58, 64; 139, 154]
    Eigen::MatrixXi expected(2, 2);
    expected << 58, 64,
                139, 154;
    
    EXPECT_EQ(C, expected);
}

TEST_F(FreivaldTest, ProverMatrixVectorMultiplication) {
    Eigen::MatrixXi A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    
    Eigen::VectorXi x(3);
    x << 1, 2, 3;
    
    Eigen::VectorXi y = Prover::multiply(A, x);
    
    // Expected result: [14, 32]
    Eigen::VectorXi expected(2);
    expected << 14, 32;
    
    EXPECT_EQ(y, expected);
}

TEST_F(FreivaldTest, ProverIncompatibleDimensions) {
    Eigen::MatrixXi A(2, 3);
    Eigen::MatrixXi B(4, 2); // Incompatible: A.cols() != B.rows()
    
    EXPECT_THROW(Prover::multiply(A, B), std::invalid_argument);
}

// Test Verifier - Correct Multiplications
TEST_F(FreivaldTest, VerifierCorrectMultiplication) {
    Eigen::MatrixXi A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    
    Eigen::MatrixXi B(3, 2);
    B << 7, 8,
         9, 10,
         11, 12;
    
    Eigen::MatrixXi C = Prover::multiply(A, B);
    
    // Verify with multiple repetitions
    EXPECT_TRUE(verifier->verify(A, B, C, 20));
}

TEST_F(FreivaldTest, VerifierIncorrectMultiplication) {
    Eigen::MatrixXi A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    
    Eigen::MatrixXi B(3, 2);
    B << 7, 8,
         9, 10,
         11, 12;
    
    // Create incorrect result
    Eigen::MatrixXi C_wrong(2, 2);
    C_wrong << 1, 2,
               3, 4;
    
    // Should fail verification with high probability
    EXPECT_FALSE(verifier->verify(A, B, C_wrong, 20));
}

TEST_F(FreivaldTest, VerifierSingleRound) {
    Eigen::MatrixXi A(2, 2);
    A << 1, 2,
         3, 4;
    
    Eigen::MatrixXi B(2, 2);
    B << 5, 6,
         7, 8;
    
    Eigen::MatrixXi C_correct = Prover::multiply(A, B);
    Eigen::MatrixXi C_wrong(2, 2);
    C_wrong << 0, 0,
               0, 0;
    
    // Test multiple single rounds to check probabilistic behavior
    int correct_passes = 0;
    int wrong_passes = 0;
    int trials = 100;
    
    for (int i = 0; i < trials; ++i) {
        Verifier v(i); // Different seed each time
        if (v.verifySingleRound(A, B, C_correct)) correct_passes++;
        if (v.verifySingleRound(A, B, C_wrong)) wrong_passes++;
    }
    
    // Correct result should pass most/all times
    EXPECT_GT(correct_passes, trials * 0.9);
    
    // Wrong result should pass roughly half the time (probabilistic)
    EXPECT_LT(wrong_passes, trials * 0.7); // Allow some variance
}

// Test Error Probability Calculations
TEST_F(FreivaldTest, ErrorProbabilityCalculation) {
    EXPECT_DOUBLE_EQ(Verifier::getErrorProbability(1), 0.5);
    EXPECT_DOUBLE_EQ(Verifier::getErrorProbability(2), 0.25);
    EXPECT_DOUBLE_EQ(Verifier::getErrorProbability(10), 1.0/1024.0);
}

TEST_F(FreivaldTest, RequiredRepetitionsCalculation) {
    EXPECT_EQ(Verifier::getRequiredRepetitions(0.5), 1);
    EXPECT_EQ(Verifier::getRequiredRepetitions(0.25), 2);
    EXPECT_EQ(Verifier::getRequiredRepetitions(0.01), 7); // ceil(log(0.01)/log(0.5))
}

// Test Double Precision Support
TEST_F(FreivaldTest, DoubleMatrixVerification) {
    Eigen::MatrixXd A(2, 2);
    A << 1.5, 2.5,
         3.5, 4.5;
    
    Eigen::MatrixXd B(2, 2);
    B << 0.5, 1.5,
         2.5, 3.5;
    
    Eigen::MatrixXd C = Prover::multiply(A, B);
    
    EXPECT_TRUE(verifier->verify(A, B, C, 10, 1e-10));
    
    // Test with wrong result
    Eigen::MatrixXd C_wrong = C;
    C_wrong(0, 0) += 0.1; // Small but detectable error
    
    EXPECT_FALSE(verifier->verify(A, B, C_wrong, 10, 1e-10));
}

// Test Complete System Using Individual Components
TEST_F(FreivaldTest, SystemRandomMatricesTest) {
    // Test various matrix sizes using individual components
    auto testMatrices = [this](int m, int n, int p, int repetitions) {
        auto A = generator->generateRandomMatrix<int>(m, n, -10, 10);
        auto B = generator->generateRandomMatrix<int>(n, p, -10, 10);
        auto C = Prover::multiply(A, B);
        return verifier->verify(A, B, C, repetitions);
    };
    
    EXPECT_TRUE(testMatrices(3, 4, 5, 15));
    EXPECT_TRUE(testMatrices(10, 8, 6, 20));
    EXPECT_TRUE(testMatrices(1, 1, 1, 5));
}

TEST_F(FreivaldTest, SystemWorkflowCorrect) {
    // Test correct workflow using individual components
    auto A = generator->generateRandomMatrix<int>(3, 3, -5, 5);
    auto B = generator->generateRandomMatrix<int>(3, 3, -5, 5);
    auto C = Prover::multiply(A, B);
    
    bool result = verifier->verify(A, B, C, 10);
    EXPECT_TRUE(result);
}

TEST_F(FreivaldTest, SystemWorkflowWithError) {
    // Test workflow with introduced error
    auto A = generator->generateRandomMatrix<int>(3, 3, -5, 5);
    auto B = generator->generateRandomMatrix<int>(3, 3, -5, 5);
    auto C = Prover::multiply(A, B);
    
    // Introduce error
    if (C.rows() > 0 && C.cols() > 0) {
        C(0, 0) += 1;
    }
    
    bool result = verifier->verify(A, B, C, 20);
    EXPECT_FALSE(result); // Should fail with high probability
}

// Test Edge Cases and Error Handling
TEST_F(FreivaldTest, InvalidDimensions) {
    EXPECT_THROW(generator->generateRandomMatrix<int>(0, 5, -10, 10), std::invalid_argument);
    EXPECT_THROW(generator->generateRandomMatrix<int>(5, -1, -10, 10), std::invalid_argument);
    EXPECT_THROW(generator->generateRandomVector<int>(-1, -10, 10), std::invalid_argument);
    EXPECT_THROW(generator->generateRandomBinaryVector(0), std::invalid_argument);
}

TEST_F(FreivaldTest, InvalidRepetitions) {
    Eigen::MatrixXi A = Eigen::MatrixXi::Identity(2, 2);
    Eigen::MatrixXi B = Eigen::MatrixXi::Identity(2, 2);
    Eigen::MatrixXi C = Eigen::MatrixXi::Identity(2, 2);
    
    EXPECT_THROW(verifier->verify(A, B, C, 0), std::invalid_argument);
    EXPECT_THROW(verifier->verify(A, B, C, -1), std::invalid_argument);
}

TEST_F(FreivaldTest, IncompatibleMatrixDimensions) {
    Eigen::MatrixXi A(2, 3);
    Eigen::MatrixXi B(4, 2); // Wrong dimension
    Eigen::MatrixXi C(2, 2);
    
    EXPECT_THROW(verifier->verify(A, B, C, 5), std::invalid_argument);
}

// Stress Test
TEST_F(FreivaldTest, StressTestLargerMatrices) {
    // Test with larger matrices (but still reasonable for CI)
    auto testLargeMatrices = [this](int m, int n, int p, int repetitions) {
        auto A = generator->generateRandomMatrix<int>(m, n, -10, 10);
        auto B = generator->generateRandomMatrix<int>(n, p, -10, 10);
        auto C = Prover::multiply(A, B);
        return verifier->verify(A, B, C, repetitions);
    };
    
    EXPECT_TRUE(testLargeMatrices(20, 15, 25, 5));
    EXPECT_TRUE(testLargeMatrices(50, 30, 40, 3));
}

// Test Configurable Repetition Factor
TEST_F(FreivaldTest, ConfigurableRepetitions) {
    Eigen::MatrixXi A = generator->generateRandomMatrix<int>(5, 5, -10, 10);
    Eigen::MatrixXi B = generator->generateRandomMatrix<int>(5, 5, -10, 10);
    Eigen::MatrixXi C = Prover::multiply(A, B);
    
    // Test different repetition factors
    EXPECT_TRUE(verifier->verify(A, B, C, 1));
    EXPECT_TRUE(verifier->verify(A, B, C, 5));
    EXPECT_TRUE(verifier->verify(A, B, C, 15));
    EXPECT_TRUE(verifier->verify(A, B, C, 30));
}

// Test Prover computeABr method
TEST_F(FreivaldTest, ProverComputeABr) {
    Eigen::MatrixXi A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    
    Eigen::MatrixXi B(3, 2);
    B << 7, 8,
         9, 10,
         11, 12;
    
    Eigen::VectorXi r(2);
    r << 1, 0;
    
    // Compute A * (B * r) using new method
    Eigen::VectorXi ABr = Prover::computeABr(A, B, r);
    
    // Compute manually for verification
    Eigen::VectorXi Br = Prover::multiply(B, r);
    Eigen::VectorXi expected = Prover::multiply(A, Br);
    
    EXPECT_EQ(ABr, expected);
}

// Test additional verification scenarios
TEST_F(FreivaldTest, VerifierAdditionalCorrect) {
    Eigen::MatrixXi A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    
    Eigen::MatrixXi B(3, 2);
    B << 7, 8,
         9, 10,
         11, 12;
    
    Eigen::MatrixXi C = Prover::multiply(A, B);
    
    // Verify with high repetition count for confidence
    EXPECT_TRUE(verifier->verify(A, B, C, 20));
}

TEST_F(FreivaldTest, VerifierAdditionalIncorrect) {
    Eigen::MatrixXi A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    
    Eigen::MatrixXi B(3, 2);
    B << 7, 8,
         9, 10,
         11, 12;
    
    // Create incorrect result
    Eigen::MatrixXi C_wrong(2, 2);
    C_wrong << 1, 2,
               3, 4;
    
    // Should fail verification
    EXPECT_FALSE(verifier->verify(A, B, C_wrong, 20));
} 