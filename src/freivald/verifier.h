#pragma once

#include <Eigen/Dense>
#include <random>
#include <stdexcept>
#include <type_traits>
#include "prover.h"

namespace freivald {

// Verifier class implementing Freivald's algorithm for matrix multiplication verification
class Verifier {
public:
    // Constructor with optional seed for deterministic testing
    explicit Verifier(unsigned int seed = std::random_device{}());

    // Verify if A * B = C using Freivald's algorithm
    template<typename Scalar>
    bool verify(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, 
                const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B, 
                const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& C, 
                int repetitions = 10, Scalar tolerance = Scalar(1e-10));

    // Single round of Freivald's verification algorithm
    template<typename Scalar>
    bool verifySingleRound(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, 
                          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B, 
                          const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& C, 
                          Scalar tolerance = Scalar(1e-10));

    // Error probability calculations
    static double getErrorProbability(int repetitions);
    static int getRequiredRepetitions(double desired_error_prob);

private:
    std::mt19937 rng_;

    // Generate random binary vector for Freivald's algorithm
    Eigen::VectorXi generateRandomBinaryVector(int size);

    // Check if matrices have compatible dimensions for verification
    template<typename MatrixType>
    bool checkDimensions(const MatrixType& A, const MatrixType& B, const MatrixType& C) {
        return (A.cols() == B.rows()) && (A.rows() == C.rows()) && (B.cols() == C.cols());
    }
};

// Template implementations
template<typename Scalar>
bool Verifier::verify(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, 
                     const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B, 
                     const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& C, 
                     int repetitions, Scalar tolerance) {
    if (repetitions <= 0) {
        throw std::invalid_argument("Number of repetitions must be positive");
    }
    
    if (!checkDimensions(A, B, C)) {
        throw std::invalid_argument("Matrix dimensions are incompatible for verification");
    }
    
    for (int i = 0; i < repetitions; ++i) {
        if (!verifySingleRound(A, B, C, tolerance)) {
            return false;
        }
    }
    
    return true;
}

template<typename Scalar>
bool Verifier::verifySingleRound(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A, 
                                 const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B, 
                                 const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& C, 
                                 Scalar tolerance) {
    if (!checkDimensions(A, B, C)) {
        throw std::invalid_argument("Matrix dimensions are incompatible for verification");
    }
    
    // Generate random binary vector r
    Eigen::VectorXi binary_r = generateRandomBinaryVector(B.cols());
    
    // Convert binary vector to the appropriate scalar type
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> r(binary_r.size());
    for (int i = 0; i < binary_r.size(); ++i) {
        r(i) = static_cast<Scalar>(binary_r(i));
    }
    
    // Use prover to compute A * (B * r) and C * r (prover-delegated approach)
    auto ABr = Prover::computeABr(A, B, r);
    auto Cr = Prover::multiply(C, r);
    
    // Check if A * (B * r) equals C * r
    if constexpr (std::is_integral_v<Scalar>) {
        return ABr == Cr;
    } else {
        return (ABr - Cr).norm() <= tolerance;
    }
}

} // namespace freivald 