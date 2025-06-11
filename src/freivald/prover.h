#pragma once

#include <Eigen/Dense>
#include <stdexcept>

namespace freivald {

// Prover class for computing matrix multiplication
class Prover {
public:
    // Matrix multiplication C = A * B (templated for any scalar type)
    template<typename Scalar>
    static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> multiply(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B);

    // Matrix-vector multiplication y = A * x (templated for any scalar type)
    template<typename Scalar>
    static Eigen::Matrix<Scalar, Eigen::Dynamic, 1> multiply(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x);

    // Freivald verification helper: compute A * (B * r) for random vector r (templated)
    template<typename Scalar>
    static Eigen::Matrix<Scalar, Eigen::Dynamic, 1> computeABr(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& r);

    // Check if matrices have compatible dimensions for multiplication
    template<typename MatrixType1, typename MatrixType2>
    static bool areCompatible(const MatrixType1& A, const MatrixType2& B) {
        return A.cols() == B.rows();
    }

    // Check if matrix and vector have compatible dimensions for multiplication
    template<typename MatrixType, typename VectorType>
    static bool areCompatibleMV(const MatrixType& A, const VectorType& x) {
        return A.cols() == x.size();
    }
};

// Template implementations
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Prover::multiply(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B) {
    
    if (!areCompatible(A, B)) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }
    
    return A * B;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Prover::multiply(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x) {
    
    if (!areCompatibleMV(A, x)) {
        throw std::invalid_argument("Matrix and vector dimensions are incompatible for multiplication");
    }
    
    return A * x;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Prover::computeABr(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& B,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& r) {
    
    if (!areCompatible(A, B)) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }
    if (B.cols() != r.size()) {
        throw std::invalid_argument("Matrix B and vector r dimensions are incompatible");
    }
    
    // Compute B * r first, then A * (B * r)
    auto Br = B * r;
    return A * Br;
}

} // namespace freivald 