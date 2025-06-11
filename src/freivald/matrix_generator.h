#pragma once

#include <Eigen/Dense>
#include <random>
#include <type_traits>

namespace freivald {

// Matrix generator for creating random matrices for testing
class MatrixGenerator {
public:
    // Constructor with optional seed for reproducible results
    explicit MatrixGenerator(unsigned int seed = std::random_device{}());

    // Generate random matrix with templated scalar type
    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> generateRandomMatrix(
        int rows, int cols, Scalar min_val, Scalar max_val);

    // Generate random vector with templated scalar type  
    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> generateRandomVector(
        int size, Scalar min_val, Scalar max_val);

    // Generate random binary vector (0s and 1s)
    Eigen::VectorXi generateRandomBinaryVector(int size);

private:
    std::mt19937 rng_;
};

// Template implementations
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixGenerator::generateRandomMatrix(
    int rows, int cols, Scalar min_val, Scalar max_val) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (min_val >= max_val) {
        throw std::invalid_argument("min_val must be less than max_val");
    }

    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Matrix matrix(rows, cols);

    if constexpr (std::is_integral_v<Scalar>) {
        std::uniform_int_distribution<Scalar> dist(min_val, max_val);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = dist(rng_);
            }
        }
    } else {
        std::uniform_real_distribution<Scalar> dist(min_val, max_val);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = dist(rng_);
            }
        }
    }
    
    return matrix;
}

template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> MatrixGenerator::generateRandomVector(
    int size, Scalar min_val, Scalar max_val) {
    if (size <= 0) {
        throw std::invalid_argument("Vector size must be positive");
    }
    if (min_val >= max_val) {
        throw std::invalid_argument("min_val must be less than max_val");
    }

    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    Vector vector(size);

    if constexpr (std::is_integral_v<Scalar>) {
        std::uniform_int_distribution<Scalar> dist(min_val, max_val);
        for (int i = 0; i < size; ++i) {
            vector(i) = dist(rng_);
        }
    } else {
        std::uniform_real_distribution<Scalar> dist(min_val, max_val);
        for (int i = 0; i < size; ++i) {
            vector(i) = dist(rng_);
        }
    }
    
    return vector;
}

} // namespace freivald 