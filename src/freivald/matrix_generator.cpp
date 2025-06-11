#include "matrix_generator.h"
#include <stdexcept>

namespace freivald {

MatrixGenerator::MatrixGenerator(unsigned int seed) : rng_(seed) {}

Eigen::VectorXi MatrixGenerator::generateRandomBinaryVector(int size) {
    if (size <= 0) {
        throw std::invalid_argument("Vector size must be positive");
    }

    std::uniform_int_distribution<int> dist(0, 1);
    
    Eigen::VectorXi vector(size);
    for (int i = 0; i < size; ++i) {
        vector(i) = dist(rng_);
    }
    
    return vector;
}

} // namespace freivald 