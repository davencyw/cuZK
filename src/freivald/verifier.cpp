#include "verifier.h"
#include "prover.h"  
#include <stdexcept>
#include <cmath>

namespace freivald {

Verifier::Verifier(unsigned int seed) : rng_(seed) {}

double Verifier::getErrorProbability(int repetitions) {
    if (repetitions <= 0) {
        throw std::invalid_argument("Number of repetitions must be positive");
    }
    
    // Error probability is (1/2)^repetitions
    return std::pow(0.5, repetitions);
}

int Verifier::getRequiredRepetitions(double desired_error_prob) {
    if (desired_error_prob <= 0.0 || desired_error_prob >= 1.0) {
        throw std::invalid_argument("Desired error probability must be between 0 and 1");
    }
    
    // We need (1/2)^k <= desired_error_prob
    // Taking log: k * log(1/2) <= log(desired_error_prob)
    // k >= log(desired_error_prob) / log(1/2)
    return static_cast<int>(std::ceil(std::log(desired_error_prob) / std::log(0.5)));
}

Eigen::VectorXi Verifier::generateRandomBinaryVector(int size) {
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