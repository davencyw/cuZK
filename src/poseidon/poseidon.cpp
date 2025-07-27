#include "poseidon.hpp"

#include <chrono>

#include "../common/namespace_utils.hpp"

namespace Poseidon {

USING_FIELD_CONSTANTS()
USING_FIELD_OPS()

// Initialize static constants
bool PoseidonConstants::initialized_ = false;

// Static constants (will be properly initialized in init())
std::array<FieldElement,
           PoseidonParams::TOTAL_ROUNDS * PoseidonParams::STATE_SIZE>
    PoseidonConstants::ROUND_CONSTANTS;
std::array<FieldElement,
           PoseidonParams::STATE_SIZE * PoseidonParams::STATE_SIZE>
    PoseidonConstants::MDS_MATRIX;

void PoseidonConstants::init() {
  if (initialized_)
    return;

  FieldConstants::init();
  generate_round_constants();
  generate_mds_matrix();
  initialized_ = true;
}

void PoseidonConstants::generate_round_constants() {
  // For demonstration, using simple deterministic constants
  // In practice, these would be generated using a proper method like LFSR
  for (size_t i = 0; i < ROUND_CONSTANTS.size(); ++i) {
    FieldElement base_value(i + 1);
    // Mix in some more entropy
    multiply(base_value, FieldElement(0x123456789ABCDEFULL),
             ROUND_CONSTANTS[i]);
    add(ROUND_CONSTANTS[i], FieldElement(i * 0x987654321ULL),
        ROUND_CONSTANTS[i]);
  }
}

void PoseidonConstants::generate_mds_matrix() {
  // Generate a simple MDS matrix for demonstration
  // In practice, this would be carefully chosen for optimal security
  MDS_MATRIX[0] = FieldElement(7);
  MDS_MATRIX[1] = FieldElement(23);
  MDS_MATRIX[2] = FieldElement(8);
  MDS_MATRIX[3] = FieldElement(26);
  MDS_MATRIX[4] = FieldElement(5);
  MDS_MATRIX[5] = FieldElement(4);
  MDS_MATRIX[6] = FieldElement(15);
  MDS_MATRIX[7] = FieldElement(20);
  MDS_MATRIX[8] = FieldElement(9);
}

void PoseidonHash::permutation(FieldElement state[PoseidonParams::STATE_SIZE]) {
  if (!PoseidonConstants::initialized_) {
    PoseidonConstants::init();
  }

  size_t round = 0;

  // First half of full rounds
  for (size_t r = 0; r < PoseidonParams::ROUNDS_FULL / 2; ++r) {
    add_round_constants(state, round++);
    apply_sbox(state);
    apply_mds_matrix(state);
  }

  // Partial rounds
  for (size_t r = 0; r < PoseidonParams::ROUNDS_PARTIAL; ++r) {
    add_round_constants(state, round++);
    apply_partial_sbox(state);
    apply_mds_matrix(state);
  }

  // Second half of full rounds
  for (size_t r = 0; r < PoseidonParams::ROUNDS_FULL / 2; ++r) {
    add_round_constants(state, round++);
    apply_sbox(state);
    apply_mds_matrix(state);
  }
}

FieldElement PoseidonHash::hash_single(const FieldElement &input) {
  return sponge({input}, FieldElement(1));
}

FieldElement PoseidonHash::hash_pair(const FieldElement &left,
                                     const FieldElement &right) {
  return sponge({left, right}, FieldElement(2));
}

FieldElement
PoseidonHash::hash_multiple(const std::vector<FieldElement> &inputs) {
  return sponge(inputs, FieldElement(3));
}

FieldElement PoseidonHash::sponge(const std::vector<FieldElement> &inputs, const FieldElement &domain_separator) {
  // Initialize with the provided domain separator
  FieldElement state[PoseidonParams::STATE_SIZE] = {
      domain_separator,
      ZERO, ZERO};

  // Absorb phase
  size_t input_idx = 0;
  while (input_idx < inputs.size()) {
    // Add up to RATE inputs to the state
    for (size_t i = 0; i < PoseidonParams::RATE && input_idx < inputs.size();
         ++i) {
      add(state[i + PoseidonParams::CAPACITY], inputs[input_idx],
          state[i + PoseidonParams::CAPACITY]);
      input_idx++;
    }

    // Apply permutation
    permutation(state);
  }

  // Squeeze phase - return first rate element
  return state[PoseidonParams::CAPACITY];
}

void PoseidonHash::add_round_constants(
    FieldElement state[PoseidonParams::STATE_SIZE], size_t round) {
  for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
    size_t const_idx = round * PoseidonParams::STATE_SIZE + i;
    add(state[i], PoseidonConstants::ROUND_CONSTANTS[const_idx], state[i]);
  }
}

void PoseidonHash::apply_sbox(FieldElement state[PoseidonParams::STATE_SIZE]) {
  for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
    power5(state[i], state[i]);
  }
}

void PoseidonHash::apply_partial_sbox(
    FieldElement state[PoseidonParams::STATE_SIZE]) {
  // Only apply S-box to the first element in partial rounds
  power5(state[0], state[0]);
}

void PoseidonHash::apply_mds_matrix(
    FieldElement state[PoseidonParams::STATE_SIZE]) {
  FieldElement new_state[PoseidonParams::STATE_SIZE];

  for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
    new_state[i] = ZERO;
    for (size_t j = 0; j < PoseidonParams::STATE_SIZE; ++j) {
      FieldElement temp;
      multiply(
          PoseidonConstants::MDS_MATRIX[i * PoseidonParams::STATE_SIZE + j],
          state[j], temp);
      add(new_state[i], temp, new_state[i]);
    }
  }

  // Copy back to original state
  for (size_t i = 0; i < PoseidonParams::STATE_SIZE; ++i) {
    state[i] = new_state[i];
  }
}

// Performance benchmarking
HashingStats benchmark_poseidon(size_t num_iterations) {
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_iterations; ++i) {
    FieldElement input = FieldElement::random();
    FieldElement result = PoseidonHash::hash_single(input);
    // Use result to prevent optimization
    (void)result;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  HashingStats stats;
  stats.total_time_ms = duration.count() / 1000000.0;
  stats.avg_time_per_hash_ns =
      static_cast<double>(duration.count()) / num_iterations;
  stats.hashes_per_second =
      static_cast<size_t>(1000000000.0 / stats.avg_time_per_hash_ns);
  stats.total_hashes = num_iterations;

  return stats;
}

HashingStats benchmark_poseidon_pairs(size_t num_pairs) {
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_pairs; ++i) {
    FieldElement left = FieldElement::random();
    FieldElement right = FieldElement::random();
    FieldElement result = PoseidonHash::hash_pair(left, right);
    // Use result to prevent optimization
    (void)result;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  HashingStats stats;
  stats.total_time_ms = duration.count() / 1000000.0;
  stats.avg_time_per_hash_ns =
      static_cast<double>(duration.count()) / num_pairs;
  stats.hashes_per_second =
      static_cast<size_t>(1000000000.0 / stats.avg_time_per_hash_ns);
  stats.total_hashes = num_pairs;

  return stats;
}

} // namespace Poseidon