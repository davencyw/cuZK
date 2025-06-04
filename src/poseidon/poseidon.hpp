#pragma once

#include "field_arithmetic.hpp"

namespace Poseidon {

// Poseidon configuration parameters
struct PoseidonParams {
  static constexpr size_t STATE_SIZE = 3;                // t = 3
  static constexpr size_t CAPACITY = 1;                  // c = 1
  static constexpr size_t RATE = STATE_SIZE - CAPACITY;  // r = 2
  static constexpr size_t ROUNDS_FULL = 8;               // R_F = 8
  static constexpr size_t ROUNDS_PARTIAL = 56;           // R_P = 56
  static constexpr size_t TOTAL_ROUNDS = ROUNDS_FULL + ROUNDS_PARTIAL;
  static constexpr size_t ALPHA = 5;  // S-box exponent
};

// Poseidon constants
class PoseidonConstants {
 public:
  // Round constants
  static std::array<FieldElement, PoseidonParams::TOTAL_ROUNDS * PoseidonParams::STATE_SIZE>
      ROUND_CONSTANTS;

  // MDS matrix (3x3 for t=3)
  static std::array<FieldElement, PoseidonParams::STATE_SIZE * PoseidonParams::STATE_SIZE>
      MDS_MATRIX;

  // Initialize constants (call once at startup)
  static void init();

  // Initialization status
  static bool initialized_;

 private:
  static void generate_round_constants();
  static void generate_mds_matrix();
};

// Poseidon permutation and hash functions
class PoseidonHash {
 public:
  // Core permutation function
  static void permutation(FieldElement state[PoseidonParams::STATE_SIZE]);

  // Hash functions
  static FieldElement hash_single(const FieldElement& input);
  static FieldElement hash_pair(const FieldElement& left, const FieldElement& right);
  static FieldElement hash_multiple(const std::vector<FieldElement>& inputs);

  // Sponge construction for variable-length input
  static FieldElement sponge(const std::vector<FieldElement>& inputs);

 private:
  // Internal permutation steps
  static void add_round_constants(FieldElement state[PoseidonParams::STATE_SIZE], size_t round);
  static void apply_sbox(FieldElement state[PoseidonParams::STATE_SIZE]);
  static void apply_partial_sbox(FieldElement state[PoseidonParams::STATE_SIZE]);
  static void apply_mds_matrix(FieldElement state[PoseidonParams::STATE_SIZE]);
};

// Performance testing utilities
struct HashingStats {
  double total_time_ms;
  double avg_time_per_hash_ns;
  size_t hashes_per_second;
  size_t total_hashes;
};

HashingStats benchmark_poseidon(size_t num_iterations = 10000);
HashingStats benchmark_poseidon_pairs(size_t num_pairs = 10000);

}  // namespace Poseidon