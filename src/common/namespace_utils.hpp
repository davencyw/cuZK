#pragma once

// Forward declarations for clean namespace usage
namespace Poseidon {
class FieldElement;
class PoseidonHash;
namespace FieldConstants {
extern const FieldElement ZERO;
extern const FieldElement ONE;
extern const FieldElement TWO;
extern const FieldElement MODULUS;
} // namespace FieldConstants
namespace FieldArithmetic {
void add(const FieldElement &a, const FieldElement &b, FieldElement &result);
void subtract(const FieldElement &a, const FieldElement &b,
              FieldElement &result);
void multiply(const FieldElement &a, const FieldElement &b,
              FieldElement &result);
void reduce(FieldElement &a);
void power5(const FieldElement &a, FieldElement &result);
} // namespace FieldArithmetic
} // namespace Poseidon

namespace MerkleTree {
class NaryMerkleTree;
struct MerkleTreeConfig;
struct MerkleProof;
} // namespace MerkleTree

// Macro for consistent namespace usage in implementations
// Note: These macros should be used in .cpp files after including the necessary
// headers

#define USING_CUZK_TYPES()                                                     \
  using FieldElement = Poseidon::FieldElement;                                 \
  using Hash = Poseidon::PoseidonHash;

#define USING_MERKLE_TYPES()                                                   \
  using NaryTree = MerkleTree::NaryMerkleTree;                                 \
  using TreeConfig = MerkleTree::MerkleTreeConfig;                             \
  using Proof = MerkleTree::MerkleProof;

#define USING_FIELD_CONSTANTS()                                                \
  using Poseidon::FieldConstants::ZERO;                                        \
  using Poseidon::FieldConstants::ONE;                                         \
  using Poseidon::FieldConstants::TWO;                                         \
  using Poseidon::FieldConstants::MODULUS;

#define USING_FIELD_OPS()                                                      \
  using Poseidon::FieldArithmetic::add;                                        \
  using Poseidon::FieldArithmetic::subtract;                                   \
  using Poseidon::FieldArithmetic::multiply;                                   \
  using Poseidon::FieldArithmetic::reduce;                                     \
  using Poseidon::FieldArithmetic::power5;

// Note: CUDA error handling utilities are macros and should be used directly
// Include "../common/error_handling.hpp" to access CUDA_CHECK_* macros