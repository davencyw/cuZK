#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace Poseidon {

// Field element structure for 256-bit prime field arithmetic
struct FieldElement {
  // 4 64-bit limbs to represent 256-bit numbers
  uint64_t limbs[4];

  // Constructors
  FieldElement();
  explicit FieldElement(uint64_t value);
  FieldElement(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3);
  FieldElement(const FieldElement &other);

  // Assignment operator
  FieldElement &operator=(const FieldElement &other);

  // Comparison operators
  bool operator==(const FieldElement &other) const;
  bool operator!=(const FieldElement &other) const;
  bool operator<(const FieldElement &other) const;

  // Arithmetic operations
  FieldElement operator+(const FieldElement &other) const;
  FieldElement operator-(const FieldElement &other) const;
  FieldElement operator*(const FieldElement &other) const;
  FieldElement &operator+=(const FieldElement &other);
  FieldElement &operator-=(const FieldElement &other);
  FieldElement &operator*=(const FieldElement &other);

  // Utility functions
  std::string to_hex() const;
  std::string to_dec() const;
  static FieldElement from_hex(const std::string &hex);
  static FieldElement random();
  bool is_zero() const;
  void set_zero();
};

// Field arithmetic operations
namespace FieldArithmetic {
// Basic operations
void add(const FieldElement &a, const FieldElement &b, FieldElement &result);
void subtract(const FieldElement &a, const FieldElement &b,
              FieldElement &result);
void multiply(const FieldElement &a, const FieldElement &b,
              FieldElement &result);
void square(const FieldElement &a, FieldElement &result);

// Modular operations
void reduce(FieldElement &a);

// Power operations
void power5(const FieldElement &a,
            FieldElement &result); // Optimized x^5 for S-box

// Random number generation
FieldElement random();

// Internal helper functions
void subtract_internal(const FieldElement &a, const FieldElement &b,
                       FieldElement &result);
void reduce_512(const uint64_t product[8], FieldElement &result);
} // namespace FieldArithmetic

// Constants for the prime field
namespace FieldConstants {
// BN254 scalar field modulus:
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
extern const FieldElement MODULUS;
extern const FieldElement ZERO;
extern const FieldElement ONE;
extern const FieldElement TWO;

void init();
} // namespace FieldConstants

} // namespace Poseidon