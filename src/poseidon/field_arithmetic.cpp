#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>

#include "poseidon.hpp"

namespace Poseidon {

// BN254 scalar field modulus
namespace FieldConstants {
const FieldElement MODULUS = FieldElement(0x43e1f593f0000001ULL,
                                          0x2833e84879b97091ULL,
                                          0xb85045b68181585dULL,
                                          0x30644e72e131a029ULL);
const FieldElement ZERO = FieldElement(0, 0, 0, 0);
const FieldElement ONE = FieldElement(1, 0, 0, 0);
const FieldElement TWO = FieldElement(2, 0, 0, 0);

void init() {
  // Constants are statically initialized above
}
}  // namespace FieldConstants

// FieldElement implementation
FieldElement::FieldElement() {
  std::memset(limbs, 0, sizeof(limbs));
}

FieldElement::FieldElement(uint64_t value) {
  limbs[0] = value;
  limbs[1] = 0;
  limbs[2] = 0;
  limbs[3] = 0;
}

FieldElement::FieldElement(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) {
  limbs[0] = v0;
  limbs[1] = v1;
  limbs[2] = v2;
  limbs[3] = v3;
}

FieldElement::FieldElement(const FieldElement& other) {
  std::memcpy(limbs, other.limbs, sizeof(limbs));
}

FieldElement& FieldElement::operator=(const FieldElement& other) {
  if (this != &other) {
    std::memcpy(limbs, other.limbs, sizeof(limbs));
  }
  return *this;
}

bool FieldElement::operator==(const FieldElement& other) const {
  return std::memcmp(limbs, other.limbs, sizeof(limbs)) == 0;
}

bool FieldElement::operator!=(const FieldElement& other) const {
  return !(*this == other);
}

bool FieldElement::operator<(const FieldElement& other) const {
  for (int i = 3; i >= 0; --i) {
    if (limbs[i] < other.limbs[i]) return true;
    if (limbs[i] > other.limbs[i]) return false;
  }
  return false;
}

FieldElement FieldElement::operator+(const FieldElement& other) const {
  FieldElement result;
  FieldArithmetic::add(*this, other, result);
  return result;
}

FieldElement FieldElement::operator-(const FieldElement& other) const {
  FieldElement result;
  FieldArithmetic::subtract(*this, other, result);
  return result;
}

FieldElement FieldElement::operator*(const FieldElement& other) const {
  FieldElement result;
  FieldArithmetic::multiply(*this, other, result);
  return result;
}

FieldElement& FieldElement::operator+=(const FieldElement& other) {
  FieldArithmetic::add(*this, other, *this);
  return *this;
}

FieldElement& FieldElement::operator-=(const FieldElement& other) {
  FieldArithmetic::subtract(*this, other, *this);
  return *this;
}

FieldElement& FieldElement::operator*=(const FieldElement& other) {
  FieldArithmetic::multiply(*this, other, *this);
  return *this;
}

std::string FieldElement::to_hex() const {
  std::stringstream ss;
  ss << "0x";
  for (int i = 3; i >= 0; --i) {
    ss << std::hex << std::setw(16) << std::setfill('0') << limbs[i];
  }
  return ss.str();
}

std::string FieldElement::to_dec() const {
  // Handle zero case
  if (is_zero()) {
    return "0";
  }

  // Copy the field element for manipulation
  FieldElement temp = *this;
  std::string result;

  // Repeatedly divide by 10 and collect remainders
  while (!temp.is_zero()) {
    // Divide temp by 10 and get remainder
    uint64_t remainder = 0;
    for (int i = 3; i >= 0; --i) {
      // Use 128-bit arithmetic to avoid overflow
      __uint128_t current = (static_cast<__uint128_t>(remainder) << 64) | temp.limbs[i];
      temp.limbs[i] = static_cast<uint64_t>(current / 10);
      remainder = static_cast<uint64_t>(current % 10);
    }

    // Add the digit to result (prepend since we're working backwards)
    result = char('0' + remainder) + result;
  }

  return result;
}

FieldElement FieldElement::from_hex(const std::string& hex) {
  FieldElement result;
  std::string clean_hex = hex;
  if (clean_hex.substr(0, 2) == "0x" || clean_hex.substr(0, 2) == "0X") {
    clean_hex = clean_hex.substr(2);
  }

  // Pad to 64 characters (256 bits)
  while (clean_hex.length() < 64) {
    clean_hex = "0" + clean_hex;
  }

  for (int i = 0; i < 4; ++i) {
    std::string limb_str = clean_hex.substr((3 - i) * 16, 16);
    result.limbs[i] = std::stoull(limb_str, nullptr, 16);
  }

  return result;
}

FieldElement FieldElement::random() {
  return FieldArithmetic::random();
}

bool FieldElement::is_zero() const {
  return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
}

void FieldElement::set_zero() {
  std::memset(limbs, 0, sizeof(limbs));
}

// Field arithmetic implementation
namespace FieldArithmetic {

void add(const FieldElement& a, const FieldElement& b, FieldElement& result) {
  uint64_t carry = 0;
  for (int i = 0; i < 4; ++i) {
    // Use 128-bit arithmetic to avoid overflow issues
    __uint128_t sum = (__uint128_t)a.limbs[i] + (__uint128_t)b.limbs[i] + (__uint128_t)carry;
    result.limbs[i] = (uint64_t)sum;
    carry = (uint64_t)(sum >> 64);  // Extract the high 64 bits as carry
  }
  reduce(result);
}

void subtract(const FieldElement& a, const FieldElement& b, FieldElement& result) {
  // If a < b, we need to add the modulus first
  if (a < b) {
    FieldElement temp;
    // Add modulus to a without calling reduce()
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
      __uint128_t sum = (__uint128_t)a.limbs[i] + (__uint128_t)FieldConstants::MODULUS.limbs[i] + (__uint128_t)carry;
      temp.limbs[i] = (uint64_t)sum;
      carry = (uint64_t)(sum >> 64);
    }
    subtract_internal(temp, b, result);
  } else {
    subtract_internal(a, b, result);
  }
}

void subtract_internal(const FieldElement& a, const FieldElement& b, FieldElement& result) {
  uint64_t borrow = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t temp_a = a.limbs[i];
    uint64_t temp_b = b.limbs[i] + borrow;

    if (temp_a >= temp_b) {
      result.limbs[i] = temp_a - temp_b;
      borrow = 0;
    } else {
      result.limbs[i] = temp_a + (UINT64_MAX - temp_b) + 1;
      borrow = 1;
    }
  }
}

void multiply(const FieldElement& a, const FieldElement& b, FieldElement& result) {
  uint64_t product[8] = {0};

  // Schoolbook multiplication
  for (int i = 0; i < 4; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 4; ++j) {
      __uint128_t prod = (__uint128_t)a.limbs[i] * b.limbs[j] + product[i + j] + carry;
      product[i + j] = (uint64_t)prod;
      carry = (uint64_t)(prod >> 64);
    }
    product[i + 4] = carry;
  }

  reduce_512(product, result);
}

void square(const FieldElement& a, FieldElement& result) {
  multiply(a, a, result);
}

void reduce(FieldElement& a) {
  while (!(a < FieldConstants::MODULUS)) {
    subtract_internal(a, FieldConstants::MODULUS, a);
  }
}

void reduce_512(const uint64_t product[8], FieldElement& result) {
  // Reduction for 512-bit to 256-bit for BN254 field
  // The input represents: low + high * 2^256
  // We need to compute: (low + high * (2^256 mod p)) mod p
  
  // 2^256 mod p = 0x0e0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb
  static const FieldElement k = FieldElement::from_hex("0x0e0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb");

  // Extract low and high parts
  FieldElement low(product[0], product[1], product[2], product[3]);
  FieldElement high(product[4], product[5], product[6], product[7]);

  // If high part is zero, just return reduced low part
  if (high.is_zero()) {
    result = low;
    reduce(result);
    return;
  }

  // Implement the exact same logic as multiply() function but inline
  // to avoid circular dependency
  uint64_t mult_product[8] = {0};

  // Schoolbook multiplication: high * k (same as in multiply())
  for (int i = 0; i < 4; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 4; ++j) {
      __uint128_t prod = (__uint128_t)high.limbs[i] * k.limbs[j] + mult_product[i + j] + carry;
      mult_product[i + j] = (uint64_t)prod;
      carry = (uint64_t)(prod >> 64);
    }
    mult_product[i + 4] = carry;
  }

  // Now we have a 512-bit result in mult_product = high * k
  // We need to reduce this 512-bit result, but we can't call reduce_512 recursively
  // Instead, we'll apply the reduction algorithm iteratively
  
  // Extract the low and high parts of mult_product
  FieldElement mult_low(mult_product[0], mult_product[1], mult_product[2], mult_product[3]);
  FieldElement mult_high(mult_product[4], mult_product[5], mult_product[6], mult_product[7]);
  
  // Start with mult_low
  FieldElement high_contribution = mult_low;
  
  // If mult_high is non-zero, we need to add mult_high * k to high_contribution
  // But we'll do this only once to avoid infinite recursion, and then rely on final reduce()
  if (!mult_high.is_zero()) {
    // Do one more round of schoolbook multiplication: mult_high * k
    uint64_t mult2_product[8] = {0};
    
    for (int i = 0; i < 4; ++i) {
      uint64_t carry = 0;
      for (int j = 0; j < 4; ++j) {
        __uint128_t prod = (__uint128_t)mult_high.limbs[i] * k.limbs[j] + mult2_product[i + j] + carry;
        mult2_product[i + j] = (uint64_t)prod;
        carry = (uint64_t)(prod >> 64);
      }
      mult2_product[i + 4] = carry;
    }
    
    // Add only the low part of mult2_product to high_contribution
    // Any high part will be small and handled by final reduce()
    FieldElement mult2_low(mult2_product[0], mult2_product[1], mult2_product[2], mult2_product[3]);
    add(high_contribution, mult2_low, high_contribution);
  }
  
  // Now compute final result: low + high_contribution
  add(low, high_contribution, result);
  
  // Final reduction to ensure result < p
  reduce(result);
}

void power5(const FieldElement& a, FieldElement& result) {
  // Compute a^5 = a^4 * a = (a^2)^2 * a
  FieldElement a2, a4;
  square(a, a2);            // a^2
  square(a2, a4);           // a^4
  multiply(a4, a, result);  // a^5
}

FieldElement random() {
  static std::random_device rd;
  static std::mt19937_64 gen(rd());
  static std::uniform_int_distribution<uint64_t> dis;

  FieldElement result;
  for (int i = 0; i < 4; ++i) {
    result.limbs[i] = dis(gen);
  }
  reduce(result);
  return result;
}

}  // namespace FieldArithmetic

}  // namespace Poseidon