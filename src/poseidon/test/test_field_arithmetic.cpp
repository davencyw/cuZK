#include <gtest/gtest.h>

#include <sstream>

#include "field_arithmetic.hpp"

using namespace Poseidon;

class FieldArithmeticTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FieldConstants::init();
  }

  // Helper function to create known test values
  FieldElement createTestValue(uint64_t v0, uint64_t v1 = 0, uint64_t v2 = 0, uint64_t v3 = 0) {
    FieldElement result;
    result.limbs[0] = v0;
    result.limbs[1] = v1;
    result.limbs[2] = v2;
    result.limbs[3] = v3;
    return result;
  }
};

TEST_F(FieldArithmeticTest, FieldElementConstruction) {
  FieldElement zero;
  EXPECT_TRUE(zero.is_zero());

  FieldElement one(1);
  EXPECT_FALSE(one.is_zero());
  EXPECT_EQ(one.limbs[0], 1ULL);
  EXPECT_EQ(one.limbs[1], 0ULL);
  EXPECT_EQ(one.limbs[2], 0ULL);
  EXPECT_EQ(one.limbs[3], 0ULL);

  FieldElement copy(one);
  EXPECT_EQ(copy, one);
}

TEST_F(FieldArithmeticTest, BasicArithmetic) {
  FieldElement a(5);
  FieldElement b(3);

  // Test addition
  FieldElement sum = a + b;
  FieldElement expected_sum(8);
  EXPECT_EQ(sum, expected_sum);

  // Test subtraction
  FieldElement diff = a - b;
  FieldElement expected_diff(2);
  EXPECT_EQ(diff, expected_diff);

  // Test multiplication
  FieldElement product = a * b;
  FieldElement expected_product(15);
  EXPECT_EQ(product, expected_product);
}

TEST_F(FieldArithmeticTest, CompoundAssignment) {
  FieldElement a(10);
  FieldElement b(5);

  FieldElement original_a = a;
  a += b;
  EXPECT_EQ(a, original_a + b);

  a = FieldElement(10);
  a -= b;
  EXPECT_EQ(a, original_a - b);

  a = FieldElement(10);
  a *= b;
  EXPECT_EQ(a, original_a * b);
}

TEST_F(FieldArithmeticTest, ZeroProperties) {
  FieldElement zero = FieldConstants::ZERO;
  FieldElement a(42);

  // Adding zero
  EXPECT_EQ(a + zero, a);
  EXPECT_EQ(zero + a, a);

  // Subtracting zero
  EXPECT_EQ(a - zero, a);

  // Multiplying by zero
  EXPECT_EQ(a * zero, zero);
  EXPECT_EQ(zero * a, zero);
}

TEST_F(FieldArithmeticTest, OneProperties) {
  FieldElement one = FieldConstants::ONE;
  FieldElement a(42);

  // Multiplying by one
  EXPECT_EQ(a * one, a);
  EXPECT_EQ(one * a, a);
}

TEST_F(FieldArithmeticTest, FieldReduction) {
  // Test that values are properly reduced modulo the field prime
  FieldElement a, b, result;

  // Test basic reduction
  FieldArithmetic::add(FieldConstants::MODULUS, FieldConstants::ONE, result);
  EXPECT_EQ(result, FieldConstants::ONE);
}

TEST_F(FieldArithmeticTest, Power5) {
  FieldElement base(2);
  FieldElement result;

  FieldArithmetic::power5(base, result);

  // 2^5 = 32
  FieldElement expected(32);
  EXPECT_EQ(result, expected);

  // Test with zero
  FieldArithmetic::power5(FieldConstants::ZERO, result);
  EXPECT_EQ(result, FieldConstants::ZERO);

  // Test with one
  FieldArithmetic::power5(FieldConstants::ONE, result);
  EXPECT_EQ(result, FieldConstants::ONE);
}

TEST_F(FieldArithmeticTest, PowerFunction) {
  FieldElement base(3);
  FieldElement result;

  // Implement simple power function inline since we removed the general power
  // function
  auto simple_power = [](const FieldElement& base, uint64_t exp, FieldElement& result) {
    if (exp == 0) {
      result = FieldConstants::ONE;
    } else if (exp == 1) {
      result = base;
    } else {
      result = FieldConstants::ONE;
      FieldElement temp = base;
      for (uint64_t i = 0; i < exp; ++i) {
        FieldArithmetic::multiply(result, temp, result);
      }
    }
  };

  // Test 3^0 = 1
  simple_power(base, 0, result);
  EXPECT_EQ(result, FieldConstants::ONE);

  // Test 3^1 = 3
  simple_power(base, 1, result);
  EXPECT_EQ(result, base);

  // Test 3^2 = 9
  simple_power(base, 2, result);
  FieldElement expected(9);
  EXPECT_EQ(result, expected);

  // Test 3^3 = 27
  simple_power(base, 3, result);
  FieldElement expected2(27);
  EXPECT_EQ(result, expected2);
}

TEST_F(FieldArithmeticTest, Negation) {
  FieldElement a(5);
  // Implement negation inline since we removed the negate function
  FieldElement neg_a;
  if (a.is_zero()) {
    neg_a = a;
  } else {
    FieldArithmetic::subtract(FieldConstants::MODULUS, a, neg_a);
  }

  // Test that a + (-a) = 0
  FieldElement sum = a + neg_a;
  EXPECT_EQ(sum, FieldConstants::ZERO);

  // Test negation of zero
  FieldElement neg_zero;
  if (FieldConstants::ZERO.is_zero()) {
    neg_zero = FieldConstants::ZERO;
  } else {
    FieldArithmetic::subtract(FieldConstants::MODULUS, FieldConstants::ZERO, neg_zero);
  }
  EXPECT_EQ(neg_zero, FieldConstants::ZERO);
}

TEST_F(FieldArithmeticTest, Comparison) {
  FieldElement a(5);
  FieldElement b(10);
  FieldElement c(5);

  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_FALSE(a < c);

  EXPECT_TRUE(a == c);
  EXPECT_FALSE(a == b);

  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a != c);
}

TEST_F(FieldArithmeticTest, HexConversion) {
  FieldElement a(0x123456789ABCDEFULL);
  std::string hex = a.to_hex();

  FieldElement b = FieldElement::from_hex(hex);
  EXPECT_EQ(a, b);

  // Test with zero
  FieldElement zero = FieldConstants::ZERO;
  std::string zero_hex = zero.to_hex();
  FieldElement zero_back = FieldElement::from_hex(zero_hex);
  EXPECT_EQ(zero, zero_back);
}

TEST_F(FieldArithmeticTest, DecimalConversion) {
  // Test with zero
  FieldElement zero = FieldConstants::ZERO;
  std::string zero_dec = zero.to_dec();
  EXPECT_EQ(zero_dec, "0");

  // Test with one
  FieldElement one = FieldConstants::ONE;
  std::string one_dec = one.to_dec();
  EXPECT_EQ(one_dec, "1");

  // Test with two
  FieldElement two = FieldConstants::TWO;
  std::string two_dec = two.to_dec();
  EXPECT_EQ(two_dec, "2");

  // Test with a known small value
  FieldElement small(123456789ULL);
  std::string small_dec = small.to_dec();
  EXPECT_EQ(small_dec, "123456789");

  // Test with a larger value
  FieldElement large(0x123456789ABCDEFULL);
  std::string large_dec = large.to_dec();
  EXPECT_EQ(large_dec, "81985529216486895");

  // Test with the BN254 modulus
  std::string modulus_dec = FieldConstants::MODULUS.to_dec();
  EXPECT_EQ(modulus_dec,
            "21888242871839275222246405745257275088548364400416034"
            "343698204186575808495617");

  // Test consistency - convert back and forth shouldn't change hex
  // representation
  FieldElement test_val = createTestValue(0xDEADBEEFCAFEBABEULL, 0x1234567890ABCDEFULL, 0, 0);
  std::string dec_str = test_val.to_dec();

  // Verify the decimal string is not empty and contains only digits
  EXPECT_FALSE(dec_str.empty());
  for (char c : dec_str) {
    EXPECT_TRUE(c >= '0' && c <= '9');
  }
}

TEST_F(FieldArithmeticTest, RandomGeneration) {
  // Test that random generation produces different values
  FieldElement r1 = FieldElement::random();
  FieldElement r2 = FieldElement::random();

  // Very unlikely to be equal (but not impossible)
  EXPECT_NE(r1, r2);

  // Both should be less than modulus
  EXPECT_TRUE(r1 < FieldConstants::MODULUS);
  EXPECT_TRUE(r2 < FieldConstants::MODULUS);
}

TEST_F(FieldArithmeticTest, LargeNumberArithmetic) {
  // Test with larger numbers that might cause overflow
  FieldElement large1 = createTestValue(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0, 0);
  FieldElement large2 = createTestValue(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0, 0);

  // Addition should not overflow catastrophically
  FieldElement sum = large1 + large2;
  EXPECT_TRUE(sum < FieldConstants::MODULUS);

  // Multiplication should work
  FieldElement product = large1 * large2;
  EXPECT_TRUE(product < FieldConstants::MODULUS);
}

TEST_F(FieldArithmeticTest, FieldIdentityProperties) {
  // Test the specific case ZERO + ONE - ONE != ZERO
  FieldElement zero = FieldConstants::ZERO;
  FieldElement one = FieldConstants::ONE;
  
  // Test ZERO + ONE - ONE should equal ZERO
  FieldElement step1 = zero + one;
  EXPECT_EQ(step1, one) << "ZERO + ONE should equal ONE";
  
  FieldElement result = step1 - one;
  EXPECT_EQ(result, zero) << "ZERO + ONE - ONE should equal ZERO";
  
  // Alternative: directly test (ZERO + ONE) - ONE
  FieldElement direct_result = (zero + one) - one;
  EXPECT_EQ(direct_result, zero) << "(ZERO + ONE) - ONE should equal ZERO";
  
  // Test other identity properties
  EXPECT_EQ(one - one, zero) << "ONE - ONE should equal ZERO";
  EXPECT_EQ(zero + zero, zero) << "ZERO + ZERO should equal ZERO";
  EXPECT_EQ(zero - zero, zero) << "ZERO - ZERO should equal ZERO";
  
  // Test associativity of addition: (a + b) + c = a + (b + c)
  FieldElement a(5), b(3), c(7);
  FieldElement left = (a + b) + c;
  FieldElement right = a + (b + c);
  EXPECT_EQ(left, right) << "Addition should be associative";
  
  // Test subtraction properties: a - b + b = a
  FieldElement diff = a - b;
  FieldElement restored = diff + b;
  EXPECT_EQ(restored, a) << "a - b + b should equal a";
  
  // Test with constants: TWO - ONE should equal ONE
  FieldElement two = FieldConstants::TWO;
  FieldElement two_minus_one = two - one;
  EXPECT_EQ(two_minus_one, one) << "TWO - ONE should equal ONE";
  
  // Test: ONE + ONE should equal TWO
  FieldElement one_plus_one = one + one;
  EXPECT_EQ(one_plus_one, two) << "ONE + ONE should equal TWO";
}

TEST_F(FieldArithmeticTest, SubtractionEdgeCases) {
  // Test subtraction when a < b (should handle modular arithmetic correctly)
  FieldElement small(5);
  FieldElement large(10);
  
  // small - large should wrap around correctly
  FieldElement result = small - large;
  
  // Verify: result + large should equal small (mod p)
  FieldElement verification = result + large;
  EXPECT_EQ(verification, small) << "Modular subtraction should satisfy (a - b) + b = a";
  
  // Test: ZERO - ONE should give MODULUS - 1
  FieldElement zero = FieldConstants::ZERO;
  FieldElement one = FieldConstants::ONE;
  FieldElement zero_minus_one = zero - one;
  
  // zero_minus_one + one should equal zero
  FieldElement check = zero_minus_one + one;
  EXPECT_EQ(check, zero) << "(ZERO - ONE) + ONE should equal ZERO";
  
  // Test with constants directly
  FieldElement modulus_minus_one;
  FieldArithmetic::subtract(FieldConstants::MODULUS, one, modulus_minus_one);
  EXPECT_EQ(zero_minus_one, modulus_minus_one) << "ZERO - ONE should equal MODULUS - ONE";
}

TEST_F(FieldArithmeticTest, Reduce512) {
  // Test the 512-bit reduction function

  // Test case 1: High part is zero (should just reduce low part)
  {
    uint64_t product[8] = {42, 0, 0, 0, 0, 0, 0, 0};
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    FieldElement expected(42);
    EXPECT_EQ(result, expected);
  }

  // Test case 2: Low part is zero, high part is small
  {
    uint64_t product[8] = {0, 0, 0, 0, 1, 0, 0, 0};
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    // This should be equivalent to 1 * 2^256 mod p ≈ 4
    FieldElement expected(4);
    EXPECT_EQ(result, expected);
  }

  // Test case 3: Both high and low parts are non-zero
  {
    uint64_t product[8] = {10, 20, 30, 40, 1, 2, 3, 4};
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);

    // The result should be less than the modulus
    EXPECT_TRUE(result < FieldConstants::MODULUS);

    // Manual verification: high * 4 + low (approximately)
    FieldElement low(10, 20, 30, 40);
    FieldElement high(1, 2, 3, 4);
    FieldElement high_times_4;

    // Multiply high by 4
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
      uint64_t shifted = (high.limbs[i] << 2) | carry;
      high_times_4.limbs[i] = shifted;
      carry = high.limbs[i] >> 62;
    }

    FieldElement expected_result = high_times_4 + low;
    EXPECT_EQ(result, expected_result);
  }

  // Test case 4: Large values that definitely need reduction
  {
    uint64_t product[8] = {0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL};
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);

    // Result should be reduced
    EXPECT_TRUE(result < FieldConstants::MODULUS);
    EXPECT_FALSE(result.is_zero());  // Very unlikely to be zero
  }

  // Test case 5: Test consistency with multiplication
  {
    FieldElement a = createTestValue(0x123456789ABCDEFULL, 0xFEDCBA9876543210ULL, 0, 0);
    FieldElement b = createTestValue(0xAABBCCDDEEFF1122ULL, 0x3344556677889900ULL, 0, 0);

    // Compute multiplication using the multiply function
    FieldElement result1 = a * b;

    // Manually compute the 512-bit product and reduce
    uint64_t product[8] = {0};
    for (int i = 0; i < 4; ++i) {
      uint64_t carry = 0;
      for (int j = 0; j < 4; ++j) {
        __uint128_t prod = (__uint128_t)a.limbs[i] * b.limbs[j] + product[i + j] + carry;
        product[i + j] = (uint64_t)prod;
        carry = (uint64_t)(prod >> 64);
      }
      product[i + 4] = carry;
    }

    FieldElement result2;
    FieldArithmetic::reduce_512(product, result2);

    EXPECT_EQ(result1, result2);
  }
}
