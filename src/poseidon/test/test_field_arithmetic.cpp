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

TEST_F(FieldArithmeticTest, Reduce512EdgeCases) {
  // Test edge cases that might expose bugs in reduce_512

  // Test case 1: Test the 2^256 mod p approximation accuracy
  {
    // Create a product where the high part is exactly 1 (representing 2^256)
    uint64_t product[8] = {0, 0, 0, 0, 1, 0, 0, 0};
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    
    // Compute the exact value of 2^256 mod p for comparison
    // We'll compute this by repeated squaring: 2^256 = (2^128)^2
    FieldElement two_to_128 = FieldConstants::ONE;
    for (int i = 0; i < 128; ++i) {
      two_to_128 = two_to_128 + two_to_128; // 2 * two_to_128
    }
    FieldElement exact_two_to_256 = two_to_128 * two_to_128;
    
    EXPECT_EQ(result, exact_two_to_256) << "2^256 mod p approximation should be exact";
  }

  // Test case 2: Test potential overflow in the left shift by 2
  {
    // Use high values that could cause overflow when shifted left by 2
    uint64_t product[8] = {0, 0, 0, 0, 
                           0xFFFFFFFFFFFFFFFFULL,  // Will overflow when shifted left by 2
                           0x3FFFFFFFFFFFFFFFULL,   // Near maximum value that should work
                           0,
                           0};
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    
    // Result should still be valid (less than modulus)
    EXPECT_TRUE(result < FieldConstants::MODULUS);
  }

  // Test case 3: Test maximum possible high value
  {
    uint64_t product[8] = {0, 0, 0, 0,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL,
                           0xFFFFFFFFFFFFFFFFULL};
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    
    EXPECT_TRUE(result < FieldConstants::MODULUS);
  }

  // Test case 4: Test that different representations of the same value give the same result
  {
    // Test value: p + 5, which should reduce to 5
    uint64_t product1[8] = {5, 0, 0, 0, 0, 0, 0, 0}; // Just 5
    uint64_t product2[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // Will be set to p + 5
    
    // Add modulus to the low part manually
    uint64_t carry = 5;
    for (int i = 0; i < 4; ++i) {
      __uint128_t sum = (__uint128_t)FieldConstants::MODULUS.limbs[i] + carry;
      product2[i] = (uint64_t)sum;
      carry = (uint64_t)(sum >> 64);
    }
    
    FieldElement result1, result2;
    FieldArithmetic::reduce_512(product1, result1);
    FieldArithmetic::reduce_512(product2, result2);
    
    EXPECT_EQ(result1, result2) << "Different representations of p + 5 should reduce to the same value";
    
    FieldElement expected(5);
    EXPECT_EQ(result1, expected) << "p + 5 should reduce to 5";
  }

  // Test case 5: Test multiple reduction iterations needed
  {
    // Create a value that's multiple times the modulus
    uint64_t product[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Set to approximately 2 * modulus using high part
    // If 2^256 ≈ 4 (mod p), then 2^257 ≈ 8 (mod p)
    // So we want something that when reduced gives us around 2p
    uint64_t target_multiplier = 2;  // We want ~2p
    product[4] = target_multiplier;
    
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    
    EXPECT_TRUE(result < FieldConstants::MODULUS) << "Result should be properly reduced even when multiple iterations needed";
  }

  // Test case 6: Test the exactness of the 2^256 ≡ 4 (mod p) approximation
  {
    // Verify that the approximation is actually correct for BN254
    // BN254 modulus is: 0x30644e72e131a029b85045b68181585d2833e84879b97091480f3e5934b9f43e1f593f0000001
    
    // Test several powers of 2^256 to see if the approximation holds
    for (int power = 1; power <= 5; ++power) {
      uint64_t product[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      product[4] = power;  // power * 2^256
      
      FieldElement result;
      FieldArithmetic::reduce_512(product, result);
      
      // The result should be equivalent to (power * 4) mod p
      FieldElement expected;
      uint64_t expected_val = power * 4;
      if (expected_val < (1ULL << 63)) { // Check if it fits in a single limb
        expected = FieldElement(expected_val);
      } else {
        // Handle larger values
        expected = FieldElement(expected_val);
        FieldArithmetic::reduce(expected);
      }
      
      EXPECT_EQ(result, expected) << "2^256 * " << power << " mod p should equal " << (power * 4) << " mod p";
    }
  }

  // Test case 7: Test boundary conditions near modulus
  {
    // Test with low part near modulus and small high part
    uint64_t product[8] = {0, 0, 0, 0, 1, 0, 0, 0}; // high part = 1
    
    // Set low part to modulus - 1
    for (int i = 0; i < 4; ++i) {
      product[i] = FieldConstants::MODULUS.limbs[i];
    }
    product[0] -= 1; // modulus - 1
    
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    
    EXPECT_TRUE(result < FieldConstants::MODULUS);
    
    // Verify the computation: (modulus - 1) + 1*4 = modulus + 3 ≡ 3 (mod p)
    FieldElement expected(3);
    EXPECT_EQ(result, expected) << "(modulus - 1) + 2^256 should equal 3 mod p";
  }
}

TEST_F(FieldArithmeticTest, Reduce512ConsistencyWithModularArithmetic) {
  // Test that reduce_512 is consistent with modular arithmetic properties
  
  // Test case 1: Linearity test - reduce_512(a + b) should equal reduce_512(a) + reduce_512(b) mod p
  {
    uint64_t product_a[8] = {100, 200, 300, 400, 1, 2, 3, 4};
    uint64_t product_b[8] = {50, 60, 70, 80, 5, 6, 7, 8};
    uint64_t product_sum[8];
    
    // Add the two products (with carry handling)
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
      __uint128_t sum = (__uint128_t)product_a[i] + product_b[i] + carry;
      product_sum[i] = (uint64_t)sum;
      carry = (uint64_t)(sum >> 64);
    }
    
    FieldElement result_a, result_b, result_sum, expected_sum;
    FieldArithmetic::reduce_512(product_a, result_a);
    FieldArithmetic::reduce_512(product_b, result_b);
    FieldArithmetic::reduce_512(product_sum, result_sum);
    
    expected_sum = result_a + result_b;
    
    EXPECT_EQ(result_sum, expected_sum) << "reduce_512 should be linear: reduce_512(a + b) = reduce_512(a) + reduce_512(b)";
  }

  // Test case 2: Test with known multiplication results
  {
    // Use small known values where we can verify the result manually
    FieldElement a(123456);
    FieldElement b(789012);
    
    // Compute using regular multiplication
    FieldElement expected = a * b;
    
    // Compute the 512-bit product manually and use reduce_512
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
    
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    
    EXPECT_EQ(result, expected) << "reduce_512 should give same result as regular multiplication";
  }

  // Test case 3: Test commutativity through reduce_512
  {
    // Test that a*b and b*a give the same result when reduced via reduce_512
    FieldElement a = createTestValue(0x123456789ABCDEFULL, 0x1111222233334444ULL, 0, 0);
    FieldElement b = createTestValue(0xFEDCBA9876543210ULL, 0x5555666677778888ULL, 0, 0);
    
    // Compute a*b product
    uint64_t product_ab[8] = {0};
    for (int i = 0; i < 4; ++i) {
      uint64_t carry = 0;
      for (int j = 0; j < 4; ++j) {
        __uint128_t prod = (__uint128_t)a.limbs[i] * b.limbs[j] + product_ab[i + j] + carry;
        product_ab[i + j] = (uint64_t)prod;
        carry = (uint64_t)(prod >> 64);
      }
      product_ab[i + 4] = carry;
    }
    
    // Compute b*a product
    uint64_t product_ba[8] = {0};
    for (int i = 0; i < 4; ++i) {
      uint64_t carry = 0;
      for (int j = 0; j < 4; ++j) {
        __uint128_t prod = (__uint128_t)b.limbs[i] * a.limbs[j] + product_ba[i + j] + carry;
        product_ba[i + j] = (uint64_t)prod;
        carry = (uint64_t)(prod >> 64);
      }
      product_ba[i + 4] = carry;
    }
    
    FieldElement result_ab, result_ba;
    FieldArithmetic::reduce_512(product_ab, result_ab);
    FieldArithmetic::reduce_512(product_ba, result_ba);
    
    EXPECT_EQ(result_ab, result_ba) << "reduce_512 should preserve commutativity: reduce_512(a*b) = reduce_512(b*a)";
  }
}

TEST_F(FieldArithmeticTest, Reduce512SpecialValues) {
  // Test reduce_512 with special mathematical values
  
  // Test case 1: Powers of 2
  {
    for (int power = 0; power < 8; ++power) {
      uint64_t product[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (power < 4) {
        product[power] = 1; // 2^(64*power) in low part
      } else {
        product[power] = 1; // 2^(64*power) in high part
      }
      
      FieldElement result;
      FieldArithmetic::reduce_512(product, result);
      
      EXPECT_TRUE(result < FieldConstants::MODULUS) << "2^" << (64 * power) << " mod p should be properly reduced";
    }
  }

  // Test case 2: All ones in different positions
  {
    for (int pos = 0; pos < 8; ++pos) {
      uint64_t product[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      product[pos] = 0xFFFFFFFFFFFFFFFFULL;
      
      FieldElement result;
      FieldArithmetic::reduce_512(product, result);
      
      EXPECT_TRUE(result < FieldConstants::MODULUS) << "All-ones at position " << pos << " should be properly reduced";
    }
  }

  // Test case 3: Alternating bit patterns
  {
    uint64_t patterns[] = {
      0xAAAAAAAAAAAAAAAAULL,
      0x5555555555555555ULL,
      0xCCCCCCCCCCCCCCCCULL,
      0x3333333333333333ULL
    };
    
    for (uint64_t pattern : patterns) {
      uint64_t product[8] = {pattern, pattern, pattern, pattern,
                            pattern, pattern, pattern, pattern};
      
      FieldElement result;
      FieldArithmetic::reduce_512(product, result);
      
      EXPECT_TRUE(result < FieldConstants::MODULUS) << "Pattern 0x" << std::hex << pattern << " should be properly reduced";
    }
  }

  // Test case 4: Test exact modulus multiples
  {
    // Test 0 * modulus (should be 0)
    uint64_t product[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    EXPECT_EQ(result, FieldConstants::ZERO) << "0 * modulus should equal 0";
    
    // Test 1 * modulus (should be 0)
    // This is tricky to set up exactly, so we'll use an approximation
    // We know that if 2^256 ≈ 4 (mod p), then p ≈ 2^256 / 4 = 2^254
    // So we'll test with a high part that should approximate the modulus
    uint64_t product2[8] = {0, 0, 0, 0, 0, 0, 0, 1ULL << 62}; // Approximately 2^254 * 4 ≈ 2^256 ≈ p
    
    FieldElement result2;
    FieldArithmetic::reduce_512(product2, result2);
    
    // The result should be small (close to 0) since it's approximately 1 * modulus
    EXPECT_TRUE(result2 < FieldElement(1000)) << "Approximate modulus should reduce to a small value";
  }
}

TEST_F(FieldArithmeticTest, Reduce512MathematicalCorrectness) {
  // This test verifies the mathematical correctness of the reduce_512 implementation
  // by testing the core assumption: 2^256 ≡ ? (mod p) for BN254
  
  // Test case 1: Verify the exact value of 2^256 mod p
  {
    // Compute 2^256 mod p using repeated doubling (more reliable)
    FieldElement two_to_256 = FieldConstants::ONE;
    for (int i = 0; i < 256; ++i) {
      two_to_256 = two_to_256 + two_to_256; // Double the value
    }
    
    // Test reduce_512 with input representing exactly 2^256
    uint64_t product[8] = {0, 0, 0, 0, 1, 0, 0, 0}; // 1 * 2^256
    FieldElement reduce_512_result;
    FieldArithmetic::reduce_512(product, reduce_512_result);
    
    EXPECT_EQ(reduce_512_result, two_to_256) 
      << "reduce_512(2^256) should match exact computation of 2^256 mod p"
      << "\nExact 2^256 mod p: " << two_to_256.to_hex()
      << "\nreduce_512 result: " << reduce_512_result.to_hex();
    
    // Also verify that the approximation "2^256 ≈ 4 (mod p)" is close
    FieldElement four(4);
    FieldElement diff1 = (two_to_256 < four) ? four - two_to_256 : two_to_256 - four;
    
    // The approximation should be reasonably close (within some tolerance)
    // For BN254, this should actually be exactly 4, but let's verify
    if (two_to_256 != four) {
      std::cout << "WARNING: 2^256 mod p = " << two_to_256.to_hex() 
                << " which is NOT exactly 4. The approximation error is " << diff1.to_hex() << std::endl;
    }
  }
  
  // Test case 2: Test multiple powers of 2^256 to verify linearity
  {
    for (uint64_t multiplier = 1; multiplier <= 10; ++multiplier) {
      // Test multiplier * 2^256 using reduce_512
      uint64_t product[8] = {0, 0, 0, 0, multiplier, 0, 0, 0};
      FieldElement reduce_result;
      FieldArithmetic::reduce_512(product, reduce_result);
      
      // Compute the expected result: multiplier * (2^256 mod p) mod p
      FieldElement two_to_256 = FieldConstants::ONE;
      for (int i = 0; i < 256; ++i) {
        two_to_256 = two_to_256 + two_to_256;
      }
      
      FieldElement expected = FieldConstants::ZERO;
      for (uint64_t i = 0; i < multiplier; ++i) {
        expected = expected + two_to_256;
      }
      
      EXPECT_EQ(reduce_result, expected) 
        << "reduce_512(" << multiplier << " * 2^256) failed"
        << "\nExpected: " << expected.to_hex()
        << "\nGot:      " << reduce_result.to_hex();
    }
  }
  
  // Test case 3: Test the shift-by-2 logic for potential overflow
  {
    // Test cases where the high part, when shifted left by 2, would overflow
    struct TestCase {
      uint64_t high_limbs[4];
      const char* description;
    };
    
    TestCase test_cases[] = {
      {{0x4000000000000000ULL, 0, 0, 0}, "High limb with bit 62 set"},
      {{0x8000000000000000ULL, 0, 0, 0}, "High limb with bit 63 set"},
      {{0xFFFFFFFFFFFFFFFFULL, 0, 0, 0}, "All ones in lowest high limb"},
      {{0, 0x4000000000000000ULL, 0, 0}, "Overflow potential in second limb"},
      {{0, 0, 0, 0x4000000000000000ULL}, "Overflow potential in highest limb"},
      {{0xC000000000000000ULL, 0xC000000000000000ULL, 0xC000000000000000ULL, 0xC000000000000000ULL}, "All limbs with potential overflow"},
    };
    
    for (const auto& test_case : test_cases) {
      uint64_t product[8] = {0, 0, 0, 0, 
                            test_case.high_limbs[0], 
                            test_case.high_limbs[1], 
                            test_case.high_limbs[2], 
                            test_case.high_limbs[3]};
      
      FieldElement result;
      FieldArithmetic::reduce_512(product, result);
      
      // The result should always be properly reduced (less than modulus)
      EXPECT_TRUE(result < FieldConstants::MODULUS) 
        << "Test case '" << test_case.description << "' resulted in unreduced value: " << result.to_hex();
      
      // Verify by manual computation of the shift
      FieldElement high(test_case.high_limbs[0], test_case.high_limbs[1], 
                       test_case.high_limbs[2], test_case.high_limbs[3]);
      
      // Manual shift by 2 with proper carry handling
      FieldElement manual_shifted;
      uint64_t carry = 0;
      for (int i = 0; i < 4; ++i) {
        uint64_t shifted = (high.limbs[i] << 2) | carry;
        manual_shifted.limbs[i] = shifted;
        carry = high.limbs[i] >> 62;
      }
      
      // If there's a final carry, the shifted value would overflow 256 bits
      // In such cases, we need to handle the overflow properly
      if (carry > 0) {
        // The overflowed bits represent additional multiples of 2^256
        // Each unit of carry represents 2^258 = 4 * 2^256
        FieldElement two_to_256 = FieldConstants::ONE;
        for (int i = 0; i < 256; ++i) {
          two_to_256 = two_to_256 + two_to_256;
        }
        
        FieldElement overflow_contribution = FieldConstants::ZERO;
        for (uint64_t i = 0; i < carry; ++i) {
          overflow_contribution = overflow_contribution + two_to_256;
          overflow_contribution = overflow_contribution + two_to_256;
          overflow_contribution = overflow_contribution + two_to_256;
          overflow_contribution = overflow_contribution + two_to_256; // Add 4 * 2^256
        }
        
        FieldElement expected_result = manual_shifted + overflow_contribution;
        EXPECT_EQ(result, expected_result) 
          << "Overflow handling incorrect for test case '" << test_case.description << "'"
          << "\nExpected: " << expected_result.to_hex()
          << "\nGot:      " << result.to_hex();
      }
    }
  }
  
  // Test case 4: Verify that the function handles the modulus correctly
  {
    // Test input that is exactly p (should reduce to 0)
    uint64_t product[8] = {
      FieldConstants::MODULUS.limbs[0],
      FieldConstants::MODULUS.limbs[1],
      FieldConstants::MODULUS.limbs[2],
      FieldConstants::MODULUS.limbs[3],
      0, 0, 0, 0
    };
    
    FieldElement result;
    FieldArithmetic::reduce_512(product, result);
    EXPECT_EQ(result, FieldConstants::ZERO) << "Input equal to modulus should reduce to zero";
    
    // Test input that is exactly 2*p (should also reduce to 0)
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
      __uint128_t doubled = (__uint128_t)FieldConstants::MODULUS.limbs[i] * 2 + carry;
      product[i] = (uint64_t)doubled;
      carry = (uint64_t)(doubled >> 64);
    }
    product[4] = carry;
    product[5] = product[6] = product[7] = 0;
    
    FieldArithmetic::reduce_512(product, result);
    EXPECT_EQ(result, FieldConstants::ZERO) << "Input equal to 2*modulus should reduce to zero";
  }
}
