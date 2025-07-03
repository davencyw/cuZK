#include "field_arithmetic.hpp"
#include <gtest/gtest.h>

using namespace Poseidon;

class FieldAccumulationTest : public ::testing::Test {
protected:
  void SetUp() override { FieldConstants::init(); }
};

TEST_F(FieldAccumulationTest, CoefficientAccumulation) {
  // Test the sequence: 1 - 2 - 3 + 4 = 0
  FieldElement result = FieldConstants::ZERO;

  // Add 1
  result += FieldElement(1);
  EXPECT_EQ(result.to_dec(), "1");

  // Subtract 2
  result += (FieldConstants::ZERO - FieldElement(2));
  EXPECT_EQ(result.to_dec(), "2188824287183927522224640574525727508854836440041"
                             "6034343698204186575808495616");

  // Subtract 3
  result += (FieldConstants::ZERO - FieldElement(3));
  EXPECT_EQ(result.to_dec(), "2188824287183927522224640574525727508854836440041"
                             "6034343698204186575808495613");

  // Add 4
  result += FieldElement(4);
  EXPECT_EQ(result.to_dec(), "0");

  // Final result should be zero
  EXPECT_TRUE(result.is_zero()) << "Result should be zero: " << result.to_dec();
}

TEST_F(FieldAccumulationTest, DirectComputation) {
  // Test direct computation
  FieldElement direct =
      FieldElement(1) - FieldElement(2) - FieldElement(3) + FieldElement(4);

  EXPECT_TRUE(direct.is_zero())
      << "Direct computation should be zero: " << direct.to_dec();
  EXPECT_EQ(direct.to_dec(), "0");
}