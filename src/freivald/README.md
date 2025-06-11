# Freivald's Algorithm - Matrix Multiplication Verification

This library implements matrix multiplication verification using Freivald's probabilistic algorithm. It leverages the Eigen library for efficient matrix operations and provides a configurable repetition factor to control the error probability.

## Overview

Freivald's algorithm is a probabilistic method for verifying matrix multiplication. Given matrices A, B, and C, it determines whether A×B = C with high probability and low computational cost.

### Algorithm Description

The algorithm works by:
1. Generating a random binary vector **r** (elements are 0 or 1)
2. Computing **y₁ = A×(B×r)** and **y₂ = C×r**
3. Checking if **y₁ = y₂**
4. If they're equal, the multiplication is likely correct
5. If they're different, the multiplication is definitely incorrect

The key insight is that if A×B ≠ C, then A×(B×r) ≠ C×r with probability ≥ 1/2 for a random vector r.

### Error Probability

- **Single round error probability**: ≤ 1/2
- **k rounds error probability**: ≤ (1/2)ᵏ
- For k=10 rounds: error probability ≤ 1/1024 ≈ 0.001
- For k=20 rounds: error probability ≤ 1/1,048,576 ≈ 10⁻⁶

## Components

### 1. MatrixGenerator
Generates random matrices and vectors for testing purposes.

**Features:**
- Templated matrix generation for any scalar type (int, float, double, etc.)
- Configurable value ranges
- Random binary vector generation
- Reproducible results with seed support
- Automatic type-appropriate distribution selection (integer vs. real)

### 2. Prover
Performs matrix multiplication using Eigen's optimized operations.

**Features:**
- Templated matrix-matrix multiplication for any scalar type
- Templated matrix-vector multiplication for any scalar type
- Dimension compatibility checking
- Automatic type deduction from input matrices
- Support for all arithmetic types (int, float, double, long, etc.)

### 3. Verifier
Implements Freivald's algorithm for matrix multiplication verification.

**Features:**
- Configurable repetition factor
- Single-round and multi-round verification
- Error probability calculations
- Templated support for all arithmetic types with automatic tolerance handling
- Type-appropriate comparison (exact for integers, tolerance-based for floating-point)
- Prover-delegated architecture for clean separation of concerns



## Usage

### Basic Example

```cpp
#include "matrix_generator.h"
#include "prover.h"
#include "verifier.h"
using namespace freivald;

// Create components with fixed seed for reproducibility
MatrixGenerator generator(12345);
Verifier verifier(12345);

// Generate random matrices and verify (integer matrices)
auto A = generator.generateRandomMatrix<int>(10, 8, -10, 10);
auto B = generator.generateRandomMatrix<int>(8, 12, -10, 10);
auto C = Prover::multiply(A, B);
bool result = verifier.verify(A, B, C, 15);
std::cout << "Verification: " << (result ? "PASSED" : "FAILED") << std::endl;

// Example with double precision
auto A_double = generator.generateRandomMatrix<double>(5, 5, -1.0, 1.0);
auto B_double = generator.generateRandomMatrix<double>(5, 5, -1.0, 1.0);
auto C_double = Prover::multiply(A_double, B_double);
bool result_double = verifier.verify(A_double, B_double, C_double, 10, 1e-10);
std::cout << "Double verification: " << (result_double ? "PASSED" : "FAILED") << std::endl;
```

### Using Individual Components

```cpp
// Generate matrices
MatrixGenerator generator(seed);
Eigen::MatrixXi A = generator.generateRandomMatrix<int>(3, 4, -10, 10);
Eigen::MatrixXi B = generator.generateRandomMatrix<int>(4, 5, -10, 10);

// Compute multiplication
Eigen::MatrixXi C = Prover::multiply(A, B);

// Verify result
Verifier verifier(seed);
bool is_correct = verifier.verify(A, B, C, 20); // 20 repetitions

// Example with different types
auto A_float = generator.generateRandomMatrix<float>(2, 3, -1.0f, 1.0f);
auto B_float = generator.generateRandomMatrix<float>(3, 2, -1.0f, 1.0f);
auto C_float = Prover::multiply(A_float, B_float);
bool is_correct_float = verifier.verify(A_float, B_float, C_float, 15, 1e-6f);
```

### Configurable Repetition Factor

```cpp
Verifier verifier;

// Different repetition counts for different confidence levels
verifier.verify(A, B, C, 1);   // 50% confidence
verifier.verify(A, B, C, 10);  // 99.9% confidence  
verifier.verify(A, B, C, 20);  // 99.9999% confidence

// Calculate required repetitions for desired error probability
int reps = Verifier::getRequiredRepetitions(0.001); // For 0.1% error probability
std::cout << "Need " << reps << " repetitions" << std::endl;
```

## Project Structure

```
# Project root
├── Makefile                    # Top-level convenience commands
└── src/freivald/
    ├── CMakeLists.txt          # Build configuration
    ├── README.md               # This documentation
    ├── example.cpp             # Usage examples and demos
    ├── matrix_generator.h/.cpp # Random matrix generation
    ├── prover.h                # Matrix multiplication (header-only)
    ├── verifier.h/.cpp        # Freivald's algorithm implementation
    └── test/                   # Test suite directory
        ├── test_freivald.cpp           # Comprehensive test suite (25 tests)
        └── benchmark_verification.cpp # Performance benchmark suite
```

## Building

### Prerequisites
- C++17 compatible compiler
- CMake 3.14 or later
- Eigen3 (automatically downloaded if not found)

### Build Instructions

```bash
# From the project root
mkdir build && cd build
cmake ..
make freivald_tests     # Build tests
make freivald_example   # Build example
make freivald_benchmark # Build performance benchmark

# Run tests
./src/freivald/freivald_tests

# Run example
./src/freivald/freivald_example

# Run performance benchmark
./src/freivald/freivald_benchmark
```

### Quick Commands (from project root)

For convenience, a top-level Makefile is provided that automatically handles build directory navigation:

```bash
# Show all available commands
make help

# Build and run everything (recommended)
make all-demo          # Runs tests, example, and benchmark

# Individual commands  
make freivald-tests     # Build and run tests
make freivald-example   # Build and run example  
make freivald-benchmark # Build and run performance benchmark


# Setup and cleanup
make setup             # Create and configure build directory
make clean             # Clean build directory
```

**Note**: The top-level Makefile automatically handles the `build/` directory navigation for you. You can also run CMake commands directly from the `build/` directory if preferred.

### Integration with Main Project

The matrix multiplication verification is integrated into the main project's test suite:

```bash
# Run all tests including matrix multiplication verification
make run_all_tests

# Run just CPU tests (including this module)
make run_cpu_tests

# Run using CTest
ctest -R freivald_tests
```

## Testing

The project includes comprehensive tests covering:

### Functionality Tests
- Matrix and vector generation
- Matrix multiplication correctness
- Freivald's algorithm verification
- Error detection capabilities
- Double precision support

### Edge Cases
- Invalid dimensions
- Invalid repetition counts
- Matrix incompatibility
- Error handling

### Performance Tests
- Large matrix verification
- Stress testing with various sizes
- Comprehensive benchmarking suite comparing Freivald's vs. recomputation
- Timing analysis across different matrix sizes and scalar types

### Probabilistic Behavior
- Single round verification statistics
- Error probability validation
- False positive/negative rates

## Performance Characteristics

### Time Complexity
- **Standard verification** (recomputing A×B): O(mnp) for A(m×n) × B(n×p)
- **Freivald's algorithm**: O(k(mn + np + mp)) for k repetitions

### Space Complexity
- O(mn + np + mp) for matrices plus O(p) for random vectors

### Benchmark Methodology Note
**Important:** The benchmark correctly measures:
- **Freivald verification**: Performs k probabilistic rounds as required by the algorithm
- **Recomputation verification**: Performs matrix multiplication once per verification (realistic approach)

Previous versions incorrectly repeated recomputation k times, artificially inflating speedup results.


### Benchmark Results (Corrected)
Our comprehensive benchmark suite tests various matrix sizes and types. **Previous results were artificially inflated due to incorrect recomputation methodology.** The corrected benchmark shows realistic performance comparisons:

**Sample Results (Corrected Benchmark):**
```
------------------------------------------------------------------------------------------
|   Type |   Size | Reps | Freivald (ms) | Recompute (ms) |    Speedup | Status |
------------------------------------------------------------------------------------------
|    int |     50 |   20 |        0.031 |        0.011 |       0.35x |     OK |
|    int |    100 |   20 |        0.078 |        0.029 |       0.37x |     OK |
|    int |    200 |   20 |        0.132 |        0.192 |       1.45x |     OK |
|    int |    500 |   20 |        0.617 |        2.834 |       4.59x |     OK |
|    int |   1000 |   20 |        2.791 |       22.770 |       8.16x |     OK |
|    int |   1500 |   20 |        9.203 |       76.752 |       8.34x |     OK |
|    int |   2000 |   20 |       19.068 |      176.228 |       9.24x |     OK |
|    int |   3000 |   20 |       47.872 |      595.165 |      12.43x |     OK |
|    int |   4000 |   20 |       71.654 |     1389.431 |      19.39x |     OK |
|    int |   5000 |   20 |       97.098 |     2733.320 |      28.15x |     OK |
| double |     50 |   20 |        0.017 |        0.006 |       0.35x |     OK |
| double |    100 |   20 |        0.067 |        0.052 |       0.78x |     OK |
| double |    200 |   20 |        0.233 |        0.403 |       1.73x |     OK |
| double |    500 |   20 |        1.301 |        5.788 |       4.45x |     OK |
| double |   1000 |   20 |        6.733 |       45.816 |       6.80x |     OK |
| double |   1500 |   20 |       21.724 |      154.067 |       7.09x |     OK |
| double |   2000 |   20 |       33.355 |      360.281 |      10.80x |     OK |
| double |   3000 |   20 |       67.134 |     1241.259 |      18.49x |     OK |
| double |   4000 |   20 |      149.386 |     2863.544 |      19.17x |     OK |
| double |   5000 |   20 |      216.358 |     5651.482 |      26.12x |     OK |
|  float |     50 |   20 |        0.011 |        0.003 |       0.29x |     OK |
|  float |    100 |   20 |        0.035 |        0.024 |       0.69x |     OK |
|  float |    200 |   20 |        0.133 |        0.187 |       1.40x |     OK |
|  float |    500 |   20 |        0.635 |        2.959 |       4.66x |     OK |
|  float |   1000 |   20 |        2.921 |       23.224 |       7.95x |     OK |
|  float |   1500 |   20 |        8.718 |       79.740 |       9.15x |     OK |
|  float |   2000 |   20 |       19.636 |      183.639 |       9.35x |     OK |
|  float |   3000 |   20 |       51.000 |      628.961 |      12.33x |     OK |
|  float |   4000 |   20 |       69.280 |     1449.506 |      20.92x |     OK |
|  float |   5000 |   20 |       94.300 |     2819.081 |      29.89x |     OK |
------------------------------------------------------------------------------------------
```


Run `make freivald-benchmark` from the root of this repo for complete performance analysis on your hardware.

## Error Probability Analysis

| Repetitions | Error Probability | Confidence Level |
|-------------|-------------------|------------------|
| 1           | ≤ 50%            | 50%              |
| 5           | ≤ 3.125%         | 96.875%          |
| 10          | ≤ 0.098%         | 99.902%          |
| 15          | ≤ 0.003%         | 99.997%          |
| 20          | ≤ 0.0001%        | 99.9999%         |

## Architecture Design

The system uses a clean prover-delegated architecture:

### Prover-Delegated Design
- **Verifier** generates random vector r and orchestrates verification
- **Verifier** delegates A×(B×r) computation to prover via `computeABr(A, B, r)`
- **Verifier** computes C×r using prover's multiply function
- **Clean separation of concerns**: verifier focuses on verification logic, prover handles matrix operations
- **Modular and testable**: each component can be tested independently
- **Extensible**: easy to modify matrix operations without affecting verification logic

## Advantages of Freivald's Algorithm

1. **Speed**: Faster than recomputing A×B for verification
2. **Simplicity**: Easy to implement and understand
3. **Scalability**: Performance advantage increases with matrix size
4. **Flexibility**: Configurable confidence level
5. **Memory Efficient**: Uses O(n) additional space
6. **Modular Design**: Clean separation between verification logic and matrix operations

## Limitations

1. **Probabilistic**: Small chance of false positives
2. **No Error Location**: Doesn't identify where errors occur
3. **Binary Result**: Only yes/no answer, no error magnitude

## API Reference

### MatrixGenerator
```cpp
MatrixGenerator(seed)                                    // Constructor
generateRandomMatrix<T>(rows, cols, min_val, max_val)   // Generate matrix of type T
generateRandomVector<T>(size, min_val, max_val)         // Generate vector of type T
generateRandomBinaryVector(size)                        // Generate binary vector (always int)
```

**Supported Types**: `int`, `float`, `double`, `long`, and other arithmetic types

### Prover
```cpp
static multiply(A, B)                    // Matrix multiplication
static multiply(A, x)                    // Matrix-vector multiplication
static computeABr(A, B, r)              // Compute A×(B×r) for Freivald's algorithm
static areCompatible(A, B)              // Check dimension compatibility
static areCompatibleMV(A, x)            // Check matrix-vector compatibility
```

### Verifier
```cpp
Verifier(seed)                                    // Constructor
verify(A, B, C, repetitions, tolerance=1e-10)    // Verify A×B=C using Freivald's algorithm
verifySingleRound(A, B, C, tolerance=1e-10)      // Single verification round
static getErrorProbability(repetitions)          // Calculate error probability
static getRequiredRepetitions(desired_error_prob) // Calculate needed repetitions
```