# cuZK - Various Zero-Knowledge Relevant Implementations in C++ and CUDA

An educational C++ and CUDA implementation of zero-knowledge cryptography primitives, designed for learning and understanding ZK-proof systems.

## ⚠️ Security Warning

**This implementation is for educational purposes only and should not be used in production environments.**

For example, the MDS matrix and round constants generation must be replaced with secure, spec-compliant algorithms if this is meant for cryptographic purposes.

## ⚠️ Disclaimer

**The author does not guarantee the correctness of any implemented algorithms.** This code is provided as-is for educational and research purposes. If you discover any errors, inconsistencies, you can report them by opening an issue on the project repository.

## 🚀 Performance Highlights

**GPU acceleration delivers game-changing performance:**
- **Merkle Tree Building**: Up to **45.78x faster** on GPU (50K elements in 282ms vs 12.9s on CPU)
- **Batch Proof Verification**: Up to **87.26x speedup** (5K proofs in 14.8ms vs 1.52s on CPU)

## Projects

### Poseidon Hash
- Complete implementation of the Poseidon hash function
- Optimized for BN254 scalar field arithmetic
- Support for single element, pair, and multiple element hashing
- Sponge construction for variable-length inputs
- CUDA-accelerated batch hashing for high-throughput applications
- Comprehensive test suite with performance benchmarks

### N-ary Merkle Tree
- Generic n-ary merkle tree implementation with configurable arity (2-8)
- Integration with Poseidon hash function for cryptographic security
- Efficient proof generation and verification
- Support for batch operations and multiple proof types
- **CUDA-accelerated tree building and batch proof verification**
- Comprehensive test suite with performance benchmarks for different arities
- Optimized bottom-up tree construction for consistent structure

## Quick Start

### Prerequisites
- CMake 3.14 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- Git (for Google Test download)
- CUDA Toolkit 11.0+ (optional, for GPU acceleration)

### Building

Using the Makefile (recommended):
```bash
# Build everything
make build

# Debug build
make debug
```

### Running Tests

```bash
# Run all tests
make test

# Run tests with verbose output
make test-verbose

# Run CUDA tests (if CUDA is available)
make test-cuda

# Run benchmark tests
make benchmark

# Run benchmarks
./run_poseidon_benchmark.sh
./run_merkle_benchmarks.sh
```


## Code Quality

```bash
# Format code (requires clang-format)
make format

# Lint code (requires cppcheck)
make lint
```

## Technical Details

### Poseidon Parameters
- State size: 3 elements (t=3)
- Capacity: 1 element (c=1)
- Rate: 2 elements (r=2)
- Full rounds: 8 (R_F=8)
- Partial rounds: 56 (R_P=56)
- S-box: x^5
- Field: BN254 scalar field

### N-ary Merkle Tree Features
- **Configurable Arity**: Support for 2-ary (binary) through 8-ary trees
- **Bottom-up Construction**: Efficient tree building with consistent structure
- **Poseidon Integration**: Uses Poseidon hash for cryptographic security
- **Proof Generation**: Direct indexing approach for reliable proof construction
- **Batch Operations**: Support for multiple proof generation and verification
- **GPU Acceleration**: CUDA-powered tree building and batch verification for large datasets
- **Performance Optimization**: Benchmarks show optimal arity selection guidelines:
  - Binary trees: Best for small datasets (< 1000 leaves)
  - 4-ary trees: Optimal balance for most use cases
  - 8-ary trees: Best for very large datasets (> 100,000 leaves)

### Field Arithmetic
- Prime: 21888242871839275222246405745257275088548364400416034343698204186575808495617
- Representation: 4 × 64-bit limbs (256 bits)
- Reduction: Simple comparison-based reduction
- Multiplication: Schoolbook multiplication with reduction

### CUDA Acceleration
- **Batch Processing**: Optimized for processing thousands of elements simultaneously
- **Kernel Optimization**: Custom CUDA kernels for field arithmetic and hashing operations

Example poseidon benchmark on a A100:
```bash

Configuration: Large Scale (Batch Size: 4096, Total Hashes: 1000000)
========================================================================================================================

Pair Hash Results:
------------------------------------------------------------------------------------------------------------------------
Implementation       |       Time (ns) |          Hash/s | Total Time (ms) |  Speedup vs CPU |
------------------------------------------------------------------------------------------------------------------------
CPU                  |        68101.40 |           14683 |        68101.40 |             1.0x |
CUDA                 |          466.19 |         2145027 |          466.19 |           146.1x |
------------------------------------------------------------------------------------------------------------------------

Single Hash Results:
------------------------------------------------------------------------------------------------------------------------
Implementation       |       Time (ns) |          Hash/s | Total Time (ms) |  Speedup vs CPU |
------------------------------------------------------------------------------------------------------------------------
CPU                  |        67834.45 |           14741 |        67834.45 |             1.0x |
CUDA                 |          570.91 |         1751596 |          570.91 |           118.8x |
------------------------------------------------------------------------------------------------------------------------
```


## References
- [Poseidon: A New Hash Function for Zero-Knowledge Proof Systems](https://eprint.iacr.org/2019/458.pdf)
- [BN254 Curve Specification](https://hackmd.io/@jpw/bn254)
