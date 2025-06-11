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

**Freivald's algorithm shows significant algorithmic advantages:**
- **Matrix Verification**: Up to **76x faster** than recomputation (1000×1000 matrices: 8.6ms vs 651ms)
- **Scalable Verification**: O(n²) complexity vs O(n³) for traditional verification
- **High Confidence**: Configurable error probability (e.g., ≤ 0.1% with 10 repetitions)

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

### Freivald's Algorithm (Matrix Multiplication Verification)
- Implementation of Freivald's algorithm for efficient matrix multiplication verification
- Probabilistic verification with configurable error bounds
- O(n²) time complexity vs O(n³) for recomputation
- Fully templated design supporting integer and floating-point types
- Comprehensive test suite with 25 test cases
- Performance benchmarks showing 6-76x speedup for practical matrix sizes
- Modular architecture with separate matrix generator, prover, and verifier components

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

# Run benchmark tests (includes merkle tree benchmarks)
make benchmark
```

### Freivald's Algorithm

```bash
# Run Freivald's algorithm tests
make freivald-tests

# Run Freivald's algorithm examples
make freivald-example  

# Run Freivald's algorithm performance benchmark
make freivald-benchmark
```

## Usage Examples

### GPU-Accelerated Poseidon Hashing

```cpp
#include "poseidon.hpp"
#include "poseidon_cuda.cuh"
using namespace Poseidon;
using namespace Poseidon::PoseidonCUDA;

int main() {
    // Initialize CUDA Poseidon
    if (!CudaPoseidonHash::initialize()) {
        std::cerr << "Failed to initialize CUDA Poseidon" << std::endl;
        return 1;
    }
    
    // Prepare batch data
    std::vector<FieldElement> inputs;
    for (size_t i = 0; i < 10000; ++i) {
        inputs.push_back(FieldElement(i));
    }
    
    // Batch hash on GPU
    std::vector<FieldElement> results;
    bool success = CudaPoseidonHash::batch_hash_single(inputs, results);
    
    if (success) {
        std::cout << "Hashed " << inputs.size() << " elements on GPU" << std::endl;
        std::cout << "First result: " << results[0].to_hex() << std::endl;
    }
    
    // Cleanup
    CudaPoseidonHash::cleanup();
    return 0;
}
```

### GPU-Accelerated Merkle Tree Operations

```cpp
#include "merkle_tree.hpp"
#include "merkle_tree_cuda.cuh"
using namespace MerkleTree;
using namespace MerkleTree::MerkleTreeCUDA;

int main() {
    // Initialize CUDA
    if (!CudaNaryMerkleTree::initialize_cuda()) {
        std::cerr << "Failed to initialize CUDA" << std::endl;
        return 1;
    }
    
    // Initialize Poseidon constants
    Poseidon::PoseidonConstants::init();
    
    // Generate large dataset for GPU acceleration benefits
    std::vector<FieldElement> leaves;
    for (size_t i = 0; i < 10000; ++i) {
        leaves.push_back(FieldElement(i + 1));
    }
    
    // Create CUDA-accelerated tree
    MerkleTreeConfig config(4); // Quaternary tree
    CudaNaryMerkleTree cuda_tree(leaves, config);
    
    std::cout << "Built tree with " << cuda_tree.get_leaf_count() << " leaves" << std::endl;
    std::cout << "Tree height: " << cuda_tree.get_tree_height() << std::endl;
    std::cout << "Root hash: " << cuda_tree.get_root_hash().to_hex().substr(0, 16) << "..." << std::endl;
    
    // Generate batch proofs (GPU-accelerated for large batches)
    std::vector<size_t> proof_indices;
    for (size_t i = 0; i < 1000; i += 10) {
        proof_indices.push_back(i);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto proofs = cuda_tree.generate_batch_proofs(proof_indices);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Generated " << proofs.size() << " proofs in " << duration.count() << "ms" << std::endl;
    
    // Verify batch proofs
    std::vector<FieldElement> proof_values;
    for (size_t idx : proof_indices) {
        proof_values.push_back(leaves[idx]);
    }
    
    start = std::chrono::high_resolution_clock::now();
    bool batch_valid = cuda_tree.verify_batch_proofs(proofs, proof_values);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Verified " << proofs.size() << " proofs in " << duration.count() << "ms" << std::endl;
    std::cout << "Batch verification result: " << (batch_valid ? "Valid" : "Invalid") << std::endl;
    
    // Compare with CPU implementation
    NaryMerkleTree cpu_tree(leaves, config);
    bool trees_match = cuda_tree.compare_with_cpu_tree(cpu_tree);
    std::cout << "GPU tree matches CPU: " << (trees_match ? "Yes" : "No") << std::endl;
    
    // Cleanup
    CudaNaryMerkleTree::cleanup_cuda();
    return 0;
}
```

### Freivald's Algorithm (Matrix Multiplication Verification)

```cpp
#include "matrix_generator.h"
#include "prover.h"
#include "verifier.h"
using namespace freivald;

int main() {
    // Generate random matrices for testing
    MatrixGenerator<int> generator;
    
    auto A = generator.generateMatrix(100, 80, 42);  // 100x80 matrix, seed=42
    auto B = generator.generateMatrix(80, 120, 43);  // 80x120 matrix, seed=43
    
    // Prover computes matrix multiplication A × B = C
    Prover<int> prover;
    auto C = prover.computeAB(A, B);
    
    std::cout << "Computed matrix multiplication: " 
              << A.rows() << "×" << A.cols() << " × " 
              << B.rows() << "×" << B.cols() << " = " 
              << C.rows() << "×" << C.cols() << std::endl;
    
    // Verifier checks if A × B = C using Freivald's algorithm
    Verifier<int> verifier;
    
    // Verify with 10 repetitions for high confidence (error probability ≤ (1/2)^10)
    bool is_correct = verifier.verify(A, B, C, prover, 10);
    
    std::cout << "Verification result: " << (is_correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Error probability: ≤ " << verifier.getErrorProbability(10) << std::endl;
    
    // Demonstrate incorrect multiplication detection
    auto C_wrong = C;
    if (C_wrong.rows() > 0 && C_wrong.cols() > 0) {
        C_wrong(0, 0) += 1;  // Introduce error
    }
    
    bool is_wrong_detected = !verifier.verify(A, B, C_wrong, prover, 10);
    std::cout << "Error detection works: " << (is_wrong_detected ? "YES" : "NO") << std::endl;
    
    return 0;
}
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

Example benchmark:
```bash
====================================================================================================
  CUDA SCALABILITY ANALYSIS
====================================================================================================
Dataset Size | Build Time (ms) | Trees/sec | Speedup vs CPU
-----------------------------------------------------------------
         100 |           77.51 |       129 |          0.55x
         500 |          106.28 |        94 |          1.66x
        1000 |          110.72 |        90 |          2.02x
        5000 |          194.30 |        51 |         12.81x
       10000 |          200.09 |        49 |         14.89x
       50000 |          282.57 |        35 |         45.78x
====================================================================================================

====================================================================================================
  BATCH PROOF PERFORMANCE: CPU vs GPU
====================================================================================================
Batch Size | CPU Proof Gen (ms) | GPU Proof Gen (ms) | CPU Verify (ms) | GPU Verify (ms) | Speedup
----------------------------------------------------------------------------------------------------
        10 |               0.01 |               0.01 |            3.11 |            3.00 |    1.04x
        50 |               0.04 |               0.03 |           15.08 |            8.06 |    1.87x
       100 |               0.09 |               0.05 |           30.34 |            8.17 |    3.70x
       500 |               0.41 |               0.25 |          151.62 |           11.60 |   12.82x
      1000 |               0.82 |               0.50 |          303.94 |           12.05 |   24.29x
      5000 |               4.15 |               2.62 |         1519.26 |           14.84 |   87.26x
====================================================================================================
```

## Performance Benchmarks

The implementation includes comprehensive benchmarking tools:

```bash
# Run CPU benchmarks
make benchmark

# Run CUDA benchmarks (if available)
make test-cuda

# Run specific benchmark script
./run_merkle_benchmarks.sh
```

## References
- [Poseidon: A New Hash Function for Zero-Knowledge Proof Systems](https://eprint.iacr.org/2019/458.pdf)
- [BN254 Curve Specification](https://hackmd.io/@jpw/bn254)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
