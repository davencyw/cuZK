#!/bin/bash

# Script to build and run Merkle Tree CPU vs GPU benchmarks
# Usage: ./run_merkle_benchmarks.sh [quick|full|cpu-only]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default mode
MODE=${1:-quick}

print_status "Starting Merkle Tree Benchmark Suite"
print_status "Mode: $MODE"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    print_status "Creating build directory..."
    mkdir build
fi

cd build

# Configure and build
print_status "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

print_status "Building the project..."
make -j$(nproc) 2>/dev/null || make -j4

# Check if CUDA is available
CUDA_AVAILABLE=false
if [ -f "src/merkle_tree/merkle_tree_cuda_benchmark_tests" ]; then
    CUDA_AVAILABLE=true
    print_success "CUDA support detected"
else
    print_warning "CUDA support not detected - will run CPU-only benchmarks"
fi

echo ""
echo "=========================================="
echo "       MERKLE TREE BENCHMARK SUITE       "
echo "=========================================="
echo ""

case $MODE in
    "quick")
        print_status "Running quick performance check..."
        echo ""
        
        # Run CPU benchmarks
        print_status "=== CPU BENCHMARKS ==="
        ./src/merkle_tree/merkle_tree_tests --gtest_filter="*Benchmark*OptimalArityAnalysis*" || true
        
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo ""
            print_status "=== GPU BENCHMARKS ==="
            ./src/merkle_tree/merkle_tree_cuda_benchmark_tests --gtest_filter="*QuickPerformanceCheck*" || true
        fi
        ;;
        
    "full")
        print_status "Running comprehensive benchmarks..."
        echo ""
        
        # Run all CPU benchmarks
        print_status "=== COMPREHENSIVE CPU BENCHMARKS ==="
        ./src/merkle_tree/merkle_tree_tests --gtest_filter="*Benchmark*" || true
        
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo ""
            print_status "=== COMPREHENSIVE GPU BENCHMARKS ==="
            ./src/merkle_tree/merkle_tree_cuda_benchmark_tests || true
        fi
        ;;
        
    "cpu-only")
        print_status "Running CPU-only benchmarks..."
        echo ""
        
        print_status "=== CPU BENCHMARKS ==="
        ./src/merkle_tree/merkle_tree_tests --gtest_filter="*Benchmark*" || true
        ;;
        
    "gpu-only")
        if [ "$CUDA_AVAILABLE" = true ]; then
            print_status "Running GPU-only benchmarks..."
            echo ""
            
            print_status "=== GPU BENCHMARKS ==="
            ./src/merkle_tree/merkle_tree_cuda_benchmark_tests || true
        else
            print_error "CUDA not available - cannot run GPU-only benchmarks"
            exit 1
        fi
        ;;
        
    *)
        print_error "Unknown mode: $MODE"
        echo "Available modes:"
        echo "  quick    - Run quick performance check"
        echo "  full     - Run comprehensive benchmarks"
        echo "  cpu-only - Run only CPU benchmarks"
        echo "  gpu-only - Run only GPU benchmarks"
        exit 1
        ;;
esac

echo ""
print_success "Benchmark suite completed!"

if [ "$CUDA_AVAILABLE" = true ]; then
    echo ""
    print_status "You can also run specific benchmark categories:"
    echo "  CPU vs GPU comparison: ./src/merkle_tree/merkle_tree_cuda_benchmark_tests --gtest_filter='*ComprehensiveCPUvsGPUComparison*'"
    echo "  Scalability analysis:  ./src/merkle_tree/merkle_tree_cuda_benchmark_tests --gtest_filter='*ScalabilityAnalysis*'"
    echo "  Batch proof performance: ./src/merkle_tree/merkle_tree_cuda_benchmark_tests --gtest_filter='*BatchProofPerformanceComparison*'"
    echo "  Optimal configuration: ./src/merkle_tree/merkle_tree_cuda_benchmark_tests --gtest_filter='*OptimalConfigurationAnalysis*'"
fi 