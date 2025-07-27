#!/bin/bash

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
MODE=${1:-full}

print_status "Starting Poseidon Hash Benchmark Suite"
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
if [ -f "src/poseidon/poseidon_cuda_tests" ]; then
    CUDA_AVAILABLE=true
    print_success "CUDA support detected"
else
    print_warning "CUDA support not detected - will run CPU-only benchmarks"
fi

echo ""
echo "=========================================="
echo "       POSEIDON HASH BENCHMARK SUITE      "
echo "=========================================="
echo ""

case $MODE in
    "full")
        print_status "Running comprehensive benchmarks..."
        echo ""
        
        # Run all CPU benchmarks
        print_status "=== COMPREHENSIVE CPU TESTS ==="
        ./src/poseidon/poseidon_tests --gtest_filter="*Benchmark*" || true
        
        if [ "$CUDA_AVAILABLE" = true ]; then
            echo ""
            print_status "=== COMPREHENSIVE GPU TESTS ==="
            ./src/poseidon/poseidon_cuda_tests --gtest_filter="*Performance*" || true
        fi
        
        echo ""
        print_status "=== COMPREHENSIVE PERFORMANCE BENCHMARK ==="
        ./src/poseidon/poseidon_benchmark || true
        ;;
    *)
        print_error "Unknown mode: $MODE"
        echo "Available modes:"
        echo "  full     - Run comprehensive benchmarks"
        exit 1
        ;;
esac

echo ""
print_success "Benchmark suite completed!"


