#!/bin/bash

# Simple Poseidon CUDA Performance Benchmark Script
# Measures kernel performance without requiring external profiling tools

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_BATCH_SIZE=8192
DEFAULT_ITERATIONS=100
DEFAULT_KERNEL_TYPE="both"
OUTPUT_DIR="benchmark_results"
BUILD_DIR="build"

# Check prerequisites
check_prerequisites() {
    # Check CUDA device availability
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: No NVIDIA GPU detected or nvidia-smi not available.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ CUDA GPU detected${NC}"
    
    # Show GPU info
    local gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1)
    echo -e "${BLUE}GPU: $gpu_info${NC}"
}

# Build the project if needed
build_project() {
    echo -e "${BLUE}Building project...${NC}"
    
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        cmake ..
    else
        cd "$BUILD_DIR"
    fi
    
    make poseidon_cuda_profiler -j$(nproc)
    
    if [ ! -f "src/poseidon/poseidon_cuda_profiler" ]; then
        echo -e "${RED}Error: Failed to build poseidon_cuda_profiler executable${NC}"
        exit 1
    fi
    
    cd ..
    echo -e "${GREEN}✓ Build completed${NC}"
}

# Create output directory
setup_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    echo -e "${GREEN}✓ Output directory created: $OUTPUT_DIR${NC}"
}

# Run a single benchmark configuration
run_benchmark() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local description="$4"
    
    echo -e "${YELLOW}Running benchmark: $description${NC}"
    echo -e "  Batch size: $batch_size, Iterations: $iterations, Kernel type: $kernel_type"
    
    # Run the profiler and capture output
    local output=$("$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type" 2>&1)
    
    # Extract metrics from output
    local total_hashes=$(echo "$output" | grep "Total hashes:" | awk '{print $3}')
    local total_time=$(echo "$output" | grep "Total time:" | awk '{print $3}')
    local avg_time_per_hash=$(echo "$output" | grep "Average time per hash:" | awk '{print $5}')
    local hashes_per_second=$(echo "$output" | grep "Hashes per second:" | awk '{print $4}')
    
    # Calculate throughput
    local throughput_mhash_s=$(echo "scale=2; $hashes_per_second / 1000000" | bc -l 2>/dev/null || echo "N/A")
    
    # Store results
    echo "$batch_size,$iterations,$kernel_type,$total_hashes,$total_time,$avg_time_per_hash,$hashes_per_second,$throughput_mhash_s" >> "$OUTPUT_DIR/benchmark_results.csv"
    
    # Display results
    printf "  %-20s: %s\n" "Total hashes" "$total_hashes"
    printf "  %-20s: %s ms\n" "Total time" "$total_time"
    printf "  %-20s: %s ns\n" "Avg time per hash" "$avg_time_per_hash"
    printf "  %-20s: %s\n" "Hashes per second" "$hashes_per_second"
    printf "  %-20s: %s MHash/s\n" "Throughput" "$throughput_mhash_s"
    echo ""
}

# Run scaling analysis (different batch sizes)
run_scaling_analysis() {
    local iterations=$1
    local kernel_type=$2
    
    echo -e "${BLUE}Running scaling analysis...${NC}"
    
    local batch_sizes=(1024 2048 4096 8192 16384 32768 65536)
    
    for batch_size in "${batch_sizes[@]}"; do
        # Adjust iterations for larger batch sizes to keep runtime reasonable
        local adjusted_iterations=$iterations
        if [ "$batch_size" -gt 16384 ]; then
            adjusted_iterations=$((iterations / 2))
        fi
        if [ "$batch_size" -gt 32768 ]; then
            adjusted_iterations=$((iterations / 4))
        fi
        
        run_benchmark "$batch_size" "$adjusted_iterations" "$kernel_type" "Scaling test (batch_size=$batch_size)"
    done
}

# Run comprehensive benchmarks
run_comprehensive_benchmarks() {
    echo -e "${BLUE}Running comprehensive benchmarks...${NC}"
    
    # Initialize CSV file
    echo "batch_size,iterations,kernel_type,total_hashes,total_time_ms,avg_time_per_hash_ns,hashes_per_second,throughput_mhash_s" > "$OUTPUT_DIR/benchmark_results.csv"
    
    # Test different configurations
    echo -e "${YELLOW}=== Single Hash Kernel Tests ===${NC}"
    run_scaling_analysis 100 "single"
    
    echo -e "${YELLOW}=== Pairs Hash Kernel Tests ===${NC}"
    run_scaling_analysis 100 "pairs"
    
    # Performance comparison at optimal batch size
    echo -e "${YELLOW}=== Kernel Comparison ===${NC}"
    local optimal_batch=8192
    run_benchmark "$optimal_batch" 200 "single" "Single kernel comparison"
    run_benchmark "$optimal_batch" 200 "pairs" "Pairs kernel comparison"
}

# Generate performance analysis report
generate_analysis_report() {
    local report_file="$OUTPUT_DIR/performance_analysis.txt"
    
    echo -e "${BLUE}Generating performance analysis report...${NC}"
    
    cat > "$report_file" << EOF
Poseidon CUDA Kernel Performance Analysis
========================================

Generated on: $(date)
GPU Information: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1)

Benchmark Configuration:
- Batch sizes tested: 1024, 2048, 4096, 8192, 16384, 32768, 65536
- Kernel types: single hash, pairs hash
- Multiple iterations for statistical accuracy

Key Performance Metrics:
EOF

    # Find best performing configurations
    if [ -f "$OUTPUT_DIR/benchmark_results.csv" ]; then
        echo "" >> "$report_file"
        echo "Top Performing Configurations:" >> "$report_file"
        echo "-----------------------------" >> "$report_file"
        
        # Best single hash throughput
        local best_single=$(grep ",single," "$OUTPUT_DIR/benchmark_results.csv" | sort -t',' -k8 -nr | head -1)
        if [ -n "$best_single" ]; then
            local batch_size=$(echo "$best_single" | cut -d',' -f1)
            local throughput=$(echo "$best_single" | cut -d',' -f8)
            echo "Best Single Hash: $throughput MHash/s (batch_size=$batch_size)" >> "$report_file"
        fi
        
        # Best pairs hash throughput
        local best_pairs=$(grep ",pairs," "$OUTPUT_DIR/benchmark_results.csv" | sort -t',' -k8 -nr | head -1)
        if [ -n "$best_pairs" ]; then
            local batch_size=$(echo "$best_pairs" | cut -d',' -f1)
            local throughput=$(echo "$best_pairs" | cut -d',' -f8)
            echo "Best Pairs Hash: $throughput MHash/s (batch_size=$batch_size)" >> "$report_file"
        fi
        
        # Add scaling analysis
        echo "" >> "$report_file"
        echo "Scaling Analysis:" >> "$report_file"
        echo "----------------" >> "$report_file"
        echo "Batch Size | Single (MHash/s) | Pairs (MHash/s)" >> "$report_file"
        echo "-----------|------------------|----------------" >> "$report_file"
        
        local batch_sizes=(1024 2048 4096 8192 16384 32768 65536)
        for batch_size in "${batch_sizes[@]}"; do
            local single_perf=$(grep "^$batch_size,.*,single," "$OUTPUT_DIR/benchmark_results.csv" | cut -d',' -f8 | head -1)
            local pairs_perf=$(grep "^$batch_size,.*,pairs," "$OUTPUT_DIR/benchmark_results.csv" | cut -d',' -f8 | head -1)
            printf "%10s | %15s | %15s\n" "$batch_size" "${single_perf:-N/A}" "${pairs_perf:-N/A}" >> "$report_file"
        done
    fi
    
    cat >> "$report_file" << EOF

Optimization Recommendations:
----------------------------
1. Use batch sizes >= 8192 for optimal throughput
2. For maximum performance, use the largest batch size your memory allows
3. Consider the trade-off between latency and throughput
4. Monitor GPU utilization - low utilization suggests room for optimization

Files Generated:
---------------
- benchmark_results.csv: Raw performance data
- performance_analysis.txt: This analysis report

For detailed kernel-level profiling, use:
./profile_poseidon_nsys.sh (requires NVIDIA Nsight Systems)
EOF
    
    echo -e "${GREEN}✓ Analysis report generated: $report_file${NC}"
}

# Generate simple plots if gnuplot is available
generate_plots() {
    if command -v gnuplot &> /dev/null; then
        echo -e "${BLUE}Generating performance plots...${NC}"
        
        # Create gnuplot script for throughput scaling
        cat > "$OUTPUT_DIR/plot_scaling.gp" << 'EOF'
set terminal png size 1200,800
set output 'scaling_performance.png'
set title 'Poseidon CUDA Kernel Scaling Performance'
set xlabel 'Batch Size'
set ylabel 'Throughput (MHash/s)'
set logscale x
set grid
set key top left

plot 'benchmark_results.csv' using 1:($3=="single" ? $8 : 1/0) with linespoints title 'Single Hash' linewidth 2, \
     'benchmark_results.csv' using 1:($3=="pairs" ? $8 : 1/0) with linespoints title 'Pairs Hash' linewidth 2
EOF
        
        cd "$OUTPUT_DIR"
        gnuplot plot_scaling.gp
        cd ..
        
        echo -e "${GREEN}✓ Performance plot generated: $OUTPUT_DIR/scaling_performance.png${NC}"
    else
        echo -e "${YELLOW}gnuplot not available - skipping plot generation${NC}"
    fi
}

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                    Show this help message"
    echo "  -b, --batch-size SIZE         Set batch size for single test (default: $DEFAULT_BATCH_SIZE)"
    echo "  -i, --iterations COUNT        Set iteration count (default: $DEFAULT_ITERATIONS)"
    echo "  -k, --kernel-type TYPE        Set kernel type: single, pairs, both (default: $DEFAULT_KERNEL_TYPE)"
    echo "  -o, --output-dir DIR          Set output directory (default: $OUTPUT_DIR)"
    echo "  -c, --comprehensive           Run comprehensive benchmarks with scaling analysis"
    echo "  --scaling-only                Run only scaling analysis"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run single benchmark with default settings"
    echo "  $0 -c                        # Run comprehensive benchmarks"
    echo "  $0 -b 16384 -i 50 -k single # Benchmark specific configuration"
    echo "  $0 --scaling-only            # Run scaling analysis only"
}

# Main script
main() {
    # Parse command line arguments
    local batch_size=$DEFAULT_BATCH_SIZE
    local iterations=$DEFAULT_ITERATIONS
    local kernel_type=$DEFAULT_KERNEL_TYPE
    local comprehensive=false
    local scaling_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -b|--batch-size)
                batch_size="$2"
                shift 2
                ;;
            -i|--iterations)
                iterations="$2"
                shift 2
                ;;
            -k|--kernel-type)
                kernel_type="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -c|--comprehensive)
                comprehensive=true
                shift
                ;;
            --scaling-only)
                scaling_only=true
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Validate kernel type
    if [[ ! "$kernel_type" =~ ^(single|pairs|both)$ ]]; then
        echo -e "${RED}Error: Invalid kernel type. Must be 'single', 'pairs', or 'both'.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Poseidon CUDA Performance Benchmark${NC}"
    echo -e "${GREEN}===================================${NC}"
    echo ""
    
    # Setup
    check_prerequisites
    build_project
    setup_output_dir
    
    echo ""
    echo -e "${BLUE}Starting benchmarks...${NC}"
    echo ""
    
    # Run benchmarks based on options
    if [ "$comprehensive" = true ]; then
        run_comprehensive_benchmarks
        generate_analysis_report
        generate_plots
    elif [ "$scaling_only" = true ]; then
        echo "batch_size,iterations,kernel_type,total_hashes,total_time_ms,avg_time_per_hash_ns,hashes_per_second,throughput_mhash_s" > "$OUTPUT_DIR/benchmark_results.csv"
        run_scaling_analysis "$iterations" "$kernel_type"
        generate_analysis_report
        generate_plots
    else
        # Single benchmark run
        echo "batch_size,iterations,kernel_type,total_hashes,total_time_ms,avg_time_per_hash_ns,hashes_per_second,throughput_mhash_s" > "$OUTPUT_DIR/benchmark_results.csv"
        run_benchmark "$batch_size" "$iterations" "$kernel_type" "Single benchmark run"
    fi
    
    echo ""
    echo -e "${GREEN}Benchmarking completed successfully!${NC}"
    echo -e "Results are available in: ${BLUE}$OUTPUT_DIR${NC}"
    if [ "$comprehensive" = true ] || [ "$scaling_only" = true ]; then
        echo -e "Check ${BLUE}$OUTPUT_DIR/performance_analysis.txt${NC} for detailed analysis."
    fi
}

# Run main function with all arguments
main "$@" 