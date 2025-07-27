#!/bin/bash

# Poseidon CUDA Optimized Kernel Profiler Script using NVIDIA Nsight Compute
# This script runs ncu with various configurations to analyze GPU kernel performance in detail

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_BATCH_SIZE=8192
DEFAULT_ITERATIONS=10  # Lower for ncu since it's more intensive
DEFAULT_KERNEL_TYPE="both"
OUTPUT_DIR="ncu_profiling_results"
BUILD_DIR="build"

# Global variable to track GPU permissions
GPU_PERF_COUNTERS_AVAILABLE=false

# Check if ncu is available
check_ncu() {
    if ! command -v ncu &> /dev/null; then
        echo -e "${RED}Error: ncu not found. Please ensure NVIDIA Nsight Compute is installed.${NC}"
        echo -e "${YELLOW}Install with: sudo apt install nsight-compute${NC}"
        exit 1
    fi
    
    # Check CUDA device availability
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: No NVIDIA GPU detected or nvidia-smi not available.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ ncu and CUDA GPU detected${NC}"
    
    # Show ncu version
    echo -e "${BLUE}ncu version: $(ncu --version | head -1)${NC}"
    
    # Check GPU performance counter permissions
    check_gpu_permissions
}

# Check GPU performance counter permissions
check_gpu_permissions() {
    echo -e "${BLUE}Checking GPU performance counter permissions...${NC}"
    
    # Test if we can access performance counters
    if ncu --list-sets &> /dev/null; then
        echo -e "${GREEN}✓ GPU performance counters accessible${NC}"
        GPU_PERF_COUNTERS_AVAILABLE=true
    else
        echo -e "${YELLOW}⚠️  GPU performance counters access restricted${NC}"
        echo -e "${YELLOW}   This will limit the available profiling metrics${NC}"
        echo ""
        echo -e "${BLUE}To enable full GPU profiling capabilities:${NC}"
        echo -e "  1. Run as root: sudo ./profile_poseidon_ncu.sh [options]"
        echo -e "  2. Or enable permissions permanently:"
        echo -e "     sudo sh -c 'echo options nvidia NVreg_RestrictProfilingToAdminUsers=0 > /etc/modprobe.d/nvidia-nsight.conf'"
        echo -e "     sudo update-initramfs -u"
        echo -e "     # Then reboot"
        echo ""
        echo -e "${YELLOW}Continuing with limited profiling capabilities...${NC}"
        GPU_PERF_COUNTERS_AVAILABLE=false
    fi
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

# Run basic kernel analysis
run_basic_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/basic_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running basic ncu profiling (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    # Try to run with performance counters, fallback if permission denied
    echo -e "${YELLOW}Attempting profiling with performance counters...${NC}"
    local temp_log="/tmp/ncu_output_$$.log"
    
    if ncu \
        --target-processes application-only \
        --kernel-name-base function \
        --launch-skip-before-match 0 \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section ComputeWorkloadAnalysis \
        --force-overwrite \
        --export "$output_file" \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type" 2>&1 | tee "$temp_log"; then
        echo -e "${GREEN}✓ Full profiling completed${NC}"
    elif grep -q "ERR_NVGPUCTRPERM" "$temp_log" 2>/dev/null; then
        echo -e "${YELLOW}Performance counters access denied, retrying with basic metrics...${NC}"
        ncu \
            --target-processes application-only \
            --kernel-name-base function \
            --launch-skip-before-match 0 \
            --section LaunchStats \
            --force-overwrite \
            --export "$output_file" \
            "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
        echo -e "${YELLOW}✓ Basic profiling completed (limited metrics)${NC}"
    else
        echo -e "${RED}✗ Profiling failed${NC}"
        return 1
    fi
    
    # Cleanup temp log
    rm -f "$temp_log"
    
    echo -e "${GREEN}✓ Basic profiling completed. Results saved to: ${output_file}.ncu-rep${NC}"
}

# Run detailed kernel analysis with full metrics
run_detailed_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/detailed_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running detailed ncu profiling (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    ncu \
        --target-processes application-only \
        --kernel-name-base function \
        --launch-skip-before-match 0 \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section ComputeWorkloadAnalysis \
        --section SchedulerStats \
        --section WarpStateStats \
        --section InstructionStats \
        --section LaunchStats \
        --section Occupancy \
        --export "$output_file" \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
    
    echo -e "${GREEN}✓ Detailed profiling completed. Results saved to: ${output_file}.ncu-rep${NC}"
}

# Run memory-focused analysis
run_memory_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/memory_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running memory profiling (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    ncu \
        --target-processes application-only \
        --kernel-name-base function \
        --launch-skip-before-match 0 \
        --section MemoryWorkloadAnalysis \
        --section MemoryWorkloadAnalysis_Chart \
        --section MemoryWorkloadAnalysis_Tables \
        --section LaunchStats \
        --section Occupancy \
        --export "$output_file" \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
    
    echo -e "${GREEN}✓ Memory profiling completed. Results saved to: ${output_file}.ncu-rep${NC}"
}

# Run roofline analysis
run_roofline_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/roofline_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running roofline analysis (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    ncu \
        --target-processes application-only \
        --kernel-name-base function \
        --launch-skip-before-match 0 \
        --section SpeedOfLight \
        --section ComputeWorkloadAnalysis \
        --section MemoryWorkloadAnalysis \
        --section SpeedOfLight_RooflineChart \
        --export "$output_file" \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
    
    echo -e "${GREEN}✓ Roofline analysis completed. Results saved to: ${output_file}.ncu-rep${NC}"
}

# Run comprehensive profiling for optimization
run_optimization_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/optimization_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running optimization profiling (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    ncu \
        --target-processes application-only \
        --kernel-name-base function \
        --launch-skip-before-match 0 \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section ComputeWorkloadAnalysis \
        --section SchedulerStats \
        --section WarpStateStats \
        --section InstructionStats \
        --section LaunchStats \
        --section Occupancy \
        --section SpeedOfLight_RooflineChart \
        --section MemoryWorkloadAnalysis_Chart \
        --section ComputeWorkloadAnalysis_Chart \
        --export "$output_file" \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
    
    echo -e "${GREEN}✓ Optimization profiling completed. Results saved to: ${output_file}.ncu-rep${NC}"
}

# Run specific kernel analysis
run_kernel_specific_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    
    echo -e "${BLUE}Running kernel-specific analysis...${NC}"
    
    # Profile single kernel if requested or both
    if [[ "$kernel_type" == "single" || "$kernel_type" == "both" ]]; then
        local output_file="$OUTPUT_DIR/kernel_single_${batch_size}x${iterations}"
        echo -e "${YELLOW}Profiling single hash kernel...${NC}"
        
        ncu \
            --target-processes application-only \
            --kernel-name "batch_hash_single_kernel_optimized" \
            --launch-skip-before-match 0 \
            --section SpeedOfLight \
            --section MemoryWorkloadAnalysis \
            --section ComputeWorkloadAnalysis \
            --section Occupancy \
            --export "$output_file" \
            "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "single"
        
        echo -e "${GREEN}✓ Single kernel analysis completed: ${output_file}.ncu-rep${NC}"
    fi
    
    # Profile pairs kernel if requested or both
    if [[ "$kernel_type" == "pairs" || "$kernel_type" == "both" ]]; then
        local output_file="$OUTPUT_DIR/kernel_pairs_${batch_size}x${iterations}"
        echo -e "${YELLOW}Profiling pairs hash kernel...${NC}"
        
        ncu \
            --target-processes application-only \
            --kernel-name "batch_hash_pairs_kernel_optimized" \
            --launch-skip-before-match 0 \
            --section SpeedOfLight \
            --section MemoryWorkloadAnalysis \
            --section ComputeWorkloadAnalysis \
            --section Occupancy \
            --export "$output_file" \
            "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "pairs"
        
        echo -e "${GREEN}✓ Pairs kernel analysis completed: ${output_file}.ncu-rep${NC}"
    fi
}

# Run comprehensive profiling with multiple configurations
run_comprehensive_profiling() {
    echo -e "${BLUE}Starting comprehensive ncu profiling...${NC}"
    
    # Different batch sizes for comprehensive analysis
    local batch_sizes=(1024 4096 8192 16384)
    local iterations=(10 8 6 4)  # Fewer iterations for larger batch sizes
    
    for i in "${!batch_sizes[@]}"; do
        local batch_size=${batch_sizes[$i]}
        local iter=${iterations[$i]}
        
        echo -e "${BLUE}--- Profiling configuration: ${batch_size} batch size, ${iter} iterations ---${NC}"
        
        # Run different analysis types
        run_basic_profiling "$batch_size" "$iter" "both"
        run_memory_profiling "$batch_size" "$iter" "both"
        
        # Only run detailed analysis for middle configuration to save time
        if [ "$i" -eq 2 ]; then
            run_detailed_profiling "$batch_size" "$iter" "both"
            run_roofline_profiling "$batch_size" "$iter" "both"
            run_optimization_profiling "$batch_size" "$iter" "both"
            run_kernel_specific_profiling "$batch_size" "$iter" "both"
        fi
    done
}

# Generate summary report
generate_summary() {
    local summary_file="$OUTPUT_DIR/profiling_summary.txt"
    
    echo -e "${BLUE}Generating profiling summary...${NC}"
    
    cat > "$summary_file" << EOF
Poseidon CUDA Optimized Kernel Profiling Summary (NVIDIA Nsight Compute)
========================================================================

Generated on: $(date)
GPU Information: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1)
ncu version: $(ncu --version | head -1)

Profiling Results:
EOF
    
    # List all generated files
    echo "" >> "$summary_file"
    echo "Generated Profile Files:" >> "$summary_file"
    echo "------------------------" >> "$summary_file"
    
    for file in "$OUTPUT_DIR"/*.ncu-rep; do
        if [ -f "$file" ]; then
            local filesize=$(du -h "$file" | cut -f1)
            echo "  $(basename "$file") - $filesize" >> "$summary_file"
        fi
    done
    
    echo "" >> "$summary_file"
    echo "How to Analyze Results:" >> "$summary_file"
    echo "----------------------" >> "$summary_file"
    echo "1. View .ncu-rep files in NVIDIA Nsight Compute GUI:" >> "$summary_file"
    echo "   ncu-ui ${OUTPUT_DIR}/*.ncu-rep" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "2. Key metrics to analyze:" >> "$summary_file"
    echo "   - Speed of Light (SOL) - Shows limiting factors" >> "$summary_file"
    echo "   - Memory Workload Analysis - Memory efficiency" >> "$summary_file"
    echo "   - Compute Workload Analysis - Arithmetic intensity" >> "$summary_file"
    echo "   - Occupancy - Warp and thread block utilization" >> "$summary_file"
    echo "   - Roofline Chart - Performance vs memory bandwidth" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "3. Command line analysis:" >> "$summary_file"
    echo "   ncu --csv --page details <profile>.ncu-rep" >> "$summary_file"
    echo "   ncu --page SpeedOfLight <profile>.ncu-rep" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "4. Focus areas for optimization:" >> "$summary_file"
    echo "   - Check if memory or compute bound" >> "$summary_file"
    echo "   - Analyze memory access patterns" >> "$summary_file"
    echo "   - Review occupancy and warp efficiency" >> "$summary_file"
    echo "   - Compare single vs pairs kernel performance" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "For detailed optimization guidance, examine individual profile files in ncu-ui." >> "$summary_file"
    
    echo -e "${GREEN}✓ Summary generated: $summary_file${NC}"
}

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                    Show this help message"
    echo "  -b, --batch-size SIZE         Set batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  -i, --iterations COUNT        Set iteration count (default: $DEFAULT_ITERATIONS)"
    echo "  -k, --kernel-type TYPE        Set kernel type: single, pairs, both (default: $DEFAULT_KERNEL_TYPE)"
    echo "  -o, --output-dir DIR          Set output directory (default: $OUTPUT_DIR)"
    echo "  -c, --comprehensive           Run comprehensive profiling with multiple configurations"
    echo "  --basic-only                  Run only basic profiling"
    echo "  --memory-only                 Run only memory profiling"
    echo "  --roofline-only               Run only roofline analysis"
    echo "  --detailed-only               Run only detailed profiling"
    echo "  --optimization-only           Run only optimization profiling"
    echo "  --kernel-specific-only        Run only kernel-specific profiling"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run basic profiling with default settings"
    echo "  $0 -c                        # Run comprehensive profiling"
    echo "  $0 -b 16384 -i 5 -k single   # Profile single kernel with specific settings"
    echo "  $0 --memory-only -b 8192     # Run only memory profiling"
    echo "  $0 --roofline-only           # Run roofline analysis"
    echo "  $0 --kernel-specific-only    # Profile individual kernels"
}

# Main script
main() {
    # Parse command line arguments
    local batch_size=$DEFAULT_BATCH_SIZE
    local iterations=$DEFAULT_ITERATIONS
    local kernel_type=$DEFAULT_KERNEL_TYPE
    local comprehensive=false
    local basic_only=false
    local memory_only=false
    local roofline_only=false
    local detailed_only=false
    local optimization_only=false
    local kernel_specific_only=false
    
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
            --basic-only)
                basic_only=true
                shift
                ;;
            --memory-only)
                memory_only=true
                shift
                ;;
            --roofline-only)
                roofline_only=true
                shift
                ;;
            --detailed-only)
                detailed_only=true
                shift
                ;;
            --optimization-only)
                optimization_only=true
                shift
                ;;
            --kernel-specific-only)
                kernel_specific_only=true
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
    
    echo -e "${GREEN}Poseidon CUDA Optimized Kernel Profiler (NVIDIA Nsight Compute)${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo ""
    
    # Setup
    check_ncu
    build_project
    setup_output_dir
    
    echo ""
    echo -e "${BLUE}Starting profiling with configuration:${NC}"
    echo -e "  Batch size: $batch_size"
    echo -e "  Iterations: $iterations"
    echo -e "  Kernel type: $kernel_type"
    echo -e "  Output directory: $OUTPUT_DIR"
    echo ""
    
    # Run profiling based on options
    if [ "$comprehensive" = true ]; then
        run_comprehensive_profiling
    elif [ "$basic_only" = true ]; then
        run_basic_profiling "$batch_size" "$iterations" "$kernel_type"
    elif [ "$memory_only" = true ]; then
        run_memory_profiling "$batch_size" "$iterations" "$kernel_type"
    elif [ "$roofline_only" = true ]; then
        run_roofline_profiling "$batch_size" "$iterations" "$kernel_type"
    elif [ "$detailed_only" = true ]; then
        run_detailed_profiling "$batch_size" "$iterations" "$kernel_type"
    elif [ "$optimization_only" = true ]; then
        run_optimization_profiling "$batch_size" "$iterations" "$kernel_type"
    elif [ "$kernel_specific_only" = true ]; then
        run_kernel_specific_profiling "$batch_size" "$iterations" "$kernel_type"
    else
        # Run basic profiling by default
        run_basic_profiling "$batch_size" "$iterations" "$kernel_type"
        run_memory_profiling "$batch_size" "$iterations" "$kernel_type"
    fi
    
    generate_summary
    
    echo ""
    echo -e "${GREEN}Profiling completed successfully!${NC}"
    echo -e "Results are available in: ${BLUE}$OUTPUT_DIR${NC}"
    echo -e "Check ${BLUE}$OUTPUT_DIR/profiling_summary.txt${NC} for an overview."
    echo ""
    echo -e "${YELLOW}To view results graphically, run:${NC}"
    echo -e "${BLUE}ncu-ui $OUTPUT_DIR/*.ncu-rep${NC}"
}

# Run main function with all arguments
main "$@" 