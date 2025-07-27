#!/bin/bash

# Poseidon CUDA Optimized Kernel Profiler Script using NVIDIA Nsight Systems
# This script runs nsys with various configurations to profile the optimized Poseidon kernels

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_BATCH_SIZE=8192
DEFAULT_ITERATIONS=50
DEFAULT_KERNEL_TYPE="both"
OUTPUT_DIR="nsys_profiling_results"
BUILD_DIR="build"

# Check if nsys is available
check_nsys() {
    if ! command -v nsys &> /dev/null; then
        echo -e "${RED}Error: nsys not found. Please ensure NVIDIA Nsight Systems is installed.${NC}"
        echo -e "${YELLOW}Install with: sudo apt install nvidia-nsight-systems-cli${NC}"
        exit 1
    fi
    
    # Check CUDA device availability
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: No NVIDIA GPU detected or nvidia-smi not available.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ nsys and CUDA GPU detected${NC}"
    
    # Show nsys version
    echo -e "${BLUE}nsys version: $(nsys --version | head -1)${NC}"
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

# Run basic CUDA tracing with nsys
run_basic_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/basic_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running basic nsys profiling (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    nsys profile \
        --output="$output_file" \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
    
    echo -e "${GREEN}✓ Basic profiling completed. Results saved to: ${output_file}.nsys-rep${NC}"
    
    # Generate text report
    echo -e "${BLUE}Generating text summary...${NC}"
    nsys stats --report gputrace,cudaapi "$output_file.nsys-rep" > "$output_file.txt"
    echo -e "${GREEN}✓ Text summary saved to: ${output_file}.txt${NC}"
}

# Run detailed kernel analysis
run_detailed_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/detailed_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running detailed kernel profiling (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    nsys profile \
        --output="$output_file" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi \
        --stop-on-range-end=true \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
    
    echo -e "${GREEN}✓ Detailed profiling completed. Results saved to: ${output_file}.nsys-rep${NC}"
    
    # Generate comprehensive text reports
    echo -e "${BLUE}Generating detailed text reports...${NC}"
    nsys stats --report gputrace,gpukernsum,gpumemtimesum,cudaapi "$output_file.nsys-rep" > "$output_file.txt"
    echo -e "${GREEN}✓ Detailed text summary saved to: ${output_file}.txt${NC}"
}

# Run memory-focused profiling
run_memory_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/memory_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running memory profiling (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    nsys profile \
        --output="$output_file" \
        --force-overwrite=true \
        --trace=cuda \
        --cuda-memory-usage=true \
        --cuda-um-cpu-page-faults=true \
        --cuda-um-gpu-page-faults=true \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
    
    echo -e "${GREEN}✓ Memory profiling completed. Results saved to: ${output_file}.nsys-rep${NC}"
    
    # Generate memory-focused text report
    echo -e "${BLUE}Generating memory analysis report...${NC}"
    nsys stats --report gpumemtimesum,gpumemordersum,cudaapi "$output_file.nsys-rep" > "$output_file.txt"
    echo -e "${GREEN}✓ Memory analysis saved to: ${output_file}.txt${NC}"
}

# Run timeline profiling for visualization
run_timeline_profiling() {
    local batch_size=$1
    local iterations=$2
    local kernel_type=$3
    local output_file="$OUTPUT_DIR/timeline_profile_${kernel_type}_${batch_size}x${iterations}"
    
    echo -e "${YELLOW}Running timeline profiling (batch_size=$batch_size, iterations=$iterations, kernel_type=$kernel_type)...${NC}"
    
    nsys profile \
        --output="$output_file" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --sample=cpu \
        --cpuctxsw=process-tree \
        "$BUILD_DIR/src/poseidon/poseidon_cuda_profiler" "$batch_size" "$iterations" "$kernel_type"
    
    echo -e "${GREEN}✓ Timeline profiling completed. Results saved to: ${output_file}.nsys-rep${NC}"
    echo -e "${BLUE}Open ${output_file}.nsys-rep in NVIDIA Nsight Systems GUI for visualization${NC}"
}

# Run comprehensive profiling
run_comprehensive_profiling() {
    echo -e "${BLUE}Starting comprehensive nsys profiling...${NC}"
    
    # Different batch sizes for comprehensive analysis
    local batch_sizes=(1024 4096 8192 16384 32768)
    local iterations=(50 40 30 20 10)
    
    for i in "${!batch_sizes[@]}"; do
        local batch_size=${batch_sizes[$i]}
        local iter=${iterations[$i]}
        
        echo -e "${BLUE}--- Profiling configuration: ${batch_size} batch size, ${iter} iterations ---${NC}"
        
        # Run all profiling types for both kernels
        run_basic_profiling "$batch_size" "$iter" "both"
        run_detailed_profiling "$batch_size" "$iter" "both"
        run_memory_profiling "$batch_size" "$iter" "both"
        
        # Only run timeline for one configuration to avoid too many large files
        if [ "$i" -eq 2 ]; then  # Middle configuration
            run_timeline_profiling "$batch_size" "$iter" "both"
        fi
    done
}

# Generate summary report
generate_summary() {
    local summary_file="$OUTPUT_DIR/profiling_summary.txt"
    
    echo -e "${BLUE}Generating profiling summary...${NC}"
    
    cat > "$summary_file" << EOF
Poseidon CUDA Optimized Kernel Profiling Summary (NVIDIA Nsight Systems)
=========================================================================

Generated on: $(date)
GPU Information: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1)
nsys version: $(nsys --version | head -1)

Profiling Results:
EOF
    
    # List all generated files
    echo "" >> "$summary_file"
    echo "Generated Profile Files:" >> "$summary_file"
    echo "------------------------" >> "$summary_file"
    
    for file in "$OUTPUT_DIR"/*.nsys-rep; do
        if [ -f "$file" ]; then
            local filesize=$(du -h "$file" | cut -f1)
            echo "  $(basename "$file") - $filesize" >> "$summary_file"
        fi
    done
    
    echo "" >> "$summary_file"
    echo "Generated Text Reports:" >> "$summary_file"
    echo "----------------------" >> "$summary_file"
    
    for file in "$OUTPUT_DIR"/*.txt; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "profiling_summary.txt" ]; then
            echo "  $(basename "$file")" >> "$summary_file"
        fi
    done
    
    echo "" >> "$summary_file"
    echo "How to Analyze Results:" >> "$summary_file"
    echo "----------------------" >> "$summary_file"
    echo "1. View .nsys-rep files in NVIDIA Nsight Systems GUI:" >> "$summary_file"
    echo "   nsight-sys ${OUTPUT_DIR}/timeline_profile_*.nsys-rep" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "2. Key metrics to look for in text reports:" >> "$summary_file"
    echo "   - Kernel execution time and duration" >> "$summary_file"
    echo "   - Memory transfer times (H2D/D2H)" >> "$summary_file"
    echo "   - GPU utilization percentage" >> "$summary_file"
    echo "   - Memory bandwidth utilization" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "3. Additional analysis commands:" >> "$summary_file"
    echo "   nsys stats --report gpukernsum <profile>.nsys-rep" >> "$summary_file"
    echo "   nsys stats --report gputrace --format csv <profile>.nsys-rep" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "For detailed analysis, examine individual profile files." >> "$summary_file"
    
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
    echo "  --timeline-only               Run only timeline profiling"
    echo "  --detailed-only               Run only detailed profiling"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run basic profiling with default settings"
    echo "  $0 -c                        # Run comprehensive profiling"
    echo "  $0 -b 16384 -i 25 -k single # Profile single kernel with specific settings"
    echo "  $0 --memory-only -b 8192     # Run only memory profiling"
    echo "  $0 --timeline-only           # Run timeline profiling for GUI analysis"
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
    local timeline_only=false
    local detailed_only=false
    
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
            --timeline-only)
                timeline_only=true
                shift
                ;;
            --detailed-only)
                detailed_only=true
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
    
    echo -e "${GREEN}Poseidon CUDA Optimized Kernel Profiler (NVIDIA Nsight Systems)${NC}"
    echo -e "${GREEN}===============================================================${NC}"
    echo ""
    
    # Setup
    check_nsys
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
    elif [ "$timeline_only" = true ]; then
        run_timeline_profiling "$batch_size" "$iterations" "$kernel_type"
    elif [ "$detailed_only" = true ]; then
        run_detailed_profiling "$batch_size" "$iterations" "$kernel_type"
    else
        # Run all profiling types
        run_basic_profiling "$batch_size" "$iterations" "$kernel_type"
        run_detailed_profiling "$batch_size" "$iterations" "$kernel_type"
        run_memory_profiling "$batch_size" "$iterations" "$kernel_type"
        run_timeline_profiling "$batch_size" "$iterations" "$kernel_type"
    fi
    
    generate_summary
    
    echo ""
    echo -e "${GREEN}Profiling completed successfully!${NC}"
    echo -e "Results are available in: ${BLUE}$OUTPUT_DIR${NC}"
    echo -e "Check ${BLUE}$OUTPUT_DIR/profiling_summary.txt${NC} for an overview."
    echo ""
    echo -e "${YELLOW}To view results graphically, run:${NC}"
    echo -e "${BLUE}nsight-sys $OUTPUT_DIR/timeline_profile_*.nsys-rep${NC}"
}

# Run main function with all arguments
main "$@" 