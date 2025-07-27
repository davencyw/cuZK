# Poseidon CUDA Kernel Profiling Guide

This guide shows you how to profile the optimized Poseidon CUDA kernels using NVIDIA Nsight Systems for detailed performance analysis.

## Setup

### Installing NVIDIA Nsight Systems

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install nvidia-nsight-systems-cli

# Or download from NVIDIA Developer site:
# https://developer.nvidia.com/nsight-systems

# Verify installation
nsys --version
```

### Project Structure

```
cuZK/
├── profile_poseidon_nsys.sh           # Nsight Systems profiling script
├── benchmark_poseidon_cuda.sh         # Simple performance benchmarking
└── src/poseidon/cuda/
    └── poseidon_cuda_profiler.cpp     # Profiler executable source
```

## Quick Start

### Basic Profiling

```bash
# Run basic profiling with default settings
./profile_poseidon_nsys.sh --basic-only

# Profile specific configuration
./profile_poseidon_nsys.sh --basic-only -b 8192 -i 25 -k single
```

### Comprehensive Analysis

```bash
# Run comprehensive profiling with multiple configurations
./profile_poseidon_nsys.sh -c

# Run timeline profiling for GUI visualization
./profile_poseidon_nsys.sh --timeline-only
```

## Profiling Modes

### 1. Basic Profiling (`--basic-only`)
- **Purpose**: Quick performance overview
- **Traces**: CUDA kernels, memory transfers
- **Output**: `.nsys-rep` file + text summary
- **Best for**: Initial performance assessment

### 2. Detailed Profiling (`--detailed-only`)
- **Purpose**: Comprehensive kernel analysis
- **Traces**: CUDA kernels, NVTX markers, OS runtime
- **Output**: Detailed kernel statistics
- **Best for**: Understanding kernel behavior

### 3. Memory Profiling (`--memory-only`)
- **Purpose**: Memory transfer and bandwidth analysis
- **Traces**: CUDA memory operations, unified memory
- **Output**: Memory usage and efficiency metrics
- **Best for**: Optimizing memory access patterns

### 4. Timeline Profiling (`--timeline-only`)
- **Purpose**: Visual timeline analysis
- **Traces**: Full application timeline with CPU sampling
- **Output**: Rich GUI-viewable profile
- **Best for**: Visual analysis in Nsight Systems GUI

## Understanding the Results

### Text Reports

After profiling, you'll get text reports like:

```
nsys_profiling_results/
├── basic_profile_both_8192x50.nsys-rep    # Binary profile
├── basic_profile_both_8192x50.txt         # Text summary
├── detailed_profile_both_8192x50.nsys-rep
├── detailed_profile_both_8192x50.txt
└── profiling_summary.txt                  # Overview
```

### Key Metrics to Look For

#### 1. Kernel Execution Time
```
Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)   Max (ns)   Name
--------  ---------------  ---------  ---------  ---------  ---------  ---------  ----
   45.2        123456789         50    2469136    2456789    2234567    2789123  batch_hash_single_kernel_optimized
   38.7        105234567         50    2104691    2089765    1987654    2298765  batch_hash_pairs_kernel_optimized
```

#### 2. Memory Transfer Analysis
```
Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)   Max (ns)   Size (MB)  BW (GB/s)
--------  ---------------  -----  ---------  ---------  ---------  ---------  ---------  ---------
   12.3         34567890     25     1382716    1376543    1234567    1543210      12.3      8.92
```

#### 3. GPU Utilization
- Look for high GPU utilization (>80%)
- Identify idle time between kernel launches
- Check for memory transfer bottlenecks

### GUI Analysis

Open profile files in NVIDIA Nsight Systems GUI:

```bash
# Launch GUI with profile
nsight-sys nsys_profiling_results/timeline_profile_both_8192x50.nsys-rep

# Or just launch GUI and open file
nsight-sys
```

In the GUI, you can:
- **Timeline View**: See kernel execution timeline
- **GPU Metrics**: Monitor GPU utilization, memory bandwidth
- **CUDA Context**: Analyze memory transfers and kernel launches
- **Source Code**: Link performance to source code (if debug info available)

## Advanced Profiling Commands

### Custom nsys Commands

```bash
# Profile with specific GPU metrics
nsys profile --output=custom_profile \
    --trace=cuda,nvtx \
    --gpu-metrics-device=0 \
    --cuda-memory-usage=true \
    build/src/poseidon/poseidon_cuda_profiler 8192 100 both

# Generate specific reports
nsys stats --report gpukernsum custom_profile.nsys-rep
nsys stats --report gputrace --format csv custom_profile.nsys-rep
nsys stats --report gpumemtimesum custom_profile.nsys-rep
```

### Export Data for Analysis

```bash
# Export kernel statistics to CSV
nsys stats --report gpukernsum --format csv profile.nsys-rep > kernel_stats.csv

# Export GPU trace to CSV  
nsys stats --report gputrace --format csv profile.nsys-rep > gpu_trace.csv

# Export memory statistics
nsys stats --report gpumemtimesum --format csv profile.nsys-rep > memory_stats.csv
```

## Performance Optimization Tips

### Based on Profiling Results

1. **High Memory Transfer Time**
   - Increase batch sizes to amortize transfer overhead
   - Consider using CUDA streams for overlap
   - Check for unnecessary CPU-GPU synchronization

2. **Low GPU Utilization**
   - Increase grid/block sizes
   - Reduce register usage per thread
   - Check for divergent branches

3. **Memory Bandwidth Issues**
   - Optimize memory access patterns
   - Use shared memory for frequently accessed data
   - Consider memory coalescing

### Kernel-Specific Optimizations

#### `batch_hash_single_kernel_optimized`
- **Focus**: Individual hash performance
- **Metrics**: Average time per hash, GPU occupancy
- **Optimization**: Thread block size, register usage

#### `batch_hash_pairs_kernel_optimized`  
- **Focus**: Pair hash throughput
- **Metrics**: Pairs processed per second, memory efficiency
- **Optimization**: Memory access patterns, shared memory usage

## Example Workflow

### 1. Initial Assessment
```bash
# Quick profiling to understand baseline
./profile_poseidon_nsys.sh --basic-only -b 4096 -i 20 -k both
```

### 2. Deep Dive Analysis
```bash
# Detailed profiling for optimization targets
./profile_poseidon_nsys.sh --detailed-only -b 8192 -i 50 -k single
./profile_poseidon_nsys.sh --memory-only -b 16384 -i 25 -k pairs
```

### 3. Visual Analysis
```bash
# Timeline profiling for GUI analysis
./profile_poseidon_nsys.sh --timeline-only -b 8192 -i 30 -k both

# Open in GUI
nsight-sys nsys_profiling_results/timeline_profile_both_8192x30.nsys-rep
```

### 4. Optimization Validation
```bash
# Compare before/after optimization
./profile_poseidon_nsys.sh -c
# Analyze comprehensive results to validate improvements
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x profile_poseidon_nsys.sh
   ```

2. **nsys Not Found**
   ```bash
   sudo apt install nvidia-nsight-systems-cli
   ```

3. **GPU Not Detected**
   - Check NVIDIA driver installation
   - Verify GPU is CUDA-compatible
   - Run `nvidia-smi` to test

4. **Large Profile Files**
   - Use `--basic-only` for quick analysis
   - Reduce iteration count for initial testing
   - Clean up old profile files regularly

### Performance Tips

- Start with small batch sizes and iterations for testing
- Use `--basic-only` for quick iteration during development
- Save `--comprehensive` profiling for final analysis
- Archive or delete old `.nsys-rep` files (they can be large)

## Integration with Development

### CMake Integration

The profiler is integrated with the CMake build system:

```bash
# Build profiler
cd build && make poseidon_cuda_profiler

# Run profiling targets
make profile_poseidon_cuda              # Basic nsys profiling
make profile_poseidon_comprehensive     # Comprehensive analysis
```

### Continuous Profiling

Consider integrating profiling into your development workflow:

1. **Feature Development**: Use `--basic-only` for quick checks
2. **Pre-commit**: Run `--detailed-only` for specific kernels
3. **Release**: Use `--comprehensive` for full analysis
4. **Regression Testing**: Compare profile results over time

This profiling setup provides comprehensive insights into your Poseidon CUDA kernel performance and helps identify optimization opportunities. 