# NVIDIA Nsight Compute CLI Analysis Guide

## 🚀 Quick Analysis Commands

### Basic Profile Overview
```bash
ncu --import <profile>.ncu-rep --page details
```

### Raw Metrics with Summary
```bash
ncu --import <profile>.ncu-rep --page raw --print-summary per-kernel
```

### CSV Export for Data Analysis
```bash
ncu --import <profile>.ncu-rep --csv --page details > analysis.csv
```

## 📈 Key Performance Metrics

### 1. **Occupancy Analysis**
```bash
# Check waves per SM (0.024 = very low occupancy!)
ncu --import <profile>.ncu-rep --page raw | grep "launch__waves_per_multiprocessor"

# Register usage limiting occupancy
ncu --import <profile>.ncu-rep --page raw | grep "launch__occupancy_limit_registers"
```

### 2. **Throughput Analysis** 
```bash
# Memory throughput (1.22% = very low!)
ncu --import <profile>.ncu-rep --page raw | grep "gpu__compute_memory_throughput.*pct_of_peak_sustained_elapsed"

# SM compute throughput (6% = compute underutilized)
ncu --import <profile>.ncu-rep --page raw | grep "sm__throughput.*pct_of_peak_sustained_elapsed"
```

### 3. **Memory Efficiency**
```bash
# L1 cache hit rate (99%+ = excellent!)
ncu --import <profile>.ncu-rep --page raw | grep "l1tex__t_sector_hit_rate"

# L2 cache hit rate (99%+ = excellent!)
ncu --import <profile>.ncu-rep --page raw | grep "lts__t_sector_hit_rate"
```

### 4. **Instruction Analysis**
```bash
# ALU pipeline utilization (41% = well utilized)
ncu --import <profile>.ncu-rep --page raw | grep "sm__pipe_alu_cycles_active.*pct_of_peak_sustained_active"

# Instructions per cycle
ncu --import <profile>.ncu-rep --page raw | grep "sm__inst_executed.*per_cycle_active"
```

## 🎯 **Performance Bottleneck Identification**

### Current Issues Found:
1. **Grid Size Too Small**: Only 0.024 waves per SM
   - Grid: (16,1,1) × Block: (128,1,1) = 2048 threads total
   - A100 has 108 SMs, needs ~15,000+ threads for full occupancy

2. **Memory Throughput**: 1.22% of peak (severely underutilized)

3. **Compute Throughput**: 6% of peak (not compute-bound)

## 🔧 **Advanced Analysis Commands**

### Kernel Comparison
```bash
# Compare single vs pairs kernels
ncu --import <profile>.ncu-rep --page raw --print-summary per-kernel | grep -A1 -B1 "batch_hash"
```

### Source Code Analysis 
```bash
# View SASS assembly with metrics correlation
ncu --import <profile>.ncu-rep --page source --print-source sass
```

### Session Information
```bash
# Device and configuration details
ncu --import <profile>.ncu-rep --page session
```

## 📊 **Optimization Recommendations**

### 1. **Increase Grid Size**
```bash
# Current: 16 blocks × 128 threads = 2,048 threads
# Target: ~108 × 32 × 4 = 13,824+ threads for better occupancy
```

### 2. **Profile with Larger Batch Sizes**
```bash
sudo ./profile_poseidon_ncu.sh --basic-only -b 16384 -i 5
```

### 3. **Memory Optimization Analysis**
```bash
# Check memory access patterns
ncu --import <profile>.ncu-rep --csv | grep -E "Memory|Throughput|Bandwidth"
```

## 🎮 **Interactive Analysis**

### Create Custom Reports
```bash
# Extract specific metrics to file
ncu --import <profile>.ncu-rep --page raw --print-summary per-kernel | \
grep -E "throughput|occupancy|hit_rate" > performance_summary.txt
```

### Filter by Kernel Name  
```bash
# Analyze specific kernels
ncu --import <profile>.ncu-rep --csv | grep "batch_hash_single_kernel"
ncu --import <profile>.ncu-rep --csv | grep "batch_hash_pairs_kernel"
```

## 📄 **Generate Reports**

### Comprehensive Analysis
```bash
# Full analysis script
echo "=== POSEIDON KERNEL ANALYSIS ===" > full_analysis.txt
echo "Grid Size Issues:" >> full_analysis.txt
ncu --import <profile>.ncu-rep --page raw | grep "launch__waves_per_multiprocessor" >> full_analysis.txt
echo -e "\nMemory Efficiency:" >> full_analysis.txt  
ncu --import <profile>.ncu-rep --page raw | grep "hit_rate" >> full_analysis.txt
echo -e "\nThroughput Issues:" >> full_analysis.txt
ncu --import <profile>.ncu-rep --page raw | grep "throughput.*pct_of_peak_sustained_elapsed" >> full_analysis.txt
```

## 🎯 **Key Findings Summary**

Your Poseidon kernels show:
- ✅ **Excellent cache efficiency** (99%+ hit rates)
- ✅ **Well-optimized ALU usage** (41% utilization) 
- ❌ **Severe occupancy issues** (0.024 waves/SM vs optimal ~1.0)
- ❌ **Memory bandwidth underutilization** (1.22% vs target 80%+)
- ❌ **Grid too small** for GPU parallelism

**Primary optimization**: Increase batch sizes and grid dimensions for better GPU utilization! 