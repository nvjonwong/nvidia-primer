<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kokkos Profiling Guide

Profiling Kokkos tests with Nsight Systems and detecting synchronization issues.

---

This guide and `profile_test.sh` operate on the test executable built by this repository; `profile_test.sh` does not accept an arbitrary application. To create an NCU rename map from your own application's Nsight Systems capture, follow [the application workflow](../07-profiling-and-debugging.md#generate-an-ncu-map-from-an-nsight-systems-report) and run `profiling_name_tools.py` on the exported SQLite database.

## Quick Start

Run the common profiling workflow with:

```bash
# Profile a single test
cd test_suite
./profile_test.sh ExecutionSpace.DefaultSpace

# Profile with timeline
./profile_test.sh profile ExecutionSpace.DefaultSpace

# View in GUI
nsys-ui build/profile.nsys-rep

# Extract sync counts
grep -E "Synchronize" build/profile_timeline.txt
```

---

## Table of Contents

1. [Overview](#overview)
2. [Profiling Tools](#profiling-tools)
3. [Quick Start Workflows](#quick-start-workflows)
4. [Timeline Extraction](#timeline-extraction)
5. [Comparing Kokkos vs CUDA](#comparing-kokkos-vs-cuda)
6. [Detecting Synchronization Issues](#detecting-synchronization-issues)
7. [Performance Analysis](#performance-analysis)
8. [Advanced Profiling](#advanced-profiling)
9. [Quick Reference](#quick-reference)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What Profiling Reveals

- **CUDA API calls** - malloc, free, synchronize, memcpy
- **Kernel launches** - Grid/block configuration, execution time
- **NVTX markers** - Named regions, custom markers
- **Memory operations** - Allocations, transfers, page faults
- **Synchronization points** - Both implicit and explicit
- **Stream ordering** - Concurrent execution patterns

### Tools Provided

| Tool | Purpose |
|------|---------|
| `profiling_name_tools.py` | Create an NCU rename YAML file by streaming an exported Nsight Systems SQLite database; usable with any application |
| `verify_profiling_names.sh` | Validate the naming workflow against the Primer test executable |
| `profile_test.sh` | Profile Primer GoogleTests and list, export, or inspect their results |
| `extract_timeline.py` | Detailed chronological timeline extraction |
| `extract_simplified_timeline.py` | Regression-friendly timeline for diffing |
| `create_all_baselines.sh` | Baseline profiling for all test categories |
| `compare_all_baselines.sh` | Compare baselines between Kokkos versions |

---

## Profiling Tools

### 1. profile_test.sh - Main Profiling Script

**Commands:**
```bash
./profile_test.sh list              # Show test suites
./profile_test.sh list-all          # Show all test cases
./profile_test.sh profile <filter>  # Profile matching tests
./profile_test.sh export <file>     # Export summary events
./profile_test.sh timeline <file>   # Export chronological timeline
./profile_test.sh help              # Show usage
```

**Example:**
```bash
# Profile execution space tests
./profile_test.sh profile 'ExecutionSpace.*'

# View results
cat build/profile_timeline.txt
```

**What You Get:**
- Nsight profile (`.nsys-rep`)
- Summary statistics
- Timeline with CUDA API + NVTX + kernels
- Automatic CUDA API summary

### 2. extract_timeline.py - Detailed Timeline

Extracts chronological timeline combining:
- CUDA API calls
- Kernel launches  
- NVTX markers

**Usage:**
```bash
# Via profile_test.sh (recommended)
./profile_test.sh timeline build/profile.nsys-rep

# Direct usage
nsys export -t sqlite build/profile.nsys-rep -o build/profile_data
./extract_timeline.py build/profile_data
```

**Options:**
- `--type CUDA_API` - Show only CUDA API
- `--type NVTX` - Show only NVTX markers
- `--type KERNEL` - Show only kernels
- `--correlation` - Show correlation IDs
- `-o file.txt` - Save to file

**Output Format:**
```text
        Time |     Rel.Time |     Duration |       Type | Event
----------------------------------------------------------------------------------------
    571.85ms |       5.86ms |        960ns |     KERNEL | cuda_parallel_launch...
    571.88ms |       5.90ms |       1.15us |     KERNEL | cuda_parallel_launch...
    571.98ms |       6.00ms |          0ns |       NVTX | Before sum reduction
    572.05ms |       6.06ms |       4.61us |     KERNEL | cuda_parallel_launch...
    572.07ms |       6.29ms |      28.70us |   CUDA_API | cudaMemcpyAsync_v3020
    572.10ms |       6.32ms |       4.38us |   CUDA_API | cudaStreamSynchronize_v3020
```

### 3. extract_simplified_timeline.py - Regression Timeline

Creates diff-friendly timeline for regression testing.

**Usage:**
```bash
./extract_simplified_timeline.py profile.sqlite --output baseline.txt --summary
```

**Options:**
- `--summary` - Include statistics summary
- `--durations` - Include event durations
- `-o file` - Save to file

**Output Format:**
```text
# Simplified Timeline
# Format: RELATIVE_TIME_NS | TYPE | EVENT_NAME

              0 | NVTX       | Before reduce to pinned
          15234 | CUDA_API   | cudaLaunchKernel
          23456 | KERNEL     | Kokkos::parallel_for::init
          45678 | CUDA_API   | cudaLaunchKernel
          56789 | KERNEL     | Kokkos::parallel_reduce::sum
          78901 | NVTX       | After reduce to pinned
```

**Use Case:** Easy to diff between versions
```bash
diff baselines/v1/test_timeline.txt baselines/v2/test_timeline.txt
```

---

## Kernel Naming Verification

The profiling-name harness verifies both the data transformation and the real profiler behavior:

```bash
# No GPU required
./verify_profiling_names.sh unit
./verify_profiling_names.sh preflight

# Requires a CUDA build, nsys, ncu, and profiling permissions
KOKKOS_PRIMER_BUILD_DIR="$PWD/build" ./verify_profiling_names.sh all
```

The earliest configuration tested for this guide used Nsight Systems 2026.2.1 and Nsight Compute 2026.1.1 on an NVIDIA GH200. These are reference versions, not minimum supported versions.

The unit tests create small SQLite fixtures to exercise innermost-range selection, overloaded-symbol handling, and YAML serialization without requiring a GPU. The end-to-end test exports a real Nsight Systems capture to SQLite before generating the rename map. The workload nests the functor launch inside outer and inner NVTX ranges to verify that both the mapper and the `:nvtx-name` reports select the inner range.

The end-to-end sequence is:

1. Run `ProfilingNames.StableKernelNames`, which launches separately named lambda, functor, and work-tag kernels inside named NVTX ranges. The functor range is nested inside an outer range.
2. Capture CUDA and NVTX with Nsight Systems to create an `.nsys-rep` file.
3. Export the capture to SQLite and stream its CUDA runtime, kernel, and NVTX tables to generate `ncu-kernel-renames.yaml` before running NCU.
4. Verify `cuda_gpu_kern_sum:nvtx-name` selects the inner alias and `nvtx_kern_sum` shows both nested ranges.
5. Start a fresh NCU collection with the rename map preloaded, filtering on a stable fragment of the original demangled functor name. Filtering on the original name supports NCU releases that apply kernel filters before display renaming.
6. Import the resulting report and confirm the inner alias is displayed.

Generated files default to `profiling-name-results/`. Override the locations independently:

```bash
KOKKOS_PRIMER_BUILD_DIR=/path/to/test/build \
KOKKOS_PRIMER_PROFILE_OUTPUT_DIR=/tmp/kokkos-profile-names \
./verify_profiling_names.sh all
```

To discover names supported by the installed Nsight version:

```bash
./profile_test.sh recipes
nsys stats --help-reports
nsys recipe --help
```

---

## Quick Start Workflows

### Profile a Single Test

Profile one test through the helper script or directly with Nsight Systems:

```bash
cd test_suite

# Option 1: Using profile_test.sh
./profile_test.sh profile 'ExecutionSpace.DefaultSpace'

# Option 2: Manual
cd build
nsys profile --trace=cuda,nvtx --output=my_profile ./kokkos_tests --gtest_filter='ExecutionSpace.DefaultSpace'
```

### Profile All Tests in a Suite

Use a GoogleTest wildcard to profile a complete suite:

```bash
./profile_test.sh profile 'ExecutionSpace.*'
```

### Profile with Custom Name

Pass a second argument to choose the output name:

```bash
./profile_test.sh profile 'Atomics.AtomicAdd' atomics_test
```

### View Results

Inspect the generated report and text summaries with:

```bash
# View in Nsight Systems GUI
nsys-ui build/profile.nsys-rep

# View timeline in terminal
cat build/profile_timeline.txt

# View summary stats
cat build/profile_summary.txt

# Count synchronizations
grep -c "cudaDeviceSynchronize" build/profile_timeline.txt
```

---

## Timeline Extraction

### Why Extract Timelines?

- **Verify synchronization** - Count sync calls
- **Compare versions** - Diff timelines
- **Debug issues** - See exact operation order
- **Performance analysis** - Identify bottlenecks

### Detailed Timeline (Chronological)

**Create:**
```bash
./profile_test.sh timeline build/profile.nsys-rep
```

**Analyze:**
```bash
# Count specific API calls
grep -c "cudaDeviceSynchronize" build/profile_timeline.txt
grep -c "cudaMallocAsync" build/profile_timeline.txt

# Find NVTX markers
grep "NVTX" build/profile_timeline.txt

# See kernels only
grep "KERNEL" build/profile_timeline.txt
```

### Simplified Timeline (Regression-Friendly)

**Create:**
```bash
./extract_simplified_timeline.py build/profile.sqlite \
    --summary --output baseline.txt
```

**Compare Versions:**
```bash
# After Kokkos upgrade
diff baselines/kokkos_5.0.1/test_timeline.txt \
     baselines/kokkos_5.1.0/test_timeline.txt
```

**What to Look For:**
- New synchronization calls
- Changed order of operations
- Missing/extra allocations
- Kernel launch pattern changes

---

## Comparing Kokkos vs CUDA

### Purpose

Verify Kokkos generates equivalent GPU code to hand-written CUDA.

### Workflow

**1. Profile Kokkos Implementation**

```bash
./profile_test.sh profile 'ViewHelpers.MakeViewUninitialized' kokkos_view
```

**2. Profile CUDA Reference**

```bash
./profile_test.sh profile 'CudaReference.ViewUninitialized' cuda_view
```

**3. Compare Kernel Statistics**

```bash
nsys stats build/kokkos_view.nsys-rep --report cuda_gpu_kern_sum
nsys stats build/cuda_view.nsys-rep --report cuda_gpu_kern_sum
```

**4. Compare API Calls**

```bash
nsys stats build/kokkos_view.nsys-rep --report cuda_api_sum
nsys stats build/cuda_view.nsys-rep --report cuda_api_sum
```

### What to Compare

**Grid and Block Configuration:**
- Block size (threads per block)
- Grid size (number of blocks)
- Shared memory usage

Expected: Similar configurations

**Memory Bandwidth:**
- Global memory load/store transactions
- L1/L2 cache hit rates
- Memory bandwidth utilization

Expected: Within 5-10%

**Occupancy:**
- Active warps per SM
- Register usage per thread
- Shared memory per block

Expected: Similar occupancy

**Execution Time:**
- Kernel execution times
- Total GPU time

Expected: Within 5-10% for optimized code

### Visual Comparison

Open both reports side by side:

```bash
# Open both profiles side-by-side
nsys-ui build/kokkos_view.nsys-rep &
nsys-ui build/cuda_view.nsys-rep &
```

Look for:
- Similar kernel timeline
- Matching memory operations
- Equivalent synchronization patterns

---

## Detecting Synchronization Issues

### Common Synchronization Problems

1. **Unexpected cudaDeviceSynchronize**
   - Problem: Global sync when not needed
   - Impact: Destroys concurrency
   - Detection: Check timeline for device syncs

2. **Allocation/Deallocation Syncs**
   - Problem: View destruction synchronizes
   - Impact: Serializes stream operations
   - Detection: Count syncs around malloc/free

3. **Reduce to Host Syncs**
   - Problem: Result copied to host synchronizes
   - Impact: Prevents overlapping work
   - Detection: Check reduce operations

4. **Deep Copy Syncs**
   - Problem: Implicit fence before copy
   - Impact: Unnecessary synchronization
   - Detection: Timeline shows sync before copy

### Verification Workflow

**Step 1: Profile Test**

```bash
./profile_test.sh profile 'ExecutionSpace.ReduceToPinnedNoSync'
```

**Step 2: Extract Timeline**

```bash
./profile_test.sh timeline build/ExecutionSpace_ReduceToPinnedNoSync.nsys-rep
```

**Step 3: Count Synchronizations**

```bash
# Device syncs (worst)
grep -c "cudaDeviceSynchronize" build/ExecutionSpace_ReduceToPinnedNoSync_timeline.txt

# Stream syncs (better)
grep -c "cudaStreamSynchronize" build/ExecutionSpace_ReduceToPinnedNoSync_timeline.txt
```

**Step 4: Verify Expected Behavior**

For zero-sync test, both counts should be **0**.

### Automated Verification

Create a baseline and inspect its synchronization statistics:

```bash
cd test_suite

# Create baseline for sync behavior category
./create_all_baselines.sh 5.0.1

# Check sync stats
cat baselines/kokkos_5.0.1/SyncBehavior_sync.txt
```

### Manual Inspection

Open the report and inspect the CUDA, NVTX, and kernel rows:

```bash
# Open in GUI
nsys-ui build/profile.nsys-rep

# Look for:
# - CUDA Runtime API row
# - CUDA events (look for Synchronize)
# - NVTX markers (your test annotations)
# - Kernel execution (verify async)
```

---

## Performance Analysis

### Kernel Analysis with Nsight Compute

For detailed kernel-level profiling:

```bash
# Profile specific kernel
ncu --set full --export kernel_profile ./kokkos_tests --gtest_filter='YourTest'

# View report
ncu --import kernel_profile.ncu-rep --page details
```

**Metrics to Check:**
- SM efficiency
- Memory throughput
- Warp execution efficiency
- Register usage
- Occupancy

### Memory Analysis

**Check Memory Operations:**
```bash
nsys stats build/profile.nsys-rep --report cuda_gpu_mem_size_sum
```

**Analyze:**
- Transfer sizes
- Transfer directions (H2D, D2H, D2D)
- Transfer timing
- Async vs sync transfers

**Red Flags:**
- Many small transfers
- Synchronous transfers
- Unnecessary D2H transfers

### Concurrency Analysis

**Visual Timeline:**
```bash
nsys-ui build/profile.nsys-rep
```

Look for:
- Multiple kernels running simultaneously
- Overlapped memory operations
- Stream utilization
- Gaps indicating serialization

**Quantitative:**
```bash
# Kernel overlap
nsys stats build/profile.nsys-rep --report cuda_gpu_kern_sum --format csv

# Check Duration column for overlaps
```

### Identifying Bottlenecks

Sort the API and kernel summaries to identify expensive operations:

```bash
# Top time-consuming operations
nsys stats build/profile.nsys-rep --report cuda_api_sum | head -20

# Kernel execution times
nsys stats build/profile.nsys-rep --report cuda_gpu_kern_sum
```

---

## Advanced Profiling

### Custom NVTX Markers

Add to your tests:
```cpp
#include <nvtx3/nvToolsExt.h>

// Mark a point
nvtxMarkA("Important Event");

// Range
nvtxRangePushA("My Operation");
// ... code ...
nvtxRangePop();
```

**View in Profile:**
These appear as annotations in Nsight Systems timeline.

### Profiling Specific GPU

Select one GPU before launching the profiling helper:

```bash
CUDA_VISIBLE_DEVICES=1 ./profile_test.sh profile 'YourTest'
```

### Kernel-Level Profiling

Collect detailed metrics or filter the collection by kernel name:

```bash
# Detailed kernel metrics
ncu --set full ./kokkos_tests --gtest_filter='YourTest'

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./kokkos_tests --gtest_filter='YourTest'

# Compare kernels
ncu --kernel-name "kernel1|kernel2" ./kokkos_tests
```

### Memory Trace

Enable CUDA memory tracing for the selected test:

```bash
# Capture memory operations
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
    --output=memory_profile ./kokkos_tests --gtest_filter='YourTest'
```

### Python Analysis Script

Create custom analysis:
```python
import sqlite3

# Open exported SQLite
conn = sqlite3.connect('profile.sqlite')

# Query CUDA API calls
cursor = conn.execute('''
    SELECT name, COUNT(*), SUM(end-start) as total_time
    FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE name LIKE '%Synchronize%'
    GROUP BY name
''')

for row in cursor:
    print(f"{row[0]}: {row[1]} calls, {row[2]/1e6:.2f} ms")
```

---

## Quick Reference

### Common Commands

Use these commands for the most common profiling tasks:

```bash
# Profile single test
./profile_test.sh profile 'TestName'

# Extract timeline
./profile_test.sh timeline build/profile.nsys-rep

# Count syncs
grep -c "Synchronize" build/profile_timeline.txt

# View in GUI
nsys-ui build/profile.nsys-rep

# Export stats
nsys stats build/profile.nsys-rep --report cuda_api_sum
```

### Finding Issues

Search an exported timeline for common synchronization and memory operations:

```bash
# Device syncs (bad)
grep "cudaDeviceSynchronize" timeline.txt

# Stream syncs (better)
grep "cudaStreamSynchronize" timeline.txt

# Allocations
grep "cudaMalloc" timeline.txt

# Memory copies
grep "cudaMemcpy" timeline.txt
```

### Expected Patterns

**Good (Zero-Sync):**
```text
cudaLaunchKernel
cudaLaunchKernel
cudaLaunchKernel
# No synchronize calls
```

**Bad (Excessive Sync):**
```text
cudaLaunchKernel
cudaDeviceSynchronize      ← BAD
cudaLaunchKernel
cudaDeviceSynchronize      ← BAD
```

---

## Troubleshooting

### "nsys: command not found"

**Solution:**
```bash
export PATH=/opt/nvidia/nsight-systems/bin:$PATH
# Or
export PATH=/usr/local/cuda/bin:$PATH
```

### Profile File Not Created

**Check:**
```bash
# Nsys version
nsys --version

# Run test manually
cd build
./kokkos_tests --gtest_filter='YourTest'
```

**Common Issues:**
- Test crashes (fix test first)
- Insufficient permissions
- Disk full

### Timeline Extraction Failed

**Solution:**
```bash
# Check SQLite file exists
ls -lh *.sqlite

# Check schema
sqlite3 profile.sqlite ".schema"

# Try different nsys version
```

### GUI Won't Open

**Remote System:**
```bash
# Enable X11 forwarding
ssh -X user@remote
nsys-ui profile.nsys-rep

# Or download file locally
scp user@remote:profile.nsys-rep .
nsys-ui profile.nsys-rep
```

### Excessive Profiling Overhead

**Solutions:**
- Profile smaller tests
- Reduce trace categories: `--trace=cuda`
- Skip sampling: `--sample=none`
- Disable backtrace: `--backtrace=none`

### Different Results on Different Runs

**Causes:**
- GPU thermal throttling
- Other processes on GPU
- CUDA launch blocking set

**Solutions:**
```bash
# Clear GPU
nvidia-smi

# Ensure consistent environment
unset CUDA_LAUNCH_BLOCKING
./profile_test.sh profile 'YourTest'
```
