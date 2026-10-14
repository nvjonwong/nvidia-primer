<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kokkos Testing Guide

Testing, baselines, and regression tracking for the Kokkos NVIDIA GPU Primer.

---

## Quick Start

```bash
# 1. Build tests
cd test_suite && ./build.sh

# 2. Run all tests
cd build && ./kokkos_tests

# 3. Create baseline (for version tracking)
cd .. && ./create_all_baselines.sh 5.0.1

# 4. Compare versions (after Kokkos upgrade)
./compare_all_baselines.sh 5.0.1 5.1.0
```

---

## Table of Contents

1. [Test Suite Overview](#test-suite-overview)
2. [Building and Running Tests](#building-and-running-tests)
3. [Test Categories](#test-categories)
4. [Baseline Management](#baseline-management)
5. [Regression Testing](#regression-testing)
6. [40KB Allocation Threshold Tests](#40kb-allocation-threshold-tests)
7. [Synchronization Verification](#synchronization-verification)
8. [StreamOrderedView Tests](#streamorderedview-tests)
9. [Quick Reference Commands](#quick-reference-commands)
10. [Troubleshooting](#troubleshooting)

---

## Test Suite Overview

### What's Tested

Comprehensive test suite validating all patterns from `NvidiaKokkosPrimer.md`:

- **View Management** - Creation, initialization, transfers
- **Execution Spaces** - Streams, events, dependencies
- **Atomic Operations** - Explicit atomics, view traits
- **Memory Traits** - Restrict, const, atomic, unmanaged
- **MDRangePolicy** - 2D/3D parallel operations
- **Synchronization Behavior** - Implicit vs explicit syncs
- **StreamOrderedView** - Zero-sync allocations
- **UVM and ILP** - Unified memory, instruction-level parallelism
- **Allocation Thresholds** - 40KB boundary behavior

---

## Building and Running Tests

### Prerequisites

- CMake 3.16+
- Kokkos installation (with CUDA backend)
- C++17 compatible compiler
- CUDA toolkit
- Google Test (optional, for framework)
- Nsight Systems (for profiling)

### Quick Build

```bash
cd test_suite
./build.sh
```

### Manual Build

```bash
mkdir -p test_suite/build && cd test_suite/build

cmake .. \
    -DKokkos_DIR=/path/to/kokkos/install/lib/cmake/Kokkos \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17

make -j$(nproc)
```

### Build with Specific Kokkos Configuration

```bash
cmake .. \
    -DKokkos_DIR=/path/to/kokkos/install/lib/cmake/Kokkos \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86" \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

**Critical:** Enable `Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC` for meaningful 40KB threshold testing!

### Running Tests

```bash
cd build

# Run all tests
./kokkos_tests

# Run specific test suite
./kokkos_tests --gtest_filter="ExecutionSpace.*"

# Run specific test
./kokkos_tests --gtest_filter="ExecutionSpace.DefaultSpace"

# List all tests
./kokkos_tests --gtest_list_tests

# Run with verbose output
./kokkos_tests --gtest_verbose
```

### Running in Docker

```bash
docker run --ulimit memlock=-1 --rm -it \
  --privileged --gpus all \
  -v "$(pwd):/workspace" \
  nvidia/cuda:12.0.0-devel-ubuntu22.04 \
  bash -c "cd /workspace/test_suite && ./build.sh && cd build && ./kokkos_tests"
```

---

## Test Categories

### 1. View Management (`test_views.cpp`)

**Tests (8):**
- View default creation (zero-initialized)
- View without initialization
- Helper functions (`make_view_uninitialized`, `make_view_filled`)
- 2D view creation
- Deep copy operations
- Host-device transfers
- UVM view creation

**Key Validations:**
- Views are properly allocated
- Uninitialized views avoid overhead
- Deep copy works correctly
- UVM views accessible from both sides

### 2. Execution Spaces (`test_execution_spaces.cpp`)

**Tests (9):**
- Default execution space
- Explicit CUDA execution spaces
- Multiple independent execution spaces
- Asynchronous deep_copy
- Event-based stream dependencies
- Event chains for complex dependencies
- Reduce to `CudaHostPinnedSpace` (no sync!)
- Multiple reductions to pinned memory

**Key Validations:**
- Streams operate independently
- Events synchronize correctly
- Pinned memory reduces don't synchronize
- Stream ordering respected

### 3. Atomic Operations (`test_atomics_prefetch.cpp`)

**Tests (7):**
- Explicit `atomic_add`, `atomic_fetch_add`
- Selective atomics vs atomic view trait
- Atomic min/max operations
- Memory prefetching for UVM views
- Prefetch to device and host
- Prefetch no-op on non-UVM memory

**Key Validations:**
- Atomics work correctly
- Prefetching improves UVM performance
- Non-UVM prefetch is safe no-op

### 4. View Traits & MDRange (`test_view_traits_mdrange.cpp`)

**Tests (11):**
- View trait detection (Atomic, Restrict, Const, Unmanaged)
- Restrict and const optimizations
- Memory trait introspection
- MDRangePolicy for 2D and 3D
- Recommended tile sizes
- Comparison of MDRange vs nested loops

**Key Validations:**
- Traits detected correctly
- MDRange generates efficient kernels
- Tile sizes are optimal

### 5. Synchronization Behavior (`test_sync_behavior.cpp`)

**Tests (8):**
- Verifies when parallel operations synchronize
- Tests `parallel_for`, `parallel_reduce`, `parallel_scan`
- Tests `deep_copy` synchronization patterns
- Uses profiling to verify actual behavior

**Key Validations:**
- Implicit syncs detected
- Zero-sync patterns confirmed
- Timeline analysis accurate

### 6. StreamOrderedView (`test_stream_ordered_view.cpp`)

**Tests (13):**
- Basic allocation and cleanup
- `parallel_for` usage
- `parallel_reduce` usage
- Operator* accessor
- Implicit conversion to UnManaged view
- Move semantics
- Move assignment
- 2D views
- `make_stream_ordered_view` helper
- Multiple allocations on same stream
- Different data types
- Stream consistency
- Performance comparison

**Key Validations:**
- Zero synchronization on allocation
- Zero synchronization on deallocation
- Correct stream ordering
- Performance equivalent to CUDA

### 7. UVM and ILP (`test_uvm_aliasing_ilp.cpp`)

**Tests:**
- UVM aliasing patterns
- Avoiding read-after-write from global memory
- Register-based computation
- Minimizing global memory round-trips

**Key Validations:**
- ILP patterns improve performance
- UVM aliasing works correctly

### 8. Allocation Threshold (`test_allocation_threshold.cpp`)

**Tests (5):**
- Configuration verification (`VerifyAsyncMallocEnabled`)
- Small allocation (<40KB) behavior
- Large allocation (≥40KB) behavior
- Boundary crossing (39KB, 40KB, 41KB)
- StreamOrderedView for large allocations

**Key Validations:**
- Async malloc flag is enabled
- Small allocations use `cudaMalloc`
- Large allocations use `cudaMallocAsync` + syncs
- StreamOrderedView avoids all syncs

---

## Baseline Management

### Overview

The comprehensive baseline system tracks Kokkos synchronization behavior across versions by:

1. Profiling all test categories with Nsight Systems
2. Extracting synchronization statistics
3. Capturing system configuration
4. Enabling automated version comparison
5. Detecting regressions and improvements

### Critical Configuration

**KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC Must Be ON**

The 40KB allocation threshold only works when this flag is enabled:

```bash
cmake /path/to/kokkos \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON \
    -DKokkos_ARCH_AMPERE80=ON \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc) && make install
```

Without this flag:
- All allocations use `cudaMalloc`/`cudaFree`
- 40KB threshold has no effect
- Threshold tests will skip with warning

### Creating a Baseline

#### First Time Setup

```bash
cd test_suite

# Build tests
./build.sh

# Create comprehensive baseline
./create_all_baselines.sh 5.0.1
```

**Time:** ~10-15 minutes  
**Disk:** ~500MB-1GB per baseline

#### What's Created

```text
baselines/kokkos_5.0.1/
├── BASELINE_SUMMARY.txt                    # Overall summary
├── system_info.txt                         # GPU, CUDA, compiler, flags
├── metadata.txt                            # Baseline creation info
│
├── AllocationThreshold_baseline.nsys-rep   # Nsight profile
├── AllocationThreshold_sync.txt            # Sync API counts
├── AllocationThreshold_timeline.txt        # Event timeline
├── AllocationThreshold_test_output.txt     # Test execution log
│
├── SyncBehavior_baseline.nsys-rep
├── SyncBehavior_sync.txt
└── ... (7 test categories total)
```

### Test Categories in Baseline

| Category | Description | Key Focus |
|----------|-------------|-----------|
| AllocationThreshold | 40KB boundary | cudaMalloc vs cudaMallocAsync |
| SyncBehavior | Sync patterns | Implicit vs explicit syncs |
| ExecutionSpace | Stream mgmt | Multi-stream behavior |
| Views | View operations | Initialization costs |
| StreamOrderedView | Zero-sync pattern | Async memory optimization |
| ViewTraits | Memory traits | Random access, atomic |
| UVMandILP | Advanced patterns | UVM aliasing, parallelism |

### Comparing Versions

After upgrading Kokkos:

```bash
# Rebuild with new Kokkos
cd build && rm -rf * && cmake .. && make -j && cd ..

# Generate new baseline
./create_all_baselines.sh 5.1.0

# Compare
./compare_all_baselines.sh 5.0.1 5.1.0

# Review report
cat baselines/comprehensive_comparison_5.0.1_vs_5.1.0.txt
```

### Interpreting Comparison Results

#### No Regressions ✅
```text
✅ NO REGRESSIONS DETECTED
🎉 Found 2 improvements!

Summary:
Total tests:     65
Regressions:     0
Improvements:    2
Unchanged:       63
```
→ Safe to upgrade!

#### Regressions Found ⚠️
```text
⚠ REGRESSION: ExecutionSpace.MultipleOpsNoSync
  Device Sync:  0 → 2
  Stream Sync:  0 → 0

❌ REGRESSIONS DETECTED!
```
→ Investigate before upgrading

### Investigating Regressions

```bash
# View detailed timeline diff
diff baselines/kokkos_5.0.1/AllocationThreshold_sync.txt \
     baselines/kokkos_5.1.0/AllocationThreshold_sync.txt

# Check synchronization points
grep "cudaDeviceSynchronize" baselines/kokkos_5.1.0/AllocationThreshold_sync.txt

# Open visual profile
nsys-ui baselines/kokkos_5.1.0/AllocationThreshold_baseline.nsys-rep
```

### Baseline Maintenance

**When to Create New Baselines:**
- Major Kokkos version update (5.0 → 5.1)
- Critical configuration change (async malloc toggle)
- GPU upgrade (different architecture)
- Compiler update
- After significant primer updates

**Disk Management:**
- Each baseline: ~500MB-1GB
- Keep: Major versions (5.0, 5.1, 6.0)
- Archive: Minor versions if no changes
- Delete: Failed/incomplete baselines

**Version Control:**
Include in Git:
```bash
git add baselines/kokkos_*/metadata.txt
git add baselines/kokkos_*/BASELINE_SUMMARY.txt
git add baselines/kokkos_*/*_sync.txt
```

Exclude from Git (`.gitignore`):
```gitignore
baselines/**/*.nsys-rep
baselines/**/*.sqlite
```

---

## Regression Testing

### Overview

Regression testing detects synchronization and performance changes when upgrading Kokkos.

### Workflow

#### 1. Generate Initial Baseline

```bash
./create_all_baselines.sh 5.0.1
```

#### 2. After Kokkos Upgrade

```bash
# Rebuild tests
cd build && rm -rf * && cmake .. && make -j && cd ..

# Generate new baseline
./create_all_baselines.sh 5.1.0
```

#### 3. Compare

```bash
./compare_all_baselines.sh 5.0.1 5.1.0
```

#### 4. Review Changes

The comparison checks:
- `cudaDeviceSynchronize` count
- `cudaStreamSynchronize` count
- Memory allocation patterns
- CUDA API call ordering

### What Constitutes a Regression?

**Critical Regressions:**
- New `cudaDeviceSynchronize` calls
- Zero-sync tests now synchronize
- StreamOrderedView tests show syncs
- Reduce-to-pinned now syncs

**Acceptable Changes:**
- Kernel order changes (if sync count unchanged)
- Small timing variations
- Memory allocation timing shifts

**Improvements:**
- Decreased `cudaDeviceSynchronize` count
- More `cudaStreamSynchronize`, less `cudaDeviceSynchronize`
- Better stream ordering

---

## 40KB Allocation Threshold Tests

### Background

Kokkos 5.0.1 has a `memory_threshold_g = 40000` bytes that determines allocation strategy:

- **< 40KB:** Uses `cudaMalloc`/`cudaFree` (2 implicit syncs)
- **≥ 40KB:** Uses `cudaMallocAsync`/`cudaFreeAsync` (3 explicit syncs!)

**Paradox:** Async allocations have MORE synchronization overhead.

**Solution:** Use `StreamOrderedView` for zero-sync allocations of any size.

### Test: Configuration Verification

```cpp
TEST_F(AllocationThresholdInfo, VerifyAsyncMallocEnabled) {
#ifdef KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC
    SUCCEED();
#else
    GTEST_SKIP() << "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC not enabled";
#endif
}
```

This test ensures the critical flag is enabled, or skips with a clear warning.

### Test: Small Allocation Behavior

```cpp
TEST_F(AllocationThresholdTest, SmallAllocationSyncBehavior) {
    const size_t N_elements = (40000 / sizeof(double)) - 1; // < 40KB
    
    NVTX_MARK("Before Small View Allocation");
    {
        Kokkos::View<double*> small_view(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "small_view"), 
            N_elements);
    }
    NVTX_MARK("After Small View Deallocation");
}
```

**Expected:** 
- Uses `cudaMalloc`/`cudaFree`
- 2 synchronizations (both implicit)

### Test: Large Allocation Behavior

```cpp
TEST_F(AllocationThresholdTest, LargeAllocationSyncBehavior) {
    const size_t N_elements = (40000 / sizeof(double)) + 1; // > 40KB
    
    NVTX_MARK("Before Large View Allocation");
    {
        Kokkos::View<double*> large_view(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "large_view"), 
            N_elements);
    }
    NVTX_MARK("After Large View Deallocation");
}
```

**Expected:**
- Uses `cudaMallocAsync`/`cudaFreeAsync`
- 3 synchronizations:
  1. After allocation: `cudaStreamSynchronize` or `cudaDeviceSynchronize`
  2. Before deallocation: `cudaDeviceSynchronize`
  3. After deallocation: `cudaDeviceSynchronize`

### Test: Boundary Crossing

```cpp
TEST_F(AllocationThresholdTest, BoundaryAllocationSyncBehavior) {
    // Test 39KB, 40KB, 41KB
    NVTX_MARK("Before 39KB Allocation");
    { Kokkos::View<double*> view_39k(..., (40000/sizeof(double)) - 1); }
    
    NVTX_MARK("Before 40KB Allocation");
    { Kokkos::View<double*> view_40k(..., (40000/sizeof(double))); }
    
    NVTX_MARK("Before 41KB Allocation");
    { Kokkos::View<double*> view_41k(..., (40000/sizeof(double)) + 1); }
}
```

**Expected:**
- Clear transition at exactly 40KB
- 39KB → `cudaMalloc`
- 40KB → `cudaMallocAsync` (threshold includes boundary)
- 41KB → `cudaMallocAsync`

### Test: StreamOrderedView Zero-Sync

```cpp
TEST_F(AllocationThresholdTest, StreamOrderedViewZeroSync) {
    const size_t N_elements = (40000 / sizeof(double)) + 100; // Large
    
    NVTX_MARK("Before StreamOrderedView Allocation");
    {
        NVKokkos::StreamOrderedView<double*> so_view(
            exec_space.cuda_stream(), N_elements);
    }
    NVTX_MARK("After StreamOrderedView Deallocation");
}
```

**Expected:**
- Uses `cudaMallocAsync`/`cudaFreeAsync`
- **0 synchronizations** (no explicit syncs added by Kokkos)

### Running Threshold Tests

```bash
cd build

# Run all threshold tests
./kokkos_tests --gtest_filter="AllocationThreshold.*"

# Run only config verification
./kokkos_tests --gtest_filter="AllocationThresholdInfo.VerifyAsyncMallocEnabled"

# Profile threshold tests
cd ..
./profile_test.sh AllocationThreshold.BoundaryAllocationSyncBehavior
```

---

## Synchronization Verification

### Purpose

Verify actual synchronization behavior matches primer documentation using profiling.

### Tests

1. **Parallel operations without fence** - Should NOT sync
2. **Explicit fence** - Should sync
3. **Reduce to host** - Should sync
4. **Reduce to CudaHostPinnedSpace** - Should NOT sync
5. **Deep copy before kernel** - Behavior depends on configuration

### Automated Verification

```bash
cd test_suite
./create_all_baselines.sh 5.0.1

# Check sync stats for SyncBehavior category
cat baselines/kokkos_5.0.1/SyncBehavior_sync.txt
```

### Manual Verification

```bash
# Profile specific test
./profile_test.sh SyncBehavior.ReduceToPinnedNoSync

# Check sync count
grep -c "cudaDeviceSynchronize" build/SyncBehavior_ReduceToPinnedNoSync_timeline.txt
```

Expected: `0` for zero-sync tests

---

## StreamOrderedView Tests

### Purpose

Validate that `StreamOrderedView` provides zero-synchronization allocations and deallocations for any size.

### Key Tests

#### BasicAllocation
- Tests allocation, validity, automatic cleanup
- Verifies zero syncs

#### ParallelForUsage
- Uses StreamOrderedView with `parallel_for`
- Verifies computation correctness

#### ParallelReduceUsage  
- Uses StreamOrderedView with `parallel_reduce`
- Verifies result accuracy

#### MoveSemantics
- Tests move constructor
- Ensures proper resource transfer

#### TwoDimensionalView
- Tests 2D StreamOrderedView
- Verifies layout and access

#### StreamConsistency
- Multiple operations on same stream
- Verifies proper ordering

#### PerformanceComparison
- Compares timing vs standard Views
- Shows overhead reduction

### Running StreamOrderedView Tests

```bash
cd build

# Run all StreamOrderedView tests
./kokkos_tests --gtest_filter="StreamOrderedView.*"

# Profile for zero-sync verification
cd ..
./profile_test.sh profile 'StreamOrderedView.*'
```

---

## Quick Reference Commands

### Building

```bash
cd test_suite && ./build.sh
cd build && cmake .. && make -j
```

### Running Tests

```bash
./kokkos_tests                                    # All tests
./kokkos_tests --gtest_filter="ExecutionSpace.*"  # Category
./kokkos_tests --gtest_filter="*.BasicTest"       # Pattern
./kokkos_tests --gtest_list_tests                 # List all
```

### Baselines

```bash
# Create comprehensive baseline
./create_all_baselines.sh 5.0.1

# Compare versions
./compare_all_baselines.sh 5.0.1 5.1.0

# View summary
cat baselines/kokkos_5.0.1/BASELINE_SUMMARY.txt

# View sync stats
cat baselines/kokkos_5.0.1/AllocationThreshold_sync.txt
```

### Profiling

```bash
# Profile single test
./profile_test.sh ExecutionSpace.DefaultSpace

# Extract timeline
./extract_simplified_timeline.py build/profile.sqlite

# Visual profiling
nsys-ui baselines/kokkos_5.0.1/AllocationThreshold_baseline.nsys-rep
```

### Configuration Verification

```bash
# Check async malloc flag
grep "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC" build/CMakeCache.txt

# Run verification test
cd build && ./kokkos_tests --gtest_filter="*.VerifyAsyncMallocEnabled"
```

### Debugging

```bash
# Enable blocking mode
export CUDA_LAUNCH_BLOCKING=1
./kokkos_tests

# Single stream mode
export KOKKOS_DEBUG_SINGLE_STREAM=1
./kokkos_tests

# Memory check
cuda-memcheck ./kokkos_tests
```

---

## Troubleshooting

### Tests Skip with Async Malloc Warning

**Problem:**
```text
⚠️  WARNING: KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC is DISABLED
   The 40KB threshold will NOT affect allocation behavior.
```

**Solution:**
Rebuild Kokkos with the flag:
```bash
cd kokkos/build
cmake .. -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON
make -j$(nproc) && make install
```

### CMake Can't Find Kokkos

**Problem:**
```text
CMake Error: Could not find package Kokkos
```

**Solution:**
```bash
cmake .. -DKokkos_DIR=/path/to/kokkos/install/lib/cmake/Kokkos
```

### Tests Pass on CPU but Fail on GPU

**Solution:**
```bash
# Enable blocking to isolate issue
export CUDA_LAUNCH_BLOCKING=1
./kokkos_tests

# Use memory checker
cuda-memcheck ./kokkos_tests
```

### Baseline Comparison Shows Unexpected Changes

**Check:**
1. System configuration matches:
   ```bash
   diff baselines/kokkos_5.0.1/system_info.txt \
        baselines/kokkos_5.1.0/system_info.txt
   ```

2. Same GPU:
   ```bash
   nvidia-smi
   ```

3. Environmental factors:
   - `CUDA_LAUNCH_BLOCKING` set?
   - Other GPU processes running?
   - Thermal throttling?

### Nsys Not Found

**Solution:**
```bash
export PATH=/opt/nvidia/nsight-systems/bin:$PATH
# Or
export PATH=/usr/local/cuda/bin:$PATH
```

### Profile Extraction Failed

**Solution:**
```bash
# Check SQLite file exists
ls -lh *.sqlite

# Try manual extraction
nsys export -t sqlite profile.nsys-rep
./extract_simplified_timeline.py profile.sqlite
```

### Large Baseline Directories

**Solution:**
```bash
# Remove large files (can regenerate if needed)
rm baselines/*/*.nsys-rep
rm baselines/*/*.sqlite

# Keep lightweight files for git
git add baselines/*/metadata.txt
git add baselines/*/*_sync.txt
git add baselines/*/*_timeline.txt
```

