<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Repository, Examples, and Tests

This chapter describes the repository layout, the `nvkokkos_helpers.hpp` single-header utility library, and the test suite — including how to build and run tests, profile synchronization behavior, and extend coverage with new test cases.

## Repository Structure

The repository is organized as follows:

```text
kokkosprimer/
├── NvidiaKokkosPrimer.md              # Main document (TOC + summary)
├── 01-introduction.md                 # Introduction to Kokkos
├── 02-gpu-architecture.md             # NVIDIA GPU architecture review
├── 03-data-management.md              # Memory management with Kokkos Views
├── 04-execution-and-streams.md        # Execution spaces and CUDA streams
├── 05-kernel-optimization.md          # Kernel performance optimization
├── 06-synchronization-reference.md    # Synchronization reference
├── 07-profiling-and-debugging.md      # Profiling & debugging Kokkos on NVIDIA
├── 08-helper-library.md               # Helper libraries
├── 09-repo-examples-and-tests.md      # This file
├── 10-summary.md                      # Summary and bottom line
├── nvkokkos_helpers.hpp               # Single-header helper library
└── test_suite/
    ├── CMakeLists.txt                 # Build configuration
    ├── build.sh                       # Quick build script
    ├── test_main.cpp                  # Main test driver
    ├── test_views.cpp                 # View creation and management tests
    ├── test_execution_spaces.cpp      # Execution spaces and events
    ├── test_atomics_prefetch.cpp      # Atomics and prefetching
    ├── test_view_traits_mdrange.cpp   # View traits, MDRange, ILP
    ├── test_helpers_streams.cpp       # Helper library utilities
    ├── test_sync_behavior.cpp         # Synchronization behavior verification
    ├── test_stream_ordered_view.cpp   # StreamOrderedView tests
    ├── test_cuda_reference.cu         # Pure CUDA reference implementations
    ├── run_sync_tests.sh              # Automated sync behavior testing
    ├── profile_test.sh                # Profiling script
    ├── profile_common.sh              # Shared profiling functions
    ├── profile_and_verify.sh          # Profile and verify script
    ├── extract_timeline.py            # Timeline extraction from nsys profiles
    ├── extract_simplified_timeline.py # Simplified timeline extraction
    ├── verify_sync_behavior.py        # Automated sync verification
    ├── verify_stream_ordered_zero_sync.sh  # Zero-sync verification
    ├── compare_sync_overhead.sh       # Standard vs StreamOrderedView comparison
    ├── compare_baselines.sh           # Baseline comparison
    ├── generate_baselines.sh          # Baseline generation
    ├── README.md                      # Test suite documentation
    ├── PROFILING_GUIDE.md             # Profiling tools and workflows
    └── TESTING_GUIDE.md               # Sync behavior test documentation
```

## Helper Library: `nvkokkos_helpers.hpp`

The repository includes a single-header helper library that provides all the utilities described in the [Helper Library](08-helper-library.md) chapter:

- `NVKokkos::Event` - CUDA event wrapper for stream dependencies
- `NVKokkos::StreamOrderedView` - RAII zero-sync allocation/deallocation
- `NVKokkos::prefetch_to_device()` / `prefetch_to_host()` - UVM prefetching
- `NVKokkos::make_view_uninitialized()` - Fast view creation
- `NVKokkos::make_mdrange_2d()` / `3d()` / `4d()` - Optimal MDRange tiling
- View introspection helpers (`is_atomic_view`, `is_restrict_view`, etc.)

**Usage:**

```cpp
#include "nvkokkos_helpers.hpp"
```

All utilities are portable and provide no-op fallbacks for non-CUDA backends.

## Test Suite

The test suite validates all patterns and best practices from the primer using Google Test.

### What's Tested

| Test File | What It Covers |
|-----------|---------------|
| `test_views.cpp` | View creation, `WithoutInitializing`, helper functions, deep copy, UVM |
| `test_execution_spaces.cpp` | Explicit CUDA execution spaces, multiple streams, async deep_copy, events |
| `test_atomics_prefetch.cpp` | Explicit atomics, selective vs trait atomics, min/max, UVM prefetching |
| `test_view_traits_mdrange.cpp` | View trait detection, restrict/const, MDRangePolicy tiling, ILP patterns |
| `test_helpers_streams.cpp` | Helper library utilities, two-stream pattern, debug single-stream mode |
| `test_sync_behavior.cpp` | Profiling-based verification of when operations synchronize |
| `test_stream_ordered_view.cpp` | StreamOrderedView RAII zero-sync allocation |
| `test_cuda_reference.cu` | Pure CUDA reference implementations for comparison |

### Building

**Prerequisites:**
- CMake 3.16+
- Kokkos installation (with CUDA backend for full testing)
- C++17 compatible compiler
- CUDA toolkit

**Quick build:**

```bash
cd test_suite
./build.sh
```

**Manual build:**

```bash
cd test_suite
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/kokkos/install -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### Running Tests

Run all tests, or filter to a specific suite using gtest filter flags:

```bash
# Run all tests
./build/kokkos_tests

# Run a specific test suite
./build/kokkos_tests --gtest_filter="ViewManagement.*"
./build/kokkos_tests --gtest_filter="ExecutionSpaces.*"
./build/kokkos_tests --gtest_filter="SyncBehavior.*"

# Debug mode: force single stream
export KOKKOS_DEBUG_SINGLE_STREAM=1
./build/kokkos_tests

# Debug mode: synchronous kernel launches
export CUDA_LAUNCH_BLOCKING=1
./build/kokkos_tests
```

### Running in Docker

The test suite requires a Kokkos installation. The easiest approach is to use an interactive container with an [NVIDIA CUDA base image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda):

```bash
# Start an interactive container from the repository root
docker run --ulimit memlock=-1 --rm -it \
  --privileged --gpus all \
  -v "$(realpath .):/workspace" \
  nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu22.04
```

Inside the container, install dependencies, build Kokkos, then build and run the tests:

```bash
apt-get update && apt-get install -y cmake git build-essential

# Build Kokkos
git clone --depth 1 https://github.com/kokkos/kokkos.git /tmp/kokkos-src
cmake -S /tmp/kokkos-src -B /tmp/kokkos-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/tmp/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_NATIVE=ON \
  -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-install
cmake --build /tmp/kokkos-build -j$(nproc)
cmake --install /tmp/kokkos-build

# Build and run tests
cd /workspace/test_suite
./build.sh --kokkos-path /tmp/kokkos-install
./build/kokkos_tests
```

The CI script `test-ci-locally.sh` in the repository root automates this full workflow.

## Verification Tools

The test suite includes profiling infrastructure to verify that the optimizations described in this guide actually achieve zero-synchronization behavior.

### Profiling Individual Tests

Use `profile_test.sh` from within `test_suite/` to profile individual test cases:

```bash
# List available tests
./profile_test.sh list

# Profile a specific test with automatic summary
./profile_test.sh profile 'ExecutionSpace.DefaultSpace'

# Export chronological timeline
./profile_test.sh timeline build/profile.nsys-rep
```

The profiler automatically shows:
- **CUDA API Summary** - All CUDA runtime calls (malloc, launch, sync, memcpy)
- **Kernel Summary** - Kernel names, grid/block configs, execution times
- **NVTX Events** - Named regions from your code and Kokkos internals

### Synchronization Behavior Verification

The sync behavior tests use NVTX markers and nsys profiling to programmatically verify when Kokkos operations synchronize:

```bash
./run_sync_tests.sh
```

This profiles each test, extracts timeline data, and verifies synchronization behavior matches expectations. Tests cover:
- `parallel_for` does NOT synchronize (release builds)
- `parallel_reduce` to host scalar DOES synchronize
- `parallel_reduce` to device View does NOT synchronize
- `parallel_scan` with scalar return DOES synchronize
- `parallel_scan` with View return does NOT synchronize
- `deep_copy` without execution space DOES synchronize
- `deep_copy` with execution space does NOT synchronize

### StreamOrderedView Zero-Sync Verification

Two scripts verify that `StreamOrderedView` achieves zero synchronization overhead compared to standard Kokkos Views:

```bash
# Verify StreamOrderedView has zero synchronization overhead
./verify_stream_ordered_zero_sync.sh

# Compare standard Kokkos Views vs StreamOrderedView
./compare_sync_overhead.sh
```

**`verify_stream_ordered_zero_sync.sh`** profiles StreamOrderedView and counts synchronization calls, confirming zero sync in allocation/deallocation paths.

**`compare_sync_overhead.sh`** compares standard Kokkos Views versus StreamOrderedView, showing the reduction in synchronization overhead.

### Timeline Extraction

Export and query a chronological timeline from an nsys profile using `profile_test.sh` or directly with `extract_timeline.py`:

```bash
# Extract chronological timeline from an nsys profile
./profile_test.sh timeline build/profile.nsys-rep

# Or use the Python script directly for more control
nsys export -t sqlite build/profile.nsys-rep -o build/profile_data
./extract_timeline.py build/profile_data

# Filter by event type
./extract_timeline.py build/profile_data --type CUDA_API
./extract_timeline.py build/profile_data --type NVTX
./extract_timeline.py build/profile_data --type KERNEL
```

**`extract_timeline.py`** combines CUDA API calls, kernel launches, and NVTX events into a single chronological view:

```text
        Time |     Rel.Time |     Duration |       Type | Event
------------------------------------------------------------------------------------------------------------------------
    571.85ms |       5.86ms |        960ns |     KERNEL | cuda_parallel_launch... [Grid(8,1,1) Block(1,128,1)]
    571.88ms |       5.90ms |       1.15us |     KERNEL | cuda_parallel_launch... [Grid(8,1,1) Block(1,128,1)]
    571.98ms |       6.00ms |          0ns |       NVTX | Before sum reduction (profiling marker)
    572.05ms |       6.06ms |       4.61us |     KERNEL | cuda_parallel_launch... [Grid(4,1,1) Block(1,256,1)]
    572.07ms |       6.29ms |      28.70us |   CUDA_API | cudaMemcpyAsync_v3020
    572.10ms |       6.32ms |       4.38us |   CUDA_API | cudaStreamSynchronize_v3020
```

Use this to spot unexpected synchronization, verify async behavior, and measure gaps between operations.

### Common Profiling Workflows

**Verify async execution:**
```bash
./profile_test.sh profile 'ExecutionSpace.DefaultSpace'
# Check: multiple cudaLaunchKernel without intervening sync
```

**Compare Kokkos vs CUDA:**
```bash
./profile_test.sh profile 'ViewHelpers.FillView'
./profile_test.sh profile 'CudaReference.FillArray'
# Compare kernel counts, grid/block configs, performance
```

**Debug synchronization issues:**
```bash
./profile_test.sh profile 'YourTest.*'
./profile_test.sh timeline build/profile.nsys-rep
grep -E "Synchronize" build/profile_timeline.txt
```

## Extending the Test Suite

To add a new test:

1. Add a test function in the appropriate `.cpp` file (or create a new one)
2. Use descriptive names for parallel regions: `Kokkos::parallel_for("descriptive_name", ...)`
3. Add NVTX markers at key points if verifying synchronization behavior
4. If adding a new `.cpp` file, add it to `CMakeLists.txt`

For synchronization behavior tests specifically, see `SYNC_BEHAVIOR_TESTS.md` for the full protocol involving NVTX markers and `verify_sync_behavior.py`.

See also:
- [test_suite/README.md](test_suite/README.md) — test suite overview and quick start
- [test_suite/PROFILING_GUIDE.md](test_suite/PROFILING_GUIDE.md) — full profiling reference
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) — complete repository navigation
