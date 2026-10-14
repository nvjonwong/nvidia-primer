<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kokkos NVIDIA GPU Test Suite

Comprehensive test suite validating all patterns from the Kokkos NVIDIA GPU Primer.

See [`TESTING_GUIDE.md`](TESTING_GUIDE.md) for complete documentation.

## Quick Start

```bash
# Build tests
./build.sh

# Run all tests
cd build && ./kokkos_tests

# Run specific suite
./kokkos_tests --gtest_filter="ExecutionSpace.*"

# List all tests
./kokkos_tests --gtest_list_tests
```

## Test Categories (58 tests)

- **Views** - Creation, initialization, transfers
- **Execution Spaces** - Streams, events, dependencies
- **Atomics** - Explicit atomics, prefetching
- **Synchronization** - Implicit vs explicit syncs
- **StreamOrderedView** - Zero-sync allocations
- **Allocation Threshold** - 40KB boundary behavior
- **View Traits** - Memory traits, MDRange
- **UVM & ILP** - Unified memory patterns

## Baselines

```bash
# Create baseline
./create_all_baselines.sh 5.0.1

# Compare after Kokkos upgrade
./compare_all_baselines.sh 5.0.1 5.1.0
```

## Profiling

```bash
./profile_test.sh ExecutionSpace.DefaultSpace
```

## Critical Configuration

For 40KB threshold tests, Kokkos must be built with:
```text
-DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON
```
