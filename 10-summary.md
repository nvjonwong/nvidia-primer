<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Summary

This guide demonstrates how to achieve CUDA-equivalent performance in Kokkos without writing raw CUDA, focusing on eliminating unnecessary synchronization and matching CUDA's stream-ordered execution model.

## Core Performance Strategies

### 1. Eliminate Implicit Synchronization

**View Initialization**: Use `Kokkos::view_alloc(Kokkos::WithoutInitializing, label)` for temporary arrays that will be overwritten. Default initialization zeros memory and synchronizes.

**View Allocation/Deallocation**: Use `NVKokkos::StreamOrderedView` for zero-synchronization allocation and deallocation. Standard Kokkos Views synchronize multiple times during construction and destruction, even with `KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC` enabled.

**Reduction Results**: Reduce to device Views or CudaHostPinnedSpace Views instead of host scalars. Reducing to a host scalar forces immediate synchronization. CudaHostPinnedSpace is device-accessible, so GPU writes directly without synchronization - ideal when you need results on host eventually.

**Data Transfers**: Always pass execution space instances to `Kokkos::deep_copy()`. Without an execution space, deep_copy synchronizes the entire device.

### 2. Stream-Ordered Execution

**Execution Spaces**: Use explicit `Kokkos::Cuda` execution space instances to control stream ordering. All operations on the same execution space execute in order without explicit synchronization.

**Multiple Streams**: Use `Kokkos::Experimental::partition_space` to derive independent execution spaces from a base execution space for concurrent execution. Use the two-stream alternating pattern for most applications - this balances concurrency with simplicity.

**Synchronization Points**: Only call `.fence()` when dependencies require it (e.g., host needs GPU results, cross-stream dependencies). Kokkos 4.x+ fixed implicit fence issues - kernels now queue asynchronously by default.

**Stream Dependencies**: Use `NVKokkos::Event` wrapper for fine-grained cross-stream dependencies via CUDA events. Standard Kokkos only provides coarse-grained `.fence()` for synchronization.

### 3. Memory Management Patterns

**Allocation Reuse**: Allocate Views outside hot loops and reuse them. Even with async allocation, reuse is faster than recreation.

**Memory Spaces**: Use `Kokkos::CudaSpace` (device memory) for best performance. `Kokkos::CudaUVMSpace` (managed memory) has migration overhead on discrete GPUs but performs better on Grace Hopper.

**Prefetching**: For UVM allocations, use `NVKokkos::prefetch_to_device()` and `prefetch_to_host()` to minimize page fault overhead.

**UnManaged Views**: Wrap externally-managed memory with UnManaged trait to prevent Kokkos from managing lifecycle. This is the foundation of `StreamOrderedView`.

### 4. Kernel Optimization

**MDRangePolicy Tiling**: Override default tile sizes with recommended values: `{1, 64}` for 2D, `{1, 1, 64}` for 3D. Defaults often use too few threads per block. Use `NVKokkos::make_mdrange_*` helpers for automatic optimal tiling.

**Avoid Read-After-Write on Global Memory**: Load values to registers once, perform all logic in registers, write back once. Don't read from global memory immediately after writing to check conditionals.

**Selective Atomics**: Use explicit `Kokkos::atomic_add()` on regular Views instead of marking entire Views with Atomic trait. Atomic trait makes all accesses atomic (significant overhead).

**Restrict and Const**: Use `Kokkos::MemoryTraits<Kokkos::Restrict>` on non-aliasing Views and mark read-only inputs as const to enable compiler optimizations.

### 5. Debugging Strategies

**Single Stream Testing**: Force all execution spaces to use one stream to isolate stream synchronization bugs. If bugs disappear with single stream, you have a dependency issue.

**CUDA_LAUNCH_BLOCKING**: Set to 1 to make all kernel launches synchronous for easier debugging of correctness issues.

**Profiling with Nsight Systems**: Use `nsys profile` to verify synchronization patterns. Look for cudaDeviceSynchronize and cudaStreamSynchronize calls - they should only appear at explicit fence points.

**Timeline Extraction**: Use `extract_timeline.py` tool to see chronological order of allocations, kernels, and synchronizations. Verify that async operations truly don't synchronize.

## Key Workarounds for Kokkos Limitations

### Zero-Sync Allocation: StreamOrderedView

Kokkos's `cudaMallocAsync` implementation still synchronizes before and after allocation/deallocation. `NVKokkos::StreamOrderedView` provides RAII wrapper around UnManaged Views with manual `cudaMallocAsync`/`cudaFreeAsync`, achieving truly zero-synchronization overhead. This is essential for performance-critical code with frequent temporary allocations.

### Event-Based Dependencies: NVKokkos::Event

Kokkos lacks native CUDA event support for fine-grained stream dependencies. The Event wrapper provides `record()` and `make_stream_wait()` methods for cross-stream synchronization without device-wide fences.

### Memory Prefetching: Helper Functions

Kokkos has no prefetch API for UVM. Helper functions check memory space at compile-time and call `cudaMemPrefetchAsync` only for UVM Views, making them no-ops for other memory spaces.

### Optimal Tile Sizes: Helper Functions

Kokkos MDRangePolicy defaults to suboptimal tile sizes (32-64 threads). Helper functions automatically apply recommended 64-thread-per-block tiling for better occupancy.

## When Kokkos Matches CUDA Performance

With these strategies, Kokkos achieves CUDA-equivalent performance:

**Stream Ordering**: Operations on same execution space execute in order, just like CUDA streams. No implicit synchronization between operations.

**Asynchronous Execution**: Kernels and data transfers queue asynchronously on streams. Host immediately continues to next operation.

**Memory Operations**: With StreamOrderedView, allocations and deallocations are fully stream-ordered with zero synchronization overhead, matching `cudaMallocAsync`/`cudaFreeAsync` behavior.

**Concurrency**: Partitioned execution space instances enable concurrent kernel execution and computation/transfer overlap, matching multi-stream CUDA patterns.

**Resource Management**: RAII wrappers (StreamOrderedView, Event) provide automatic cleanup with zero overhead, improving on manual CUDA memory management.

## Verification Tools

The test suite includes profiling infrastructure to verify zero-synchronization behavior:

**verify_stream_ordered_zero_sync.sh**: Profiles StreamOrderedView and counts synchronization calls, confirming zero sync in allocation/deallocation paths.

**compare_sync_overhead.sh**: Compares standard Kokkos Views versus StreamOrderedView, showing dramatic reduction in synchronization overhead.

**extract_timeline.py**: Extracts chronological timeline from nsys profiles to visualize operation ordering and identify unwanted synchronizations.

**profile_common.sh**: Shared library of profiling functions for consistent verification across tools.

These tools prove that the optimizations in this guide achieve true CUDA-level performance without writing raw CUDA.

## Bottom Line

Kokkos can almost achieve CUDA-equivalent performance (for most workloads) on NVIDIA GPUs by:
1. Using explicit execution spaces for stream control
2. Avoiding implicit synchronization (WithoutInitializing, device Views, execution space instances)
3. Using StreamOrderedView for zero-sync allocation/deallocation
4. Applying optimal kernel configurations (tile sizes, memory traits)
5. Leveraging helper library for missing CUDA features (events, prefetch)
6. Verifying with profiling tools that synchronizations only occur at intended points

The result is portable, high-level code with the performance characteristics of hand-tuned CUDA, without sacrificing the productivity benefits of Kokkos's abstraction layer.

See also:
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) — complete repository navigation
