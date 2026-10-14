/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "../nvkokkos_helpers.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

void test_default_space() {
    const int N = 1000;
    
    // Use explicit execution space instance to control synchronization precisely
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<int*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Pre-allocate reduction output View WITHOUT initialization to avoid sync
    Kokkos::View<int> sum_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));
    
    // Kernel 1: Initialize data (no implicit fence in Kokkos 4.x+)
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) = i;
        });
    
    // Kernel 2: Modify data (no implicit fence in Kokkos 4.x+)
    Kokkos::parallel_for("modify",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) *= 2;
        });
 
    // Place NVTX marker to mark point before fence for profiling synchronization
    #ifdef KOKKOS_ENABLE_CUDA
    nvtxMarkA("Before sum reduction (profiling marker)");
    #endif
        
    // Kernel 3: Reduce to device View (NO fence! Result stays on GPU)
    Kokkos::parallel_reduce("sum",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, int& lsum) {
            lsum += data(i);
        }, sum_device);
    // NO implicit fence here - all work queued on GPU!

    // Place NVTX marker to mark point before fence for profiling synchronization
    #ifdef KOKKOS_ENABLE_CUDA
    nvtxMarkA("Before explicit fence (profiling marker)");
    #endif
    
    // Pre-create host mirror to avoid extra allocations during copy
    auto sum_host = Kokkos::create_mirror_view(sum_device);
    
    // Use explicit deep_copy with execution space to control synchronization
    Kokkos::deep_copy(exec_space, sum_host, sum_device);
    
    // Only fence THIS execution space's stream (not entire device!)
    exec_space.fence();  // Synchronize only this stream
    
    int expected = N * (N - 1);
    EXPECT_EQ(sum_host(), expected);
}

// Note. We need a way to verify that when kokkos knows in the reduction there is a host view for the result of sum that it will implicitly synchronize, but if it's device it will not?

#ifdef KOKKOS_ENABLE_CUDA
void test_explicit_cuda_space() {
    const int N = 1000;
    Kokkos::Cuda exec_space;
    Kokkos::View<int*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Pre-allocate reduction output View WITHOUT initialization to avoid sync
    Kokkos::View<int> sum_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));
    
    // Launch kernel 1 - NO implicit fence after this
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) = i * 2;
        });
    // NO fence here! Kernels queue on same stream in order
    
    // Launch kernel 2 - Reduce to device View (NO fence!)
    Kokkos::parallel_reduce("sum",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, int& lsum) {
            lsum += data(i);
        }, sum_device);
 
    // Pre-create host mirror
    auto sum_host = Kokkos::create_mirror_view(sum_device);
    
    // Deep copy with execution space instance
    Kokkos::deep_copy(exec_space, sum_host, sum_device);
    
    // Only fence THIS execution space's stream
    exec_space.fence();
    
    int expected = N * (N - 1);
    EXPECT_EQ(sum_host(), expected);
}

// Verify no implicit fences in Kokkos 4.x+ (same pattern as test_default_space)
void test_default_space_no_implicit_fences() {
    const int N = 1000;
    
    // Use explicit execution space instance for precise stream control
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<int*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Pre-allocate reduction output View WITHOUT initialization to avoid sync
    Kokkos::View<int> sum_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));
    
    // Launch kernel 1 - NO implicit fence (Kokkos 4.x+)
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) = i * 2;
        });
    
    // Launch kernel 2 - Reduce to device View (NO fence!)
    Kokkos::parallel_reduce("sum",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, int& lsum) {
            lsum += data(i);
        }, sum_device);
    // All kernels queued asynchronously!
    
    // Pre-create host mirror
    auto sum_host = Kokkos::create_mirror_view(sum_device);
    
    // Deep copy with execution space instance
    Kokkos::deep_copy(exec_space, sum_host, sum_device);
    
    // Only fence THIS execution space's stream (not entire device!)
    exec_space.fence();
    
    int expected = N * (N - 1);
    EXPECT_EQ(sum_host(), expected);
}

// Test that parallel_reduce to CudaHostPinnedSpace does NOT synchronize
// CudaHostPinnedSpace is device-accessible, so GPU can write directly
void test_reduce_to_pinned_no_sync() {
    const int N = 10000;
    
    Kokkos::Cuda exec_space;
    Kokkos::View<double*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Allocate result in CudaHostPinnedSpace WITHOUT initialization
    Kokkos::View<double, Kokkos::CudaHostPinnedSpace> sum_pinned(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum_pinned"));
    
    // NVTX marker before operations
    #ifdef KOKKOS_ENABLE_CUDA
    nvtxMarkA("Before reduce to pinned (should NOT sync)");
    #endif
    
    // Initialize data
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) = i * 2.0;
        });
    
    // Reduce to CudaHostPinnedSpace - should NOT synchronize!
    // GPU writes directly to pinned memory
    Kokkos::parallel_reduce("sum_to_pinned",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, double& lsum) {
            lsum += data(i);
        }, sum_pinned);
    // NO synchronization here! CudaHostPinnedSpace is device-accessible
    
    // NVTX marker after reduce (before fence)
    #ifdef KOKKOS_ENABLE_CUDA
    nvtxMarkA("After reduce to pinned (verify no sync occurred)");
    #endif
    
    // Can queue more work immediately - proves no sync happened
    Kokkos::parallel_for("more_work",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) *= 3.0;
        });
    
    // Only synchronize when we actually need to read the result on host
    exec_space.fence();
    
    // Now can read result directly from pinned memory on host
    double expected = static_cast<double>(N * (N - 1));
    EXPECT_DOUBLE_EQ(sum_pinned(), expected);
}

// Test multiple reductions to CudaHostPinnedSpace with no intermediate syncs
void test_multiple_reduces_to_pinned() {
    const int N = 5000;
    
    Kokkos::Cuda exec_space;
    Kokkos::View<double*> data1(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data1"), N);
    Kokkos::View<double*> data2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data2"), N);
    
    // Allocate multiple results in CudaHostPinnedSpace
    Kokkos::View<double, Kokkos::CudaHostPinnedSpace> sum1_pinned(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum1"));
    Kokkos::View<double, Kokkos::CudaHostPinnedSpace> sum2_pinned(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum2"));
    
    // Initialize both datasets
    Kokkos::parallel_for("init1",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data1(i) = i;
            data2(i) = i * 2.0;
        });
    
    // Both reductions queue asynchronously (no sync!)
    Kokkos::parallel_reduce("sum1",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, double& lsum) {
            lsum += data1(i);
        }, sum1_pinned);
    
    Kokkos::parallel_reduce("sum2",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, double& lsum) {
            lsum += data2(i);
        }, sum2_pinned);
    
    // Can use results in another kernel - still no sync!
    Kokkos::View<double*> result(Kokkos::view_alloc(Kokkos::WithoutInitializing, "result"), N);
    Kokkos::parallel_for("combine",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            // Note: Reading pinned memory from device kernel
            result(i) = data1(i) / sum1_pinned() + data2(i) / sum2_pinned();
        });
    
    // Only one sync at the end
    exec_space.fence();
    
    double expected1 = static_cast<double>(N * (N - 1) / 2);
    double expected2 = expected1 * 2.0;
    EXPECT_DOUBLE_EQ(sum1_pinned(), expected1);
    EXPECT_DOUBLE_EQ(sum2_pinned(), expected2);
}

// Verify reducing to CudaHostPinnedSpace and synchronizing via CUDA event instead of fence
void test_reduce_to_pinned_event_sync_host() {
    const int N = 10000;
    Kokkos::Cuda exec_space;

    Kokkos::View<double*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    Kokkos::View<double, Kokkos::CudaHostPinnedSpace> sum_pinned(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum_pinned"));

    // Initialize data
    Kokkos::parallel_for("init_event_host",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) { data(i) = static_cast<double>(i); });

    // Reduce to pinned (no implicit sync)
    Kokkos::parallel_reduce("sum_event_host",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, double& lsum) { lsum += data(i); },
        sum_pinned);

    // Use CUDA event to wait for completion instead of fencing the whole stream
    NVKokkos::Event reduce_done;
    reduce_done.record(exec_space);

    // Queue unrelated work; event wait will only happen when host needs the value
    Kokkos::parallel_for("post_work_event_host",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) { /* no-op */ });

    // Host waits on event (not a full device fence)
    reduce_done.synchronize();
    const double expected = static_cast<double>(N) * static_cast<double>(N - 1) / 2.0;
    EXPECT_DOUBLE_EQ(sum_pinned(), expected);
}

// Cross-stream dependency: make a second stream wait on the reduce completion event
void test_reduce_to_pinned_event_cross_stream() {
    const int N = 4096;
    auto [exec_space1, exec_space2] =
        Kokkos::Experimental::partition_space(Kokkos::Cuda{}, 1, 1);

    Kokkos::View<double*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    Kokkos::View<double*> out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out"), N);
    Kokkos::View<double, Kokkos::CudaHostPinnedSpace> sum_pinned(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum_pinned"));

    // Init on stream 1
    Kokkos::parallel_for("init_event_cross",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space1, 0, N),
        KOKKOS_LAMBDA(int i) { data(i) = 1.0; });

    // Reduce to pinned on stream 1
    Kokkos::parallel_reduce("sum_event_cross",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space1, 0, N),
        KOKKOS_LAMBDA(int i, double& lsum) { lsum += data(i); },
        sum_pinned);

    // Record event on stream 1 and make stream 2 wait for it
    NVKokkos::Event reduce_done;
    reduce_done.record(exec_space1);
    reduce_done.make_stream_wait(exec_space2);

    // Now stream 2 can safely read sum_pinned() from a device kernel
    Kokkos::parallel_for("consume_on_stream2",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space2, 0, N),
        KOKKOS_LAMBDA(int i) { out(i) = data(i) * sum_pinned(); });

    // Only fence stream 2, not the whole device
    exec_space2.fence();

    // Verify expected: sum_pinned == N, so out(i) == 1.0 * N
    auto out_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(out_h(i), static_cast<double>(N));
    }
}
#endif

TEST(ExecutionSpace, DefaultSpace) { test_default_space(); }
#ifdef KOKKOS_ENABLE_CUDA
TEST(ExecutionSpace, ExplicitCudaSpace) { test_explicit_cuda_space(); }
TEST(ExecutionSpace, DefaultSpaceNoImplicitFences) { test_default_space_no_implicit_fences(); }
TEST(ExecutionSpace, ReduceToPinnedNoSync) { test_reduce_to_pinned_no_sync(); }
TEST(ExecutionSpace, MultipleReducesToPinned) { test_multiple_reduces_to_pinned(); }
TEST(ExecutionSpace, ReduceToPinnedEventHostSync) { test_reduce_to_pinned_event_sync_host(); }
TEST(ExecutionSpace, ReduceToPinnedEventCrossStream) { test_reduce_to_pinned_event_cross_stream(); }
#endif
