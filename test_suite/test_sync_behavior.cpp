/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#ifdef KOKKOS_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

// These tests verify the synchronization behavior of Kokkos parallel operations
// They are designed to be profiled with nsys and analyzed with extract_timeline.py

// Helper macro to add NVTX markers for profiling
#ifdef KOKKOS_ENABLE_CUDA
#define NVTX_MARK(msg) nvtxMarkA(msg)
#define NVTX_RANGE_PUSH(msg) nvtxRangePushA(msg)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_MARK(msg)
#define NVTX_RANGE_PUSH(msg)
#define NVTX_RANGE_POP()
#endif

// Test 1: parallel_for should NOT synchronize in release builds
void test_parallel_for_no_sync() {
    NVTX_RANGE_PUSH("ParallelForNoSync");
    
    const int N = 100000;
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<int*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    NVTX_MARK("Before parallel_for 1");
    
    // Launch first kernel - should not synchronize
    Kokkos::parallel_for("kernel1",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) = i;
        });
    
    NVTX_MARK("After parallel_for 1, before parallel_for 2");
    
    // Launch second kernel immediately - should not synchronize
    Kokkos::parallel_for("kernel2",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) *= 2;
        });
    
    NVTX_MARK("After parallel_for 2, before explicit fence");
    
    // Explicit synchronization
    exec_space.fence();
    
    NVTX_MARK("After explicit fence");
    
    NVTX_RANGE_POP();
}

// Test 2: parallel_reduce to host scalar SHOULD synchronize
void test_parallel_reduce_to_host_sync() {
    NVTX_RANGE_PUSH("ParallelReduceToHostSync");
    
    const int N = 100000;
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<int*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Initialize data
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) = 1;
        });
    
    NVTX_MARK("Before parallel_reduce to host scalar");
    
    // Reduce to host scalar - SHOULD synchronize
    int sum = 0;
    Kokkos::parallel_reduce("reduce_to_host",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, int& lsum) {
            lsum += data(i);
        }, sum);
    
    NVTX_MARK("After parallel_reduce to host scalar");
    
    EXPECT_EQ(sum, N);
    
    NVTX_RANGE_POP();
}

// Test 3: parallel_reduce to device View
// NOTE: This test WILL show synchronization due to View allocation, not parallel_reduce itself.
// When Views are allocated with execution space, Kokkos uses cudaMallocAsync + cudaStreamSynchronize
// even for device-accessible Views. This is a known Kokkos behavior.
void test_parallel_reduce_to_device_no_sync() {
    NVTX_RANGE_PUSH("ParallelReduceToDeviceNoSync");
    
    const int N = 100000;
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<int*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Initialize data
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) = 1;
        });
    
    NVTX_MARK("Before parallel_reduce to device view");
    
    // Reduce to device View - parallel_reduce itself does NOT synchronize,
    // but View allocation may cause cudaStreamSynchronize if cudaMallocAsync is used
    Kokkos::View<int> sum_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));
    Kokkos::parallel_reduce("reduce_to_device",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, int& lsum) {
            lsum += data(i);
        }, sum_device);
    
    NVTX_MARK("After parallel_reduce to device view, before another kernel");
    
    // Launch another kernel - View allocation here may also cause synchronization
    Kokkos::View<int*> result_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "result"), N);
    Kokkos::parallel_for("use_result",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            result_device(i) = data(i) * 2;
        });
    
    NVTX_MARK("After second kernel, before explicit fence");
    
    // Explicit synchronization
    exec_space.fence();
    
    NVTX_MARK("After explicit fence");
    
    // Copy result to host to verify
    auto sum_host = Kokkos::create_mirror_view(sum_device);
    Kokkos::deep_copy(exec_space, sum_host, sum_device);
    exec_space.fence();
    
    EXPECT_EQ(sum_host(), N);
    
    NVTX_RANGE_POP();
}

#ifdef KOKKOS_ENABLE_CUDA
#include "../nvkokkos_helpers.hpp"

// Test 3b: parallel_reduce to device View using StreamOrderedView (should NOT synchronize)
// This test uses StreamOrderedView which bypasses Kokkos's allocation synchronization,
// demonstrating that parallel_reduce itself does not synchronize when using zero-sync allocation.
void test_parallel_reduce_to_device_no_sync_stream_ordered() {
    NVTX_RANGE_PUSH("ParallelReduceToDeviceNoSyncStreamOrdered");
    
    const int N = 100000;
    Kokkos::Cuda exec_space;
    cudaStream_t stream = exec_space.cuda_stream();
    
    // Use StreamOrderedView to avoid allocation synchronization
    // Pre-allocate all Views before the test markers to avoid allocation sync during test
    NVKokkos::StreamOrderedView<int*> data(stream, N);
    NVKokkos::StreamOrderedView<int*> sum_device_array(stream, 1);  // 1-element array for scalar result
    NVKokkos::StreamOrderedView<int*> result_device(stream, N);
    
    // Extract views for lambda capture (StreamOrderedView is non-copyable)
    auto data_view = data.view();
    auto sum_device_array_view = sum_device_array.view();
    auto result_device_view = result_device.view();
    
    // Create scalar View wrapper from 1-element array for parallel_reduce
    // Use Unmanaged to wrap the existing memory without taking ownership
    Kokkos::View<int, Kokkos::CudaSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        sum_device(sum_device_array_view.data());
    
    // Initialize data
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data_view(i) = 1;
        });
    
    NVTX_MARK("Before parallel_reduce to device view (stream ordered)");
    
    // Reduce to device View - should NOT synchronize (no View allocation sync)
    Kokkos::parallel_reduce("reduce_to_device",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, int& lsum) {
            lsum += data_view(i);
        }, sum_device);
    
    NVTX_MARK("After parallel_reduce to device view, before another kernel (stream ordered)");
    
    // Launch another kernel - should queue without sync (no View allocation sync)
    Kokkos::parallel_for("use_result",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            result_device_view(i) = data_view(i) * 2;
        });
    
    NVTX_MARK("After second kernel, before explicit fence (stream ordered)");
    
    // Explicit synchronization
    exec_space.fence();
    
    NVTX_MARK("After explicit fence (stream ordered)");
    
    // Copy result to host to verify
    Kokkos::View<int, Kokkos::HostSpace> sum_host("sum_host");
    Kokkos::deep_copy(exec_space, sum_host, sum_device);
    exec_space.fence();
    
    EXPECT_EQ(sum_host(), N);
    
    NVTX_RANGE_POP();
}
#endif

// Test 4: parallel_scan with scalar return SHOULD synchronize
void test_parallel_scan_to_scalar_sync() {
    NVTX_RANGE_PUSH("ParallelScanToScalarSync");
    
    const int N = 1000;
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<int*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    NVTX_MARK("Before parallel_scan to scalar");
    
    // Scan with scalar return - SHOULD synchronize
    int total = 0;
    Kokkos::parallel_scan("scan_to_scalar",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, int& update, bool final) {
            if (final) data(i) = update;
            update += 1;
        }, total);
    
    NVTX_MARK("After parallel_scan to scalar");
    
    EXPECT_EQ(total, N);
    
    NVTX_RANGE_POP();
}

// Test 5: parallel_scan with View return should NOT immediately synchronize
void test_parallel_scan_to_view_no_sync() {
    NVTX_RANGE_PUSH("ParallelScanToViewNoSync");
    
    const int N = 1000;
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<int*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    Kokkos::View<int> total_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "total"));
    
    NVTX_MARK("Before parallel_scan to view");
    
    // Scan with View return - should NOT immediately synchronize
    Kokkos::parallel_scan("scan_to_view",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, int& update, bool final) {
            if (final) data(i) = update;
            update += 1;
        }, total_device);
    
    NVTX_MARK("After parallel_scan to view, before another kernel");
    
    // Launch another kernel - should queue without sync
    Kokkos::parallel_for("use_scan_result",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) *= 2;
        });
    
    NVTX_MARK("After second kernel, before explicit fence");
    
    // Explicit synchronization
    exec_space.fence();
    
    NVTX_MARK("After explicit fence");
    
    // Copy result to host to verify
    auto total_host = Kokkos::create_mirror_view(total_device);
    Kokkos::deep_copy(exec_space, total_host, total_device);
    exec_space.fence();
    
    EXPECT_EQ(total_host(), N);
    
    NVTX_RANGE_POP();
}

// Test 6: deep_copy without execution space SHOULD synchronize
void test_deep_copy_without_exec_space_sync() {
    NVTX_RANGE_PUSH("DeepCopyWithoutExecSpaceSync");
    
    const int N = 100000;
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<double*, Kokkos::CudaSpace> d_data("device", N);
    Kokkos::View<double*, Kokkos::HostSpace> h_data("host", N);
    
    // Initialize device data
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            d_data(i) = i * 1.5;
        });
    
    NVTX_MARK("Before deep_copy without exec space");
    
    // Deep copy without execution space - SHOULD synchronize
    Kokkos::deep_copy(h_data, d_data);
    
    NVTX_MARK("After deep_copy without exec space");
    
    EXPECT_NEAR(h_data(100), 150.0, 0.01);
    
    NVTX_RANGE_POP();
}

// Test 7: deep_copy with execution space should NOT immediately synchronize
void test_deep_copy_with_exec_space_no_sync() {
    NVTX_RANGE_PUSH("DeepCopyWithExecSpaceNoSync");
    
    const int N = 100000;
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<double*, Kokkos::CudaSpace> d_data("device", N);
    Kokkos::View<double*, Kokkos::HostSpace> h_data("host", N);
    
    // Initialize device data
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            d_data(i) = i * 1.5;
        });
    
    NVTX_MARK("Before deep_copy with exec space");
    
    // Deep copy with execution space - should NOT immediately synchronize
    Kokkos::deep_copy(exec_space, h_data, d_data);
    
    NVTX_MARK("After deep_copy with exec space, before another kernel");
    
    // Launch another kernel - should queue without sync
    Kokkos::parallel_for("modify",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            d_data(i) *= 2.0;
        });
    
    NVTX_MARK("After second kernel, before explicit fence");
    
    // Explicit synchronization
    exec_space.fence();
    
    NVTX_MARK("After explicit fence");
    
    EXPECT_NEAR(h_data(100), 150.0, 0.01);
    
    NVTX_RANGE_POP();
}

// Test 8: Multiple operations without sync
void test_multiple_ops_no_sync() {
    NVTX_RANGE_PUSH("MultipleOpsNoSync");
    
    const int N = 100000;
    Kokkos::DefaultExecutionSpace exec_space;
    
    Kokkos::View<int*> data1(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data1"), N);
    Kokkos::View<int*> data2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data2"), N);
    Kokkos::View<long long> sum1_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum1"));
    Kokkos::View<long long> sum2_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum2"));
    
    NVTX_MARK("Starting multiple operations");
    
    // Operation 1: parallel_for
    Kokkos::parallel_for("init1",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data1(i) = i;
        });
    
    NVTX_MARK("After init1");
    
    // Operation 2: parallel_reduce to device view
    Kokkos::parallel_reduce("reduce1",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, long long& lsum) {
            lsum += data1(i);
        }, sum1_device);
    
    NVTX_MARK("After reduce1");
    
    // Operation 3: Another parallel_for
    Kokkos::parallel_for("init2",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data2(i) = i * 2;
        });
    
    NVTX_MARK("After init2");
    
    // Operation 4: Another parallel_reduce to device view
    Kokkos::parallel_reduce("reduce2",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, long long& lsum) {
            lsum += data2(i);
        }, sum2_device);
    
    NVTX_MARK("After reduce2, before explicit fence");
    
    // Only now synchronize
    exec_space.fence();
    
    NVTX_MARK("After explicit fence");
    
    // Verify results
    auto sum1_host = Kokkos::create_mirror_view(sum1_device);
    auto sum2_host = Kokkos::create_mirror_view(sum2_device);
    Kokkos::deep_copy(exec_space, sum1_host, sum1_device);
    Kokkos::deep_copy(exec_space, sum2_host, sum2_device);
    exec_space.fence();
    
    EXPECT_EQ(sum1_host(), (long long)N * (N - 1) / 2);
    EXPECT_EQ(sum2_host(), (long long)N * (N - 1));
    
    NVTX_RANGE_POP();
}

// Test 9: make_view_filled with exec space should NOT synchronize
// Pre-allocates views so allocation sync doesn't pollute the measurement.
// The deep_copy(exec, view, value) overload is async on the exec space's stream.
void test_make_view_filled_no_sync() {
    NVTX_RANGE_PUSH("MakeViewFilledNoSync");

    const int N = 100000;
    Kokkos::Cuda exec_space;

    // Pre-allocate views outside the measurement window so that
    // cudaMallocAsync allocation sync doesn't appear in the profiling region.
    auto filled = NVKokkos::make_view_uninitialized<double*>("filled", N);
    Kokkos::View<double*> result(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "result"), N);
    exec_space.fence();

    NVTX_MARK("Before deep_copy fill");

    // Async fill — should NOT synchronize
    Kokkos::deep_copy(exec_space, filled, 3.14);

    NVTX_MARK("After deep_copy fill, before parallel_for");

    // Launch a kernel immediately — should queue without sync
    Kokkos::parallel_for("use_filled",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            result(i) = filled(i) * 2.0;
        });

    NVTX_MARK("After parallel_for, before explicit fence");

    exec_space.fence();

    NVTX_MARK("After explicit fence");

    auto h_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    EXPECT_DOUBLE_EQ(h_result(0), 6.28);
    EXPECT_DOUBLE_EQ(h_result(N-1), 6.28);

    NVTX_RANGE_POP();
}

// Now wrap each test function in a TEST() macro
TEST(SyncBehavior, ParallelForNoSync) {
    test_parallel_for_no_sync();
}

TEST(SyncBehavior, ParallelReduceToHostSync) {
    test_parallel_reduce_to_host_sync();
}

TEST(SyncBehavior, ParallelReduceToDeviceNoSync) {
    test_parallel_reduce_to_device_no_sync();
}

#ifdef KOKKOS_ENABLE_CUDA
TEST(SyncBehavior, ParallelReduceToDeviceNoSyncStreamOrdered) {
    test_parallel_reduce_to_device_no_sync_stream_ordered();
}
#endif

TEST(SyncBehavior, ParallelScanToScalarSync) {
    test_parallel_scan_to_scalar_sync();
}

TEST(SyncBehavior, ParallelScanToViewNoSync) {
    test_parallel_scan_to_view_no_sync();
}

TEST(SyncBehavior, DeepCopyWithoutExecSpaceSync) {
    test_deep_copy_without_exec_space_sync();
}

TEST(SyncBehavior, DeepCopyWithExecSpaceNoSync) {
    test_deep_copy_with_exec_space_no_sync();
}

TEST(SyncBehavior, MultipleOpsNoSync) {
    test_multiple_ops_no_sync();
}

TEST(SyncBehavior, MakeViewFilledNoSync) {
    test_make_view_filled_no_sync();
}
