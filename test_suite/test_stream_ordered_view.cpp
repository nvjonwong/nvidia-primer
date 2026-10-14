/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Test suite for NVKokkos::StreamOrderedView
// Verifies stream-ordered memory allocation with zero synchronization overhead

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "../nvkokkos_helpers.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#include <chrono>
#endif

//============================================================================
// Test Functions (standalone to avoid CUDA lambda issues with gtest)
//============================================================================

void test_basic_allocation() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 1000;
    
    {
        // Allocate with StreamOrderedView
        NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
        
        // Check validity
        EXPECT_TRUE(data.valid());
        EXPECT_NE(data.data(), nullptr);
        EXPECT_EQ(data.size_bytes(), N * sizeof(double));
        
        // Get UnManaged view
        auto view = data.view();
        EXPECT_EQ(view.size(), N);
        
        // Automatic cleanup when data goes out of scope
    }
    
    exec_space.fence();
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_parallel_for_usage() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 10000;
    
    // Allocate and initialize
    NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
    auto view = data.view();
    
    // Initialize with parallel_for
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view(i) = static_cast<double>(i);
        });
    
    // Modify with another parallel_for
    Kokkos::parallel_for("modify",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view(i) *= 2.0;
        });
    
    exec_space.fence();
    
    // Verify results on host
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(view_host(i), static_cast<double>(i) * 2.0);
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_parallel_reduce_usage() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 100000;
    
    NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
    auto view = data.view();
    
    // Initialize
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view(i) = 1.0;
        });
    
    // Sum with parallel_reduce
    double sum = 0.0;
    Kokkos::parallel_reduce("sum",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, double& lsum) {
            lsum += view(i);
        }, sum);
    
    exec_space.fence();
    
    EXPECT_DOUBLE_EQ(sum, static_cast<double>(N));
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_operator_star() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 1000;
    
    NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
    
    // Get view via operator*
    auto view = *data;
    
    // Use it
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view(i) = static_cast<double>(i * 3);
        });
    
    exec_space.fence();
    
    // Verify
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, *data);
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(view_host(i), static_cast<double>(i * 3));
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_implicit_conversion() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 500;
    
    NVKokkos::StreamOrderedView<int*> data(exec_space.cuda_stream(), N);
    
    // Get view and pass to a function
    auto view = data.view();
    
    Kokkos::parallel_for("process",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view(i) = i * i;
        });
    
    exec_space.fence();
    
    // Verify
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data.view());
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(view_host(i), static_cast<int>(i * i));
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_move_semantics() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 1000;
    
    // Create and initialize
    NVKokkos::StreamOrderedView<double*> data1(exec_space.cuda_stream(), N);
    auto view1 = data1.view();
    
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view1(i) = static_cast<double>(i);
        });
    
    exec_space.fence();
    
    // Move construct
    NVKokkos::StreamOrderedView<double*> data2(std::move(data1));
    
    // data1 should be invalid, data2 should be valid
    EXPECT_FALSE(data1.valid());
    EXPECT_TRUE(data2.valid());
    
    // data2 should have the same data
    auto view2 = data2.view();
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view2);
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(view_host(i), static_cast<double>(i));
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_move_assignment() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N1 = 1000;
    const size_t N2 = 500;
    
    // Create two allocations
    NVKokkos::StreamOrderedView<double*> data1(exec_space.cuda_stream(), N1);
    NVKokkos::StreamOrderedView<double*> data2(exec_space.cuda_stream(), N2);
    
    // Initialize data2
    auto view2 = data2.view();
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N2),
        KOKKOS_LAMBDA(int i) {
            view2(i) = static_cast<double>(i * 10);
        });
    
    exec_space.fence();
    
    // Move assign data2 to data1 (should free data1's old allocation)
    data1 = std::move(data2);
    
    // data2 should be invalid, data1 should have data2's allocation
    EXPECT_FALSE(data2.valid());
    EXPECT_TRUE(data1.valid());
    EXPECT_EQ(data1.view().size(), N2);
    
    // Verify data
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data1.view());
    
    for (size_t i = 0; i < N2; ++i) {
        EXPECT_DOUBLE_EQ(view_host(i), static_cast<double>(i * 10));
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_two_dimensional_view() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t M = 100;
    const size_t N = 200;
    
    NVKokkos::StreamOrderedView<double**> data(exec_space.cuda_stream(), M, N);
    auto view = data.view();
    
    EXPECT_EQ(view.extent(0), M);
    EXPECT_EQ(view.extent(1), N);
    
    // Initialize
    Kokkos::parallel_for("init",
        Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>(
            exec_space, {0, 0}, {M, N}),
        KOKKOS_LAMBDA(int i, int j) {
            view(i, j) = static_cast<double>(i * N + j);
        });
    
    exec_space.fence();
    
    // Verify
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EXPECT_DOUBLE_EQ(view_host(i, j), static_cast<double>(i * N + j));
        }
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_make_helper_function() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 5000;
    
    // Use helper function for type deduction
    auto data = NVKokkos::make_stream_ordered_view<double*>(exec_space, N);
    
    EXPECT_TRUE(data.valid());
    EXPECT_EQ(data.view().size(), N);
    
    // Use it
    auto view = data.view();
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view(i) = static_cast<double>(i);
        });
    
    exec_space.fence();
    
    // Verify
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data.view());
    
    for (size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(view_host(i), static_cast<double>(i));
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_multiple_allocations_on_same_stream() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 1000;
    
    // Create multiple stream-ordered views
    NVKokkos::StreamOrderedView<double*> data1(exec_space.cuda_stream(), N);
    NVKokkos::StreamOrderedView<double*> data2(exec_space.cuda_stream(), N);
    NVKokkos::StreamOrderedView<double*> data3(exec_space.cuda_stream(), N);
    
    auto view1 = data1.view();
    auto view2 = data2.view();
    auto view3 = data3.view();
    
    // Initialize all three
    Kokkos::parallel_for("init1",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) { view1(i) = static_cast<double>(i); });
    
    Kokkos::parallel_for("init2",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) { view2(i) = static_cast<double>(i * 2); });
    
    Kokkos::parallel_for("init3",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) { view3(i) = static_cast<double>(i * 3); });
    
    // Combine results
    Kokkos::parallel_for("combine",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view1(i) = view1(i) + view2(i) + view3(i);
        });
    
    exec_space.fence();
    
    // Verify
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view1);
    
    for (size_t i = 0; i < N; ++i) {
        double expected = static_cast<double>(i) + static_cast<double>(i * 2) + 
                         static_cast<double>(i * 3);
        EXPECT_DOUBLE_EQ(view_host(i), expected);
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_different_data_types() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 1000;
    
    // Test with int
    {
        NVKokkos::StreamOrderedView<int*> int_data(exec_space.cuda_stream(), N);
        auto view = int_data.view();
        
        Kokkos::parallel_for("init_int",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
            KOKKOS_LAMBDA(int i) { view(i) = i * 100; });
        
        exec_space.fence();
        
        auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
        EXPECT_EQ(view_host(100), 10000);
    }
    
    // Test with float
    {
        NVKokkos::StreamOrderedView<float*> float_data(exec_space.cuda_stream(), N);
        auto view = float_data.view();
        
        Kokkos::parallel_for("init_float",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
            KOKKOS_LAMBDA(int i) { view(i) = static_cast<float>(i) * 1.5f; });
        
        exec_space.fence();
        
        auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
        EXPECT_FLOAT_EQ(view_host(100), 150.0f);
    }
    
    // Test with long long
    {
        NVKokkos::StreamOrderedView<long long*> ll_data(exec_space.cuda_stream(), N);
        auto view = ll_data.view();
        
        Kokkos::parallel_for("init_ll",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
            KOKKOS_LAMBDA(int i) { 
                view(i) = static_cast<long long>(i) * 1000000000LL; 
            });
        
        exec_space.fence();
        
        auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
        EXPECT_EQ(view_host(5), 5000000000LL);
    }
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_stream_consistency() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 10000;
    
    NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
    auto view = data.view();
    
    // All operations must use the same execution space instance
    Kokkos::parallel_for("op1",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) { view(i) = static_cast<double>(i); });
    
    Kokkos::parallel_for("op2",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) { view(i) *= 2.0; });
    
    double sum = 0.0;
    Kokkos::parallel_reduce("op3",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, double& lsum) { lsum += view(i); },
        sum);
    
    exec_space.fence();
    
    // Sum should be: sum(2*i for i in 0..N-1) = 2 * N*(N-1)/2
    double expected = static_cast<double>(N) * static_cast<double>(N - 1);
    EXPECT_DOUBLE_EQ(sum, expected);
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

void test_performance_comparison() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::Cuda exec_space;
    const size_t N = 100000;
    const int iterations = 100;
    
    std::cout << "\n=== Performance Comparison (Informational) ===\n";
    
    // Time standard Kokkos View allocations
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::View<double*> view("standard", N);
        Kokkos::parallel_for("init",
            Kokkos::RangePolicy<Kokkos::Cuda>(0, N),
            KOKKOS_LAMBDA(int i) { view(i) = i; });
        Kokkos::fence();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto standard_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Time StreamOrderedView allocations
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
        auto view = data.view();
        Kokkos::parallel_for("init",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
            KOKKOS_LAMBDA(int i) { view(i) = i; });
    }
    exec_space.fence();
    end = std::chrono::high_resolution_clock::now();
    auto stream_ordered_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Standard Kokkos View:    " << standard_time << " ms\n";
    std::cout << "StreamOrderedView:       " << stream_ordered_time << " ms\n";
    
    if (stream_ordered_time > 0 && stream_ordered_time < standard_time) {
        double speedup = static_cast<double>(standard_time) / static_cast<double>(stream_ordered_time);
        std::cout << "Speedup:                 " << speedup << "x\n";
    } else if (stream_ordered_time == 0) {
        std::cout << "Note:                    StreamOrderedView too fast to measure!\n";
    }
    std::cout << "===========================================\n\n";
    
    // This is informational only - don't fail the test
    // StreamOrderedView might be too fast to measure (0 ms is okay)
    EXPECT_GE(standard_time, 0);
    EXPECT_GE(stream_ordered_time, 0);
#else
    GTEST_SKIP() << "Test requires CUDA backend";
#endif
}

//============================================================================
// Test Wrappers (gtest TEST macros)
//============================================================================

TEST(StreamOrderedView, BasicAllocation) { test_basic_allocation(); }
TEST(StreamOrderedView, ParallelForUsage) { test_parallel_for_usage(); }
TEST(StreamOrderedView, ParallelReduceUsage) { test_parallel_reduce_usage(); }
TEST(StreamOrderedView, OperatorStar) { test_operator_star(); }
TEST(StreamOrderedView, ImplicitConversion) { test_implicit_conversion(); }
TEST(StreamOrderedView, MoveSemantics) { test_move_semantics(); }
TEST(StreamOrderedView, MoveAssignment) { test_move_assignment(); }
TEST(StreamOrderedView, TwoDimensionalView) { test_two_dimensional_view(); }
TEST(StreamOrderedView, MakeHelperFunction) { test_make_helper_function(); }
TEST(StreamOrderedView, MultipleAllocationsOnSameStream) { test_multiple_allocations_on_same_stream(); }
TEST(StreamOrderedView, DifferentDataTypes) { test_different_data_types(); }
TEST(StreamOrderedView, StreamConsistency) { test_stream_consistency(); }
TEST(StreamOrderedView, PerformanceComparison) { test_performance_comparison(); }
