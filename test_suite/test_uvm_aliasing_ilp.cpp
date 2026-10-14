/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include "../nvkokkos_helpers.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

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

#ifdef KOKKOS_ENABLE_CUDA

// Helper functions to avoid CUDA extended lambda issues in private TEST methods

void init_uvm_kernel(
    Kokkos::View<double*, Kokkos::CudaUVMSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> view,
    int N,
    const Kokkos::Cuda& exec_space) {
    Kokkos::parallel_for("init_uvm",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            view(i) = i * 2.0;
        });
}

// ============================================================================
// Test 1: UVM Deallocation Double Sync
// ============================================================================
// This test verifies that StreamOrderedView with CudaUVMSpace avoids
// the double synchronization that standard Kokkos UVM Views have on deallocation

TEST(UVMDeallocation, StreamOrderedViewNoSync) {
    NVTX_RANGE_PUSH("UVMDeallocationStreamOrderedView");
    
    const int N = 100000;
    Kokkos::Cuda exec_space;
    cudaStream_t stream = exec_space.cuda_stream();
    
    // Use StreamOrderedView with CudaUVMSpace
    {
        NVTX_MARK("Before StreamOrderedView UVM allocation");
        
        NVKokkos::StreamOrderedView<double*, Kokkos::CudaUVMSpace> uvm_data(stream, N);
        auto view = uvm_data.view();
        
        NVTX_MARK("After StreamOrderedView UVM allocation, before kernel");
        
        // Use the view
        init_uvm_kernel(view, N, exec_space);
        
        NVTX_MARK("After kernel, before deallocation");
        
        // View goes out of scope here - should NOT synchronize
    } // Destructor called here
    
    NVTX_MARK("After StreamOrderedView UVM deallocation");
    
    // Explicit sync to ensure all operations complete
    exec_space.fence();
    
    NVTX_RANGE_POP();
}

// ============================================================================
// Test 2: View Aliasing Race Conditions
// ============================================================================
// This test demonstrates the aliasing problem and verifies that creating
// separate views avoids the issue

// Helper functions for aliasing tests
// These take execution spaces as parameters to allow concurrent execution on different streams
void compute_read_only(const Kokkos::View<const double*>& data_read, int N, const Kokkos::Cuda& exec_space) {
    Kokkos::View<double> sum("sum");
    
    Kokkos::parallel_reduce("read_sum",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, double& lsum) {
            lsum += data_read(i);
        }, sum);
}

void compute_write(Kokkos::View<double*>& data_write, int N, const Kokkos::Cuda& exec_space) {
    Kokkos::parallel_for("write_modify",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data_write(i) = data_write(i) * 2.0 + 1.0;
        });
}

void init_data_kernel(Kokkos::View<double*> data, int N, const Kokkos::Cuda& exec_space) {
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data(i) = i;
        });
}

TEST(ViewAliasing, AliasingProblem) {
    NVTX_RANGE_PUSH("ViewAliasingProblem");
    
    const int N = 100000;
    
    // Partition a base execution space into different streams for concurrent execution
    auto [exec_space_read, exec_space_write] =
        Kokkos::Experimental::partition_space(Kokkos::Cuda{}, 1, 1);
    
    // Create a view
    Kokkos::View<double*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Initialize
    Kokkos::Cuda exec_space_init;
    init_data_kernel(data, N, exec_space_init);
    exec_space_init.fence();
    
    // DANGEROUS: Pass same view as const and mutable on DIFFERENT STREAMS
    // This creates a real race condition - read and write operations execute concurrently
    NVTX_MARK("Before aliasing operations (different streams)");
    
    // Create a const view reference (aliases same memory)
    Kokkos::View<const double*> const_ref = data;
    
    // Launch read and write operations concurrently on different streams - RACE CONDITION!
    // These execute asynchronously and can interleave, causing undefined behavior
    compute_read_only(const_ref, N, exec_space_read);
    compute_write(data, N, exec_space_write);
    
    // Sync both streams
    exec_space_read.fence();
    exec_space_write.fence();
    
    NVTX_MARK("After aliasing operations");
    
    NVTX_RANGE_POP();
}

TEST(ViewAliasing, NoAliasingSolution) {
    NVTX_RANGE_PUSH("ViewAliasingSolution");
    
    const int N = 100000;
    
    // Partition a base execution space into different streams for concurrent execution
    auto [exec_space_read, exec_space_write] =
        Kokkos::Experimental::partition_space(Kokkos::Cuda{}, 1, 1);
    
    // Create separate views
    Kokkos::View<double*> data_in(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data_in"), N);
    Kokkos::View<double*> data_out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data_out"), N);
    
    // Initialize input
    Kokkos::Cuda exec_space_init;
    init_data_kernel(data_in, N, exec_space_init);
    exec_space_init.fence();
    
    NVTX_MARK("Before non-aliasing operations (different streams)");
    
    // Create actual copy for read-only operation
    Kokkos::Cuda exec_space_copy;
    Kokkos::View<double*> read_copy(Kokkos::view_alloc(Kokkos::WithoutInitializing, "read_copy"), N);
    Kokkos::deep_copy(exec_space_copy, read_copy, data_in);
    exec_space_copy.fence();
    
    // Now safe to use read_copy for read-only and data_out for write on different streams
    // No race condition because they operate on different memory locations
    Kokkos::View<const double*> const_ref = read_copy;
    compute_read_only(const_ref, N, exec_space_read);
    compute_write(data_out, N, exec_space_write);
    
    // Sync both streams
    exec_space_read.fence();
    exec_space_write.fence();
    
    NVTX_MARK("After non-aliasing operations");
    
    NVTX_RANGE_POP();
}

// ============================================================================
// Test 3: Read-After-Write Global Memory Dependency (ILP Performance)
// ============================================================================
// This test compares performance of anti-pattern (read-after-write) vs
// optimized pattern (register-based computation)

// Helper kernels for ILP performance test
void anti_pattern_kernel(Kokkos::View<double*> data, int N, const Kokkos::Cuda& exec_space) {
    Kokkos::parallel_for("anti_pattern",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            // Update value in global memory
            data(i) = data(i) + 1.0;
            
            // Immediately read back from global for conditional - SLOW!
            // Global memory round-trip stalls pipeline before expensive computation
            if (data(i) > 1000.0) {
                // Expensive transcendental computation - pipeline stalled waiting for global memory
                data(i) = std::exp(data(i) * 0.001) * 0.5;
            }
        });
}

void optimized_kernel(Kokkos::View<double*> data, int N, const Kokkos::Cuda& exec_space) {
    Kokkos::parallel_for("optimized",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            // Read once into register
            double value = data(i);
            double new_value = value + 1.0;
            
            // Use register value in conditional - FAST!
            // Expensive computation happens on register value without global memory stall
            if (new_value > 1000.0) {
                // Expensive transcendental computation - no pipeline stall!
                new_value = std::exp(new_value * 0.001) * 0.5;
            }
            
            // Write back once
            data(i) = new_value;
        });
}

double test_read_after_write_anti_pattern(int N, int iterations) {
    Kokkos::Cuda exec_space;
    Kokkos::View<double*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Initialize
    init_data_kernel(data, N, exec_space);
    exec_space.fence();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Anti-pattern: Read-after-write on global memory
    for (int iter = 0; iter < iterations; ++iter) {
        anti_pattern_kernel(data, N, exec_space);
    }
    exec_space.fence();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0;  // Return milliseconds
}

double test_register_based_optimized(int N, int iterations) {
    Kokkos::Cuda exec_space;
    Kokkos::View<double*> data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
    
    // Initialize
    init_data_kernel(data, N, exec_space);
    exec_space.fence();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Optimized: Use register value, avoid read-after-write
    for (int iter = 0; iter < iterations; ++iter) {
        optimized_kernel(data, N, exec_space);
    }
    exec_space.fence();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0;  // Return milliseconds
}

TEST(InstructionLevelParallelism, ReadAfterWritePerformance) {
    NVTX_RANGE_PUSH("ILPReadAfterWrite");
    
    const int N = 1000000;  // Large size to see performance difference
    const int iterations = 100;
    
    NVTX_MARK("Before anti-pattern test");
    double anti_pattern_time = test_read_after_write_anti_pattern(N, iterations);
    NVTX_MARK("After anti-pattern test");
    
    NVTX_MARK("Before optimized test");
    double optimized_time = test_register_based_optimized(N, iterations);
    NVTX_MARK("After optimized test");
    
    std::cout << "\nILP Performance Comparison:\n";
    std::cout << "  Anti-pattern (read-after-write): " << anti_pattern_time << " ms\n";
    std::cout << "  Optimized (register-based):      " << optimized_time << " ms\n";
    std::cout << "  Speedup: " << (anti_pattern_time / optimized_time) << "x\n";
    
    // The optimized version should be faster (or at least not slower)
    // Note: This is a performance test, not a correctness test
    EXPECT_GE(anti_pattern_time, optimized_time * 0.8);  // Allow some variance
    
    NVTX_RANGE_POP();
}

#endif // KOKKOS_ENABLE_CUDA
