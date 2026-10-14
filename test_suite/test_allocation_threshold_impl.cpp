/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_allocation_threshold_impl.cu
 * @brief CUDA implementation functions for allocation threshold tests
 * 
 * These functions contain the actual test logic with KOKKOS_LAMBDA.
 * They are implemented as non-member functions to avoid CUDA's restriction
 * on extended lambdas in private/protected member functions.
 */

#include <Kokkos_Core.hpp>
#include <string>

#ifdef KOKKOS_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
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

// Constants
constexpr size_t KOKKOS_THRESHOLD = 40000;

//==============================================================================
// Test 1: Small Allocation Behavior (<40KB)
//==============================================================================

void test_small_allocation_30kb() {
    NVTX_RANGE_PUSH("SmallAllocation_30KB");
    
    const size_t size = 30000;  // 30KB - well below threshold
    
    Kokkos::Cuda exec_space;
    
    NVTX_MARK("Before small allocation (30KB)");
    
    // Allocate View - should use cudaMalloc
    {
        Kokkos::View<char*> small_view(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "small_30KB"),
            size);
        
        NVTX_MARK("After small allocation, before kernel");
        
        // Use the allocation
        Kokkos::parallel_for("use_small",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, 100),
            KOKKOS_LAMBDA(int i) {
                if (i < size) small_view(i) = static_cast<char>(i % 256);
            });
        
        NVTX_MARK("After kernel, before deallocation");
    }
    
    NVTX_MARK("After small deallocation");
    
    exec_space.fence();
    NVTX_MARK("After explicit fence");
    
    NVTX_RANGE_POP();
}

//==============================================================================
// Test 2: Large Allocation Behavior (≥40KB)
//==============================================================================

void test_large_allocation_50kb() {
    NVTX_RANGE_PUSH("LargeAllocation_50KB");
    
    const size_t size = 50000;  // 50KB - above threshold
    
    Kokkos::Cuda exec_space;
    
    NVTX_MARK("Before large allocation (50KB)");
    
    // Allocate View - should use cudaMallocAsync (if enabled)
    {
        Kokkos::View<char*> large_view(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "large_50KB"),
            size);
        
        NVTX_MARK("After large allocation, before kernel");
        
        // Use the allocation
        Kokkos::parallel_for("use_large",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, 100),
            KOKKOS_LAMBDA(int i) {
                if (i < size) large_view(i) = static_cast<char>(i % 256);
            });
        
        NVTX_MARK("After kernel, before deallocation");
    }
    
    NVTX_MARK("After large deallocation");
    
    exec_space.fence();
    NVTX_MARK("After explicit fence");
    
    NVTX_RANGE_POP();
}

//==============================================================================
// Test 3: Boundary Crossing Behavior
//==============================================================================

void test_boundary_crossing() {
    NVTX_RANGE_PUSH("BoundaryCrossing");
    
    struct TestCase {
        size_t size;
        const char* label;
        const char* expected_path;
    };
    
    TestCase cases[] = {
        {39000, "below_39KB", "cudaMalloc"},
        {39999, "below_39999B", "cudaMalloc"},
        {40000, "exactly_40KB", "cudaMallocAsync"},
        {40001, "above_40001B", "cudaMallocAsync"},
        {41000, "above_41KB", "cudaMallocAsync"}
    };
    
    Kokkos::Cuda exec_space;
    
    for (const auto& test : cases) {
        std::string range_label = std::string("Boundary_") + test.label;
        NVTX_RANGE_PUSH(range_label.c_str());
        
        NVTX_MARK((std::string("Before_") + test.label).c_str());
        
        {
            std::string label_str(test.label);
            Kokkos::View<char*> view(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, label_str),
                test.size);
            
            NVTX_MARK((std::string("Allocated_") + test.label).c_str());
            
            // Capture variables needed in lambda
            size_t test_size = test.size;
            Kokkos::parallel_for(test.label,
                Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, 100),
                KOKKOS_LAMBDA(int i) {
                    if (i < test_size) view(i) = static_cast<char>(i % 256);
                });
            
            NVTX_MARK((std::string("Used_") + test.label).c_str());
        }
        
        NVTX_MARK((std::string("Deallocated_") + test.label).c_str());
        
        NVTX_RANGE_POP();
    }
    
    exec_space.fence();
    NVTX_MARK("BoundaryCrossing_FinalFence");
    
    NVTX_RANGE_POP();
}

//==============================================================================
// Test 4: Multiple Allocation Sizes Sweep
//==============================================================================

void test_size_sweep() {
    NVTX_RANGE_PUSH("SizeSweep");
    
    size_t sizes[] = {
        1000,      // 1KB - very small
        5000,      // 5KB - small
        10000,     // 10KB - small
        20000,     // 20KB - small
        30000,     // 30KB - small, close to threshold
        39000,     // 39KB - just below threshold
        40000,     // 40KB - exactly at threshold
        41000,     // 41KB - just above threshold
        50000,     // 50KB - above threshold
        100000,    // 100KB - well above threshold
        500000,    // 500KB - large
        1000000    // 1MB - very large
    };
    
    Kokkos::Cuda exec_space;
    
    for (size_t size : sizes) {
        std::string label = "size_" + std::to_string(size);
        std::string bytes_label = std::to_string(size) + "B";
        
        NVTX_RANGE_PUSH(label.c_str());
        
        NVTX_MARK((std::string("Before_") + bytes_label).c_str());
        
        {
            Kokkos::View<char*> view(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, label),
                size);
            
            NVTX_MARK((std::string("Allocated_") + bytes_label).c_str());
            
            int range_size = static_cast<int>(std::min(size, size_t(1000)));
            Kokkos::parallel_for(label,
                Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, range_size),
                KOKKOS_LAMBDA(int i) {
                    view(i) = static_cast<char>(i % 256);
                });
            
            NVTX_MARK((std::string("Used_") + bytes_label).c_str());
        }
        
        NVTX_MARK((std::string("Deallocated_") + bytes_label).c_str());
        
        NVTX_RANGE_POP();
    }
    
    exec_space.fence();
    NVTX_MARK("SizeSweep_FinalFence");
    
    NVTX_RANGE_POP();
}

//==============================================================================
// Test 5: Repeated Allocations at Boundary
//==============================================================================

void test_repeated_allocations_at_boundary() {
    NVTX_RANGE_PUSH("RepeatedAllocationsAtBoundary");
    
    const int num_iterations = 10;
    Kokkos::Cuda exec_space;
    
    // Test small allocations repeated
    NVTX_RANGE_PUSH("Repeated_Small_39KB");
    for (int i = 0; i < num_iterations; ++i) {
        std::string iter_label = "iter_" + std::to_string(i);
        NVTX_MARK((std::string("Small_") + iter_label + "_before").c_str());
        
        {
            Kokkos::View<char*> view(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, iter_label),
                39000);
            
            Kokkos::parallel_for(iter_label,
                Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, 100),
                KOKKOS_LAMBDA(int j) { view(j) = j; });
        }
        
        NVTX_MARK((std::string("Small_") + iter_label + "_after").c_str());
    }
    NVTX_RANGE_POP();
    
    // Test large allocations repeated
    NVTX_RANGE_PUSH("Repeated_Large_41KB");
    for (int i = 0; i < num_iterations; ++i) {
        std::string iter_label = "iter_" + std::to_string(i);
        NVTX_MARK((std::string("Large_") + iter_label + "_before").c_str());
        
        {
            Kokkos::View<char*> view(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, iter_label),
                41000);
            
            Kokkos::parallel_for(iter_label,
                Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, 100),
                KOKKOS_LAMBDA(int j) { view(j) = j; });
        }
        
        NVTX_MARK((std::string("Large_") + iter_label + "_after").c_str());
    }
    NVTX_RANGE_POP();
    
    exec_space.fence();
    NVTX_MARK("RepeatedAllocations_FinalFence");
    
    NVTX_RANGE_POP();
}

//==============================================================================
// Test 6: Synchronization Count Comparison
//==============================================================================

void test_synchronization_count_comparison() {
    NVTX_RANGE_PUSH("SynchronizationCountComparison");
    
    Kokkos::Cuda exec_space;
    
    // Phase 1: Small allocation lifecycle
    NVTX_RANGE_PUSH("Phase1_SmallAllocation");
    NVTX_MARK("Phase1_Start");
    
    {
        NVTX_MARK("Phase1_BeforeAlloc");
        Kokkos::View<char*> small(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "small_30k"),
            30000);
        NVTX_MARK("Phase1_AfterAlloc_BeforeUse");
        
        Kokkos::parallel_for("phase1_use",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, 100),
            KOKKOS_LAMBDA(int i) { small(i) = i; });
        
        NVTX_MARK("Phase1_AfterUse_BeforeDealloc");
    }
    NVTX_MARK("Phase1_AfterDealloc");
    NVTX_RANGE_POP();
    
    exec_space.fence();
    NVTX_MARK("Phase1_AfterFence");
    
    // Phase 2: Large allocation lifecycle
    NVTX_RANGE_PUSH("Phase2_LargeAllocation");
    NVTX_MARK("Phase2_Start");
    
    {
        NVTX_MARK("Phase2_BeforeAlloc");
        Kokkos::View<char*> large(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "large_50k"),
            50000);
        NVTX_MARK("Phase2_AfterAlloc_BeforeUse");
        
        Kokkos::parallel_for("phase2_use",
            Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, 100),
            KOKKOS_LAMBDA(int i) { large(i) = i; });
        
        NVTX_MARK("Phase2_AfterUse_BeforeDealloc");
    }
    NVTX_MARK("Phase2_AfterDealloc");
    NVTX_RANGE_POP();
    
    exec_space.fence();
    NVTX_MARK("Phase2_AfterFence");
    
    NVTX_RANGE_POP();
}

//==============================================================================
// Test 7: With and Without Execution Space
//==============================================================================

void test_with_and_without_execution_space() {
    NVTX_RANGE_PUSH("WithAndWithoutExecutionSpace");
    
    // Test allocation without execution space
    NVTX_RANGE_PUSH("Without_ExecSpace_Large");
    NVTX_MARK("NoExec_BeforeAlloc");
    {
        Kokkos::View<char*> view(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "no_exec_large"),
            50000);
        NVTX_MARK("NoExec_AfterAlloc");
    }
    NVTX_MARK("NoExec_AfterDealloc");
    NVTX_RANGE_POP();
    
    // Test allocation with execution space
    Kokkos::Cuda exec_space;
    
    NVTX_RANGE_PUSH("With_ExecSpace_Large");
    NVTX_MARK("WithExec_BeforeAlloc");
    {
        // Create view with execution space instance
        Kokkos::View<char*, Kokkos::CudaSpace> view(
            Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing, "with_exec_large"),
            50000);
        NVTX_MARK("WithExec_AfterAlloc");
    }
    NVTX_MARK("WithExec_AfterDealloc");
    NVTX_RANGE_POP();
    
    exec_space.fence();
    NVTX_MARK("WithAndWithout_FinalFence");
    
    NVTX_RANGE_POP();
}

//==============================================================================
// Test 8: Cuda API Call Verification
//==============================================================================

#ifdef KOKKOS_ENABLE_CUDA
bool test_cuda_api_verification() {
    NVTX_RANGE_PUSH("CudaAPIVerification");
    
    Kokkos::Cuda exec_space;
    cudaStream_t stream = exec_space.cuda_stream();
    
    // Get initial error state
    cudaError_t initial_error = cudaGetLastError();
    bool success = (initial_error == cudaSuccess);
    
    // Small allocation
    NVTX_MARK("API_SmallAlloc_Before");
    {
        Kokkos::View<char*> small(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "api_small"),
            30000);
        NVTX_MARK("API_SmallAlloc_After");
        
        // Verify no errors
        success = success && (cudaGetLastError() == cudaSuccess);
    }
    NVTX_MARK("API_SmallDealloc_After");
    
    // Large allocation
    NVTX_MARK("API_LargeAlloc_Before");
    {
        Kokkos::View<char*> large(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "api_large"),
            50000);
        NVTX_MARK("API_LargeAlloc_After");
        
        // Verify no errors
        success = success && (cudaGetLastError() == cudaSuccess);
    }
    NVTX_MARK("API_LargeDealloc_After");
    
    exec_space.fence();
    success = success && (cudaGetLastError() == cudaSuccess);
    
    NVTX_RANGE_POP();
    
    return success;
}
#endif
