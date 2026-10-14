/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_allocation_threshold.cpp
 * @brief Tests for Kokkos 40KB allocation threshold behavior
 * 
 * These tests verify the synchronization behavior differences between:
 * - Small allocations (<40KB) using cudaMalloc/cudaFree
 * - Large allocations (≥40KB) using cudaMallocAsync/cudaFreeAsync
 * 
 * Tests are designed to be profiled with nsys to create golden baselines
 * for tracking changes in future Kokkos versions.
 * 
 * Usage:
 *   # Run tests
 *   ./kokkos_tests --gtest_filter="AllocationThreshold.*"
 *   
 *   # Profile to create baseline
 *   nsys profile --trace=cuda,nvtx --output=threshold_baseline_5.0.1.nsys-rep \
 *       ./kokkos_tests --gtest_filter="AllocationThreshold.*"
 *   
 *   # Extract timeline
 *   python3 ../scripts/extract_timeline.py threshold_baseline_5.0.1.nsys-rep \
 *       > threshold_baseline_5.0.1.txt
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <string>

#ifdef KOKKOS_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#endif

// Forward declarations of implementation functions in .cu file
void test_small_allocation_30kb();
void test_large_allocation_50kb();
void test_boundary_crossing();
void test_size_sweep();
void test_repeated_allocations_at_boundary();
void test_synchronization_count_comparison();
void test_with_and_without_execution_space();
#ifdef KOKKOS_ENABLE_CUDA
bool test_cuda_api_verification();
#endif

// Constants matching Kokkos source code
constexpr size_t KOKKOS_THRESHOLD = 40000;  // 40KB threshold from Kokkos_CudaSpace.cpp:139

//==============================================================================
// Test 1: Small Allocation Behavior (<40KB)
//==============================================================================

TEST(AllocationThreshold, SmallAllocation_30KB) {
    const size_t size = 30000;
    ASSERT_LT(size, KOKKOS_THRESHOLD);
    test_small_allocation_30kb();
}

//==============================================================================
// Test 2: Large Allocation Behavior (≥40KB)
//==============================================================================

TEST(AllocationThreshold, LargeAllocation_50KB) {
    const size_t size = 50000;
    ASSERT_GE(size, KOKKOS_THRESHOLD);
    test_large_allocation_50kb();
}

//==============================================================================
// Test 3: Boundary Crossing Behavior
//==============================================================================

TEST(AllocationThreshold, BoundaryCrossing) {
    test_boundary_crossing();
}

//==============================================================================
// Test 4: Multiple Allocation Sizes Sweep
//==============================================================================

TEST(AllocationThreshold, SizeSweep) {
    test_size_sweep();
}

//==============================================================================
// Test 5: Repeated Allocations at Boundary
//==============================================================================

TEST(AllocationThreshold, RepeatedAllocationsAtBoundary) {
    test_repeated_allocations_at_boundary();
}

//==============================================================================
// Test 6: Synchronization Count Comparison
//==============================================================================

TEST(AllocationThreshold, SynchronizationCountComparison) {
    test_synchronization_count_comparison();
}

//==============================================================================
// Test 7: With and Without Execution Space
//==============================================================================

TEST(AllocationThreshold, WithAndWithoutExecutionSpace) {
    test_with_and_without_execution_space();
}

//==============================================================================
// Test 8: Cuda API Call Verification
//==============================================================================

#ifdef KOKKOS_ENABLE_CUDA
TEST(AllocationThreshold, CudaAPIVerification) {
    EXPECT_TRUE(test_cuda_api_verification());
}
#endif

//==============================================================================
// Test Suite Information
//==============================================================================

class AllocationThresholdInfo : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef KOKKOS_ENABLE_CUDA
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        std::cout << "\n=== Kokkos Allocation Threshold Test Suite ===\n";
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Kokkos Threshold: " << KOKKOS_THRESHOLD << " bytes (40KB)\n";
        
#ifdef KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC
        std::cout << "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC: ON\n";
#else
        std::cout << "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC: OFF\n";
#endif
        
        std::cout << "Expected behavior:\n";
        std::cout << "  <40KB: cudaMalloc/cudaFree\n";
#ifdef KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC
        std::cout << "  ≥40KB: cudaMallocAsync/cudaFreeAsync + synchronization\n";
#else
        std::cout << "  ≥40KB: cudaMalloc/cudaFree (async disabled)\n";
#endif
        std::cout << "===============================================\n\n";
#endif
    }
};

TEST_F(AllocationThresholdInfo, PrintConfiguration) {
    // This test just prints configuration info
    SUCCEED();
}

TEST_F(AllocationThresholdInfo, VerifyAsyncMallocEnabled) {
    // This test verifies that KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC is enabled
    // Without it, the 40KB threshold doesn't change allocation behavior
    
#ifdef KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC
    std::cout << "✅ KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC is enabled\n";
    std::cout << "   Threshold tests will show allocation path differences.\n";
    SUCCEED();
#else
    std::cout << "⚠️  WARNING: KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC is DISABLED\n";
    std::cout << "   The 40KB threshold will NOT affect allocation behavior.\n";
    std::cout << "   All allocations will use cudaMalloc/cudaFree.\n";
    std::cout << "\nTo enable async malloc, rebuild Kokkos with:\n";
    std::cout << "  -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON\n\n";
    
    // This is not a failure, but we want to make it very visible
    // Use GTEST_SKIP to mark the test as skipped with a message
    GTEST_SKIP() << "KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC not enabled - "
                 << "threshold tests will not show expected behavior differences";
#endif
}
