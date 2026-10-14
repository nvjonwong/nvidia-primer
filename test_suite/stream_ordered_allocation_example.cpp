/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Example: Stream-Ordered Memory Management with UnManaged Kokkos Views
// This demonstrates how to bypass Kokkos's synchronization overhead
// by using NVKokkos::StreamOrderedView from nvkokkos_helpers.hpp

#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include "../nvkokkos_helpers.hpp"

int main() {
    Kokkos::initialize();
    {
        const size_t N = 1000000;
        
        // Get the CUDA stream from a Kokkos execution space instance
        Kokkos::Cuda cuda_instance;
        
        // ============================================================
        // METHOD 1: RAII StreamOrderedView (Recommended)
        // ============================================================
        {
            std::cout << "Method 1: RAII StreamOrderedView\n";
            
            // Create RAII-managed view (allocates with cudaMallocAsync, no sync!)
            NVKokkos::StreamOrderedView<double*> managed_view(cuda_instance.cuda_stream(), N);
            
            // Get the UnManaged Kokkos::View
            auto view = managed_view.view();
            
            // Use the view normally with Kokkos operations
            Kokkos::parallel_for("init", 
                Kokkos::RangePolicy<Kokkos::Cuda>(cuda_instance, 0, N),
                KOKKOS_LAMBDA(int i) {
                    view(i) = i * 2.0;
                });
            
            // More operations on the same stream
            Kokkos::parallel_for("modify",
                Kokkos::RangePolicy<Kokkos::Cuda>(cuda_instance, 0, N),
                KOKKOS_LAMBDA(int i) {
                    view(i) *= 3.0;
                });
            
            // All operations are stream-ordered, no implicit syncs!
            
            // Only sync when we actually need results
            cuda_instance.fence();
            
            // Automatic cleanup when managed_view goes out of scope
            // (calls cudaFreeAsync, no sync!)
            std::cout << "  ✓ Stream-ordered allocation (no sync)\n";
            std::cout << "  ✓ Automatic RAII cleanup (no sync)\n";
        }
        
        // ============================================================
        // METHOD 2: Using operator* for convenience
        // ============================================================
        {
            std::cout << "\nMethod 2: Using operator* shorthand\n";
            
            NVKokkos::StreamOrderedView<double*> managed_view(cuda_instance.cuda_stream(), N);
            
            // Use operator* to get view (or implicit conversion)
            Kokkos::parallel_for("compute",
                Kokkos::RangePolicy<Kokkos::Cuda>(cuda_instance, 0, N),
                KOKKOS_LAMBDA(int i) {
                    (*managed_view)(i) = i * 5.0;
                });
            
            cuda_instance.fence();
            
            std::cout << "  ✓ Convenient access with operator*\n";
        }
        
        // ============================================================
        // METHOD 3: Manual allocation (low-level approach)
        // ============================================================
        {
            std::cout << "\nMethod 3: Manual cudaMallocAsync/cudaFreeAsync\n";
            
            // Manually allocate memory (stream-ordered, no sync)
            double* raw_ptr = nullptr;
            cudaMallocAsync(&raw_ptr, N * sizeof(double), cuda_instance.cuda_stream());
            
            // Wrap in UnManaged View - Kokkos won't manage memory lifecycle
            Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
                view_unmanaged(raw_ptr, N);
            
            Kokkos::parallel_for("manual_init", 
                Kokkos::RangePolicy<Kokkos::Cuda>(cuda_instance, 0, N),
                KOKKOS_LAMBDA(int i) {
                    view_unmanaged(i) = i * 7.0;
                });
            
            cuda_instance.fence();
            
            // Manual cleanup - stream-ordered, will execute after all kernels finish
            cudaFreeAsync(raw_ptr, cuda_instance.cuda_stream());
            
            std::cout << "  ✓ Manual memory management (most control)\n";
        }
        
        // ============================================================
        // COMPARISON: Standard Kokkos View (with synchronization)
        // ============================================================
        {
            // Standard managed view - Kokkos handles allocation
            Kokkos::View<double*, Kokkos::CudaSpace> view_managed("managed", N);
            // ^^ This allocation causes synchronization in Kokkos
            
            Kokkos::parallel_for("standard",
                Kokkos::RangePolicy<Kokkos::Cuda>(0, N),
                KOKKOS_LAMBDA(int i) {
                    view_managed(i) = i * 7.0;
                });
            
            // When view_managed goes out of scope:
            // - cudaDeviceSynchronize() <- unnecessary sync
            // - cudaFree()
            // - cudaDeviceSynchronize() <- another unnecessary sync (UVM only)
            
            std::cout << "Standard Kokkos View destructor will sync...\n";
        }
        std::cout << "Standard Kokkos View destroyed (with sync overhead)\n";
        
        // Final sync before exiting
        cuda_instance.fence();
    }
    Kokkos::finalize();
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "✓ Method 1 (Recommended): RAII StreamOrderedView - safe and convenient\n";
    std::cout << "✓ Method 2: operator* shorthand - ergonomic access\n";
    std::cout << "✓ Method 3: Manual allocation - maximum control\n";
    std::cout << "✓ All methods: Zero synchronization overhead from alloc/dealloc\n";
    std::cout << "✓ All operations remain stream-ordered for correctness\n";
    std::cout << "✓ Only sync when actually needed (fence before accessing results)\n";
    std::cout << "\nBenefits:\n";
    std::cout << "  • RAII ensures automatic cleanup (no memory leaks)\n";
    std::cout << "  • Type-safe wrapper around raw CUDA allocations\n";
    std::cout << "  • Works seamlessly with Kokkos algorithms\n";
    std::cout << "  • Move semantics for efficient resource transfer\n";
    
    return 0;
}
