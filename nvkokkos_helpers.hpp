/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file nvkokkos_helpers.hpp
 * @brief Helper utilities for optimizing Kokkos on NVIDIA GPUs
 * 
 * This library provides convenience wrappers and utilities to make Kokkos
 * development on NVIDIA GPUs easier and more performant, including:
 * - CUDA event wrappers for fine-grained stream dependencies
 * - Stream-ordered memory allocation (zero-sync overhead)
 * - Memory prefetching for UVM
 * - Optimized view creation helpers
 * - Recommended tile sizes for MDRangePolicy
 * - View introspection and diagnostics
 * 
 * All utilities are portable and provide fallbacks for non-CUDA backends.
 */

#ifndef NVKOKKOS_HELPERS_HPP
#define NVKOKKOS_HELPERS_HPP

#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace NVKokkos {

#ifdef KOKKOS_ENABLE_CUDA
namespace detail {

inline void cuda_check(cudaError_t err, const char* call) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("NVKokkos CUDA call failed: ") +
                                 call + ": " + cudaGetErrorString(err));
    }
}

inline void cuda_check_no_throw(cudaError_t err, const char* call) noexcept {
    if (err != cudaSuccess) {
        std::cerr << "NVKokkos CUDA call failed: " << call << ": "
                  << cudaGetErrorString(err) << "\n";
    }
}

} // namespace detail
#endif

// ============================================================================
// CUDA Events - Fine-grained stream dependencies
// ============================================================================

#ifdef KOKKOS_ENABLE_CUDA
/**
 * @brief Wrapper for CUDA events to enable fine-grained stream dependencies
 * 
 * Usage example:
 *   auto [exec_space1, exec_space2] =
 *       Kokkos::Experimental::partition_space(Kokkos::Cuda{}, 1, 1);
 *   NVKokkos::Event event;
 *   
 *   Work on stream 1, then record event
 *   event.record(exec_space1);
 *   
 *   Make stream 2 wait for stream 1
 *   event.make_stream_wait(exec_space2);
 *   Now stream 2 work will wait for stream 1 to complete
 */
class Event {
    cudaEvent_t event_ = nullptr;
    
public:
    Event() {
        detail::cuda_check(cudaEventCreate(&event_), "cudaEventCreate");
    }
    
    ~Event() {
        if (event_) {
            detail::cuda_check_no_throw(cudaEventDestroy(event_), "cudaEventDestroy");
        }
    }
    
    // Disable copy, enable move
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (event_) {
                detail::cuda_check_no_throw(cudaEventDestroy(event_), "cudaEventDestroy");
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    /**
     * @brief Record event on the given execution space's stream
     */
    void record(const Kokkos::Cuda& exec_space) {
        detail::cuda_check(cudaEventRecord(event_, exec_space.cuda_stream()),
                           "cudaEventRecord");
    }
    
    /**
     * @brief Make execution space's stream wait for this event
     */
    void make_stream_wait(const Kokkos::Cuda& exec_space) {
        detail::cuda_check(cudaStreamWaitEvent(exec_space.cuda_stream(), event_, 0),
                           "cudaStreamWaitEvent");
    }
    
    /**
     * @brief Block CPU until event completes
     */
    void synchronize() {
        detail::cuda_check(cudaEventSynchronize(event_), "cudaEventSynchronize");
    }
    
    /**
     * @brief Check if event has completed (non-blocking)
     * @return true if event has completed, false otherwise
     */
    bool query() {
        cudaError_t err = cudaEventQuery(event_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        detail::cuda_check(err, "cudaEventQuery");
        return false;
    }
    
    /**
     * @brief Get raw CUDA event handle
     */
    cudaEvent_t get() const { return event_; }
};

#else
// Stub implementation for non-CUDA backends
class Event {
public:
    template<typename ExecSpace>
    void record(const ExecSpace& exec_space) { 
        exec_space.fence(); 
    }
    
    template<typename ExecSpace>
    void make_stream_wait(const ExecSpace& exec_space) { 
        /* no-op */ 
    }
    
    void synchronize() { 
        Kokkos::fence(); 
    }
    
    bool query() { 
        return true; 
    }
};
#endif

// ============================================================================
// Memory Prefetching - Optimize UVM performance
// ============================================================================

/**
 * @brief Prefetch View data to device
 * @tparam View Kokkos View type
 * @tparam ExecSpace Execution space type
 * @param view View to prefetch
 * @param exec_space Execution space (determines stream and device)
 * 
 * Only performs prefetch if View is in CudaUVMSpace. No-op otherwise.
 */
template<typename View, typename ExecSpace>
void prefetch_to_device(const View& view, const ExecSpace& exec_space) {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<typename View::memory_space, Kokkos::CudaUVMSpace>) {
        detail::cuda_check(
            cudaMemPrefetchAsync(view.data(),
                                 view.size() * sizeof(typename View::value_type),
                                 exec_space.cuda_device(),
                                 exec_space.cuda_stream()),
            "cudaMemPrefetchAsync");
    }
#endif
    // No-op for other memory spaces or backends
}

/**
 * @brief Prefetch View data to host
 * @tparam View Kokkos View type
 * @tparam ExecSpace Execution space type
 * @param view View to prefetch
 * @param exec_space Execution space (determines stream)
 * 
 * Only performs prefetch if View is in CudaUVMSpace. No-op otherwise.
 */
template<typename View, typename ExecSpace>
void prefetch_to_host(const View& view, const ExecSpace& exec_space) {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<typename View::memory_space, Kokkos::CudaUVMSpace>) {
        detail::cuda_check(
            cudaMemPrefetchAsync(view.data(),
                                 view.size() * sizeof(typename View::value_type),
                                 cudaCpuDeviceId,
                                 exec_space.cuda_stream()),
            "cudaMemPrefetchAsync");
    }
#endif
}

/**
 * @brief Prefetch View data to specific device ID
 * @tparam View Kokkos View type
 * @param view View to prefetch
 * @param device_id CUDA device ID
 * 
 * Uses default stream. Only performs prefetch if View is in CudaUVMSpace.
 */
template<typename View>
void prefetch_to_device_id(const View& view, int device_id) {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<typename View::memory_space, Kokkos::CudaUVMSpace>) {
        detail::cuda_check(
            cudaMemPrefetchAsync(view.data(),
                                 view.size() * sizeof(typename View::value_type),
                                 device_id, 0),
            "cudaMemPrefetchAsync");
    }
#endif
}

// ============================================================================
// View Creation Helpers - Avoid unnecessary initialization
// ============================================================================

/**
 * @brief Create View without zero-initialization (fast)
 * @tparam T View data type (e.g., "double*", "int**")
 * @tparam Args Dimension types
 * @param label View label
 * @param args View dimensions
 * @return Uninitialized View
 * 
 * Usage:
 *   auto temp = NVKokkos::make_view_uninitialized<double*>("temp", N);
 */
template<typename T, typename... Args>
auto make_view_uninitialized(const std::string& label, Args... args) {
    return Kokkos::View<T>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, label), 
        args...);
}

/**
 * @brief Create View in specific memory space without initialization
 * @tparam T View data type
 * @tparam MemorySpace Memory space (e.g., Kokkos::CudaSpace)
 * @tparam Args Dimension types
 * @param label View label
 * @param args View dimensions
 * @return Uninitialized View in specified memory space
 * 
 * Usage:
 *   auto data = NVKokkos::make_view_uninitialized_in<double*, Kokkos::CudaSpace>("data", N);
 */
template<typename T, typename MemorySpace, typename... Args>
auto make_view_uninitialized_in(const std::string& label, Args... args) {
    return Kokkos::View<T, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, label),
        args...);
}

/**
 * @brief Create View and initialize to specific value (async, no implicit sync)
 * @tparam T View data type
 * @tparam ExecSpace Execution space type
 * @tparam Args Dimension types
 * @param exec Execution space instance for async deep_copy
 * @param label View label
 * @param value Initial value
 * @param args View dimensions
 * @return View initialized to value (fill is in-flight on exec's stream)
 * 
 * Usage:
 *   Kokkos::Cuda space;
 *   auto ones = NVKokkos::make_view_filled<double*>(space, "ones", 1.0, N);
 */
template<typename T, typename ExecSpace, typename... Args>
auto make_view_filled(const ExecSpace& exec,
                     const std::string& label, 
                     typename Kokkos::View<T>::value_type value,
                     Args... args) {
    auto view = make_view_uninitialized<T>(label, args...);
    Kokkos::deep_copy(exec, view, value);
    return view;
}

// ============================================================================
// Optimized Tile Sizes for MDRangePolicy
// ============================================================================

/**
 * @brief Recommended tile size for NVIDIA GPUs
 * 
 * Kokkos defaults (e.g., {2,16} or {2,32}) can be suboptimal for 3D.
 * For better occupancy and coalescing, use {1, ..., 64} tile patterns.
 */
constexpr int RECOMMENDED_TILE_SIZE = 64;  // 64 threads per block for NVIDIA GPUs

/**
 * @brief Create 2D MDRangePolicy with recommended tiling
 * @tparam ExecSpace Execution space (defaults to Kokkos::DefaultExecutionSpace)
 * @param start Start indices {j_start, i_start}
 * @param end End indices {j_end, i_end}
 * @param tile_size Tile size for innermost dimension (default: 64)
 * @return MDRangePolicy with {1, tile_size} tiling
 * 
 * Usage:
 *   Kokkos::parallel_for("2d", NVKokkos::make_mdrange_2d({0,0}, {M,N}),
 *       KOKKOS_LAMBDA(int j, int i) { ... });
 */
template<typename ExecSpace = Kokkos::DefaultExecutionSpace>
auto make_mdrange_2d(const std::array<int, 2>& start, 
                     const std::array<int, 2>& end,
                     int tile_size = RECOMMENDED_TILE_SIZE) {
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecSpace>(
        {start[0], start[1]}, 
        {end[0], end[1]}, 
        {1, tile_size});
}

/**
 * @brief Create 3D MDRangePolicy with recommended tiling
 * @tparam ExecSpace Execution space (defaults to Kokkos::DefaultExecutionSpace)
 * @param start Start indices {k_start, j_start, i_start}
 * @param end End indices {k_end, j_end, i_end}
 * @param tile_size Tile size for innermost dimension (default: 64)
 * @return MDRangePolicy with {1, 1, tile_size} tiling
 * 
 * Usage:
 *   Kokkos::parallel_for("3d", NVKokkos::make_mdrange_3d({0,0,0}, {X,Y,Z}),
 *       KOKKOS_LAMBDA(int k, int j, int i) { ... });
 */
template<typename ExecSpace = Kokkos::DefaultExecutionSpace>
auto make_mdrange_3d(const std::array<int, 3>& start,
                     const std::array<int, 3>& end,
                     int tile_size = RECOMMENDED_TILE_SIZE) {
    return Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecSpace>(
        {start[0], start[1], start[2]},
        {end[0], end[1], end[2]},
        {1, 1, tile_size});
}

/**
 * @brief Create 4D MDRangePolicy with recommended tiling
 * @tparam ExecSpace Execution space (defaults to Kokkos::DefaultExecutionSpace)
 * @param start Start indices {l_start, k_start, j_start, i_start}
 * @param end End indices {l_end, k_end, j_end, i_end}
 * @param tile_size Tile size for innermost dimension (default: 64)
 * @return MDRangePolicy with {1, 1, 1, tile_size} tiling
 * 
 * Usage:
 *   Kokkos::parallel_for("4d", NVKokkos::make_mdrange_4d({0,0,0,0}, {W,X,Y,Z}),
 *       KOKKOS_LAMBDA(int l, int k, int j, int i) { ... });
 */
template<typename ExecSpace = Kokkos::DefaultExecutionSpace>
auto make_mdrange_4d(const std::array<int, 4>& start,
                     const std::array<int, 4>& end,
                     int tile_size = RECOMMENDED_TILE_SIZE) {
    return Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecSpace>(
        {start[0], start[1], start[2], start[3]},
        {end[0], end[1], end[2], end[3]},
        {1, 1, 1, tile_size});
}

// ============================================================================
// View Memory Trait Checking - Identify View modifiers
// ============================================================================

/**
 * @brief Check if View has Atomic memory trait (compile-time)
 * 
 * When a view is passed by reference, it's not obvious if it has the Atomic
 * trait attached. This function helps identify if all accesses will be atomic.
 * 
 * Usage:
 *   if constexpr (NVKokkos::is_atomic_view<decltype(view)>()) {
 *       // All accesses will use atomics - may be slow!
 *   }
 */
template<typename View>
constexpr bool is_atomic_view() {
    return (View::memory_traits::is_atomic != 0);
}

/**
 * @brief Check if View has Restrict memory trait (compile-time)
 * 
 * Restrict trait tells the compiler that no aliasing occurs, enabling
 * better optimizations. Check this to verify optimization opportunities.
 */
template<typename View>
constexpr bool is_restrict_view() {
    return (View::memory_traits::is_restrict != 0);
}

/**
 * @brief Check if View has Unmanaged memory trait (compile-time)
 * 
 * Unmanaged views don't own their memory. Useful for wrapping existing
 * allocations or passing sub-views.
 */
template<typename View>
constexpr bool is_unmanaged_view() {
    return (View::memory_traits::is_unmanaged != 0);
}

/**
 * @brief Check if View has RandomAccess memory trait (compile-time)
 * 
 * RandomAccess trait can hint texture cache usage on some backends.
 */
template<typename View>
constexpr bool is_random_access_view() {
    return (View::memory_traits::is_random_access != 0);
}

/**
 * @brief Check if View is const-qualified (compile-time)
 * 
 * Const views cannot be written to. Useful for enforcing read-only access
 * and enabling compiler optimizations.
 */
template<typename View>
constexpr bool is_const_view() {
    return std::is_const_v<typename View::value_type>;
}

/**
 * @brief Get all memory trait flags as a string (for debugging)
 * @return Space-separated list of active traits, or "None"
 * 
 * Usage:
 *   std::cout << "Traits: " << NVKokkos::get_memory_traits_string<ViewType>() << "\n";
 */
template<typename View>
std::string get_memory_traits_string() {
    std::string traits;
    if (is_atomic_view<View>()) traits += "Atomic ";
    if (is_restrict_view<View>()) traits += "Restrict ";
    if (is_unmanaged_view<View>()) traits += "Unmanaged ";
    if (is_random_access_view<View>()) traits += "RandomAccess ";
    if (is_const_view<View>()) traits += "Const ";
    return traits.empty() ? "None" : traits;
}

/**
 * @brief Print comprehensive View memory trait information
 * @param view View to inspect
 * @param name Optional name override (uses view.label() if empty)
 * 
 * Prints all memory traits in a readable format. Useful for debugging
 * when views are passed through many function layers.
 * 
 * Usage:
 *   void process(const auto& view) {
 *       NVKokkos::print_view_traits(view, "input");
 *       // ... process view ...
 *   }
 */
template<typename View>
void print_view_traits(const View& view, const std::string& name = "") {
    std::cout << "View: " << (name.empty() ? view.label() : name) << "\n"
              << "  Memory Traits: " << get_memory_traits_string<View>() << "\n"
              << "  Is Atomic: " << (is_atomic_view<View>() ? "yes" : "no") << "\n"
              << "  Is Restrict: " << (is_restrict_view<View>() ? "yes" : "no") << "\n"
              << "  Is Const: " << (is_const_view<View>() ? "yes" : "no") << "\n"
              << "  Is Unmanaged: " << (is_unmanaged_view<View>() ? "yes" : "no") << "\n";
}

// ============================================================================
// Stream Management Helpers (for debugging)
// ============================================================================

#ifdef KOKKOS_ENABLE_CUDA
/**
 * @brief Create vector of execution spaces for stream management
 * @param num_streams Number of execution spaces to create
 * @param force_single_stream If true, all spaces share the same stream (for debugging)
 * @return Vector of Kokkos::Cuda execution spaces
 * 
 * Usage:
 *   // Production: multiple streams
 *   auto exec_spaces = NVKokkos::create_exec_spaces(4);
 *   
 *   // Debug: force all to one stream
 *   auto exec_spaces = NVKokkos::create_exec_spaces(4, true);
 */
inline std::vector<Kokkos::Cuda> create_exec_spaces(int num_streams, 
                                                     bool force_single_stream = false) {
    std::vector<Kokkos::Cuda> spaces;
    if (num_streams <= 0) {
        return spaces;
    }

    spaces.reserve(num_streams);
    
    if (force_single_stream) {
        // All execution spaces share the same CUDA stream
        Kokkos::Cuda shared_space;
        for (int i = 0; i < num_streams; ++i) {
            spaces.push_back(shared_space);
        }
    } else {
        // Partitioning a base space is the Kokkos-supported way to request
        // independent CUDA streams for concurrent work.
        std::vector<int> weights(static_cast<std::size_t>(num_streams), 1);
        spaces = Kokkos::Experimental::partition_space(Kokkos::Cuda{}, weights);
    }
    
    return spaces;
}

/**
 * @brief Fence all execution spaces in vector
 * @param exec_spaces Vector of execution spaces to fence
 */
inline void fence_all(const std::vector<Kokkos::Cuda>& exec_spaces) {
    for (const auto& space : exec_spaces) {
        space.fence();
    }
}
#endif

// ============================================================================
// Stream-Ordered Memory Allocation - Zero-overhead allocation/deallocation
// ============================================================================

#ifdef KOKKOS_ENABLE_CUDA
/**
 * @brief RAII wrapper for stream-ordered CUDA memory allocations
 * 
 * This class bypasses Kokkos's synchronization overhead in cudaMallocAsync/
 * cudaFreeAsync by manually managing memory with true stream-ordered operations.
 * 
 * **Problem:** Even with KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC enabled, Kokkos
 * adds synchronization calls around memory operations:
 *   - cudaMallocAsync + cudaStreamSynchronize (1 sync)
 *   - cudaDeviceSynchronize + cudaFreeAsync + cudaDeviceSynchronize (2 syncs)
 * 
 * **Solution:** Use StreamOrderedView for zero synchronization overhead:
 *   - Allocates with cudaMallocAsync (no sync!)
 *   - Deallocates with cudaFreeAsync in destructor (no sync!)
 *   - Provides UnManaged Kokkos::View for seamless integration
 * 
 * **Benefits:**
 *   ✅ Zero synchronization overhead from allocation/deallocation
 *   ✅ RAII ensures automatic cleanup and exception safety
 *   ✅ Type-safe wrapper with move semantics
 *   ✅ Works seamlessly with all Kokkos algorithms
 * 
 * **Requirements:**
 *   - CUDA 11.2+ (for cudaMallocAsync/cudaFreeAsync)
 *   - All operations must use the same stream for correctness
 * 
 * @tparam DataType View data type specification (e.g., "double*", "int**")
 * @tparam MemorySpace Memory space (default: Kokkos::CudaSpace, also supports Kokkos::CudaUVMSpace)
 * @tparam Properties Additional view properties (optional)
 * 
 * **Memory Space Support:**
 *   - `Kokkos::CudaSpace` (default): Device memory, uses cudaMallocAsync/cudaFreeAsync
 *   - `Kokkos::CudaUVMSpace`: Unified memory, uses cudaMallocAsync for managed memory (CUDA 11.2+)
 *     Falls back to cudaMallocManaged + cudaStreamAttachMemAsync if async allocation not supported
 * 
 * **Usage Example:**
 * @code
 * Kokkos::Cuda exec_space;
 * 
 * // Device memory (default)
 * NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
 * 
 * // Unified memory (UVM)
 * NVKokkos::StreamOrderedView<double*, Kokkos::CudaUVMSpace> uvm_data(exec_space.cuda_stream(), N);
 * 
 * // Get UnManaged Kokkos::View
 * auto view = data.view();  // or: auto view = *data;
 * auto uvm_view = uvm_data.view();
 * 
 * // Use with Kokkos operations
 * Kokkos::parallel_for("kernel",
 *     Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
 *     KOKKOS_LAMBDA(int i) { view(i) = i * 2.0; });
 * 
 * // Sync when needed
 * exec_space.fence();
 * 
 * // Automatic cleanup when data goes out of scope (no sync!)
 * @endcode
 * 
 * **Performance Comparison:**
 * | Approach | Alloc Syncs | Dealloc Syncs | Total |
 * |----------|-------------|---------------|-------|
 * | Standard Kokkos View | 1 | 2 | 3 ⚠️ |
 * | Kokkos + CUDA_MALLOC_ASYNC | 1 | 2 | 3 ⚠️ |
 * | StreamOrderedView | 0 | 0 | 0 ✅ |
 * 
 * **Safety Considerations:**
 * ⚠️ You are responsible for:
 *   - Ensuring all operations use the same stream
 *   - Synchronizing before accessing results on host
 *   - Not freeing memory while kernels are using it
 * 
 * ⚠️ Stream consistency example:
 * @code
 * // ✓ CORRECT: Use same execution space instance
 * Kokkos::Cuda exec_space;
 * NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
 * Kokkos::parallel_for("kernel",
 *     Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),  // Same instance!
 *     KOKKOS_LAMBDA(int i) { data.view()(i) = i; });
 * 
 * // ✗ WRONG: Default space may use different stream
 * Kokkos::parallel_for("kernel",
 *     Kokkos::RangePolicy<Kokkos::Cuda>(0, N),  // Default space - wrong!
 *     KOKKOS_LAMBDA(int i) { data.view()(i) = i; });
 * @endcode
 * 
 * **When to Use:**
 * ✅ High-performance libraries requiring maximum control
 * ✅ Applications with many temporary allocations in hot loops
 * ✅ Multi-stream pipelines where sync overhead is significant
 * ✅ After profiling shows allocation/deallocation is a bottleneck
 * 
 * ❌ Not recommended for:
 * ❌ General application code (standard Views are simpler)
 * ❌ Beginners (requires understanding of stream ordering)
 * ❌ Portable code (CUDA-specific)
 */
template <typename DataType, typename MemorySpace = Kokkos::CudaSpace, typename... Properties>
class StreamOrderedView {
private:
    using value_type = typename Kokkos::View<DataType, Properties...>::value_type;
    using unmanaged_view_type = Kokkos::View<DataType, 
                                              MemorySpace,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    
    value_type* raw_ptr_;
    size_t size_bytes_;
    cudaStream_t stream_;
    unmanaged_view_type view_;
    
    // Helper to allocate based on memory space
    void allocate_memory(size_t size_bytes, cudaStream_t stream) {
        if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaUVMSpace>) {
            // For UVM, use cudaMallocAsync with managed memory support (CUDA 11.2+)
            // This provides zero-sync allocation for managed memory
            // Note: Requires CUDA 11.2+ and device support for async managed allocation
            cudaError_t err = cudaMallocAsync(&raw_ptr_, size_bytes, stream);
            if (err != cudaSuccess) {
                // Fallback: try cudaMallocManaged if async fails (older CUDA or unsupported device)
                err = cudaMallocManaged(&raw_ptr_, size_bytes);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("StreamOrderedView: cudaMallocAsync/cudaMallocManaged failed: ") + 
                        cudaGetErrorString(err));
                }
                // Associate with stream for stream-ordered access
                err = cudaStreamAttachMemAsync(stream, raw_ptr_, size_bytes, cudaMemAttachGlobal);
                if (err != cudaSuccess) {
                    detail::cuda_check_no_throw(cudaFree(raw_ptr_), "cudaFree");
                    throw std::runtime_error(
                        std::string("StreamOrderedView: cudaStreamAttachMemAsync failed: ") + 
                        cudaGetErrorString(err));
                }
            }
        } else {
            // For device memory, use cudaMallocAsync (asynchronous, no sync!)
            cudaError_t err = cudaMallocAsync(&raw_ptr_, size_bytes, stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("StreamOrderedView: cudaMallocAsync failed: ") + 
                    cudaGetErrorString(err));
            }
        }
    }
    
    // Helper to free memory based on memory space
    void free_memory() {
        if (raw_ptr_ != nullptr) {
            if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaUVMSpace>) {
                // For UVM, use cudaFreeAsync (no sync!)
                detail::cuda_check_no_throw(cudaFreeAsync(raw_ptr_, stream_), "cudaFreeAsync");
            } else {
                // For device memory, use cudaFreeAsync (no sync!)
                detail::cuda_check_no_throw(cudaFreeAsync(raw_ptr_, stream_), "cudaFreeAsync");
            }
        }
    }
    
public:
    /**
     * @brief Constructor: allocates memory with cudaMallocAsync (no sync!)
     * @param stream CUDA stream for allocation
     * @param dims Dimensions for the view (e.g., N for 1D, N, M for 2D)
     * @throws std::runtime_error if cudaMallocAsync fails
     */
    template <typename... Dims>
    StreamOrderedView(cudaStream_t stream, Dims... dims) 
        : raw_ptr_(nullptr), stream_(stream) {
        
        // Calculate total size
        size_t total_elements = (dims * ...);
        size_bytes_ = total_elements * sizeof(value_type);
        
        // Allocate memory (handles both CudaSpace and CudaUVMSpace)
        allocate_memory(size_bytes_, stream_);
        
        // Create UnManaged view wrapping the allocation
        view_ = unmanaged_view_type(raw_ptr_, dims...);
    }
    
    /**
     * @brief Destructor: frees memory with cudaFreeAsync (no sync!)
     * 
     * Memory deallocation is queued on the stream and executes after all
     * prior operations on the stream complete. Errors from cudaFreeAsync
     * are ignored to maintain exception safety.
     */
    ~StreamOrderedView() {
        free_memory();  // NO SYNC! (handles both CudaSpace and CudaUVMSpace)
        // Note: Errors ignored for exception safety
    }
    
    // Delete copy operations (move-only type for safety)
    StreamOrderedView(const StreamOrderedView&) = delete;
    StreamOrderedView& operator=(const StreamOrderedView&) = delete;
    
    /**
     * @brief Move constructor: transfers ownership of allocation
     */
    StreamOrderedView(StreamOrderedView&& other) noexcept 
        : raw_ptr_(other.raw_ptr_)
        , size_bytes_(other.size_bytes_)
        , stream_(other.stream_)
        , view_(other.view_) {
        other.raw_ptr_ = nullptr;
    }
    
    /**
     * @brief Move assignment: transfers ownership, frees current allocation
     */
    StreamOrderedView& operator=(StreamOrderedView&& other) noexcept {
        if (this != &other) {
            // Free our current allocation (if any)
            free_memory();
            // Take ownership of other's allocation
            raw_ptr_ = other.raw_ptr_;
            size_bytes_ = other.size_bytes_;
            stream_ = other.stream_;
            view_ = other.view_;
            other.raw_ptr_ = nullptr;
        }
        return *this;
    }
    
    /**
     * @brief Get the UnManaged Kokkos::View
     * @return UnManaged view wrapping the stream-ordered allocation
     */
    unmanaged_view_type view() const { return view_; }
    
    /**
     * @brief Dereference operator for convenient access
     * @return UnManaged view wrapping the stream-ordered allocation
     */
    unmanaged_view_type operator*() const { return view_; }
    
    /**
     * @brief Arrow operator (returns view for consistency)
     */
    unmanaged_view_type operator->() const { return view_; }
    
    /**
     * @brief Implicit conversion to UnManaged view for convenience
     */
    operator unmanaged_view_type() const { return view_; }
    
    /**
     * @brief Get raw pointer (use with caution)
     * @return Raw pointer to allocated memory
     */
    value_type* data() const { return raw_ptr_; }
    
    /**
     * @brief Get the CUDA stream associated with this allocation
     * @return CUDA stream used for allocation/deallocation
     */
    cudaStream_t stream() const { return stream_; }
    
    /**
     * @brief Get allocation size in bytes
     * @return Size of allocation in bytes
     */
    size_t size_bytes() const { return size_bytes_; }
    
    /**
     * @brief Check if allocation is valid (non-null)
     * @return true if allocation exists, false otherwise
     */
    bool valid() const { return raw_ptr_ != nullptr; }
    
    /**
     * @brief Explicit boolean conversion
     */
    explicit operator bool() const { return valid(); }
};

/**
 * @brief Helper function to create StreamOrderedView with type deduction
 * @param exec_space Kokkos execution space (extracts stream from it)
 * @param dims Dimensions for the view
 * @return StreamOrderedView managing the allocation
 * 
 * Usage:
 *   auto data = NVKokkos::make_stream_ordered_view<double*>(exec_space, N);
 */
template <typename DataType, typename... Dims>
auto make_stream_ordered_view(const Kokkos::Cuda& exec_space, Dims... dims) {
    return StreamOrderedView<DataType>(exec_space.cuda_stream(), dims...);
}

#else
// Stub for non-CUDA backends - falls back to standard Kokkos Views
template <typename DataType, typename... Properties>
class StreamOrderedView {
private:
    Kokkos::View<DataType, Properties...> view_;
    
public:
    template <typename... Dims>
    StreamOrderedView(void* /* stream */, Dims... dims) 
        : view_("stream_ordered_fallback", dims...) {}
    
    auto view() const { return view_; }
    auto operator*() const { return view_; }
    operator decltype(view_)() const { return view_; }
    auto data() const { return view_.data(); }
    bool valid() const { return true; }
    explicit operator bool() const { return valid(); }
};

template <typename DataType, typename ExecSpace, typename... Dims>
auto make_stream_ordered_view(const ExecSpace& /* exec_space */, Dims... dims) {
    return StreamOrderedView<DataType>(nullptr, dims...);
}
#endif

} // namespace NVKokkos

#endif // NVKOKKOS_HELPERS_HPP
