<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Helper Libraries: Debugging Utilities and NVKokkos Reference

This chapter provides two helper libraries for Kokkos development on NVIDIA GPUs: a stream-consolidation debug utility for diagnosing stream synchronization issues without recompiling, and the NVKokkos reference library covering CUDA events, UVM prefetching, uninitialized View creation, MDRange tiling, and View trait introspection.

## Streams Helper Library (for debugging)

**Problem**: When debugging stream synchronization issues, you need to quickly force all execution spaces onto a single stream without changing application logic.

**Solution**: Helper functions to manage stream consolidation:

```cpp
#include <Kokkos_Core.hpp>
#include <vector>

namespace KokkosDebug {

// Helper to create vector of execution spaces
// Set force_single_stream=true for debugging
template<typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
std::vector<ExecutionSpace> create_exec_spaces(
    int num_streams, 
    bool force_single_stream = false) {
    
    std::vector<ExecutionSpace> spaces;
    
    if (force_single_stream) {
        // Debug mode: All "streams" use the same execution space
        ExecutionSpace single_space;
        spaces.resize(num_streams, single_space);
        
        std::cout << "DEBUG: Using single stream for all " 
                  << num_streams << " execution spaces\n";
    } else {
        // Production mode: partition one base execution space into
        // independent backend streams/resources.
        ExecutionSpace base_space;
        std::vector<int> weights(num_streams, 1);
        spaces = Kokkos::Experimental::partition_space(base_space, weights);
        
        std::cout << "Using " << num_streams 
                  << " partitioned execution spaces\n";
    }
    
    return spaces;
}

// Helper to fence all execution spaces
template<typename ExecutionSpace>
void fence_all(const std::vector<ExecutionSpace>& spaces) {
    for(auto& space : spaces) {
        space.fence();
    }
}

} // namespace KokkosDebug
```

**Usage Example:**

```cpp
#include "kokkos_debug_helpers.hpp"

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Toggle debug mode with environment variable
        bool debug_mode = (std::getenv("KOKKOS_DEBUG_SINGLE_STREAM") != nullptr);
        
        // Create execution spaces (2 streams or 1 if debug)
        auto exec_spaces = KokkosDebug::create_exec_spaces<Kokkos::Cuda>(2, debug_mode);
        
        const int N = 1000000;
        Kokkos::View<double*> data("data", N);
        
        // Alternating stream pattern
        for(int iteration = 0; iteration < 100; iteration++) {
            int idx = iteration % 2;
            
            // Work on current stream
            Kokkos::parallel_for("compute",
                Kokkos::RangePolicy<Kokkos::Cuda>(exec_spaces[idx], 0, N),
                KOKKOS_LAMBDA(int i) {
                    data(i) = i * 2.0;
                });
        }
        
        // Final synchronization
        KokkosDebug::fence_all(exec_spaces);
        
        std::cout << "Completed " << (debug_mode ? "DEBUG" : "PRODUCTION") 
                  << " mode\n";
    }
    Kokkos::finalize();
    return 0;
}
```

**Quick Debug Workflow:**

```bash
# Normal run - uses multiple streams
./my_kokkos_app

# Debug run - force single stream without code changes
export KOKKOS_DEBUG_SINGLE_STREAM=1
./my_kokkos_app

# If bug disappears with single stream -> stream synchronization issue
# If bug persists -> not stream-related
```

**Alternative: Simple Function-Based Approach**

For quick debugging without classes:

```cpp
// Simple helper - just add at top of your file
inline auto make_debug_streams(int num_streams) {
    bool debug = (std::getenv("KOKKOS_DEBUG_SINGLE_STREAM") != nullptr);
    
    if (debug) {
        Kokkos::Cuda single_stream;
        return std::vector<Kokkos::Cuda>(num_streams, single_stream);
    } else {
        return Kokkos::Experimental::partition_space(
            Kokkos::Cuda{}, std::vector<int>(num_streams, 1));
    }
}

// Usage - drop-in replacement for stream creation
auto exec_spaces = make_debug_streams(4);
// Rest of code unchanged
```

**Benefits:**
- **Debug mode flexibility**: Toggle debug mode without recompiling
- **Issue isolation**: Isolates stream synchronization issues quickly
- **Minimal integration**: Minimal code changes to existing applications
- **Runtime control**: Can be controlled via environment variables or command line
- **Clear diagnostics**: Provides clear diagnostic output

See also:
- [04-execution-and-streams.md](04-execution-and-streams.md) — execution space and stream patterns this utility is designed to debug
- `test_suite/test_execution_spaces.cpp` — tests that verify stream isolation behavior

## NVIDIA Kokkos Helper Library

A collection of helper functions and wrappers to make Kokkos development on NVIDIA GPUs easier and more performant.

### Events and Dependencies

CUDA events are lightweight synchronization markers that can be recorded on one stream and waited on by another, enabling fine-grained inter-stream dependencies without a full device fence. Kokkos does not expose `cudaEvent_t` directly.

**Problem**: Kokkos doesn't expose CUDA events directly for fine-grained stream dependencies.

**Solution**: Event wrapper for CUDA execution spaces:

```cpp
namespace NVKokkos {

#ifdef KOKKOS_ENABLE_CUDA
// CUDA Event wrapper
class Event {
    cudaEvent_t event_;
    
public:
    Event() {
        cudaEventCreate(&event_);
    }
    
    ~Event() {
        cudaEventDestroy(event_);
    }
    
    // Record event on execution space
    void record(const Kokkos::Cuda& exec_space) {
        cudaEventRecord(event_, exec_space.cuda_stream());
    }
    
    // Make execution space wait for this event
    void make_stream_wait(const Kokkos::Cuda& exec_space) {
        cudaStreamWaitEvent(exec_space.cuda_stream(), event_, 0);
    }
    
    // Block CPU until event completes
    void synchronize() {
        cudaEventSynchronize(event_);
    }
    
    // Check if event has completed
    bool query() {
        return cudaEventQuery(event_) == cudaSuccess;
    }
    
    cudaEvent_t get() const { return event_; }
};
#else
// Stub for non-CUDA backends
class Event {
public:
    template<typename ExecSpace>
    void record(const ExecSpace& exec_space) { exec_space.fence(); }
    
    template<typename ExecSpace>
    void make_stream_wait(const ExecSpace& exec_space) { /* no-op */ }
    
    void synchronize() { Kokkos::fence(); }
    bool query() { return true; }
};
#endif

} // namespace NVKokkos
```

**Usage:**
```cpp
Kokkos::Cuda base_exec;
auto [exec_space1, exec_space2] =
    Kokkos::Experimental::partition_space(base_exec, 1, 1);
NVKokkos::Event event;

// Work on stream 1
Kokkos::parallel_for("work1",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space1, 0, N),
    KOKKOS_LAMBDA(int i) { data1(i) = compute(i); });

event.record(exec_space1);  // Mark completion point

// Make stream 2 wait for stream 1
event.make_stream_wait(exec_space2);

// Work on stream 2 that depends on stream 1
Kokkos::parallel_for("work2",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space2, 0, N),
    KOKKOS_LAMBDA(int i) { data2(i) = process(data1(i)); });
```

See: [CUDA Events — Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)

### Memory Prefetching

`cudaMemPrefetchAsync` hints the CUDA runtime to migrate UVM pages to a destination device (or host) before they are accessed, reducing the number of page faults during kernel execution. Kokkos provides no built-in prefetch interface.

**Problem**: UVM page faults can be expensive. Kokkos doesn't provide prefetch interface.

**Solution**: Template helpers for UVM prefetching:

```cpp
namespace NVKokkos {

// Prefetch View to device
template<typename View, typename ExecSpace>
void prefetch_to_device(const View& view, const ExecSpace& exec_space) {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<typename View::memory_space, Kokkos::CudaUVMSpace>) {
        cudaMemPrefetchAsync(view.data(),
                            view.size() * sizeof(typename View::value_type),
                            exec_space.cuda_device(),
                            exec_space.cuda_stream());
    }
#endif
    // No-op for other memory spaces or backends
}

// Prefetch View to host
template<typename View, typename ExecSpace>
void prefetch_to_host(const View& view, const ExecSpace& exec_space) {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<typename View::memory_space, Kokkos::CudaUVMSpace>) {
        cudaMemPrefetchAsync(view.data(),
                            view.size() * sizeof(typename View::value_type),
                            cudaCpuDeviceId,
                            exec_space.cuda_stream());
    }
#endif
}

// Prefetch hint for explicit device ID
template<typename View>
void prefetch_to_device_id(const View& view, int device_id) {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<typename View::memory_space, Kokkos::CudaUVMSpace>) {
        cudaMemPrefetchAsync(view.data(),
                            view.size() * sizeof(typename View::value_type),
                            device_id, 0);
    }
#endif
}

} // namespace NVKokkos
```

**Usage:**
```cpp
Kokkos::View<double*, Kokkos::CudaUVMSpace> data("data", N);
Kokkos::Cuda exec_space;

// Prefetch to GPU before computation
NVKokkos::prefetch_to_device(data, exec_space);

Kokkos::parallel_for("compute",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
    KOKKOS_LAMBDA(int i) { data(i) *= 2.0; });

// Prefetch to host before CPU access
NVKokkos::prefetch_to_host(data, exec_space);
exec_space.fence();

for(int i = 0; i < 10; i++) {
    std::cout << data(i) << " ";
}
```

See: [`cudaMemPrefetchAsync` — Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) and [Kokkos UVM / CudaUVMSpace](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/View.html)

### Better View Creation Defaults

`Kokkos::View` zero-initializes memory by default, which is safe but incurs an extra `cudaMemset` on every allocation. The helpers below skip initialization and optionally fill with an explicit value on the user's execution space stream.

**Problem**: Default Kokkos Views initialize to zero, causing unnecessary overhead.

**Solution**: Helper functions for fast View creation:

```cpp
namespace NVKokkos {

// Create View without initialization (fast)
template<typename T, typename... Args>
auto make_view_uninitialized(const std::string& label, Args... args) {
    return Kokkos::View<T>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, label), 
        args...);
}

// Create View with specific memory space, no init
template<typename T, typename MemorySpace, typename... Args>
auto make_view_uninitialized_in(const std::string& label, Args... args) {
    return Kokkos::View<T, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, label),
        args...);
}

// Create View and fill with value (async — no implicit sync)
template<typename T, typename ExecSpace, typename... Args>
auto make_view_filled(const ExecSpace& exec,
                     const std::string& label, 
                     typename Kokkos::View<T>::value_type value,
                     Args... args) {
    auto view = make_view_uninitialized<T>(label, args...);
    Kokkos::deep_copy(exec, view, value);
    return view;
}

} // namespace NVKokkos
```

**Usage:**
```cpp
// Fast - no initialization overhead
auto temp = NVKokkos::make_view_uninitialized<double*>("temp", N);

// Specific memory space
auto device_data = NVKokkos::make_view_uninitialized_in<double*, Kokkos::CudaSpace>("data", N);

// Initialize to specific value (async on exec space's stream)
Kokkos::Cuda space;
auto initialized = NVKokkos::make_view_filled<double*>(space, "init", 1.0, N);
```

See: [Kokkos `view_alloc` and allocation properties](https://kokkos.org/kokkos-core-wiki/API/core/view/view_alloc.html)

### Recommended MDRange Tile Sizes

`Kokkos::MDRangePolicy` requires an explicit tile-size argument; the Kokkos defaults (e.g., `<2,16>` for 2D) are too small for NVIDIA GPUs and result in under-occupied thread blocks. The helpers below apply a 64-thread tile recommendation automatically.

**Problem**: Kokkos default tile sizes are suboptimal (e.g., `<2,16>` for 2D).

**Solution**: Helper functions that apply recommended tile sizes automatically:

```cpp
namespace NVKokkos {

// Recommended tile size for NVIDIA GPUs (64 threads per block)
constexpr int RECOMMENDED_TILE_SIZE = 64;

// Helper functions for MDRangePolicy with recommended tiling
template<typename ExecSpace = Kokkos::DefaultExecutionSpace>
auto make_mdrange_2d(const std::array<int, 2>& start, 
                     const std::array<int, 2>& end,
                     int tile_size = RECOMMENDED_TILE_SIZE);

template<typename ExecSpace = Kokkos::DefaultExecutionSpace>
auto make_mdrange_3d(const std::array<int, 3>& start,
                     const std::array<int, 3>& end,
                     int tile_size = RECOMMENDED_TILE_SIZE);

template<typename ExecSpace = Kokkos::DefaultExecutionSpace>
auto make_mdrange_4d(const std::array<int, 4>& start,
                     const std::array<int, 4>& end,
                     int tile_size = RECOMMENDED_TILE_SIZE);

} // namespace NVKokkos
```

**Usage:**
```cpp
#include "nvkokkos_helpers.hpp"

// 2D with automatic recommended tiling
Kokkos::parallel_for("2d_optimized",
    NVKokkos::make_mdrange_2d({0, 0}, {M, N}),
    KOKKOS_LAMBDA(int j, int i) { /* work */ });

// 3D with automatic recommended tiling
Kokkos::parallel_for("3d_optimized",
    NVKokkos::make_mdrange_3d({0, 0, 0}, {X, Y, Z}),
    KOKKOS_LAMBDA(int k, int j, int i) { /* work */ });

// 4D with automatic recommended tiling
Kokkos::parallel_for("4d_optimized",
    NVKokkos::make_mdrange_4d({0, 0, 0, 0}, {W, X, Y, Z}),
    KOKKOS_LAMBDA(int l, int k, int j, int i) { /* work */ });

// Or specify tile size directly without helper
Kokkos::parallel_for("manual",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}, {1, 64}),
    KOKKOS_LAMBDA(int j, int i) { /* work */ });
```

See: [Kokkos `MDRangePolicy`](https://kokkos.org/kokkos-core-wiki/API/core/policies/MDRangePolicy.html)

### View Attribute/Modifier Checking

**Problem**: When views are passed by reference to functions, it's not obvious if they have memory traits like `Atomic`, `Restrict`, `Const`, or `Unmanaged` attached. These traits significantly affect performance and behavior.

**Solution**: Template helpers to inspect View memory traits at compile-time:

```cpp
namespace NVKokkos {

// Check if View has Atomic memory trait
template<typename View>
constexpr bool is_atomic_view() {
    return (View::memory_traits::is_atomic != 0);
}

// Check if View has Restrict memory trait
template<typename View>
constexpr bool is_restrict_view() {
    return (View::memory_traits::is_restrict != 0);
}

// Check if View has Unmanaged memory trait
template<typename View>
constexpr bool is_unmanaged_view() {
    return (View::memory_traits::is_unmanaged != 0);
}

// Check if View has RandomAccess memory trait
template<typename View>
constexpr bool is_random_access_view() {
    return (View::memory_traits::is_random_access != 0);
}

// Check if View is const-qualified
template<typename View>
constexpr bool is_const_view() {
    return std::is_const_v<typename View::value_type>;
}

// Get all memory trait flags as a string (for debugging)
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

// Print comprehensive View information including memory traits
template<typename View>
void print_view_traits(const View& view, const std::string& name = "") {
    std::cout << "View: " << (name.empty() ? view.label() : name) << "\n"
              << "  Memory Traits: " << get_memory_traits_string<View>() << "\n"
              << "  Is Atomic: " << (is_atomic_view<View>() ? "yes" : "no") << "\n"
              << "  Is Restrict: " << (is_restrict_view<View>() ? "yes" : "no") << "\n"
              << "  Is Const: " << (is_const_view<View>() ? "yes" : "no") << "\n"
              << "  Is Unmanaged: " << (is_unmanaged_view<View>() ? "yes" : "no") << "\n";
}

} // namespace NVKokkos
```

**Usage:**
```cpp
// Define views with different memory traits
Kokkos::View<double*> normal_view("normal", N);
auto atomic_view = Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Atomic>>("atomic", N);
auto restrict_view = Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Restrict>>("restrict", N);
auto const_view = Kokkos::View<const double*>("const", N);

// Compile-time checks
static_assert(!NVKokkos::is_atomic_view<decltype(normal_view)>());
static_assert(NVKokkos::is_atomic_view<decltype(atomic_view)>());
static_assert(NVKokkos::is_restrict_view<decltype(restrict_view)>());
static_assert(NVKokkos::is_const_view<decltype(const_view)>());

// Use in function to check what was passed in
void process_view(const auto& view) {
    if constexpr (NVKokkos::is_atomic_view<decltype(view)>()) {
        std::cout << "Warning: All accesses will use atomics!\n";
    }
    
    if constexpr (!NVKokkos::is_restrict_view<decltype(view)>()) {
        std::cout << "Consider using Restrict trait for better optimization\n";
    }
    
    // Print full trait information
    NVKokkos::print_view_traits(view);
}

// Example output:
process_view(atomic_view);
// View: atomic
//   Memory Traits: Atomic 
//   Is Atomic: yes
//   Is Restrict: no
//   Is Const: no
//   Is Unmanaged: no
```

**Why This Matters:**

When passing views to functions, especially template functions, it's critical to know what traits are attached:

```cpp
// This function doesn't know what view traits it receives
template<typename ViewType>
void compute(ViewType data) {
    // Is 'data' atomic? If so, all accesses will be atomic (slow!)
    // Is 'data' restrict? If not, compiler can't optimize as aggressively
    // Is 'data' const? If so, writes will fail to compile
    
    // Use trait checks to provide helpful diagnostics
    static_assert(!NVKokkos::is_atomic_view<ViewType>(), 
                  "Don't pass Atomic view here - use explicit atomics in kernel instead");
    
    if constexpr (NVKokkos::is_restrict_view<ViewType>()) {
        // Can safely assume no aliasing
    }
}
```

See: [Kokkos `View` memory traits](https://kokkos.org/kokkos-core-wiki/API/core/view/view.html)

### Complete Helper Header

All utilities above are combined in a single header file `nvkokkos_helpers.hpp` included in the repository root. Use it directly:

```cpp
#include "nvkokkos_helpers.hpp"
```

This header brings in the `NVKokkos` namespace with all classes and functions described in this chapter. See `nvkokkos_helpers.hpp` for the full implementation.

**Benefits of Helper Library:**
- **No code changes to Kokkos**: Works alongside standard Kokkos
- **Portable**: No-ops or fallbacks for non-CUDA backends
- **Easy to use**: Simple function calls, clear names
- **Performance**: Enables optimizations not in standard Kokkos API
- **Type-safe**: Template metaprogramming for compile-time checks

See: `nvkokkos_helpers.hpp` in this repository

## Kokkos Tools Integration (stub)

This section will cover Kokkos profiling and tracing tools that complement the helpers above:

- **Kokkos Tracer**: captures kernel launches and memory operations for timeline analysis
- **Kokkos Combiner**: aggregates profiling output from multiple MPI ranks
- **NVTX Annotations**: integrating `nvtxRangePush`/`nvtxRangePop` with Kokkos kernel names for nsys recipe-based profiling workflows

See: [Kokkos Tools](https://github.com/kokkos/kokkos-tools)
