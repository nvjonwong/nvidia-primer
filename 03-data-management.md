<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Memory Management: Kokkos Views and GPU Memory Spaces

This chapter covers practical Kokkos patterns for GPU memory management, including View creation, initialization strategies, unified memory (UVM), host-pinned memory, and asynchronous allocation.

## Data Management with Kokkos Views

### View Creation and Initialization

**Problem: Default View Creation is Expensive**

Kokkos Views by default initialize memory to zero, which causes implicit synchronization:

**CUDA Pattern:**
```cpp
// Explicit control over initialization
float *d_data;
cudaMalloc(&d_data, N * sizeof(float));
// No initialization - fast but contains garbage

// Or explicit memset when needed
cudaMemset(d_data, 0, N * sizeof(float));
```

**Kokkos Default (Slow):**
```cpp
// Implicitly zeros memory AND synchronizes the device
Kokkos::View<double*> data("data", N);  
// Causes cudaMemset + cudaDeviceSynchronize()
```

**Kokkos Optimized:**
```cpp
// Skip initialization when you'll overwrite the data
Kokkos::View<double*> data(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);

// Or for managed memory
Kokkos::View<double*, Kokkos::CudaUVMSpace> data_uvm(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "data_uvm"), N);
```

**When to Use Each:**
- **Recommended:** Use `WithoutInitializing` for temporary arrays or arrays that will be fully overwritten
- Use default initialization for output arrays or partial writes to avoid undefined behavior

### View Deallocation Overhead

**Problem: UVM Deallocation Synchronizes Twice**

Kokkos UVM View destruction causes unnecessary synchronization:

```cpp
{
    Kokkos::View<double*, Kokkos::CudaUVMSpace> temp("temp", N);
    // ... use temp ...
}  // Destructor:
   // 1. cudaDeviceSynchronize()  <- unnecessary
   // 2. cudaFree(temp.data())
   // 3. cudaDeviceSynchronize()  <- unnecessary
```

**Workarounds:**
1. **Reuse allocations** instead of creating/destroying in hot loops
2. **Use device memory** (`CudaSpace`) instead of UVM when possible
3. **Batch allocations** outside of timing-critical sections
4. **Advanced: Use UnManaged Views** with manual `cudaMallocAsync`/`cudaFreeAsync` to eliminate all synchronization (see [Advanced: Truly Asynchronous Allocation](#advanced-truly-asynchronous-allocation-with-unmanaged-views))

**Potential Kokkos improvement:**
- Do not fence for the user when it doesn't match the CUDA stream execution model.

**CUDA Comparison:**
```cpp
// CUDA: cudaFree also synchronizes (implicit)
cudaFree(d_data);  // Synchronizes device

// CUDA 11.2+: Truly asynchronous deallocation
cudaFreeAsync(d_data, stream);  // No sync, queued on stream
```

**Note**: Both `cudaFree` and Kokkos View destruction synchronize. Use `cudaFreeAsync` for truly asynchronous allocation/deallocation patterns.

### Asynchronous Memory Allocation (cudaMallocAsync)

`cudaMallocAsync` and `cudaFreeAsync` (CUDA 11.2+) are stream-ordered memory allocation functions that queue allocation and deallocation operations on a CUDA stream rather than synchronizing the device immediately.

**Problem: cudaMalloc/cudaFree Cause Implicit Synchronization**

Traditional CUDA memory allocation functions synchronize the device, creating performance bottlenecks:

```cpp
// CUDA: cudaMalloc synchronizes implicitly
cudaMalloc(&d_data, size);  // Blocks until all GPU work completes

// CUDA: cudaFree also synchronizes
cudaFree(d_data);  // Blocks until all GPU work completes
```

This synchronization prevents overlapping memory allocation with computation and creates pipeline stalls.

**Solution: cudaMallocAsync/cudaFreeAsync (CUDA 11.2+)**

Asynchronous memory operations queue on streams without synchronization:

```cpp
// CUDA 11.2+: Truly asynchronous allocation
cudaMallocAsync(&d_data, size, stream);  // No sync, queued on stream
// ... launch kernels on stream ...
cudaFreeAsync(d_data, stream);  // No sync, queued on stream
```

**Benefits:**
- No implicit device synchronization
- Memory operations queue on stream like kernels
- Enables overlapping allocation with computation
- Better pipelining in multi-stream applications

**Kokkos Support: KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC**

Kokkos supports `cudaMallocAsync` and `cudaFreeAsync` via a CMake configuration option that is off by default.

**Enable in CMake:**
```bash
# When building Kokkos
cmake .. \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON \
  -DCMAKE_CXX_COMPILER=<kokkos>/bin/nvcc_wrapper

# Requires CUDA Toolkit 11.2 or later
```

**How Kokkos Uses cudaMallocAsync:**

When enabled, Kokkos uses `cudaMallocAsync` selectively based on allocation size:

Source: [`core/src/Cuda/Kokkos_CudaSpace.cpp` (Kokkos 5.0.2)](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_CudaSpace.cpp#L164-L232) — `impl_allocate_common`, simplified

```cpp
// Allocation
if (arg_alloc_size >= memory_threshold && 
    device_supports_cuda_malloc_async) {
    cudaMallocAsync(&ptr, arg_alloc_size, stream);  // Large allocations
} else {
    cudaMalloc(&ptr, arg_alloc_size);  // Small allocations
}

// Deallocation
if (arg_alloc_size >= memory_threshold) {
    cudaDeviceSynchronize();  // Still synchronizes before free!
    cudaFreeAsync(arg_alloc_ptr, stream);
    cudaDeviceSynchronize();  // Still synchronizes after free!
} else {
    cudaFree(arg_alloc_ptr);  // Synchronous
}
```

**Current Limitations:**

**Why OFF by default** (as of September 2024, see [`cmake/kokkos_enable_options.cmake` L36–38](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/cmake/kokkos_enable_options.cmake#L36-L38)):
- Known issues with Intel Cluster Checker (ICP) when enabled
- Compatibility problems with older UCX versions used as MPI communication layers

**Allocation synchronizes when using execution space:**
- When allocating a View with an execution space (e.g., `Kokkos::View<int> view(exec_space, ...)`), Kokkos passes `stream_sync_only=true` to the allocation function
- If `cudaMallocAsync` is used (for large allocations), this triggers `cudaStreamSynchronize(stream)` immediately after allocation
- This happens even for device-accessible Views that don't require synchronization
- See: [`core/src/Cuda/Kokkos_CudaSpace.cpp` L211–216](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_CudaSpace.cpp#L211-L216) — `impl_allocate_common(..., true)` vs `impl_allocate_common(..., false)`
- **Impact:** View allocation between kernels can cause unexpected synchronization, breaking asynchronous execution patterns

**Deallocation still synchronizes:**
- Even with `cudaMallocAsync` enabled, Kokkos calls `cudaDeviceSynchronize()` before and after `cudaFreeAsync`
- Similarly, allocation with `cudaMallocAsync` is followed by either `cudaStreamSynchronize()` or `cudaDeviceSynchronize()`
- This limits the benefit of asynchronous memory operations
- Synchronization added for safety/compatibility reasons

**Memory threshold:**
- Only allocations above a threshold use async allocation
- Small allocations still use synchronous `cudaMalloc`
- Threshold is hardcoded: 40 KB (40,000 bytes)
- See: [`core/src/Cuda/Kokkos_CudaSpace.cpp` L139](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_CudaSpace.cpp#L139) — `size_t memory_threshold_g = 40000;`
- Not currently configurable at runtime

**Performance Recommendations:**

1. **For new applications:** Try enabling `KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC` and profile
   - Verify no MPI communication issues (especially with UCX)
   - Measure allocation overhead with `nsys profile`
   - Look for reduced `cudaDeviceSynchronize` calls

2. **Combine with `WithoutInitializing`:** Maximum benefit when avoiding both sync sources
   ```cpp
   // Best: No initialization sync + async allocation
   Kokkos::View<double*> data(
       Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
   ```

3. **Allocation reuse still best:** Even with async allocation, reusing Views is faster than recreating
   ```cpp
   // Best practice: Allocate once, reuse many times
   Kokkos::View<double*> workspace("work", max_size);
   for(int i = 0; i < iterations; i++) {
       // Reuse workspace - no allocation overhead
   }
   ```

4. **Profile to verify benefit:** Use Nsight Systems to check if synchronizations are reduced
   ```bash
   nsys profile --stats=true ./my_kokkos_app
   # Look for reduction in cudaDeviceSynchronize calls
   ```

5. **For zero-sync allocation:** Use `StreamOrderedView` (NVIDIA-managed Kokkos UnManaged views) for truly asynchronous allocation/deallocation. See section [Advanced: Truly Asynchronous Allocation with UnManaged Views](#advanced-truly-asynchronous-allocation-with-unmanaged-views) for implementation details.

**Example: Profiling Allocation Overhead**

```bash
# Test suite includes allocation profiling
cd test_suite
./build.sh
./profile_test.sh profile "ExecutionSpace*" 
./profile_test.sh timeline "ExecutionSpace*" > timeline.txt

# Look for:
# - cudaMalloc vs cudaMallocAsync calls
# - cudaDeviceSynchronize associated with allocation
# - cudaMemset for View initialization
```

See `test_suite/test_execution_spaces.cpp` for examples using `WithoutInitializing` to eliminate initialization overhead.

### Advanced: Truly Asynchronous Allocation with UnManaged Views

**Problem: Kokkos's cudaMallocAsync Still Synchronizes**

Even when `KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC` is enabled, Kokkos's implementation includes synchronization calls (see [`core/src/Cuda/Kokkos_CudaSpace.cpp` L211–216 and L362–376](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_CudaSpace.cpp#L211-L216)):

```cpp
// Allocation:
cudaMallocAsync(&ptr, arg_alloc_size, stream);
if (stream_sync_only) {
    cudaStreamSynchronize(stream);     // SYNC after alloc
} else {
    cuda_device_synchronize(...);      // SYNC after alloc
}

// Deallocation:
cuda_device_synchronize(...);          // SYNC before free
cudaFreeAsync(arg_alloc_ptr, stream);
cuda_device_synchronize(...);          // SYNC after free
```

These synchronizations were added for safety and compatibility, but they negate the performance benefits of stream-ordered allocations and may encourage users to find more complex patterns of pipelining instead of the canonical tried-and-true stream-ordering that CUDA provides.

**Solution: Manual Memory Management with UnManaged Views**

You can bypass Kokkos's allocation/deallocation synchronization by:
1. Manually allocating memory with `cudaMallocAsync` (no sync)
2. Wrapping the raw pointer in an **UnManaged** Kokkos View
3. Manually freeing with `cudaFreeAsync` when done (no sync)

**Implementation:**

```cpp
#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>

void example_stream_ordered_allocation() {
    const size_t N = 1000000;
    
    // Get CUDA stream from Kokkos execution space
    Kokkos::Cuda cuda_instance;
    cudaStream_t stream = cuda_instance.cuda_stream();
    
    // Step 1: Manually allocate with cudaMallocAsync (NO SYNC!)
    double* raw_ptr = nullptr;
    cudaMallocAsync(&raw_ptr, N * sizeof(double), stream);
    // Note: No cudaStreamSynchronize() here
    
    // Step 2: Wrap in UnManaged View
    // The Unmanaged trait tells Kokkos not to manage memory lifecycle
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
        view_unmanaged(raw_ptr, N);
    
    // Step 3: Use view normally with Kokkos operations
    Kokkos::parallel_for("init", 
        Kokkos::RangePolicy<Kokkos::Cuda>(cuda_instance, 0, N),
        KOKKOS_LAMBDA(int i) {
            view_unmanaged(i) = i * 2.0;
        });
    
    Kokkos::parallel_for("modify",
        Kokkos::RangePolicy<Kokkos::Cuda>(cuda_instance, 0, N),
        KOKKOS_LAMBDA(int i) {
            view_unmanaged(i) *= 3.0;
        });
    
    // All operations are stream-ordered
    // No implicit synchronization between allocation and kernels
    
    // Step 4: Only sync when you need results
    cuda_instance.fence();  // or cudaStreamSynchronize(stream);
    
    // Step 5: Manual deallocation (NO SYNC!)
    cudaFreeAsync(raw_ptr, stream);
    // The free is queued on the stream, will execute after all prior ops
    
    // View destruction does NOT sync (because it's UnManaged)
}
```

**Benefits:**

- **Zero synchronization overhead** from allocation/deallocation
- **Stream-ordered execution** maintained throughout
- **Works with existing Kokkos algorithms** (parallel_for, parallel_reduce, etc.)
- **Can pass UnManaged Views** to Kokkos functions expecting regular Views

**Comparison:**

```cpp
// Standard Kokkos View (with KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC)
{
    Kokkos::View<double*> v("v", N);
    // cudaMallocAsync + cudaStreamSynchronize (SYNC!)
    
    Kokkos::parallel_for(...);
    
} // Destructor:
  // cudaDeviceSynchronize() (SYNC!)
  // cudaFreeAsync()
  // cudaDeviceSynchronize() (SYNC!)
  // Total: 3 synchronizations

// UnManaged View with manual cudaMallocAsync/cudaFreeAsync
{
    double* ptr;
    cudaMallocAsync(&ptr, N * sizeof(double), stream);  // No sync
    auto v = Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, N);
    
    Kokkos::parallel_for(...);
    
    cudaFreeAsync(ptr, stream);  // No sync
} // Destructor: Nothing happens (UnManaged)
  // Total: 0 synchronizations
```

**Safety Considerations:**

**Lifetime management**: You are responsible for ensuring the memory isn't freed while kernels are using it. Use stream synchronization (`fence()`) before accessing results on host, and track pointer lifetimes carefully.

**Stream ordering**: All operations must use the same stream. Pass the execution space instance to ensure stream consistency:
  ```cpp
  Kokkos::parallel_for("kernel",
      Kokkos::RangePolicy<Kokkos::Cuda>(cuda_instance, 0, N),  // ✓ Use same instance
      KOKKOS_LAMBDA(int i) { ... });
  ```

**Initialization**: UnManaged views don't initialize memory. Use `Kokkos::parallel_for` to initialize, or ensure `cudaMallocAsync` returns zeroed memory.

**RAII Helper Class (Recommended):**

To eliminate the manual memory management burden while keeping zero-sync performance, use the `StreamOrderedView` RAII wrapper. It provides automatic memory management (just like standard Kokkos Views) while maintaining stream-ordered, zero-sync allocation/deallocation:

```cpp
// RAII wrapper for stream-ordered View allocations
template <typename DataType, typename... Properties>
class StreamOrderedView {
private:
    using value_type = typename Kokkos::View<DataType, Properties...>::value_type;
    using unmanaged_view_type = Kokkos::View<DataType, 
                                              Kokkos::CudaSpace,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    
    value_type* raw_ptr_;
    size_t size_bytes_;
    cudaStream_t stream_;
    unmanaged_view_type view_;
    
public:
    // Constructor: allocates memory with cudaMallocAsync (no sync!)
    template <typename... Dims>
    StreamOrderedView(cudaStream_t stream, Dims... dims) 
        : raw_ptr_(nullptr), stream_(stream) {
        
        // Calculate total size
        size_t total_elements = (dims * ...);
        size_bytes_ = total_elements * sizeof(value_type);
        
        // Allocate with cudaMallocAsync (NO SYNC!)
        cudaError_t err = cudaMallocAsync(&raw_ptr_, size_bytes_, stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMallocAsync failed");
        }
        
        // Create UnManaged view wrapping the allocation
        view_ = unmanaged_view_type(raw_ptr_, dims...);
    }
    
    // Destructor: frees memory with cudaFreeAsync (no sync!)
    ~StreamOrderedView() {
        if (raw_ptr_ != nullptr) {
            cudaFreeAsync(raw_ptr_, stream_);  // NO SYNC
            // Deallocation queued on stream, executes after prior operations
        }
    }
    
    // Delete copy operations (move-only type)
    StreamOrderedView(const StreamOrderedView&) = delete;
    StreamOrderedView& operator=(const StreamOrderedView&) = delete;
    
    // Move operations
    StreamOrderedView(StreamOrderedView&& other) noexcept 
        : raw_ptr_(other.raw_ptr_)
        , size_bytes_(other.size_bytes_)
        , stream_(other.stream_)
        , view_(other.view_) {
        other.raw_ptr_ = nullptr;
    }
    
    StreamOrderedView& operator=(StreamOrderedView&& other) noexcept {
        if (this != &other) {
            // Free our current allocation
            if (raw_ptr_ != nullptr) {
                cudaFreeAsync(raw_ptr_, stream_);
            }
            // Take ownership of other's allocation
            raw_ptr_ = other.raw_ptr_;
            size_bytes_ = other.size_bytes_;
            stream_ = other.stream_;
            view_ = other.view_;
            other.raw_ptr_ = nullptr;
        }
        return *this;
    }
    
    // Get the UnManaged Kokkos::View
    unmanaged_view_type view() const { return view_; }
    
    // Convenient access operators
    unmanaged_view_type operator*() const { return view_; }
    unmanaged_view_type operator->() const { return view_; }
    
    // Implicit conversion to view
    operator unmanaged_view_type() const { return view_; }
    
    // Get raw pointer (use with caution)
    value_type* data() const { return raw_ptr_; }
    
    // Get stream
    cudaStream_t stream() const { return stream_; }
};

// Usage example:
Kokkos::Cuda exec_space;

// Create stream-ordered view (RAII managed)
StreamOrderedView<double*> managed_view(exec_space.cuda_stream(), N);

// Get the UnManaged Kokkos::View
auto view = managed_view.view();  // or just: auto view = *managed_view;

// Use with Kokkos operations
Kokkos::parallel_for("kernel",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
    KOKKOS_LAMBDA(int i) {
        view(i) = i * 2.0;
    });

// Sync when needed
exec_space.fence();

// Automatic cleanup when managed_view goes out of scope (cudaFreeAsync, no sync!)
```

**Simpler Usage:**

```cpp
void process_data(Kokkos::Cuda& exec_space, size_t N) {
    // Allocate (no sync!)
    StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
    
    // Use directly (implicit conversion)
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i) {
            data.view()(i) = i;
        });
    
    Kokkos::parallel_reduce("sum",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
        KOKKOS_LAMBDA(int i, double& sum) {
            sum += data.view()(i);
        }, result);
    
    exec_space.fence();
    
    // Automatic cleanup (no sync!)
}
```

**When to Use This Approach:**

**Recommended for:**
- High-performance libraries requiring maximum control
- Applications with many temporary allocations in hot loops
- Multi-stream pipelines where synchronization kills performance
- Profiling shows allocation/deallocation sync overhead is significant
- Code that already uses execution spaces explicitly (stream-aware)

**Not suitable for:**
- Code that needs to be portable across backends (CUDA-specific)
- Simple applications where allocation overhead is negligible
- When you're unsure about stream/execution space semantics

`StreamOrderedView` is fully RAII — it handles memory management automatically and safely, just like standard Kokkos Views. The memory is tied to a specific stream/execution space, so operations must use the correct execution space. This is not "unsafe" — it's a requirement for correctness with stream-ordered operations. Standard Views work with any execution space but pay a synchronization penalty.

**Complete Example:**

See these files for working demonstrations:
- **`nvkokkos_helpers.hpp`** - Single header library with `NVKokkos::StreamOrderedView` and other utilities
- **`test_suite/stream_ordered_allocation_example.cpp`** - Complete example comparing:
  - Method 1: RAII StreamOrderedView (recommended)
  - Method 2: operator* shorthand for convenience
  - Method 3: Manual cudaMallocAsync/cudaFreeAsync (maximum control)
  - Standard Kokkos Views (for comparison, shows sync overhead)
- **`STREAM_ORDERED_ALLOCATION_WORKAROUND.md`** - Quick reference guide

**Using in Your Project:**

Simply include the helper library:
```cpp
#include "nvkokkos_helpers.hpp"

// Now you can use it:
Kokkos::Cuda exec_space;
NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
auto view = data.view();
```

**Profiling the Difference:**

```bash
# Profile standard Kokkos allocations
nsys profile --trace=cuda,nvtx ./app_standard_views

# Profile UnManaged with manual allocation
nsys profile --trace=cuda,nvtx ./app_unmanaged_views

# Compare cudaDeviceSynchronize counts:
nsys stats standard.nsys-rep --report cuda_api_sum | grep Synchronize
nsys stats unmanaged.nsys-rep --report cuda_api_sum | grep Synchronize
```

This workaround exists because Kokkos prioritizes safety and compatibility over raw performance in its allocator.

## Memory Spaces and Access Patterns

### Aliasing and Data Hazards

**Problem: View Aliasing Can Cause Race Conditions**

```cpp
// DANGEROUS: Same underlying data used as const and mutable
Kokkos::View<double*> data1("avg", N);

// In different kernels/functions:
void compute1(const Kokkos::View<double*>& data_read) {
    // Read from data_read
}

void compute2(Kokkos::View<double*>& data_write) {
    // Write to data_write  
}

// Called with same view - compiler may optimize incorrectly
compute1(data1);  // Expects const
compute2(data1);  // Modifies data
```

**Best Practice:**
```cpp
// Create an actual copy in the same memory space
Kokkos::View<double*> read_only_copy("copy", data1.extent(0));
Kokkos::deep_copy(read_only_copy, data1);  // Real copy

// Pass the copy to read-only function
compute1(read_only_copy);  // Safe - uses copy
compute2(data1);            // Original can be modified

// Or better: Use different views from the start
Kokkos::View<double*> data_in("input", N);
Kokkos::View<double*> data_out("output", N);
```

**Note**: `create_mirror_view_and_copy` only creates a real copy if memory spaces differ. For same-space copies, use explicit allocation + `deep_copy`.

See also:
- [04-execution-and-streams.md](04-execution-and-streams.md) — stream-ordered execution patterns that use these allocation strategies
- [08-helper-library.md](08-helper-library.md) — `NVKokkos::StreamOrderedView` and `make_view_uninitialized` helpers
