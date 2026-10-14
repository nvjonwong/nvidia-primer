<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kernel Performance Optimization

This chapter covers Kokkos kernel optimization techniques: memory access modifiers, occupancy tuning, atomic operations, multi-dimensional iteration, instruction-level parallelism, and memory prefetching.

## Writing Efficient Kokkos Kernels

### View Access Modifiers

**Atomic Operations:**

**Problem: Atomic Views Apply Atomicity to ALL Accesses**

When you mark a View with the `Atomic` memory trait, **all** accesses become atomic, even when atomicity isn't needed. This can cause significant performance degradation.

**CUDA Pattern (Fine-Grained Control):**
```cpp
int* global_data;

__global__ void kernel(int* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Regular read - fast
    int value = data[i];
    
    // Atomic operation only when needed - slower
    atomicAdd(&data[0], value);
    
    // Regular write - fast
    data[i] = value * 2;
}
```

**Kokkos with Atomic View (All Accesses Atomic - Slow):**
```cpp
// Marking view as Atomic makes ALL accesses atomic
Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Atomic>> atomic_view("data", N);

Kokkos::parallel_for("work", N, KOKKOS_LAMBDA(int i) {
    int value = atomic_view(i);  // ATOMIC read (slow!)
    
    Kokkos::atomic_add(&atomic_view(0), value);  // ATOMIC add (intended)
    
    atomic_view(i) = value * 2;  // ATOMIC write (slow!)
});
```

**Kokkos Optimized (Selective Atomics):**
```cpp
// Regular view - non-atomic by default
Kokkos::View<int*> data("data", N);

Kokkos::parallel_for("work", N, KOKKOS_LAMBDA(int i) {
    int value = data(i);  // Regular read - fast
    
    // Explicit atomic only when needed
    Kokkos::atomic_add(&data(0), value);  // ATOMIC add
    
    data(i) = value * 2;  // Regular write - fast
});
```

**Best Practices:**
- **Avoid** marking entire Views with `Kokkos::Atomic` trait
- **Recommended:** Use explicit `Kokkos::atomic_add()`, `Kokkos::atomic_fetch_add()`, etc.
- **Regular Views** with selective atomic operations perform much better
- Only use Atomic Views if every access truly needs to be atomic (rare)

**Restrict and Const: Enable Aggressive Compiler Optimizations**

**The Optimization**: `const` and `__restrict__` tell the compiler that pointers don't alias, enabling optimizations like:
- Loop vectorization
- Instruction reordering
- Register caching of values
- Elimination of redundant loads

**CUDA Pattern:**
```cpp
__global__ void compute(const double* __restrict__ input,
                       double* __restrict__ output,
                       int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compiler knows:
    // - input won't change (const)
    // - input and output don't overlap (__restrict__)
    // Can safely cache input[i] in registers and optimize aggressively
    double val = input[i];
    output[i] = val * val + val;
}
```

**Kokkos Pattern:**
```cpp
// Mark read-only input as const + Restrict
Kokkos::View<const double*, Kokkos::MemoryTraits<Kokkos::Restrict>> 
    input("input", N);

// Mark output as Restrict (no aliasing)
Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Restrict>> 
    output("output", N);

Kokkos::parallel_for("compute", N, KOKKOS_LAMBDA(int i) {
    // Compiler can optimize knowing no aliasing exists
    double val = input(i);
    output(i) = val * val + val;
});
```

**When to Use:**
- Use `const` for read-only Views - always safe and beneficial
- Use `Restrict` when Views don't alias (point to different memory)
- Easy to add - just modify the View declaration
- Don't use `Restrict` if Views might overlap or alias
- **Note:** Unmanaged + Restrict is useful for wrapping external pointers

**Example: External Pointer Wrapping**
```cpp
void process_external_data(double* external_ptr, int N) {
    // Wrap external pointer without taking ownership
    Kokkos::View<double*, 
        Kokkos::MemoryTraits<Kokkos::Restrict | Kokkos::Unmanaged>>
        wrapped(external_ptr, N);
    
    Kokkos::parallel_for("process", N, KOKKOS_LAMBDA(int i) {
        wrapped(i) *= 2.0;
    });
}
```

**Performance Benefit**:
- Benefits vary by workload - most noticeable in memory-bound kernels with complex expressions
- Compiler can better optimize when it knows memory regions don't overlap
- Particularly effective when:
  - Multiple Views are accessed in same kernel
  - Complex arithmetic operations on loaded values
  - Loops with potential for vectorization
- Profile to measure impact for your specific use case

See: [CUDA C Best Practices Guide - Use Restrict](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#use-restrict)

### MDRangePolicy for Better Coalescing

`Kokkos::MDRangePolicy` distributes a multi-dimensional iteration space across threads in a way that allows the runtime to choose a thread-tile mapping that promotes coalesced memory access.

**Problem: Nested Loops May Not Coalesce**

```cpp
// Poor coalescing - inner loop over i
Kokkos::parallel_for("nested", M, KOKKOS_LAMBDA(int j) {
    for(int i = 0; i < N; i++) {
        data(i, j) = i + j;  // Strided access if LayoutLeft
    }
});
```

**Better: MDRangePolicy**

```cpp
// Better coalescing - threads distributed across both dimensions
Kokkos::parallel_for("md_range",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}),
    KOKKOS_LAMBDA(int j, int i) {
        data(i, j) = i + j;  // Coalesced for LayoutLeft
    });

// With explicit tiling for cache locality - specify tile size in constructor
Kokkos::parallel_for("md_range_tiled",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}, {16, 16}),
    KOKKOS_LAMBDA(int j, int i) {
        data(i, j) = i + j;
    });
```

**Problem: Kokkos Default Tiling is Suboptimal**

Kokkos uses default tile sizes if you don't specify them:
- **2D default**: `<2, 16>` or `<2, 32>` depending on backend
- **3D default**: `<2, 4, 4>` or similar

**Issue with Defaults:**
```cpp
// 2D: Default tiling is okay but not great
Kokkos::parallel_for("2d",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}),
    KOKKOS_LAMBDA(int j, int i) { /* work */ });
// Uses <2, 16> or <2, 32> - only 32 or 64 threads per block
// Not enough threads to hide latency

// 3D: Default tiling is poor
Kokkos::parallel_for("3d",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {X, Y, Z}),
    KOKKOS_LAMBDA(int k, int j, int i) { /* work */ });
// Uses <2, 4, 4> - only 32 threads per block
// Very low occupancy
```

**Recommended: Use {1, 64} or {1, 128} as Better Default**

Specify tile sizes directly as the third argument to `MDRangePolicy`:

```cpp
// 2D with recommended tiling: 64 threads per block
Kokkos::parallel_for("2d_optimized",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}, {1, 64}),
    KOKKOS_LAMBDA(int j, int i) { /* work */ });

// 3D with recommended tiling: 64 threads per block
Kokkos::parallel_for("3d_optimized",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {X, Y, Z}, {1, 1, 64}),
    KOKKOS_LAMBDA(int k, int j, int i) { /* work */ });

// 4D with recommended tiling: 64 threads per block
Kokkos::parallel_for("4d_optimized",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {W, X, Y, Z}, {1, 1, 1, 64}),
    KOKKOS_LAMBDA(int l, int k, int j, int i) { /* work */ });
```

For convenience, use the helper functions from `nvkokkos_helpers.hpp`:

```cpp
#include "nvkokkos_helpers.hpp"

// 2D
Kokkos::parallel_for("2d",
    NVKokkos::make_mdrange_2d({0, 0}, {M, N}),
    KOKKOS_LAMBDA(int j, int i) { /* work */ });

// 3D
Kokkos::parallel_for("3d",
    NVKokkos::make_mdrange_3d({0, 0, 0}, {X, Y, Z}),
    KOKKOS_LAMBDA(int k, int j, int i) { /* work */ });

// 4D
Kokkos::parallel_for("4d",
    NVKokkos::make_mdrange_4d({0, 0, 0, 0}, {W, X, Y, Z}),
    KOKKOS_LAMBDA(int l, int k, int j, int i) { /* work */ });
```

**Why {1, 64} or {1, 128} Works Better:**
- **More threads per block**: 64-128 vs 32-64 from defaults
- **Better occupancy**: More warps to hide latency
- **Works for 2D and 3D**: Single pattern that works well for both
- **Coalescing preserved**: Innermost dimension still consecutive threads
- **Note:** May need tuning: Profile and adjust for your specific kernel

**Occupancy Comparison:**
| Configuration | Threads/Block | Warps/Block | Good For |
|---------------|---------------|-------------|----------|
| Default 2D `<2,16>` | 32 | 1 | **Avoid:** Too few threads |
| Default 2D `<2,32>` | 64 | 2 | **Note:** Marginal |
| Default 3D `<2,4,4>` | 32 | 1 | **Avoid:** Too few threads |
| Recommended `{1,64}` | 64 | 2 | **Good:** 2D and 3D |

## Instruction-Level Parallelism (ILP): Avoid Global Memory Dependencies

Instruction-level parallelism (ILP) is the ability of a processor to execute multiple instructions simultaneously. On GPU hardware, ILP is limited when a thread must wait for a high-latency global memory round-trip before it can proceed.

**The Problem: Read-After-Write Dependency on Global Memory**

When you update a value in global memory and immediately read it back for a conditional, you create a high-latency dependency:

**CUDA Anti-Pattern (High Latency):**
```cpp
__global__ void update_kernel(double* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update value in global memory
    data[i] = data[i] + compute_delta(i);
    
    // Immediately read back from global for conditional - inefficient
    if(data[i] > threshold) {
        // More work...
    }
    // Waiting for global memory round-trip creates pipeline stall
}
```

**The latency chain:**
1. Read `data[i]` from global memory (~400-800 cycles)
2. Compute delta
3. Write result to global memory (~400-800 cycles)
4. **Wait for write to complete**
5. Read `data[i]` back from global memory (~400-800 cycles)
6. Check conditional

Total: ~1200-2400 cycles of latency

**CUDA Optimized (Low Latency):**
```cpp
__global__ void update_kernel(double* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double value = data[i];        // Read once
    double delta = compute_delta(i);
    double new_value = value + delta;
    
    data[i] = new_value;           // Write to global
    
    // Use register value in conditional - efficient
    if(new_value > threshold) {
        // More work...
    }
    // No global memory dependency - pipeline continues
}
```

**Kokkos Anti-Pattern:**
```cpp
Kokkos::View<double*> data("data", N);

Kokkos::parallel_for("update", N, KOKKOS_LAMBDA(int i) {
    // Update in place
    data(i) = data(i) + compute_delta(i);
    
    // Immediately use updated value - inefficient
    if(data(i) > threshold) {  // Read-after-write from global
        // More work...
    }
});
```

**Kokkos Optimized:**
```cpp
Kokkos::View<double*> data("data", N);

Kokkos::parallel_for("update", N, KOKKOS_LAMBDA(int i) {
    // Load to register once
    double value = data(i);
    double delta = compute_delta(i);
    double new_value = value + delta;
    
    // Write to global memory
    data(i) = new_value;
    
    // Use register value - no global memory dependency
    if(new_value > threshold) {
        // More work...
    }
});
```

**More Complex Example: Multiple Updates**

**Before (Multiple Global Memory Dependencies):**
```cpp
Kokkos::parallel_for("complex", N, KOKKOS_LAMBDA(int i) {
    data(i) += delta1(i);
    if(data(i) > threshold1) {      // Read 1
        data(i) *= scale;
        if(data(i) < threshold2) {  // Read 2
            data(i) += delta2(i);
        }
    }
    result(i) = data(i);            // Read 3
});
// Three round-trips to global memory
```

**After (Register-Based Logic):**
```cpp
Kokkos::parallel_for("complex", N, KOKKOS_LAMBDA(int i) {
    // Load once, work in registers
    double value = data(i);
    
    value += delta1(i);
    if(value > threshold1) {
        value *= scale;
        if(value < threshold2) {
            value += delta2(i);
        }
    }
    
    // Write final result once
    data(i) = value;
    result(i) = value;
});
// Only one read, one write - more efficient
```

**Performance Benefits:**
- **Eliminates global memory round-trips**: Work stays in registers
- **Reduces instruction latency**: No waiting for memory subsystem
- **Improves ILP**: More independent instructions for scheduler
- **Better register usage**: Compiler can optimize register allocation

**When to Apply:**
- **Recommended for:** Any time you update and immediately check the same value
- **Recommended for:** Multiple sequential updates to same variable
- **Recommended for:** Complex conditionals based on updated values
- Watch register pressure — profile occupancy

**Profiling Tip:**

In Nsight Compute, look for:
- High `Stall Long Scoreboard` - indicates waiting for memory operations
- Low ILP (Instructions per cycle) - indicates sequential dependencies
- Memory replay - repeated accesses to same address


## Memory Prefetching for UVM

**Problem: UVM Page Faults Can Be Expensive**

```cpp
// CUDA: Explicit prefetch
cudaMemPrefetchAsync(data, size, deviceId, stream);

// Kokkos: Manual prefetch for UVM views
template<typename View>
void prefetch_to_device(const View& view, const Kokkos::Cuda& space) {
    if constexpr (std::is_same_v<typename View::memory_space, 
                                  Kokkos::CudaUVMSpace>) {
        cudaMemPrefetchAsync(view.data(), 
                            view.size() * sizeof(typename View::value_type),
                            space.cuda_device(),
                            space.cuda_stream());
    }
}

// Usage
Kokkos::View<double*, Kokkos::CudaUVMSpace> data("data", N);
Kokkos::Cuda exec_space;

prefetch_to_device(data, exec_space);  // Hint: bring to GPU

Kokkos::parallel_for("work",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
    KOKKOS_LAMBDA(int i) { data(i) *= 2.0; });
```

See also:
- [06-synchronization-reference.md](06-synchronization-reference.md) — comprehensive reference for when Kokkos operations synchronize
- [08-helper-library.md](08-helper-library.md) — `NVKokkos::make_mdrange_*` helpers for optimal MDRange tiling
