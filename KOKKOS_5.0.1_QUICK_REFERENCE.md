<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kokkos 5.0.1 Synchronization Quick Reference

## Confirmed: All Issues from 4.x Still Present

### Issue Summary with Line Numbers

| Problem | File | Lines | Fix |
|---------|------|-------|-----|
| **cudaMallocAsync syncs after alloc** | `Kokkos_CudaSpace.cpp` | 212, 214-215 | Use `StreamOrderedView` |
| **cudaFreeAsync syncs twice** | `Kokkos_CudaSpace.cpp` | 363, 367 | Use `StreamOrderedView` |
| **40KB threshold hardcoded** | `Kokkos_CudaSpace.cpp` | 139 | N/A (Kokkos limitation) |
| **parallel_reduce to host scalar syncs** | `Kokkos_Cuda_Parallel_Range.hpp` | 346-365 | Use device View or pinned memory |
| **deep_copy without exec space syncs** | `Kokkos_CudaSpace.cpp` | 73 | Pass exec space |
| **cuda_device_synchronize syncs ALL GPUs** | `Kokkos_Cuda_Instance.cpp` | N/A | Use exec space instances |

---

## Quick Code Patterns (Kokkos 5.0.1)

### **Avoid:** Default View Allocation

This allocates a zeroed View and forces a host-device synchronization:

```cpp
Kokkos::View<double*> data("data", N);  
// Zeros memory + synchronizes (avoid)
```

### **Recommended:** Skip Initialization

This skips zeroing and avoids the synchronization:

```cpp
Kokkos::View<double*> data(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "data"), N);
// No sync (recommended)
```

### **Best:** Zero-Sync Allocation

`StreamOrderedView` provides zero synchronization on both allocation and deallocation:

```cpp
NVKokkos::StreamOrderedView<double*> data(exec_space.cuda_stream(), N);
// Zero sync on alloc AND dealloc (best)
```

---

### **Avoid:** Reduce to Host Scalar

Reducing to a host scalar forces an immediate device-wide synchronization before returning:

```cpp
int sum = 0;
Kokkos::parallel_reduce("sum", N, lambda, sum);  
// Immediate device-wide sync (avoid)
```

### **Recommended:** Reduce to Device View

Reducing to a device View queues the reduction on the stream without synchronizing:

```cpp
Kokkos::View<int> sum(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));
Kokkos::parallel_reduce("sum", N, lambda, sum);  
// No sync, can queue more work (recommended)
```

### **Best:** Reduce to Pinned Memory

Reducing to pinned memory provides zero-sync reduction while keeping the result host-accessible after a fence:

```cpp
Kokkos::View<int, Kokkos::CudaHostPinnedSpace> sum(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));
Kokkos::parallel_reduce("sum", N, lambda, sum);  
// No sync, result accessible on host after fence (best)
// This is the RECOMMENDED approach, not a workaround!
```

**Note**: `CudaHostPinnedSpace` is device-accessible, so the GPU writes directly to it without synchronization. This is correct behavior by design, not a bug being worked around.

---

### **Avoid:** deep_copy Without Execution Space

A `deep_copy` without an execution space implicitly synchronizes the entire device:

```cpp
Kokkos::deep_copy(host_view, device_view);  
// Syncs entire device (avoid)
```

### **Recommended:** deep_copy With Execution Space

Passing an execution space binds the copy to a stream and avoids the synchronization:

```cpp
Kokkos::Cuda exec_space;
Kokkos::deep_copy(exec_space, host_view, device_view);
// Async on stream, no sync (recommended)
```

---

## Key Findings

### Allocation (lines 200-221 of Kokkos_CudaSpace.cpp)
- **Large allocations (≥40KB)** using `cudaMallocAsync`:
  - **With execution space**: `cudaStreamSynchronize(stream)` after alloc
  - **Without execution space**: `cuda_device_synchronize()` (syncs ALL GPUs!) after alloc
- **Small allocations (<40KB)**: Use synchronous `cudaMalloc` (no explicit sync needed, implicitly synchronous)

**Note**: Kokkos consistently pairs allocation/deallocation methods:
- `cudaMalloc` ↔ `cudaFree` for small allocations
- `cudaMallocAsync` ↔ `cudaFreeAsync` for large allocations
- Per NVIDIA docs, `cudaFree` on async-allocated memory would also work and not synchronize

### Deallocation (lines 362-370 of Kokkos_CudaSpace.cpp)
- **Large allocations (≥40KB)**: 
  - `cuda_device_synchronize()` BEFORE `cudaFreeAsync`
  - `cuda_device_synchronize()` AFTER `cudaFreeAsync`
  - **Total: 2 full device synchronizations per deallocation**
  - Each `cuda_device_synchronize()` syncs **ALL CUDA devices**
- **Small allocations (<40KB)**: 
  - Use synchronous `cudaFree` (pairs with `cudaMalloc`)
  - 1 implicit synchronization

### cuda_device_synchronize (Kokkos_Cuda_Instance.cpp)

The implementation iterates over all registered CUDA devices:

```cpp
void cuda_device_synchronize(const std::string &name) {
  for (int cuda_device : Kokkos::Impl::CudaInternal::cuda_devices) {
    cudaSetDevice(cuda_device);
    cudaDeviceSynchronize();  // Syncs every GPU in system (avoid)
  }
}
```

---

## What Changed in Kokkos 5.0

### New Feature: __grid_constant__ Launch
- Changed kernel launch mechanism
- "Can lead to subtle synchronization behavior change for mid-size functors"
- **Action Required**: Profile your application on 5.0 vs 4.x
- Check for new synchronization patterns in Nsight Systems

### What Didn't Change
- All synchronization issues from 4.x remain
- All best practices from NvidiaKokkosPrimer.md still apply
- Memory threshold still hardcoded at 40KB

**Note**: Using `CudaHostPinnedSpace` for reductions is not a "workaround" but the recommended best practice for zero-sync reductions with host accessibility.

---

## Verification Commands

Check your code for synchronization issues:

```bash
cd /path/to/kokkos/src/core/src/Cuda

# Check allocation synchronization
grep -n "cudaMallocAsync\|cudaStreamSynchronize" Kokkos_CudaSpace.cpp

# Check deallocation synchronization  
sed -n '362,370p' Kokkos_CudaSpace.cpp

# Check memory threshold
grep -n "memory_threshold_g" Kokkos_CudaSpace.cpp

# Check cuda_device_synchronize
grep -A 20 "void cuda_device_synchronize" Kokkos_Cuda_Instance.cpp
```

---

## Profiling Checklist

Use Nsight Systems to verify zero-sync optimizations:

```bash
nsys profile --trace=cuda,nvtx ./my_app

# Check for:
# 1. cudaDeviceSynchronize count (should be minimal)
# 2. cudaStreamSynchronize on allocation/deallocation
# 3. Memory operations between kernels
# 4. Grid constant launch patterns (new in 5.0)
```

Expected with optimizations:
- **Target:** Zero `cudaDeviceSynchronize` calls except explicit fences
- **Target:** Zero `cudaStreamSynchronize` on alloc/dealloc
- **Target:** Async operations queue on stream without gaps

---

## Bottom Line

**Kokkos 5.0.1 requires the SAME best practices as 4.x for optimal performance:**

1. Use `WithoutInitializing` for temp arrays
2. Use `StreamOrderedView` for zero-sync alloc/dealloc
3. Reduce to device Views (`CudaSpace`) or pinned memory (`CudaHostPinnedSpace`), not host scalars
4. Pass execution space to `deep_copy()`
5. Use execution space instances everywhere
6. **NEW**: Profile for grid constant launch impact

**Important**: Using `CudaHostPinnedSpace` for reductions is the recommended best practice, not a workaround. It provides zero-sync reduction with host accessibility - this is correct behavior by design.

See `KOKKOS_5.0.1_SYNCHRONIZATION_VERIFICATION.md` for complete analysis with line numbers.
See `NvidiaKokkosPrimer.md` for detailed best practices.
