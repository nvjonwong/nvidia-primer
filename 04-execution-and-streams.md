<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Execution Spaces and CUDA Streams

This chapter covers Kokkos execution spaces, CUDA stream management, multi-stream asynchronous patterns, event synchronization, and memory prefetching for UVM.

## Execution Spaces and Streams

### Stream Management

In older versions of Kokkos (pre-4.x), launching kernels without passing an execution space instance caused implicit synchronization after each kernel. This behavior was fixed in Kokkos 4.x — kernels now queue asynchronously on the default stream regardless of whether you pass an execution space instance.

See `test_suite/test_execution_spaces.cpp` for tests that verify no implicit fences occur in either case. Profile with `nsys` to confirm (see `test_suite/PROFILING_GUIDE.md`).

**CUDA Pattern:**
```cpp
// Explicit stream control
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block, 0, stream2>>>();  // Concurrent

cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream1);
// No implicit synchronization
```

**Kokkos Current Behavior (4.x+):**
```cpp
// Both patterns now avoid implicit fences

// Pattern 1: Without explicit execution space (works fine now)
Kokkos::parallel_for("kernel1", N, KOKKOS_LAMBDA(int i) {
    // work
});  // <- NO implicit fence (fixed in Kokkos 4.x+)

Kokkos::parallel_for("kernel2", N, KOKKOS_LAMBDA(int i) {
    // work  
});  // <- NO implicit fence

// Pattern 2: With explicit execution space (also no implicit fence)
Kokkos::DefaultExecutionSpace exec_space;

Kokkos::parallel_for("kernel1",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
    KOKKOS_LAMBDA(int i) {
        // work
    });  // <- NO implicit fence

Kokkos::parallel_for("kernel2",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
    KOKKOS_LAMBDA(int i) {
        // work
    });  // <- NO implicit fence

// Explicit synchronization only when needed
exec_space.fence();  // Or Kokkos::fence() for default space
```

**When to Use Explicit Execution Space Instances:**

With the fence behavior fixed, explicit execution space instances are now primarily useful for:
- **Multiple independent streams** - use `Kokkos::Experimental::partition_space` to derive independent streams from a base execution space
- **Fine-grained control** - explicit `fence()` calls where you need them
- **Stream dependencies** - using events for cross-stream synchronization
- **Note:** Single-stream code - not necessary anymore, default behavior is good

**Kokkos with Multiple Execution Spaces:**
```cpp
// Partition one base execution space into independent streams
Kokkos::Cuda base_exec;
auto [exec_space1, exec_space2] =
    Kokkos::Experimental::partition_space(base_exec, 1, 1);

// Launch on different execution spaces - can overlap
Kokkos::parallel_for("kernel1",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space1, 0, N),
    KOKKOS_LAMBDA(int i) {
        // work on exec_space1
    });

Kokkos::parallel_for("kernel2",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space2, 0, N),
    KOKKOS_LAMBDA(int i) {
        // work on exec_space2
    });

// Explicit synchronization when needed
exec_space1.fence();
exec_space2.fence();
```

### Stream Dependencies with Events

**Problem: Kokkos Lacks Native Event Support**

CUDA events provide fine-grained stream dependencies:

```cpp
// CUDA: Event-based dependencies
cudaEvent_t event;
cudaEventCreate(&event);

kernel1<<<..., stream1>>>(...);
cudaEventRecord(event, stream1);  // Mark completion point

// stream2 waits for stream1's event
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<..., stream2>>>(...);  // Only starts after kernel1
```

**Kokkos Workaround: Custom Event Wrapper**

```cpp
// Helper for CUDA execution spaces
template<typename ExecutionSpace>
struct KokkosEvent {
    void fence() {
        exec_space.fence();
    }
    ExecutionSpace exec_space;
};

// Specialization for CUDA with actual events
template<>
struct KokkosEvent<Kokkos::Cuda> {
    cudaEvent_t event;
    
    KokkosEvent(const Kokkos::Cuda& space) {
        cudaEventCreate(&event);
        cudaEventRecord(event, space.cuda_stream());
    }
    
    ~KokkosEvent() {
        cudaEventDestroy(event);
    }
    
    void fence() {
        cudaEventSynchronize(event);
    }
    
    // Make stream wait for this event
    void make_stream_wait(const Kokkos::Cuda& space) {
        cudaStreamWaitEvent(space.cuda_stream(), event, 0);
    }
};

// Usage
Kokkos::Cuda base_exec;
auto [exec_space1, exec_space2] =
    Kokkos::Experimental::partition_space(base_exec, 1, 1);

Kokkos::parallel_for("work1",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space1, 0, N),
    KOKKOS_LAMBDA(int i) { /* work */ });

KokkosEvent<Kokkos::Cuda> event1(exec_space1);  // Capture completion

event1.make_stream_wait(exec_space2);  // exec_space2 depends on exec_space1

Kokkos::parallel_for("work2",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space2, 0, N),
    KOKKOS_LAMBDA(int i) { /* work */ });
```

## Asynchronous Data Transfers

**CUDA Pattern:**
```cpp
cudaMemcpyAsync(h_data, d_data, size, 
                cudaMemcpyDeviceToHost, stream);
// Returns immediately, transfer happens asynchronously
```

**Kokkos Pattern:**
```cpp
Kokkos::Cuda exec_space;
Kokkos::View<double*, Kokkos::CudaSpace> d_view("device", N);
Kokkos::View<double*, Kokkos::CudaHostPinnedSpace> h_view("host", N);

// Asynchronous deep copy
Kokkos::deep_copy(exec_space, h_view, d_view);

// Can launch work on same execution space - will queue after copy
Kokkos::parallel_for("work",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
    KOKKOS_LAMBDA(int i) { /* work */ });

exec_space.fence();  // Wait for both copy and work
```

## Avoiding Premature Synchronization in Reductions

**Problem: parallel_reduce to Host Scalar Forces Immediate Sync**

When you pass a host scalar as the reduction result, Kokkos must synchronize immediately to return the value:

```cpp
// BAD: Forces immediate synchronization
int sum = 0;
Kokkos::parallel_reduce("sum", N,
    KOKKOS_LAMBDA(int i, int& lsum) {
        lsum += data(i);
    }, sum);  // <- cudaDeviceSynchronize() here

// Can't queue more work without waiting for sum
```

This is problematic because:
- It forces GPU to CPU synchronization even if you don't need the result yet
- It prevents overlapping the reduction with other work
- It creates unnecessary pipeline stalls

**Solution: Reduce to Device View**

Keep the result on the GPU by using a device View. **Important**: 
1. Pre-allocate the View without initialization to avoid sync from zero-initialization
2. Use execution space instance for stream-specific synchronization

```cpp
// GOOD: Result stays on GPU, no immediate sync
// Use explicit execution space for precise stream control
Kokkos::DefaultExecutionSpace exec_space;

// Pre-allocate WITHOUT initialization (avoids cudaMemset + sync)
Kokkos::View<int> sum_device(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));

Kokkos::parallel_reduce("sum",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
    KOKKOS_LAMBDA(int i, int& lsum) {
        lsum += data(i);
    }, sum_device);  // <- NO synchronization

// Can continue queuing GPU work
Kokkos::parallel_for("more_work", N, KOKKOS_LAMBDA(int i) {
    // Use sum_device(0) in subsequent kernels if needed
    result(i) = data(i) * sum_device();  // Result still on GPU
});

// Only synchronize when you actually need the result on host
// IMPORTANT: Use execution space instance for both deep_copy and fence
auto sum_host = Kokkos::create_mirror_view(sum_device);
Kokkos::deep_copy(exec_space, sum_host, sum_device);  // Async copy on this stream
exec_space.fence();  // Stream-specific sync (efficient)

// DON'T use create_mirror_view_and_copy() - it does device-wide syncs
// DON'T use Kokkos::fence() - that syncs ALL streams on the device

std::cout << "Sum: " << sum_host() << "\n";
```

**Note on reducer initialization**
- Built-in Kokkos reducers (`Kokkos::Sum`, `Prod`, `Min`, `Max`, etc.) initialize the per-thread/league reduction value to the operation’s identity (e.g., sum → 0) via `Kokkos::reduction_identity`. The destination rank-0 `Kokkos::View` is overwritten with the final result; prior contents are ignored. You do not need to zero-initialize it, so `WithoutInitializing` is correct here.
- Pre-initialize only if you purposely accumulate across multiple reductions into the same destination or if you implement a custom reducer with a non-default identity and no `init()` implementation.

**Benefits:**
- Delays synchronization until explicitly needed
- Allows pipelining multiple reductions
- Can use reduction results in subsequent GPU kernels without sync
- Better overlap of computation and data movement

**When to Use Each:**

| Pattern | When to Use | Synchronization | Efficiency |
|---------|-------------|----------------|-----------|
| Reduce to host scalar | Immediate result needed, simple code | Immediate (forced) | **Avoid:** Device-wide sync |
| Reduce to device View + `create_mirror_view_and_copy` | Convenience | Multiple device-wide syncs | **Avoid:** Multiple syncs |
| Reduce to device View + explicit `deep_copy` | Performance-critical | Delayed (stream-specific) | **Good:** Single stream sync |
| **Reduce to CudaHostPinnedSpace View** | **Result needed on host eventually** | **NO synchronization** | **Best:** **Zero sync + host accessible** |

**Special Case: CudaHostPinnedSpace for Zero-Sync Reductions**

`CudaHostPinnedSpace` (pinned host memory) is device-accessible, meaning the GPU can write directly to it without triggering synchronization. This provides the best of both worlds:

```cpp
// BEST: Zero synchronization + host accessible result
Kokkos::View<double, Kokkos::CudaHostPinnedSpace> sum_pinned(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));

Kokkos::parallel_reduce("sum", N,
    KOKKOS_LAMBDA(int i, double& lsum) {
        lsum += data(i);
    }, sum_pinned);  // <- NO synchronization

// Can queue more GPU work immediately
Kokkos::parallel_for("more_work", N, KOKKOS_LAMBDA(int i) {
    result(i) = data(i) * 2.0;
});

// Only sync when you need to read the result on host
exec_space.fence();
std::cout << "Sum: " << sum_pinned() << "\n";  // Direct host read
```

**Alternative: Use a CUDA event instead of fencing the stream**
- Record an event on the same execution space immediately after the reduction. Keep queueing GPU work. Only synchronize the event right before you read `sum_pinned()` on the host.

```cpp
Kokkos::Cuda exec_space;
Kokkos::View<double, Kokkos::CudaHostPinnedSpace> sum_pinned(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));

Kokkos::parallel_reduce("sum", N,
    KOKKOS_LAMBDA(int i, double& lsum) { lsum += data(i); },
    sum_pinned);  // writes directly to pinned host memory, no sync

// Record event on the same stream after the reduce
NVKokkos::Event reduce_done;
reduce_done.record(exec_space);

// Queue more work here (same or other streams) without blocking...
// Kokkos::parallel_for(...);

// Only when the host needs the result, wait on the event (not the whole stream)
reduce_done.synchronize();
std::cout << "Sum: " << sum_pinned() << "\n";
```

Tip: To consume the pinned result on another stream without host sync, make that stream wait on the event:

```cpp
Kokkos::Cuda base_exec;
auto [exec_space1, exec_space2] =
    Kokkos::Experimental::partition_space(base_exec, 1, 1);
// ... launch reduce on exec_space1 into sum_pinned ...
NVKokkos::Event reduce_done;
reduce_done.record(exec_space1);

// exec_space2 depends on reduce completion
reduce_done.make_stream_wait(exec_space2);

// Now kernels on exec_space2 can safely read sum_pinned() on device
Kokkos::parallel_for("use_sum",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space2, 0, N),
    KOKKOS_LAMBDA(int i) {
        result(i) = data(i) * sum_pinned();  // device can read pinned host memory
    });
```

**Why CudaHostPinnedSpace doesn't synchronize:**

Kokkos checks if the result memory is device-accessible using `MemorySpaceAccess<CudaSpace, ResultMemorySpace>::accessible`. For `CudaHostPinnedSpace`, this returns `true`, so:
- GPU writes result directly to pinned memory (no device→host copy needed)
- No synchronization during `parallel_reduce` call
- Host can read result after explicit `fence()` (when you actually need it)
- More efficient than device View + deep_copy if you eventually need value on host

**When to use CudaHostPinnedSpace for reductions:**
- Result will eventually be needed on host
- Want to avoid explicit deep_copy
- Need to accumulate multiple results host-side
- **Note:** Pinned memory allocation is more expensive (allocate once, reuse)
- **Note:** Limited pinned memory available on system

**Example: Pipelining Multiple Reductions**

```cpp
// Pre-allocate result Views WITHOUT initialization
Kokkos::View<double> sum1(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum1"));
Kokkos::View<double> sum2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum2"));

// Both reductions queue asynchronously (no sync!)
Kokkos::parallel_reduce("sum1", N, KOKKOS_LAMBDA(int i, double& ls) {
    ls += data1(i);
}, sum1);

Kokkos::parallel_reduce("sum2", N, KOKKOS_LAMBDA(int i, double& ls) {
    ls += data2(i);  
}, sum2);

// Use results in another kernel - still no sync
Kokkos::parallel_for("combine", N, KOKKOS_LAMBDA(int i) {
    result(i) = data1(i) / sum1() + data2(i) / sum2();
});

// Only one sync at the end
Kokkos::fence();
```

See also:
- `test_suite/test_execution_spaces.cpp` — working examples of asynchronous execution and reductions
