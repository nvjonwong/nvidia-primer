<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Synchronization Reference: When Do Kokkos Operations Synchronize?

This chapter provides a comprehensive breakdown of when each Kokkos parallel operation synchronizes, based on the Kokkos 5.0.2 source code.

## When Do Parallel Operations Synchronize?

Understanding when Kokkos parallel operations (`parallel_for`, `parallel_reduce`, `parallel_scan`) synchronize is critical for performance. Here's a comprehensive breakdown based on the source code:

### parallel_for: Almost Never Synchronizes

**General Rule**: `parallel_for` launches kernels asynchronously and does not synchronize in production builds.

**Exception: Debug Mode**
```cpp
// If compiled with KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
Kokkos::parallel_for("work", N, KOKKOS_LAMBDA(int i) {
    data(i) = i;
});  // <- Implicit fence here in debug mode only
```

Source: [`core/src/Cuda/Kokkos_Cuda_KernelLaunch.hpp` L709–713](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_Cuda_KernelLaunch.hpp#L709-L713) (Kokkos 5.0.2)
- **Code**:
```cpp
#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetLastError());
  cuda_instance->fence(
      "Kokkos::Impl::launch_kernel: Debug Only Check for Execution Error");
#endif
```

**Best Practice**: Never rely on implicit synchronization. Always use explicit `fence()` when needed.

### parallel_reduce: Synchronizes When Result Goes to Host

**General Rule**: `parallel_reduce` synchronizes only when the result pointer is not device-accessible (i.e., it's a host pointer or scalar).

**Case 1: Reduce to Host Scalar (SYNCHRONIZES)**
```cpp
int sum = 0;
Kokkos::parallel_reduce("sum", N,
    KOKKOS_LAMBDA(int i, int& lsum) {
        lsum += data(i);
    }, sum);  // <- Host scalar: IMMEDIATE SYNCHRONIZATION
// GPU work completes here, sum is available immediately
```

**Why it synchronizes**:
- Result must be copied from GPU scratch space to host memory
- `DeepCopy<HostSpace, CudaSpace>` is called, which synchronizes

Source: [`core/src/Cuda/Kokkos_Cuda_Parallel_Range.hpp` L346–363](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_Cuda_Parallel_Range.hpp#L346-L363) (Kokkos 5.0.2)
```cpp
if (!m_result_ptr_device_accessible) {
  if (m_result_ptr) {
    if (m_unified_space) {
      m_policy.space().fence(
          "Kokkos::Impl::ParallelReduce<Cuda, RangePolicy>::execute: "
          "Result Not Device Accessible");
      // ... copy from unified memory ...
    } else {
      const int size = m_functor_reducer.get_reducer().value_size();
      DeepCopy<HostSpace, CudaSpace, Cuda>(m_policy.space(), m_result_ptr,
                                           m_scratch_space, size);
      // ^ This synchronizes
    }
  }
}
```

**Case 2: Reduce to Device View (NO SYNCHRONIZATION)**
```cpp
Kokkos::View<int> sum_device(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));

Kokkos::parallel_reduce("sum", N,
    KOKKOS_LAMBDA(int i, int& lsum) {
        lsum += data(i);
    }, sum_device);  // <- Device view: NO SYNCHRONIZATION

// Can queue more GPU work immediately
Kokkos::parallel_for("use_sum", N, KOKKOS_LAMBDA(int i) {
    result(i) = data(i) / sum_device();  // Result still on GPU
});

// Only synchronize when you need the value on host
auto sum_host = Kokkos::create_mirror_view(sum_device);
Kokkos::deep_copy(sum_host, sum_device);  // This synchronizes
```

**Key Insight**: Reducing to a device View keeps everything asynchronous until you explicitly copy to host.

### parallel_scan: Synchronizes When Result is a Value (Not a View)

**General Rule**: `parallel_scan` synchronizes when the return type is a scalar value, not a View.

**Case 1: Scan with Scalar Return (SYNCHRONIZES)**
```cpp
int total = 0;
Kokkos::parallel_scan("scan", N,
    KOKKOS_LAMBDA(int i, int& update, bool final) {
        if (final) data(i) = update;
        update += i;
    }, total);  // <- Scalar return: SYNCHRONIZATION

// total is available immediately on host
std::cout << "Total: " << total << "\n";
```

Source: [`core/src/Kokkos_Parallel.hpp` L445–447](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Kokkos_Parallel.hpp#L445-L447) (Kokkos 5.0.2)
```cpp
if (!Kokkos::is_view<ReturnType>::value)
  policy.space().fence(
      "Kokkos::parallel_scan: fence due to result being a value, not a view");
```

**Case 2: Scan with View Return (NO IMMEDIATE SYNCHRONIZATION)**
```cpp
Kokkos::View<int> total_device("total");

Kokkos::parallel_scan("scan", N,
    KOKKOS_LAMBDA(int i, int& update, bool final) {
        if (final) data(i) = update;
        update += i;
    }, total_device);  // <- View: NO IMMEDIATE SYNCHRONIZATION

// Can queue more work
// Synchronize only when needed
```

**Implementation Detail**: `parallel_scan` launches two kernels sequentially on the same stream:
1. First pass: Initial scan to compute block totals
2. Second pass: Final scan using block totals

Source: [`core/src/Cuda/Kokkos_Cuda_Parallel_Range.hpp` L675–688](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_Cuda_Parallel_Range.hpp#L675-L688) (Kokkos 5.0.2)
```cpp
m_final = false;
CudaParallelLaunch<ParallelScan, LaunchBounds>(
    *this, grid, block, shmem,
    m_policy.space().impl_internal_space_instance());  // First pass

m_final = true;
CudaParallelLaunch<ParallelScan, LaunchBounds>(
    *this, grid, block, shmem,
    m_policy.space().impl_internal_space_instance());  // Second pass
```

The two kernels execute sequentially on the same stream, so they're ordered correctly without explicit synchronization between them.

### When DeepCopy Synchronizes

`DeepCopy` behavior depends on whether you pass an execution space instance:

**Case 1: DeepCopy WITHOUT Execution Space (ALWAYS SYNCHRONIZES)**
```cpp
Kokkos::View<double*, Kokkos::CudaSpace> d_data("device", N);
Kokkos::View<double*, Kokkos::HostSpace> h_data("host", N);

// This synchronizes
Kokkos::deep_copy(h_data, d_data);
```

Source: [`core/src/Cuda/Kokkos_CudaSpace.cpp` L65–73](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_CudaSpace.cpp#L65-L73) (Kokkos 5.0.2)
```cpp
void DeepCopyAsyncCuda(void *dst, const void *src, size_t n) {
  cudaStream_t s = cuda_get_deep_copy_stream();
  KOKKOS_IMPL_CUDA_SAFE_CALL((CudaInternal::singleton()
      .cuda_memcpy_async_wrapper(dst, src, n, cudaMemcpyDefault, s)));
  // Immediately synchronizes the stream
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(s));
}
```

**Case 2: DeepCopy WITH Execution Space (ASYNCHRONOUS)**
```cpp
Kokkos::Cuda exec_space;

// Asynchronous copy on exec_space's stream
Kokkos::deep_copy(exec_space, h_data, d_data);

// Can queue more work on same execution space
Kokkos::parallel_for("work",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
    KOKKOS_LAMBDA(int i) { /* work */ });

// Synchronize only when needed
exec_space.fence();
```

**However**: If the execution space is not `Kokkos::Cuda`, it fences before the copy:

Source: [`core/src/Cuda/Kokkos_CudaSpace.hpp` L521–523](https://github.com/kokkos/kokkos/blob/dfceb77277e23361cccfbcf72c0468fd8475153e/core/src/Cuda/Kokkos_CudaSpace.hpp#L521-L523) (Kokkos 5.0.2)
```cpp
// For non-Cuda execution spaces
inline DeepCopy(const ExecutionSpace& exec, void* dst, const void* src,
                size_t n) {
  exec.fence(fence_string());  // Fence the execution space first
  DeepCopyAsyncCuda(dst, src, n);
}
```

### Summary Table

| Operation | Synchronization Condition | When It Happens |
|-----------|---------------------------|-----------------|
| **parallel_for** | Debug mode only | `KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK` defined |
| **parallel_reduce** | Result → host scalar or HostSpace pointer | After kernel, before returning |
| **parallel_reduce** | Result → device View (CudaSpace/UVM) | **NO synchronization** |
| **parallel_reduce** | Result → CudaHostPinnedSpace View | **NO synchronization** (device-accessible) |
| **parallel_scan** | Result is scalar value | After second kernel pass |
| **parallel_scan** | Result is View | **NO synchronization** |
| **deep_copy** | No execution space passed | Always synchronizes |
| **deep_copy** | Cuda exec space passed | Asynchronous on stream |
| **deep_copy** | Non-Cuda exec space passed | Fences exec space first |

### Best Practices for Avoiding Unwanted Synchronization

1. **Use Device Views for Reductions**
```cpp
// BAD: Immediate sync
int sum;
Kokkos::parallel_reduce("sum", N, lambda, sum);

// GOOD: Stays on GPU
Kokkos::View<int> sum_device(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "sum"));
Kokkos::parallel_reduce("sum", N, lambda, sum_device);
```

2. **Pass Execution Spaces to deep_copy**
```cpp
// BAD: Synchronizes immediately
Kokkos::deep_copy(host_data, device_data);

// GOOD: Asynchronous on stream
Kokkos::Cuda exec_space;
Kokkos::deep_copy(exec_space, host_data, device_data);
```

3. **Disable Debug Mode in Production**
```bash
# Don't use this in production builds:
-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON
```

4. **Profile to Verify**
```bash
# Use nsys to see actual synchronization points
nsys profile --stats=true ./my_app
# Look for cudaDeviceSynchronize and cudaStreamSynchronize calls
```

See: `test_suite/test_execution_spaces.cpp` for working examples; `test_suite/PROFILING_GUIDE.md` for profiling synchronization points.

## Common Pitfalls Summary

| Issue | CUDA Default | Kokkos 4.x+ Default | Kokkos Optimized |
|-------|-------------|---------------------|------------------|
| View initialization | No init (fast) | Zero init + sync (slow) | `WithoutInitializing` |
| View allocation | `cudaMallocAsync` (no sync) | `cudaMalloc` or `cudaMallocAsync` + sync | `StreamOrderedView` (zero sync) |
| View deallocation | `cudaFreeAsync` (no sync) | Sync + free + sync (slow) | `StreamOrderedView` (zero sync) |
| Kernel launch | Async on stream | **Async (fixed!)** | Same behavior |
| Single stream control | Manual `cudaStreamSynchronize` | **No auto-sync (fixed!)** | Explicit `.fence()` when needed |
| Multiple streams | Multiple `cudaStream_t` | Single stream default | Multiple execution space instances |
| Data transfers | Async with stream | Sync without space | Pass execution space |
| Stream dependencies | `cudaStreamWaitEvent` | Only `.fence()` | Custom event wrapper (`NVKokkos::Event`) |
| Memory prefetch | `cudaMemPrefetchAsync` | Not available | Manual CUDA call (`NVKokkos::prefetch_*`) |
| Atomic operations | Per-operation (`atomicAdd`) | `Atomic` trait = all accesses | Explicit `atomic_add()` on regular View |

**Note**: Implicit fence issues were fixed in Kokkos 4.x+. See `test_suite/test_execution_spaces.cpp` and `test_suite/PROFILE_FENCE_BEHAVIOR.md` for verification.

**Key Optimization**: `NVKokkos::StreamOrderedView` provides RAII-managed, zero-synchronization allocation/deallocation by using `cudaMallocAsync`/`cudaFreeAsync` with UnManaged Views. This is the recommended approach for performance-critical code with temporary allocations. See section [Advanced: Truly Asynchronous Allocation with UnManaged Views](03-data-management.md#advanced-truly-asynchronous-allocation-with-unmanaged-views) and `nvkokkos_helpers.hpp`.

## Debugging Stream Issues

**Enable Blocking Mode:**
```bash
# Forces all kernel launches to synchronize
export CUDA_LAUNCH_BLOCKING=1

# Run your app - easier to find where NaNs/errors occur
./my_kokkos_app
```

**Force Single Stream for Testing:**
```cpp
// Use default execution space (single stream) everywhere
Kokkos::parallel_for("work", N, KOKKOS_LAMBDA(int i) {
    // All kernels serialize
});
```

**Force All Execution Spaces to Single Stream:**

If your code already uses multiple execution spaces but you suspect a stream synchronization issue, you can force everything onto one stream for debugging:

```cpp
// Original code with multiple streams
std::vector<Kokkos::Cuda> exec_spaces;
for(int i = 0; i < num_streams; i++) {
    exec_spaces.push_back(Kokkos::Cuda());  // Each has own stream
}

// Use different streams in kernels
Kokkos::parallel_for("work1",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_spaces[0], 0, N),
    KOKKOS_LAMBDA(int i) { /* work */ });

Kokkos::parallel_for("work2",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_spaces[1], 0, N),
    KOKKOS_LAMBDA(int i) { /* work */ });
```

**Debug: Force to Single Stream**

```cpp
// Create single execution space
Kokkos::Cuda single_stream;

// Make all array elements point to same stream
std::vector<Kokkos::Cuda> exec_spaces(num_streams, single_stream);

// Now all kernels serialize on same stream
Kokkos::parallel_for("work1",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_spaces[0], 0, N),
    KOKKOS_LAMBDA(int i) { /* work */ });  // Uses single_stream

Kokkos::parallel_for("work2",
    Kokkos::RangePolicy<Kokkos::Cuda>(exec_spaces[1], 0, N),
    KOKKOS_LAMBDA(int i) { /* work */ });  // Uses single_stream

// If problem disappears, you have a stream dependency issue
```

**Alternative: Replace Vector Initialization**

```cpp
// Before (multiple streams):
std::vector<Kokkos::Cuda> exec_spaces;
for(int i = 0; i < num_streams; i++) {
    exec_spaces.push_back(Kokkos::Cuda());
}

// Debug version (single stream):
Kokkos::Cuda shared_stream;
std::vector<Kokkos::Cuda> exec_spaces(num_streams, shared_stream);

// Or even simpler - just reuse first stream:
std::vector<Kokkos::Cuda> exec_spaces;
exec_spaces.push_back(Kokkos::Cuda());  // Create first
for(int i = 1; i < num_streams; i++) {
    exec_spaces.push_back(exec_spaces[0]);  // Reuse first stream
}
```

**What This Tells You:**

If the bug disappears, the problem is a stream synchronization issue:
- Missing fences between dependent operations
- Race condition between kernels on different streams
- Need to add event dependencies

If the bug persists, the problem is not stream-related:
- Logic error in kernel code
- Memory corruption unrelated to streams
- Data race within single kernel

**Better Stream Strategy: Alternate Between Two Streams**

**Common Anti-Pattern: Too Many Streams**

Because of `Kokkos::View`'s implicit synchronization, developers often create many streams to avoid fences:

```cpp
// Anti-pattern: Create stream for every operation
std::vector<Kokkos::Cuda> exec_spaces;
for(int i = 0; i < 100; i++) {
    exec_spaces.push_back(Kokkos::Cuda());
}

// Defeats stream-ordered execution benefits
for(int i = 0; i < 100; i++) {
    Kokkos::parallel_for("work",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_spaces[i], 0, N),
        KOKKOS_LAMBDA(int j) { /* work */ });
}

// Lost benefits:
// - No automatic ordering within stream
// - Explicit synchronization needed everywhere
// - Hard to manage dependencies
```

**Recommended: Two-Stream Alternating Pattern**

Use just two execution spaces and alternate between them:

```cpp
// Create array of two execution spaces
std::array<Kokkos::Cuda, 2> exec_spaces;

for(int iteration = 0; iteration < num_iterations; iteration++) {
    // Alternate between two streams: 0, 1, 0, 1, ...
    int current_idx = iteration % 2;
    
    // Launch work on current stream
    Kokkos::parallel_for("compute",
        Kokkos::RangePolicy<Kokkos::Cuda>(exec_spaces[current_idx], 0, N),
        KOKKOS_LAMBDA(int i) { 
            data(i) = compute_expensive(data(i)); 
        });
    
    Kokkos::deep_copy(exec_spaces[current_idx], host_output, device_data);
    
    // Only fence when dependencies cross streams
    if(needs_synchronization(iteration)) {
        exec_spaces[0].fence();
        exec_spaces[1].fence();
    }
}
```

**Benefits of Two-Stream Pattern:**

**Stream-ordered execution**: Operations on same stream execute in order
- No fence needed between operations on same stream
- CUDA automatically handles dependencies

**Overlap opportunities**: Two streams can run concurrently
- Stream A computing while Stream B transfers
- Good balance of concurrency and simplicity

**Minimal synchronization**: Only fence when truly needed
- Between operations with cross-stream dependencies
- At end of major computation phases

**Easy to understand**: Simple alternating pattern
- Clear which stream is active
- Easy to add dependencies when needed

**Example: Computation Pipeline**

```cpp
Kokkos::Cuda stream_compute;
Kokkos::Cuda stream_transfer;

for(int step = 0; step < num_steps; step++) {
    // Transfer input on transfer stream
    Kokkos::deep_copy(stream_transfer, d_input, h_input[step]);
    
    // Compute on compute stream (can overlap with transfer)
    Kokkos::parallel_for("compute",
        Kokkos::RangePolicy<Kokkos::Cuda>(stream_compute, 0, N),
        KOKKOS_LAMBDA(int i) { d_output(i) = process(d_input(i)); });
    
    // Only fence when compute needs transferred data
    stream_transfer.fence();  // Ensure transfer complete before compute uses it
}
```

**Key Principle**: Start with 1-2 streams and leverage stream-ordered execution. Only add more streams if profiling shows opportunity for additional concurrency.

## Alternative: Use MPS or MIG to Overcome Synchronization Without Code Changes

**The Challenge**: Kokkos's implicit synchronization can limit concurrency, and refactoring to use explicit execution spaces requires code changes throughout your application.

**Solution**: Use **NVIDIA MPS (Multi-Process Service)** or **MIG (Multi-Instance GPU)** to enable concurrency at the system level without modifying your Kokkos code.

### Multi-Process Service (MPS)

**What MPS Does:**
- Allows multiple CPU processes to share a single GPU
- Each process can have its own CUDA context and streams
- GPU work from different processes can execute concurrently
- Particularly effective for small kernels that don't saturate the GPU

**Using MPS with Kokkos:**

```bash
# Start MPS daemon
nvidia-cuda-mps-control -d

# Run multiple Kokkos processes that will share GPU
mpirun -n 4 ./my_kokkos_app &
mpirun -n 4 ./another_kokkos_app &

# Stop MPS daemon when done
echo quit | nvidia-cuda-mps-control
```

**Benefits for Kokkos Applications:**
- **No code changes**: Works with existing Kokkos code using default execution space
- **Process-level concurrency**: Multiple MPI ranks can overlap GPU work
- **Overcomes implicit fences**: Each process has independent context
- **Good for small kernels**: Helps utilize GPU when individual kernels are small

**When MPS Helps:**
```cpp
// Each MPI rank runs Kokkos code with implicit synchronization
Kokkos::parallel_for("work", N, KOKKOS_LAMBDA(int i) {
    // Small kernel
});  // Implicit fence in each process

// With MPS: Different ranks can run concurrently on GPU
// Without MPS: Ranks serialize GPU access
```

**MPS Limitations:**
- Each process has independent memory - no direct data sharing
- Best for compute-bound workloads, not memory-bound
- Total memory usage is sum of all processes
- Requires compatible GPU (Volta+) for best performance

### Multi-Instance GPU (MIG)

**What MIG Does** (A100, H100, H200):
- Partitions single GPU into multiple independent GPU instances
- Each instance has dedicated SM, memory, and cache resources
- True hardware isolation between instances
- Guaranteed QoS and fault isolation

**Using MIG with Kokkos:**

```bash
# Enable MIG mode (requires root, system reboot)
sudo nvidia-smi -mig 1

# Create MIG instances (example: 3 equal partitions on A100)
sudo nvidia-smi mig -cgi 19,19,19 -C

# List MIG instances
nvidia-smi -L

# Run Kokkos application on specific MIG instance
CUDA_VISIBLE_DEVICES=0 ./kokkos_app1 &
CUDA_VISIBLE_DEVICES=1 ./kokkos_app2 &
CUDA_VISIBLE_DEVICES=2 ./kokkos_app3 &
```

**Benefits for Kokkos Applications:**
- **True hardware isolation**: Each instance is independent GPU
- **Guaranteed resources**: No competition between instances
- **QoS**: Predictable performance per instance
- **No code changes**: Each instance appears as separate GPU

**When MIG Helps:**
- Multiple independent Kokkos applications sharing same physical GPU
- Need guaranteed performance isolation
- Testing/development: One instance per developer
- Mixed workloads: Production + monitoring on same GPU

**MIG Configuration Examples (A100 80GB):**

| Profile | SMs | Memory | Use Case |
|---------|-----|--------|----------|
| 1g.10gb | 14 | 10GB | 7 instances, small jobs |
| 2g.20gb | 28 | 20GB | 3 instances, medium jobs |
| 3g.40gb | 42 | 40GB | 2 instances, large jobs |
| 7g.80gb | 98 | 80GB | 1 instance, full GPU |

### MPS vs MIG Comparison

| Feature | MPS | MIG |
|---------|-----|-----|
| **Code changes** | None | None |
| **Hardware requirement** | Volta+ | A100/H100/H200 |
| **Isolation** | Software (shared resources) | Hardware (dedicated resources) |
| **Overhead** | Low | None (hardware) |
| **Memory sharing** | No (per-process) | No (per-instance) |
| **QoS guarantee** | No | Yes |
| **Setup complexity** | Easy (runtime) | Moderate (requires reboot) |
| **Best for** | Small kernels, MPI apps | Multiple independent apps |

### Practical Guidance

**Use MPS when:**
- Running MPI-parallel Kokkos application with many ranks
- Individual kernels are small and don't saturate GPU
- Want quick improvement without code refactoring
- Can tolerate lack of QoS guarantees

**Use MIG when:**
- Multiple independent Kokkos applications
- Need guaranteed resources and isolation
- Have A100/H100/H200 GPUs
- Development/testing environments

**Use explicit streams when:**
- Single application needs fine-grained concurrency control
- Can modify code to use execution spaces
- Need precise dependency management
- Want maximum control over execution

**Example: MPI Application with MPS**

```bash
#!/bin/bash
# Launch script for MPI Kokkos app with MPS

# Start MPS
nvidia-cuda-mps-control -d

# Run MPI application - each rank gets independent CUDA context
mpirun -n 8 --map-by ppr:8:socket \
    --bind-to core \
    ./my_kokkos_mpi_app input.dat

# MPS enables concurrent execution across ranks
# Without MPS: ranks would serialize GPU access

# Cleanup
echo quit | nvidia-cuda-mps-control
```

**Monitoring MPS:**
```bash
# Check MPS status
nvidia-cuda-mps-control -d

# View MPS server logs
cat /tmp/nvidia-mps/control.log

# Monitor GPU utilization with multiple processes
nvidia-smi dmon
```

See also:
- [CUDA MPS Documentation](https://docs.nvidia.com/deploy/mps/index.html)
- [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html)
- [Kokkos View Documentation](https://kokkos.github.io/kokkos-core-wiki/API/core/view/view.html)
- [Kokkos Execution Spaces](https://kokkos.github.io/kokkos-core-wiki/API/core/execution_spaces.html)

