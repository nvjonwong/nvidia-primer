<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Review of NVIDIA GPU architecture

## Thread Hierarchy and Execution Model
NVIDIA GPUs execute work using a hierarchical parallelism model ([Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model)):

**Grid → Blocks → Warps → Threads**
- **Threads**: Individual execution units that run in lockstep groups
- **Warps**: Groups of 32 threads that execute in SIMT (Single Instruction, Multiple Thread) fashion
- **Thread Blocks**: Collections of threads (typically 128-1024) that can cooperate via shared memory and synchronization
- **Grid**: The complete set of thread blocks launched by a kernel

See: [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)

## Memory Hierarchy
GPUs have multiple memory spaces with different access speeds and scopes ([Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy)):

- **Registers**: Fastest, per-thread private memory
- **Shared Memory**: Fast on-chip memory shared within a thread block (~100x faster than global memory)
- **L1/L2 Cache**: Automatic caching of global memory accesses
- **Global Memory**: Large (GBs) but high-latency device memory (DRAM)
  - Accessed via: `Kokkos::View<T*, Kokkos::CudaSpace>`
- **Constant Memory**: Read-only, broadcast-optimized memory (64KB limit)
  - **Not directly exposed in Kokkos** - no dedicated constant memory space
  - **Workarounds**:
    - Small read-only data passed as kernel arguments may use constant cache automatically
    - Use `KOKKOS_INLINE_FUNCTION` with constexpr data
    - For advanced use: manually define `__constant__` variables and access via raw CUDA interop
- **Texture Memory**: Read-only, cached memory with spatial locality optimizations and hardware interpolation
  - **Not directly exposed in Kokkos** - no texture memory abstraction
  - **Workarounds**:
    - Read-only Views may benefit from L1/L2 cache with similar access patterns
    - For true texture memory: use raw CUDA texture objects within Kokkos kernels (see example below)


See: [CUDA Constant Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant-memory), [CUDA Texture Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory)

### Managed Memory and Pinned Memory

**Managed (Unified) Memory**
- Available via `cudaMallocManaged` or Kokkos' `Kokkos::CudaUVMSpace`; allocations are automatically migrated between CPU and GPU by the CUDA driver.
- Pros:
  - Simplifies code by eliminating explicit device↔host copies
  - Enables oversubscription (using more memory than fits in the GPU, at a performance cost)
- Cons:
  - Performance overhead from implicit page migration
  - Lack of control over migration can introduce latency spikes
- Usage in Kokkos:
  - Enable with `-DKokkos_ENABLE_CUDA_UVM=ON` during Kokkos configuration
  - Example:
    ```cpp
    Kokkos::View<float*, Kokkos::CudaSpace> a("a", N);               // Regular device memory
    Kokkos::View<float*, Kokkos::CudaUVMSpace> b("b", N);            // Managed (Unified) memory
    ```

**How UVM works:**
- Memory is allocated with `cudaMallocManaged()` (CUDA), or in Kokkos with `Kokkos::CudaUVMSpace`
- The programmer can access the memory from host or device code, and CUDA/Kokkos handles migration
- Hardware or the driver will automatically migrate pages between host and device depending on access
- Address space is unified: host and device share the same pointer value for a given allocation
- Explicit synchronization (e.g., `Kokkos::fence()`, `cudaDeviceSynchronize()`) is required before accessing memory from the other side

- **Kokkos Example**:
  ```cpp
  Kokkos::View<float*, Kokkos::CudaUVMSpace> x_managed("x", N);
  // Accessible from both host and device
  Kokkos::parallel_for("Init", N, KOKKOS_LAMBDA(int i) {
    x_managed(i) = i * 2.0f;
  });
  // Can use x_managed on the CPU after fence/synchronization
  ```

**When to use UVM:**
- When you have complex pointer-based data structures to share between host and device
- When you want to prototype GPU code without manually managing data copies
- In memory-constrained environments where you need to use more memory than the GPU physically has (be aware of migration overhead)
- Page migration overhead is significantly lower on Grace Hopper (see [below](#uvm-on-grace-hopper-architecture))

**Performance considerations:**
- Page migration incurs performance penalties (potentially large for random access or large transfers)
- UVM-aware page prefetching (`cudaMemPrefetchAsync`) or careful staged access can reduce overhead, but explicit memory management still provides the best performance
- For best performance and control, consider using explicit device and host memory (default in Kokkos), and use UVM only where it meaningfully simplifies the codebase

See: [CUDA Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd), [Kokkos CudaUVMSpace](https://kokkos.org/kokkos-core-wiki/API/core/memory_spaces.html#kokkos-cudauvmspace)

**Pinned (Page-Locked) Host Memory**
- Standard CUDA: Allocate via `cudaMallocHost` (or Kokkos `HostPinnedSpace`).
- Host memory pages that are page-locked (cannot be swapped); enables higher throughput for host↔device transfers.
- Pros:
  - Asynchronous data transfers between host and device (`cudaMemcpyAsync`/Kokkos `deep_copy`)
  - Higher PCIe transfer bandwidth compared to pageable memory
- Cons:
  - Excessive use reduces system memory available for paging
  - Allocation is slower and should be reused if possible
- Usage in Kokkos:
  - `Kokkos::View<double*, Kokkos::HostPinnedSpace> host_buf("buf", N);`
  - Use for staging buffers to overlap data movement and computation; combine with CUDA streams for maximal performance.

### UVM on Grace Hopper Architecture

**NVIDIA Grace Hopper** is a unique architecture that integrates the Grace CPU and Hopper GPU with **NVLink-C2C (chip-to-chip)** interconnect, providing cache-coherent unified memory with significantly different characteristics than traditional discrete GPU systems.

**Key Differences from Discrete GPUs:**

1. **Cache Coherent Unified Memory**
   - Grace CPU and Hopper GPU share a unified address space with hardware cache coherence
   - No explicit page migration required for many access patterns
   - Much higher bandwidth (900 GB/s via NVLink-C2C vs ~64 GB/s PCIe Gen5)
   - Lower latency for CPU-GPU memory access

2. **Access-Counter-Based Migration (CUDA 12.4+)**
   - Hardware access counters monitor memory page access patterns
   - Automatically migrates pages to frequently accessed memory (CPU or GPU)
   - Improves data locality dynamically
   - Can be disabled if it impacts performance: `uvm_perf_access_counter_mimc_migration_enable=0`
   
   See: [CUDA Toolkit 12.4 Grace Hopper Enhancements](https://developer.nvidia.com/blog/cuda-toolkit-12-4-enhances-support-for-nvidia-grace-hopper-and-confidential-computing/)

**Kokkos on Grace Hopper:**

Kokkos includes explicit support for unified memory architectures like Grace Hopper (4.5+, including 5.x):

```cpp
// UVM on Grace Hopper - much better performance than discrete GPUs
Kokkos::View<double*, Kokkos::CudaUVMSpace> data("data", N);

// Can access from host without explicit migration penalties (compared to discrete)
Kokkos::parallel_for("gpu_work", N, KOKKOS_LAMBDA(int i) {
    data(i) = i * 2.0;
});

Kokkos::fence();  // Still need explicit synchronization

// Host access is efficient due to cache coherence
for (int i = 0; i < 100; ++i) {
    std::cout << data(i) << " ";
}
```

**Best Practices for Grace Hopper:**

1. **NUMA Awareness**
   - Grace Hopper has multiple NUMA domains
   - Bind processes and memory correctly using `numactl`:
     ```bash
     numactl --cpunodebind=0 --membind=0 ./myapp
     ```
   - Incorrect NUMA binding can severely degrade performance

2. **Configuration**
   - Enable UVM: `-DKokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY=ON`
   - On multi-GPU without NVLink, may need: `export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`

3. **When UVM Makes Sense on Grace Hopper**
   - UVM performance penalty is much lower on Grace Hopper than discrete GPUs
   - More practical for production workloads with mixed CPU-GPU access patterns
   - Consider UVM as a reasonable default for Grace Hopper applications

4. **Profiling**
   - Use Nsight Systems to verify migration patterns
   - Check if access-counter-based migration is beneficial for your workload
   - Compare UVM vs explicit memory management - gap is smaller on Grace Hopper

**Performance Guidance:**

| System Type | UVM Performance | When to Use UVM |
|-------------|----------------|-----------------|
| Discrete GPU (PCIe) | 5-50% overhead typical | Prototyping only |
| Grace Hopper | <5-10% overhead typical | Production viable |

**Example: Disabling Access-Counter Migration**
```bash
# If access-counter migration causes issues
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm uvm_perf_access_counter_mimc_migration_enable=0
```

---

**Typical Workflow:**
1. Use page-locked host memory for high-bandwidth H↔D transfers (e.g., staging arrays).
2. Use managed memory for easier code and oversubscription, but assess performance impact (profile memory migration).
3. **On Grace Hopper**: UVM is much more performant - consider as a reasonable default for mixed CPU-GPU workloads.
4. For best performance and control on discrete GPUs, use explicit device memory (default for Kokkos on CUDA).

See also:
- [Kokkos Memory Spaces](https://kokkos.org/kokkos-core-wiki/API/core/memory_spaces.html)
- [CUDA Managed Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-unified-memory-programming-hd)
- [CUDA Pinned Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory)
- [03-data-management.md](03-data-management.md) — practical Kokkos patterns for the memory spaces described here


## Asynchronous Execution and Streams
GPUs achieve high throughput through pipelining and concurrent execution ([Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)):

**CUDA Streams** enable:
- Overlapping kernel execution with data transfers ([Overlap of Data Transfer and Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlap-of-data-transfer-and-kernel-execution))
- Concurrent kernel execution on the same GPU ([Concurrent Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-kernel-execution))
- Asynchronous host-device operations

**Multi-Process Service (MPS)**: Enables multiple CPU processes to share a single GPU with improved utilization ([Multi-Device System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-device-system))

## Performance Considerations

**Occupancy**
- Ratio of active warps to maximum possible warps on an SM
- Higher occupancy helps hide memory latency but isn't always optimal
- Balance register usage, shared memory, and thread count
- Target: 50-100% occupancy depending on workload characteristics

**Performance Bottleneck Identification**
Categorize your kernel to focus optimization efforts:

1. **Compute-Bound**: Limited by arithmetic throughput
   - Increase arithmetic intensity (compute per memory access)
   - Use specialized units (Tensor Cores, intrinsics)

2. **Memory-Bound**: Limited by memory bandwidth
   - Optimize memory access patterns (coalescing)
   - Maximize cache utilization
   - Reduce memory transactions

3. **Latency-Bound**: Limited by instruction/memory latency
   - Increase occupancy to hide latency
   - Improve instruction-level parallelism (ILP)
   - Reduce branch divergence

## Optimization Strategy Hierarchy

**1. Amdahl's Law and Bottleneck Analysis**
- Focus on the largest time consumers first
- Profile before optimizing: use Nsight Systems for system-level view, Nsight Compute for kernel analysis
- 10% improvement to 90% of runtime >> 90% improvement to 10% of runtime

**2. Maximizing GPU Utilization (Resource Efficiency)**
- Keep SMs busy: sufficient parallelism, minimal synchronization
- Maximize memory bandwidth: coalesced access, optimal transfer sizes
- Balance resource usage: registers, shared memory, thread blocks per SM

**3. Overlapping Latencies (Time Efficiency)**
- Use streams to overlap computation and communication
- Leverage asynchronous operations to keep the GPU pipeline full
- Hide memory latency with sufficient active warps
- Pipeline data transfers with kernel execution

**Key Principle**: The GPU is a throughput-oriented device. Feed it enough parallel work to keep all execution units busy while data is in flight.
