<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Introduction to Kokkos

## What is Kokkos?
Kokkos is an open-source C++ performance portability library.
It was started by Sandia National Laboratories
and is now managed by the High Performance Software Foundation,
a multi-institutional consortium under the Linux Foundation.
Kokkos provides a unified programming model that abstracts parallel execution and memory management across diverse hardware architectures, including CPUs, NVIDIA GPUs, AMD GPUs, and Intel GPUs. The core philosophy is to write code once and run it efficiently everywhere without sacrificing performance.

## Key Benefits of Using Kokkos on NVIDIA GPUs

**1. Performance Portability**
- Single codebase runs efficiently across NVIDIA GPUs and other platforms
- No need to maintain separate CUDA-specific implementations
- Full access to GPU computational power without hardware lock-in

**2. Simplified Parallel Programming**
- High-level abstractions (`parallel_for`, `parallel_reduce`, `parallel_scan`)
- Eliminates boilerplate CUDA kernel code
- Automatic mapping to efficient CUDA kernel launches

**3. Optimized Memory Management**
- `Kokkos::View` abstraction for multi-dimensional arrays
- Automatic memory layout optimization for GPU coalescing
- Support for multiple memory spaces (device, host, managed)

**4. Hierarchical Parallelism**
- Team and vector-level parallelism map naturally to CUDA thread hierarchy
- `TeamPolicy` → thread blocks, `ThreadVectorRange` → warp-level operations
- Enables fine-grained control without writing raw CUDA

**5. Scalability**
- Multi-GPU support through execution spaces
- Integration with MPI for distributed computing
- Ability to integrate with NVSHMEM or similar libraries
- Scales from laptops to leadership-class supercomputers

**6. Active Ecosystem**
- Kokkos Kernels library for optimized linear algebra operations
- Integration with profiling tools (Nsight Systems, Nsight Compute)
- Strong community and NVIDIA collaboration

## How Kokkos Interacts with CUDA

Kokkos provides a templated higher-level programming paradigm that abstracts CUDA:

**Kernel Definition**
- Kokkos kernels are C++ lambdas with `KOKKOS_LAMBDA` macro
  - Translates to: `[=] __host__ __device__`
  - Extended lambdas with some caveats (must capture by value for GPU)
- CUDA API calls hidden behind Kokkos abstraction layer
- Automatic kernel launch configuration and resource management

**Execution Model**
- `ExecutionSpace` abstracts CUDA streams and device management
- `MemorySpace` abstracts device memory, managed memory, pinned host memory
- Policies (`RangePolicy`, `TeamPolicy`, `MDRangePolicy`) control parallelism granularity

**Under the Hood**
- Kokkos compiles to native CUDA kernels using `nvcc_wrapper`
- Direct mapping to CUDA runtime API
- Minimal overhead compared to hand-written CUDA

## Profiling Kokkos Applications on NVIDIA GPUs

**Nsight Systems (nsys)**
- Kokkos operations appear with mangled C++ names
- Use `kokkos-tools` with NVTX connector for readable region names
- Example: `export KOKKOS_TOOLS_LIBS=libkp_nvtx_connector.so`

**Nsight Compute (ncu)**
- Kernel-level profiling works similar to regular CUDA
- Name mangling presents same challenges as nsys
- Use NVTX annotations for clearer profiling regions

**Best Practice**
- Enable Kokkos Tools ecosystem for enhanced profiling visibility
- Combine with NVTX for hierarchical performance analysis
- Use `kernel-logger` and `kernel-tracer` tools for detailed execution tracking

See also:
- [02-gpu-architecture.md](02-gpu-architecture.md) — GPU architecture background that complements the Kokkos abstractions described here
