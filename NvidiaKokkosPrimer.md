<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kokkos Best Practices for Nvidia GPUs

## Table of Contents

1. [Introduction to Kokkos](01-introduction.md)
   - What is Kokkos?
   - Key Benefits of Using Kokkos on NVIDIA GPUs
   - How Kokkos Interacts with CUDA
   - Profiling Kokkos Applications on NVIDIA GPUs

2. [Review of NVIDIA GPU Architecture](02-gpu-architecture.md)
   - Thread Hierarchy and Execution Model
   - Memory Hierarchy (Managed Memory, Pinned Memory, UVM, Grace Hopper)
   - Asynchronous Execution and Streams
   - Performance Considerations
   - Optimization Strategy Hierarchy

3. [Memory Management: Kokkos Views and GPU Memory Spaces](03-data-management.md)
   - Data Management with Kokkos Views (View Creation, Deallocation, cudaMallocAsync, UnManaged Views)
   - Memory Spaces and Access Patterns

4. [Execution Spaces and CUDA Streams](04-execution-and-streams.md)
   - Execution Spaces and Streams
   - Stream Management and Stream Dependencies with Events
   - Asynchronous Data Transfers
   - Avoiding Premature Synchronization in Reductions

5. [Kernel Performance Optimization](05-kernel-optimization.md)
   - Writing Efficient Kokkos Kernels (View Access Modifiers, MDRangePolicy Coalescing)
   - Instruction-Level Parallelism (ILP)
   - Memory Prefetching for UVM

6. [Synchronization Reference: When Do Kokkos Operations Synchronize?](06-synchronization-reference.md)
   - When Do Parallel Operations Synchronize?
   - Common Pitfalls Summary
   - Debugging Stream Issues
   - Alternative: MPS and MIG

7. [Profiling and Debugging Kokkos on NVIDIA](07-profiling-and-debugging.md)
   - Option 1: KOKKOS_TOOLS (NVTX connector, Nsight integration)
   - Option 2: Caliper
   - Debugging (CUDA_LAUNCH_BLOCKING, compute-sanitizer, stream consolidation)

8. [Helper Libraries: Debugging Utilities and NVKokkos Reference](08-helper-library.md)
   - Streams Helper Library (for debugging)
   - NVIDIA Kokkos Helper Library (Events, Prefetching, View Creation, MDRange Tiles, View Introspection)
   - Kokkos Tools Integration

9. [Repository, Examples, and Tests](09-repo-examples-and-tests.md)
   - Repository Structure
   - Helper Library (`nvkokkos_helpers.hpp`)
   - Test Suite (building, running, Docker)
   - Verification Tools (profiling, sync verification, timeline extraction)
   - Extending the Test Suite

10. [Summary](10-summary.md)
    - Core Performance Strategies
    - Key Workarounds for Kokkos Limitations
    - When Kokkos Matches CUDA Performance
    - Verification Tools
    - Bottom Line
