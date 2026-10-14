<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KokkosPrimer Documentation Index

## Core Documentation

### [`NvidiaKokkosPrimer.md`](NvidiaKokkosPrimer.md)
Main primer covering all Kokkos synchronization issues, solutions, workarounds, and code examples. Verified for Kokkos 4.x and 5.0.1.

### [`KOKKOS_5.0.1_QUICK_REFERENCE.md`](KOKKOS_5.0.1_QUICK_REFERENCE.md)
One-page cheat sheet: pitfalls summary table, quick solutions, common workarounds.

---

## Chapters

### [`01-introduction.md`](01-introduction.md)
Kokkos overview: what it is, key benefits, CUDA interaction model, and profiling.

### [`02-gpu-architecture.md`](02-gpu-architecture.md)
GPU architecture: thread hierarchy, memory hierarchy, warp execution, and occupancy.

### [`03-data-management.md`](03-data-management.md)
Memory management: View creation, initialization strategies, UVM, host-pinned memory, and asynchronous allocation.

### [`04-execution-and-streams.md`](04-execution-and-streams.md)
Execution spaces and CUDA streams: stream management, multi-stream patterns, event synchronization, and memory prefetching.

### [`05-kernel-optimization.md`](05-kernel-optimization.md)
Kernel performance: memory access modifiers, occupancy tuning, atomics, MDRangePolicy, and instruction-level parallelism.

### [`06-synchronization-reference.md`](06-synchronization-reference.md)
Synchronization reference: when `parallel_for`, `parallel_reduce`, `parallel_scan`, and `deep_copy` synchronize, plus stream debugging and MPS/MIG.

### [`07-profiling-and-debugging.md`](07-profiling-and-debugging.md)
Profiling and debugging Kokkos on NVIDIA: KOKKOS_TOOLS, NVTX connector, Caliper, nv-combiner, debugging strategies.
Includes kernel naming and remapping, Nsight report discovery, and profiler workflow verification.

### [`08-helper-library.md`](08-helper-library.md)
Helper libraries: stream-consolidation debug utility and NVKokkos reference (CUDA events, UVM prefetching, uninitialized View creation, MDRange tiling, View trait introspection).

### [`09-repo-examples-and-tests.md`](09-repo-examples-and-tests.md)
Repository structure, `nvkokkos_helpers.hpp` header, test suite build/run instructions, verification tools (sync verification, timeline extraction), and extending the suite.

### [`10-summary.md`](10-summary.md)
Summary of core performance strategies, key workarounds for Kokkos limitations, when Kokkos matches CUDA performance, and bottom-line conclusions.

---

## Guides (in `test_suite/`)

### [`test_suite/TESTING_GUIDE.md`](test_suite/TESTING_GUIDE.md)
Complete testing and baseline management: building/running tests, all test categories, baseline creation and comparison, regression testing, troubleshooting.

### [`test_suite/PROFILING_GUIDE.md`](test_suite/PROFILING_GUIDE.md)
Profiling with Nsight Systems and Nsight Compute: timeline extraction, synchronization detection, kernel-name remapping verification, recipe discovery, Kokkos vs CUDA comparison, and performance analysis.

### [`test_suite/INFRASTRUCTURE_GUIDE.md`](test_suite/INFRASTRUCTURE_GUIDE.md)
CI/CD and infrastructure: Docker container setup, GitLab CI/CD configuration, version management, baseline storage, troubleshooting.

---

## Build Scripts (in `scripts/`)

### `scripts/build-dev.sh`
Quick build for development: builds Kokkos from source (auto-detects GPU arch) and the test suite.

### `scripts/build-and-test.sh`
Full CI-style build: fetches Kokkos, builds, builds test suite, and runs tests.

### `scripts/run-docker-interactive.sh`
Launch an interactive Docker container with GPU access.

### `scripts/test-ci-locally.sh`
Run the CI pipeline locally for testing.

---

## Structure

The repository is organized as follows:

```text
KokkosPrimer/
├── NvidiaKokkosPrimer.md                     # Main primer
├── KOKKOS_5.0.1_QUICK_REFERENCE.md           # Quick reference
├── DOCUMENTATION_INDEX.md                    # This file
├── 01-introduction.md
├── 02-gpu-architecture.md
├── 03-data-management.md
├── 04-execution-and-streams.md
├── 05-kernel-optimization.md
├── 06-synchronization-reference.md
├── 07-profiling-and-debugging.md             # Profiling & debugging
├── 08-helper-library.md
├── 09-repo-examples-and-tests.md
├── 10-summary.md
├── style-guide/
│   ├── STYLE_GUIDE.md
│   ├── CHAPTER_TEMPLATE.md
│   ├── MAINTENANCE.md                        # Index file update protocol
│   └── lint_docs.py
├── .gitlab-ci.yml                            # CI pipeline
├── scripts/
│   ├── build-dev.sh                          # Dev build script
│   ├── build-and-test.sh                     # Full build & test script
│   ├── run-docker-interactive.sh             # Docker launcher
│   └── test-ci-locally.sh                    # Local CI runner
└── test_suite/
    ├── README.md
    ├── TESTING_GUIDE.md                      # Testing & baselines guide
    ├── PROFILING_GUIDE.md                    # Profiling guide
    ├── INFRASTRUCTURE_GUIDE.md               # CI/CD & infrastructure
    ├── CMakeLists.txt                        # Build configuration
    ├── build.sh                              # Test suite build helper
    ├── profile_test.sh                       # Profiling tool
    ├── create_all_baselines.sh               # Baseline creation
    ├── compare_all_baselines.sh              # Baseline comparison
    ├── extract_timeline.py                   # Timeline extraction
    ├── extract_simplified_timeline.py        # Regression-friendly timeline
    ├── verify_sync_behavior.py               # Sync verification
    ├── test_*.cpp / test_*.cu                # Test source files (58 tests)
    └── baselines/                            # Profiling baselines
```
