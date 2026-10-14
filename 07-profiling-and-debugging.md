<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling and Debugging Kokkos on NVIDIA

Profiling Kokkos applications on NVIDIA GPUs is essential for understanding where time is spent, how kernels and memory operations are ordered, and how to optimize. This chapter focuses on two options that work well with Kokkos: **KOKKOS_TOOLS** (the Kokkos profiling interface and its plugins) and **Caliper**. Both can be used without recompiling your application and give you readable region names and structured performance data.

## Chapter overview

- [Two profiling options](#two-profiling-options)
  - [Option 1: Kokkos Tools NVTX connector](#option-1-kokkos-tools-nvtx-connector)
  - [Option 2: Caliper](#option-2-caliper)
- [Capture and focus an Nsight Systems profile](#capture-and-focus-the-profile)
- [Map Kokkos labels to Nsight Compute kernel names](#kernel-naming-and-renaming)
- [Validate the workflow with the included demo](#validate-the-scripts-with-the-included-demo)
- [Debug Kokkos applications](#debugging)

## Two profiling options

- **KOKKOS_TOOLS** is Kokkos’s native profiling hook system. Plugins (e.g., NVTX connector, kernel logger) attach to the Kokkos runtime and annotate kernels, regions, and memory operations. This is the standard way to get Kokkos-aware labels into Nsight Systems and other NVIDIA tools.
- **Caliper** is a general-purpose performance measurement and analysis toolkit that supports Kokkos via the same KOKKOS_TOOLS interface. It gives you flexible, hierarchical profiling (including GPU time and memory copies).

## Capture and focus the profile

Nsight Systems can capture Kokkos dispatch labels when a Kokkos-aware tool creates enclosing NVTX ranges around kernels. Use either the Kokkos Tools NVTX connector or Caliper's NVTX service to create NVTX regions from Kokkos kernels and regions at run time. The option-specific sections later in this chapter explain how to configure each integration.

After configuring and installing one of the two options, capture CUDA activity and NVTX ranges with the following command pattern:

```bash
nsys profile --trace=cuda,nvtx --output=my_profile ./my_kokkos_app
```

This command creates `my_profile.nsys-rep`. Open the report in the Nsight Systems GUI to inspect kernel names, region hierarchy, and synchronization calls such as `cudaStreamSynchronize` and `cudaDeviceSynchronize`. For GPU memory usage, add `--cuda-memory-usage=true` to the collection command.

Nsight Systems collection generally has low overhead, but the generated reports can be large. Processing and viewing reports over 4 GB in the GUI can be resource-intensive. With the default collection settings, try to limit application runs to less than five minutes.

Nsight Compute overhead depends on the selected metrics and replay passes. Use Nsight Systems to identify important NVTX ranges, then restrict Nsight Compute to a few kernels and launches.

Kokkos kernels normally appear in Nsight Compute with long demangled C++ names containing template arguments, lambda captures, or functor types. Selecting them by an enclosing NVTX label is usually easier. The [Kernel naming and renaming](#kernel-naming-and-renaming) section describes how to verify the association, generate a rename map, and use the renamed label for an Nsight Compute collection.

## Option 1: Kokkos Tools NVTX connector

Kokkos provides a plugin-based profiling interface, often referred to as Kokkos Tools or KokkosP. Instrumentation hooks are built into the Kokkos runtime, and tools are implemented as dynamic libraries that are loaded at runtime. Your application does not need to be recompiled when you switch or add tools.

Kokkos is typically built with `Kokkos_ENABLE_LIBDL=ON`, which is the default, so it can load shared libraries at startup. Set `KOKKOS_TOOLS_LIBS`, or `KOKKOS_PROFILE_LIBRARY` in some setups, to the path of the plugin's shared library. The plugin receives callbacks for kernel launches, region push and pop operations, memory allocations, and other events.

### Capture with the NVTX connector

The [Kokkos NVTX connector](https://github.com/kokkos/kokkos-tools/wiki/NVTXConnector) creates NVTX ranges from Kokkos regions and encloses named kernels in NVTX ranges. Configure the connector, then run the shared Nsight Systems capture command:

```bash
export INSTALL_PATH_TO_KTOOLS=/path/to/kokkos-tools/build/install/lib
export KOKKOS_TOOLS_LIBS=${INSTALL_PATH_TO_KTOOLS}/libkp_nvtx_connector.so

nsys profile --trace=cuda,nvtx --output=my_profile ./my_kokkos_app
```

The NVTX connector does not implement memory allocation hooks. Use Nsight Systems' `--cuda-memory-usage=true` option when you need allocation visibility.

The connector can insert global fences so that its ranges accurately delimit asynchronous GPU work. These fences make timing easier to interpret but can hide application concurrency. Disable the extra fences before profiling performance when the connector supports this setting:

```bash
export KOKKOS_TOOLS_GLOBALFENCES=0
```

This setting applies to both the focused and regular connectors. The focused connector records more detail. Verify the resulting timeline because an NVTX host range can end before its asynchronous GPU work completes. The global fences are typically unnecessary with Nsight Systems because it distinguishes the CPU and GPU timelines. The fences are irrelevant to Nsight Compute.

### Other Kokkos Tools

The [kokkos-tools](https://github.com/kokkos/kokkos-tools) repository provides additional plugins:

- **KernelLogger** and **SimpleKernelTimer** – lightweight kernel inspection and timing.
- **KernelFilter** and **KernelSampler** – utilities that limit which kernels other tools analyze.
- **MemoryHighWater**, **MemoryUsage**, **MemoryEvents** – memory analysis.
- **SpaceTimeStack** - Reports timing and memory usage for nested profiling regions.

Build the tools you need and set `KOKKOS_TOOLS_LIBS` to the appropriate `.so`. No change to your application code is required.

### Combining multiple tools

Some tools can operate simultaneously. See the [Kokkos Tools documentation](https://github.com/kokkos/kokkos-tools) for current compatibility information.

1. **Semicolon-separated list** – Set `KOKKOS_TOOLS_LIBS` to multiple shared-library paths separated by semicolons, such as `lib1.so;lib2.so`. Kokkos loads each library and invokes its callbacks. Confirm that your Kokkos Tools version supports the selected combination.
2. **Monolithic build (`KokkosTools_ENABLE_SINGLE`)** – A single Kokkos Tools library can link multiple profilers with `SINGLELIB_PROFILERS`. Multi-profiler callback dispatch has not worked consistently across versions, so verify that every selected profiler receives events. Prefer the semicolon-separated list for supported combinations.

### Optional: explicit regions in your code

You can add custom regions so that both KOKKOS_TOOLS plugins and Caliper see them:

```cpp
#include <Kokkos_Core.hpp>

int main() {
  Kokkos::initialize();
  {
    Kokkos::Profiling::pushRegion("my_algorithm");
    // ... your code ...
    Kokkos::Profiling::popRegion();
  }
  Kokkos::finalize();
}
```

Or use the RAII helper:

```cpp
{
  Kokkos::Profiling::ProfilingSection section("my_region");
  // ... code to profile ...
}
```

These regions will appear in NVTX timelines and in Caliper if you use the Kokkos Tools connector.

## Option 2: Caliper

[Caliper](https://software.llnl.gov/Caliper/) is a performance instrumentation and measurement library from LLNL. It can act as a **Kokkos Tools plugin**, so it automatically gets Kokkos kernel and region events. In addition, Caliper can profile **CUDA API calls** and **GPU activity** (kernel and memory copy time) when built with CUPTI support, which is very useful on NVIDIA GPUs.

### How to use Caliper with Kokkos

1. **Build Caliper** with Kokkos Tools and CUDA support, for example:
   - Enable the Kokkos Tools connector (Caliper’s Kokkos integration).
   - For NVIDIA GPUs: `-DWITH_CUPTI=On` so you can get GPU time and kernel-level data.

2. **Run your Kokkos application** under Nsight Systems with Caliper loaded as the Kokkos Tools library. Use `profile.kokkos` in `CALI_CONFIG` so Caliper records Kokkos regions and kernels, and enable the NVTX service so Nsight Systems sees those regions:

   ```bash
   export KOKKOS_TOOLS_LIBS=/path/to/lib/libcaliper.so

   export CALI_SERVICES_ENABLE=nvtx
   export CALI_CONFIG="runtime-report(output=stdout,profile.kokkos)"
   nsys profile --trace=cuda,nvtx --output=my_profile ./my_kokkos_app
   ```

   This command creates a Caliper text report and an Nsight Systems report containing the Kokkos-derived NVTX ranges. Do not enable `profile.cuda` in this run. Caliper and Nsight Systems would both attempt to subscribe to CUPTI and could produce `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`. Collect Caliper CUDA metrics in a separate run.

### What you get with Caliper on NVIDIA

- **Host-side CUDA API** – Caliper's [`profile.cuda`](https://software.llnl.gov/Caliper/GPUProfiling.html#profiling-host-side-api-functions) option reports time in CUDA API calls such as `cudaMemcpy`, `cudaLaunchKernel`, and `cudaDeviceSynchronize`.
- **Device-side GPU time** – Caliper's [`cuda.gputime`](https://software.llnl.gov/Caliper/GPUProfiling.html#profiling-device-side-activities) option reports GPU activity time from kernels and memory copies.
- **Structured reports** – [`runtime-report`](https://software.llnl.gov/Caliper/BuiltinConfigurations.html#built-in-configs) prints region timing, while [`cuda-activity-report` and `cuda-activity-profile`](https://software.llnl.gov/Caliper/GPUProfiling.html#profiling-device-side-activities) provide GPU activity tables or machine-readable files for tools such as Hatchet or `cali-query`.
- **Memory copy volume** – Caliper's [`cuda.memcpy`](https://software.llnl.gov/Caliper/GPUProfiling.html#profiling-memory-copies) option reports CPU-to-GPU and GPU-to-CPU copy volume.

### When to use which

- Use **KOKKOS_TOOLS + NVTX** when your workflow is centered on **Nsight Systems** (timelines, traces, synchronization inspection). See also the [Profiling Guide](test_suite/PROFILING_GUIDE.md) in the test suite for Nsight and timeline extraction.
- Use **Caliper** when integrating with other Caliper-based workflows (e.g., MPI + Kokkos, or mixed Fortran + Kokkos when Caliper is built with `WITH_KOKKOS=ON` and Fortran support).

## Kernel naming and renaming

Use Kokkos labels to give GPU kernels readable names in Nsight Systems and Nsight Compute:

1. **Label each Kokkos dispatch.** Without a label, profiling tools fall back to the C++ functor or lambda type.

   ```cpp
   Kokkos::parallel_for("my_func", N, KOKKOS_LAMBDA(int i) {
     // ...
   });
   ```

2. **Export the labels as NVTX ranges.** Use the Kokkos Tools NVTX connector or Caliper, and capture both CUDA and NVTX activity. The option-specific capture commands above provide the required configuration.

3. **Verify that Nsight Systems associates the kernels with their labels.**

   ```bash
   nsys stats --report cuda_gpu_kern_sum:nvtx-name my_profile.nsys-rep
   nsys stats --report nvtx_kern_sum my_profile.nsys-rep
   ```

The [`cuda_gpu_kern_sum:nvtx-name` report](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#cuda-gpu-kern-sum-nvtx-name-base-mangled-cuda-gpu-kernel-summary) uses the innermost enclosing NVTX range in each kernel name. The [`nvtx_kern_sum` report](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-kern-sum-base-mangled-nvtx-range-kernel-summary) groups kernels by NVTX range. These reports change the displayed output; they do not rename kernels in raw SQLite or Parquet exports.

The next section generates an Nsight Compute rename map from the supported Nsight Systems report output.

### Generate an NCU map from an Nsight Systems report

After collecting `my_profile.nsys-rep`, export the report to a SQLite database and invoke the python mapper `test_suite/profiling_name_tools.py` provided in this repository:

```bash
nsys export -t sqlite my_profile.nsys-rep -o my_profile.sqlite
python3 test_suite/profiling_name_tools.py \
    my_profile.sqlite --output ncu-kernel-renames.yaml
```

The mapper reads the SQLite tables with the standard-library `sqlite3` module, streams NVTX ranges per host thread, and selects the innermost enclosing range. On first use it adds a reusable `(globalTid, start, end)` index to the generated SQLite database.

The YAML file now contains the mappings NCU needs to target kernels by their enclosing Kokkos labels. To include only labels beginning with a particular prefix, add `--label-prefix MyKokkosRegion`. The prefix match is literal; a trailing period has no special meaning.

When multiple NVTX range names enclose the same symbol, we key the overloaded symbol to the intersection between NVTX range names with wildcards for differing characters. For example, default initialization of multiple named Views of the same non-trival types will commonly cause such collisions. These collisions are saved to `ncu-kernel-renames.collisions.yaml` (or `--collisions-output PATH`) for review.

Profiler report formats and command-line options can change. Use the [validation harness](test_suite/PROFILING_GUIDE.md) described in [Validate the scripts with the included demo](#validate-the-scripts-with-the-included-demo) to check the installed versions.

**Apply the Nsight Compute rename map.** Current [Nsight Compute releases](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#kernel-renaming) accept a YAML mapping from an original demangled name to a display name:

```yaml
-
  - Original: "void VeryLongKokkosKernelType(int, unsigned int *, ...)"
  - Renamed: "my_func"
```

Apply the generated map when running Nsight Compute:

```bash
ncu --rename-kernels-path ncu-kernel-renames.yaml \
    --kernel-name-base demangled \
    --kernel-name my_func \
    --export my_func_profile \
    ./my_kokkos_app
```

This command creates `my_func_profile.ncu-rep`, which can be opened with the Nsight Compute GUI. Optional controls include:

- `--section LaunchStats` to collect launch statistics.
- `--launch-count 1` to profile one launch of the selected kernel.
- `--kill yes` to terminate the application after collecting the requested launches.
- `--rename-kernels off` to display the original demangled names.

If the full renamed label is `MyKokkosRegion.my_func`, pass `--kernel-name MyKokkosRegion.my_func`. The `--label-prefix MyKokkosRegion` option controls which labels enter the YAML map; it does not alter those labels.

### Validate the scripts with the included demo

Run the complete validation workflow:

```bash
cd test_suite
KOKKOS_PRIMER_BUILD_DIR="$PWD/build" ./verify_profiling_names.sh all
```

A successful run confirms that:

- Nsight Systems associates the demo kernels with their enclosing NVTX names.
- `profiling_name_tools.py` creates an NCU YAML map from the supported report output.
- NCU uses a renamed Kokkos label to filter the initial collection before any `.ncu-rep` is imported.
- The renamed label remains visible when the resulting NCU report is imported.

The end-to-end demo requires a CUDA-enabled `build/kokkos_tests`, `nsys`, `ncu`, access to GPU performance counters, and profiling permissions. Use the individual modes to isolate a failure:

```bash
./verify_profiling_names.sh unit       # Test CSV matching and YAML generation
./verify_profiling_names.sh preflight  # Check installed NCU flags and NSYS reports
./verify_profiling_names.sh nsys       # Capture the demo and generate its YAML map
./verify_profiling_names.sh ncu        # Use the existing map for first-pass NCU filtering
```

See the [Kokkos Profiling Guide](test_suite/PROFILING_GUIDE.md#kernel-naming-verification) for the demo workload, generated files, tested tool versions, and configurable build and output paths.

## Summary

| Approach                               | Best for                                                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **KOKKOS_TOOLS** (e.g. NVTX connector) | Nsight Systems / Nsight Compute timelines, readable Kokkos regions and kernel names, no recompile.                 |
| **Caliper** (as Kokkos Tools + CUPTI)  | Hierarchical reports, GPU time, CUDA API time, memory copy stats, JSON/analysis pipelines; can emit NVTX for Nsight Systems. |

Both approaches rely on the same **Kokkos profiling hooks**; switch between them by setting `KOKKOS_TOOLS_LIBS` to the appropriate library.

## Debugging

The profiling tools above are also the primary debugging tools for Kokkos on NVIDIA GPUs. In addition to profiling, consider these debugging strategies:

- `CUDA_LAUNCH_BLOCKING=1` — Forces synchronous kernel launches so errors are reported at the correct call site. Useful for isolating which kernel produces NaNs or illegal memory accesses.
- **Single-stream consolidation** — The helper library provides a drop-in utility (`NVKokkos::make_debug_streams`) that routes all execution spaces onto a single CUDA stream via an environment variable, without recompiling. This isolates stream synchronization bugs by serializing all GPU work. See [08-helper-library.md](08-helper-library.md).
- `compute-sanitizer` — NVIDIA's runtime checker for out-of-bounds access, race conditions, and uninitialized memory on the GPU:
  ```bash
  compute-sanitizer --tool memcheck ./my_kokkos_app
  compute-sanitizer --tool racecheck ./my_kokkos_app
  ```
- **Debug builds** — Build Kokkos with `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON` for bounds checking on View accesses (adds implicit fences after every kernel; do not use in production).
