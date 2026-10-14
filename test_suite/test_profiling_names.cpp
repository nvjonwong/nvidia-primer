/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_profiling_names.cpp
 * @brief Stable kernel-name and NVTX-range workload for Nsight integration tests.
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#ifdef KOKKOS_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

namespace {

constexpr int kProfilingNameElements = 4096;

#ifdef KOKKOS_ENABLE_CUDA
class ScopedNvtxRange {
 public:
  explicit ScopedNvtxRange(const char* name) { nvtxRangePushA(name); }
  ~ScopedNvtxRange() { nvtxRangePop(); }

  ScopedNvtxRange(const ScopedNvtxRange&) = delete;
  ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
};
#endif

struct ProfilingNameFunctor {
  Kokkos::View<int*> values;

  KOKKOS_FUNCTION void operator()(const int i) const { values(i) += 2; }
};

struct ProfilingNameWorkTag {};

struct ProfilingNameTaggedFunctor {
  Kokkos::View<int*> values;

  KOKKOS_FUNCTION void operator()(ProfilingNameWorkTag, const int i) const {
    values(i) += 3;
  }
};

#ifdef KOKKOS_ENABLE_CUDA
bool run_profiling_name_kernels() {
  Kokkos::View<int*> values("profiling_name_values", kProfilingNameElements);
  Kokkos::deep_copy(values, 0);

  {
    ScopedNvtxRange range{"KokkosPrimer.NcuRename.Lambda"};
    Kokkos::parallel_for(
        "KokkosPrimer.NcuRename.LambdaDispatch", kProfilingNameElements,
        KOKKOS_LAMBDA(const int i) { values(i) += 1; });
  }

  {
    ScopedNvtxRange outer{"KokkosPrimer.NcuRename.Outer"};
    {
      ScopedNvtxRange range{"KokkosPrimer.NcuRename.Functor"};
      Kokkos::parallel_for(
          "KokkosPrimer.NcuRename.FunctorDispatch", kProfilingNameElements,
          ProfilingNameFunctor{values});
    }
  }

  {
    ScopedNvtxRange range{"KokkosPrimer.NcuRename.WorkTag"};
    using TaggedPolicy =
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, ProfilingNameWorkTag>;
    Kokkos::parallel_for(
        "KokkosPrimer.NcuRename.WorkTagDispatch",
        TaggedPolicy(0, kProfilingNameElements),
        ProfilingNameTaggedFunctor{values});
    Kokkos::fence("ProfilingNames.StableKernelNames");
  }

  const auto host_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
  for (int i = 0; i < kProfilingNameElements; ++i) {
    if (host_values(i) != 6) {
      return false;
    }
  }
  return true;
}
#endif

}  // namespace

TEST(ProfilingNames, StableKernelNames) {
#ifndef KOKKOS_ENABLE_CUDA
  GTEST_SKIP() << "Kernel renaming integration requires the Kokkos CUDA backend";
#else
  ASSERT_TRUE(run_profiling_name_kernels());
#endif
}
