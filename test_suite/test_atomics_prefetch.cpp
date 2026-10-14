/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "../nvkokkos_helpers.hpp"

void test_atomic_add() {
    const int N = 1000;
    Kokkos::View<int*> counter("counter", 1);
    
    Kokkos::parallel_for("atomic_add", N, KOKKOS_LAMBDA(int i) {
        Kokkos::atomic_add(&counter(0), 1);
    });
    Kokkos::fence();
    
    auto h_counter = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counter);
    EXPECT_EQ(h_counter(0), N);
}

TEST(Atomics, AtomicAdd) { test_atomic_add(); }
