/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "../nvkokkos_helpers.hpp"

void test_helpers_make_view_uninitialized() {
    const int N = 1000;
    auto view = NVKokkos::make_view_uninitialized<double*>("test", N);
    
    Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(int i) {
        view(i) = i * 3.14;
    });
    Kokkos::fence();
    
    double sum = 0;
    Kokkos::parallel_reduce("sum", N,
        KOKKOS_LAMBDA(int i, double& lsum) {
            lsum += view(i);
        }, sum);
    
    double expected = N * (N - 1) / 2.0 * 3.14;
    EXPECT_NEAR(sum, expected, 1e-6);
}

TEST(Helpers, MakeViewUninitialized) { test_helpers_make_view_uninitialized(); }
