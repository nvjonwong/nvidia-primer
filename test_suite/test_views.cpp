/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_views.cpp
 * @brief Tests for Kokkos View creation - uses free functions for CUDA lambda compatibility
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "../nvkokkos_helpers.hpp"

// Free functions for CUDA lambda compatibility (extended lambdas cant be in private/protected methods)

void test_default_creation() {
    const int N = 1000;
    Kokkos::View<double*> view("test_view", N);
    
    EXPECT_EQ(view.extent(0), N);
    EXPECT_STREQ(view.label().c_str(), "test_view");
}

void test_without_initialization() {
    const int N = 1000;
    auto view = Kokkos::View<double*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "uninit"),
        N
    );
    
    EXPECT_EQ(view.extent(0), N);
    Kokkos::deep_copy(view, 42.0);
    Kokkos::fence();
    
    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    EXPECT_DOUBLE_EQ(h_view(0), 42.0);
    EXPECT_DOUBLE_EQ(h_view(N-1), 42.0);
}

void test_make_view_uninitialized() {
    const int N = 1000;
    auto view = NVKokkos::make_view_uninitialized<double*>("helper_uninit", N);
    
    EXPECT_EQ(view.extent(0), N);
    
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

void test_make_view_filled() {
    const int N = 1000;
    Kokkos::Cuda exec_space;
    auto view = NVKokkos::make_view_filled<double*>(exec_space, "helper_filled", 7.5, N);

    EXPECT_EQ(view.extent(0), N);

    exec_space.fence();
    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    EXPECT_DOUBLE_EQ(h_view(0), 7.5);
    EXPECT_DOUBLE_EQ(h_view(N/2), 7.5);
    EXPECT_DOUBLE_EQ(h_view(N-1), 7.5);
}

// Wrapper TEST macros that call the free functions
TEST(ViewCreation, DefaultCreation) { test_default_creation(); }
TEST(ViewCreation, WithoutInitialization) { test_without_initialization(); }
TEST(ViewHelpers, MakeViewUninitialized) { test_make_view_uninitialized(); }
TEST(ViewHelpers, MakeViewFilled) { test_make_view_filled(); }
