/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "../nvkokkos_helpers.hpp"

void test_atomic_view() {
    using AtomicView = Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Atomic>>;
    constexpr bool is_atomic = NVKokkos::is_atomic_view<AtomicView>();
    EXPECT_TRUE(is_atomic);
}

void test_mdrange_2d() {
    const int M = 100;
    const int N = 200;
    Kokkos::View<int**> data("data", M, N);
    
    Kokkos::parallel_for("md_tiled",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}, {1, 64}),
        KOKKOS_LAMBDA(int i, int j) {
            data(i, j) = i + j;
        });
    Kokkos::fence();
    
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    EXPECT_EQ(h_data(0, 0), 0);
    EXPECT_EQ(h_data(M-1, N-1), M + N - 2);
}

void test_mdrange_2d_helper() {
    const int M = 100;
    const int N = 200;
    Kokkos::View<int**> data("data", M, N);
    
    // Use helper function with recommended tiling
    Kokkos::parallel_for("md_helper",
        NVKokkos::make_mdrange_2d({0, 0}, {M, N}),
        KOKKOS_LAMBDA(int i, int j) {
            data(i, j) = i * 10 + j;
        });
    Kokkos::fence();
    
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    EXPECT_EQ(h_data(0, 0), 0);
    EXPECT_EQ(h_data(5, 7), 57);
}

void test_mdrange_3d_helper() {
    const int X = 10;
    const int Y = 20;
    const int Z = 30;
    Kokkos::View<int***> data("data", X, Y, Z);
    
    // Use 3D helper function with recommended tiling
    Kokkos::parallel_for("md_3d_helper",
        NVKokkos::make_mdrange_3d({0, 0, 0}, {X, Y, Z}),
        KOKKOS_LAMBDA(int i, int j, int k) {
            data(i, j, k) = i + j + k;
        });
    Kokkos::fence();
    
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    EXPECT_EQ(h_data(0, 0, 0), 0);
    EXPECT_EQ(h_data(X-1, Y-1, Z-1), X + Y + Z - 3);
}

TEST(ViewTraits, AtomicView) { test_atomic_view(); }
TEST(MDRangePolicy, 2DTiling) { test_mdrange_2d(); }
TEST(MDRangePolicy, 2DHelper) { test_mdrange_2d_helper(); }
TEST(MDRangePolicy, 3DHelper) { test_mdrange_3d_helper(); }
