/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_main.cpp
 * @brief Main test driver with Kokkos initialization
 */

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Custom test environment to initialize/finalize Kokkos once for all tests
class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        int argc = 0;
        char** argv = nullptr;
        Kokkos::initialize(argc, argv);
    }

    void TearDown() override {
        Kokkos::finalize();
    }
};

// Register the Kokkos environment
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
    return RUN_ALL_TESTS();
}
