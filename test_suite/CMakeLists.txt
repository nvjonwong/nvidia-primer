# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)
project(KokkosPrimerTests LANGUAGES CXX CUDA)

# Download CPM.cmake
set(CPM_DOWNLOAD_VERSION 0.38.1)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
         https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
         ${CPM_DOWNLOAD_LOCATION}
    )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# Set C++ and CUDA standards - Kokkos 5.x requires C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures if not already defined
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Common NVIDIA GPU architectures for CUDA 12.0+
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100 120)
endif()

# Find CUDA Toolkit first
find_package(CUDAToolkit QUIET)

# Find Kokkos
find_package(Kokkos REQUIRED)

# Fetch Google Test using CPM
CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.14.0
  VERSION 1.14.0
  OPTIONS
    "INSTALL_GTEST OFF"
    "BUILD_GMOCK OFF"
)

# Test executable
add_executable(kokkos_tests
    test_main.cpp
    test_views.cpp
    test_execution_spaces.cpp
    test_atomics_prefetch.cpp
    test_view_traits_mdrange.cpp
    test_helpers_streams.cpp
    test_sync_behavior.cpp
    test_stream_ordered_view.cpp
    test_uvm_aliasing_ilp.cpp
    test_allocation_threshold.cpp
    test_allocation_threshold_impl.cpp
    test_cuda_reference.cu
    test_profiling_names.cpp
)

# Add CUDA-specific compile options
if(Kokkos_ENABLE_CUDA)
    target_compile_options(kokkos_tests PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CXX>:--extended-lambda>
    )
    # Pass OpenMP flag to nvcc for .cu files
    if(Kokkos_ENABLE_OPENMP)
        target_compile_options(kokkos_tests PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
        )
    endif()
endif()

# Link against Kokkos and GTest
target_link_libraries(kokkos_tests 
    Kokkos::kokkos
    gtest
    gtest_main
)

# NVTX3 is header-only, no linking required
if(TARGET CUDA::nvtx3)
    target_link_libraries(kokkos_tests CUDA::nvtx3)
endif()

# Include the parent directory for nvkokkos_helpers.hpp
target_include_directories(kokkos_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Enable testing
enable_testing()

# Add test
add_test(NAME KokkosPrimerTests COMMAND kokkos_tests)

# Print configuration
message(STATUS "Kokkos Test Suite Configuration:")
message(STATUS "  Kokkos version: ${Kokkos_VERSION}")
message(STATUS "  Kokkos backends: ${Kokkos_DEVICES}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
if(Kokkos_ENABLE_CUDA)
    message(STATUS "  CUDA enabled: YES")
    message(STATUS "  CUDA standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "  CUDA enabled: NO")
endif()
