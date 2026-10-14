#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Quick build script for development inside the container
# Run from /tmp/kokkos/KokkosPrimer directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

KOKKOS_SRC="$PROJECT_DIR/kokkos/src"
KOKKOS_BUILD="$PROJECT_DIR/kokkos/build"
KOKKOS_INSTALL="$PROJECT_DIR/kokkos/install"
TEST_BUILD="$PROJECT_DIR/test_suite/build"

echo "==================================="
echo "Development Build Script"
echo "==================================="

# Check if Kokkos source exists
if [ ! -d "$KOKKOS_SRC/.git" ]; then
    echo "ERROR: Kokkos source not found at $KOKKOS_SRC"
    echo "Please clone Kokkos first:"
    echo "  mkdir -p kokkos/src"
    echo "  git clone --branch 5.0.1 https://github.com/kokkos/kokkos.git kokkos/src"
    exit 1
fi

# Check if Kokkos is installed
if [ ! -f "$KOKKOS_INSTALL/lib/cmake/Kokkos/KokkosConfig.cmake" ]; then
    echo ""
    echo "=== Building Kokkos ==="
    
    # Clean and create directories
    rm -rf "$KOKKOS_BUILD"
    mkdir -p "$KOKKOS_BUILD" "$KOKKOS_INSTALL/bin"
    
    # Copy nvcc_wrapper
    cp "$KOKKOS_SRC/bin/nvcc_wrapper" "$KOKKOS_INSTALL/bin/nvcc_wrapper"
    chmod +x "$KOKKOS_INSTALL/bin/nvcc_wrapper"
    
    # Detect GPU architecture
    ARCH_FLAGS=""
    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        case "$COMPUTE_CAP" in
            90) ARCH_FLAGS="-D Kokkos_ARCH_HOPPER90=ON" ;;
            89) ARCH_FLAGS="-D Kokkos_ARCH_ADA89=ON" ;;
            86) ARCH_FLAGS="-D Kokkos_ARCH_AMPERE86=ON" ;;
            80) ARCH_FLAGS="-D Kokkos_ARCH_AMPERE80=ON" ;;
            *)  ARCH_FLAGS="-D Kokkos_ARCH_NATIVE=ON" ;;
        esac
        echo "Detected compute capability: ${COMPUTE_CAP} → ${ARCH_FLAGS}"
    else
        ARCH_FLAGS="-D Kokkos_ARCH_NATIVE=ON"
        echo "No GPU detected, using NATIVE arch"
    fi
    
    # Configure
    cmake -S "$KOKKOS_SRC" -B "$KOKKOS_BUILD" \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_CXX_COMPILER="$KOKKOS_INSTALL/bin/nvcc_wrapper" \
        -D Kokkos_ENABLE_CUDA=ON \
        -D Kokkos_ENABLE_OPENMP=ON \
        -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON \
        ${ARCH_FLAGS} \
        -D CMAKE_INSTALL_PREFIX="$KOKKOS_INSTALL"
    
    # Build and install
    cmake --build "$KOKKOS_BUILD" --parallel $(nproc)
    cmake --install "$KOKKOS_BUILD"
    
    echo "✓ Kokkos installed to $KOKKOS_INSTALL"
else
    echo "✓ Kokkos already installed at $KOKKOS_INSTALL"
fi

echo ""
echo "=== Building Test Suite ==="

# Clean test build
rm -rf "$TEST_BUILD"
mkdir -p "$TEST_BUILD"

# Configure test suite
cmake -S "$PROJECT_DIR/test_suite" -B "$TEST_BUILD" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER="$KOKKOS_INSTALL/bin/nvcc_wrapper" \
    -D CMAKE_PREFIX_PATH="$KOKKOS_INSTALL"

# Build
cmake --build "$TEST_BUILD" --parallel $(nproc)

echo ""
echo "==================================="
echo "✓ Build complete!"
echo "==================================="
echo ""
echo "To run tests:"
echo "  cd test_suite/build"
echo "  ctest --output-on-failure"
echo ""
echo "Or run directly:"
echo "  ./test_suite/build/kokkos_tests"
echo ""
