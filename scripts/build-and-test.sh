#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the same steps as CI, but directly in your current container
# This is the simplest option when you're already inside a container

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CI_PROJECT_DIR="${CI_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
export KOKKOS_SRC_DIR="$CI_PROJECT_DIR/kokkos/src"
export KOKKOS_BUILD_DIR="$CI_PROJECT_DIR/kokkos/build"
export KOKKOS_INSTALL_DIR="$CI_PROJECT_DIR/kokkos/install"
export KOKKOS_REPO_URL="${KOKKOS_REPO_URL:-ssh://git@gitlab-master.nvidia.com:12051/Devtech-Compute/kokkos.git}"
export KOKKOS_VERSION="${KOKKOS_VERSION:-}"

echo "=================================="
echo "Building Kokkos Primer Test Suite"
echo "=================================="
echo "Project dir: $CI_PROJECT_DIR"
echo "Kokkos repo: $KOKKOS_REPO_URL"
echo "Kokkos version: ${KOKKOS_VERSION:-default}"
echo "Build type: $CMAKE_BUILD_TYPE"
echo ""

# Step 1: Fetch Kokkos
echo "=== Step 1: Fetching Kokkos ==="
mkdir -p "$KOKKOS_SRC_DIR"

if [ ! -d "$KOKKOS_SRC_DIR/.git" ]; then
    if [ -n "$KOKKOS_VERSION" ] && echo "$KOKKOS_VERSION" | grep -qE "^[0-9a-f]{7,40}$"; then
        echo "Cloning Kokkos and checking out commit: $KOKKOS_VERSION"
        git clone "$KOKKOS_REPO_URL" "$KOKKOS_SRC_DIR"
        cd "$KOKKOS_SRC_DIR"
        git checkout "$KOKKOS_VERSION"
        cd -
    elif [ -n "$KOKKOS_VERSION" ]; then
        echo "Cloning Kokkos branch/tag: $KOKKOS_VERSION"
        git clone --branch "$KOKKOS_VERSION" --depth 1 "$KOKKOS_REPO_URL" "$KOKKOS_SRC_DIR"
    else
        echo "Cloning Kokkos (default branch)"
        git clone --depth 1 "$KOKKOS_REPO_URL" "$KOKKOS_SRC_DIR"
    fi
else
    echo "Kokkos already exists at $KOKKOS_SRC_DIR, skipping clone"
    if [ -n "$KOKKOS_VERSION" ]; then
        echo "WARNING: KOKKOS_VERSION=$KOKKOS_VERSION specified but directory exists"
        echo "         Delete kokkos/ directory to use a different version"
    fi
fi

# Step 2: Build Kokkos
echo ""
echo "=== Step 2: Building Kokkos ==="

# Clean and create directories
rm -rf "$KOKKOS_BUILD_DIR"
mkdir -p "$KOKKOS_BUILD_DIR" "$KOKKOS_INSTALL_DIR/bin"

# Copy nvcc_wrapper
cp "$KOKKOS_SRC_DIR/bin/nvcc_wrapper" "$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper"
chmod +x "$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper"

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

# Configure Kokkos
cmake -S "$KOKKOS_SRC_DIR" -B "$KOKKOS_BUILD_DIR" \
    -D CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -D CMAKE_CXX_COMPILER="$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper" \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_DEPRECATION_WARNINGS=ON \
    -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON \
    ${ARCH_FLAGS} \
    -D CMAKE_INSTALL_PREFIX="$KOKKOS_INSTALL_DIR"

# Build and install
cmake --build "$KOKKOS_BUILD_DIR" --parallel $(nproc)
cmake --install "$KOKKOS_BUILD_DIR"

echo "✓ Kokkos built and installed"

# Step 3: Build test suite
echo ""
echo "=== Step 3: Building test suite ==="

if [ ! -d "$CI_PROJECT_DIR/test_suite" ]; then
    echo "ERROR: test_suite directory not found at $CI_PROJECT_DIR/test_suite"
    exit 1
fi

# Clean and create build directory
rm -rf "$CI_PROJECT_DIR/test_suite/build"
mkdir -p "$CI_PROJECT_DIR/test_suite/build"

# Configure test suite
cmake -S "$CI_PROJECT_DIR/test_suite" -B "$CI_PROJECT_DIR/test_suite/build" \
    -D CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -D CMAKE_CXX_COMPILER="$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper" \
    -D CMAKE_PREFIX_PATH="$KOKKOS_INSTALL_DIR" \
    -D Kokkos_ROOT="$KOKKOS_INSTALL_DIR"

# Build test suite
cmake --build "$CI_PROJECT_DIR/test_suite/build" --parallel $(nproc)

echo "✓ Test suite built"

# Step 4: Run tests
echo ""
echo "=== Step 4: Running tests ==="

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

ctest --test-dir "$CI_PROJECT_DIR/test_suite/build" \
    --output-on-failure \
    --parallel $(nproc)

echo ""
echo "=================================="
echo "✓ All steps completed successfully!"
echo "=================================="
