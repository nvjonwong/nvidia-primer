#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build script for Kokkos test suite

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Kokkos Test Suite Build Script${NC}"
echo "================================"

# Parse arguments
BUILD_TYPE=${1:-Release}
KOKKOS_PATH=${2:-""}
BUILD_DIR="build"
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --kokkos-path)
            KOKKOS_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean           Clean build directory before building"
            echo "  --debug           Build in Debug mode (default: Release)"
            echo "  --kokkos-path     Path to Kokkos installation"
            echo "  --help            Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  KOKKOS_ROOT       Alternative way to specify Kokkos path"
            echo ""
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Try to find Kokkos if not specified
if [ -z "$KOKKOS_PATH" ]; then
    if [ -n "$KOKKOS_ROOT" ]; then
        KOKKOS_PATH="$KOKKOS_ROOT"
        echo -e "${YELLOW}Using KOKKOS_ROOT: $KOKKOS_PATH${NC}"
    else
        echo -e "${YELLOW}Warning: Kokkos path not specified${NC}"
        echo "Will rely on CMake's find_package. If build fails, use:"
        echo "  ./build.sh --kokkos-path /path/to/kokkos/install"
        echo "or set KOKKOS_ROOT environment variable"
    fi
fi

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo ""
echo -e "${GREEN}Configuring with CMake...${NC}"
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

if [ -n "$KOKKOS_PATH" ]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$KOKKOS_PATH")
fi

cmake .. "${CMAKE_ARGS[@]}"

# Build
echo ""
echo -e "${GREEN}Building...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Success message
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build successful!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Test executable: $BUILD_DIR/kokkos_tests"
echo ""
echo "To run tests:"
echo "  cd $BUILD_DIR && ./kokkos_tests"
echo ""
echo "Or use CTest:"
echo "  cd $BUILD_DIR && ctest --verbose"
echo ""

# Detect CUDA and provide additional info
if cmake -LA -N .. 2>/dev/null | grep -q "Kokkos_ENABLE_CUDA.*ON"; then
    echo -e "${YELLOW}CUDA backend detected${NC}"
    echo ""
    echo "Additional run options:"
    echo "  - Debug mode (single stream):"
    echo "      export KOKKOS_DEBUG_SINGLE_STREAM=1"
    echo "      ./kokkos_tests"
    echo ""
    echo "  - CUDA launch blocking (for debugging):"
    echo "      export CUDA_LAUNCH_BLOCKING=1"
    echo "      ./kokkos_tests"
    echo ""
    echo "  - Profile with Nsight Systems:"
    echo "      nsys profile --stats=true ./kokkos_tests"
    echo ""
fi

exit 0

