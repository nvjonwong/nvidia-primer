#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to test GitLab CI pipeline locally using NVIDIA CUDA container
# This simulates what GitLab CI does without needing gitlab-runner
# NOTE: This must be run on the machine where Docker is running (not via Remote-SSH)
#
# Build directories are isolated in /tmp inside the container to avoid conflicts
# with development builds in your mounted source directory.
#
# Usage:
#   ./test-ci-locally.sh                    # Use default (latest from repo)
#   KOKKOS_VERSION=4.5.00 ./test-ci-locally.sh    # Use specific tag
#   KOKKOS_VERSION=develop ./test-ci-locally.sh   # Use specific branch
#   KOKKOS_VERSION=abc123def ./test-ci-locally.sh # Use specific commit hash

set -e

# Configuration using NVIDIA CUDA container
IMAGE="nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu22.04"
PROJECT_DIR="/tmp/kokkos-primer"
KOKKOS_REPO_URL="${KOKKOS_REPO_URL:-https://github.com/kokkos/kokkos.git}"
# Alternative (requires SSH access): ssh://git@gitlab-master.nvidia.com:12051/Devtech-Compute/kokkos.git
KOKKOS_VERSION="${KOKKOS_VERSION:-}"  # Tag, branch, or commit hash (empty = default branch)

# Get absolute path to the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(realpath "$SCRIPT_DIR")"

echo "Testing GitLab CI locally with NVIDIA CUDA container..."
echo "Project directory: $SCRIPT_DIR"
echo "Docker image: $IMAGE"
echo "Kokkos repository: $KOKKOS_REPO_URL"
if [ -n "$KOKKOS_VERSION" ]; then
    echo "Kokkos version: $KOKKOS_VERSION"
else
    echo "Kokkos version: default (latest from repo)"
fi
echo ""

# Verify the directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "ERROR: Directory does not exist: $SCRIPT_DIR"
    exit 1
fi


# Check if Docker can access this path
echo "Checking Docker host..."
DOCKER_HOST_INFO=$(docker info --format '{{.OperatingSystem}}' 2>/dev/null || echo "unknown")
echo "Docker OS: $DOCKER_HOST_INFO"
echo ""

# Check SSH agent
if [ -z "$SSH_AUTH_SOCK" ]; then
    echo "WARNING: SSH_AUTH_SOCK not set. SSH git operations may fail."
    echo "Run: eval \$(ssh-agent) && ssh-add"
fi

# Check if gitlab-master.nvidia.com is in known_hosts
if [ -f "$HOME/.ssh/known_hosts" ] && ! grep -q "gitlab-master.nvidia.com" "$HOME/.ssh/known_hosts"; then
    echo "WARNING: gitlab-master.nvidia.com not in known_hosts."
    echo "You may need to run: ssh-keyscan gitlab-master.nvidia.com >> ~/.ssh/known_hosts"
fi

# Build SSH mount options
SSH_MOUNTS=""
SSH_ENV=""
if [ -S "${SSH_AUTH_SOCK}" ] || [ -L "${SSH_AUTH_SOCK}" ]; then
    # Resolve symlink to get the real socket path (VS Code Remote-SSH uses symlinks)
    REAL_SSH_SOCK=$(readlink -f "${SSH_AUTH_SOCK}")
    
    if [ -S "$REAL_SSH_SOCK" ]; then
        # Get the directory containing the actual socket
        SSH_SOCK_DIR="$(dirname "${REAL_SSH_SOCK}")"
        
        # Mount the directory and use the real socket path
        SSH_MOUNTS="-v ${SSH_SOCK_DIR}:${SSH_SOCK_DIR}:ro"
        SSH_ENV="-e SSH_AUTH_SOCK=${REAL_SSH_SOCK}"
        echo "Mounting SSH agent from: ${SSH_AUTH_SOCK} -> ${REAL_SSH_SOCK}"
    else
        echo "WARNING: SSH socket symlink found but target doesn't exist"
    fi
fi

# Don't mount known_hosts - we'll create it inside the container
# This avoids issues when Docker runs on a different host (e.g., WSL Docker with Remote-SSH)
echo "Note: known_hosts will be created inside the container"

# Get current user's UID and GID to fix permissions at the end
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Run the CI pipeline in Docker as root (for package installation)
# Will fix file ownership at the end
# Check if we have a TTY for interactive mode
if [ -t 0 ]; then
    TTY_FLAGS="-it"
else
    TTY_FLAGS=""
fi

docker run --rm $TTY_FLAGS \
  --ulimit memlock=-1 \
  --gpus all \
  -v "$SCRIPT_DIR:$PROJECT_DIR" \
  $SSH_MOUNTS \
  $SSH_ENV \
  -w "$PROJECT_DIR" \
  -h "$(hostname)" \
  -e USER_ID="$USER_ID" \
  -e GROUP_ID="$GROUP_ID" \
  -e CI_PROJECT_DIR="$PROJECT_DIR" \
  -e CMAKE_BUILD_TYPE=Release \
  -e KOKKOS_SRC_DIR="/tmp/ci-kokkos/src" \
  -e KOKKOS_BUILD_DIR="/tmp/ci-kokkos/build" \
  -e KOKKOS_INSTALL_DIR="/tmp/ci-kokkos/install" \
  -e KOKKOS_REPO_URL="$KOKKOS_REPO_URL" \
  -e KOKKOS_VERSION="$KOKKOS_VERSION" \
  -e TEST_SUITE_BUILD_DIR="/tmp/ci-test-suite/build" \
  "$IMAGE" \
  bash -c '
    set -e
    
    echo ""
    echo "=== Installing dependencies (as root) ==="
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git build-essential ca-certificates openssh-client sudo wget
    
    echo ""
    echo "=== Installing CMake 3.28.3 (for CUDA20 support) ==="
    wget -q https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz
    tar -xzf cmake-3.28.3-linux-x86_64.tar.gz -C /opt
    rm cmake-3.28.3-linux-x86_64.tar.gz
    ln -sf /opt/cmake-3.28.3-linux-x86_64/bin/cmake /usr/local/bin/cmake
    ln -sf /opt/cmake-3.28.3-linux-x86_64/bin/ctest /usr/local/bin/ctest
    ln -sf /opt/cmake-3.28.3-linux-x86_64/bin/cpack /usr/local/bin/cpack
    
    echo ""
    echo "=== Installing Nsight Systems 2025.5.2 (for profiling) ==="
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2025.5.2_2025.5.2.266-1_amd64.deb
    DEBIAN_FRONTEND=noninteractive apt-get install -y ./nsight-systems-2025.5.2_2025.5.2.266-1_amd64.deb
    rm nsight-systems-2025.5.2_2025.5.2.266-1_amd64.deb
    
    echo ""
    echo "=== Checking environment ==="
    git --version
    cmake --version
    nvcc --version
    nsys --version 2>/dev/null || echo "Warning: nsys not found (profiling will not be available)"
    
    echo ""
    echo "=== Creating user ${USER_ID}:${GROUP_ID} ==="
    # Create group and user matching host IDs
    groupadd -g ${GROUP_ID} hostgroup 2>/dev/null || true
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash hostuser 2>/dev/null || true
    
    # Add hostuser to sudo group and allow passwordless sudo
    usermod -aG sudo hostuser 2>/dev/null || true
    echo "hostuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    
    echo ""
    echo "=== Setting up SSH agent forwarding for hostuser ==="
    # Make SSH socket accessible to hostuser
    if [ -n "${SSH_AUTH_SOCK}" ] && [ -S "${SSH_AUTH_SOCK}" ]; then
        SSH_SOCK_DIR=$(dirname "${SSH_AUTH_SOCK}")
        # Make socket directory and socket accessible
        chmod 755 "${SSH_SOCK_DIR}" 2>/dev/null || true
        chmod 666 "${SSH_AUTH_SOCK}" 2>/dev/null || true
        echo "SSH socket configured for hostuser: ${SSH_AUTH_SOCK}"
    fi
    
    echo ""
    echo "=== Switching to user ${USER_ID}:${GROUP_ID} for build ==="
    
    # Run the build as the host user to match file permissions
    runuser -u hostuser -- bash <<EOFUSER
        set -e
        export HOME=/tmp/home-${USER_ID}
        export SSH_AUTH_SOCK="${SSH_AUTH_SOCK}"
        export CI_PROJECT_DIR="${CI_PROJECT_DIR}"
        export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        export KOKKOS_SRC_DIR="${KOKKOS_SRC_DIR}"
        export KOKKOS_BUILD_DIR="${KOKKOS_BUILD_DIR}"
        export KOKKOS_INSTALL_DIR="${KOKKOS_INSTALL_DIR}"
        export KOKKOS_REPO_URL="${KOKKOS_REPO_URL}"
        export TEST_SUITE_BUILD_DIR="${TEST_SUITE_BUILD_DIR}"
        
        echo ""
        echo "=== Verifying SSH agent access as hostuser ==="
        if [ -S "\$SSH_AUTH_SOCK" ]; then
            echo "SSH agent socket: \$SSH_AUTH_SOCK"
            ssh-add -l 2>&1 || echo "Cannot access SSH agent"
        else
            echo "WARNING: SSH agent socket not found"
        fi
        
        echo ""
        echo "=== Fetching Kokkos (as hostuser) ==="
        mkdir -p "\$KOKKOS_SRC_DIR"
        if [ ! -d "\$KOKKOS_SRC_DIR/.git" ]; then
            # Determine if KOKKOS_VERSION is a commit hash (7-40 hex characters)
            if [ -n "\$KOKKOS_VERSION" ] && echo "\$KOKKOS_VERSION" | grep -qE "^[0-9a-f]{7,40}$"; then
                echo "Cloning Kokkos and checking out commit hash: \$KOKKOS_VERSION"
                # For commit hashes, we need full history (can'\''t use --depth 1)
                GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 12051" \
                    git clone "\$KOKKOS_REPO_URL" "\$KOKKOS_SRC_DIR"
                cd "\$KOKKOS_SRC_DIR"
                git checkout "\$KOKKOS_VERSION"
                cd -
            elif [ -n "\$KOKKOS_VERSION" ]; then
                echo "Cloning Kokkos branch/tag: \$KOKKOS_VERSION"
                # For branches and tags, use --depth 1 for faster clone
                GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 12051" \
                    git clone --branch "\$KOKKOS_VERSION" --depth 1 "\$KOKKOS_REPO_URL" "\$KOKKOS_SRC_DIR"
            else
                echo "Cloning Kokkos (default branch)"
                # No version specified, use default behavior
                GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 12051" \
                    git clone --depth 1 "\$KOKKOS_REPO_URL" "\$KOKKOS_SRC_DIR"
            fi
        else
            echo "Kokkos already exists at \$KOKKOS_SRC_DIR, skipping clone"
            if [ -n "\$KOKKOS_VERSION" ]; then
                echo "WARNING: KOKKOS_VERSION=\$KOKKOS_VERSION specified but directory already exists"
                echo "         To use a different version, delete kokkos/ directory first"
            fi
        fi
        
        echo ""
        echo "=== Building Kokkos ==="
        mkdir -p "$KOKKOS_BUILD_DIR" "$KOKKOS_INSTALL_DIR/bin"
        cp "$KOKKOS_SRC_DIR/bin/nvcc_wrapper" "$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper"
        chmod +x "$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper"
        
        # Note: Kokkos 5.x uses C++20 features by default
        # Building with C++20 ensures the headers are compatible with test suite
        cmake -S "$KOKKOS_SRC_DIR" -B "$KOKKOS_BUILD_DIR" \
            -D CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -D CMAKE_CXX_COMPILER="$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper" \
            -D Kokkos_ENABLE_CUDA=ON \
            -D Kokkos_ENABLE_OPENMP=ON \
            -D Kokkos_ENABLE_DEPRECATION_WARNINGS=ON \
            -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=ON \
            -D Kokkos_ARCH_NATIVE=ON \
            -D CMAKE_INSTALL_PREFIX="$KOKKOS_INSTALL_DIR"
        
        cmake --build "$KOKKOS_BUILD_DIR" --parallel
        cmake --install "$KOKKOS_BUILD_DIR"
        
        echo ""
        echo "=== Building test suite ==="
        if [ -d "$CI_PROJECT_DIR/test_suite" ]; then
            mkdir -p "$TEST_SUITE_BUILD_DIR"
            cmake -S "$CI_PROJECT_DIR/test_suite" -B "$TEST_SUITE_BUILD_DIR" \
                -D CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
                -D CMAKE_CXX_COMPILER="$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper" \
                -D CMAKE_PREFIX_PATH="$KOKKOS_INSTALL_DIR"
            cmake --build "$TEST_SUITE_BUILD_DIR" --parallel
            
            echo ""
            echo "=== Running tests ==="
            export OMP_NUM_THREADS=4
            ctest --test-dir "$TEST_SUITE_BUILD_DIR" --output-on-failure --parallel $(nproc)
        else
            echo "No test_suite directory found, skipping tests"
        fi
        
        echo ""
        echo "=== CI pipeline completed successfully! ==="
EOFUSER
  '
