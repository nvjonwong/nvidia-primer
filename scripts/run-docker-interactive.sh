#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to launch an interactive Docker environment
# This provides a consistent development environment without running CI tests
#
# Usage:
#   ./run-docker-interactive.sh

set -e

# Configuration using NVIDIA CUDA container
IMAGE="nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu22.04"
PROJECT_DIR="/tmp/kokkos/KokkosPrimer"

# Get absolute path to the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(realpath "$SCRIPT_DIR")"

echo "Starting interactive Docker environment..."
echo "Project directory (host): $SCRIPT_DIR"
echo "Project directory (container): $PROJECT_DIR"
echo "Docker image: $IMAGE"
echo ""

# Verify the directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "ERROR: Directory does not exist: $SCRIPT_DIR"
    exit 1
fi

# Check Docker
echo "Checking Docker..."
docker info > /dev/null 2>&1 || {
    echo "ERROR: Docker is not running or not accessible"
    exit 1
}

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
        echo "SSH agent forwarding: enabled (${REAL_SSH_SOCK})"
    else
        echo "WARNING: SSH socket symlink found but target doesn't exist"
        echo "SSH agent forwarding: disabled"
    fi
else
    echo "SSH agent forwarding: disabled (SSH_AUTH_SOCK not set)"
fi

# Get current user's UID and GID
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo ""
echo "Starting container..."
echo "Press Ctrl+D or type 'exit' to leave the container"
echo "============================================"
echo ""

# Check if we need to build a custom image with packages pre-installed
CUSTOM_IMAGE="kokkos-dev-env:latest"

if ! docker image inspect "$CUSTOM_IMAGE" > /dev/null 2>&1; then
    echo "Building custom development image (one-time setup)..."
    echo "This may take a few minutes..."
    echo ""
    
    # Create a temporary Dockerfile
    DOCKERFILE=$(mktemp)
    cat > "$DOCKERFILE" << 'EOFDOCKERFILE'
FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu22.04

# Install development tools
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git build-essential ca-certificates openssh-client sudo vim nano gosu wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Install CMake 3.28 from official releases
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.28.3-linux-x86_64.tar.gz -C /opt && \
    rm cmake-3.28.3-linux-x86_64.tar.gz && \
    ln -sf /opt/cmake-3.28.3-linux-x86_64/bin/cmake /usr/local/bin/cmake && \
    ln -sf /opt/cmake-3.28.3-linux-x86_64/bin/ctest /usr/local/bin/ctest && \
    ln -sf /opt/cmake-3.28.3-linux-x86_64/bin/cpack /usr/local/bin/cpack

# Install Nsight Systems CLI 2025.6.1 for profiling
RUN wget -q https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb && \
    apt-get update -qq && \
    apt-get install -y -qq libglib2.0-0 && \
    dpkg -i NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb && \
    rm NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure sudo to allow any UID to use sudo without password
# This is needed when running with arbitrary --user UIDs
RUN echo 'ALL ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chmod 0440 /etc/sudoers

# Pre-create /home/devuser directory with permissive ownership
# The entrypoint will fix ownership for the specific UID/GID
RUN mkdir -p /home/devuser && chmod 777 /home/devuser

# Pre-create X11 socket directory to avoid permission issues when attaching from VS Code
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

# Create an entrypoint that sets up the user and runs as that user
COPY <<'EOF' /entrypoint.sh
#!/bin/bash
set -e

# Create group if it doesn't exist
if ! getent group ${GROUP_ID} >/dev/null 2>&1; then
    groupadd -g ${GROUP_ID} devgroup
fi

# Create user if it doesn't exist, with bash as the shell
if ! getent passwd ${USER_ID} >/dev/null 2>&1; then
    useradd -u ${USER_ID} -g ${GROUP_ID} -d /home/devuser -s /bin/bash -m devuser
fi

# Set up CUDA environment if .bashrc doesn't exist
if [ ! -f /home/devuser/.bashrc ]; then
    cat > /home/devuser/.bashrc <<'EOFBASHRC'
# CUDA environment
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda

# Standard bashrc
if [ -f /etc/bash.bashrc ]; then
    . /etc/bash.bashrc
fi

# Enable color prompt
export PS1='\[\e[01;32m\]\u@\h\[\e[00m\]:\[\e[01;34m\]\w\[\e[00m\]\$ '

# Useful aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
EOFBASHRC
    chown ${USER_ID}:${GROUP_ID} /home/devuser/.bashrc
fi

# Set up SSH agent permissions if needed
if [ -n "${SSH_AUTH_SOCK}" ] && [ -S "${SSH_AUTH_SOCK}" ]; then
    SSH_SOCK_DIR=$(dirname "${SSH_AUTH_SOCK}")
    chmod 755 "${SSH_SOCK_DIR}" 2>/dev/null || true
    chmod 666 "${SSH_AUTH_SOCK}" 2>/dev/null || true
fi

# Configure root's bashrc to auto-switch to devuser
# This ensures that docker exec (which defaults to root) will become devuser
cat > /root/.bashrc <<'EOFROOTBASHRC'
# Auto-switch to devuser if running as root in an interactive shell
if [ "$(id -u)" = "0" ] && [ -n "$BASH_VERSION" ] && [ -t 0 ]; then
    export HOME=/home/devuser
    cd /home/devuser 2>/dev/null || cd /
    exec gosu devuser bash --login
fi
EOFROOTBASHRC

# Switch to devuser and execute the command
exec gosu devuser "$@"
EOF

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
EOFDOCKERFILE
    
    # Build the image
    docker build -t "$CUSTOM_IMAGE" -f "$DOCKERFILE" . > /dev/null 2>&1
    rm -f "$DOCKERFILE"
    
    echo "Custom image built successfully!"
    echo ""
fi

CONTAINER_NAME="kokkos-dev-${USER_ID}"

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' is already running."
    echo "Attaching to existing container..."
    echo ""
    docker exec -it --user devuser -e HOME="/home/devuser" "$CONTAINER_NAME" bash --login
    exit 0
fi

# Check if container exists but is stopped
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Found stopped container '${CONTAINER_NAME}'. Starting it..."
    docker start "$CONTAINER_NAME"
    docker exec -it --user devuser -e HOME="/home/devuser" "$CONTAINER_NAME" bash --login
    exit 0
fi

echo "Creating new container: ${CONTAINER_NAME}"
echo "User: devuser (UID=${USER_ID}, GID=${GROUP_ID})"
echo ""

# Set up SSH agent permissions if needed
if [ -n "${SSH_AUTH_SOCK}" ] && [ -S "${SSH_AUTH_SOCK}" ]; then
    echo "SSH agent forwarding: enabled"
fi

echo ""
echo "==================================================================="
echo "Container will run in the background."
echo "This shell will attach to it automatically."
echo ""
echo "To stop the container later, run:"
echo "  docker stop ${CONTAINER_NAME}"
echo ""
echo "To remove the container, run:"
echo "  docker rm ${CONTAINER_NAME}"
echo ""
echo "The container will persist across sessions until you remove it."
echo "==================================================================="
echo ""

# Start container as root temporarily to set up the user
docker run -d \
  --name "$CONTAINER_NAME" \
  --ulimit memlock=-1 \
  --gpus all \
  --label "devcontainer.metadata=[{\"remoteUser\":\"devuser\"}]" \
  --label "com.visualstudio.code.devcontainers.user=devuser" \
  -v "$SCRIPT_DIR:$PROJECT_DIR" \
  $SSH_MOUNTS \
  $SSH_ENV \
  -w "$PROJECT_DIR" \
  -h "docker-kokkos-dev" \
  -e USER_ID="$USER_ID" \
  -e GROUP_ID="$GROUP_ID" \
  --entrypoint /bin/bash \
  "$CUSTOM_IMAGE" \
  -c "
    # Create group and user with bash shell
    groupadd -g ${GROUP_ID} devgroup 2>/dev/null || true
    useradd -u ${USER_ID} -g ${GROUP_ID} -d /home/devuser -s /bin/bash devuser 2>/dev/null || true
    
    # Set up devuser environment
    if [ ! -f /home/devuser/.bashrc ]; then
      cat > /home/devuser/.bashrc <<'EOFBASHRC'
export PATH=/usr/local/cuda/bin:\${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda
if [ -f /etc/bash.bashrc ]; then . /etc/bash.bashrc; fi
export PS1='\[\e[01;32m\]\u@\h\[\e[00m\]:\[\e[01;34m\]\w\[\e[00m\]\$ '
alias ll='ls -alF'; alias la='ls -A'; alias l='ls -CF'
EOFBASHRC
      chown ${USER_ID}:${GROUP_ID} /home/devuser/.bashrc
    fi
    
    # Configure root's bashrc to auto-switch to devuser for interactive shells
    # This ensures docker exec uses devuser by default
    cat >> /root/.bashrc <<EOFROOTBASHRC

# Auto-switch to devuser if running as root in an interactive shell
if [ \\\"\\\$(id -u)\\\" = \\\"0\\\" ] && [ -n \\\"\\\$BASH_VERSION\\\" ] && [ -t 0 ]; then
    export HOME=/home/devuser
    cd ${PROJECT_DIR} 2>/dev/null || cd /home/devuser 2>/dev/null || cd /
    exec gosu devuser bash --login
fi
EOFROOTBASHRC
    
    # Now exec as devuser and sleep infinity
    exec gosu devuser sleep infinity
  "

# Wait a moment for the entrypoint to complete setup
sleep 2

echo "Environment ready!"
echo ""
echo "==================================================================="
echo "Container: ${CONTAINER_NAME}"
echo "User: devuser (UID=${USER_ID}, GID=${GROUP_ID})"
echo "Working directory: $PROJECT_DIR"
echo ""
echo "Available tools:"
echo "  - git, cmake 3.28.3, nvcc (CUDA 13.0.2)"
echo "  - C++ compiler (g++)"
echo "  - nsys (Nsight Systems 2025.6.1) for profiling"
echo ""
echo "Your source files are mounted from the host."
echo ""
echo "The container is running as devuser by default."
echo "VS Code/Cursor will automatically use the correct user and permissions."
echo "All file operations will work correctly."
echo ""
echo "Type 'exit' or press Ctrl+D to leave (container keeps running)."
echo "==================================================================="
echo ""

# Attach to the container (already running as devuser)
docker exec -it --user devuser -e HOME="/home/devuser" "$CONTAINER_NAME" bash --login

echo ""
echo "Exited Docker container."
