<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kokkos Infrastructure Guide

Guide to CI/CD setup, Docker containers, and version management for the Kokkos NVIDIA GPU Primer.

---

## Docker Container Setup

### Quick Docker Run

#### Interactive Shell

```bash
docker run --ulimit memlock=-1 --rm -it \
  --privileged --gpus all \
  -h $(hostname) \
  -v "$(realpath .):/tmp/kokkos" \
  nvidia/cuda:12.0.0-devel-ubuntu22.04
```

#### Run Tests Directly

```bash
docker run --ulimit memlock=-1 --rm -it \
  --privileged --gpus all \
  -v "$(realpath .):/tmp/kokkos" \
  nvidia/cuda:12.0.0-devel-ubuntu22.04 \
  bash -c "cd /tmp/kokkos/KokkosPrimer/test_suite && ./build.sh && cd build && ./kokkos_tests"
```

### Existing Container (GitLab Runner)

If you have a GitLab runner container:

```bash
# Connect to container
docker exec -it gitlab-runner-container bash

# Navigate to project
cd /tmp/kokkos/KokkosPrimer/test_suite

# Build and test
./build.sh
cd build && ./kokkos_tests
```

### Directory Mapping

**Host Path → Container Path:**
- `/home/jonathanw/scratch_asim/kokkos/KokkosPrimer` → `/tmp/kokkos/KokkosPrimer`

**Important:** Use container paths in scripts and CI configs!

### GPU Access in Docker

**Requirements:**
- NVIDIA Docker runtime installed
- `--gpus all` flag
- `--privileged` flag (for profiling)

**Verify GPU Access:**
```bash
docker run --gpus all nvidia/cuda:12.0.0-base nvidia-smi
```

---

## GitLab CI/CD Configuration

### Basic Configuration

`.gitlab-ci.yml`:
```yaml
image: nvidia/cuda:12.0.0-devel-ubuntu22.04

variables:
  GIT_SUBMODULE_STRATEGY: recursive

stages:
  - build
  - test
  - baseline

before_script:
  - apt-get update && apt-get install -y cmake build-essential

build_tests:
  stage: build
  script:
    - cd KokkosPrimer/test_suite
    - ./build.sh
  artifacts:
    paths:
      - KokkosPrimer/test_suite/build/
    expire_in: 1 hour

run_tests:
  stage: test
  dependencies:
    - build_tests
  script:
    - cd KokkosPrimer/test_suite/build
    - ./kokkos_tests
  artifacts:
    when: on_failure
    paths:
      - KokkosPrimer/test_suite/build/Testing/
```

### With Baseline Comparison

```yaml
generate_baseline:
  stage: baseline
  dependencies:
    - build_tests
  script:
    - cd KokkosPrimer/test_suite
    - export KOKKOS_VERSION=$(git describe --tags)
    - ./create_all_baselines.sh baselines/ci_${CI_COMMIT_SHA}
  artifacts:
    paths:
      - KokkosPrimer/test_suite/baselines/
    expire_in: 1 week

compare_baseline:
  stage: baseline
  dependencies:
    - generate_baseline
  script:
    - cd KokkosPrimer/test_suite
    - ./compare_all_baselines.sh baselines/main baselines/ci_${CI_COMMIT_SHA}
  artifacts:
    when: on_failure
    paths:
      - KokkosPrimer/test_suite/baselines/comprehensive_comparison_*.txt
```

### With GPU Runner

```yaml
run_tests_gpu:
  stage: test
  tags:
    - gpu
    - cuda
  script:
    - nvidia-smi  # Verify GPU access
    - cd KokkosPrimer/test_suite
    - ./build.sh
    - cd build && ./kokkos_tests
```

---

## Version Management

### Tracking Kokkos Versions

#### Using Git Tags

```bash
# Tag current version
cd kokkos
git tag -a v5.0.1 -m "Kokkos 5.0.1"

# Generate baseline with version
cd ../KokkosPrimer/test_suite
export KOKKOS_VERSION="5.0.1"
./create_all_baselines.sh baselines/kokkos_5.0.1
```

#### Using Git Describe

```bash
export KOKKOS_VERSION=$(cd kokkos && git describe --tags)
./create_all_baselines.sh baselines/kokkos_${KOKKOS_VERSION}
```

#### Manual Version

```bash
export KOKKOS_VERSION="5.0.1-custom"
./create_all_baselines.sh baselines/kokkos_5.0.1_custom
```

### Version Comparison Strategy

**Keep Baselines For:**
- Major versions (5.0, 5.1, 6.0)
- Release candidates (5.1.0-rc1)
- Critical patches (5.0.2)

**Archive/Remove:**
- Development commits
- Failed baselines
- Duplicate configurations

### Baseline Naming Convention

```text
baselines/
├── kokkos_5.0.1/                    # Release version
├── kokkos_5.0.1_async_on/           # Configuration variant
├── kokkos_5.1.0-rc1/                # Release candidate
├── kokkos_5.1.0/                    # Next release
└── kokkos_main_20260109/            # Development snapshot
```

---

## Baseline Storage Strategy

### What to Store in Git

**Include:**
```bash
git add baselines/*/metadata.txt
git add baselines/*/BASELINE_SUMMARY.txt
git add baselines/*/*_sync.txt
git add baselines/*/*_timeline.txt
git add baselines/*/*.md
```

**Exclude (`.gitignore`):**
```gitignore
# Large binary files
baselines/**/*.nsys-rep
baselines/**/*.sqlite
baselines/**/*.qdrep

# Temporary files
baselines/**/build/
baselines/**/*.tmp
```

### Size Management

**Per Baseline:**
- `.nsys-rep` files: ~50-100 MB each (7 categories)
- Text files: ~1-5 MB total
- **Total:** ~500 MB - 1 GB

**Optimization:**
```bash
# Remove large files after analysis
find baselines -name "*.nsys-rep" -delete
find baselines -name "*.sqlite" -delete

# Keep only summaries
find baselines -type f ! -name "*_sync.txt" ! -name "*_timeline.txt" ! -name "*.md" -delete
```

---

## Troubleshooting

### Docker Issues

#### GPU Not Accessible

**Problem:**
```text
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Solution:**
```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Permission Denied

**Problem:**
```text
docker: Got permission denied while trying to connect to the Docker daemon socket
```

**Solution:**
```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### Container Can't See Files

**Problem:**
```text
bash: ./build.sh: No such file or directory
```

**Solution:**
Check volume mount:
```bash
docker run -v $(pwd):/workspace ...
# Use $(pwd) not relative paths
```

### GitLab CI Issues

#### Pipeline Stuck

**Check:**
```bash
# GitLab Runner status
gitlab-runner status

# Runner logs
gitlab-runner --debug run
```

#### GPU Not Available in CI

**Solution:**
Tag jobs for GPU runners:
```yaml
test:
  tags:
    - gpu
    - cuda
```

Register runner with GPU:
```bash
gitlab-runner register --executor docker --docker-image nvidia/cuda:12.0.0-devel-ubuntu22.04 --docker-gpus all
```

#### Baseline Comparison Fails

**Problem:**
```text
Error: baseline not found
```

**Solution:**
Ensure baseline artifact from previous stage:
```yaml
compare:
  dependencies:
    - generate_baseline
  script:
    - ls -la baselines/  # Verify baseline exists
    - ./compare_all_baselines.sh ...
```

### Build Issues in CI

#### CMake Can't Find Kokkos

**Solution:**
```yaml
before_script:
  - export Kokkos_DIR=/path/to/kokkos/install/lib/cmake/Kokkos
```

#### Out of Memory

**Solution:**
Limit parallel builds:
```yaml
script:
  - make -j2  # Instead of -j$(nproc)
```

#### Disk Space Issues

**Solution:**
Clean up between stages:
```yaml
after_script:
  - rm -rf build/
  - docker system prune -f
```

