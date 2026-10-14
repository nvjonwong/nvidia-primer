<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Baseline Profiles Directory

This directory stores baseline profiles for regression testing across Kokkos versions.

## Structure

Each subdirectory represents a Kokkos version:

```
baselines/
├── kokkos_5.0.1/
│   ├── metadata.txt                  # Version info, GPU, date
│   ├── *_timeline.txt                # Simplified, diffable timelines
│   ├── *_sync.txt                    # Synchronization statistics
│   └── *.nsys-rep                    # Full profiles (not in Git)
└── kokkos_5.1.0/
    └── ...
```

## Usage

```bash
# Generate baseline
cd test_suite
./create_all_baselines.sh 5.0.1

# Compare baselines
./compare_all_baselines.sh 5.0.1 5.1.0
```

## Version Control

**Commit to Git:** `metadata.txt`, `*_timeline.txt`, `*_sync.txt`

**Exclude from Git:** `*.nsys-rep`, `*.sqlite` (large, regenerable)
