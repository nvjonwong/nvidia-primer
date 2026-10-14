<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA Kokkos Primer

The NVIDIA Kokkos Primer is an educational repository for writing performant
Kokkos code on NVIDIA GPUs. The project includes documentation, tutorial-style
examples, test and validation code, build and configuration files, and a small
reusable header-only Kokkos extension library.

The project only uses public APIs for Kokkos and CUDA libraries.

Start with [NvidiaKokkosPrimer.md](NvidiaKokkosPrimer.md). The complete file
index is in [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md).

## Scope

The project provides:

- Introductory documentation for Kokkos concepts as they apply to NVIDIA GPU
  programming
- Tutorial-style C++ and Kokkos examples
- Test and validation code for examples
- Best practices, common pitfalls, and suggested workarounds for Kokkos on
  NVIDIA hardware
- A small reusable Kokkos extension library, currently header-only

The header-only extension library is intended to support missing or emerging
Kokkos functionality used by the primer and examples. Over time, portions of
this extension layer may shrink or be removed as equivalent functionality is
added directly to Kokkos.

## License and Notices

This project is released under the Apache License, Version 2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).

Third-party dependency and attribution information is documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contributions

This project accepts external contributions under the Developer Certificate of
Origin (DCO). Expected contributions include documentation updates for Kokkos
behavior, tutorial improvements, test improvements, and bug fixes or
enhancements to the Kokkos extension library.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution policy.
