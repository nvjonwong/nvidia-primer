<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Notices

This repository does not vendor third-party source code or third-party binaries
in the planned source distribution. The local `kokkos/` checkout, CMake build
directories, downloaded dependency trees, and downloaded packages are local
developer artifacts and are excluded from the project distribution.

The items below are external dependencies or referenced tools used to build,
test, profile, or document the project.

## Kokkos

- Project: Kokkos
- Use: Required external dependency for examples, helper code, and tests.
- Distribution status: Not vendored in this repository distribution.
- License: Apache-2.0 WITH LLVM-exception
- Attribution: Copyright Contributors to the Kokkos Project.
- Source: <https://github.com/kokkos/kokkos>

## GoogleTest

- Project: GoogleTest
- Use: Test dependency fetched by CMake through CPM.cmake.
- Distribution status: Not vendored in this repository distribution.
- Version referenced by build files: v1.14.0
- License: BSD-3-Clause
- Attribution: Copyright 2008, Google Inc.
- Source: <https://github.com/google/googletest>

## CPM.cmake

- Project: CPM.cmake
- Use: CMake dependency manager downloaded at configure time by the test suite.
- Distribution status: Not vendored in this repository distribution.
- Version referenced by build files: 0.38.1
- License: MIT
- Attribution: Copyright (c) 2019-2022 Lars Melchior and contributors.
- Source: <https://github.com/cpm-cmake/CPM.cmake>

## Optional Tools Referenced in Documentation

The documentation references optional external profiling and analysis tools.
These tools are not vendored or redistributed by this repository.

- Kokkos Tools: Apache-2.0 WITH LLVM-exception,
  Copyright Contributors to the Kokkos Project,
  <https://github.com/kokkos/kokkos-tools>
- Caliper: BSD-3-Clause, Copyright (c) 2015-2024, Lawrence Livermore
  National Security, LLC, <https://github.com/LLNL/Caliper>

## Developer Certificate of Origin

- Project: Developer Certificate of Origin
- Use: Contribution certification text in `CONTRIBUTING.md`.
- Copyright: Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
- Copying terms: Everyone is permitted to copy and distribute verbatim copies of
  this license document, but changing it is not allowed.

## License Texts

### Apache-2.0 WITH LLVM-exception

The Apache License, Version 2.0 text is distributed with this project in
`LICENSE`.

Kokkos and Kokkos Tools add the following LLVM exception:

```text
As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
```

### BSD-3-Clause

The BSD-3-Clause license terms are:

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### MIT

The MIT license terms are:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
