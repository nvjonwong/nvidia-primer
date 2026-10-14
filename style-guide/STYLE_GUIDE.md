<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kokkos Primer Style Guide

This guide defines the writing, formatting, and content standards for all documentation in this repository. It is derived from `01-introduction.md` and `02-gpu-architecture.md`, which serve as the canonical reference examples of well-written content.

New contributions should be checked against this guide before merging.

---

## Table of Contents

1. [Document Structure](#1-document-structure)
2. [Writing Voice and Tone](#2-writing-voice-and-tone)
3. [Code Blocks](#3-code-blocks)
4. [Information Architecture within Sections](#4-information-architecture-within-sections)
5. [CUDA-to-Kokkos Comparison Patterns](#5-cuda-to-kokkos-comparison-patterns)
6. [External References and Citations](#6-external-references-and-citations)
7. [Markdown Formatting Conventions](#7-markdown-formatting-conventions)
8. [Scope and Depth Calibration](#8-scope-and-depth-calibration)
9. [Source Code Citations](#9-source-code-citations)
10. [Guidance and Warning Conventions](#10-guidance-and-warning-conventions)
11. [Contributor Checklist](#11-contributor-checklist)

---

## 1. Document Structure

### Heading Hierarchy

Use at most three heading levels. Do not use H4 or deeper.

| Level | Usage | Example |
|-------|-------|---------|
| H1 (`#`) | Document title only — one per file | `# Introduction to Kokkos` |
| H2 (`##`) | Major conceptual sections | `## Memory Hierarchy` |
| H3 (`###`) | Detailed subsections under an H2 | `### Managed Memory and Pinned Memory` |

For short lists of sub-topics within an H2 section that do not warrant a full H3, use bold subsection labels instead:

```markdown
## How Kokkos Interacts with CUDA

**Kernel Definition**
- Kokkos kernels are C++ lambdas with `KOKKOS_LAMBDA` macro
  - Translates to: `[=] __host__ __device__`

**Execution Model**
- `ExecutionSpace` abstracts CUDA streams and device management
```

### Section Flow

Sections within a file should follow a progression from broad to specific:

- Introduction files: Definition → Benefits → CUDA relationship → Practical use
- Technical deep-dive files: Conceptual model → Technical details → Practical considerations → Performance

Use a horizontal rule (`---`) only when content changes character relative to the surrounding material — for example, before a synthesis block ("Typical Workflow", "Summary", "Best Practices" wrap-up) that follows a sequence of detailed technical sections, or before an architecturally distinct appendix-like section at the end of a long file.

Do not use `---` as a default separator between regular H2 sections. In particular, short files (under ~200 lines) rarely need any `---` at all. The canonical example is `02-gpu-architecture.md`, which uses `---` exactly once: before the "Typical Workflow" block that synthesizes the entire memory hierarchy discussion above it.

This rule is a matter of editorial judgment and is not enforced by the automated linter. It belongs to the manual review checklist.

### File Length

There is no strict line limit, but use file length as a signal of scope creep. As a reference point:

- A focused overview chapter: ~80–150 lines
- A technical deep-dive chapter: ~200–600 lines
- A file significantly longer than ~600 lines is a candidate for splitting into multiple chapters

When a file grows long, ask whether any sections are self-contained enough to become their own numbered file in the series.

---

## 2. Writing Voice and Tone

### General Principles

- Write in direct, declarative sentences. State facts; avoid hedging unless uncertainty is genuinely relevant.
- Use active voice: "Kokkos provides...", not "A unified programming model is provided by Kokkos."
- Be precise but accessible. Define CUDA/GPU terms before using them, especially when bridging to Kokkos equivalents.
- Do not use italics for emphasis. Use bold or inline code instead (see [Section 7](#7-markdown-formatting-conventions)).

### Sentence and Paragraph Length

- Use short sentences (one clause) for definitions and rules.
- Use longer sentences for explanations that require connecting two related ideas.
- Keep paragraphs to 1–3 sentences. When a topic requires more detail, switch to a bullet list.

**Example of the pattern:**

> Kokkos is an open-source C++ performance portability library. It was started by Sandia National Laboratories and is now managed by the High Performance Software Foundation, a multi-institutional consortium under the Linux Foundation.

One short definition sentence, then one elaborating sentence. The reader can stop after the first sentence and understand the key fact.

### Bridging CUDA Knowledge to Kokkos

This repository targets a mixed audience. Assume the reader has CUDA experience and is learning Kokkos. When introducing a Kokkos concept that maps to an existing CUDA concept:

1. Name the Kokkos concept.
2. Immediately map it to its CUDA equivalent using `→` notation or a parenthetical translation.

**Examples:**
- `` `TeamPolicy` → thread blocks, `ThreadVectorRange` → warp-level operations ``
- `` `KOKKOS_LAMBDA` macro — Translates to: `[=] __host__ __device__` ``
- `` `ExecutionSpace` abstracts CUDA streams and device management ``

Do not assume the reader knows Kokkos-specific vocabulary without an introduction. Do not over-explain CUDA fundamentals that the reader already knows.

---

## 3. Code Blocks

### Language Tags

Always specify a language tag on every fenced code block. Never leave a code block untagged.

| Content | Tag |
|---------|-----|
| C++/Kokkos code | ` ```cpp ` |
| Shell commands, environment variables | ` ```bash ` |
| CMake configuration | ` ```bash ` or ` ```cmake ` |

### Introducing Code Blocks

Every code block must be introduced by a line of prose or a bold label that sets context. Do not place a code block immediately after a heading with no intervening text.

Accepted introduction patterns:

```markdown
- **Kokkos Example**:
  ```cpp
  ...
  ```

- Usage in Kokkos:
  ```cpp
  ...
  ```

- Example:
  ```cpp
  ...
  ```
```

For standalone examples (not inside a list), use a bold label or a short prose sentence:

```markdown
**Enable in CMake:**
```bash
cmake .. -DKokkos_ENABLE_CUDA=ON
```
```

### Inline Comments in Code

Keep inline comments minimal. A code block should be self-documenting through clear variable and function names. Use comments only to:

- Identify Kokkos-specific behavior that is not obvious from the API name
- Mark synchronization points explicitly
- Distinguish between memory spaces when multiple appear in one snippet

**Good comment use:**
```cpp
Kokkos::View<float*, Kokkos::CudaSpace> a("a", N);        // Regular device memory
Kokkos::View<float*, Kokkos::CudaUVMSpace> b("b", N);     // Managed (Unified) memory
```

**Avoid comments that merely restate the code:**
```cpp
// Create a view   <- unnecessary
Kokkos::View<double*> data("data", N);
```

### Code Completeness

Prefer complete, compilable snippets. Avoid pseudocode. If a snippet must omit boilerplate for brevity, use a comment placeholder like `// ... other setup ...` rather than leaving an unexplained gap.

---

## 4. Information Architecture within Sections

Each section or subsection should follow this progression:

1. **Definition** — What is this thing? One to two sentences.
2. **Mechanism / Availability** — How does it work? How is it used in Kokkos?
3. **Implications** — When should you use it? Trade-offs?
4. **Example** — Code or concrete illustration.
5. **Limitations / Workarounds** — Where does Kokkos fall short? What is the workaround?

Not every section requires all five parts. Omit steps that are not relevant, but do not skip step 1 (definition).

### List Type Selection

| Situation | List Type |
|-----------|-----------|
| Parallel, unordered items (feature list, benefit list) | Bullet list (`-`) |
| Sequential steps or ordered hierarchy (numbered concepts) | Numbered list (`1.`) |
| Trade-offs with clear opposing sides | `Pros:` / `Cons:` sub-lists |
| Missing Kokkos feature with alternatives | `**Workarounds**:` with bullet sub-list |

### Pros/Cons Pattern

Use this pattern for any feature with meaningful trade-offs:

```markdown
**Managed (Unified) Memory**
- Available via `cudaMallocManaged` or Kokkos' `Kokkos::CudaUVMSpace`
- Pros:
  - Simplifies code by eliminating explicit device↔host copies
  - Enables oversubscription
- Cons:
  - Performance overhead from implicit page migration
  - Lack of control over migration can introduce latency spikes
- Usage in Kokkos:
  ```cpp
  Kokkos::View<float*, Kokkos::CudaUVMSpace> b("b", N);
  ```
```

### Missing Feature / Workaround Pattern

When a CUDA feature has no direct Kokkos equivalent:

```markdown
- **Constant Memory**: Read-only, broadcast-optimized memory (64KB limit)
  - **Not directly exposed in Kokkos** — no dedicated constant memory space
  - **Workarounds**:
    - Small read-only data passed as kernel arguments may use constant cache automatically
    - Use `KOKKOS_INLINE_FUNCTION` with constexpr data
```

---

## 5. CUDA-to-Kokkos Comparison Patterns

### Arrow Notation for Direct Mappings

Use `→` inline to map a Kokkos concept to its CUDA equivalent. This is the lightest-weight comparison form and is preferred when the mapping is 1:1.

```markdown
- `TeamPolicy` → thread blocks
- `ThreadVectorRange` → warp-level operations
- `ExecutionSpace` → CUDA streams and device management
```

### Prose-Level Comparisons

For comparisons that require more explanation than a single arrow, describe the CUDA concept first, then explain the Kokkos abstraction:

```markdown
**Kernel Definition**
- Kokkos kernels are C++ lambdas with `KOKKOS_LAMBDA` macro
  - Translates to: `[=] __host__ __device__`
  - Extended lambdas with some caveats (must capture by value for GPU)
- CUDA API calls hidden behind Kokkos abstraction layer
```

### Side-by-Side Code Comparisons

Use side-by-side code blocks when the syntax difference between CUDA and Kokkos is itself the point being illustrated. Label each block clearly:

```markdown
**CUDA Pattern:**
```cpp
cudaMalloc(&d_data, N * sizeof(float));
```

**Kokkos Equivalent:**
```cpp
Kokkos::View<float*> data("data", N);
```
```

Do not use side-by-side code comparisons for conceptual explanations that can be conveyed in prose.

### Comparison Tables

Use a table when comparing multiple options across multiple dimensions (e.g., system types vs. performance characteristics). Do not use a table when a bullet list is sufficient.

```markdown
| System Type         | UVM Performance          | When to Use UVM     |
|---------------------|--------------------------|---------------------|
| Discrete GPU (PCIe) | 5-50% overhead typical   | Prototyping only    |
| Grace Hopper        | <5-10% overhead typical  | Production viable   |
```

---

## 6. External References and Citations

### Reference Style by Placement

Use the appropriate reference style based on where in the text the reference appears:

| Style | When to Use | Example |
|-------|-------------|---------|
| Inline `([Concept](url))` | Immediately after the term being defined | `([Programming Model](https://docs.nvidia.com/...))` |
| `See: [Text](url)` | Single reference at the end of a definition or subsection | `See: [Thread Hierarchy](https://...)` |
| `See also:` with bullet list | Multiple references at the end of a section | See example below |
| `References:` block | Formal citation block at end of a major H2 section | See example below |

**"See also:" pattern:**
```markdown
See also:
- [Kokkos Memory Spaces](https://kokkos.github.io/kokkos-core-wiki/API/core/memory_spaces/)
- [CUDA Managed Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-unified-memory-programming-hd)
- [CUDA Pinned Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory)
```

**"References:" pattern:**
```markdown
References:
- [CUDA Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
- [Kokkos Unified Memory](https://kokkos.github.io/kokkos-core-wiki/API/core/memory_spaces/#cudauvmsapce)
```

### Approved Reference Sources

Prefer links to the following canonical sources:

- **NVIDIA CUDA documentation**: `https://docs.nvidia.com/cuda/cuda-c-programming-guide/`
- **NVIDIA CUDA Best Practices Guide**: `https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/`
- **Kokkos Core Wiki**: `https://kokkos.github.io/kokkos-core-wiki/`
- **NVIDIA Developer Blog**: `https://developer.nvidia.com/blog/`

Avoid linking to third-party tutorials or Stack Overflow. Avoid linking to specific version-tagged documentation unless the version matters (e.g., a CUDA 12.4-specific feature).

---

## 7. Markdown Formatting Conventions

### Emphasis

| Element | Use for |
|---------|---------|
| **Bold** (`**text**`) | Section labels, important concept names, key terms being defined |
| `Inline code` (backticks) | API names, macros, types, function names, flags, file paths |
| *Italic* | Not used in this repository |

Use bold for the label at the start of a bullet item when that item introduces a named concept:

```markdown
- **Registers**: Fastest, per-thread private memory
- **Shared Memory**: Fast on-chip memory shared within a thread block
```

Use inline code for any Kokkos or CUDA API identifier, even in prose:

```markdown
The `Kokkos::View` abstraction supports multiple memory spaces.
Enable with `-DKokkos_ENABLE_CUDA_UVM=ON` during configuration.
```

### Horizontal Rules

Use `---` to separate major conceptual boundaries within a file — for example, between a detailed memory-type subsection and a "Typical Workflow" summary. Do not use `---` between every section; reserve it for genuine topic shifts.

### Tables

Use tables for:
- Multi-dimensional comparisons (rows = options, columns = attributes)
- Summary grids at the end of a section (e.g., "When to use X vs Y")

Do not use tables for:
- Simple lists of items with one attribute (use bullets instead)
- Content where the rows have very different lengths (use a definition-style bullet list instead)

### Numbered Bold Subsections

For a list of 3–6 parallel concepts under an H2, use numbered bold labels rather than H3 headings:

```markdown
## Key Benefits of Using Kokkos on NVIDIA GPUs

**1. Performance Portability**
- Single codebase runs efficiently across NVIDIA GPUs and other platforms

**2. Simplified Parallel Programming**
- High-level abstractions (`parallel_for`, `parallel_reduce`, `parallel_scan`)
```

Use H3 when a subsection is long enough to warrant its own internal structure (multiple paragraphs and/or code blocks).

---

## 8. Scope and Depth Calibration

### Introduction Files (01-style)

Introduction files answer "what is this?" and "why does it matter?". They are high-level surveys, not implementation guides.

**Include:**
- Definition of the technology and its purpose
- Key benefits, summarized concisely
- How it relates to CUDA (at the conceptual level)
- Brief overview of major capabilities
- References to where to learn more

**Exclude:**
- Deep implementation details, workarounds, or source-code references
- Advanced or architecture-specific topics (e.g., Grace Hopper specifics)
- Performance benchmarks or optimization techniques
- Error-handling patterns

### Technical Deep-Dive Files (02-style)

Technical files answer "how does it work?" and "when should I use it?". They are the primary reference material.

**Include:**
- Full conceptual model with mechanism explained
- Code examples for both basic and advanced usage
- Pros/Cons trade-offs
- Workarounds for missing Kokkos features
- Architecture-specific content (clearly labeled with H3)
- Performance guidance and best practices
- External references to source code or documentation

**Exclude:**
- Repetition of introductory material from earlier files
- Speculation without basis in documentation or profiling results
- Content that belongs in a subsequent, more advanced file

### Signals That a Topic Should Be Deferred

Use the following signals to indicate that a topic is intentionally deferred:

- A brief mention followed by a "See:" reference to the relevant later section or file
- A `**Note**:` that names the more advanced topic without explaining it
- A cross-reference within the repository (e.g., `See \`03-optimizing-kokkos.md\``)

Do not start explaining an advanced topic and then abandon it mid-explanation. Either cover it fully in the appropriate file or defer it with a reference.

---

## 9. Source Code Citations

When referencing Kokkos or CUDA source code to substantiate a claim about internal behavior, citations are allowed and encouraged in technical deep-dive files. They must follow these rules:

### When to Cite Source

Cite source code when:
- Explaining non-obvious internal behavior (e.g., when Kokkos calls `cudaDeviceSynchronize`)
- Documenting a known limitation or workaround tied to a specific implementation detail
- Showing simplified code extracted from the library to illustrate a mechanism

Do not cite source code in introduction or overview files.

### Citation Format

Use a `Source:` label with an inline code path and line reference:

```markdown
Source: [`core/src/Cuda/Kokkos_CudaSpace.cpp` L211–216](https://github.com/kokkos/kokkos/blob/297e182502896611ccc699b4935d1bcc703293c9/core/src/Cuda/Kokkos_CudaSpace.cpp#L211-L216)
```

Or for a simplified excerpt shown directly:

```markdown
// From kokkos/core/src/Cuda/Kokkos_Cuda_KernelLaunch.hpp:686-690 (simplified)
```cpp
#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
  cuda_instance->fence("...");
#endif
```
```

### Permalink Requirement

All source citations **must** use a permalink tied to a specific git commit, not a branch or tag. This prevents citations from silently becoming incorrect as the Kokkos codebase evolves.

Correct permalink format (GitHub):
```text
https://github.com/kokkos/kokkos/blob/297e182502896611ccc699b4935d1bcc703293c9/core/src/Cuda/Kokkos_CudaSpace.cpp#L211-L216
```

Do not link to `main`, `develop`, or a version tag like `4.5.0`. Use the full commit SHA.

When adding a new source citation, record the Kokkos version and commit SHA in a comment near the citation so future maintainers can verify whether the behavior has changed:

```markdown
Source: [`core/src/Cuda/Kokkos_CudaSpace.cpp` L211–216](https://github.com/kokkos/kokkos/blob/297e182502896611ccc699b4935d1bcc703293c9/core/src/Cuda/Kokkos_CudaSpace.cpp#L211-L216)
(Verified against Kokkos 5.0.1, commit `297e182502896611ccc699b4935d1bcc703293c9`)
```

---

## 10. Guidance and Warning Conventions

Avoid ad-hoc callout labels such as `**Important:**`, `**Historical Note:**`, `**Future Kokkos Improvements:**`, or `**Verification:**`. These create inconsistent formatting and are difficult to maintain at scale.

### Preferred Alternatives

**For critical caveats or warnings**: Incorporate the information as the first or last sentence of the relevant paragraph, or use a dedicated bullet point under a `**Best Practices:**` list.

Instead of:
```markdown
**Important:** Do not use `Atomic` Views unless every access truly needs to be atomic.
```

Write:
```markdown
**Best Practices:**
- Avoid marking entire Views with `Kokkos::Atomic` — it makes every access atomic.
  Use explicit `Kokkos::atomic_add()` instead.
```

**For version-specific behavioral changes**: Include the version information inline as part of a prose sentence or a parenthetical:

Instead of:
```markdown
**Historical Note:** In older versions of Kokkos (pre-4.x), launching kernels without passing
an execution space caused implicit fences. This has been fixed in Kokkos 4.x+.
```

Write:
```markdown
In Kokkos versions prior to 4.x, launching kernels without an explicit execution space caused
an implicit fence after each kernel. This was fixed in Kokkos 4.x+.
```

**For future improvements**: Omit speculative content about possible future Kokkos changes unless you have an upstream issue or RFC to cite. If citing an open issue, use a standard "See also:" reference.

**For test/verification references**: Use a plain "See also:" reference to the relevant test file.

---

## 11. Contributor Checklist

Before submitting a new or revised document, verify the following:

**Structure**
- [ ] Single H1 title at the top of the file
- [ ] No heading deeper than H3
- [ ] Sections flow from broad to specific
- [ ] File length is appropriate to its scope (see [Section 1](#1-document-structure))

**Writing**
- [ ] Declarative sentences; active voice throughout
- [ ] No hedging language (e.g., "likely", "probably", "as shown above")
- [ ] No italics used for emphasis
- [ ] CUDA concepts introduced before Kokkos mappings
- [ ] Paragraphs are 1–3 sentences; longer content uses lists

**Code Blocks**
- [ ] Every code block has a language tag (`cpp`, `bash`, etc.)
- [ ] Every code block is introduced by a label or prose sentence
- [ ] Inline comments explain behavior, not narrate the code
- [ ] Snippets are complete and compilable (or use `// ...` placeholders explicitly)

**Content**
- [ ] Each section follows Definition → Mechanism → Implications → Example → Limitations
- [ ] Pros/Cons lists used for features with genuine trade-offs
- [ ] `**Workarounds**:` pattern used for missing Kokkos features
- [ ] Tables used only for multi-dimensional comparisons

**References**
- [ ] Inline `([Concept](url))` for immediate context citations
- [ ] `See:` for 1–2 end-of-section references
- [ ] `See also:` for 3+ end-of-section references
- [ ] All links point to approved canonical sources (see [Section 6](#6-external-references-and-citations))
- [ ] No duplicate `See also:` or `References:` blocks at end of file

**Source Code Citations** (deep-dive files only; not applicable to overview files)
- [ ] Source citations use the `Source:` label format defined in [Section 9](#9-source-code-citations)
- [ ] Every source citation links to a GitHub permalink at a specific commit SHA, not a branch or tag
- [ ] Each cited file and line range has been manually verified against the linked commit
- [ ] A Kokkos version and commit SHA annotation is present near each citation

**Guidance and Warning Conventions**
- [ ] No `**Important:**` labels — critical information incorporated into prose or `**Best Practices:**` lists
- [ ] No `**Historical Note:**` labels — version-specific information written as inline prose
- [ ] No `**Future Kokkos Improvements:**` labels — speculative content removed or replaced with a link to an upstream issue
- [ ] No `**Verification:**` labels — test file references use `See also:` instead

**Markdown**
- [ ] Bold used for section labels and key terms only; not for emphasis within prose sentences
- [ ] Inline code used for all API names, macros, types, flags, and file paths
- [ ] `---` horizontal rules used only before content that changes character (synthesis blocks, appendix-like sections) — not between regular H2 sections
- [ ] No H4+ headings used
