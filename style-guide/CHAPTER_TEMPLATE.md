<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# [Chapter Title]

[One to two sentence overview of this chapter's scope. State what the reader will learn and
why it matters. Do not repeat content from a prior chapter — refer to it instead.]

---

## [First Major Topic]

[One to two sentence definition of this topic. Answer "what is this?" before anything else.]

[Mechanism paragraph: explain how it works, or how it is used in Kokkos. Keep to 1–3 sentences.
If more detail is needed, switch to a list.]

- [Key point]
- [Key point]
- [Key point]

See: link Concept Name or Kokkos Docs to the appropriate canonical URL.

### [Subtopic A]

[Definition sentence for this subtopic.]

[Mechanism or availability in Kokkos.]

- Pros:
  - [Benefit]
  - [Benefit]
- Cons:
  - [Limitation]
  - [Limitation]
- Usage in Kokkos:

```cpp
// Brief, compilable example
Kokkos::View<double*, Kokkos::CudaSpace> example("example", N);
```

**When to Use:**
- [Condition under which this is the right choice]
- [Condition under which this is the wrong choice]

### [Subtopic B — Missing Feature or Workaround]

[Name of the missing CUDA feature or unsupported pattern.]

- **Not directly exposed in Kokkos** — [brief reason or alternative]
- **Workarounds:**
  - [Workaround option 1]
  - [Workaround option 2]

**Example:**
```cpp
// Show the workaround in use
```

---

## [Second Major Topic: CUDA-to-Kokkos Comparison]

[Definition or context sentence.]

**CUDA Pattern:**
```cpp
// Canonical CUDA way of doing this
```

**Kokkos Equivalent:**
```cpp
// Kokkos way of doing this
```

[One to two sentence explanation of the key difference or trade-off between the two.]

**Best Practices:**
- [Actionable guidance]
- [Actionable guidance]

---

## [Third Major Topic: Performance Consideration]

[Definition of the performance concept being discussed.]

[Explanation of why this matters on GPU hardware — connect to thread model or memory
hierarchy where relevant.]

**Anti-Pattern:**
```cpp
// Code that exhibits the problem
```

**Optimized Pattern:**
```cpp
// Code that avoids the problem
```

[Explanation of why the optimized pattern is better. Reference hardware behavior
(e.g., coalescing, register pressure, occupancy) where applicable.]

**Performance Impact:**
- [What improves and why]
- [Any trade-offs or caveats]

---

## [Optional: Summary / Quick Reference]

Use a summary table only if this chapter covers three or more options that readers will
need to choose between in practice.

| Option | When to Use | Synchronization | Notes |
|--------|-------------|----------------|-------|
| [Option A] | [Condition] | [Yes/No] | [Note] |
| [Option B] | [Condition] | [Yes/No] | [Note] |

---

See also:
- Kokkos Docs — relevant page: replace with the appropriate canonical URL
- CUDA Programming Guide — relevant section: replace with the appropriate canonical URL
