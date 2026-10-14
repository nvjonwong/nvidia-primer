<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Maintenance Guide

When content in this repository changes, two index files must be kept in sync.

## Index Files

| File | Tracks | Granularity |
|------|--------|-------------|
| `NvidiaKokkosPrimer.md` | All 10 chapters | H1 title as linked heading + H2 headings as bullet points |
| `DOCUMENTATION_INDEX.md` | All chapters, guides, and scripts | Single-sentence description per file |

## When to Update

| Change | NvidiaKokkosPrimer.md | DOCUMENTATION_INDEX.md |
|--------|----------------------|------------------------|
| Add a chapter file | Add numbered item with H2 bullets | Add entry under `## Chapters` |
| Rename a chapter file | Update link path and title text | Update filename and description |
| Add or rename an H2 section | Update bullet points | Update description if scope changes |
| Add a guide in `test_suite/` | No change | Add entry under `## Guides (in test_suite/)` |
| Add a script in `scripts/` | No change | Add entry under `## Build Scripts (in scripts/)` |
| Delete a chapter | Remove the numbered item | Remove the entry |
| Minor edits (examples, typos, code) | No change | No change |

## Granularity Rules

**`NvidiaKokkosPrimer.md`**: List H2 section headings as bullet points. H3 sub-sections may be mentioned parenthetically only when they represent a distinct major concept. Do not enumerate every H3.

**`DOCUMENTATION_INDEX.md`**: One sentence per file. Describe the scope and key topics — do not list heading names verbatim. Entries use linked headings: `### [`filename`](relative/path)` where the path is relative to the repo root. Script entries (`scripts/*.sh`) are plain text headings without links.

## Style Linter

Run the linter on any chapter file you modify before finishing:

```bash
python style-guide/lint_docs.py <modified-file.md>
```

The linter enforces `STYLE_GUIDE.md` rules: heading hierarchy, no italic, no hedging language, code block formatting, source citation permalinks, and local link validity. Exit code 0 means no violations.

**`SG-8-LINK`**: A local file link target does not exist on disk. Fix the path or remove the link.

**`SG-8-ANCHOR`**: A heading anchor fragment does not exist in the target file. Check that the heading text matches exactly — anchors are generated using GitHub's normalization: lowercase, spaces to hyphens, punctuation stripped.

**`SG-1-LEN` (file length)**: This warning is advisory. A modest overage above 600 lines is acceptable if the additional content is necessary for completeness and clarity of presentation. Large regressions — or length increases that could be addressed by splitting a chapter — should still be resolved.
