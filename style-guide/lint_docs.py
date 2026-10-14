#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
lint_docs.py — Style linter for KokkosPrimer documentation.

Checks Markdown files against STYLE_GUIDE.md. Exits with code 1 if any
violations are found, 0 if all files pass.

Usage:
    python style-guide/lint_docs.py [file_or_glob ...]

    # Lint all top-level .md files (run from repo root)
    python style-guide/lint_docs.py *.md

    # Lint a specific file
    python style-guide/lint_docs.py 03-data-management.md

    # Lint everything including subdirectories
    python style-guide/lint_docs.py **/*.md

    # With no arguments, lints all *.md in the repo root
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Style guide §1: deep-dive files above this line count get a warning.
FILE_LENGTH_WARN = 600

# Style guide §9: source citations must link to a full commit SHA (40 hex chars).
# A valid GitHub permalink looks like: /blob/<40-char-sha>/
PERMALINK_SHA_RE = re.compile(r"/blob/[0-9a-f]{40}/")

# Patterns that indicate a source citation (kokkos internal path reference).
SOURCE_CITATION_RE = re.compile(
    r"kokkos/(?:core|cmake|containers|algorithms|simd)/src/\S+"
)

# Style guide §10: banned callout labels.
BANNED_CALLOUTS = [
    r"\*\*Important:\*\*",
    r"\*\*Historical Note[^*]*:\*\*",
    r"\*\*Future Kokkos Improvements[^*]*:\*\*",
    r"\*\*Verification:\*\*",
]
BANNED_CALLOUT_RE = re.compile("|".join(BANNED_CALLOUTS))

# Style guide §7: italic is never used.
# Detect opening italic markers (* or _) that:
#   - Are preceded by whitespace or start-of-string (not mid-word)
#   - Are not part of ** (bold)
#   - Are not immediately followed by whitespace (which would not render as italic)
# This avoids false positives on underscores in identifiers (parallel_for)
# and asterisks in bold (**text**).
ITALIC_RE = re.compile(r"(?:(?<=\s)|(?<=^))(?<!\*)\*(?!\*)(?!\s)|(?:(?<=\s)|(?<=^))(?<!_)_(?!_)(?!\s)")

# (Bold prose emphasis is checked inline per-line; no top-level regex needed.)

# §8: Local link validity — matches [text](target) but not image links ![alt](src).
# Captures the URL portion, stopping before any whitespace or quoted title attribute.
MARKDOWN_LINK_RE = re.compile(r'(?<!!)\[[^\]]*\]\(([^)\s"\']+)')

# ---------------------------------------------------------------------------
# Violation collector
# ---------------------------------------------------------------------------

class Violation:
    def __init__(self, path: Path, line_no: int, rule: str, message: str):
        self.path = path
        self.line_no = line_no
        self.rule = rule
        self.message = message

    def __str__(self):
        return f"  {self.path}:{self.line_no}  [{self.rule}]  {self.message}"


# ---------------------------------------------------------------------------
# Link-checking helpers (§8)
# ---------------------------------------------------------------------------

def _github_anchor(heading_text: str) -> str:
    """Convert a heading string to its GitHub-style anchor fragment."""
    text = re.sub(r"`([^`]+)`", r"\1", heading_text)  # strip backticks, keep content
    text = re.sub(r"[*_]+", "", text)                  # strip bold/italic markers
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)               # keep word chars, spaces, hyphens
    return re.sub(r"\s+", "-", text.strip())


_anchor_cache: dict[Path, set[str]] = {}


def _file_anchors(path: Path) -> set[str]:
    """Return the set of valid anchor fragments for all headings in a file."""
    if path in _anchor_cache:
        return _anchor_cache[path]
    anchors: set[str] = set()
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*`{3,}", line):
            in_code = not in_code
        if in_code:
            continue
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            anchors.add(_github_anchor(m.group(1)))
    _anchor_cache[path] = anchors
    return anchors


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_file(path: Path) -> list[Violation]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    violations: list[Violation] = []

    def add(line_no: int, rule: str, message: str):
        violations.append(Violation(path, line_no, rule, message))

    in_code_block = False
    code_fence_lang = None
    prev_nonblank_line = ""
    prev_nonblank_line_no = 0
    h1_count = 0
    see_also_count = 0

    # Pre-compute anchors for same-file anchor checks (§8).
    own_anchors = _file_anchors(path)

    for i, raw_line in enumerate(lines, start=1):
        line = raw_line

        # ------------------------------------------------------------------
        # Track code fence state
        # ------------------------------------------------------------------
        # Match fences at any indentation level (e.g., inside list items).
        fence_match = re.match(r"^\s*(`{3,})(.*)", line)
        if fence_match:
            if not in_code_block:
                in_code_block = True
                lang = fence_match.group(2).strip()
                code_fence_lang = lang

                # §3: Every code block must have a language tag.
                if not lang:
                    add(i, "SG-3-TAG",
                        "Code block opened without a language tag (use `cpp`, `bash`, etc.)")

                # §3: Every code block must be introduced by a label or prose.
                # The preceding non-blank line must not be a heading, blank, or
                # another fence closing.
                intro = prev_nonblank_line.strip()
                if intro == "" or re.match(r"^#{1,6}\s", intro) or re.match(r"^`{3}", intro):
                    add(i, "SG-3-INTRO",
                        "Code block not preceded by an introductory label or sentence")
            else:
                in_code_block = False
                code_fence_lang = None

        # Skip style checks inside code blocks (except the fence line itself).
        if in_code_block and not fence_match:
            prev_nonblank_line = line
            prev_nonblank_line_no = i
            continue

        # ------------------------------------------------------------------
        # §1: Heading hierarchy
        # ------------------------------------------------------------------
        heading_match = re.match(r"^(#{1,6})\s", line)
        if heading_match:
            level = len(heading_match.group(1))

            if level == 1:
                h1_count += 1
                if h1_count > 1:
                    add(i, "SG-1-H1", "More than one H1 heading in this file")

            if level >= 4:
                add(i, "SG-1-H4", f"H{level} heading used — maximum depth is H3")


        # §1: Horizontal rule placement is a matter of editorial judgment
        # (use only before content that changes character, not between every H2)
        # and is not enforced by the linter. See STYLE_GUIDE.md §1.

        # ------------------------------------------------------------------
        # §1: File length (checked once, at end — see post-loop check)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # §2: Writing — italic and hedging
        # ------------------------------------------------------------------
        if not in_code_block:
            # Strip inline code spans and URLs before checking for italic so
            # that underscores in identifiers (e.g. `nvcc_wrapper`) and
            # asterisks in URLs don't produce false positives.
            line_stripped = re.sub(r"`[^`]+`", "", line)          # inline code
            line_stripped = re.sub(r"\[.*?\]\(.*?\)", "", line_stripped)  # links
            if ITALIC_RE.search(line_stripped):
                add(i, "SG-2-ITALIC", "Italic formatting used — use bold or inline code instead")

            hedging = re.search(
                r"\b(likely|probably|possibly|perhaps|it seems|as shown above|as mentioned)\b",
                line, re.IGNORECASE
            )
            if hedging:
                add(i, "SG-2-HEDGE",
                    f"Hedging language \"{hedging.group(0)}\" — use declarative statements")

        # ------------------------------------------------------------------
        # §7: Markdown — bold used as prose emphasis
        # ------------------------------------------------------------------
        if not in_code_block:
            # Heuristic: flag bold that appears after a lowercase word + space
            # (mid-sentence emphasis), where the bold text does not end with a
            # colon (which would make it a label).
            # This avoids false positives on definition-list bullets like
            # "- **Threads**: ..." and subsection headers like "**Best Practices:**".
            # Only flag bold text that starts with a lowercase letter — this
            # avoids false positives on bolded proper nouns and product names
            # (e.g., **NVLink-C2C**) which are legitimate term emphasis.
            for m in re.finditer(r"(?<=[a-z] )\*\*([a-z][^*:]*)\*\*", line):
                add(i, "SG-7-BOLD",
                    f"Bold used as prose emphasis: **{m.group(1)}** — "
                    f"use plain text or restructure as a label (see STYLE_GUIDE.md §7)")
                break  # one violation per line is enough

        # ------------------------------------------------------------------
        # §10: Banned callout labels
        # ------------------------------------------------------------------
        if BANNED_CALLOUT_RE.search(line):
            matched = BANNED_CALLOUT_RE.search(line).group(0)
            add(i, "SG-10-CALLOUT",
                f"Banned callout label {matched} — incorporate into prose or use "
                f"**Best Practices:** list (see STYLE_GUIDE.md §10)")

        # ------------------------------------------------------------------
        # §6 / §9: References — source citations and permalinks
        # ------------------------------------------------------------------
        if SOURCE_CITATION_RE.search(line):
            # If there's a URL in this line or the next few lines, check for SHA.
            context_window = "\n".join(lines[i - 1: min(i + 3, len(lines))])
            urls = re.findall(r"https://github\.com/kokkos/kokkos/blob/[^\s)\"]+", context_window)
            if not urls:
                add(i, "SG-9-NOLINK",
                    "Kokkos source citation has no GitHub permalink — "
                    "add a link to a specific commit SHA (see STYLE_GUIDE.md §9)")
            else:
                for url in urls:
                    if not PERMALINK_SHA_RE.search(url):
                        add(i, "SG-9-SHA",
                            f"Source citation URL does not use a full commit SHA: {url} — "
                            f"replace branch/tag with a 40-character commit SHA")

        # ------------------------------------------------------------------
        # §6: Duplicate See also: blocks
        # ------------------------------------------------------------------
        if re.match(r"^See also:", line.strip(), re.IGNORECASE):
            see_also_count += 1
            if see_also_count > 1:
                add(i, "SG-6-DUPE",
                    "Duplicate \"See also:\" block — merge all See also: references into one")

        # ------------------------------------------------------------------
        # §8: Local link validity
        # ------------------------------------------------------------------
        if not in_code_block:
            for lm in MARKDOWN_LINK_RE.finditer(line):
                target = lm.group(1).strip()
                # Skip external URLs and mailto.
                if re.match(r"https?://|mailto:", target):
                    continue
                file_part, _, anchor_part = target.partition("#")
                if not file_part:
                    # Same-file anchor only (e.g. [text](#heading)).
                    if anchor_part and anchor_part not in own_anchors:
                        add(i, "SG-8-ANCHOR",
                            f"Anchor #{anchor_part} not found in this file")
                else:
                    target_path = (path.parent / file_part).resolve()
                    if not target_path.exists():
                        add(i, "SG-8-LINK",
                            f"Broken link — target does not exist: {file_part}")
                    elif anchor_part:
                        if anchor_part not in _file_anchors(target_path):
                            add(i, "SG-8-ANCHOR",
                                f"Anchor #{anchor_part} not found in {file_part}")

        # ------------------------------------------------------------------
        # Update rolling context
        # ------------------------------------------------------------------
        if line.strip():
            prev_nonblank_line = line
            prev_nonblank_line_no = i

    # ------------------------------------------------------------------
    # Post-loop checks
    # ------------------------------------------------------------------

    # §1: File length warning.
    if len(lines) > FILE_LENGTH_WARN:
        violations.append(Violation(
            path, len(lines), "SG-1-LEN",
            f"File is {len(lines)} lines — style guide recommends ~200–600 for deep-dive "
            f"chapters. Consider splitting into multiple chapters."
        ))

    # §1: Exactly one H1.
    if h1_count == 0:
        violations.append(Violation(path, 1, "SG-1-H1", "No H1 heading found — every file must have exactly one"))


    return violations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parent.parent

    if len(sys.argv) < 2:
        paths = sorted(repo_root.glob("*.md"))
    else:
        paths = []
        for arg in sys.argv[1:]:
            p = Path(arg)
            if "*" in arg:
                expanded = sorted(repo_root.glob(arg))
            elif p.is_absolute():
                expanded = [p]
            else:
                expanded = [repo_root / p]
            paths.extend(expanded)

    EXCLUDED = {"STYLE_GUIDE.md", "CHAPTER_TEMPLATE.md", "03-EDITORIAL-NOTES.md"}
    paths = [p for p in paths if p.name not in EXCLUDED and p.suffix == ".md"]

    if not paths:
        print("lint_docs.py: no files to lint.")
        sys.exit(0)

    all_violations: list[Violation] = []
    for path in paths:
        if not path.exists():
            print(f"lint_docs.py: file not found: {path}", file=sys.stderr)
            continue
        file_violations = check_file(path)
        all_violations.extend(file_violations)

    if not all_violations:
        print(f"lint_docs.py: {len(paths)} file(s) checked — no violations found.")
        sys.exit(0)

    # Group violations by file for readable output.
    by_file: dict[Path, list[Violation]] = {}
    for v in all_violations:
        by_file.setdefault(v.path, []).append(v)

    total = 0
    for file_path, file_violations in sorted(by_file.items()):
        print(f"\n{file_path}  ({len(file_violations)} violation(s))")
        for v in sorted(file_violations, key=lambda x: x.line_no):
            print(f"  line {v.line_no:4d}  [{v.rule}]  {v.message}")
            total += 1

    print(f"\n{total} violation(s) across {len(by_file)} file(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
