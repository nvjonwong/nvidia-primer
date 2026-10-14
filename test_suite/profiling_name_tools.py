#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Nsight reports/recipes and build NCU maps from NSYS reports."""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


class ProfilingNameError(RuntimeError):
    """Raised when a trace cannot produce an unambiguous rename map."""


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        row[1]
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _require_columns(
    connection: sqlite3.Connection, table: str, required: Iterable[str]
) -> set[str]:
    columns = _table_columns(connection, table)
    if not columns:
        raise ProfilingNameError(f"Required SQLite table is missing: {table}")
    missing = set(required) - columns
    if missing:
        raise ProfilingNameError(
            f"{table} is missing required columns: {', '.join(sorted(missing))}"
        )
    return columns


def _resolve_string(value: object, strings: dict[int, str]) -> str:
    if isinstance(value, int):
        return strings.get(value, str(value))
    return "" if value is None else str(value)


def kernel_nvtx_sqlite_mapping(
    sqlite_path: Path, label_prefix: str = ""
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Stream innermost NVTX labels for CUDA launches from an NSYS SQLite file.

    CUDA runtime calls on one host thread are chronological. Reading launches
    and NVTX ranges in that order permits a sweep that retains only active
    ranges, avoiding the former launch-times-all-ranges quadratic scan.
    """

    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        _require_columns(connection, "StringIds", {"id", "value"})
        kernel_columns = _require_columns(
            connection, "CUPTI_ACTIVITY_KIND_KERNEL", {"correlationId"}
        )
        if (
            "demangledName" not in kernel_columns
            and "shortName" not in kernel_columns
        ):
            raise ProfilingNameError(
                "CUPTI_ACTIVITY_KIND_KERNEL has neither demangledName nor shortName"
            )
        _require_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            {"correlationId", "start", "end", "globalTid"},
        )
        nvtx_columns = _require_columns(
            connection, "NVTX_EVENTS", {"start", "end", "globalTid"}
        )
        if "text" not in nvtx_columns and "textId" not in nvtx_columns:
            raise ProfilingNameError("NVTX_EVENTS has neither text nor textId")

        # This persistent index lets each per-thread cursor stream in timestamp
        # order instead of making SQLite sort a potentially multi-million-row
        # NVTX table for every map generation.
        connection.execute(
            "CREATE INDEX IF NOT EXISTS "
            "KokkosPrimer_NVTX_EVENTS_tid_start_end "
            "ON NVTX_EVENTS(globalTid, start, end)"
        )
        connection.commit()

        strings = {
            int(row["id"]): str(row["value"])
            for row in connection.execute("SELECT id, value FROM StringIds")
        }
        original_column = (
            "demangledName" if "demangledName" in kernel_columns else "shortName"
        )
        nvtx_select = ["start", "end"]
        if "text" in nvtx_columns:
            nvtx_select.append("text")
        if "textId" in nvtx_columns:
            nvtx_select.append("textId")

        launches = connection.execute(
            f"SELECT r.globalTid, r.start, r.end, k.{original_column} AS original "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL AS k "
            "JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r "
            "ON k.correlationId = r.correlationId "
            "WHERE r.start IS NOT NULL AND r.end IS NOT NULL "
            "ORDER BY r.globalTid, r.start, r.end"
        )
        aliases_by_original: dict[str, set[str]] = defaultdict(set)
        launch = next(launches, None)
        while launch is not None:
            tid = int(launch["globalTid"])
            ranges = connection.execute(
                f"SELECT {', '.join(nvtx_select)} FROM NVTX_EVENTS "
                "WHERE globalTid = ? AND end IS NOT NULL AND end > start "
                "ORDER BY start, end",
                (tid,),
            )
            next_range = next(ranges, None)
            active: set[int] = set()
            starts: list[tuple[int, int, int, int, str]] = []
            ends: list[tuple[int, int]] = []
            range_id = 0
            previous_end: int | None = None

            while launch is not None and int(launch["globalTid"]) == tid:
                launch_start = int(launch["start"])
                launch_end = int(launch["end"])
                if previous_end is not None and launch_end < previous_end:
                    raise ProfilingNameError(
                        "CUDA runtime calls overlap on one host thread; "
                        "cannot safely stream NVTX ranges"
                    )
                previous_end = launch_end

                while (
                    next_range is not None
                    and int(next_range["start"]) <= launch_start
                ):
                    text = ""
                    if (
                        "text" in next_range.keys()
                        and next_range["text"] is not None
                    ):
                        text = str(next_range["text"])
                    elif (
                        "textId" in next_range.keys()
                        and next_range["textId"] is not None
                    ):
                        text = _resolve_string(next_range["textId"], strings)
                    if text and (not label_prefix or text.startswith(label_prefix)):
                        range_start = int(next_range["start"])
                        range_end = int(next_range["end"])
                        active.add(range_id)
                        # Latest start, then shortest duration, is innermost.
                        heapq.heappush(
                            starts,
                            (
                                -range_start,
                                range_end - range_start,
                                range_id,
                                range_end,
                                text,
                            ),
                        )
                        heapq.heappush(ends, (range_end, range_id))
                        range_id += 1
                    next_range = next(ranges, None)

                while ends and ends[0][0] < launch_end:
                    _, expired_id = heapq.heappop(ends)
                    active.discard(expired_id)
                while starts and starts[0][2] not in active:
                    heapq.heappop(starts)
                # Expired ranges below the best active candidate remain in a
                # heapq. Rebuild occasionally so memory tracks active nesting,
                # not every range encountered since the first launch.
                if len(starts) > 2 * len(active) + 1024:
                    starts = [entry for entry in starts if entry[2] in active]
                    heapq.heapify(starts)
                if starts:
                    original = _resolve_string(launch["original"], strings)
                    aliases_by_original[original].add(starts[0][4])
                launch = next(launches, None)

        collisions = {
            original: aliases
            for original, aliases in aliases_by_original.items()
            if len(aliases) > 1
        }
        mapping = {
            original: next(iter(aliases))
            for original, aliases in sorted(aliases_by_original.items())
            if len(aliases) == 1
        }
        return mapping, collisions
    finally:
        connection.close()


def write_ncu_yaml(output_path: Path, mapping: dict[str, str]) -> None:
    """Write the list-of-pairs YAML format accepted by Nsight Compute."""

    with output_path.open("w", encoding="utf-8") as output:
        for original, renamed in mapping.items():
            output.write("-\n")
            output.write(f"  - Original: {json.dumps(original)}\n")
            output.write(f"  - Renamed: {json.dumps(renamed)}\n")


def write_collision_yaml(
    output_path: Path, collisions: dict[str, set[str]]
) -> None:
    """Write symbols with multiple candidate NVTX labels for later review."""

    with output_path.open("w", encoding="utf-8") as output:
        for original, labels in sorted(collisions.items()):
            output.write(f"- Original: {json.dumps(original)}\n")
            output.write("  Candidates:\n")
            for label in sorted(labels):
                output.write(f"    - {json.dumps(label)}\n")


def parse_nsys_names(help_text: str) -> list[str]:
    """Parse the identifiers printed by nsys stats/recipe help."""

    names = []
    for line in help_text.splitlines():
        match = re.match(r"^  ([a-z][a-z0-9_]+)(?=\s|\[|:)", line)
        if match:
            names.append(match.group(1))
    return sorted(set(names))


def discover_nsys() -> dict[str, list[str]]:
    stats = subprocess.run(
        ["nsys", "stats", "--help-reports"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    recipes = subprocess.run(
        ["nsys", "recipe", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "stats_reports": parse_nsys_names(stats.stdout),
        "recipes": parse_nsys_names(recipes.stdout),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sqlite", type=Path, nargs="?", help="NSYS-exported SQLite database to map"
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--collisions-output",
        type=Path,
        help="YAML file for symbols that map to more than one NVTX label "
        "(default: <output>.collisions.yaml)",
    )
    parser.add_argument(
        "--label-prefix",
        default="",
        help="Only use NVTX labels beginning with this literal prefix",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="List installed NSYS stats reports and recipes instead of mapping a database",
    )
    parser.add_argument("--json", action="store_true", help="Format --discover output as JSON")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.discover:
            if args.sqlite or args.output:
                parser.error("--discover cannot be combined with a database or --output")
            discovered = discover_nsys()
            if args.json:
                print(json.dumps(discovered, indent=2))
            else:
                for category, names in discovered.items():
                    print(f"{category}:")
                    for name in names:
                        print(f"  {name}")
        else:
            if args.sqlite is None or args.output is None:
                parser.error("sqlite database and --output are required unless --discover is used")
            mapping, collisions = kernel_nvtx_sqlite_mapping(
                args.sqlite, args.label_prefix
            )
            if not mapping:
                raise ProfilingNameError(
                    "No kernel launches were enclosed by matching named NVTX ranges"
                )

            collisions_output = args.collisions_output or args.output.with_suffix(
                ".collisions.yaml"
            )
            write_collision_yaml(collisions_output, collisions)
            if collisions:
                print(
                    f"WARNING: {len(collisions)} overloaded symbols; "
                    f"see {collisions_output}",
                    file=sys.stderr,
                )
            mapping.update(
                {
                    original: "".join(
                        chars[0] if len(set(chars)) == 1 else "*"
                        for chars in zip(*aliases)
                    )
                    for original, aliases in collisions.items()
                }
            )

            write_ncu_yaml(args.output, mapping)
            for original, renamed in mapping.items():
                print(f"{renamed} <- {original}")
    except (
        OSError,
        sqlite3.Error,
        subprocess.CalledProcessError,
        ProfilingNameError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
