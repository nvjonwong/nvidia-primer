#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import tempfile
import unittest
from pathlib import Path

from profiling_name_tools import (
    _build_parser,
    kernel_nvtx_sqlite_mapping,
    parse_nsys_names,
    write_collision_yaml,
    write_ncu_yaml,
)


class ProfilingNameToolsTest(unittest.TestCase):
    def make_sqlite_database(self, collide=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "profile.sqlite"
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                correlationId INTEGER, demangledName INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                correlationId INTEGER, start INTEGER, end INTEGER,
                globalTid INTEGER
            );
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER, textId INTEGER
            );
            INSERT INTO StringIds VALUES
                (1, 'void StableFunctor::kernel()'),
                (2, 'KokkosPrimer.NcuRename.Outer'),
                (3, 'KokkosPrimer.NcuRename.Target'),
                (4, 'KokkosPrimer.NcuRename.Other');
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (7, 1);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (7, 120, 125, 99);
            INSERT INTO NVTX_EVENTS VALUES (100, 200, 99, 2);
            INSERT INTO NVTX_EVENTS VALUES (110, 140, 99, 3);
            """
        )
        if collide:
            connection.executescript(
                """
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (8, 1);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (8, 320, 325, 99);
                INSERT INTO NVTX_EVENTS VALUES (300, 340, 99, 4);
                """
            )
        connection.commit()
        connection.close()
        return path

    def test_sqlite_mapper_uses_innermost_range(self):
        mapping, collisions = kernel_nvtx_sqlite_mapping(
            self.make_sqlite_database(), "KokkosPrimer.NcuRename."
        )
        self.assertEqual(
            mapping,
            {"void StableFunctor::kernel()": "KokkosPrimer.NcuRename.Target"},
        )
        self.assertEqual(collisions, {})

    def test_sqlite_mapper_writes_overloaded_symbols_to_collision_data(self):
        mapping, collisions = kernel_nvtx_sqlite_mapping(
            self.make_sqlite_database(collide=True), "KokkosPrimer.NcuRename."
        )
        self.assertEqual(mapping, {})
        self.assertEqual(
            collisions,
            {
                "void StableFunctor::kernel()": {
                    "KokkosPrimer.NcuRename.Target",
                    "KokkosPrimer.NcuRename.Other",
                }
            },
        )

    def test_writes_collision_yaml_without_pyyaml(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "ncu-kernel-collisions.yaml"
        write_collision_yaml(
            output,
            {"void Kernel<int>()": {"Readable Kernel A", "Readable Kernel B"}},
        )
        self.assertEqual(
            output.read_text(encoding="utf-8"),
            '- Original: "void Kernel<int>()"\n'
            "  Candidates:\n"
            '    - "Readable Kernel A"\n'
            '    - "Readable Kernel B"\n',
        )

    def test_writes_ncu_yaml_without_pyyaml(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "ncu-kernel-renames.yaml"
        write_ncu_yaml(output, {"void Kernel<int>()": "Readable Kernel"})
        self.assertEqual(
            output.read_text(encoding="utf-8"),
            '-\n  - Original: "void Kernel<int>()"\n'
            '  - Renamed: "Readable Kernel"\n',
        )

    def test_parses_stats_reports_and_recipes(self):
        sample = """
The following built-in reports are available:
  cuda_gpu_kern_sum[:nvtx-name][:base|:mangled] -- summary
  cuda_gpu_trace[:nvtx-name][:base|:mangled] -- trace
  nvtx_kern_sum[:base|:mangled] -- NVTX summary
The following built-in recipes are available:
  cuda_gpu_kern_sum -- recipe
  nvtx_gpu_proj_trace -- recipe
"""
        self.assertEqual(
            parse_nsys_names(sample),
            [
                "cuda_gpu_kern_sum",
                "cuda_gpu_trace",
                "nvtx_gpu_proj_trace",
                "nvtx_kern_sum",
            ],
        )


    def test_default_cli_uses_sqlite_mapper(self):
        args = _build_parser().parse_args(["profile.sqlite", "--output", "renames.yaml"])
        self.assertEqual(args.sqlite, Path("profile.sqlite"))
        self.assertEqual(args.output, Path("renames.yaml"))
        self.assertFalse(args.discover)

    def test_discovery_is_opt_in(self):
        args = _build_parser().parse_args(["--discover", "--json"])
        self.assertTrue(args.discover)
        self.assertTrue(args.json)


if __name__ == "__main__":
    unittest.main()
