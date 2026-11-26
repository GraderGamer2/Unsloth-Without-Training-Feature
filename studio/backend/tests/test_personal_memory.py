# SPDX-License-Identifier: AGPL-3.0-only
"""Standalone tests; no torch or Studio initialization needed."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "personal_memory", Path(__file__).parents[1] / "core/inference/personal_memory.py"
)
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)
MIB = memory.MIB
check = memory.hybrid_memory_shortfall


class PersonalMemoryTest(unittest.TestCase):
    def test_safe_load(self):
        self.assertIsNone(check(16384, 4096 * MIB, 512 * MIB))

    def test_oversized_load(self):
        self.assertIsNotNone(check(4096, 4096 * MIB, 0))

    def test_exact_budget(self):
        self.assertIsNone(check(8192, 4096 * MIB, 0))
        self.assertIsNotNone(check(8192, 4096 * MIB, 1))

    def test_unknown_budget(self):
        self.assertIsNotNone(check(None, 4096 * MIB, 0))
        self.assertIsNotNone(check(8192, 0, 0))
        self.assertIsNotNone(check(8192, 4096 * MIB, -1))

    def test_large_context(self):
        self.assertIsNotNone(check(16384, 4096 * MIB, 10000 * MIB))


if __name__ == "__main__":
    unittest.main()
