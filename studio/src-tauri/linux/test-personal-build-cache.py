#!/usr/bin/env python3
"""Check cache fingerprints without compiling any native engine."""
from pathlib import Path
import runpy
import unittest

SCRIPT = Path(__file__).resolve().with_name("personal-build-cache-key.py")
MOD = runpy.run_path(str(SCRIPT))
ROOT = SCRIPT.parent


class CacheKeyTests(unittest.TestCase):
    def test_per_component_build_source(self):
        source = (ROOT / "build-personal-runtimes.sh").read_text()
        for engine in MOD["COMPONENTS"]:
            selected = MOD["build_source"](engine)
            self.assertIn("checkout_source " + engine + " ", selected)
            for other in MOD["COMPONENTS"]:
                if other != engine:
                    self.assertNotIn("checkout_source " + other + " ", selected)
        self.assertIn("-march=znver3", source)
        self.assertIn("-DGGML_LTO=ON", source)

    def test_runtime_key_changes_with_toolchain(self):
        for engine in MOD["COMPONENTS"]:
            first = MOD["runtime"](engine, "toolchain-a")
            self.assertEqual(first, MOD["runtime"](engine, "toolchain-a"))
            self.assertNotEqual(first, MOD["runtime"](engine, "toolchain-b"))

    def test_runtime_keys_are_distinct(self):
        keys = {MOD["runtime"](engine, "toolchain") for engine in MOD["COMPONENTS"]}
        self.assertEqual(len(keys), len(MOD["COMPONENTS"]))

    def test_unambiguous_digest(self):
        self.assertNotEqual(MOD["digest"](["a", "bc"]), MOD["digest"](["ab", "c"]))


if __name__ == "__main__":
    unittest.main()
