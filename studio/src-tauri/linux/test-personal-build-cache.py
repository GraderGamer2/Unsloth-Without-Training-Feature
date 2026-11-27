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


    def test_literal_patch_amendments_do_not_invalidate_unrelated_engines(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "probe.py"
            baseline = ('LLAMA = []\nENGINE_EXTRA = {"audio": []}\n'
                        'def patch_specs(component):\n'
                        '    return LLAMA if component == "llama" else ENGINE_EXTRA[component]\n')
            source.write_text(baseline)
            with patch.dict(MOD["patch_fingerprint"].__globals__, {"ROOT": root}):
                a = MOD["patch_fingerprint"]("probe.py", "audio")
                l = MOD["patch_fingerprint"]("probe.py", "llama")
                source.write_text(baseline + 'LLAMA += [("x", "before", "after")]\n')
                self.assertEqual(a, MOD["patch_fingerprint"]("probe.py", "audio"))
                self.assertNotEqual(l, MOD["patch_fingerprint"]("probe.py", "llama"))
                source.write_text(baseline + 'ENGINE_EXTRA["audio"] += [("a", "b", "c")]\n')
                self.assertNotEqual(a, MOD["patch_fingerprint"]("probe.py", "audio"))
                self.assertEqual(l, MOD["patch_fingerprint"]("probe.py", "llama"))
                source.write_text(baseline + 'LLAMA += list()\n')
                self.assertNotEqual(a, MOD["patch_fingerprint"]("probe.py", "audio"))

if __name__ == "__main__":
    unittest.main()
