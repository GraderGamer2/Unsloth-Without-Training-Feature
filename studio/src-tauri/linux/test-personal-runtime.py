#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path
import runpy
import tempfile
import unittest

patch = runpy.run_path(str(Path(__file__).with_name("apply-personal-runtime.py")))


class RuntimePatchTests(unittest.TestCase):
    def test_composition_and_idempotence(self):
        for component in ("llama", "diffusion", "whisper", "audio"):
            for newline in ("\n", "\r\n"):
                with self.subTest(component=component, newline=newline), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    originals = {}
                    for path, old, _new in patch["patch_specs"](component):
                        originals[path] = originals.get(path, "") + old + "\n// separator\n"
                    for path, content in originals.items():
                        target = root / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content.replace("\n", newline).encode())
                    patch["apply"](component, root)
                    after = {p: (root / p).read_bytes() for p in originals}
                    for p, _old, new in patch["patch_specs"](component):
                        self.assertEqual(after[p].count(new.replace("\n", newline).encode()), 1)
                    patch["apply"](component, root)
                    self.assertEqual(after, {p: (root / p).read_bytes() for p in originals})

    def test_anchor_failure_leaves_all_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = patch["patch_specs"]("llama")
            first, old, _new = specs[0]
            target = root / first
            target.parent.mkdir(parents=True)
            target.write_text(old)
            with self.assertRaises(FileNotFoundError):
                patch["apply"]("llama", root)
            self.assertEqual(target.read_text(), old)


if __name__ == "__main__":
    unittest.main()
