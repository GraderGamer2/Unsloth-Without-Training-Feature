#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Source-patch composition and control-flow regression tests (no GPU required)."""
from pathlib import Path
import runpy
import tempfile
import unittest

PATCH = runpy.run_path(str(Path(__file__).with_name("apply-personal-llama-vulkan.py")))
ROOT = PATCH["ROOT"]
SPECS = PATCH["patch_specs"]()


class VulkanPatchTests(unittest.TestCase):
    def fixture(self, root):
        originals = {}
        for relative, old, _new, count in SPECS:
            path = root / ROOT / relative
            originals[path] = originals.get(path, "") + (old + "\n// fixture separator\n") * count
        for path, contents in originals.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        return originals

    def test_composition_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = self.fixture(root)
            PATCH["apply"](root)
            patched = {path: path.read_text(encoding="utf-8") for path in originals}
            for relative, _old, new, count in SPECS:
                self.assertEqual(patched[root / ROOT / relative].count(new), count)
            PATCH["apply"](root)
            self.assertEqual(patched, {path: path.read_text(encoding="utf-8") for path in originals})

    def test_invalid_anchor_does_not_modify_any_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = self.fixture(root)
            relative, old, _new, _count = SPECS[-1]
            path = root / ROOT / relative
            originals[path] = originals[path].replace(old, "// upstream source changed")
            path.write_text(originals[path], encoding="utf-8")
            with self.assertRaises(RuntimeError):
                PATCH["apply"](root)
            self.assertEqual(originals, {p: p.read_text(encoding="utf-8") for p in originals})

    def test_cooperative_mask_reduction_keeps_upstream_barrier(self):
        # The scalar kernel has an earlier barrier; cm1 does not. Removing the
        # latter's barrier would permit inter-subgroup overwrite on a skip.
        self.assertFalse(any(p == "vulkan-shaders/flash_attn_cm1.comp"
                             for p, old, new, count in SPECS))

    def test_empty_tail_has_no_active_blocks(self):
        for group_size in (16, 32, 64, 128, 256):
            groups = group_size // 16
            for blocks in range(1, 257):
                remainder = blocks % groups
                full = blocks - remainder
                old_active = [full + lane for lane in range(groups) if full + lane < blocks]
                new_active = old_active if remainder else []
                self.assertEqual(old_active, new_active)
                self.assertEqual(len(new_active), remainder)


if __name__ == "__main__":
    unittest.main()
