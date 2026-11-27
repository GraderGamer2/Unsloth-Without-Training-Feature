#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path
import runpy
import tempfile
import unittest

PATCH = runpy.run_path(str(Path(__file__).with_name("apply-personal-throughput.py")))

class ThroughputTests(unittest.TestCase):
    def test_fixtures_and_idempotence(self):
        for engine in ("llama", "diffusion", "whisper", "audio"):
            for nl in ("\n", "\r\n"):
                with self.subTest(engine=engine, newline=nl), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    original = {}
                    for p, old, _ in PATCH["patch_specs"](engine):
                        original[p] = original.get(p, "") + old + "\n// separator\n"
                    for p, data in original.items():
                        path = root / p
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data.replace("\n", nl).encode())
                    PATCH["apply"](engine, root)
                    after = {p: (root / p).read_bytes() for p in original}
                    for p, _, new in PATCH["patch_specs"](engine):
                        self.assertEqual(after[p].count(new.replace("\n", nl).encode()), 1)
                    PATCH["apply"](engine, root)
                    self.assertEqual(after, {p: (root / p).read_bytes() for p in original})

    def test_activation_load_addresses(self):
        for blocks in range(1, 65):
            for lanes in (1, 2, 4, 8, 16):
                for start in range(0, blocks+lanes, lanes):
                    for rows in (1, 2, 4):
                        for cols in (1, 2, 4):
                            old = [(i,c) for i in range(start,start+lanes) if i<blocks
                                   for _ in range(rows) for c in range(cols)]
                            new = [(i,c) for i in range(start,start+lanes) if i<blocks
                                   for _ in range(1 if cols==1 and rows>1 else rows)
                                   for c in range(cols)]
                            self.assertEqual(set(old),set(new))
                            self.assertLessEqual(len(new),len(old))

    def test_iq_ternary_cached_addresses(self):
        # Compare every lane address, including inactive tail blocks.
        for width, stride, count in ((4, 1, 4), (2, 8, 8)):
            for base in (0, 1, 4, 127, 256):
                for y in range(0, 256, 2):
                    cached = [(base + y) // width + stride * k for k in range(count)]
                    for cols in (1, 2, 4):
                        for rows in (1, 2, 4):
                            reuse = cols == 1 and rows > 1
                            for valid in (False, True):
                                if not valid:
                                    # No cached load occurs for an inactive TQ2 lane.
                                    self.assertFalse(reuse and valid)
                                    continue
                                for row in range(rows):
                                    for col in range(cols):
                                        expected = [(col * 1024 + base + y) // width + stride * k
                                                    for k in range(count)]
                                        actual = cached if reuse else expected
                                        self.assertEqual(actual, expected)

    def test_ring_indices(self):
        for cap in range(1,65):
            for first in range(cap):
                for size in range(1,cap+1):
                    for i in range(size):
                        idx = first+size-i-1
                        self.assertEqual(idx%cap,idx-cap if idx>=cap else idx)

if __name__ == "__main__":
    unittest.main()
