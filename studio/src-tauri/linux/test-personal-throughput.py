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

    def test_ring_indices(self):
        for cap in range(1,65):
            for first in range(cap):
                for size in range(1,cap+1):
                    for i in range(size):
                        idx = first+size-i-1
                        self.assertEqual(idx%cap,idx-cap if idx>=cap else idx)

if __name__ == "__main__":
    unittest.main()
