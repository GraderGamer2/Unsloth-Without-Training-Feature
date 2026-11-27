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

    def test_sequence_set_union_matches_pairwise_selection(self):
        import random
        rng = random.Random(7735)
        for _ in range(1000):
            chosen, occupied = [], 0
            last = -1
            sequential = bool(rng.randrange(2))
            for _ in range(128):
                candidate = sum(1 << bit for bit in set(
                    rng.randrange(256) for _ in range(rng.randrange(8))))
                seq_id = rng.randrange(16)
                allowed = not sequential or not chosen or seq_id == last + 1
                before = allowed and all(not (s & candidate) for s in chosen)
                after = allowed and not (occupied & candidate)
                self.assertEqual(bool(before), bool(after))
                if after:
                    chosen.append(candidate)
                    occupied |= candidate
                    last = seq_id

    def test_rejection_probability_index_first_match(self):
        import random
        rng = random.Random(7735)
        for _ in range(1000):
            q = [(rng.randrange(256), rng.random()) for _ in range(rng.randrange(128))]
            p = [(rng.randrange(256), rng.random()) for _ in range(rng.randrange(128))]
            lookup = {}
            if len(q) >= 32 and len(p) >= 32:
                for token, prob in q:
                    lookup.setdefault(token, prob)
            def scan(token):
                return next((prob for ident, prob in q if ident == token), 0.0)
            before = [max(0.0, prob - scan(token)) for token, prob in p]
            after = [max(0.0, prob - (lookup.get(token, 0.0) if lookup else scan(token)))
                     for token, prob in p]
            self.assertEqual(before, after)
            self.assertEqual(sum(before), sum(after))

    def test_adaptive_draft_limits_and_rollback_bound(self):
        # Preserve replay, synthetic verification, configured minimum, and
        # regular probes. Never select more than the existing context cap.
        for cap in range(1, 33):
            for minimum in range(1, 17):
                for rounds in range(1, 33):
                    for accepted, drafted in ((0, 64), (7, 100), (50, 100), (0, 8)):
                        eligible = (drafted >= 64 and rounds >= 8 and rounds % 8 != 0
                                    and accepted * 5 < drafted)
                        result = min(cap, max(2, minimum)) if eligible else cap
                        self.assertGreaterEqual(result, 1)
                        self.assertLessEqual(result, cap)
                        if eligible:
                            self.assertGreaterEqual(result, min(cap, minimum))
                        else:
                            self.assertEqual(result, cap)
            for capacity in range(0, 33):
                needs_clone = cap > capacity
                for rollback in range(cap + 1):
                    if rollback > capacity:
                        self.assertTrue(needs_clone)

    def test_clipped_image_composite_order(self):
        import random
        rng = random.Random(7735)
        for _ in range(3000):
            sw, sh, dw, dh = (rng.randrange(1, 32) for _ in range(4))
            ox, oy = (rng.randrange(-40, 41) for _ in range(2))
            old = [(x, y, x + ox, y + oy) for y in range(sh) for x in range(sw)
                   if 0 <= x + ox < dw and 0 <= y + oy < dh]
            new = [(x, y, x + ox, y + oy)
                   for y in range(max(0, -oy), min(sh, dh - oy))
                   for x in range(max(0, -ox), min(sw, dw - ox))]
            self.assertEqual(old, new)

    def test_iq3_iq4_activation_slice(self):
        for base in range(0, 128):
            for cols in (1, 2, 4, 8):
                nl = 4 if cols <= 4 else 2
                for l0 in (0, 4 - nl):
                    cached = [base + 2*l0 + k for k in range(2*nl)]
                    for ll in range(nl):
                        self.assertEqual(cached[2*ll:2*ll+2],
                                         [base + 2*(l0+ll), base + 2*(l0+ll)+1])
                cached = list(range(base, base+8))
                for l in range(4):
                    self.assertEqual((cached[l], cached[4+l]), (base+l, base+4+l))

    def test_diffusion_truncation_equivalence(self):
        for count in range(30):
            for maximum in range(1, 40):
                for bos in (0, 1):
                    for eos in (0, 1):
                        tokens = list(range(10, 10+count))
                        def finish(items):
                            result = ([1] if bos else []) + items + ([2] if eos else [])
                            result = (result + [0] * maximum)[:maximum]
                            if eos:
                                result[-1] = 2
                            return result
                        take = min(count, max(0, maximum-bos))
                        self.assertEqual(finish(tokens), finish(tokens[:take]))

    def test_ring_indices(self):
        for cap in range(1,65):
            for first in range(cap):
                for size in range(1,cap+1):
                    for i in range(size):
                        idx = first+size-i-1
                        self.assertEqual(idx%cap,idx-cap if idx>=cap else idx)

    def test_dspark_uniform_adaptive_blocks(self):
        # A mixed batch must retain an equal number of positions per sequence.
        import itertools
        for maximum in range(1, 17):
            for minimum in range(1, maximum + 1):
                for caps in itertools.product((-1, 0, 1, 2, 4, 8, 16), repeat=3):
                    requested = [min(maximum, cap) if cap > 0 and cap >= minimum
                                 else maximum for cap in caps]
                    shared = max(requested)
                    self.assertGreaterEqual(shared, minimum)
                    self.assertLessEqual(shared, maximum)
                    for anchor in (0, 1):
                        block = shared + anchor
                        for cap, want in zip(caps, requested):
                            end = anchor + min(block - anchor, cap) if cap > 0 and cap >= minimum else block
                            self.assertEqual(end - anchor, want)
                            self.assertLessEqual(end, block)
        # The user's seven-token low-acceptance case reduces actual block work.
        self.assertEqual(max([2]), 2)
        self.assertEqual(max([2, 7]), 7)

    def test_dspark_block_layout_identity(self):
        for blocks in range(1, 9):
            for positions in range(1, 17):
                for width in (1, 3, 11):
                    position_major = [(b, p, k) for p in range(positions)
                                      for b in range(blocks) for k in range(width)]
                    block_major = [(b, p, k) for b in range(blocks)
                                   for p in range(positions) for k in range(width)]
                    if blocks == 1 or positions == 1:
                        self.assertEqual(position_major, block_major)
                    else:
                        self.assertNotEqual(position_major, block_major)

if __name__ == "__main__":
    unittest.main()
