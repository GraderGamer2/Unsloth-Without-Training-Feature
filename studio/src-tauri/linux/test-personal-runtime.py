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

    def test_whisper_silent_tail_partition(self):
        # Model the old interleaved per-worker tail and the new contiguous fill.
        for total in (1, 16, 3000, 3001):
            for first_silent in (0, 1, total // 2, total):
                if first_silent > total:
                    continue
                for threads in (1, 2, 3, 4, 8, 16):
                    written = set()
                    for ith in range(threads):
                        i = ith
                        while i < first_silent:
                            i += threads
                        written.update(range(i, total, threads))
                    self.assertEqual(written, set(range(first_silent, total)))

    def test_whisper_padding_worker_bounds(self):
        # The spectrogram frame count retains the full logical padding.
        # The worker sees only the input and leading reflection.
        frame_size, step = 400, 160
        leading, silence = 200, 480000
        for samples in (201, 399, 400, 401, 1600, 16001, 480000, 480001):
            physical = samples + leading
            old_size = samples + silence + leading * 2
            logical_size = physical + silence + leading
            self.assertEqual(old_size, logical_size)
            frames = (logical_size - frame_size) // step
            for i in range(min(physical // step + 1, frames)):
                offset = i * step
                copied = min(frame_size, physical - offset)
                self.assertGreaterEqual(copied, 0)
                self.assertLessEqual(offset + copied, physical)

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
