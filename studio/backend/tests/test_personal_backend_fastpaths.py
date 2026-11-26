# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only regressions for model metadata and document chunking fast paths."""
from __future__ import annotations

import ast
import bisect
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import re
import runpy
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Callable, Optional
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]


def source_defs(path, names, *, class_name=None, globals_=None):
    source = BACKEND / path
    body = ast.parse(source.read_text(encoding="utf-8")).body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    for n in nodes:
        if isinstance(n, ast.FunctionDef) and class_name:
            n.decorator_list = []
    namespace = dict(globals_ or {}, __name__=__name__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class FastPathsTest(unittest.TestCase):
    def test_unknown_quant_offsets(self):
        constants = ModuleType("gguf.constants")
        constants.GGML_QUANT_SIZES = {0: (1, 4)}
        layout = ModuleType("core.inference.offload_layout")
        layout._BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
        layout._MOE_EXPERT_RE = re.compile(r"^ffn_(up|gate|down|gate_up)_(exps|chexps)\.weight$")
        scan = source_defs(
            "core/inference/llama_cpp.py", {"_gguf_scan_tensor_bytes"},
            class_name="LlamaCppBackend",
            globals_=dict(struct=struct, math=math, os=os, bisect=bisect,
                          Optional=Optional, Callable=Callable, _LAZY_READ_TENSORS={}),
        )["_gguf_scan_tensor_bytes"]
        rng = random.Random(7735)
        with patch.dict(sys.modules, {"gguf.constants": constants, "core.inference.offload_layout": layout}):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "test.gguf"
                for _ in range(120):
                    offsets = [rng.randrange(0, 20) * 32 for _ in range(50)]
                    rows = [(f"blk.{i}.ffn_up_exps.weight" if i % 3 == 0 else f"tensor.{i}",
                             0 if i % 5 == 0 else 999, offset)
                            for i, offset in enumerate(offsets)]
                    header = bytearray(struct.pack("<IIQQ", 0x46554747, 2, len(rows), 0))
                    for name, quant, offset in rows:
                        raw = name.encode()
                        header.extend(struct.pack("<Q", len(raw)) + raw)
                        header.extend(struct.pack("<IQIQ", 1, 4, quant, offset))
                    header.extend(bytes((-len(header)) % 32))
                    data_end = 640
                    path.write_bytes(header + bytes(data_end))
                    ordered = sorted(offsets)
                    expected = {}
                    experts = 0
                    kept = {name for i, (name, _, _) in enumerate(rows) if i % 2 == 0}
                    for i, (name, quant, offset) in enumerate(rows):
                        later = ordered[bisect.bisect_right(ordered, offset):]
                        size = 16 if quant == 0 else max(0, (later[0] if later else data_end) - offset)
                        if i % 3 == 0:
                            experts += size
                        if name in kept:
                            expected[name] = size
                    arch, named, expert_bytes = scan(str(path), also=kept.__contains__)
                    self.assertIsNone(arch)
                    self.assertEqual(named, expected)
                    self.assertEqual(expert_bytes, experts)

    def test_split_layout_reader_reuse(self):
        lookup = ModuleType("core.inference.lookup_tables")
        lookup.is_per_layer_embedding = lambda name: False
        gguf = ModuleType("gguf")
        gguf.GGUFEndian = object()
        gguf.GGUFValueType = object()
        calls = []

        class Reader:
            def __init__(self, path):
                calls.append(path)
                first = "00001-of-" in path
                values = {"split.count": 2}
                if first:
                    values.update({
                        "general.architecture": "llama",
                        "llama.block_count": 2,
                        "llama.attention.head_count_kv": 1,
                        "llama.attention.key_length": 4,
                        "llama.attention.value_length": 4,
                    })
                self.fields = {k: SimpleNamespace(contents=lambda v=v: v) for k, v in values.items()}
                rows = ([("token_embd.weight", 64), ("blk.0.ffn_up.weight", 80)] if first
                        else [("blk.1.ffn_down.weight", 96), ("output.weight", 128)])
                self.tensors = [SimpleNamespace(name=n, n_bytes=b) for n, b in rows]

        gguf.GGUFReader = Reader
        with patch.dict(sys.modules, {
            "core.inference.lookup_tables": lookup, "gguf": gguf, "numpy": ModuleType("numpy"),
        }):
            read = runpy.run_path(str(BACKEND / "core/inference/offload_layout.py"))["layout_from_gguf"]
            with tempfile.TemporaryDirectory() as tmp:
                paths = [str(Path(tmp) / f"model-{i:05d}-of-00002.gguf") for i in (1, 2)]
                for p in paths:
                    Path(p).touch()
                for start in paths:
                    calls.clear()
                    result = read(start, all_shards=True)
                    self.assertTrue(result.complete)
                    self.assertEqual(result.arch, "llama")
                    self.assertEqual(result.tensor_bytes, 368)
                    self.assertEqual([b.index for b in result.blocks], [0, 1])
                    self.assertEqual(len(calls), 2)
                    self.assertCountEqual(calls, paths)
                Path(paths[1]).unlink()
                self.assertFalse(read(paths[0], all_shards=True).complete)

    def test_rag_overlap_equivalence(self):
        namespace = source_defs("core/rag/chunking.py", {"_Piece", "_merge"},
                                globals_={"dataclass": dataclass})
        piece, merge = namespace["_Piece"], namespace["_merge"]

        def reference(pieces, starts, maximum, overlap):
            chunks, buf, positions, tokens = [], [], [], 0

            def flush():
                raw = "".join(p.text for p in buf)
                if raw.strip():
                    chunks.append((raw.strip(), positions[0] + len(raw) - len(raw.lstrip()),
                                   positions[0] + len(raw.rstrip())))

            for p, start in zip(pieces, starts):
                if buf and tokens + p.token_count > maximum:
                    flush()
                    budget = min(overlap, max(0, maximum - p.token_count))
                    carry, carry_pos, used = [], [], 0
                    for prev, pos in zip(reversed(buf), reversed(positions)):
                        if used + prev.token_count > budget:
                            break
                        carry.insert(0, prev)
                        carry_pos.insert(0, pos)
                        used += prev.token_count
                    buf, positions, tokens = carry, carry_pos, used
                buf.append(p)
                positions.append(start)
                tokens += p.token_count
            if buf:
                flush()
            return chunks

        rng = random.Random(680)
        for _ in range(1000):
            pieces = [piece(rng.choice(["x", " y", " ", "\n", "abc"]), rng.randrange(0, 8))
                      for _ in range(rng.randrange(0, 250))]
            starts, pos = [], 0
            for p in pieces:
                starts.append(pos)
                pos += len(p.text)
            maximum, overlap = rng.randrange(1, 40), rng.randrange(0, 50)
            self.assertEqual(merge(pieces, starts, maximum, overlap),
                             reference(pieces, starts, maximum, overlap))


if __name__ == "__main__":
    unittest.main()
