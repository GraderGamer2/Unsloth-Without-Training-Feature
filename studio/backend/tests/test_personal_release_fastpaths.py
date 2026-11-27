# SPDX-License-Identifier: AGPL-3.0-only
"""Dependency-free checks of personal release fast paths and diagnostic policy."""
from dataclasses import dataclass
import heapq
import random
import unittest
from test_personal_backend_fastpaths import source_defs, BACKEND


class PersonalReleaseTests(unittest.TestCase):
    def test_rrf_matches_stable_reference(self):
        ns = source_defs("core/rag/retrieval.py", {"Hit", "_rrf", "filter_min_score"},
                         globals_=dict(dataclass=dataclass, heapq=heapq))
        Hit, rrf = ns["Hit"], ns["_rrf"]
        rng = random.Random(7735)
        for _ in range(200):
            lexical = [Hit(str(i), 0, lexical_score=0.5) for i in rng.sample(range(90), 30)]
            dense = [Hit(str(i), 0, dense_score=rng.choice([0.2, 0.5, 0.9]))
                     for i in rng.sample(range(90), 30)]
            for top_k in (-2, 0, 1, 8, 60):
                for floor in (0, 0.5, 0.95, float("nan")):
                    ranks = {}
                    for ranking in (lexical, dense):
                        for rank, hit in enumerate(ranking):
                            score, lex, den = ranks.get(hit.chunk_id, (0.0, None, None))
                            ranks[hit.chunk_id] = (score + 1.0 / (61 + rank),
                                                   lex if lex is not None else hit.lexical_score,
                                                   den if den is not None else hit.dense_score)
                    expected = [(cid, s, l, d) for cid, (s, l, d) in
                                sorted(ranks.items(), key=lambda item: item[1][0], reverse=True)
                                if floor <= 0 or d is None or d >= floor][:top_k]
                    actual = [(h.chunk_id, h.score, h.lexical_score, h.dense_score)
                              for h in rrf([lexical, dense], 60, top_k, floor)]
                    self.assertEqual(actual, expected)

    def test_no_managed_engine_file_open_in_personal_build(self):
        import ast
        import sys
        from types import ModuleType
        from unittest.mock import patch
        path = BACKEND / "core/inference/managed_engine_log.py"
        body = ast.parse(path.read_text()).body
        cls = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == "ManagedEngineLog")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "open")
        method.decorator_list = []
        method.returns = None
        ns = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        policy = ModuleType("utils.gpu_policy")
        policy.personal_vulkan = lambda: True
        with patch.dict(sys.modules, {"utils.gpu_policy": policy}):
            self.assertIsNone(ns["open"](object(), "llama"))


if __name__ == "__main__":
    unittest.main()
