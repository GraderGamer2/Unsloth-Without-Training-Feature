# SPDX-License-Identifier: AGPL-3.0-only
"""Source collection and idle lifecycle tests without a model or network."""
import ast
import importlib.util
import os
import sqlite3
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, BACKEND / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

policy = load("core.inference.web_access_policy", "core/inference/web_access_policy.py")
web = load("personal_web_sources", "core/rag/web_sources.py")
idle = load("personal_embed_idle", "core/rag/embed_idle.py")

class SourceTests(unittest.TestCase):
    def test_policy_and_deduplication(self):
        output = "\n\n---\n\n".join(
            f"Title: Paper\nURL: {url}\nSnippet: evidence"
            for url in ("https://example.org/p?a=1#part", "https://example.org/p?a=1#other",
                        "https://example.org/p?a=2", "https://blocked.org/p")
        )
        results = web.search_candidates(output, {"allowedDomains": ["example.org"]})
        self.assertEqual([r["url"] for r in results],
                         ["https://example.org/p?a=1", "https://example.org/p?a=2"])
        self.assertEqual(web.search_candidates(output, limit=1), results[:1])
        with self.assertRaises(ValueError):
            web.canonical_source_url("file:///etc/passwd")
        with self.assertRaises(ValueError):
            web.canonical_source_url("https://name:password@example.org")
        with self.assertRaises(ValueError):
            web.intersect_policies({"allowedDomains": ["a.org"]}, {"allowedDomains": ["b.org"]})
        combined = web.intersect_policies({"allowedDomains": ["a.org"]},
                                         {"allowedDomains": ["papers.a.org"], "blockedDomains": ["bad.a.org"]})
        self.assertEqual(combined["allowedDomains"], ["papers.a.org"])
        self.assertEqual(combined["blockedDomains"], ["bad.a.org"])

    def test_snapshot_stable_and_bounded(self):
        url = "https://example.org/p"
        _, a = web.snapshot_markdown(url, "First title", "Evidence")
        _, b = web.snapshot_markdown(url, "Another title", "Evidence")
        self.assertEqual(a, b)
        self.assertIn(url, a)
        self.assertIn("Evidence", a)
        self.assertIn("incomplete", a)
        with self.assertRaises(ValueError):
            web.snapshot_markdown(url, "Empty", " ")
        _, large = web.snapshot_markdown(url, "Large", "a" * (web.MAX_SOURCE_CHARS + 1000))
        self.assertLess(len(large), web.MAX_SOURCE_CHARS + 300)

    def test_fetch_does_not_save_errors_or_search_snippets(self):
        tools = types.ModuleType("core.inference.tools")
        tools.EMPTY_SEARCH_RESULTS = {"No results"}
        tools.is_unreadable_page = lambda value: value == "Unreadable"
        controller = types.ModuleType("core.inference.tool_loop_controller")
        controller.is_tool_error = lambda value: value.startswith("Error:")
        controller.strip_result_for_model = lambda value, _name: value
        calls = []
        def execute(name, args, **kwargs):
            calls.append((name, args, kwargs))
            return "Original evidence"
        tools.execute_tool = execute
        with patch.dict(sys.modules, {tools.__name__: tools, controller.__name__: controller}):
            self.assertEqual(web.fetch_source_text("https://example.org"), "Original evidence")
            self.assertEqual(calls[0][1], {"url": "https://example.org"})
            for failure in ("Error: blocked", "Unreadable", "No results", ""):
                tools.execute_tool = lambda *a, **kw: failure
                with self.assertRaises(ValueError):
                    web.fetch_source_text("https://example.org")

class SelectionTests(unittest.TestCase):
    def test_lexical_and_dense_selection_before_limit(self):
        source = ast.parse((BACKEND / "core/rag/store.py").read_text())
        wanted = {"search_lexical", "_drop_incompatible", "_COMPATIBLE_CHUNKS", "_ID_BATCH"}
        nodes = [n for n in source.body if (
            isinstance(n, ast.FunctionDef) and n.name in wanted
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)
        )]
        namespace = {"sqlite3": sqlite3, "_scopes": lambda scope: [scope],
                     "_match_query": lambda query: query,
                     "_lexical_scope_filter": lambda conn, scopes, placeholders: ("chunks_fts.scope IN (" + placeholders + ")", scopes),
                     "linked_folder_rows_exist": lambda conn: False}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<store>", "exec"), namespace)
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        conn.executescript("""
            CREATE TABLE documents(id TEXT PRIMARY KEY, scope TEXT, source_enabled INTEGER, linked_folder_id TEXT, embedding_model TEXT);
            CREATE TABLE chunks(id TEXT PRIMARY KEY, document_id TEXT);
            CREATE TABLE linked_folder_retired_scopes(scope TEXT);
            CREATE TABLE linked_folder_files(document_id TEXT);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, scope UNINDEXED, text);
            INSERT INTO documents VALUES ('off','project_p',0,NULL,'m'), ('on','project_p',1,NULL,'m'), ('other','project_q',1,NULL,'m');
            INSERT INTO chunks VALUES ('c0','off'),('c1','on'),('c2','other');
            INSERT INTO chunks_fts VALUES ('c0','project_p','needle needle needle'),('c1','project_p','needle'),('c2','project_q','needle needle');
        """)
        hits = namespace["search_lexical"](conn, "project_p", "needle", 1)
        self.assertEqual([row[0] for row in hits], ["c1"])
        self.assertEqual(namespace["_drop_incompatible"](conn, [("c0", 1.), ("c1", .8)], "m", None), [("c1", .8)])
        self.assertEqual(namespace["_drop_incompatible"](conn, [("c1", .8)], "different", None), [])

class IdleTests(unittest.TestCase):
    def test_timer_never_releases_active_work_or_stale_generation(self):
        timers = []
        releases = []
        class Timer:
            def __init__(self, delay, callback, args):
                self.callback, self.args = callback, args
                timers.append(self)
            def cancel(self): pass
            def start(self): pass
            def fire(self): self.callback(*self.args)
        lease = idle.IdleRelease(lambda: releases.append("released"), delay=lambda: 60, timer_factory=Timer)
        with lease.activity():
            with lease.activity():
                self.assertFalse(timers)
            self.assertFalse(timers)
        old = timers[-1]
        with lease.activity():
            old.fire()
            self.assertFalse(releases)
        timers[-1].fire()
        self.assertEqual(releases, ["released"])
        timers[-1].fire()
        self.assertEqual(releases, ["released"])
        with lease.activity(): pass
        timer = timers[-1]
        lease.manual_release(lambda: releases.append("manual"))
        timer.fire()
        self.assertEqual(releases, ["released", "manual"])

    def test_idle_can_be_disabled(self):
        with patch.dict(os.environ, {"UNSLOTH_EMBED_IDLE_SECONDS": "0"}):
            self.assertEqual(idle.idle_seconds(), 0)
        with patch.dict(os.environ, {"UNSLOTH_EMBED_IDLE_SECONDS": "60"}):
            self.assertEqual(idle.idle_seconds(), 60)

if __name__ == "__main__":
    unittest.main()
