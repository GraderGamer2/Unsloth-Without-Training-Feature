# SPDX-License-Identifier: AGPL-3.0-only
"""No model/network required: temporary-source integrity and coverage scheduling."""
from hashlib import sha256
from pathlib import Path
import importlib.util
import runpy
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
from urllib.parse import urldefrag

BACKEND = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("temporary_sources", BACKEND / "core/research/temporary_sources.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidateCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = module.CandidateCache(max_bytes=8, max_entries=2)
        self.addCleanup(self.cache.discard, ())
        self.scope = ("account", "project", "run")
        self.policy = {"allowedDomains": ["example.org"], "blockedDomains": []}
        self.url = "https://example.org/document"
        self.payload = b"PDF!"
        self.digest = sha256(self.payload).hexdigest()

    def put(self, url=None, payload=None):
        return self.cache.put(self.scope, url or self.url, "file.pdf", payload or self.payload,
                              {"sourceUrl": self.url, "text": "not cached as metadata"}, self.policy, self.temp.name)

    def test_original_bytes_and_scope(self):
        self.assertTrue(self.put())
        self.assertEqual(self.cache.read(self.scope, self.url, self.policy, self.digest),
                         ("file.pdf", self.payload, {"sourceUrl": self.url}))
        for scope in (("another", "project", "run"), ("account", "other", "run"), ("account", "project", "other")):
            self.assertIsNone(self.cache.read(scope, self.url, self.policy, self.digest))
        self.assertIsNone(self.cache.read(self.scope, self.url, {"allowedDomains": ["other.org"]}, self.digest))
        self.assertIsNone(self.cache.read(self.scope, self.url, self.policy, "bad"))

    def test_eviction_and_discard(self):
        for suffix in ("", "/2", "/3"):
            self.assertTrue(self.put(self.url + suffix))
        self.assertIsNone(self.cache.read(self.scope, self.url, self.policy, self.digest))
        self.assertEqual(self.cache.size, 8)
        self.cache.discard(self.scope)
        self.assertEqual(self.cache.size, 0)

    def test_expiry(self):
        self.cache.ttl = 0
        self.assertTrue(self.put())
        self.cache.prune()
        self.assertEqual(self.cache.size, 0)

    def test_oversized(self):
        self.assertFalse(self.put(payload=b"123456789"))
        self.assertEqual(self.cache.size, 0)


class DiscoveryCoverageTests(unittest.TestCase):
    def setUp(self):
        policy = ModuleType("core.inference.web_access_policy")
        policy.check_url_access = lambda url, _: (url.startswith("https://example.org/"), "", "")
        redaction = ModuleType("core.research.redaction")
        redaction._sanitize_public_query = lambda q: q.strip()[:500]
        importer = ModuleType("core.rag.research_source_import")
        importer.canonical_source_url = lambda url: urldefrag(url)[0]
        importer.fetch_source_payload = None
        importer.retained_source = lambda docs, url: None
        self.patches = patch.dict(sys.modules, {policy.__name__: policy, redaction.__name__: redaction, importer.__name__: importer})
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.discovery = runpy.run_path(str(BACKEND / "core/research/discovery.py"))

    def test_link_and_coverage_followup(self):
        frontier = self.discovery["DiscoveryFrontier"]("academic research forest water", None, True, 100)
        first = {"action": "search", "query": "forest water", "title": "Initial"}
        self.assertEqual(frontier.next_action(first, [], set(), set()), first)
        frontier.observe([{"url": "https://example.org/paper.pdf", "title": "Forest paper", "snippet": "forest"},
                          {"url": "https://example.org/paper.pdf#page=1", "title": "Duplicate", "snippet": ""}])
        self.assertEqual(len(frontier.pending), 1)
        next_action = frontier.next_action(first, [], set(), {"query1", "query2"})
        self.assertEqual(next_action["url"], "https://example.org/paper.pdf")
        frontier.record_batch("search", [{"url": "https://example.org/paper.pdf", "title": "Forest", "snippet": "forest", "verification": "downloaded"}])
        self.assertEqual(frontier.summary()["downloaded"], 1)
        self.assertIn("water", frontier.summary()["uncoveredTerms"])
        self.assertTrue(frontier.followups)

    def test_openalex_rejects_disallowed_locations(self):
        providers = runpy.run_path(str(BACKEND / "core/research/academic_providers.py"))
        found = providers["_locations"]({"display_name": "Paper", "doi": "https://blocked.org/paper",
            "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/paper.pdf",
                                 "landing_page_url": "https://blocked.org/landing"}}, {})
        self.assertEqual([item["url"] for item in found], ["https://example.org/paper.pdf"])


if __name__ == "__main__":
    unittest.main()
