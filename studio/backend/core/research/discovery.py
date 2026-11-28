# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic coverage and source verification for the existing research supervisor."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import contextvars
import hashlib
from html.parser import HTMLParser
import json
import re
import threading
import time
from urllib.parse import urlencode, urljoin, urlsplit

from core.inference.web_access_policy import check_url_access
from core.research.redaction import _sanitize_public_query
from core.rag.research_source_import import (
    canonical_source_url, fetch_source_payload, retained_source,
)

_STOP = set("about after also and are best can could does find for from have how into more only research search should that the their them these this those use using what when where which will with would".split())
_ACADEMIC = re.compile(r"\b(?:academic|scholarly|peer.reviewed|literature review|journal articles?|research papers?|doi|systematic review|scientific studies|thesis|dissertation|meta.analysis|jurnal|artikel ilmiah|skripsi|penelitian)\b", re.I)
_TECH = re.compile(r"\b(?:software|code|coding|github|linux|windows|minecraft|performance|driver|vulkan|game|mod|library|framework|benchmark|optimization|api|compile|compiler|build)\b", re.I)
_DOCUMENT = re.compile(r"\.(?:pdf|docx?|xlsx?|pptx?|od[stp]|epub|rtf|csv|tsv|json|xml|png|jpe?g|webp)(?:$|[?#])", re.I)
_CROSSREF_LOCK = threading.Lock()
_crossref_next = 0.0


def words(value):
    return {word for word in re.findall(r"[^\W_]+", value.casefold()) if len(word) >= 3 and word not in _STOP}


def fallback_plan(question, max_steps):
    query = _sanitize_public_query(question)
    if _ACADEMIC.search(question):
        aspects = ("peer reviewed studies", "systematic reviews", "open access full text", "recent findings", "methods limitations")
    elif _TECH.search(question):
        aspects = ("documentation", "benchmarks", "community discussions", "source code", "known issues")
    else:
        aspects = ("primary sources", "independent evidence", "recent findings", "comparisons", "limitations")
    queries = [query]
    for aspect in aspects:
        try:
            expanded = _sanitize_public_query(f"{query} {aspect}")
        except ValueError:
            continue
        if expanded not in queries:
            queries.append(expanded)
    return {"title": "Discover relevant sources", "steps": [
        {"title": q[:120], "query": q} for q in queries[:max_steps]
    ]}


def research_search(query, policy, cancel, timeout, exhaustive):
    """Use the existing search engine, retaining its policy and throttling behavior."""
    from core.inference.tools import _web_search
    return _web_search(_sanitize_public_query(query), max_results=15 if exhaustive else 8,
                       timeout=timeout, cancel_event=cancel, website_policy=policy)


def complementary_search(query, question, policy, cancel, timeout):
    """Public Crossref metadata, only for explicitly academic questions.

    No account/email is fabricated; one shared request every two seconds and
    Retry-After backoff. Metadata links remain subject to ordinary web policy.
    """
    if not _ACADEMIC.search(question) or cancel.is_set():
        return ""
    from core.inference.tools import _fetch_url_raw
    global _crossref_next
    with _CROSSREF_LOCK:
        now = time.monotonic()
        if now < _crossref_next:
            return ""
        _crossref_next = now + 2.0
    url = "https://api.crossref.org/works?" + urlencode({
        "query.bibliographic": _sanitize_public_query(query), "rows": 8,
        "select": "DOI,title,URL,link,published",
    })
    if not check_url_access(url, policy)[0]:
        return ""
    metadata = {}
    deadline = time.monotonic() + min(timeout, 15)
    err, raw, _ = _fetch_url_raw(url, timeout=min(timeout, 15), deadline=deadline,
                                cancel_event=cancel, website_policy=policy,
                                raw_bytes_max=1024*1024, any_status=True, meta_out=metadata)
    if metadata.get("status") == 429:
        retry = next((value for key, value in metadata.get("headers", []) if key.casefold() == "retry-after"), "60")
        with _CROSSREF_LOCK:
            _crossref_next = max(_crossref_next, time.monotonic() + min(3600, max(60, int(retry) if str(retry).isdigit() else 60)))
        return ""
    if err or not isinstance(raw, bytes) or cancel.is_set():
        return ""
    try:
        items = json.loads(raw)["message"]["items"]
    except (ValueError, KeyError, TypeError):
        return ""
    found = []
    seen = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        titles = item.get("title")
        title = " ".join(str(titles[0] if isinstance(titles, list) and titles else "Scholarly publication").split())[:300]
        links = [item.get("URL")]
        for link in item.get("link") or []:
            if isinstance(link, dict) and link.get("content-type") == "application/pdf":
                links.append(link.get("URL"))
        for candidate in links:
            if not isinstance(candidate, str) or candidate in seen or not check_url_access(candidate, policy)[0]:
                continue
            seen.add(candidate)
            doi = " ".join(str(item.get("DOI") or "").split())[:200]
            found.append(f"Title: {title}\nURL: {candidate}\nSnippet: Crossref publication metadata; DOI {doi}. Full-text access not yet verified.")
    return "\n\n---\n\n".join(found)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if len(self.links) >= 400:
            return
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.current = [values["href"], values.get("title") or ""]
        elif tag == "meta" and (values.get("name") or values.get("property") or "").casefold() in {"citation_pdf_url", "og:pdf"} and values.get("content"):
            self.links.append((values["content"], "full text PDF"))
        elif tag == "img" and values.get("src") and values.get("alt"):
            self.links.append((values["src"], values["alt"]))

    def handle_data(self, data):
        if self.current is not None and len(self.current[1]) < 500:
            self.current[1] += " " + data[:500]

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            if len(self.links) < 400:
                self.links.append(tuple(self.current))
            self.current = None


def relevant_links(html, base, question, policy, limit=16):
    parser = _Links()
    try:
        parser.feed(html[:1_000_000])
        parser.close()
    except ValueError:
        return []
    keywords = words(question)
    results = []
    seen = set()
    for href, label in parser.links:
        try:
            url = urljoin(base, href.strip())
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
                continue
            key = canonical_source_url(url)
            if len(url) > 4096 or key in seen or not check_url_access(url, policy)[0]:
                continue
        except ValueError:
            continue
        label = " ".join(label.split())[:200]
        score = len(keywords & words(label + " " + parsed.path))
        if _DOCUMENT.search(url) and (score or "full text" in label.casefold() or "download" in label.casefold() or parsed.path.lower().endswith(".pdf")):
            score += 2
        if score <= 0:
            continue
        seen.add(key)
        results.append((score, url, label or url))
    results.sort(key=lambda item: -item[0])
    return [{"url": url, "title": label, "snippet": "Related link found in a fetched source."} for _, url, label in results[:limit]]


def retained_documents(project_id):
    if not project_id:
        return []
    from core.rag import account_db, store
    conn = account_db.get_connection()
    try:
        return store.list_documents(conn, store.project_scope(project_id))
    finally:
        conn.close()


class DiscoveryFrontier:
    """Bounded search frontier. Existing project sources are excluded only for source discovery."""
    def __init__(self, question, project_id, exhaustive, max_sources):
        self.question = question
        self.project_id = project_id
        self.exhaustive = exhaustive
        self.limit = max_sources
        self.docs = retained_documents(project_id)
        self.known_hashes = {str(doc.get("sha256")) for doc in self.docs if doc.get("status") != "failed" or doc.get("source_metadata")}
        for doc in self.docs:
            try:
                meta = json.loads(doc.get("source_metadata") or "{}")
                if meta.get("contentSha256"):
                    self.known_hashes.add(meta["contentSha256"])
            except (TypeError, ValueError):
                pass
        self.seen = set()
        self.hashes = set()
        self.pending = deque()
        self.pending_keys = set()
        self.count = 0
        self.attempts = 0
        self.skipped = 0
        self.successful_searches = 0

    def restore(self, sources):
        for source in sources:
            try:
                self.seen.add(canonical_source_url(source["url"]))
                if source.get("finalUrl"):
                    self.seen.add(canonical_source_url(source["finalUrl"]))
                if source.get("sha256"):
                    self.hashes.add(source["sha256"])
                self.count += 1
            except (KeyError, ValueError):
                pass

    def retained(self, url):
        return bool(retained_source(self.docs, url))

    def observe(self, candidates):
        for item in candidates:
            try:
                key = canonical_source_url(item["url"])
            except (KeyError, ValueError):
                continue
            if key in self.seen or key in self.pending_keys:
                continue
            if self.retained(item["url"]):
                self.seen.add(key)
                self.skipped += 1
                continue
            if len(self.pending) >= self.limit * 3:
                break
            self.pending_keys.add(key)
            self.pending.append(dict(item))

    def next_action(self, seed, explicit_urls, fetched_urls, used_queries):
        if self.count >= self.limit or self.attempts >= self.limit * 3:
            return None
        for url in explicit_urls:
            if url not in fetched_urls and canonical_source_url(url) not in self.seen and not self.retained(url):
                return {"action": "fetch", "title": "Read supplied source", "url": url}
        # Prioritize planned coverage, then investigate discovered links.
        if seed is not None:
            return seed
        minimum = 6 if self.exhaustive else 3
        if len(used_queries) < minimum:
            for step in fallback_plan(self.question, minimum)["steps"]:
                if step["query"] not in used_queries:
                    return {"action": "search", "title": step["title"], "query": step["query"]}
        while self.pending:
            item = self.pending.popleft()
            key = canonical_source_url(item["url"])
            self.pending_keys.discard(key)
            if key not in self.seen and item["url"] not in fetched_urls and not self.retained(item["url"]):
                return {"action": "fetch", "title": item["title"][:120], "url": item["url"], "source": item}
        return None

    def verify(self, candidates, policy, cancel, timeout):
        """Verify in bounded pairs, retaining actual acquisition status and content hashes."""
        from core.rag import config
        incoming = list(candidates)
        self.observe(incoming)
        priority = set()
        for item in incoming:
            try:
                priority.add(canonical_source_url(item["url"]))
            except (KeyError, ValueError):
                continue
        if priority:
            self.pending = deque(sorted(self.pending, key=lambda item: canonical_source_url(item["url"]) not in priority))
        deadline = time.monotonic() + max(5, timeout)
        verified = []
        batch_limit = 10 if self.exhaustive else 6
        candidates = []
        while self.pending and len(candidates) < batch_limit and self.count + len(candidates) < self.limit and self.attempts + len(candidates) < self.limit * 3:
            item = self.pending.popleft()
            key = canonical_source_url(item["url"])
            self.pending_keys.discard(key)
            if key in self.seen or not check_url_access(item["url"], policy)[0]:
                continue
            if self.retained(item["url"]):
                self.skipped += 1
                self.seen.add(key)
                continue
            self.seen.add(key)
            candidates.append(item)
        self.attempts += len(candidates)

        def acquire(item):
            if cancel.is_set() or time.monotonic() >= deadline:
                return None
            details = {}
            remaining = max(1, min(45, int(deadline - time.monotonic())))
            try:
                filename, payload = fetch_source_payload(
                    item["url"], item["title"], policy, config.MAX_UPLOAD_BYTES,
                    details=details, cancel_event=cancel, timeout=remaining,
                    skip_url=self.retained,
                )
            except (ValueError, OSError) as error:
                if details.get("skippedExisting"):
                    return {"skip": True}
                return {**item, "verification": "inaccessible", "error": str(error)[:300]}
            if cancel.is_set():
                return None
            sha = hashlib.sha256(payload).hexdigest()
            fmt = filename.rsplit(".", 1)[-1].upper()
            content_hash = details.get("contentSha256") or sha
            return {**item, "verification": "downloaded", "format": fmt,
                    "sha256": sha, "contentSha256": content_hash,
                    "finalUrl": details.get("finalUrl") or item["url"],
                    "links": relevant_links(details.get("html") or "", details.get("finalUrl") or item["url"],
                                            self.question, policy, 24 if self.exhaustive else 12),
                    "text": details.get("text") or ""}

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(contextvars.copy_context().run, acquire, item) for item in candidates]
            for future in futures:
                result = future.result()
                if result is None:
                    continue
                if result.get("skip"):
                    self.skipped += 1
                    continue
                final = result.get("finalUrl")
                hashes = {result[key] for key in ("sha256", "contentSha256") if result.get(key)}
                if result["verification"] != "inaccessible" and (hashes & (self.known_hashes | self.hashes) or (final and self.retained(final))):
                    self.skipped += 1
                    continue
                if final:
                    self.seen.add(canonical_source_url(final))
                self.hashes.update(hashes)
                self.observe(result.pop("links", []))
                self.count += 1
                verified.append(result)
        # Reuse the configured, already-loaded embedder; never load/download one
        # just to rank temporary candidates. Lexical ordering is the fallback.
        from core.rag import config, embeddings
        eligible = [item for item in verified if item.get("text")]
        keywords = words(self.question)
        scores = {item["url"]: len(keywords & words(item.get("title", "") + " " + item.get("text", "")[:8000]))
                  for item in verified}
        try:
            model = config.effective_embedding_model()
            if eligible and embeddings.backend_is_loaded(model):
                vectors, _ = embeddings.encode_with_identity(
                    [self.question] + [item["text"][:4000] for item in eligible],
                    model_name=model, normalize=True,
                )
                if len(vectors) == len(eligible) + 1:
                    for item, vector in zip(eligible, vectors[1:]):
                        cosine = sum(float(a) * float(b) for a, b in zip(vectors[0], vector))
                        scores[item["url"]] += max(0, cosine) * max(1, len(keywords))
        except Exception:
            pass
        verified.sort(key=lambda item: (item["verification"] == "inaccessible", -scores[item["url"]]))
        for item in verified:
            item.pop("text", None)
        return verified
