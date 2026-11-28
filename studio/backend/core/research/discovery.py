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
_ACADEMIC = re.compile(r"\b(?:academic|scholarly|peer.reviewed|literature review|journal articles?|research papers?|doi|systematic review|scientific studies|thesis|dissertation|meta.analysis|psychology|psychological|interpersonal|social dynamics|social behaviour|social behavior|human behaviour|human behavior|peer influence|jurnal|artikel ilmiah|skripsi|penelitian|psikologi)\b", re.I)
_TECH = re.compile(r"\b(?:software|code|coding|github|linux|windows|minecraft|performance|driver|vulkan|game|mod|library|framework|benchmark|optimization|api|compile|compiler|build)\b", re.I)
_DOCUMENT = re.compile(r"\.(?:pdf|docx?|xlsx?|pptx?|od[stp]|epub|rtf|csv|tsv|json|xml)(?:$|[?#])", re.I)
_IMAGE_URL = re.compile(r"\.(?:png|jpe?g|webp|gif|tiff?|bmp|svg|avif)(?:$|[?#])", re.I)
_VISUAL_REQUEST = re.compile(r"\b(?:images?|photos?|photographs?|pictures?|illustrations?|screenshots?|diagrams?|charts?|graphs?|infographics?|visual evidence|gambar|foto|grafik|diagram)\b", re.I)
_EVIDENCE_IMAGE = re.compile(r"\b(?:data chart|chart|graph|diagram|schematic|infographic|microscopy|scatter plot|histogram|heatmap|flowchart|grafik)\b", re.I)


def relevant_image(question, label, url):
    """Do not turn incidental article illustrations into independent research sources."""
    return (
        url in question
        or bool(_VISUAL_REQUEST.search(question))
        or bool(_EVIDENCE_IMAGE.search(label) and words(question) & words(label))
    )
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


def _crossref_search(query, question, policy, cancel, timeout):
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


def complementary_search(query, question, policy, cancel, timeout):
    if not _ACADEMIC.search(question) or cancel.is_set():
        return ""
    from core.research.academic_providers import openalex_search, europe_pmc_search, as_search_results
    try:
        crossref = _crossref_search(query, question, policy, cancel, min(8, timeout))
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError):
        crossref = ""
    if cancel.is_set():
        return crossref
    other = openalex_search(query, policy, cancel, min(8, timeout))
    if re.search(r"\b(?:medical|medicine|clinical|health|biology|biomedical|disease|pharmac|cancer|patient|vaccine|epidemiol)\w*\b", question, re.I):
        other.extend(europe_pmc_search(query, policy, cancel, min(8, timeout)))
    return "\n\n---\n\n".join(part for part in (crossref, as_search_results(other)) if part)


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
            self.current = [values["href"], values.get("title") or "", False]
        elif tag == "meta" and (values.get("name") or values.get("property") or "").casefold() in {"citation_pdf_url", "og:pdf"} and values.get("content"):
            self.links.append((values["content"], "full text PDF", False))
        elif tag == "img" and values.get("src") and values.get("alt"):
            self.links.append((values["src"], values["alt"], True))

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
    for href, label, is_image in parser.links:
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
        if (is_image or _IMAGE_URL.search(url)) and not relevant_image(question, label, url):
            continue
        score = len(keywords & words(label + " " + parsed.path))
        if _DOCUMENT.search(url) and (score or "full text" in label.casefold() or "download" in label.casefold() or parsed.path.lower().endswith(".pdf")):
            score += 2
        if score <= 0:
            continue
        # A single generic word is weak evidence of relevance for recursive links.
        # Explicit PDF/download links are already boosted above.
        if len(keywords) >= 4 and score < 2 and not (is_image or _IMAGE_URL.search(url)):
            continue
        seen.add(key)
        results.append((score, url, label or url))
    results.sort(key=lambda item: -item[0])
    return [{"url": url, "title": label, "snippet": "Related link found in a fetched source."} for _, url, label in results[:limit]]


def select_project_evidence(candidates, previous, limit):
    """Prefer source diversity without discarding the highest-ranked unique evidence."""
    if limit <= 0:
        return []
    used = {item.get("chunkId") for item in previous if item.get("chunkId")}
    seen_text = {" ".join(str(item.get("snippet") or item.get("text") or "").split()).casefold()[:1500]
                 for item in previous}
    counts = {}
    for item in previous:
        doc = item.get("documentId") or item.get("filename")
        counts[doc] = counts.get(doc, 0) + 1
    pool = []
    for item in candidates:
        key = item.get("chunkId")
        body = " ".join(str(item.get("text") or "").split()).casefold()[:1500]
        if not key or not body or key in used or body in seen_text:
            continue
        used.add(key)
        seen_text.add(body)
        pool.append(item)
    out = []
    while pool and len(out) < limit:
        # Keep the strongest initial hit; subsequently favour documents not
        # represented yet, preserving original relevance order on ties.
        index = 0 if not out and not previous else min(
            range(len(pool)),
            key=lambda i: (i + 4 * counts.get(pool[i].get("documentId") or pool[i].get("filename"), 0), i),
        )
        item = pool.pop(index)
        doc = item.get("documentId") or item.get("filename")
        counts[doc] = counts.get(doc, 0) + 1
        out.append(item)
    return out


def legacy_interstitial_documents(project_id):
    """Exclude known challenge pages from analysis without modifying retained originals."""
    from pathlib import Path
    from core.rag import account_db, store
    from core.rag.research_source_import import _raw_html_challenge_reason
    from utils.paths import rag_uploads_root

    root = Path(rag_uploads_root()).resolve()
    invalid = set()
    conn = account_db.get_connection()
    try:
        for doc in store.list_documents(conn, store.project_scope(project_id)):
            if not doc.get("source_metadata") or not str(doc.get("filename") or "").lower().endswith((".html", ".htm")):
                continue
            record = store.get_document(conn, doc["id"]) or {}
            if not record.get("stored_path"):
                continue
            try:
                path = Path(record["stored_path"]).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    continue
                with path.open("rb") as stream:
                    head = stream.read(65536).decode("utf-8", errors="replace")
                if _raw_html_challenge_reason(head):
                    invalid.add(doc["id"])
            except (OSError, ValueError, RuntimeError):
                continue
    finally:
        conn.close()
    return invalid


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
    def __init__(self, question, project_id, exhaustive, max_sources, run_id=None):
        self.question = question
        self.project_id = project_id
        self.run_id = run_id
        self.exhaustive = exhaustive
        self.limit = max_sources
        self.docs = []
        self.known_hashes = set()
        self.refresh_retained()
        self.seen = set()
        self.hashes = set()
        self.pending = deque()
        self.pending_keys = set()
        self.count = 0
        self.attempts = 0
        self.skipped = 0
        self.successful_searches = 0
        self.verified_count = 0
        self.domains = set()
        self.covered_words = set()
        try:
            self.question_words = words(_sanitize_public_query(question))
        except ValueError:
            self.question_words = set()
        self.dry_searches = 0
        self.followups = deque()
        self.followup_keys = set()
        self.query_vector = None
        self.query_identity = None
        self.fetch_turn = False
        self.stop_reason = ""

    def refresh_retained(self):
        self.docs = retained_documents(self.project_id)
        self.known_hashes = {str(doc.get("sha256")) for doc in self.docs if doc.get("status") != "failed" or doc.get("source_metadata")}
        for doc in self.docs:
            try:
                meta = json.loads(doc.get("source_metadata") or "{}")
                if isinstance(meta, dict) and meta.get("contentSha256"):
                    self.known_hashes.add(meta["contentSha256"])
            except (TypeError, ValueError):
                pass
    def restore(self, sources):
        for source in sources:
            try:
                self.seen.add(canonical_source_url(source["url"]))
                if source.get("finalUrl"):
                    self.seen.add(canonical_source_url(source["finalUrl"]))
                for field in ("sha256", "contentSha256"):
                    if source.get(field):
                        self.hashes.add(source[field])
                self.count += 1
                if source.get("verification") in {"verified", "downloaded", "unreadable"}:
                    self.verified_count += 1
                    self.domains.add(urlsplit(source.get("finalUrl") or source["url"]).hostname or "")
                    self.covered_words.update(self.question_words & words(source.get("title", "") + " " + (source.get("snippet") or "")))
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
            if _IMAGE_URL.search(item["url"]) and not relevant_image(self.question, item.get("title", ""), item["url"]):
                continue
            if self.retained(item["url"]):
                self.seen.add(key)
                self.skipped += 1
                continue
            if len(self.pending) >= self.limit * 3:
                break
            self.pending_keys.add(key)
            self.pending.append(dict(item))

    def _next_pending(self, fetched_urls):
        self.pending = deque(sorted(self.pending, key=lambda item: (
            -len(self.question_words & words(item.get("title", "") + " " + (item.get("snippet") or ""))),
            not bool(_DOCUMENT.search(item["url"])),
            urlsplit(item["url"]).hostname in self.domains,
        )))
        while self.pending:
            item = self.pending.popleft()
            key = canonical_source_url(item["url"])
            self.pending_keys.discard(key)
            if key in self.seen or item["url"] in fetched_urls:
                continue
            if self.retained(item["url"]):
                self.skipped += 1
                self.seen.add(key)
                continue
            return {"action": "fetch", "title": item["title"][:120], "url": item["url"], "source": item}
        return None

    def _follow_up(self, query):
        try:
            value = _sanitize_public_query(query)
        except ValueError:
            return
        if value not in self.followup_keys and len(self.followup_keys) < (24 if self.exhaustive else 8):
            self.followup_keys.add(value)
            self.followups.append(value)

    def record_batch(self, action, items):
        good = [item for item in items if item.get("verification") != "inaccessible"]
        for item in good:
            self.verified_count += 1
            self.domains.add(urlsplit(item.get("finalUrl") or item["url"]).hostname or "")
            self.covered_words.update(self.question_words & words(item.get("title", "") + " " + (item.get("snippet") or "")))
        if action == "search":
            self.dry_searches = 0 if good else self.dry_searches + 1
        if _ACADEMIC.search(self.question):
            for item in items:
                if item.get("verification") == "inaccessible":
                    title = item.get("title", "")
                    if title and not title.startswith(("http://", "https://")):
                        self._follow_up(title[:220] + " open access repository full text")
        missing = self.question_words - self.covered_words
        if missing and good:
            self._follow_up(self.question[:330] + " " + " ".join(sorted(missing)[:6]) + " evidence")

    def summary(self):
        return {"downloaded": self.verified_count, "skippedExisting": self.skipped,
                "distinctDomains": len(self.domains - {""}),
                "uncoveredTerms": sorted(self.question_words - self.covered_words)[:12],
                "stopReason": self.stop_reason}

    def next_action(self, seed, explicit_urls, fetched_urls, used_queries):
        if self.count >= self.limit or self.attempts >= self.limit * 3:
            self.stop_reason = "Source/verification budget reached"
            return None
        for url in explicit_urls:
            if url not in fetched_urls and canonical_source_url(url) not in self.seen:
                if self.retained(url):
                    self.seen.add(canonical_source_url(url))
                    self.skipped += 1
                    continue
                return {"action": "fetch", "title": "Read supplied source", "url": url}
        minimum = 6 if self.exhaustive else 3
        if len(used_queries) >= 2 and self.pending and self.fetch_turn:
            self.fetch_turn = False
            pending = self._next_pending(fetched_urls)
            if pending:
                return pending
        if seed is not None:
            self.fetch_turn = True
            return seed
        if len(used_queries) < minimum:
            for step in fallback_plan(self.question, minimum)["steps"]:
                if step["query"] not in used_queries:
                    self.fetch_turn = True
                    return {"action": "search", "title": step["title"], "query": step["query"]}
        pending = self._next_pending(fetched_urls)
        if pending:
            return pending
        if self.dry_searches >= (5 if self.exhaustive else 3):
            self.stop_reason = "Diminishing returns after planned searches and links"
            return None
        while self.followups:
            query = self.followups.popleft()
            if query not in used_queries:
                return {"action": "search", "title": "Investigate remaining coverage", "query": query}
        self.stop_reason = "Planned and discovered search paths exhausted"
        return None

    def verify(self, candidates, policy, cancel, timeout):
        """Verify in bounded pairs, retaining actual acquisition status and content hashes."""
        from core.rag import config
        incoming = list(candidates)
        self.refresh_retained()
        self.observe(incoming)
        priority = set()
        for item in incoming:
            try:
                priority.add(canonical_source_url(item["url"]))
            except (KeyError, ValueError):
                continue
        # Rank by the original topic, then diversify domains before consuming
        # new search results. Freshness alone must not displace a better pending result.
        self.pending = deque(sorted(self.pending, key=lambda item: (
            -len(self.question_words & words(item.get("title", "") + " " + (item.get("snippet") or ""))),
            urlsplit(item["url"]).hostname in self.domains,
            not bool(_DOCUMENT.search(item["url"])),
            canonical_source_url(item["url"]) not in priority,
        )))
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
                from core.research.academic_providers import alternatives_for_doi
                alternatives = alternatives_for_doi(item["url"], item["title"], policy, cancel, max(1, int(deadline-time.monotonic()))) if time.monotonic() < deadline else []
                return {**item, "verification": "inaccessible", "error": str(error)[:300], "links": alternatives}
            if cancel.is_set():
                return None
            sha = hashlib.sha256(payload).hexdigest()
            fmt = filename.rsplit(".", 1)[-1].upper()
            if fmt in {"PNG", "JPG", "JPEG", "WEBP", "GIF", "TIF", "TIFF", "BMP", "SVG", "AVIF"} and not relevant_image(self.question, item["title"], item["url"]):
                return {"filtered": True}
            content_hash = details.get("contentSha256") or sha
            from core.research.temporary_sources import remember
            cached = remember(self.project_id, self.run_id, item["url"], filename, payload, details, policy)
            extracted = details.get("text") or ""
            return {**item, "verification": "verified" if extracted else "downloaded", "format": fmt,
                    "cached": cached, "sha256": sha, "contentSha256": content_hash,
                    "snippet": extracted[:800] if extracted else item.get("snippet", ""),
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
                if result.get("filtered"):
                    continue
                if result.get("skip"):
                    self.skipped += 1
                    continue
                final = result.get("finalUrl")
                hashes = {result[key] for key in ("sha256", "contentSha256") if result.get(key)}
                if result["verification"] != "inaccessible" and (hashes & (self.known_hashes | self.hashes) or (final and self.retained(final))):
                    self.skipped += 1
                    if self.project_id and self.run_id:
                        from core.research.temporary_sources import cache, run_scope
                        cache.remove(run_scope(self.project_id, self.run_id), result['url'])
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
                texts = [item["text"][:4000] for item in eligible]
                identity = embeddings.embedding_identity(model)
                reuse = self.query_vector is not None and self.query_identity == identity
                vectors, served_identity = embeddings.encode_with_identity(
                    texts if reuse else [self.question] + texts, model_name=model, normalize=True,
                )
                if reuse and served_identity != self.query_identity:
                    reuse = False
                    vectors, served_identity = embeddings.encode_with_identity(
                        [self.question] + texts, model_name=model, normalize=True)
                if not reuse and len(vectors) == len(eligible) + 1:
                    self.query_vector = vectors[0]
                    self.query_identity = served_identity
                    vectors = vectors[1:]
                if self.query_vector is not None and len(vectors) == len(eligible) and self.query_identity == served_identity:
                    for item, vector in zip(eligible, vectors):
                        cosine = sum(float(a) * float(b) for a, b in zip(self.query_vector, vector))
                        scores[item["url"]] += max(0, cosine) * max(1, len(keywords))
        except Exception:
            pass
        verified.sort(key=lambda item: (item["verification"] == "inaccessible", -scores[item["url"]]))
        for item in verified:
            item.pop("text", None)
        return verified
