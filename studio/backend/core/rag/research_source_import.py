# SPDX-License-Identifier: AGPL-3.0-only
"""Import only real, recorded research sources into the ordinary document ingestion path."""
from __future__ import annotations

import hashlib
import re
import threading
from urllib.parse import urldefrag, urlsplit

from core.inference.web_access_policy import check_url_access, normalize_website_policy

# Concurrency controls, not source-count quotas.
FETCH_SLOTS = threading.BoundedSemaphore(2)
INDEX_SLOTS = threading.BoundedSemaphore(2)


def resolve_source(run, requested_url, current_policy):
    candidate = next((s for s in run.get("sources", []) if s.get("url") == requested_url), None)
    if candidate is None:
        raise ValueError("URL is not a recorded research source.")
    original = normalize_website_policy((run.get("config") or {}).get("websitePolicy"))
    current = normalize_website_policy(current_policy)
    left, right = original["allowedDomains"], current["allowedDomains"]
    if left and right:
        allowed = list(dict.fromkeys(
            narrower for a in left for b in right
            for narrower in ([a] if a == b or a.endswith("." + b) else [b] if b.endswith("." + a) else [])
        ))
        if not allowed:
            raise ValueError("Research and current website allowlists do not overlap.")
    else:
        allowed = left or right
    policy = normalize_website_policy({
        "allowedDomains": allowed,
        "blockedDomains": list(dict.fromkeys(original["blockedDomains"] + current["blockedDomains"])),
    })
    ok, reason, _ = check_url_access(requested_url, policy)
    if not ok:
        raise ValueError(reason)
    url = urldefrag(requested_url)[0]
    return url, str(candidate.get("title") or url), policy


def _challenge_page_reason(text):
    """Recognize extracted challenge pages, including HTTP 200 interstitials."""
    lines = [line.strip().lstrip("#").strip().casefold()
             for line in text[:8192].splitlines() if line.strip()]
    first = lines[:8]
    body = "\n".join(lines)
    if (
        any(line in {"prove your humanity", "verify you are human", "verify that you are human",
                     "checking your browser", "just a moment...", "attention required! | cloudflare",
                     "access denied", "request blocked", "robot check"} for line in first)
        and (len(text) < 1500 or any(marker in body for marker in (
            "complete the challenge", "not for bots", "captcha", "cloudflare",
            "verify that you are human", "checking your browser", "automated access",
            "security verification", "enable javascript and cookies",
        )))
    ):
        return "The website returned a bot verification or access-denied page, not the source content."
    if "complete the challenge below" in body and "not for bots" in body:
        return "The website returned a bot verification page, not the source content."
    return None


def _pdf_candidate(url):
    """Only direct PDF links and well-defined arXiv abstract-to-PDF mappings."""
    parts = urlsplit(url)
    if parts.path.lower().endswith(".pdf"):
        return url
    if parts.hostname == "arxiv.org":
        match = re.fullmatch(r"/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)/?", parts.path)
        if match:
            return "https://arxiv.org/pdf/" + match[1]
    return None


def fetch_source_payload(url, title, policy, max_upload_bytes):
    """Return original PDF bytes or a validated web-text snapshot, using the native fetcher."""
    candidate = _pdf_candidate(url)
    if candidate is not None:
        from core.inference.tools import _fetch_url_raw, _has_pdf_magic
        import time

        allowed, reason, _ = check_url_access(candidate, policy)
        if not allowed:
            raise ValueError(reason)
        # The normal upload cap applies too; keep the in-memory download bounded
        # even when local uploads are configured to be unlimited.
        cap = min(max_upload_bytes or (32 * 1024 * 1024), 32 * 1024 * 1024)
        error, data, _ = _fetch_url_raw(
            candidate, timeout = 60, deadline = time.monotonic() + 60,
            cancel_event = threading.Event(), website_policy = policy,
            raw_bytes_max = cap,
        )
        if error:
            raise ValueError(str(error)[:500])
        if not isinstance(data, bytes) or not _has_pdf_magic(data):
            raise ValueError("The PDF link did not return a PDF file; it may require login or be blocked.")
        return _source_filename(url, title, ".pdf"), data
    name, markdown = snapshot(url, title, fetch_source_text(url, policy))
    return name, markdown.encode("utf-8")


def fetch_source_text(url, policy):
    # Reuse the existing SSRF-safe fetch, redirect policy, document extraction,
    # timeout and content limits. Never index search snippets or an AI report.
    from core.inference.tools import execute_tool, is_unreadable_page, EMPTY_SEARCH_RESULTS
    from core.inference.tool_loop_controller import is_tool_error, strip_result_for_model

    result = execute_tool("web_search", {"url": url}, cancel_event = threading.Event(),
                          timeout = 60, website_policy = policy)
    if not isinstance(result, str):
        raise ValueError("No readable source content.")
    result = strip_result_for_model(result, "web_search").strip()
    if is_tool_error(result):
        raise ValueError(result[:500])
    if not result or result in EMPTY_SEARCH_RESULTS or is_unreadable_page(result):
        raise ValueError(result[:300] if result and is_unreadable_page(result)
                         else "The source has no extractable text.")
    reason = _challenge_page_reason(result)
    if reason:
        raise ValueError(reason)
    return result


def _source_filename(url, title, extension):
    title = " ".join(title.split())[:120] or urlsplit(url).netloc
    suffix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{title} [web-{suffix}]{extension}"


def snapshot(url, title, body):
    body = body.strip()
    if not body:
        raise ValueError("No readable source content.")
    reason = _challenge_page_reason(body)
    if reason:
        raise ValueError(reason)
    # Stable bytes enable native content-hash deduplication on repeated imports.
    filename = _source_filename(url, title, ".md")
    content = f"Original URL: {url}\n\nImported source text (web extraction may be incomplete).\n\n{body}\n"
    return filename, content
