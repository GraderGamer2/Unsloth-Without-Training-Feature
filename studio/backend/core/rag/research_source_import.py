# SPDX-License-Identifier: AGPL-3.0-only
"""Import only real, recorded research sources into the ordinary document ingestion path."""
from __future__ import annotations

import hashlib
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
    if not result or result in EMPTY_SEARCH_RESULTS or is_tool_error(result) or is_unreadable_page(result):
        raise ValueError("The source is inaccessible or has no extractable text.")
    return result


def snapshot(url, title, body):
    body = body.strip()
    if not body:
        raise ValueError("No readable source content.")
    title = " ".join(title.split())[:120] or urlsplit(url).netloc
    suffix = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    # Stable bytes enable native content-hash deduplication on repeated imports.
    filename = f"{title} [web-{suffix}].md"
    content = f"Original URL: {url}\n\nImported source text (web extraction may be incomplete).\n\n{body}\n"
    return filename, content
