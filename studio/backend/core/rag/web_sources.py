# SPDX-License-Identifier: AGPL-3.0-only
"""Persistent web evidence helpers. Never synthesize or index a generated report as evidence."""

from __future__ import annotations

import re
import threading
from urllib.parse import urldefrag, urlsplit

from core.inference.web_access_policy import check_url_access, normalize_website_policy

# Same search result envelope used by Deep Research. Body size is deliberately
# bounded because web tools are designed for model context, not full archives.
URL_BLOCK = re.compile(
    r"Title:\s*(?P<title>[^\n]*)\nURL:\s*(?P<url>https?://[^\s]+)\nSnippet:\s*(?P<snippet>.*?)(?=\n\n---|\Z)",
    re.DOTALL,
)
MAX_SOURCE_CHARS = 256_000


def canonical_source_url(url: str, policy: dict | None = None) -> str:
    url = url.strip()
    allowed, reason, _ = check_url_access(url, policy)
    if not allowed:
        raise ValueError(reason)
    # Remove only the fragment; paths, query parameters and scheme are not
    # interchangeable. Do not guess redirects or strip arbitrary tracking keys.
    return urldefrag(url)[0]


def intersect_policies(first: dict | None, second: dict | None) -> dict:
    a, b = normalize_website_policy(first), normalize_website_policy(second)
    left, right = a["allowedDomains"], b["allowedDomains"]
    if left and right:
        allowed = list(dict.fromkeys(
            narrower
            for x in left for y in right
            for narrower in ([x] if x == y or x.endswith("." + y) else [y] if y.endswith("." + x) else [])
        ))
        if not allowed:
            raise ValueError("The current and research website allowlists do not overlap.")
    else:
        allowed = left or right
    return normalize_website_policy({
        "allowedDomains": allowed,
        "blockedDomains": list(dict.fromkeys(a["blockedDomains"] + b["blockedDomains"])),
    })


def search_candidates(result: str, policy: dict | None = None, limit: int = 12) -> list[dict]:
    out = []
    if limit <= 0:
        return out
    seen = set()
    for match in URL_BLOCK.finditer(result):
        try:
            url = canonical_source_url(match.group("url"), policy)
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append({
            "url": url,
            "title": match.group("title").strip()[:200] or url,
            "snippet": match.group("snippet").strip()[:1200],
        })
        if len(out) >= limit:
            break
    return out


def snapshot_markdown(url: str, title: str, body: str) -> tuple[str, str]:
    # Keep the original source URL in the bytes, so distinct URLs remain
    # distinct even when a boilerplate body is identical. No timestamp in
    # the content: stable repeated imports reuse the existing content hash.
    title = " ".join(title.split())[:140] or urlsplit(url).netloc
    body = body.strip()
    if not body:
        raise ValueError("The website returned no readable text.")
    text = (
        f"Original URL: {url}\n\n"
        "Imported web text snapshot (extraction may be incomplete).\n\n"
        f"{body[:MAX_SOURCE_CHARS]}\n"
    )
    return title, text


def fetch_source_text(url: str, policy: dict | None = None) -> str:
    # Use precisely the existing SSRF-guarded fetch implementation, redirect
    # policy and response decoding used by Deep Research, not a new HTTP client.
    from core.inference.tools import execute_tool, is_unreadable_page, EMPTY_SEARCH_RESULTS
    from core.inference.tool_loop_controller import is_tool_error, strip_result_for_model

    result = execute_tool("web_search", {"url": url}, cancel_event = threading.Event(), timeout = 60, website_policy = policy)
    if not isinstance(result, str):
        raise ValueError("Website did not return readable text.")
    result = strip_result_for_model(result, "web_search").strip()
    if not result or result in EMPTY_SEARCH_RESULTS or is_tool_error(result) or is_unreadable_page(result):
        raise ValueError("Website is unavailable or returned no readable evidence.")
    return result[:MAX_SOURCE_CHARS]


def discover_sources(query: str, policy: dict | None = None) -> list[dict]:
    from core.inference.tools import execute_tool
    from core.inference.tool_loop_controller import is_tool_error, strip_result_for_model

    result = execute_tool("web_search", {"query": query}, cancel_event = threading.Event(), timeout = 60, website_policy = policy)
    if not isinstance(result, str):
        raise ValueError("Web search failed.")
    result = strip_result_for_model(result, "web_search")
    if is_tool_error(result):
        raise ValueError("Web search failed. Check the configured search provider.")
    return search_candidates(result, policy)
