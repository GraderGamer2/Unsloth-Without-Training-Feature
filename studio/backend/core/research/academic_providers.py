# SPDX-License-Identifier: AGPL-3.0-only
"""Complementary public scholarly discovery. No automatic credentials or paid API."""
import json
from email.utils import parsedate_to_datetime
import os
import re
import threading
import time
from urllib.parse import quote, urlencode, urlsplit, unquote
from core.inference.web_access_policy import check_url_access
from core.research.redaction import _sanitize_public_query

_LOCK = threading.Lock()
_NEXT = {}
_DOI = re.compile(r"^10\.\d{4,9}/[^\s?#]+$", re.I)


def _get(provider, url, policy, cancel, timeout):
    if cancel.is_set() or not check_url_access(url, policy)[0]:
        return {}
    with _LOCK:
        now = time.monotonic()
        if now < _NEXT.get(provider, 0):
            return {}
        _NEXT[provider] = now + 3
    from core.inference.tools import _fetch_url_raw
    meta = {}
    seconds = max(1, min(8, timeout))
    try:
        error, data, _ = _fetch_url_raw(url, timeout=seconds, deadline=time.monotonic()+seconds,
            cancel_event=cancel, website_policy=policy, raw_bytes_max=2*1024*1024,
            any_status=True, meta_out=meta)
        if meta.get("status") in (401, 403, 429):
            with _LOCK:
                delay = 120 if meta["status"] == 429 else 3600
                headers = meta.get("headers") or ()
                if isinstance(headers, dict):
                    headers = headers.items()
                for name, value in headers:
                    if str(name).casefold() == "retry-after":
                        try:
                            delay = max(delay, int(value))
                        except (TypeError, ValueError):
                            try:
                                delay = max(delay, parsedate_to_datetime(value).timestamp() - time.time())
                            except (TypeError, ValueError, OverflowError):
                                pass
                _NEXT[provider] = time.monotonic() + delay
            return {}
        if error or cancel.is_set() or not isinstance(data, bytes):
            return {}
        result = json.loads(data)
        return result if isinstance(result, dict) else {}
    except (ValueError, OSError, RuntimeError):
        return {}


def _entry(title, url, snippet, policy):
    if not isinstance(url, str) or len(url)>4096 or not check_url_access(url, policy)[0]:
        return None
    return {"title":" ".join(str(title or "Scholarly work").split())[:300],
            "url":url,"snippet":snippet}


def _unique(entries):
    seen = set()
    out = []
    for item in entries:
        if item and item["url"] not in seen:
            seen.add(item["url"])
            out.append(item)
    return out


def _locations(work, policy):
    if not isinstance(work, dict):
        return []
    title = work.get("display_name") or work.get("title")
    location = work.get("best_oa_location")
    out = []
    if isinstance(location, dict) and location.get("is_oa") is True:
        for key in ("pdf_url", "landing_page_url"):
            out.append(_entry(title, location.get(key), "OpenAlex open-access location; full text not yet verified.", policy))
    out.append(_entry(title, work.get("doi"), "OpenAlex publication metadata; full text not yet verified.", policy))
    return _unique(out)


def openalex_search(query, policy, cancel, timeout):
    url = "https://api.openalex.org/works?" + urlencode({
        "search":_sanitize_public_query(query), "per_page":8,
        "select":"display_name,doi,best_oa_location,publication_year"})
    works = _get("openalex", url, policy, cancel, timeout).get("results")
    return _unique(item for work in works[:8] for item in _locations(work, policy)) if isinstance(works, list) else []


def europe_pmc_search(query, policy, cancel, timeout):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode({
        "query":_sanitize_public_query(query), "resultType":"core", "format":"json", "pageSize":8})
    container = _get("europe-pmc", url, policy, cancel, timeout).get("resultList")
    works = container.get("result") if isinstance(container, dict) else None
    out = []
    for work in works[:8] if isinstance(works, list) else []:
        if not isinstance(work, dict):
            continue
        links = work.get("fullTextUrlList")
        items = links.get("fullTextUrl") if isinstance(links, dict) else None
        for loc in items[:8] if isinstance(items, list) else []:
            if isinstance(loc, dict) and loc.get("availabilityCode") == "OA":
                out.append(_entry(work.get("title"), loc.get("url"), "Europe PMC open-access location; full text not yet verified.", policy))
        doi = work.get("doi")
        if isinstance(doi,str) and _DOI.fullmatch(doi):
            out.append(_entry(work.get("title"), "https://doi.org/"+quote(doi,safe="/"), "Europe PMC publication metadata.",policy))
    return _unique(out)


def alternatives_for_doi(url, title, policy, cancel, timeout):
    try:
        parts = urlsplit(url)
        doi = unquote(parts.path.lstrip("/"))
        if parts.hostname not in {"doi.org","dx.doi.org"} or not _DOI.fullmatch(doi):
            return []
    except (TypeError, ValueError):
        return []
    out = _locations(_get("openalex","https://api.openalex.org/works/doi:"+quote(doi,safe="/"),policy,cancel,timeout),policy)
    # Unpaywall requires a real contact email. Never infer or invent it.
    email = os.environ.get("UNSLOTH_UNPAYWALL_EMAIL","").strip()
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",email):
        data = _get("unpaywall","https://api.unpaywall.org/v2/"+quote(doi,safe="/")+"?"+urlencode({"email":email}),policy,cancel,timeout)
        loc = data.get("best_oa_location")
        if isinstance(loc,dict):
            for key in ("url_for_pdf","url_for_landing_page"):
                out.append(_entry(title,loc.get(key),"Unpaywall open-access location; full text not yet verified.",policy))
    return [item for item in _unique(out) if item["url"] != url]


def as_search_results(entries):
    return "\n\n---\n\n".join(f"Title: {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}" for item in _unique(entries))
