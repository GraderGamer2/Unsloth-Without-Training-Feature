# SPDX-License-Identifier: AGPL-3.0-only
"""Import only real, recorded research sources into the ordinary document ingestion path."""
from __future__ import annotations

import hashlib
import io
import posixpath
import re
import threading
import time
import zipfile
from urllib.parse import unquote, unquote_plus, urldefrag, urlsplit, urlunsplit

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


# Keep acquisition limits separate from the number of sources in a project.
_MAX_REMOTE_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_SNAPSHOT_CHARS = 1_000_000
_ODF_MIME = {
    b"application/vnd.oasis.opendocument.text": ".odt",
    b"application/vnd.oasis.opendocument.spreadsheet": ".ods",
    b"application/vnd.oasis.opendocument.presentation": ".odp",
    b"application/vnd.oasis.opendocument.text-template": ".ott",
    b"application/vnd.oasis.opendocument.spreadsheet-template": ".ots",
    b"application/vnd.oasis.opendocument.presentation-template": ".otp",
    b"application/epub+zip": ".epub",
}
# The package structure, not a URL or declared MIME type, distinguishes these.
_OOXML_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml": ".docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml": ".dotx",
    "application/vnd.ms-word.document.macroenabled.main+xml": ".docm",
    "application/vnd.ms-word.template.macroenabledtemplate.main+xml": ".dotm",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml": ".xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml": ".xltx",
    "application/vnd.ms-excel.sheet.macroenabled.main+xml": ".xlsm",
    "application/vnd.ms-excel.template.macroenabled.main+xml": ".xltm",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml": ".pptx",
    "application/vnd.openxmlformats-officedocument.presentationml.template.main+xml": ".potx",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow.main+xml": ".ppsx",
    "application/vnd.ms-powerpoint.presentation.macroenabled.main+xml": ".pptm",
    "application/vnd.ms-powerpoint.template.macroenabled.main+xml": ".potm",
    "application/vnd.ms-powerpoint.slideshow.macroenabled.main+xml": ".ppsm",
}
_TEXT_MIME = {
    "text/csv": ".csv", "text/tab-separated-values": ".tsv",
    "application/json": ".json", "application/xml": ".xml", "text/xml": ".xml",
    "application/yaml": ".yaml", "text/yaml": ".yaml", "text/markdown": ".md",
    "text/plain": ".txt", "text/x-python": ".py",
}
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".py", ".js", ".ts",
    ".jsx", ".tsx", ".c", ".h", ".cpp", ".hpp", ".rs", ".go", ".java", ".sh",
    ".sql", ".rst", ".tex", ".bib", ".log", ".css",
}
_BINARY_SUFFIXES = set(_OOXML_TYPES.values()) | set(_ODF_MIME.values()) | {
    ".pdf", ".doc", ".xls", ".ppt", ".rtf",
}


def _archive_extension(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise ValueError("Document archive contains too many entries.")
            names = set()
            total = 0
            for member in members:
                name = member.filename.rstrip("/")
                if (not name or name.startswith("/") or "\\" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or name.casefold() in names or member.flag_bits & 1
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError("Document archive contains unsafe, duplicate or encrypted entries.")
                names.add(name.casefold())
                total += member.file_size
                if member.file_size > _MAX_MEMBER_BYTES or total > _MAX_ARCHIVE_BYTES:
                    raise ValueError("Expanded document archive exceeds the safe parsing limit.")
            if "mimetype" in names:
                name = next(item.filename for item in members if item.filename.casefold() == "mimetype")
                if archive.getinfo(name).file_size > 256:
                    raise ValueError("Invalid document media type.")
                ext = _ODF_MIME.get(archive.read(name).strip())
                if ext:
                    # Do not treat an arbitrary ZIP with a forged mimetype as an office file.
                    required = "meta-inf/container.xml" if ext == ".epub" else "content.xml"
                    if required not in names:
                        raise ValueError("Document archive is missing its required content.")
                    return ext
            if "[content_types].xml" in names:
                name = next(item.filename for item in members if item.filename.casefold() == "[content_types].xml")
                if archive.getinfo(name).file_size > 1024 * 1024:
                    raise ValueError("Document content-type catalog exceeds the safe parsing limit.")
                catalog = archive.read(name).lower()
                # Inspect the declared main-part type without executing or expanding XML.
                matches = {ext for mime, ext in _OOXML_TYPES.items() if
                           (b'"' + mime.encode() + b'"') in catalog or
                           (b"'" + mime.encode() + b"'") in catalog}
                if len(matches) == 1:
                    ext = matches.pop()
                    required = ("word/document.xml" if ext.startswith((".doc", ".dot"))
                                else "xl/workbook.xml" if ext.startswith(".xl")
                                else "ppt/presentation.xml")
                    if required in names:
                        return ext
                raise ValueError("Unsupported or malformed Office document archive.")
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ValueError("Invalid or encrypted document archive.") from exc
    raise ValueError("This archive is not a supported Office, OpenDocument or EPUB file.")


def _native_binary_extension(data):
    from core.inference.tools import _has_pdf_magic
    if _has_pdf_magic(data):
        return ".pdf"
    if data.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return _archive_extension(data)
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        from core.rag.cfb import CompoundFile, CompoundFileError
        try:
            document = CompoundFile(data)
            if document.exists("EncryptedPackage"):
                raise ValueError("The document is password protected.")
            found = [ext for ext, streams in (
                (".doc", ("WordDocument",)), (".xls", ("Workbook", "Book")),
                (".ppt", ("PowerPoint Document",)),
            ) if any(document.exists(stream) for stream in streams)]
            if len(found) == 1:
                return found[0]
        except CompoundFileError as exc:
            raise ValueError("Invalid legacy Office document.") from exc
        raise ValueError("Unsupported or ambiguous legacy Office document.")
    if data.lstrip().startswith(b"{\\rtf"):
        return ".rtf"
    return None


def _suffix(name):
    return posixpath.splitext(unquote(str(name or "")))[1].casefold()


IMAGE_EXTS = frozenset({".png", ".jpg", ".gif", ".webp", ".tif", ".bmp"})


def _image_extension(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    if data.startswith(b"BM"):
        return ".bmp"
    return None


def fetch_source_payload(url, title, policy, max_upload_bytes, *,
                         details=None, cancel_event=None, timeout=60, skip_url=None):
    """Preserve supported native bytes, otherwise extract a truthful web-text snapshot."""
    from importlib import import_module
    tools = import_module("core.inference.tools")
    details = {} if details is None else details
    cancel_event = cancel_event or threading.Event()
    details.update(sourceUrl=url, title=title, retrievedAt=int(time.time() * 1000))

    candidate = _pdf_candidate(url) or url
    allowed, reason, _ = check_url_access(candidate, policy)
    if not allowed:
        raise ValueError(reason)
    if skip_url is not None and skip_url(candidate):
        details["skippedExisting"] = True
        raise ValueError("Already retained in this project.")
    # Preserve the existing GitHub README/blob handling for repository pages.
    if _pdf_candidate(url) is None and urlsplit(url).hostname == "github.com":
        body = fetch_source_text(url, policy, cancel_event=cancel_event, timeout=timeout)
        name, markdown = snapshot(url, title, body)
        details.update(finalUrl=url, representation="web text snapshot",
                       contentSha256=hashlib.sha256(body.strip().encode("utf-8")).hexdigest(),
                       text=body[:12000])
        return name, markdown.encode("utf-8")
    cap = min(max_upload_bytes or _MAX_REMOTE_BYTES, _MAX_REMOTE_BYTES)
    meta = {}
    def redirect(target, method, has_body, headers):
        if skip_url is not None and skip_url(target):
            details["skippedExisting"] = True
            return "Redirect points to a source already retained in this project."
        return None

    error, data, content_type = tools._fetch_url_raw(
        candidate, timeout = timeout, deadline = time.monotonic() + timeout,
        cancel_event = cancel_event, website_policy = policy,
        raw_bytes_max = cap, meta_out = meta, on_redirect=redirect,
    )
    if error:
        raise ValueError(str(error)[:500])
    if not isinstance(data, bytes) or not data:
        raise ValueError("The source returned no readable content.")
    details.update(finalUrl=meta.get("url") or candidate,
                   contentType=content_type, originalFilename=meta.get("filename"),
                   representation="original file", contentSha256=hashlib.sha256(data).hexdigest())
    extension = _native_binary_extension(data) or _image_extension(data)
    if extension and (not _pdf_candidate(url) or extension == ".pdf"):
        return _source_filename(url, title, extension), data
    if _pdf_candidate(url):
        raise ValueError("The PDF link did not return a PDF file; it may require login or be blocked.")
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    filename_suffix = _suffix(meta.get("filename"))
    url_suffix = _suffix(urlsplit(meta.get("url") or url).path)
    if mime.startswith("image/"):
        raise ValueError("The image is not a supported PNG, JPEG, GIF, WebP, TIFF or BMP file.")
    if filename_suffix in _BINARY_SUFFIXES or url_suffix in _BINARY_SUFFIXES:
        raise ValueError("The download did not contain the requested document format; it may require login or be blocked.")
    if tools._has_binary_magic(data) or (mime != "application/octet-stream" and not tools._is_text_candidate_content_type(mime)):
        raise ValueError("The source returned an unsupported binary format.")
    from core.rag.parsers import _decode_text
    # Native text decoding supports BOMs, non-UTF-8 sources and HTML meta charsets.
    # Use the HTTP charset when valid, retaining the native decoder as a fallback.
    from codecs import lookup
    charset = meta.get("charset")
    text = None
    if charset:
        try:
            text = data.decode(lookup(charset).name)
        except (LookupError, ValueError, UnicodeError, TypeError):
            pass
    if text is None:
        text = _decode_text(data, html = "html" in mime or tools._looks_like_html(data[:4096].decode("latin-1")))
    if tools._looks_binary(text):
        raise ValueError("The source returned binary data disguised as text.")
    is_html = "html" in mime or tools._looks_like_html(text)
    if is_html:
        # The raw fetcher checks HTTP redirects; the ordinary page fetcher also
        # follows HTML meta refresh through the same SSRF/website-policy gate.
        if tools._meta_refresh_target(data, meta.get("url") or url):
            body = fetch_source_text(url, policy, cancel_event=cancel_event, timeout=timeout)
            name, markdown = snapshot(url, title, body)
            details.update(representation="web text snapshot", text=body[:12000],
                           contentSha256=hashlib.sha256(body.strip().encode("utf-8")).hexdigest())
            return name, markdown.encode("utf-8")
        body = tools._render_page(("html", meta.get("url") or url, text), _MAX_SNAPSHOT_CHARS)
        # Reject challenge/empty pages before retaining HTML. The native parser
        # extracts text without executing HTML or requesting embedded resources.
        snapshot(url, title, body)
        details.update(representation="original HTML", text=body[:12000],
                       contentSha256=hashlib.sha256(body.strip().encode("utf-8")).hexdigest(),
                       html=text[:1_000_000])
        return _source_filename(url, title, ".html"), data
    reason = _challenge_page_reason(text)
    if reason:
        raise ValueError(reason)
    # A real text attachment is saved byte-for-byte. A generic text response
    # without a filename stays a snapshot to retain its source URL.
    extension = next((ext for ext in (filename_suffix, url_suffix) if ext in _TEXT_SUFFIXES), None)
    if mime == "application/octet-stream" and not extension:
        raise ValueError("The source returned an unidentified binary download.")
    extension = extension or (_TEXT_MIME.get(mime) if mime != "text/plain" else None)
    details["text"] = text[:12000]
    if extension:
        if not text.strip():
            raise ValueError("The source contains no extractable text.")
        return _source_filename(url, title, extension), data
    name, markdown = snapshot(url, title, text)
    details.update(representation="web text snapshot",
                   contentSha256=hashlib.sha256(text.strip().encode("utf-8")).hexdigest())
    return name, markdown.encode("utf-8")


def fetch_source_text(url, policy, *, cancel_event=None, timeout=60):
    # Reuse the existing SSRF-safe fetch, redirect policy, document extraction,
    # timeout and content limits. Never index search snippets or an AI report.
    from core.inference.tools import execute_tool, is_unreadable_page, EMPTY_SEARCH_RESULTS
    from core.inference.tool_loop_controller import is_tool_error, strip_result_for_model

    result = execute_tool("web_search", {"url": url}, cancel_event = cancel_event or threading.Event(),
                          timeout = timeout, website_policy = policy)
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


# Only remove explicitly recognized tracking parameters; preserve signed URLs and
# all other raw query components (including order and encoding).
_TRACKING_KEYS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid"}


def canonical_source_url(url):
    parts = urlsplit(urldefrag(url)[0])
    query = []
    for field in parts.query.split("&") if parts.query else []:
        key = unquote_plus(field.partition("=")[0]).casefold()
        if key.startswith("utm_") or key in _TRACKING_KEYS:
            continue
        query.append(field)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", "&".join(query), ""))


def source_url_key(url):
    return hashlib.sha256(canonical_source_url(url).encode("utf-8")).hexdigest()[:10]


def retained_source(documents, url):
    # Existing imports encode a URL hash in the filename. Keep compatibility
    # with legacy exact URLs and arXiv abstract/PDF aliases.
    urls = [url, urldefrag(url)[0], canonical_source_url(url)]
    candidate = _pdf_candidate(url)
    if candidate:
        urls.extend([candidate, canonical_source_url(candidate)])
    parts = urlsplit(url)
    if parts.hostname == "arxiv.org" and candidate:
        identifier = urlsplit(candidate).path.rsplit("/", 1)[-1]
        urls.append("https://arxiv.org/abs/" + identifier)
    keys = {hashlib.sha256(value.encode("utf-8")).hexdigest()[:10] for value in urls}
    for doc in documents:
        try:
            import json
            metadata = json.loads(doc.get("source_metadata") or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except (TypeError, ValueError):
            metadata = {}
        # Failed research documents retain their original; retry them in place.
        if doc.get("status") not in {"completed", "pending", "running"} and not metadata:
            continue
        for alias in (metadata.get("sourceUrl"), metadata.get("finalUrl")):
            if isinstance(alias, str) and source_url_key(alias) in keys:
                return doc
        match = re.search(r"\[web-([0-9a-f]{10})\]\.[^.]+$", doc.get("filename", ""))
        if match and match.group(1) in keys:
            return doc
    return None


def _source_filename(url, title, extension):
    title = " ".join(title.split())[:120] or urlsplit(url).netloc
    suffix = source_url_key(url)
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
