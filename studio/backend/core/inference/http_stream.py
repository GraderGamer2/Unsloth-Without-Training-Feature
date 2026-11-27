# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""HTTP stream cleanup shared by local and external OpenAI transports."""

import re

# SSE lines end only at CRLF, LF or CR. str.splitlines() (and httpx's aiter_lines) also cuts at
# U+2028, U+2029 and U+0085, which JSON leaves unescaped inside a delta.
SSE_LINE_END = re.compile(r"\r\n|\r|\n")


def _split_complete_sse_lines(pending: str) -> tuple[list[str], str]:
    """Return complete SSE lines and the unfinished tail, holding a trailing CR."""
    # Most llama-server and provider frames use LF only. The literal split avoids
    # a regex scan and still preserves Unicode line separators in JSON deltas.
    if "\r" not in pending:
        lines = pending.split("\n")
        rest = lines.pop()
    else:
        # A trailing CR may be the first half of a CRLF.
        cut = len(pending) - pending.endswith("\r")
        lines = SSE_LINE_END.split(pending[:cut])
        rest = lines.pop() + pending[cut:]
    return lines, rest


async def aiter_sse_lines(response):
    """``response.aiter_lines()`` that splits only at SSE line ends."""
    if not hasattr(response, "aiter_text"):
        # Not an httpx response: its own lines are already whole.
        lines = response.aiter_lines()
        try:
            async for line in lines:
                yield line
        finally:
            await lines.aclose()
        return
    # Accumulate fragments without repeatedly copying a long, unfinished SSE line.
    parts: list[str] = []
    # Closed here, not by the finalizer in another task (see routes.inference passthrough).
    chunks = response.aiter_text()
    try:
        async for text in chunks:
            if not text:
                continue
            held_cr = bool(parts) and parts[-1].endswith("\r")
            if not held_cr and "\n" not in text and "\r" not in text:
                parts.append(text)
                continue
            if parts:
                parts.append(text)
                pending = "".join(parts)
                parts.clear()
            else:
                pending = text
            lines, rest = _split_complete_sse_lines(pending)
            if rest:
                parts.append(rest)
            for line in lines:
                yield line
    finally:
        await chunks.aclose()
    if parts:
        pending = "".join(parts)
        lines = SSE_LINE_END.split(pending)
        if lines[-1] == "":
            lines.pop()
        for line in lines:
            yield line


async def closing_response_lines(response):
    """Close the HTTP response before the line iterator, including cancellation."""
    lines = aiter_sse_lines(response)
    try:
        async for line in lines:
            yield line
    finally:
        await response.aclose()
        await lines.aclose()
