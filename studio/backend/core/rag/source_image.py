# SPDX-License-Identifier: AGPL-3.0-only
"""Process a retained image only with existing local vision/OCR capabilities."""
from __future__ import annotations

import io
import warnings

from . import captioner, config, pdf_ocr, parsers


def image_pages(path):
    # Pillow is already part of the native image stack; no new dependency or model.
    from PIL import Image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.width * image.height > 32_000_000:
                    raise ValueError("Image exceeds the 32-megapixel safety limit")
                image.seek(0)
                image.thumbnail((2048, 2048))
                with io.BytesIO() as png:
                    image.convert("RGB").save(png, format="PNG")
                    data = png.getvalue()
    except Exception as exc:
        raise ValueError(f"Downloaded, not indexed: image could not be safely decoded ({exc}).") from exc
    endpoint = captioner.vision_endpoint()
    if endpoint is not None:
        text = captioner._caption_one(*endpoint, data, config.CAPTION_TIMEOUT_S)
        if text and text.strip():
            return parsers.parse_text("Visual model transcription/description:\n" + captioner._collapse_runaway(text))
    # The native OCR implementation checks whether Tesseract/language data is
    # available; it returns no text if unavailable. It never downloads anything.
    try:
        text, _ = pdf_ocr.ocr_image_blocks(path)
    except (ImportError, RuntimeError, OSError, ValueError):
        text = ""
    if text.strip():
        return parsers.parse_text("Local OCR transcription (not visual interpretation):\n" + text)
    raise ValueError("Downloaded, not indexed: no usable vision or local OCR text. Load a compatible vision model or configure local OCR, then retry this retained original.")
