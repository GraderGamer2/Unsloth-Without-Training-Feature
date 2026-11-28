# SPDX-License-Identifier: AGPL-3.0-only
"""Process a retained image only with existing local vision/OCR capabilities."""
from __future__ import annotations

import io
import warnings
import zipfile

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
        try:
            text = captioner._caption_one(*endpoint, data, config.CAPTION_TIMEOUT_S)
        except (OSError, RuntimeError, ValueError):
            text = None
        if text and text.strip():
            return parsers.parse_text("Visual model transcription/description:\n" + captioner._collapse_runaway(text))
    # The native OCR implementation checks whether Tesseract/language data is
    # available; it returns no text if unavailable. It never downloads anything.
    try:
        text, _ = pdf_ocr.ocr_image_blocks(data)
    except (ImportError, RuntimeError, OSError, ValueError):
        text = ""
    if text.strip():
        return parsers.parse_text("Local OCR transcription (not visual interpretation):\n" + text)
    raise ValueError("Downloaded, not indexed: no usable vision or local OCR text. Load a compatible vision model or configure local OCR, then retry this retained original.")


def embedded_visual_pages(path):
    """Bounded raster media captions/OCR, without extracting archive paths."""
    if not zipfile.is_zipfile(path):
        return [], ""
    limit = min(8, max(0, config.CAPTION_MAX_IMAGES))
    try:
        with zipfile.ZipFile(path) as archive:
            media = [member for member in archive.infolist()
                     if member.filename.casefold().startswith(("word/media/", "xl/media/", "ppt/media/", "pictures/"))
                     and member.filename.casefold().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"))]
            if not media:
                return [], ""
            pages = []
            processed = 0
            for member in media[:limit]:
                if member.file_size > 8 * 1024 * 1024 or member.flag_bits & 1:
                    continue
                try:
                    visual = image_pages(io.BytesIO(archive.read(member)))
                    for page in visual:
                        text = f"Embedded image {processed + 1} (page location not verified):\n{page.text}"
                        pages.append(parsers.Page(text=text, char_count=len(text)))
                    processed += 1
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile):
                    # Avoid repeatedly calling an unavailable OCR/vision engine.
                    break
            warning = "" if processed == len(media) else f"Visual extraction partial: {processed} of {len(media)} embedded raster images processed (local capability and safety limits apply)."
            return pages, warning
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile):
        return [], "Embedded visual content could not be processed; original file retained."
