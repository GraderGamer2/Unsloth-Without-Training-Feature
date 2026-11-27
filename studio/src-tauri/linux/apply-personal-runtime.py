#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned source-only optimizations. Keep functional output, errors and device discovery."""
from pathlib import Path
import sys

DEBUG = "#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)"
DEBUG_OFF = "#define GGML_LOG_DEBUG(...) ((void)0)"
LLAMA = (
    ("common/log.h", "#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG,  __VA_ARGS__)", "#define LOG_DBG(...) ((void)0)"),
    ("common/log.h", "#define LOG_TRC(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_TRACE,  __VA_ARGS__)", "#define LOG_TRC(...) ((void)0)"),
    ("common/log.h", "#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)", "#define LOG_DBGV(verbosity, ...) ((void)0)"),
    ("common/log.h", "#define LOG_TRCV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_TRACE, verbosity, __VA_ARGS__)", "#define LOG_TRCV(verbosity, ...) ((void)0)"),
    ("common/sampling.cpp", "            first = (first + 1) % capacity;", "            if (++first == capacity) { first = 0; }"),
    ("common/sampling.cpp", "        pos = (pos + 1) % capacity;", "        if (++pos == capacity) { pos = 0; }"),
    ("common/sampling.cpp", """        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }""", """        if (sz != 0) {
            // At most two contiguous spans; no division per token.
            const size_t contiguous = std::min(sz, capacity - first);
            result.insert(result.end(), data.begin() + first, data.begin() + first + contiguous);
            result.insert(result.end(), data.begin(), data.begin() + (sz - contiguous));
        }"""),
)
WHISPER = (
    ("src/whisper.cpp", "#define WHISPER_LOG_INFO(...)  whisper_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)", "#define WHISPER_LOG_INFO(...)  ((void)0)"),
)
DIFFUSION = (
    ("src/tokenizers/tokenizer.cpp", """    std::regex pattern(R"( ,)");
    return std::regex_replace(text, pattern, ",");""", """    // Literal substitution; preserve non-overlapping regex semantics.
    if (text.find(" ,") == std::string::npos) {
        return text;
    }
    std::string result;
    result.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == ' ' && i + 1 < text.size() && text[i + 1] == ',') {
            continue;
        }
        result.push_back(text[i]);
    }
    return result;"""),
    ("src/tokenizers/tokenizer.cpp", """        while (offset < tokens.size()) {
            const size_t remaining = tokens.size() - offset;
            const size_t take      = std::min(tokens_per_chunk, remaining);

            std::vector<int> chunk_tokens;
            std::vector<float> chunk_weights;
            std::vector<float> chunk_mask;
""", """        // build_sequence clears its output; retain the allocated capacity per chunk.
        std::vector<int> chunk_tokens;
        std::vector<float> chunk_weights;
        std::vector<float> chunk_mask;
        while (offset < tokens.size()) {
            const size_t remaining = tokens.size() - offset;
            const size_t take      = std::min(tokens_per_chunk, remaining);
"""),
)
AUDIO = (
    ("src/community_models/echo_tts/tokenizer.cpp", """    std::vector<int32_t> ids;
    ids.reserve(out.normalized_text.size() + 1);
    ids.push_back(0);  // BOS
    for (const char byte : out.normalized_text) {
        ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(byte)));
    }

    const auto encoded_length = static_cast<int64_t>(ids.size());""", """    // A byte tokenizer has exactly one id per byte plus BOS. Avoid the
    // intermediate vector, including work on bytes discarded by truncation.
    const auto encoded_length = static_cast<int64_t>(out.normalized_text.size()) + 1;"""),
    ("src/community_models/echo_tts/tokenizer.cpp", """    for (int64_t i = 0; i < length; ++i) {
        out.input_ids[static_cast<size_t>(i)] = ids[static_cast<size_t>(i)];
        out.mask[static_cast<size_t>(i)] = 1.0F;
    }""", """    // max_length is positive and the BOS id is already zero.
    out.mask[0] = 1.0F;
    for (int64_t i = 1; i < length; ++i) {
        out.input_ids[static_cast<size_t>(i)] =
            static_cast<int32_t>(static_cast<unsigned char>(out.normalized_text[static_cast<size_t>(i - 1)]));
        out.mask[static_cast<size_t>(i)] = 1.0F;
    }"""),
)


def patch_specs(component):
    root = "external/ggml" if component == "audio" else "ggml"
    additional = {"llama": LLAMA, "whisper": WHISPER, "diffusion": DIFFUSION, "audio": AUDIO}
    return ((root + "/src/ggml-impl.h", DEBUG, DEBUG_OFF), *additional[component])


def apply(component, source):
    originals = {}
    changes = {}
    for relative, old, new in patch_specs(component):
        path = source / relative
        if path not in originals:
            originals[path] = path.read_bytes()
            changes[path] = originals[path].decode("utf-8")
        # Some upstream files use CRLF. Match in their existing newline style.
        if "\r\n" in changes[path]:
            old = old.replace("\n", "\r\n")
            new = new.replace("\n", "\r\n")
        text = changes[path]
        if text.count(new) == 1 and old not in text.replace(new, ""):
            continue
        if text.count(old) != 1 or new in text:
            raise RuntimeError(f"Unexpected pinned runtime source: {path}")
        changes[path] = text.replace(old, new, 1)
    for path, contents in changes.items():
        data = contents.encode("utf-8")
        if data != originals[path]:
            path.write_bytes(data)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ENGINE SOURCE_DIRECTORY")
    apply(sys.argv[1], Path(sys.argv[2]))
