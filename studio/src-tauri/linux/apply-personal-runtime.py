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
    ("src/whisper.cpp", "    // Initialize a vector and copy data from C array to it.\n    std::vector<float> samples_padded;\n    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);\n    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);\n\n    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio\n    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);", "    // The worker reads only n_samples + stage_2_pad samples and synthesizes\n    // all trailing zeros itself. Keep the original logical padded length.\n    const size_t padded_size = n_samples + stage_1_pad + stage_2_pad * 2;\n    std::vector<float> samples_padded(n_samples + stage_2_pad);\n    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);"),
    ("src/whisper.cpp", "    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;", "    mel.n_len     = (padded_size - frame_size) / frame_step;"),
    ("src/whisper.cpp", "    // Dump log_mel_spectrogram\n    if (debug) {\n        std::ofstream outFile(\"log_mel_spectrogram.json\");\n        outFile << \"[\";\n        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {\n            outFile << mel.data[i] << \", \";\n        }\n        outFile << mel.data[mel.data.size() - 1] << \"]\";\n        outFile.close();\n    }", "    // The personal runtime never writes a diagnostic spectrogram file.\n    (void) debug;"),
    ("src/whisper.cpp", "        std::string str = text;\n        std::string pat = ", "        auto begin = text.cbegin();\n        std::string pat = "),
    ("src/whisper.cpp", "        while (std::regex_search(str, m, re)) {\n            for (auto x : m) {\n                words.push_back(x);\n            }\n            str = m.suffix();\n        }", "        while (std::regex_search(begin, text.cend(), m, re)) {\n            for (const auto & x : m) {\n                words.push_back(x);\n            }\n            begin = m.suffix().first;\n        }"),
    ("src/whisper.cpp", "#define WHISPER_LOG_INFO(...)  whisper_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)", "#define WHISPER_LOG_INFO(...)  ((void)0)"),
    ("src/whisper.cpp", "    // Otherwise fft_out are all zero\n    double sum = log10(1e-10);\n    for (; i < mel.n_len; i += n_threads) {\n        for (int j = 0; j < mel.n_mel; j++) {\n            mel.data[j * mel.n_len + i] = sum;\n        }\n    }", "    // The parent fills the contiguous silent tail after all FFT workers join."),
    ("src/whisper.cpp", "    // clamping and normalization\n    double mmax = -1e20;", "    // The workers computed every frame before this boundary. Fill the\n    // remaining frames row-wise instead of interleaving writes across threads.\n    const int samples_for_fft = n_samples + stage_2_pad;\n    const int silent_begin = std::min(samples_for_fft / frame_step + 1, mel.n_len);\n    const float silent_value = log10(1e-10);\n    for (int j = 0; j < mel.n_mel; ++j) {\n        auto row = mel.data.begin() + j * mel.n_len;\n        std::fill(row + silent_begin, row + mel.n_len, silent_value);\n    }\n\n    // clamping and normalization\n    double mmax = -1e20;"),
)
DIFFUSION = (
    ("src/tokenizers/tokenizer.cpp", "            text += piece + \" \";", "            text += piece;\n            text.push_back(' ');"),
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
    ("src/tokenizers/tokenizer.cpp", "        for (size_t i = 0; i < count; ++i) {\n            out_tokens.push_back(tokens[begin + i]);\n            if (use_weights) {\n                out_weights.push_back((*weights)[begin + i]);\n            }\n            if (use_mask) {\n                out_mask.push_back(1.0f);\n            }\n        }", "        if (count != 0) {\n            out_tokens.insert(out_tokens.end(), tokens.begin() + begin, tokens.begin() + begin + count);\n            if (use_weights) {\n                out_weights.insert(out_weights.end(), weights->begin() + begin, weights->begin() + begin + count);\n            }\n            if (use_mask) {\n                out_mask.insert(out_mask.end(), count, 1.0f);\n            }\n        }"),
    ("src/tokenizers/tokenizer.cpp", "        if (add_bos_token) {\n            out_tokens.push_back(BOS_TOKEN_ID);", "        // Build left padding before the payload instead of shifting all tokens later.\n        const size_t pad_count = final_length - base_length;\n        if (pad_left && pad_count != 0) {\n            out_tokens.insert(out_tokens.end(), pad_count, PAD_TOKEN_ID);\n            if (use_weights) {\n                out_weights.insert(out_weights.end(), pad_count, 1.0f);\n            }\n            if (use_mask) {\n                out_mask.insert(out_mask.end(), pad_count, 0.0f);\n            }\n        }\n\n        if (add_bos_token) {\n            out_tokens.push_back(BOS_TOKEN_ID);"),
    ("src/tokenizers/tokenizer.cpp", "        if (final_length > out_tokens.size()) {\n            const size_t pad_count = final_length - out_tokens.size();\n            if (pad_left) {\n                out_tokens.insert(out_tokens.begin(), pad_count, PAD_TOKEN_ID);\n\n                if (use_weights) {\n                    out_weights.insert(out_weights.begin(), pad_count, 1.0f);\n                }\n                if (use_mask) {\n                    out_mask.insert(out_mask.begin(), pad_count, 0.0f);\n                }\n            } else {\n                out_tokens.insert(out_tokens.end(), pad_count, PAD_TOKEN_ID);\n\n                if (use_weights) {\n                    out_weights.insert(out_weights.end(), pad_count, 1.0f);\n                }\n                if (use_mask) {\n                    out_mask.insert(out_mask.end(), pad_count, 0.0f);\n                }\n            }\n        }", "        if (!pad_left && pad_count != 0) {\n            out_tokens.insert(out_tokens.end(), pad_count, PAD_TOKEN_ID);\n            if (use_weights) {\n                out_weights.insert(out_weights.end(), pad_count, 1.0f);\n            }\n            if (use_mask) {\n                out_mask.insert(out_mask.end(), pad_count, 0.0f);\n            }\n        }"),
)
AUDIO = (
    ("src/community_models/echo_tts/tokenizer.cpp", "    std::string out = text;\n\n    replace_all(out, kEllipsis, \"...\");\n    replace_all(out, kRightSingleQuote, \"'\");\n    // Upstream applies the right-double-quote rewrite twice and never rewrites\n    // the *left* double quote (U+201C). Reproduced verbatim so token streams\n    // match the reference implementation; see inference.py::tokenizer_encode.\n    replace_all(out, kRightDoubleQuote, \"\\\"\");\n    replace_all(out, kRightDoubleQuote, \"\\\"\");\n    replace_all(out, \"\\n\", \" \");\n    replace_all(out, \":\", \",\");\n    replace_all(out, \";\", \",\");\n    replace_all(out, kEmDash, \", \");", "    // All replacements are nonempty ASCII, so none can create another\n    // multi-byte match or a character that a later replacement would rewrite.\n    // Preserve the reference's intentionally unchanged left double quote.\n    std::string out;\n    out.reserve(text.size());\n    const std::string_view input(text);\n    for (size_t i = 0; i < input.size();) {\n        const auto rest = input.substr(i);\n        if (rest.size() >= 3 && rest[0] == '\\xE2' && rest[1] == '\\x80') {\n            if (rest.substr(0, 3) == kEllipsis) {\n                out += \"...\";\n            } else if (rest.substr(0, 3) == kRightSingleQuote) {\n                out.push_back('\\'');\n            } else if (rest.substr(0, 3) == kRightDoubleQuote) {\n                out.push_back('\"');\n            } else if (rest.substr(0, 3) == kEmDash) {\n                out += \", \";\n            } else {\n                out.push_back(input[i++]);\n                continue;\n            }\n            i += 3;\n        } else {\n            const char ch = input[i++];\n            out.push_back(ch == '\\n' ? ' ' : (ch == ':' || ch == ';') ? ',' : ch);\n        }\n    }"),
    ("src/community_models/echo_tts/tokenizer.cpp", """    std::vector<int32_t> ids;
    ids.reserve(out.normalized_text.size() + 1);
    ids.push_back(0);  // BOS
    for (const char byte : out.normalized_text) {
        ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(byte)));
    }

    const auto encoded_length = static_cast<int64_t>(ids.size());""", """    // A byte tokenizer has exactly one id per byte plus BOS. Avoid the
    // intermediate vector, including work on bytes discarded by truncation.
    const auto encoded_length = static_cast<int64_t>(out.normalized_text.size()) + 1;"""),
    ("src/community_models/echo_tts/tokenizer.cpp", """    out.mask.assign(static_cast<size_t>(output_length), 0.0F);
    for (int64_t i = 0; i < length; ++i) {
        out.input_ids[static_cast<size_t>(i)] = ids[static_cast<size_t>(i)];
        out.mask[static_cast<size_t>(i)] = 1.0F;
    }""", """    // Reserve once and initialize each mask element exactly once.
    out.mask.reserve(static_cast<size_t>(output_length));
    out.mask.assign(static_cast<size_t>(length), 1.0F);
    out.mask.resize(static_cast<size_t>(output_length), 0.0F);
    // max_length is positive and the BOS id is already zero.
    for (int64_t i = 1; i < length; ++i) {
        out.input_ids[static_cast<size_t>(i)] =
            static_cast<int32_t>(static_cast<unsigned char>(out.normalized_text[static_cast<size_t>(i - 1)]));
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
