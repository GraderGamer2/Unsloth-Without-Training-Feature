#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Apply conservative native Vulkan optimizations to the pinned personal llama.cpp.

Do not alter memory placement, shader precision, subgroup selection or synchronization.
Fail closed if upstream source changes rather than guessing at a patch location.
"""
from pathlib import Path
import sys

ROOT = "ggml/src/ggml-vulkan/"
SHADERS = (
    "mul_mat_vec.comp",
    "mul_mat_vecq.comp",
    "mul_mat_vec_q2_k.comp",
    "mul_mat_vec_q3_k.comp",
    "mul_mat_vec_q4_k.comp",
    "mul_mat_vec_q5_k.comp",
    "mul_mat_vec_q6_k.comp",
    "mul_mat_vec_iq1_m.comp",
    "mul_mat_vec_iq1_s.comp",
    "mul_mat_vec_iq2_s.comp",
    "mul_mat_vec_iq2_xs.comp",
    "mul_mat_vec_iq2_xxs.comp",
    "mul_mat_vec_iq3_s.comp",
    "mul_mat_vec_iq3_xxs.comp",
    "mul_mat_vec_iq4_xs.comp",
    "mul_mat_vec_tq1_0.comp",
    "mul_mat_vec_tq2_0.comp",
)

QUEUE_BEFORE = "    VK_LOG_DEBUG(\"ggml_vk_submit(\" << ctx << \", \" << fence << \")\");\n\n    std::vector<std::vector<uint64_t>> tl_wait_vals;"
QUEUE_AFTER = "    VK_LOG_DEBUG(\"ggml_vk_submit(\" << ctx << \", \" << fence << \")\");\n\n    // A single command buffer without semaphore dependencies needs no heap-backed\n    // submit arrays or timeline-semaphore pNext chain. Keep the original queue\n    // wrapper, fence, submission order and all recorded memory barriers.\n    if (ctx->seqs.size() == 1 && ctx->seqs.front().size() == 1) {\n        const auto & submission = ctx->seqs.front().front();\n        if (submission.wait_semaphores.empty() && submission.signal_semaphores.empty()) {\n            const vk::SubmitInfo si{\n                0, nullptr, nullptr,\n                1, &submission.buffer->buf,\n                0, nullptr,\n            };\n            ctx->p->q->handle->submit({ si }, fence);\n            ctx->seqs.clear();\n            return;\n        }\n    }\n\n    std::vector<std::vector<uint64_t>> tl_wait_vals;"
HELPER_BEFORE = "layout (constant_id = 2) const uint NUM_COLS = 1;\n"
HELPER_AFTER = "layout (constant_id = 2) const uint NUM_COLS = 1;\n\n#ifdef MUL_MAT_ID\n// Every invocation in a workgroup reads the same expert id in get_offsets.\n// A masked slot already produces exact zero even with fused bias or scale.\n// Return uniformly before matrix/activation reads and reductions in that case.\nbool write_masked_expert(const uint first_row, const uint num_rows, const uint d_offset) {\n    if (!expert_skip) {\n        return false;\n    }\n    if (gl_LocalInvocationID.x == 0) {\n        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {\n            [[unroll]] for (uint n = 0; n < num_rows; ++n) {\n                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(0);\n            }\n        }\n    }\n    return true;\n}\n#endif\n"
OFFSET_BEFORE = "    get_offsets(a_offset, b_offset, d_offset);\n"
OFFSET_AFTER = "    get_offsets(a_offset, b_offset, d_offset);\n#ifdef MUL_MAT_ID\n    if (write_masked_expert(first_row, num_rows, d_offset)) {\n        return;\n    }\n#endif\n"
ROUND_BEFORE = "pack32(i8vec4(round(vals)))"
ROUND_AFTER = "pack32(i8vec4(vals))"


def replace_checked(text, old, new, count, path):
    # Some replacement blocks contain the original anchor; remove those first
    # when checking whether a previous build already applied the patch.
    if text.count(new) == count and old not in text.replace(new, ""):
        return text
    if text.count(old) != count or new in text:
        raise RuntimeError(f"Unexpected pinned Vulkan source in {path}")
    return text.replace(old, new)


def apply(source):
    specs = [
        ("ggml-vulkan.cpp", QUEUE_BEFORE, QUEUE_AFTER, 1),
        ("vulkan-shaders/mul_mat_vec_base.glsl", HELPER_BEFORE, HELPER_AFTER, 1),
        *[("vulkan-shaders/" + name, OFFSET_BEFORE, OFFSET_AFTER, 1)
          for name in SHADERS],
        # vals was rounded immediately before packing; round(round(x)) is
        # redundant for finite quantized inputs and both packing layouts.
        ("vulkan-shaders/quantize_q8_1.comp", ROUND_BEFORE, ROUND_AFTER, 2),
    ]
    changes = []
    for relative, old, new, count in specs:
        path = source / ROOT / relative
        original = path.read_text(encoding="utf-8")
        modified = replace_checked(original, old, new, count, path)
        if modified != original:
            changes.append((path, modified))
    # Validate every anchor before changing any source file.
    for path, modified in changes:
        path.write_text(modified, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} LLAMA_SOURCE_DIRECTORY")
    apply(Path(sys.argv[1]))
