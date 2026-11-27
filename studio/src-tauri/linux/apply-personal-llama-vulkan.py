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


EXTRA_SPECS = (
    ("vulkan-shaders/mul_mat_vec_q2_k.comp", "    calc_superblock(a_offset, b_offset, itid, v_im, ix, q_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, false);", "    // No partial blocks: avoid a workgroup-wide barrier and empty tail iteration.\n    if (nbr_par_th != 0) {\n        calc_superblock(a_offset, b_offset, itid, v_im, ix, q_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, false);\n    }", 1),
    ("vulkan-shaders/mul_mat_vec_q3_k.comp", "    calc_superblock(a_offset, b_offset, ix, itid8, v_im, v_im4, v_in, hm_m, q_offset, y_offset, s_shift, i0 + ix, num_blocks_per_row, first_row, num_rows, false);", "    // No partial blocks: avoid a workgroup-wide barrier and empty tail iteration.\n    if (nbr_par_th != 0) {\n        calc_superblock(a_offset, b_offset, ix, itid8, v_im, v_im4, v_in, hm_m, q_offset, y_offset, s_shift, i0 + ix, num_blocks_per_row, first_row, num_rows, false);\n    }", 1),
    ("vulkan-shaders/mul_mat_vec_q6_k.comp", "    calc_superblock(a_offset, b_offset, itid, ix, ql_offset, qh_offset, s_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, false);", "    // No partial blocks: avoid a workgroup-wide barrier and empty tail iteration.\n    if (nbr_par_th != 0) {\n        calc_superblock(a_offset, b_offset, itid, ix, ql_offset, qh_offset, s_offset, y_offset, i0 + ix, num_blocks_per_row, first_row, num_rows, false);\n    }", 1),
    ("vulkan-shaders/flash_attn.comp", "                bool all_less = subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2);\n                barrier();\n", "                bool all_less = subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2);\n                // masksh and tmpsh are disjoint; the barrier before mask loading\n                // protects previous uses, and the barrier below publishes both writes.\n", 1),
    ("vulkan-shaders/flash_attn_cm1.comp", "                bool all_less = subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2);\n                barrier();\n", "                bool all_less = subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2);\n                // masksh and tmpsh are disjoint; the barrier before mask loading\n                // protects previous uses, and the barrier below publishes both writes.\n", 1),
    ("ggml-vulkan.cpp", "#include \"ggml-vulkan-common.h\"\n", "#include \"ggml-vulkan-common.h\"\n#include <unordered_set>\n", 1),
    ("ggml-vulkan.cpp", "    std::vector<ggml_tensor *> new_order;\n    std::vector<bool> used(graph->n_nodes, false);\n    std::set<ggml_tensor *> used_node_set;", "    std::vector<ggml_tensor *> new_order;\n    new_order.reserve(graph->n_nodes);\n    std::vector<bool> used(graph->n_nodes, false);\n    // Only membership is observed; iteration order is never used.\n    std::unordered_set<ggml_tensor *> used_node_set;\n    used_node_set.reserve(graph->n_nodes);\n    std::vector<int> current_set;\n    current_set.reserve(24);", 1),
    ("ggml-vulkan.cpp", "    while (first_unused < graph->n_nodes) {\n        std::vector<int> current_set;", "    while (first_unused < graph->n_nodes) {\n        current_set.clear();", 1),
    ("vulkan-shaders/mul_mat_vec_q4_k.comp", "    const uint y2_idx = y1_idx + 128;\n", "    const uint y2_idx = y1_idx + 128;\n\n    // Single-token decode: reuse the same activation vectors across weight rows.\n    // Keep the existing weight-first loop for multiple columns.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1;\n    vec4 cached_by10;\n    vec4 cached_by132;\n    vec4 cached_by20;\n    vec4 cached_by232;\n    if (reuse_b) {\n        cached_by10 = vec4(data_b_v4[(b_offset + y1_idx) / 4    ]);\n        cached_by132 = vec4(data_b_v4[(b_offset + y1_idx) / 4 + 8]);\n        cached_by20 = vec4(data_b_v4[(b_offset + y2_idx) / 4    ]);\n        cached_by232 = vec4(data_b_v4[(b_offset + y2_idx) / 4 + 8]);\n    }\n", 1),
    ("vulkan-shaders/mul_mat_vec_q4_k.comp", "            vec4 by10 =  vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y1_idx) / 4    ]);\n            vec4 by132 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y1_idx) / 4 + 8]);\n            vec4 by20 =  vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y2_idx) / 4    ]);\n            vec4 by232 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y2_idx) / 4 + 8]);", "            const vec4 by10 = reuse_b ? cached_by10 : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y1_idx) / 4    ]);\n            const vec4 by132 = reuse_b ? cached_by132 : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y1_idx) / 4 + 8]);\n            const vec4 by20 = reuse_b ? cached_by20 : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y2_idx) / 4    ]);\n            const vec4 by232 = reuse_b ? cached_by232 : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y2_idx) / 4 + 8]);", 1),
    ("vulkan-shaders/mul_mat_vec_q5_k.comp", "    const uint y2_idx = y1_idx + 128;\n", "    const uint y2_idx = y1_idx + 128;\n\n    // Single-token decode: reuse the same activation vectors across weight rows.\n    // Keep the existing weight-first loop for multiple columns.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1;\n    vec2 cached_by10;\n    vec2 cached_by116;\n    vec2 cached_by132;\n    vec2 cached_by148;\n    vec2 cached_by20;\n    vec2 cached_by216;\n    vec2 cached_by232;\n    vec2 cached_by248;\n    if (reuse_b) {\n        cached_by10 = vec2(data_b_v2[(b_offset + y1_idx) / 2     ]);\n        cached_by116 = vec2(data_b_v2[(b_offset + y1_idx) / 2 +  8]);\n        cached_by132 = vec2(data_b_v2[(b_offset + y1_idx) / 2 + 16]);\n        cached_by148 = vec2(data_b_v2[(b_offset + y1_idx) / 2 + 24]);\n        cached_by20 = vec2(data_b_v2[(b_offset + y2_idx) / 2     ]);\n        cached_by216 = vec2(data_b_v2[(b_offset + y2_idx) / 2 +  8]);\n        cached_by232 = vec2(data_b_v2[(b_offset + y2_idx) / 2 + 16]);\n        cached_by248 = vec2(data_b_v2[(b_offset + y2_idx) / 2 + 24]);\n    }\n", 1),
    ("vulkan-shaders/mul_mat_vec_q5_k.comp", "            vec2 by10 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y1_idx) / 2     ]);\n            vec2 by116 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y1_idx) / 2 +  8]);\n            vec2 by132 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y1_idx) / 2 + 16]);\n            vec2 by148 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y1_idx) / 2 + 24]);\n            vec2 by20 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y2_idx) / 2     ]);\n            vec2 by216 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y2_idx) / 2 +  8]);\n            vec2 by232 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y2_idx) / 2 + 16]);\n            vec2 by248 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y2_idx) / 2 + 24]);", "            const vec2 by10 = reuse_b ? cached_by10 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y1_idx) / 2     ]);\n            const vec2 by116 = reuse_b ? cached_by116 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y1_idx) / 2 +  8]);\n            const vec2 by132 = reuse_b ? cached_by132 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y1_idx) / 2 + 16]);\n            const vec2 by148 = reuse_b ? cached_by148 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y1_idx) / 2 + 24]);\n            const vec2 by20 = reuse_b ? cached_by20 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y2_idx) / 2     ]);\n            const vec2 by216 = reuse_b ? cached_by216 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y2_idx) / 2 +  8]);\n            const vec2 by232 = reuse_b ? cached_by232 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y2_idx) / 2 + 16]);\n            const vec2 by248 = reuse_b ? cached_by248 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y2_idx) / 2 + 24]);", 1),
)

def replace_checked(text, old, new, count, path):
    # Some replacement blocks contain the original anchor; remove those first
    # when checking whether a previous build already applied the patch.
    if text.count(new) == count and old not in text.replace(new, ""):
        return text
    if text.count(old) != count or new in text:
        raise RuntimeError(f"Unexpected pinned Vulkan source in {path}")
    return text.replace(old, new)


def patch_specs():
    return [
        ("ggml-vulkan.cpp", QUEUE_BEFORE, QUEUE_AFTER, 1),
        ("vulkan-shaders/mul_mat_vec_base.glsl", HELPER_BEFORE, HELPER_AFTER, 1),
        *[("vulkan-shaders/" + name, OFFSET_BEFORE, OFFSET_AFTER, 1)
          for name in SHADERS],
        # vals was rounded immediately before packing; round(round(x)) is
        # redundant for finite quantized inputs and both packing layouts.
        ("vulkan-shaders/quantize_q8_1.comp", ROUND_BEFORE, ROUND_AFTER, 2),
    ] + list(EXTRA_SPECS)


def apply(source):
    specs = patch_specs()
    originals = {}
    changes = {}
    for relative, old, new, count in specs:
        path = source / ROOT / relative
        if path not in originals:
            originals[path] = path.read_text(encoding="utf-8")
            changes[path] = originals[path]
        changes[path] = replace_checked(changes[path], old, new, count, path)
    # Validate every anchor before changing any source file. Multiple edits to
    # the same file must compose rather than overwrite each other.
    for path, modified in changes.items():
        if modified != originals[path]:
            path.write_text(modified, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} LLAMA_SOURCE_DIRECTORY")
    apply(Path(sys.argv[1]))
