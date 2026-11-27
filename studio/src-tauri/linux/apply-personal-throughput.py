#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned hot-path changes. Preserve GPU allocation policy and numerical precision."""
from pathlib import Path
import sys

COMMON = [
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_k.comp",
    "    const uint y_idx = i * QUANT_K + y_offset;\n",
    "    const uint y_idx = i * QUANT_K + y_offset;\n\n    // Single-column decode only; inactive tail lanes must not fetch B.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec2 cached_b0;\n    vec2 cached_b16;\n    vec2 cached_b32;\n    vec2 cached_b48;\n    vec2 cached_b64;\n    vec2 cached_b80;\n    vec2 cached_b96;\n    vec2 cached_b112;\n    if (reuse_b && i < num_blocks_per_row) {\n        cached_b0 = vec2(data_b_v2[(b_offset + y_idx) / 2 +  0]);\n        cached_b16 = vec2(data_b_v2[(b_offset + y_idx) / 2 +  8]);\n        cached_b32 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 16]);\n        cached_b48 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 24]);\n        cached_b64 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 32]);\n        cached_b80 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 40]);\n        cached_b96 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 48]);\n        cached_b112 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 56]);\n    }\n"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_k.comp",
    "            vec2 b0 =   vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  0]);\n            vec2 b16 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  8]);\n            vec2 b32 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 16]);\n            vec2 b48 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 24]);\n            vec2 b64 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 32]);\n            vec2 b80 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 40]);\n            vec2 b96 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 48]);\n            vec2 b112 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 56]);",
    "            const vec2 b0 = reuse_b ? cached_b0 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  0]);\n            const vec2 b16 = reuse_b ? cached_b16 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  8]);\n            const vec2 b32 = reuse_b ? cached_b32 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 16]);\n            const vec2 b48 = reuse_b ? cached_b48 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 24]);\n            const vec2 b64 = reuse_b ? cached_b64 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 32]);\n            const vec2 b80 = reuse_b ? cached_b80 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 40]);\n            const vec2 b96 = reuse_b ? cached_b96 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 48]);\n            const vec2 b112 = reuse_b ? cached_b112 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 56]);"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q3_k.comp",
    "    const uint y_idx = i * QUANT_K + y_offset;\n",
    "    const uint y_idx = i * QUANT_K + y_offset;\n\n    // Single-column decode only; inactive tail lanes must not fetch B.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec2 cached_b0;\n    vec2 cached_b16;\n    vec2 cached_b32;\n    vec2 cached_b48;\n    vec2 cached_b64;\n    vec2 cached_b80;\n    vec2 cached_b96;\n    vec2 cached_b112;\n    if (reuse_b && i < num_blocks_per_row) {\n        cached_b0 = vec2(data_b_v2[(b_offset + y_idx) / 2 +  0]);\n        cached_b16 = vec2(data_b_v2[(b_offset + y_idx) / 2 +  8]);\n        cached_b32 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 16]);\n        cached_b48 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 24]);\n        cached_b64 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 32]);\n        cached_b80 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 40]);\n        cached_b96 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 48]);\n        cached_b112 = vec2(data_b_v2[(b_offset + y_idx) / 2 + 56]);\n    }\n"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q3_k.comp",
    "            vec2 b0 =   vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  0]);\n            vec2 b16 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  8]);\n            vec2 b32 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 16]);\n            vec2 b48 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 24]);\n            vec2 b64 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 32]);\n            vec2 b80 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 40]);\n            vec2 b96 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 48]);\n            vec2 b112 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 56]);",
    "            const vec2 b0 = reuse_b ? cached_b0 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  0]);\n            const vec2 b16 = reuse_b ? cached_b16 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  8]);\n            const vec2 b32 = reuse_b ? cached_b32 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 16]);\n            const vec2 b48 = reuse_b ? cached_b48 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 24]);\n            const vec2 b64 = reuse_b ? cached_b64 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 32]);\n            const vec2 b80 = reuse_b ? cached_b80 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 40]);\n            const vec2 b96 = reuse_b ? cached_b96 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 48]);\n            const vec2 b112 = reuse_b ? cached_b112 : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 56]);"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp",
    "    const uint y_idx = i * QUANT_K + y_offset;\n",
    "    const uint y_idx = i * QUANT_K + y_offset;\n\n    // Single-column decode only; inactive tail lanes must not fetch B.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec4 cached_by0;\n    vec4 cached_by32;\n    vec4 cached_by64;\n    vec4 cached_by96;\n    if (reuse_b && i < num_blocks_per_row) {\n        cached_by0 = vec4(data_b_v4[(b_offset + y_idx) / 4     ]);\n        cached_by32 = vec4(data_b_v4[(b_offset + y_idx) / 4 +  8]);\n        cached_by64 = vec4(data_b_v4[(b_offset + y_idx) / 4 + 16]);\n        cached_by96 = vec4(data_b_v4[(b_offset + y_idx) / 4 + 24]);\n    }\n"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q6_k.comp",
    "            vec4 by0  = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4     ]);\n            vec4 by32 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 +  8]);\n            vec4 by64 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 16]);\n            vec4 by96 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 24]);",
    "            const vec4 by0 = reuse_b ? cached_by0 : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4     ]);\n            const vec4 by32 = reuse_b ? cached_by32 : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 +  8]);\n            const vec4 by64 = reuse_b ? cached_by64 : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 16]);\n            const vec4 by96 = reuse_b ? cached_by96 : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 24]);"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/flash_attn_mask_opt.comp",
    "\n    [[unroll]] for (uint block_x = 0; block_x < 16; ++block_x) {\n        float min_v = FLT_MAX_OVER_2;",
    "\n    [[unroll]] for (uint block_x = 0; block_x < 16; ++block_x) {\n        // Empty trailing tiles have the same code as the original reduction.\n        if ((i0 * 16 + block_x) * Bc >= nem0) {\n            if (tid == 0) { result |= 1u << (2 * block_x); }\n            continue;\n        }\n        float min_v = FLT_MAX_OVER_2;"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/flash_attn_mask_opt.comp",
    "\n        [[unroll]] for (uint block_x = 0; block_x < 16; ++block_x) {\n            float min_v = FLT_MAX_OVER_2;",
    "\n        [[unroll]] for (uint block_x = 0; block_x < 16; ++block_x) {\n            // Empty trailing tiles have the same code as the original reduction.\n            if ((i0 * 16 + block_x) * Bc >= nem0) {\n                if (tid == 0) { result |= 1u << (2 * block_x); }\n                continue;\n            }\n            float min_v = FLT_MAX_OVER_2;"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/flash_attn.comp",
    "                if (gl_SubgroupInvocationID == 0) {\n                    tmpsh[gl_SubgroupID] = all_less ? NEG_FLT_MAX_OVER_2 : 0.0f;\n                }\n                barrier();\n                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {\n                    max_mask = max(max_mask, tmpsh[s]);\n                }",
    "                if (gl_NumSubgroups == 1) {\n                    // Vote covers the workgroup; retain the publication barrier.\n                    max_mask = max(max_mask, all_less ? NEG_FLT_MAX_OVER_2 : 0.0f);\n                    barrier();\n                } else {\n                    if (gl_SubgroupInvocationID == 0) {\n                        tmpsh[gl_SubgroupID] = all_less ? NEG_FLT_MAX_OVER_2 : 0.0f;\n                    }\n                    barrier();\n                    [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {\n                        max_mask = max(max_mask, tmpsh[s]);\n                    }\n                }"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp",
    "                if (gl_SubgroupInvocationID == 0) {\n                    tmpsh[gl_SubgroupID] = all_less ? NEG_FLT_MAX_OVER_2 : 0.0f;\n                }\n                barrier();\n                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {\n                    max_mask = max(max_mask, tmpsh[s]);\n                }",
    "                if (gl_NumSubgroups == 1) {\n                    // Vote covers the workgroup; retain the publication barrier.\n                    max_mask = max(max_mask, all_less ? NEG_FLT_MAX_OVER_2 : 0.0f);\n                    barrier();\n                } else {\n                    if (gl_SubgroupInvocationID == 0) {\n                        tmpsh[gl_SubgroupID] = all_less ? NEG_FLT_MAX_OVER_2 : 0.0f;\n                    }\n                    barrier();\n                    [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {\n                        max_mask = max(max_mask, tmpsh[s]);\n                    }\n                }"
  ]
]
COMMON += [
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_s.comp",
    "    const uint y_idx = i * QUANT_K + 16 * itid;\n",
    "    const uint y_idx = i * QUANT_K + 16 * itid;\n    // This activation slice is invariant across output rows.\n    // Single-column decode only, to bound additional register pressure.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec4 cached_b[4];\n    if (reuse_b) {\n        const uint base_b = (b_offset + y_idx) / 4;\n        [[unroll]] for (uint k = 0; k < 4; ++k) {\n            cached_b[k] = vec4(data_b_v4[base_b + k]);\n        }\n    }\n"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_s.comp",
    "                vec4 b0 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 0]);\n                vec4 b4 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 1]);",
    "                const vec4 b0 = reuse_b ? cached_b[2*l] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 0]);\n                const vec4 b4 = reuse_b ? cached_b[2*l + 1] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 1]);"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xs.comp",
    "    const uint y_idx = i * QUANT_K + 16 * itid;\n",
    "    const uint y_idx = i * QUANT_K + 16 * itid;\n    // This activation slice is invariant across output rows.\n    // Single-column decode only, to bound additional register pressure.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec4 cached_b[4];\n    if (reuse_b) {\n        const uint base_b = (b_offset + y_idx) / 4;\n        [[unroll]] for (uint k = 0; k < 4; ++k) {\n            cached_b[k] = vec4(data_b_v4[base_b + k]);\n        }\n    }\n"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xs.comp",
    "                const vec4 b0 = vec4(data_b_v4[b_idx + 0]);\n                const vec4 b4 = vec4(data_b_v4[b_idx + 1]);",
    "                const vec4 b0 = reuse_b ? cached_b[2*l] : vec4(data_b_v4[b_idx + 0]);\n                const vec4 b4 = reuse_b ? cached_b[2*l + 1] : vec4(data_b_v4[b_idx + 1]);"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xxs.comp",
    "    const uint y_idx = i * QUANT_K + 16 * itid;\n",
    "    const uint y_idx = i * QUANT_K + 16 * itid;\n    // This activation slice is invariant across output rows.\n    // Single-column decode only, to bound additional register pressure.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec4 cached_b[4];\n    if (reuse_b) {\n        const uint base_b = (b_offset + y_idx) / 4;\n        [[unroll]] for (uint k = 0; k < 4; ++k) {\n            cached_b[k] = vec4(data_b_v4[base_b + k]);\n        }\n    }\n"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq2_xxs.comp",
    "                const vec4 b0 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 0]);\n                const vec4 b4 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 1]);",
    "                const vec4 b0 = reuse_b ? cached_b[2*l] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 0]);\n                const vec4 b4 = reuse_b ? cached_b[2*l + 1] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 1]);"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_xxs.comp",
    "    const uint y_idx = i * QUANT_K + 16 * itid;\n",
    "    const uint y_idx = i * QUANT_K + 16 * itid;\n    // This activation slice is invariant across output rows.\n    // Single-column decode only, to bound additional register pressure.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec4 cached_b[4];\n    if (reuse_b) {\n        const uint base_b = (b_offset + y_idx) / 4;\n        [[unroll]] for (uint k = 0; k < 4; ++k) {\n            cached_b[k] = vec4(data_b_v4[base_b + k]);\n        }\n    }\n"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_xxs.comp",
    "                const vec4 b0 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 0]);\n                const vec4 b4 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 1]);",
    "                const vec4 b0 = reuse_b ? cached_b[2*l] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 0]);\n                const vec4 b4 = reuse_b ? cached_b[2*l + 1] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 1]);"
  ]
]
TERNARY = [
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq2_0.comp",
    "    const uint y_idx = i * QUANT_K + y_offset;\n",
    "    const uint y_idx = i * QUANT_K + y_offset;\n    // Guard inactive tail lanes exactly as the original weight and B reads do.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec2 cached_b[8];\n    if (reuse_b && i < num_blocks_per_row) {\n        const uint base_b = (b_offset + y_idx) / 2;\n        [[unroll]] for (uint k = 0; k < 8; ++k) {\n            cached_b[k] = vec2(data_b_v2[base_b + 8*k]);\n        }\n    }\n"
  ],
  [
    "src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq2_0.comp",
    "            vec2 b0 =   vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  0]);\n            vec2 b16 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  8]);\n            vec2 b32 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 16]);\n            vec2 b48 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 24]);\n            vec2 b64 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 32]);\n            vec2 b80 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 40]);\n            vec2 b96 =  vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 48]);\n            vec2 b112 = vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 56]);",
    "            const vec2 b0 = reuse_b ? cached_b[0] : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  0]);\n            const vec2 b16 = reuse_b ? cached_b[1] : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 +  8]);\n            const vec2 b32 = reuse_b ? cached_b[2] : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 16]);\n            const vec2 b48 = reuse_b ? cached_b[3] : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 24]);\n            const vec2 b64 = reuse_b ? cached_b[4] : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 32]);\n            const vec2 b80 = reuse_b ? cached_b[5] : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 40]);\n            const vec2 b96 = reuse_b ? cached_b[6] : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 48]);\n            const vec2 b112 = reuse_b ? cached_b[7] : vec2(data_b_v2[(j*p.batch_stride_b + b_offset + y_idx) / 2 + 56]);"
  ]
]
LLAMA = [
  [
    "src/llama-kv-cache.cpp",
    "    llama_kv_cache::slot_info_vec_t res;\n\n    struct state_t {",
    "    llama_kv_cache::slot_info_vec_t res;\n    res.reserve(ubatches.size());\n\n    struct state_t {"
  ],
  [
    "src/llama-kv-cache.cpp",
    "    std::vector<state_t> states;\n\n    bool success = true;",
    "    std::vector<state_t> states;\n    states.reserve(ubatches.size());\n\n    bool success = true;"
  ],
  [
    "src/llama-kv-cache.cpp",
    "        const auto sinfo_new = find_slot(ubatch, false);",
    "        auto sinfo_new = find_slot(ubatch, false);"
  ],
  [
    "src/llama-kv-cache.cpp",
    "            state_t state = { sinfo_new, v_heads, {} };\n\n            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {\n                auto & cells = v_cells[sinfo_new.strm[s]];\n\n                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));\n            }",
    "            state_t state = { std::move(sinfo_new), v_heads, {} };\n            state.v_cells.reserve(state.sinfo.n_stream());\n\n            for (uint32_t s = 0; s < state.sinfo.n_stream(); ++s) {\n                auto & cells = v_cells[state.sinfo.strm[s]];\n\n                state.v_cells.push_back(cells.cp(state.sinfo.idxs[s]));\n            }"
  ],
  [
    "src/llama-kv-cache.cpp",
    "        apply_ubatch(sinfo_new, ubatch);",
    "        apply_ubatch(res.back(), ubatch);"
  ],
  [
    "src/llama-kv-cache.cpp",
    "    const T mask_drop = llama_cast<T>(-INFINITY);\n\n    // the min position",
    "    const T mask_drop = llama_cast<T>(-INFINITY);\n    const bool reuse_mask = !alibi && n_tps > 1;\n\n    // the min position"
  ],
  [
    "src/llama-kv-cache.cpp",
    "            auto & idxs = seq_idxs[seq_id];\n\n            if (!alibi) {\n                if (seq_srct.find(seq_id) != seq_srct.end()) {\n                    const uint32_t srct = seq_srct[seq_id];\n\n                    const uint64_t idst_prev = n_kv*srct;\n\n                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);\n\n                    prev = true;\n                } else {\n                    idxs.clear();\n                    idxs.reserve(ubatch->n_tokens + n_swa + 32);\n\n                    seq_srct[seq_id] = i;\n                }\n            }",
    "            // A one-token stream has no later row to reuse.\n            std::vector<uint32_t> * idxs = nullptr;\n            if (reuse_mask) {\n                const auto inserted = seq_masks.try_emplace(seq_id, i);\n                auto & mask = inserted.first->second;\n                idxs = &mask.idxs;\n                if (!inserted.second) {\n                    const uint64_t idst_prev = n_kv*mask.src;\n                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);\n                    prev = true;\n                } else {\n                    idxs->reserve(ubatch->n_tokens + n_swa + 32);\n                }\n            }"
  ],
  [
    "src/llama-kv-cache.cpp",
    "                if (!alibi) {\n                    if (prev) {\n                        if (jj >= idxs.size()) {\n                            break;\n                        }\n\n                        j = idxs[jj];\n                    }\n                }",
    "                if (prev) {\n                    if (jj >= idxs->size()) {\n                        break;\n                    }\n\n                    j = (*idxs)[jj];\n                }"
  ],
  [
    "src/llama-kv-cache.cpp",
    "                if (!alibi) {\n                    if (!prev) {\n                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32\n                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {\n                            idxs.push_back(j);\n                        }\n                    }\n                }",
    "                if (reuse_mask && !prev) {\n                    // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32\n                    if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {\n                        idxs->push_back(j);\n                    }\n                }"
  ],
  [
    "common/sampling.cpp",
    "        T value = data[first];\n        first = (first + 1) % capacity;\n        sz--;",
    "        T value = data[first];\n        if (++first == capacity) { first = 0; }\n        sz--;"
  ],
  [
    "common/sampling.cpp",
    "        return data[(first + sz - i - 1) % capacity];",
    "        const size_t idx = first + sz - i - 1;\n        return data[idx >= capacity ? idx - capacity : idx];"
  ],
  [
    "src/llama-vocab.cpp",
    "                        text += fragment.raw_text.substr(fragment.offset, fragment.length);",
    "                        text.append(fragment.raw_text, fragment.offset, fragment.length);"
  ],
  [
    "ggml/src/ggml-vulkan/ggml-vulkan.cpp",
    "    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;\n\n    while (to_alloc > 0) {",
    "    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;\n    ctx->descriptor_sets.reserve(needed);\n\n    while (to_alloc > 0) {"
  ],
  [
    "ggml/src/ggml-vulkan/ggml-vulkan.cpp",
    "        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);\n        for (uint32_t i = 0; i < alloc_count; i++) {\n            layouts[i] = device->dsl;\n        }",
    "        std::vector<vk::DescriptorSetLayout> layouts(alloc_count, device->dsl);"
  ],
  [
    "ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp",
    "    std::vector<vk::BufferCopy> slices(1);\n    if (width == spitch && width == dpitch) {\n        // Only do single write if stride is equal\n        slices[0].srcOffset = offset;\n        slices[0].dstOffset = buf_offset;\n        slices[0].size = width * height;\n    } else {\n        slices.resize(height);\n        for (size_t i = 0; i < height; i++) {\n            slices[i].srcOffset = offset + i * spitch;\n            slices[i].dstOffset = buf_offset + i * dpitch;\n            slices[i].size = width;\n        }\n    }\n\n    if (buf != nullptr) {\n        // Memory is pinned, use as staging buffer\n        ggml_vk_sync_buffers(nullptr, subctx);\n        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);\n\n        return true;\n    }",
    "    if (buf != nullptr) {\n        // Build regions only when the pinned path is taken.\n        if (width == spitch && width == dpitch) {\n            const vk::BufferCopy slice{offset, buf_offset, width * height};\n            ggml_vk_sync_buffers(nullptr, subctx);\n            subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, {slice});\n        } else {\n            std::vector<vk::BufferCopy> slices(height);\n            for (size_t i = 0; i < height; i++) {\n                slices[i].srcOffset = offset + i * spitch;\n                slices[i].dstOffset = buf_offset + i * dpitch;\n                slices[i].size = width;\n            }\n            ggml_vk_sync_buffers(nullptr, subctx);\n            subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);\n        }\n        return true;\n    }"
  ],
  [
    "src/llama-kv-cache.cpp",
    "        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;\n        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;",
    "        struct seq_mask {\n            uint32_t src;\n            std::vector<uint32_t> idxs;\n            explicit seq_mask(uint32_t src_) : src(src_) {}\n        };\n        std::unordered_map<llama_seq_id, seq_mask> seq_masks;"
  ]
]


def patch_specs(component):
    if component not in ("llama", "diffusion", "whisper", "audio"):
        raise ValueError(component)
    root = "external/ggml/" if component == "audio" else "ggml/"
    return [(root + p, a, b) for p, a, b in COMMON + (TERNARY if component in ("llama", "diffusion") else [])] + (LLAMA if component == "llama" else [])

def apply(component, source):
    originals, changes = {}, {}
    for relative, old, new in patch_specs(component):
        path = source / relative
        if path not in originals:
            originals[path] = path.read_bytes()
            changes[path] = originals[path].decode("utf-8")
        if "\r\n" in changes[path]:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        current = changes[path]
        if current.count(new) == 1 and old not in current.replace(new, ""):
            continue
        if current.count(old) != 1 or new in current:
            raise RuntimeError(f"Unexpected pinned throughput source in {path}")
        changes[path] = current.replace(old, new)
    for path, content in changes.items():
        updated = content.encode("utf-8")
        if updated != originals[path]:
            path.write_bytes(updated)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ENGINE SOURCE_DIRECTORY")
    apply(sys.argv[1], Path(sys.argv[2]))
