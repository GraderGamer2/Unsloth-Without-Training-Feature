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


# Pinned llama.cpp batch, history and speculative sampling hot paths.
LLAMA += [
  [
    "src/llama-batch.cpp",
    "    std::vector<seq_set_t> cur_seq_set;\n\n    llama_seq_id last_seq_id = -1;",
    "    std::vector<seq_set_t> cur_seq_set;\n    seq_set_t occupied_seqs{};\n\n    llama_seq_id last_seq_id = -1;"
  ],
  [
    "src/llama-batch.cpp",
    "        bool add = true;\n\n        for (uint32_t s = 0; s < cur_seq_set.size(); ++s) {\n            // no overlap with existing sequence sets:\n            if (!(cur_seq_set[s] & seq_set[i]).none()) {\n                add = false;\n                break;\n            }\n        }",
    "        // Disjoint from every selected set iff disjoint from their union.\n        // This replaces the per-candidate scan with one fixed-size bitset test.\n        bool add = (occupied_seqs & seq_set[i]).none();"
  ],
  [
    "src/llama-batch.cpp",
    "            cur_seq_set.push_back(seq_set[i]);\n\n            last_seq_id = batch.seq_id[i][0];",
    "            cur_seq_set.push_back(seq_set[i]);\n            occupied_seqs |= seq_set[i];\n\n            last_seq_id = batch.seq_id[i][0];"
  ],
  [
    "src/llama-kv-cache.cpp",
    "        const llama_seq_id seq_id = ubatch.seq_id[i][0];\n\n        for (uint32_t j = 0; j < n; ++j) {\n            const llama_pos d = (llama_pos) (n - j);\n\n            llama_pos p;\n            if (by_order) {\n                const auto & v = seq_idx[seq_id];",
    "        const llama_seq_id seq_id = ubatch.seq_id[i][0];\n        // The index is invariant across all n predecessors of this token.\n        const auto * indices = by_order ? &seq_idx.at(seq_id) : nullptr;\n\n        for (uint32_t j = 0; j < n; ++j) {\n            const llama_pos d = (llama_pos) (n - j);\n\n            llama_pos p;\n            if (by_order) {\n                const auto & v = *indices;"
  ],
  [
    "common/sampling.cpp",
    "        // rejected: tokens outside q's support keep all of p\n        residual.clear();\n        float sum = 0.0f;\n        for (size_t k = 0; k < cur_p->size; ++k) {\n            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);",
    "        // Rejection is the only path that needs all draft probabilities.\n        // For large supports index q once instead of scanning it for every\n        // target candidate. emplace preserves prob_of's first-match semantics.\n        std::unordered_map<llama_token, float> q_probs;\n        if (q.size() >= 32 && cur_p->size >= 32) {\n            q_probs.reserve(q.size());\n            for (const auto & e : q) {\n                q_probs.emplace(e.id, e.p);\n            }\n        }\n        const auto q_probability = [&](llama_token id) {\n            if (q_probs.empty()) {\n                return prob_of(q.data(), q.size(), id);\n            }\n            const auto it = q_probs.find(id);\n            return it == q_probs.end() ? 0.0f : it->second;\n        };\n\n        // rejected: tokens outside q's support keep all of p\n        residual.clear();\n        float sum = 0.0f;\n        for (size_t k = 0; k < cur_p->size; ++k) {\n            const float r = p_of(k) - q_probability(cur_p->data[k].id);"
  ]
]


# Further pinned per-engine optimizations, separate from shared ggml patches.
LLAMA += [
  [
    "tools/server/server-context.cpp",
    "                const int n_draft_max = slot.get_n_draft_max();\n\n                if (n_draft_max > 0) {",
    "                int n_draft_max = slot.get_n_draft_max();\n\n                // Limit only fresh drafts on drafters that honor\n                // the cap before decoding. Replays and synthetic tests retain\n                // the original behavior. Probe the full length every 8 rounds.\n                const auto & types = params_base.speculative.types;\n                if (n_draft_max > 0 && slot.spec_draft.empty() &&\n                    types.size() == 1 &&\n                    (types.front() == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE ||\n                     types.front() == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||\n                     types.front() == COMMON_SPECULATIVE_TYPE_DRAFT_MTP ||\n                     types.front() == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) &&\n                    common_speculative_get_synth_probs(spec.get()).empty() &&\n                    slot.stats.n_draft_tokens >= 64 &&\n                    slot.stats.n_draft_verif_steps >= 8 &&\n                    slot.stats.n_draft_verif_steps % 8 != 0 &&\n                    int64_t(slot.stats.n_draft_accepted) * 5 < slot.stats.n_draft_tokens) {\n                    n_draft_max = std::min(n_draft_max, std::max(2, params_base.speculative.draft.n_min));\n                }\n\n                if (n_draft_max > 0) {"
  ],
  [
    "common/speculative.cpp",
    "                if (params.n_max <= (int) result.size()) {\n                    drafting[seq_id] = false;\n                    n_drafting--;\n                    continue;\n                }\n\n                const int32_t idx = batch.add(id, pending_pos_last[seq_id] + (i + 1), seq_id, true);",
    "                // Respect the server's cap before another draft decode.\n                // Below n_min, keep the original generate-then-truncate behavior.\n                if (params.n_max <= (int) result.size() ||\n                    (dp.n_max > 0 && dp.n_max >= params.n_min && dp.n_max <= (int) result.size())) {\n                    drafting[seq_id] = false;\n                    n_drafting--;\n                    continue;\n                }\n\n                const int32_t idx = batch.add(id, pending_pos_last[seq_id] + (i + 1), seq_id, true);"
  ],
  [
    "common/speculative.cpp",
    "                if (params.n_max <= (int) result.size()) {\n                    drafting[seq_id] = false;\n                    n_drafting--;\n                    continue;\n                }\n\n                if (chain_heads) {",
    "                // Avoid producing draft tokens the server will discard.\n                if (params.n_max <= (int) result.size() ||\n                    (dp.n_max > 0 && dp.n_max >= params.n_min && dp.n_max <= (int) result.size())) {\n                    drafting[seq_id] = false;\n                    n_drafting--;\n                    continue;\n                }\n\n                if (chain_heads) {"
  ],
  [
    "tools/server/server-context.cpp",
    "                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));\n\n                GGML_ASSERT(spec_i_batch.size() == n_draft + 1);",
    "                // The saved sampler is only read when a rejected draft needs\n                // checkpoint replay. With removable KV, or enough recurrent\n                // rollback slots for the entire draft, that path is impossible.\n                common_sampler_ptr smpl_save;\n                if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||\n                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > llama_n_rs_seq(ctx_tgt))) {\n                    smpl_save.reset(common_sampler_clone(slot.smpl.get()));\n                }\n\n                GGML_ASSERT(spec_i_batch.size() == n_draft + 1);"
  ],
  [
    "tools/mtmd/mtmd-image.cpp",
    "        const auto src_size = src.get_size();\n        const auto dst_size = dst.get_size();\n        for (int y = 0; y < src_size.height; ++y) {\n            for (int x = 0; x < src_size.width; ++x) {\n                int dx = x + offset_x;\n                int dy = y + offset_y;\n                // skip pixels that would be out of bounds in the destination\n                if (dx < 0 || dy < 0 || dx >= dst_size.width || dy >= dst_size.height) {\n                    continue;\n                }\n                dst.set_pixel(dx, dy, src.get_pixel(x, y));\n            }\n        }",
    "        if (dst.is_placeholder()) {\n            return;\n        }\n        const auto src_size = src.get_size();\n        const auto dst_size = dst.get_size();\n        // Clip once, preserving the exact row-major copy order. Widen offset\n        // arithmetic so even extreme signed offsets cannot overflow.\n        const int64_t x0 = std::max<int64_t>(0, -int64_t(offset_x));\n        const int64_t y0 = std::max<int64_t>(0, -int64_t(offset_y));\n        const int64_t x1 = std::min<int64_t>(src_size.width, int64_t(dst_size.width) - offset_x);\n        const int64_t y1 = std::min<int64_t>(src_size.height, int64_t(dst_size.height) - offset_y);\n        if (x0 >= x1 || y0 >= y1) {\n            return;\n        }\n        for (int64_t y = y0; y < y1; ++y) {\n            for (int64_t x = x0; x < x1; ++x) {\n                dst.set_pixel(int(x + offset_x), int(y + offset_y), src.get_pixel(int(x), int(y)));\n            }\n        }"
  ],
  [
    "tools/mtmd/mtmd-image.cpp",
    "void mtmd_image_preproc_out::append(const clip_hparams & hparams, const std::vector<clip_image_u8> & imgs, bool normalized) {\n    for (const auto & img : imgs) {",
    "void mtmd_image_preproc_out::append(const clip_hparams & hparams, const std::vector<clip_image_u8> & imgs, bool normalized) {\n    // Reserve for the complete tile group before moving image buffers.\n    entries.reserve(entries.size() + imgs.size());\n    for (const auto & img : imgs) {"
  ],
  [
    "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq4_xs.comp",
    "    const uint y_idx = i * QUANT_K + 32 * ib32;\n\n    uint ibi = a_offset + first_row * num_blocks_per_row + i;",
    "    const uint y_idx = i * QUANT_K + 32 * ib32;\n\n    // Called only for valid superblocks. Reuse the activation slice across\n    // output rows during single-column decode; leave prefill unchanged.\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec4 cached_b[8];\n    if (reuse_b) {\n        const uint base_b = (b_offset + y_idx) / 4;\n        [[unroll]] for (uint k = 0; k < 8; ++k) {\n            cached_b[k] = vec4(data_b_v4[base_b + k]);\n        }\n    }\n\n    uint ibi = a_offset + first_row * num_blocks_per_row + i;"
  ],
  [
    "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq4_xs.comp",
    "                const vec4 b0 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + l]);\n                const vec4 b1 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 4 + l]);",
    "                const vec4 b0 = reuse_b ? cached_b[l] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + l]);\n                const vec4 b1 = reuse_b ? cached_b[4 + l] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 4 + l]);"
  ],
  [
    "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_s.comp",
    "    const uint y_idx = i * QUANT_K + 32 * ib32;\n\n    uint ibi = a_offset + first_row * num_blocks_per_row + i;",
    "    const uint y_idx = i * QUANT_K + 32 * ib32;\n\n    const bool reuse_b = NUM_COLS == 1 && NUM_ROWS > 1 && num_rows > 1;\n    vec4 cached_b[2 * NL];\n    if (reuse_b) {\n        const uint base_b = (b_offset + y_idx) / 4 + 2*l0;\n        [[unroll]] for (uint k = 0; k < 2 * NL; ++k) {\n            cached_b[k] = vec4(data_b_v4[base_b + k]);\n        }\n    }\n\n    uint ibi = a_offset + first_row * num_blocks_per_row + i;"
  ],
  [
    "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_s.comp",
    "                const vec4 b0 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 0]);\n                const vec4 b4 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 1]);",
    "                const vec4 b0 = reuse_b ? cached_b[2*ll] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 0]);\n                const vec4 b4 = reuse_b ? cached_b[2*ll + 1] : vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 2*l + 1]);"
  ]
]

ENGINE_EXTRA = {
    "whisper": [
  [
    "src/whisper.cpp",
    "            float * dst = wstate.inp_mel.data();\n            memset(dst, 0, ggml_nbytes(mel));\n\n            const int i0 = std::min(mel_offset,           mel_inp.n_len);\n            const int i1 = std::min(mel_offset + 2*n_ctx, mel_inp.n_len);\n\n            for (int j = 0; j < mel_inp.n_mel; ++j) {\n                for (int i = i0; i < i1; ++i) {\n                    dst[j*2*n_ctx + (i - i0)] = mel_inp.data[j*mel_inp.n_len + i];\n                }\n            }",
    "            float * dst = wstate.inp_mel.data();\n\n            const int i0 = std::min(mel_offset,           mel_inp.n_len);\n            const int i1 = std::min(mel_offset + 2*n_ctx, mel_inp.n_len);\n            const int n_copy = std::max(0, i1 - i0);\n            // Full windows overwrite the entire tensor; avoid zeroing it first.\n            if (n_copy != 2*n_ctx || int64_t(mel_inp.n_mel) * 2*n_ctx != ggml_nelements(mel)) {\n                memset(dst, 0, ggml_nbytes(mel));\n            }\n            if (n_copy > 0) {\n                for (int j = 0; j < mel_inp.n_mel; ++j) {\n                    std::copy_n(mel_inp.data.data() + size_t(j)*mel_inp.n_len + i0,\n                                n_copy, dst + size_t(j)*2*n_ctx);\n                }\n            }"
  ]
],
    "diffusion": [
  [
    "src/tokenizers/tokenizer.cpp",
    "    } else if (!allow_overflow_expand) {\n        build_sequence(0, tokens.size(), 0, new_tokens, new_weights, new_mask);\n\n        new_tokens.resize(max_length);",
    "    } else if (!allow_overflow_expand) {\n        // Tokens beyond the truncation boundary cannot survive resize().\n        // Keep BOS accounting, and retain the existing final EOS replacement.\n        const size_t take = std::min(tokens.size(), max_length > bos_count ? max_length - bos_count : 0);\n        build_sequence(0, take, 0, new_tokens, new_weights, new_mask);\n\n        new_tokens.resize(max_length);"
  ]
],
    "audio": [
  [
    "src/framework/audio/conversion.cpp",
    "    return convert_wav_to_mono_linear_resampled(\n        WavData{sample_rate_hz, channel_count, interleaved_samples},\n        target_sample_rate_hz);",
    "    // The temporary WavData copied the entire input before mixdown.\n    // Validate and operate on the original immutable sample vector instead.\n    if (sample_rate_hz <= 0 || target_sample_rate_hz <= 0) {\n        throw std::runtime_error(\"audio sample rates must be positive\");\n    }\n    auto mono = mixdown_interleaved_to_mono_average(interleaved_samples, channel_count);\n    if (sample_rate_hz != target_sample_rate_hz) {\n        mono = resample_mono_linear(mono, sample_rate_hz, target_sample_rate_hz);\n    }\n    return mono;"
  ],
  [
    "src/framework/audio/waveform_ops.cpp",
    "    if (peak > 1.0F) {\n        for (float & sample : samples) {\n            sample /= peak;\n        }\n    }\n    for (float & sample : samples) {\n        sample = std::clamp(sample, -1.0F, 1.0F);\n    }",
    "    if (peak > 1.0F) {\n        // Keep the same floating-point division and clamp, in one pass.\n        for (float & sample : samples) {\n            sample /= peak;\n            sample = std::clamp(sample, -1.0F, 1.0F);\n        }\n    } else {\n        for (float & sample : samples) {\n            sample = std::clamp(sample, -1.0F, 1.0F);\n        }\n    }"
  ]
],
}


# DSpark adaptive blocks and graph layout fast paths.
LLAMA += [
  [
    "common/speculative.cpp",
    "        // build one batch holding every drafting sequence's noise block into a single decode)\n        // record where each block starts and its size\n        std::vector<int32_t> i_block_beg(n_seq, -1);\n        std::vector<int32_t> n_block    (n_seq,  0);",
    "        // DSpark's Markov head requires uniform block lengths across sequences.\n        // Choose the largest requested cap, bounded by the trained block size.\n        // If any sequence needs the original block, retain it for the entire batch.\n        int32_t dspark_n_draft = params.n_max;\n        if (is_dspark && !is_lilicorr && !is_dflash2 && params.n_max > 0) {\n            int32_t requested = 0;\n            for (const auto & dp : dparams) {\n                if (!dp.drafting) {\n                    continue;\n                }\n                const int32_t cap = dp.n_max > 0 && dp.n_max >= params.n_min\n                    ? std::min(params.n_max, dp.n_max) : params.n_max;\n                requested = std::max(requested, cap);\n            }\n            if (requested > 0) {\n                dspark_n_draft = requested;\n            }\n        }\n\n        // Keep block bookkeeping allocations across decoding rounds.\n        i_block_beg.assign(n_seq, -1);\n        n_block.assign(n_seq, 0);"
  ],
  [
    "common/speculative.cpp",
    "    // dspark speculators\n    bool sample_from_anchor = true;",
    "    // DSpark speculators and reusable per-batch bookkeeping.\n    bool sample_from_anchor = true;\n    std::vector<int32_t> i_block_beg;\n    std::vector<int32_t> n_block;"
  ],
  [
    "common/speculative.cpp",
    "            const int32_t n_draft = params.n_max;\n            const int32_t n_block_tokens = is_lilicorr ? block_size : n_draft + (is_dspark && sample_from_anchor ? 0 : 1);",
    "            const int32_t n_draft = is_dspark ? dspark_n_draft : params.n_max;\n            const int32_t n_block_tokens = is_lilicorr ? block_size : n_draft + (is_dspark && sample_from_anchor ? 0 : 1);"
  ],
  [
    "common/speculative.cpp",
    "                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;\n                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {",
    "                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;\n                // A longer block may be required by another sequence. Do not\n                // sample this sequence's positions that would be truncated.\n                const int32_t end = dp.n_max > 0 && dp.n_max >= params.n_min\n                    ? i_draft_beg + std::min(n_block_tokens - i_draft_beg, dp.n_max)\n                    : n_block_tokens;\n                for (int32_t i = i_draft_beg; i < end; ++i) {"
  ],
  [
    "src/models/dflash.cpp",
    "    // cat is position-major; restore ubatch block-major order\n    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);\n    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]\n    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);\n\n    if (has_conf) {\n        ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);\n        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));\n        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);",
    "    // Position-major and block-major are identical for one sequence or\n    // one position. Avoid building a redundant full-vocabulary copy.\n    const auto restore_block_order = [&](ggml_tensor * tensor, int64_t width) {\n        if (n_blocks == 1 || block_drafts == 1) {\n            return ggml_reshape_2d(ctx0, ggml_is_contiguous(tensor) ? tensor : ggml_cont(ctx0, tensor), width, n_tok);\n        }\n        ggml_tensor * ordered = ggml_reshape_3d(ctx0, tensor, width, n_blocks, block_drafts);\n        ordered = ggml_cont(ctx0, ggml_permute(ctx0, ordered, 0, 2, 1, 3));\n        return ggml_reshape_2d(ctx0, ordered, width, n_tok);\n    };\n    ggml_tensor * out = restore_block_order(cat, n_vocab);\n\n    if (has_conf) {\n        ggml_tensor * conf = restore_block_order(cat_conf, 1);"
  ]
]


def patch_specs(component):
    if component not in ("llama", "diffusion", "whisper", "audio"):
        raise ValueError(component)
    root = "external/ggml/" if component == "audio" else "ggml/"
    return [(root + p, a, b) for p, a, b in COMMON + (TERNARY if component in ("llama", "diffusion") else [])] + (LLAMA if component == "llama" else ENGINE_EXTRA[component])

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
