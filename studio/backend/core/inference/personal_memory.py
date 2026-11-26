# SPDX-License-Identifier: AGPL-3.0-only
"""Conservative load admission for the personal AMD APU build."""

MIB = 1024**2


def hybrid_memory_shortfall(available_mib, weight_bytes, kv_bytes):
    """Count CPU and iGPU allocations against one pool; never credit advertised VRAM."""
    if available_mib is None or weight_bytes <= 0 or kv_bytes < 0:
        return "Cannot verify the shared RAM budget for this CPU/hybrid load."
    weights_mib = (weight_bytes + MIB - 1) // MIB
    kv_mib = (kv_bytes + MIB - 1) // MIB
    # Leave room for the desktop and for native compute/staging buffers.
    desktop_mib = 3072
    scratch_mib = max(1024, (weights_mib + 7) // 8)
    needed_mib = weights_mib + kv_mib + scratch_mib + desktop_mib
    if needed_mib > available_mib:
        return (
            f"CPU/hybrid load blocked: estimated {needed_mib / 1024:.2f} GiB "
            f"including a 3 GiB desktop reserve; only {available_mib / 1024:.2f} GiB "
            "of system RAM is available. The integrated GPU shares system memory. "
            "Use fewer CPU layers by choosing automatic GPU placement, a smaller "
            "GGUF/context, or free RAM."
        )
    return None
