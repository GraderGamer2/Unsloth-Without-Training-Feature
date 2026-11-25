# SPDX-License-Identifier: AGPL-3.0-only

"""GPU policy shared by the Linux desktop bootstrap and runtime."""

import os
import sys


def amd_only() -> bool:
    return sys.platform.startswith("linux") and os.environ.get("UNSLOTH_AMD_ONLY") == "1"
