# SPDX-License-Identifier: AGPL-3.0-only

"""GPU policy shared by the Linux desktop bootstrap and runtime."""

import os
import sys
from pathlib import Path

_PERSONAL_WHEEL = Path(__file__).resolve().parent.parent.joinpath("personal-vulkan").is_file()


def amd_only() -> bool:
    return sys.platform.startswith("linux") and os.environ.get("UNSLOTH_AMD_ONLY") == "1"


def personal_vulkan() -> bool:
    """The dedicated Ryzen desktop distribution has fixed local inference capabilities."""
    return sys.platform.startswith("linux") and (
        _PERSONAL_WHEEL or os.environ.get("UNSLOTH_PERSONAL_VULKAN") == "1"
    )
