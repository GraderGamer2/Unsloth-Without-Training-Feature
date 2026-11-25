# SPDX-License-Identifier: AGPL-3.0-only

import dataclasses
import json
import sys

import pytest

from studio import install_audio_cpp_prebuilt as audio
from studio import install_llama_prebuilt as llama
from studio import install_sd_cpp_prebuilt as diffusion
from studio import install_whisper_prebuilt as whisper
from studio.backend.utils.prebuilt.llama_backend import requestable_backends


@pytest.fixture(autouse = True)
def amd_desktop(monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux desktop GPU policy")
    monkeypatch.setenv("UNSLOTH_AMD_ONLY", "1")
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)


def forbidden(*args, **kwargs):
    pytest.fail("NVIDIA probe or download was invoked")


def test_llama_amd_profile_ignores_nvidia_and_intel(monkeypatch):
    monkeypatch.setattr(llama, "nvidia_library_inventory", forbidden)
    monkeypatch.setattr(llama.shutil, "which", lambda name: forbidden() if name == "nvidia-smi" else None)
    host = llama.detect_host()
    assert not host.has_physical_nvidia
    assert not host.has_usable_nvidia
    assert not host.has_intel_gpu

    mixed_host = dataclasses.replace(
        host, has_physical_nvidia = True, has_usable_nvidia = True, has_intel_gpu = True, has_rocm = True
    )
    route = llama.route_backend_request(
        backend = "rocm", published_repo = "", published_release_tag = "", host = mixed_host
    )
    assert route.host.has_rocm
    assert not route.host.has_usable_nvidia
    assert not route.host.has_intel_gpu


def test_saved_cuda_choice_returns_to_auto(tmp_path):
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"backend": "cuda", "backend_request": "cuda"}), encoding = "utf-8"
    )
    assert llama.effective_backend_request(install_dir = tmp_path) == ("auto", False)
    assert not llama._kept_install_payload_is_healthy(tmp_path, llama.platform_only_host())
    assert requestable_backends() == ("auto", "cpu", "rocm", "vulkan")
    with pytest.raises(llama.BackendUnavailable, match = "cannot install the cuda backend"):
        llama.route_backend_request(backend = "cuda", published_repo = "", published_release_tag = "")


def test_llama_skips_cuda_wheel_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(llama, "_linux_cuda_runtime_wheels_to_supply", forbidden)
    monkeypatch.setattr(llama, "_missing_cuda_runtime_wheels", forbidden)
    llama.ensure_linux_cuda_runtime_before_selection(None)
    llama.ensure_cuda_runtime_wheels(tmp_path)


def test_companion_installers_disallow_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "nvidia_driver_cuda_version", forbidden)
    monkeypatch.setattr(audio, "_amd_gpu_present", lambda: True)
    assert audio.detect_accelerator() == "vulkan"
    monkeypatch.setattr(audio, "_amd_gpu_present", lambda: False)
    assert audio.detect_accelerator() == "cpu"
    with pytest.raises(RuntimeError, match = "cannot install the cuda audio.cpp backend"):
        audio.install(install_dir = tmp_path, accelerator = "cuda")
    with pytest.raises(RuntimeError, match = "cannot install the cuda sd.cpp backend"):
        diffusion.install(install_dir = tmp_path, accelerator = "cuda")
    with pytest.raises(whisper.PrebuiltFallback, match = "CUDA whisper.cpp"):
        whisper.resolve_backend(whisper.detect_host(), "cuda", cpu_fallback = False)
