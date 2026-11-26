#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

if [[ $# -lt 2 || "$(uname -sm)" != "Linux x86_64" ]]; then
    echo "Usage on Linux x86_64: $0 WORK_DIRECTORY OUTPUT_DIRECTORY [llama|diffusion|whisper|audio ...]" >&2
    exit 2
fi

mkdir -p -- "$1" "$2"
work=$(realpath -- "$1")
output=$(realpath -- "$2")
shift 2
jobs=${UNSLOTH_BUILD_JOBS:-4}
components=("$@")
if [[ ${#components[@]} -eq 0 ]]; then
    components=(llama diffusion whisper audio)
fi

checkout_source() {
    local name=$1 repository=$2 revision=$3
    local source="$work/sources/$name"
    mkdir -p "$source"
    if [[ ! -d "$source/.git" ]]; then
        git -C "$source" init --quiet
        git -C "$source" remote add origin "https://github.com/unslothai/$repository.git"
    fi
    if [[ "$(git -C "$source" rev-parse --verify HEAD 2>/dev/null || true)" != "$revision" ]]; then
        git -C "$source" fetch --depth 1 origin "$revision"
        git -C "$source" checkout --quiet --detach FETCH_HEAD
    fi
}

compile() {
    local component=$1
    shift
    local -a targets=()
    while [[ $# -gt 0 && "$1" != -- ]]; do
        targets+=("$1")
        shift
    done
    shift
    local source="$work/sources/$component" build="$work/native/$component"
    local target binary library
    local shared=OFF
    # Studio probes ggml-base and ggml-vulkan directly to discover device ordinals and RAM budgets.
    # Static llama-server can run Vulkan but is invisible to that probe.
    if [[ "$component" == llama ]]; then
        shared=ON
    fi
    local native_flags="-O3 -march=znver3 -mtune=znver3 -DNDEBUG -ffunction-sections -fdata-sections"
    local ggml="$source/ggml"
    if [[ "$component" == audio ]]; then
        ggml="$source/external/ggml"
    fi
    # These optional shaders require NVIDIA Vulkan extensions that Rembrandt cannot use.
    python3 - "$ggml/src/ggml-vulkan/CMakeLists.txt" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
filtered = re.sub(
    r'(?m)^[ \t]*test_shader_extension_support\(\s*"GL_NV_[^"]+"\s*"[^"]+"\s*"[^"]+"\s*\)\n',
    "",
    text,
)
if filtered != text:
    path.write_text(filtered)
PY
    cmake -S "$source" -B "$build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS_RELEASE="$native_flags" \
        -DCMAKE_CXX_FLAGS_RELEASE="$native_flags" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--gc-sections -Wl,--as-needed" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DBUILD_SHARED_LIBS="$shared" -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
        -DGGML_NATIVE=OFF -DGGML_LTO=ON -DGGML_BACKEND_DL=OFF \
        -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF \
        -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF "$@"
    cmake --build "$build" --parallel "$jobs" --target "${targets[@]}"
    mkdir -p "$output/$component"
    for target in "${targets[@]}"; do
        binary=$(find "$build/bin" -maxdepth 1 -type f -name "$target" -print -quit)
        test -x "$binary"
        install -s -m 755 "$binary" "$output/$component/$target"
    done
    if [[ "$component" == llama ]]; then
        # Copy the actual shared objects and their SONAME symlinks together, not the static archives.
        shopt -s nullglob
        local -a libraries=("$build/bin/"*.so*)
        ((${#libraries[@]} > 0))
        cp -a -- "${libraries[@]}" "$output/$component/"
        for library in "$output/$component/"*.so*; do
            if [[ -f "$library" && ! -L "$library" ]]; then
                strip --strip-unneeded "$library"
            fi
        done
        for library in libggml-base.so libggml-vulkan.so libggml-cpu.so; do
            test -f "$output/$component/$library"
        done
    fi
    install -m 644 "$source/LICENSE" "$output/$component/LICENSE"
}

for component in "${components[@]}"; do
    case "$component" in
        llama)
            checkout_source llama llama.cpp 5c5347de57213dca9f40a1ea3ad11c6072d1b548
            compile llama llama-server -- -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
                -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
                -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF \
                -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
            ;;
        diffusion)
            checkout_source diffusion stable-diffusion.cpp 9455ee1fa10cba979e9b02533550bf4d71d654fe
            git -C "$work/sources/diffusion" submodule update --init --depth 1 \
                ggml thirdparty/libwebp thirdparty/libwebm
            compile diffusion sd-cli sd-server -- -DSD_VULKAN=ON -DSD_CUDA=OFF \
                -DSD_HIPBLAS=OFF -DSD_METAL=OFF -DSD_BUILD_EXAMPLES=ON \
                -DSD_BUILD_SHARED_LIBS=OFF -DSD_BUILD_SHARED_GGML_LIB=OFF \
                -DSD_SERVER_BUILD_FRONTEND=OFF
            ;;
        whisper)
            checkout_source whisper whisper.cpp aadc60b91fa111c1a16dd0c833ab36d13e818bbe
            compile whisper whisper-server -- -DWHISPER_BUILD_TESTS=OFF \
                -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=ON \
                -DWHISPER_CURL=OFF -DWHISPER_SDL2=OFF
            ;;
        audio)
            checkout_source audio audio.cpp 39588d42c4762e47f4a9cd111c89110d7b069abc
            compile audio audiocpp_server -- -DENGINE_ENABLE_NATIVE_CPU=OFF \
                -DENGINE_ENABLE_VULKAN=ON -DENGINE_ENABLE_CUDA=OFF \
                -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_METAL=OFF
            ;;
        *)
            echo "Unknown runtime: $component" >&2
            exit 2
            ;;
    esac
done
