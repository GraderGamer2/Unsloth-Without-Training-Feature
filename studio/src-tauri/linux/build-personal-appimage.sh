#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 || "$(uname -sm)" != "Linux x86_64" ]] ||
   [[ -n "${5:-}" && "$5" != "--package-only" ]]; then
    echo "Usage on Debian 13 x86_64: $0 WORK_DIRECTORY SEED_APPIMAGE APPIMAGETOOL OUTPUT_APPIMAGE [--package-only]" >&2
    exit 2
fi

mkdir -p -- "$1"
work=$(realpath -- "$1")
seed=$(realpath -- "$2")
packer=$(realpath -- "$3")
output=$(realpath -m -- "$4")
script_dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(cd -- "$script_dir/../../.." && pwd -P)
stage=$(mktemp -d "$work/appimage.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
test -x "$seed"
test -x "$packer"
test ! -e "$output"

jobs=${UNSLOTH_BUILD_JOBS:-4}
if [[ "${5:-}" != "--package-only" ]]; then
    "$script_dir/build-personal-runtimes.sh" "$work" "$work/runtimes"
    (
        cd "$repo"
        export UNSLOTH_PERSONAL_VULKAN=1 VITE_UNSLOTH_PERSONAL_VULKAN=1
        export CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
        export CARGO_PROFILE_RELEASE_STRIP=symbols CARGO_PROFILE_RELEASE_OPT_LEVEL=3
        export CARGO_TARGET_DIR="$work/tauri-target" CARGO_BUILD_JOBS="$jobs"
        export RUSTFLAGS="-C target-cpu=znver3 ${RUSTFLAGS:-}"
        npx --prefix studio tauri build --no-bundle \
            --target x86_64-unknown-linux-gnu --features personal-vulkan
    )
fi

for runtime in llama/llama-server diffusion/sd-cli diffusion/sd-server \
    whisper/whisper-server audio/audiocpp_server; do
    test -x "$work/runtimes/$runtime"
done

cd "$stage"
"$seed" --appimage-extract > /dev/null
appdir="$stage/squashfs-root"
resources="$appdir/usr/lib/Unsloth"
test -d "$resources"
install -m 755 "$script_dir/appimage-apprun.sh" "$appdir/AppRun.wrapped"
install -m 755 "$work/tauri-target/x86_64-unknown-linux-gnu/release/unsloth-studio" \
    "$appdir/usr/bin/unsloth-studio"
install -m 755 "$repo/install.sh" "$resources/install.sh"
rm -f -- "$resources/backend-wheel"/unsloth-*.whl
install -m 644 "$script_dir/../backend-wheel"/unsloth-*.whl "$resources/backend-wheel/"
mkdir -p "$resources/personal-runtimes"
cp -a -- "$work/runtimes/." "$resources/personal-runtimes/"

install -m 755 "$script_dir/finalize-complete-appimage.sh" "$stage/finalize-complete-appimage.sh"
install -m 644 "$appdir/usr/share/unsloth/fonts/UnslothSafeEmoji.ttf" "$stage/UnslothSafeEmoji.ttf"
install -m 644 "$appdir/usr/share/doc/unsloth-safe-emoji/copyright" "$stage/UnslothSafeEmoji.LICENSE"
install -m 644 "$appdir/usr/etc/fonts/unsloth-appimage.conf" "$stage/unsloth-appimage-fonts.conf"
"$stage/finalize-complete-appimage.sh" "$appdir"

head -c "$("$seed" --appimage-offset)" "$seed" > "$stage/runtime"
mkdir -p -- "$(dirname -- "$output")"
ARCH=x86_64 "$packer" --no-appstream --runtime-file "$stage/runtime" \
    --mksquashfs-opt=-processors --mksquashfs-opt="$jobs" "$appdir" "$output"
UNSLOTH_APPIMAGE_GLIBC_FLOOR=2.41 "$script_dir/verify-complete-appimage.sh" "$output"
(
    cd "$(dirname -- "$output")"
    sha256sum "$(basename -- "$output")" > SHA256SUMS
)
