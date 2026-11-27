#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
mode=${1:?mode}; component=${2:?component}; archive=${3:?archive directory}; dest=${4:-}
case "$component" in
    llama) binaries=(llama-server); libs=(libggml-base.so libggml-vulkan.so libggml-cpu.so);;
    diffusion) binaries=(sd-cli sd-server); libs=();;
    whisper) binaries=(whisper-server); libs=();;
    audio) binaries=(audiocpp_server); libs=();;
    *) exit 2;;
esac
check_tree() {
    local root=$1 name
    for name in "${binaries[@]}"; do
        test -x "$root/$component/$name" || return 1
        readelf -h "$root/$component/$name" | grep -F 'Advanced Micro Devices X86-64' >/dev/null || return 1
    done
    for name in "${libs[@]}"; do
        test -f "$root/$component/$name" || return 1
        readelf -h "$root/$component/$name" | grep -F 'Advanced Micro Devices X86-64' >/dev/null || return 1
    done
    test -f "$root/$component/LICENSE"
}
verify() (
    set -euo pipefail
    cd "$archive" || exit 1
    test -f SHA256SUMS && test -f runtime.tar || exit 1
    grep -Ex '[0-9a-f]{64}  runtime\.tar' SHA256SUMS >/dev/null || exit 1
    [[ $(wc -l < SHA256SUMS) == 1 ]] || exit 1
    sha256sum --strict -c SHA256SUMS || exit 1
    # Runtime packages contain one flat component directory. Reject absolute
    # names, traversal, unexpected directories and escaping link targets.
    python3 - "$component" <<'PY' || exit 1
import sys
import tarfile
from pathlib import PurePosixPath

component = sys.argv[1]
with tarfile.open("runtime.tar", "r:") as archive:
    seen = set()
    for entry in archive:
        parts = PurePosixPath(entry.name).parts
        if not parts or parts[0] != component or len(parts) > 2 or ".." in parts:
            raise SystemExit("Invalid runtime archive member")
        if entry.name in seen:
            raise SystemExit("Duplicate runtime archive member")
        seen.add(entry.name)
        if len(parts) == 1:
            if not entry.isdir():
                raise SystemExit("Invalid component directory")
        elif entry.issym():
            target = PurePosixPath(entry.linkname)
            if len(target.parts) != 1 or target.name in ("", ".", "..") or target.is_absolute():
                raise SystemExit("Invalid runtime symlink")
        elif entry.islnk():
            target = PurePosixPath(entry.linkname).parts
            if len(target) != 2 or target[0] != component or ".." in target:
                raise SystemExit("Invalid runtime hard link")
        elif not entry.isfile():
            raise SystemExit("Unexpected runtime archive member type")
PY
    local scratch
    scratch=$(mktemp -d) || exit 1
    trap 'rm -rf "$scratch"' EXIT
    tar -xf runtime.tar -C "$scratch" || exit 1
    check_tree "$scratch" || exit 1
)
case "$mode" in
    pack)
        check_tree "$dest"
        mkdir -p "$archive"
        tar -C "$dest" -cf "$archive/runtime.tar" "$component"
        (cd "$archive" && sha256sum runtime.tar > SHA256SUMS)
        verify
        ;;
    verify) verify;;
    unpack)
        verify
        mkdir -p "$dest"
        tar -xf "$archive/runtime.tar" -C "$dest"
        check_tree "$dest"
        ;;
    *) exit 2;;
esac
