#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(CDPATH= cd -- "$script_dir/../../.." && pwd -P)
resource_dir="$script_dir/../backend-wheel"
wheel_dir=$(mktemp -d)
trap 'rm -rf -- "$wheel_dir"' EXIT

python3 -m pip wheel --disable-pip-version-check --no-deps --wheel-dir "$wheel_dir" "$repo"
set -- "$wheel_dir"/unsloth-*.whl
[ "$#" -eq 1 ] && [ -f "$1" ] || { echo "Expected one Unsloth backend wheel" >&2; exit 1; }
if [ "${UNSLOTH_PERSONAL_VULKAN:-0}" = 1 ]; then
    python3 "$script_dir/prune-personal-wheel.py" "$1"
fi
rm -f -- "$resource_dir"/unsloth-*.whl
mv -- "$1" "$resource_dir/"
