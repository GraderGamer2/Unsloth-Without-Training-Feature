#!/usr/bin/env python3
"""Content-based, per-engine cache fingerprint. Never restore a binary by prefix."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMPONENTS = ("llama", "diffusion", "whisper", "audio")


def digest(items):
    h = hashlib.sha256()
    for item in items:
        raw = item.encode("utf-8") if isinstance(item, str) else item
        h.update(len(raw).to_bytes(8, "big"))
        h.update(raw)
    return h.hexdigest()


def toolchain():
    commands = [
        ["dpkg-query", "-W"],
        ["gcc", "--version"], ["g++", "--version"], ["ld", "--version"],
        ["cmake", "--version"], ["ninja", "--version"], ["glslc", "--version"],
        ["glslangValidator", "--version"], ["spirv-opt", "--version"],
        ["ccache", "--version"], ["python3", "--version"], ["uname", "-m"],
    ]
    # Include environment inputs that may change compiler and CMake configuration.
    flags = ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
             "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "CMAKE_GENERATOR",
             "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER",
             "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR", "PKG_CONFIG_SYSROOT_DIR")
    return digest([Path("/etc/os-release").read_bytes(),
                   json.dumps({name: os.environ.get(name) for name in flags}, sort_keys=True)] +
                  [subprocess.check_output(cmd) for cmd in commands])


def build_source(component):
    source = (ROOT / "build-personal-runtimes.sh").read_text()
    marker = 'for component in "${components[@]}"; do\n    case "$component" in\n'
    # If the shell structure changes, conservatively invalidate all engines.
    if source.count(marker) != 1:
        return source
    common, cases = source.split(marker)
    parts = re.split(r"(?m)^        (llama|diffusion|whisper|audio|\*)\)\n", cases)
    if parts[0] or tuple(parts[1::2]) != COMPONENTS + ("*",):
        return source
    blocks = dict(zip(parts[1::2], parts[2::2]))
    if any(not blocks[name].rstrip().endswith(";;") for name in COMPONENTS):
        return source
    return common + marker + component + blocks[component] + blocks["*"]


def patch_fingerprint(name, component):
    path = ROOT / name
    source = path.read_text()
    spec = importlib.util.spec_from_file_location("patch_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = module.patch_specs(component)
    # All code and unknown assignments are hashed. Only current declarative
    # patch tables are replaced by the actual selected patch specifications.
    known_tables = {"COMMON", "LLAMA", "WHISPER", "DIFFUSION", "AUDIO", "DEBUG", "DEBUG_OFF", "TERNARY", "ENGINE_EXTRA"}
    tree = ast.parse(source)
    kept = []
    for node in tree.body:
        # Only ignore declarative literal tables (including literal += entries).
        # Arbitrary executable modifications still invalidate every component.
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            target = node.target
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            if (target.value.id == "ENGINE_EXTRA" and isinstance(target.slice, ast.Constant)
                    and target.slice.value in COMPONENTS):
                target = target.value
        if isinstance(target, ast.Name) and target.id in known_tables:
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
            else:
                continue
        kept.append(ast.dump(node, include_attributes=False))
    return json.dumps([kept, selected], ensure_ascii=False, separators=(",", ":"))


def runtime(component, tools):
    inputs = [
        tools, component, Path(__file__).read_bytes(),
        (ROOT / "personal-runtime-archive.sh").read_bytes(),
        build_source(component),
        patch_fingerprint("apply-personal-runtime.py", component),
        patch_fingerprint("apply-personal-throughput.py", component),
    ]
    if component == "llama":
        inputs.append((ROOT / "apply-personal-llama-vulkan.py").read_bytes())
    return digest(inputs)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMPONENTS:
        raise SystemExit("usage: personal-build-cache-key.py ENGINE")
    tools = toolchain()
    print("toolchain=" + tools)
    print("runtime=" + runtime(sys.argv[1], tools))
