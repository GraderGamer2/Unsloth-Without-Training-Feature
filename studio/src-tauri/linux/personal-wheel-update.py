# SPDX-License-Identifier: AGPL-3.0-only
"""Offline replacement of a compatible personal backend wheel.

Executed by the desktop with its runtime mutation lock held. Only the installed
Unsloth distribution is backed up; the environment, dependencies and user data
are not recreated. A durable backup is published before invoking uv.
"""
import csv
import email
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile

ROOTS = {"studio", "unsloth", "unsloth_cli"}
MANIFEST = "unsloth_install_manifest.json"
JOURNAL = ".personal-wheel-rollback"
LOCK_FD_ENV = "UNSLOTH_PERSONAL_WHEEL_LOCK_FD"
DIST = re.compile(r"unsloth-[A-Za-z0-9_.+!-]+\.dist-info")
CAPABILITIES = (
    "studio_install_ok", "supports_api_only", "supports_provision_desktop_auth",
    "supports_desktop_backend_ownership", "supports_fixed_desktop_storage",
    "supports_amd_desktop", "supports_personal_vulkan",
)


class Incompatible(Exception):
    pass


def require(ok, reason):
    if not ok:
        raise Incompatible(reason)


def location(root):
    venv = root / "unsloth_studio"
    site = Path(sysconfig.get_path("purelib"))
    require(Path(sys.prefix).resolve() == venv.resolve() and
            site.resolve().is_relative_to(venv.resolve()) and
            site == site.resolve() and not venv.is_symlink(),
            "Not the owned managed Python environment")
    return venv, site


def allowed_entry(path):
    parts = PurePosixPath(path).parts
    return (len(parts) == 2 and parts[0] == "site" and
            (parts[1] in ROOTS or DIST.fullmatch(parts[1]))) or path in {
                "bin/unsloth", MANIFEST,
            }


def target(venv, site, key):
    if not allowed_entry(key):
        raise RuntimeError("Invalid backend rollback entry")
    if key.startswith("site/"):
        return site / key.split("/", 1)[1]
    return venv / key


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            json.dump(value, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
            os.replace(temporary, path)
            sync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def copy_entry(src, dst):
    if src.is_symlink():
        raise RuntimeError("Refusing symlinked backend package entry")
    if src.is_dir():
        # Preserve directory/file symlinks as links, never follow them.
        shutil.copytree(src, dst, symlinks=True)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        raise RuntimeError("Missing backend package entry")


def remove_entry(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def retire_journal(journal):
    # Rename the completed journal out of the recovery slot before deleting it.
    # A crash during deletion cannot destroy the only active rollback record.
    with tempfile.TemporaryDirectory(prefix=".personal-wheel-retired-", dir=journal.parent) as temp:
        retired = Path(temp) / "journal"
        journal.rename(retired)
        sync_dir(journal.parent)


def recover(venv, site):
    journal = venv / JOURNAL
    if not journal.exists() and not journal.is_symlink():
        return
    if journal.is_symlink() or not journal.is_dir():
        raise RuntimeError("Invalid backend rollback directory")
    state = json.loads((journal / "state.json").read_text())
    entries = state["entries"]
    present = state["present"]
    if (not isinstance(entries, list) or len(entries) > 8 or not all(
        isinstance(key, str) and allowed_entry(key) for key in entries
    ) or len(set(entries)) != len(entries) or not isinstance(present, list)
        or not all(isinstance(key, str) and key in entries for key in present)):
        raise RuntimeError("Invalid backend rollback journal")
    if state.get("committed") is True:
        retire_journal(journal)
        return
    # Keep the backup until every original entry is restored. Replaying an
    # interrupted rollback is safe because the backup is never consumed.
    for key in entries:
        if key in state["present"] and not (journal / "backup" / key).exists():
            raise RuntimeError("Backend rollback backup is incomplete")
    for key in entries:
        live = target(venv, site, key)
        remove_entry(live)
        if key in state["present"]:
            copy_entry(journal / "backup" / key, live)
    retire_journal(journal)


def lock_fds():
    value = os.environ.get(LOCK_FD_ENV)
    if value is None:
        raise RuntimeError("Offline update requires the desktop runtime lock")
    fd = int(value)
    # flock on this inherited open-file description retains the parent's lock.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return (fd,)


def capabilities(venv, expected):
    proc = subprocess.run(
        [sys.executable, "-I", "-B", str(venv / "bin/unsloth"),
         "studio", "desktop-capabilities", "--json"],
        capture_output=True, text=True, timeout=30, check=True, pass_fds=lock_fds(),
    )
    data = json.loads(proc.stdout)
    require(all(data.get(key) is True for key in CAPABILITIES) and
            data.get("desktop_protocol_version") == 1 and
            data.get("desktop_manageability_version", 0) >= 2 and
            data.get("llama_runtime_ok") is not False and data.get("version") == expected,
            "Backend capability verification failed")


def metadata_contract(metadata):
    return (metadata.get("Requires-Python"), sorted(metadata.get_all("Requires-Dist", [])),
            sorted(metadata.get_all("Provides-Extra", [])))


def requirements(site):
    directory = site / "studio/backend/requirements"
    return {p.relative_to(site).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file() and
            "__pycache__" not in p.parts and p.suffix != ".pyc"}


def wheel_contract(wheel, site, old_dist, expected):
    with zipfile.ZipFile(wheel) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        require(len(set(names)) == len(names), "Duplicate wheel members")
        metas = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metas) == 1, "Unexpected backend wheel metadata")
        new_dist = metas[0].split("/")[0]
        require(bool(DIST.fullmatch(new_dist)), "Unexpected backend distribution")
        for n in names:
            parts = n.split("/")
            require(len(parts) >= 2 and all(p not in {"", ".", ".."} for p in parts)
                    and "\\" not in n and parts[0] in ROOTS | {new_dist},
                    "Wheel contains files outside the personal backend packages")
        meta = email.message_from_bytes(archive.read(metas[0]))
        old_meta = email.message_from_bytes((site / old_dist / "METADATA").read_bytes())
        require(meta.get("Name", "").lower() == "unsloth" and meta.get("Version") == expected and
                metadata_contract(meta) == metadata_contract(old_meta),
                "Backend dependency metadata changed")
        wheel_meta = email.message_from_bytes(archive.read(new_dist + "/WHEEL"))
        require(wheel_meta.get("Root-Is-Purelib", "").lower() == "true" and
                "py3-none-any" in wheel_meta.get_all("Tag", []),
                "Backend wheel is not platform-independent Python")
        require("studio/backend/personal-vulkan" in names, "Not a personal Vulkan backend")
        require(archive.read(new_dist + "/entry_points.txt") ==
                (site / old_dist / "entry_points.txt").read_bytes(), "CLI entry points changed")
        new_requirements = {n: hashlib.sha256(archive.read(n)).hexdigest()
                            for n in names if n.startswith("studio/backend/requirements/")}
        require(new_requirements == requirements(site), "Backend requirement files changed")
        # Reject overlap with a foreign installed package before uv can write.
        current_record = set(row[0] for row in csv.reader(
            (site / old_dist / "RECORD").read_text().splitlines()) if row)
        for n in names:
            if n.startswith(new_dist + "/"):
                continue
            existing = site / n
            require(not any(parent.is_symlink() for parent in (existing, *existing.parents)
                            if parent.is_relative_to(site)),
                    "New wheel would write through a symlink")
            require(not existing.exists() or n in current_record,
                    "New wheel overlaps an unowned installed file")
        for n in current_record:
            parts = PurePosixPath(n).parts
            if (site / n).resolve() == (Path(sys.prefix) / "bin/unsloth").resolve():
                continue
            require(len(parts) >= 2 and parts[0] in ROOTS | {old_dist}
                    and ".." not in parts, "Installed backend owns unexpected paths")
        return new_dist


def update(root, wheel, expected):
    venv, site = location(root)
    recover(venv, site)
    dist = importlib.metadata.distribution("unsloth")
    old_version = dist.version
    dirs = list(site.glob("unsloth-*.dist-info"))
    require(len(dirs) == 1 and DIST.fullmatch(dirs[0].name) and
            not dirs[0].is_symlink(), "Ambiguous installed backend metadata")
    old_dist = dirs[0].name
    manifest_path = venv / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("package") == "unsloth" and manifest.get("no_torch") is True and
            manifest.get("package_version") == old_version and
            Path(manifest.get("prefix", "")).resolve() == venv.resolve() and
            (site / "studio/backend/personal-vulkan").is_file(),
            "Existing install manifest is not a compatible personal environment")
    capabilities(venv, old_version)
    if old_version == expected:
        return
    new_dist = wheel_contract(wheel, site, old_dist, expected)
    require(not (site / new_dist).exists(), "New backend metadata already exists")
    uv = next((p for p in (root / "uv/uv", root / "bin/uv")
               if p.is_file() and os.access(p, os.X_OK)), None)
    require(uv is not None, "Managed uv executable is unavailable")
    entries = sorted("site/" + name for name in ROOTS | {old_dist, new_dist})
    entries += ["bin/unsloth", MANIFEST]
    require(not (venv / "bin").is_symlink() and
            all(not target(venv, site, key).is_symlink() for key in entries),
            "Backend package entries are symlinked")
    present = [key for key in entries if target(venv, site, key).exists()]
    journal = venv / JOURNAL
    # Bound the rollback copy to package files. An unexpected generated tree
    # is not a reason to duplicate a multi-gigabyte runtime.
    backup_bytes = 0
    for key in present:
        live = target(venv, site, key)
        paths = live.rglob("*") if live.is_dir() else [live]
        for item in paths:
            if item.is_file() and not item.is_symlink():
                backup_bytes += item.stat().st_size
                require(backup_bytes <= 512 * 1024 * 1024,
                        "Installed backend contains unusually large package files")
    # Copy only distribution files, not the Python environment or caches.
    # A half-copied temporary directory is never used as a rollback journal.
    with tempfile.TemporaryDirectory(prefix=".personal-wheel-stage-", dir=venv) as temp:
        stage = Path(temp)
        for key in present:
            dst = stage / "backup" / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_entry(target(venv, site, key), dst)
        state = {"entries": entries, "present": present, "committed": False}
        publish_json(stage / "state.json", state)
        stage.rename(journal)
        sync_dir(venv)
    try:
        # Explicit flags plus a clean uv/pip environment prevent inherited
        # target/prefix, indexes or cache policy from redirecting this operation.
        uv_env = {key: value for key, value in os.environ.items()
                  if not key.startswith(("UV_", "PIP_", "PYTHON"))
                  and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}
        subprocess.run([str(uv), "--no-config", "pip", "install", "--offline", "--no-cache",
                        "--no-deps", "--no-python-downloads", "--reinstall", "--link-mode",
                        "copy", "--python", sys.executable, str(wheel)], check=True, timeout=180,
                       env=uv_env, pass_fds=lock_fds())
        # Only the revision changes; the requirement digests and environment
        # were validated against the old install before mutation.
        manifest["package_version"] = expected
        publish_json(manifest_path, manifest)
        capabilities(venv, expected)
        state["committed"] = True
        publish_json(journal / "state.json", state)
    except BaseException as error:
        recover(venv, site)
        raise RuntimeError("Offline backend replacement failed; previous backend restored") from error
    recover(venv, site)


def main():
    root = Path(sys.argv[1])
    lock_fds()
    if sys.argv[2] == "--recover":
        recover(*location(root))
        return 0
    try:
        update(root, Path(sys.argv[2]), sys.argv[3])
    except Incompatible as error:
        # A failed post-install capability check must not be mistaken for an
        # ordinary dependency change: recover() has already restored the old tree.
        print(str(error), file=sys.stderr)
        return 42
    return 0


if __name__ == "__main__":
    sys.exit(main())
