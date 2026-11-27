#!/usr/bin/env python3
"""Check cache fingerprints without compiling any native engine."""
from pathlib import Path
import runpy
import unittest

SCRIPT = Path(__file__).resolve().with_name("personal-build-cache-key.py")
MOD = runpy.run_path(str(SCRIPT))
ROOT = SCRIPT.parent


class CacheKeyTests(unittest.TestCase):
    def test_per_component_build_source(self):
        source = (ROOT / "build-personal-runtimes.sh").read_text()
        for engine in MOD["COMPONENTS"]:
            selected = MOD["build_source"](engine)
            self.assertIn("checkout_source " + engine + " ", selected)
            for other in MOD["COMPONENTS"]:
                if other != engine:
                    self.assertNotIn("checkout_source " + other + " ", selected)
        self.assertIn("-march=znver3", source)
        self.assertIn("-DGGML_LTO=ON", source)

    def test_runtime_key_changes_with_toolchain(self):
        for engine in MOD["COMPONENTS"]:
            first = MOD["runtime"](engine, "toolchain-a")
            self.assertEqual(first, MOD["runtime"](engine, "toolchain-a"))
            self.assertNotEqual(first, MOD["runtime"](engine, "toolchain-b"))

    def test_runtime_keys_are_distinct(self):
        keys = {MOD["runtime"](engine, "toolchain") for engine in MOD["COMPONENTS"]}
        self.assertEqual(len(keys), len(MOD["COMPONENTS"]))

    def test_generated_checkout_is_restored_on_rebuild(self):
        import subprocess
        import tempfile
        source = (ROOT / "build-personal-runtimes.sh").read_text()
        start = source.index("checkout_source() {")
        end = source.index("\ncompile() {", start)
        checkout = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            repo = work / "sources" / "llama"
            repo.mkdir(parents=True)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
            tracked = repo / "source.cpp"
            tracked.write_text("pristine")
            subprocess.run(["git", "-C", str(repo), "add", "source.cpp"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "baseline"], check=True)
            revision = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            tracked.write_text("previous patch")
            subprocess.run(["bash", "-c", checkout + '\nwork="$1"\ncheckout_source llama llama.cpp "$2"',
                            "test", str(work), revision], check=True)
            self.assertEqual(tracked.read_text(), "pristine")

    def test_runtime_archive_rejects_aliases_and_existing_destinations(self):
        import hashlib
        import io
        import shutil
        import subprocess
        import tarfile
        import tempfile
        script = ROOT / "personal-runtime-archive.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / "source"
            component = src / "whisper"
            component.mkdir(parents=True)
            shutil.copy2("/bin/true", component / "whisper-server")
            (component / "LICENSE").write_text("test")
            (component / "valid-link").symlink_to("whisper-server")
            archive = root / "archive"
            def run(mode, dest=None, check=True):
                args = ["bash", str(script), mode, "whisper", str(archive)]
                if dest is not None:
                    args.append(str(dest))
                return subprocess.run(args, check=check, capture_output=True)
            run("pack", src)
            dest = root / "output"
            run("unpack", dest)
            self.assertTrue((dest / "whisper" / "valid-link").is_symlink())
            self.assertNotEqual(run("unpack", dest, check=False).returncode, 0)
            outside = root / "outside"
            outside.mkdir()
            other = root / "other"
            other.mkdir()
            (other / "whisper").symlink_to(outside, target_is_directory=True)
            self.assertNotEqual(run("unpack", other, check=False).returncode, 0)
            self.assertEqual(list(outside.iterdir()), [])
            def corrupt(name, link=None):
                run("pack", src)
                with tarfile.open(archive / "runtime.tar", "a") as bundle:
                    entry = tarfile.TarInfo(name)
                    if link is not None:
                        entry.type = tarfile.SYMTYPE
                        entry.linkname = link
                        bundle.addfile(entry)
                    else:
                        entry.size = 1
                        bundle.addfile(entry, io.BytesIO(b"x"))
                digest = hashlib.sha256((archive / "runtime.tar").read_bytes()).hexdigest()
                (archive / "SHA256SUMS").write_text(digest + "  runtime.tar\n")
                self.assertNotEqual(run("verify", check=False).returncode, 0)
            corrupt("whisper/./LICENSE")
            corrupt("whisper/missing-link", "missing")
            corrupt("whisper/cycle", "cycle")

    def test_unambiguous_digest(self):
        self.assertNotEqual(MOD["digest"](["a", "bc"]), MOD["digest"](["ab", "c"]))


    def test_literal_patch_amendments_do_not_invalidate_unrelated_engines(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "probe.py"
            baseline = ('LLAMA = []\nENGINE_EXTRA = {"audio": []}\n'
                        'def patch_specs(component):\n'
                        '    return LLAMA if component == "llama" else ENGINE_EXTRA[component]\n')
            source.write_text(baseline)
            with patch.dict(MOD["patch_fingerprint"].__globals__, {"ROOT": root}):
                a = MOD["patch_fingerprint"]("probe.py", "audio")
                l = MOD["patch_fingerprint"]("probe.py", "llama")
                source.write_text(baseline + 'LLAMA += [("x", "before", "after")]\n')
                self.assertEqual(a, MOD["patch_fingerprint"]("probe.py", "audio"))
                self.assertNotEqual(l, MOD["patch_fingerprint"]("probe.py", "llama"))
                source.write_text(baseline + 'ENGINE_EXTRA["audio"] += [("a", "b", "c")]\n')
                self.assertNotEqual(a, MOD["patch_fingerprint"]("probe.py", "audio"))
                self.assertEqual(l, MOD["patch_fingerprint"]("probe.py", "llama"))
                source.write_text(baseline + 'LLAMA += list()\n')
                self.assertNotEqual(a, MOD["patch_fingerprint"]("probe.py", "audio"))

if __name__ == "__main__":
    unittest.main()
