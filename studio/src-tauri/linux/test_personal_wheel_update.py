# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the real offline updater and uv using tiny synthetic wheels, never PyPI."""
import base64
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

HELPER = Path(__file__).with_name('personal-wheel-update.py')
UV = shutil.which('uv')
MANIFEST = 'unsloth_install_manifest.json'
JOURNAL = '.personal-wheel-rollback'


def make_wheel(directory, version, *, requirements=b'unchanged\n', bad=False):
    info = f'unsloth-{version}.dist-info'
    cli = '''import importlib.metadata, json, sys
from pathlib import Path
def app():
    version = importlib.metadata.version('unsloth')
    manifest = json.loads((Path(sys.prefix) / 'unsloth_install_manifest.json').read_text())
    result = dict.fromkeys(('supports_api_only', 'supports_provision_desktop_auth', 'supports_desktop_backend_ownership', 'supports_fixed_desktop_storage', 'supports_amd_desktop', 'supports_personal_vulkan'), True)
    result.update(desktop_protocol_version=1, desktop_manageability_version=2, llama_runtime_ok=None, version=version, studio_install_ok=manifest['package_version'] == version and not BAD)
    print(json.dumps(result))
'''.replace('BAD', repr(bad))
    files = {
        'unsloth/__init__.py': f'VERSION = {version!r}\n'.encode(),
        'unsloth_cli/__init__.py': cli.encode(),
        'studio/__init__.py': b'',
        'studio/backend/personal-vulkan': b'',
        'studio/backend/requirements/runtime.txt': requirements,
        info + '/METADATA': f'Metadata-Version: 2.1\nName: unsloth\nVersion: {version}\nRequires-Python: >=3.9\n\n'.encode(),
        info + '/WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        info + '/entry_points.txt': b'[console_scripts]\nunsloth = unsloth_cli:app\n',
    }
    record = io.StringIO(newline='')
    writer = csv.writer(record)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow((name, f'sha256={digest}', len(data)))
    writer.writerow((info + '/RECORD', '', ''))
    files[info + '/RECORD'] = record.getvalue().encode()
    wheel = directory / f'unsloth-{version}-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return wheel


@unittest.skipUnless(sys.platform.startswith('linux') and UV, 'Linux and uv required')
class OfflineWheelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.venv = self.root / 'unsloth_studio'
        self.python = self.venv / 'bin/python'
        (self.root / 'uv').mkdir()
        (self.root / 'uv/uv').symlink_to(UV)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('UV_', 'PIP_', 'PYTHON')) and k not in ('VIRTUAL_ENV', 'CONDA_PREFIX')}
        self.env.update(UV_OFFLINE='1', UV_NO_CACHE='1')
        subprocess.run([UV, '--no-config', 'venv', '--offline', '--no-cache', '--no-python-downloads', '--python', sys.executable, str(self.venv)], env=self.env, check=True, capture_output=True)
        self.old = make_wheel(self.root, '2026.11.11.post1')
        self.new = make_wheel(self.root, '2026.11.11.post2')
        subprocess.run([UV, '--no-config', 'pip', 'install', '--offline', '--no-cache', '--no-deps', '--python', str(self.python), str(self.old)], env=self.env, check=True, capture_output=True)
        self.manifest = self.venv / MANIFEST
        self.manifest.write_text(json.dumps(dict(package='unsloth', package_version='2026.11.11.post1', no_torch=True, prefix=str(self.venv), requirement_files={'sentinel': 'unchanged'})))
        self.site = Path(subprocess.check_output([str(self.python), '-I', '-c', 'import sysconfig;print(sysconfig.get_path("purelib"))'], text=True).strip())
        self.unrelated = self.site / 'unrelated.py'
        self.unrelated.write_bytes(b'keep this dependency\n')
        self.project = self.root / 'project.sqlite'
        self.project.write_bytes(b'user data\n')

    def run_helper(self, *args):
        with (self.root / '.studio-runtime.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return subprocess.run([str(self.python), '-I', '-B', str(HELPER), str(self.root), *map(str, args)], env={**self.env, 'UNSLOTH_PERSONAL_WHEEL_LOCK_FD': str(lock.fileno())}, pass_fds=(lock.fileno(),), capture_output=True, text=True, timeout=60)

    def assert_version(self, version):
        self.assertEqual(json.loads(self.manifest.read_text())['package_version'], version)
        self.assertEqual(subprocess.check_output([str(self.python), '-I', '-c', 'import importlib.metadata;print(importlib.metadata.version("unsloth"))'], text=True).strip(), version)
        self.assertEqual(self.unrelated.read_bytes(), b'keep this dependency\n')
        self.assertEqual(self.project.read_bytes(), b'user data\n')
        self.assertFalse((self.venv / JOURNAL).exists())

    def test_upgrade_downgrade_and_noop(self):
        for wheel, version in [(self.new, '2026.11.11.post2'), (self.new, '2026.11.11.post2'), (self.old, '2026.11.11.post1')]:
            result = self.run_helper(wheel, version)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_version(version)

    def test_incompatible_requirements_do_not_mutate(self):
        wheel = make_wheel(self.root, '2026.11.11.post3', requirements=b'changed\n')
        result = self.run_helper(wheel, '2026.11.11.post3')
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assert_version('2026.11.11.post1')

    def test_failed_capability_rolls_back(self):
        wheel = make_wheel(self.root, '2026.11.11.post4', bad=True)
        result = self.run_helper(wheel, '2026.11.11.post4')
        self.assertNotIn(result.returncode, (0, 42), result.stderr)
        self.assertIn('previous backend restored', result.stderr)
        self.assert_version('2026.11.11.post1')

    def test_recover_interrupted_package_update(self):
        journal = self.venv / JOURNAL
        backup = journal / 'backup/site/unsloth'
        shutil.copytree(self.site / 'unsloth', backup)
        (self.site / 'unsloth/__init__.py').write_bytes(b'broken\n')
        (journal / 'state.json').write_text(json.dumps(dict(entries=['site/unsloth'], present=['site/unsloth'], committed=False)))
        result = self.run_helper('--recover')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.site / 'unsloth/__init__.py').read_bytes(), b"VERSION = '2026.11.11.post1'\n")
        self.assert_version('2026.11.11.post1')


if __name__ == '__main__':
    unittest.main()
