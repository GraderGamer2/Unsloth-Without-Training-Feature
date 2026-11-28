# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded temporary originals; never permanent project ingestion."""
from collections import OrderedDict
from hashlib import sha256
from tempfile import TemporaryFile
from threading import RLock
from time import monotonic
from urllib.parse import urldefrag

_KEYS = ("sourceUrl", "finalUrl", "title", "retrievedAt", "contentType",
         "originalFilename", "representation", "contentSha256")


def policy_key(policy):
    policy = policy or {}
    return tuple(tuple(sorted(set(policy.get(key) or ()))) for key in ("allowedDomains", "blockedDomains"))


class CandidateCache:
    def __init__(self, max_bytes=512 * 1024 * 1024, max_entries=256, ttl=3600):
        self.max_bytes, self.max_entries, self.ttl = max_bytes, max_entries, ttl
        self.entries = OrderedDict()
        self.size = 0
        self.lock = RLock()

    def _remove(self, key):
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.size -= entry[1]
            entry[0].close()

    def _expire(self):
        for key, entry in list(self.entries.items()):
            if entry[2] <= monotonic():
                self._remove(key)

    def prune(self):
        with self.lock:
            self._expire()

    def put(self, scope, url, filename, payload, details, policy, directory):
        if not payload or len(payload) > self.max_bytes:
            return False
        key = (*scope, urldefrag(url)[0])
        metadata = {k: details[k] for k in _KEYS if k in details}
        try:
            with self.lock:
                self._expire()
                self._remove(key)
                while self.entries and (self.size + len(payload) > self.max_bytes or len(self.entries) >= self.max_entries):
                    self._remove(next(iter(self.entries)))
                # On Linux this is unlinked, including after a crash.
                handle = TemporaryFile(mode="w+b", dir=directory)
                try:
                    handle.write(payload)
                    handle.flush()
                    self.entries[key] = (handle, len(payload), monotonic() + self.ttl,
                                         filename, metadata, policy_key(policy), sha256(payload).hexdigest())
                    self.size += len(payload)
                except BaseException:
                    handle.close()
                    raise
        except OSError:
            return False
        return True

    def read(self, scope, url, policy, expected_hash):
        if not expected_hash:
            return None
        key = (*scope, urldefrag(url)[0])
        with self.lock:
            self._expire()
            entry = self.entries.get(key)
            if entry is None:
                return None
            handle, size, _, filename, metadata, saved_policy, digest = entry
            if saved_policy != policy_key(policy) or expected_hash != digest:
                return None
            try:
                handle.seek(0)
                payload = handle.read(size + 1)
            except OSError:
                self._remove(key)
                return None
            if len(payload) != size or sha256(payload).hexdigest() != digest:
                self._remove(key)
                return None
            self.entries.move_to_end(key)
            return filename, payload, dict(metadata)

    def remove(self, scope, url):
        with self.lock:
            self._remove((*scope, urldefrag(url)[0]))

    def discard(self, scope_prefix):
        with self.lock:
            for key in list(self.entries):
                if key[:len(scope_prefix)] == scope_prefix:
                    self._remove(key)


cache = CandidateCache()


def run_scope(project_id, run_id):
    from utils.account_context import current_account_id
    return current_account_id(), project_id, run_id


def remember(project_id, run_id, url, filename, payload, details, policy):
    from core.training.account_jobs import account_is_retired
    if not project_id or not run_id or account_is_retired():
        return False
    from utils.paths import ensure_dir, rag_uploads_root
    try:
        directory = ensure_dir(rag_uploads_root())
    except OSError:
        return False
    return cache.put(run_scope(project_id, run_id), url, filename, payload, details, policy, directory)
