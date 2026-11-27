# SPDX-License-Identifier: AGPL-3.0-only
"""Lease-based idle release: no background work touches a leased embedder."""
from __future__ import annotations

import math
import os
import threading
from contextlib import contextmanager
from functools import wraps


def idle_seconds():
    try:
        value = float(os.environ.get("UNSLOTH_EMBED_IDLE_SECONDS", "60"))
        return min(value, 86400.0) if math.isfinite(value) and value > 0 else 0.0
    except ValueError:
        return 60.0


class IdleRelease:
    def __init__(self, release, delay=idle_seconds, timer_factory=threading.Timer):
        self._release = release
        self._delay = delay
        self._timer_factory = timer_factory
        self._lock = threading.RLock()
        self._active = 0
        self._generation = 0
        self._timer = None

    def _cancel(self):
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    @contextmanager
    def activity(self):
        with self._lock:
            self._cancel()
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                delay = self._delay()
                if self._active == 0 and delay > 0:
                    generation = self._generation
                    timer = self._timer_factory(delay, self._expire, args=(generation,))
                    timer.daemon = True
                    self._timer = timer
                    timer.start()

    def _expire(self, generation):
        with self._lock:
            if generation != self._generation or self._active:
                return
            self._timer = None
            self._generation += 1
            # Holding the admission lock through release prevents a new encode
            # from selecting a backend that is already being stopped.
            self._release()

    def manual_release(self, release):
        with self._lock:
            self._cancel()
            return release()

    def guarded(self, fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with self.activity():
                return fn(*args, **kwargs)
        return wrapped
