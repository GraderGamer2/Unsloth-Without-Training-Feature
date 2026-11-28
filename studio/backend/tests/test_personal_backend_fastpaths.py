# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only regressions for model metadata, document chunking and streaming fast paths."""
from __future__ import annotations

import ast
import asyncio
import bisect
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import re
import runpy
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Callable, Optional
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]


def source_defs(path, names, *, class_name=None, globals_=None):
    source = BACKEND / path
    body = ast.parse(source.read_text(encoding="utf-8")).body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    for n in nodes:
        if isinstance(n, ast.FunctionDef) and class_name:
            n.decorator_list = []
    namespace = dict(globals_ or {}, __name__=__name__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class FastPathsTest(unittest.TestCase):
    def test_unknown_quant_offsets(self):
        constants = ModuleType("gguf.constants")
        constants.GGML_QUANT_SIZES = {0: (1, 4)}
        layout = ModuleType("core.inference.offload_layout")
        layout._BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
        layout._MOE_EXPERT_RE = re.compile(r"^ffn_(up|gate|down|gate_up)_(exps|chexps)\.weight$")
        scan = source_defs(
            "core/inference/llama_cpp.py", {"_gguf_scan_tensor_bytes"},
            class_name="LlamaCppBackend",
            globals_=dict(struct=struct, math=math, os=os, bisect=bisect,
                          Optional=Optional, Callable=Callable, _LAZY_READ_TENSORS={}),
        )["_gguf_scan_tensor_bytes"]
        rng = random.Random(7735)
        with patch.dict(sys.modules, {"gguf.constants": constants, "core.inference.offload_layout": layout}):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "test.gguf"
                for _ in range(120):
                    offsets = [rng.randrange(0, 20) * 32 for _ in range(50)]
                    rows = [(f"blk.{i}.ffn_up_exps.weight" if i % 3 == 0 else f"tensor.{i}",
                             0 if i % 5 == 0 else 999, offset)
                            for i, offset in enumerate(offsets)]
                    header = bytearray(struct.pack("<IIQQ", 0x46554747, 2, len(rows), 0))
                    for name, quant, offset in rows:
                        raw = name.encode()
                        header.extend(struct.pack("<Q", len(raw)) + raw)
                        header.extend(struct.pack("<IQIQ", 1, 4, quant, offset))
                    header.extend(bytes((-len(header)) % 32))
                    data_end = 640
                    path.write_bytes(header + bytes(data_end))
                    ordered = sorted(offsets)
                    expected = {}
                    experts = 0
                    kept = {name for i, (name, _, _) in enumerate(rows) if i % 2 == 0}
                    for i, (name, quant, offset) in enumerate(rows):
                        later = ordered[bisect.bisect_right(ordered, offset):]
                        size = 16 if quant == 0 else max(0, (later[0] if later else data_end) - offset)
                        if i % 3 == 0:
                            experts += size
                        if name in kept:
                            expected[name] = size
                    arch, named, expert_bytes = scan(str(path), also=kept.__contains__)
                    self.assertIsNone(arch)
                    self.assertEqual(named, expected)
                    self.assertEqual(expert_bytes, experts)

    def test_split_layout_reader_reuse(self):
        lookup = ModuleType("core.inference.lookup_tables")
        lookup.is_per_layer_embedding = lambda name: False
        gguf = ModuleType("gguf")
        gguf.GGUFEndian = object()
        gguf.GGUFValueType = object()
        calls = []

        class Reader:
            def __init__(self, path):
                calls.append(path)
                first = "00001-of-" in path
                values = {"split.count": 2}
                if first:
                    values.update({
                        "general.architecture": "llama",
                        "llama.block_count": 2,
                        "llama.attention.head_count_kv": 1,
                        "llama.attention.key_length": 4,
                        "llama.attention.value_length": 4,
                    })
                self.fields = {k: SimpleNamespace(contents=lambda v=v: v) for k, v in values.items()}
                rows = ([("token_embd.weight", 64), ("blk.0.ffn_up.weight", 80)] if first
                        else [("blk.1.ffn_down.weight", 96), ("output.weight", 128)])
                self.tensors = [SimpleNamespace(name=n, n_bytes=b) for n, b in rows]

        gguf.GGUFReader = Reader
        with patch.dict(sys.modules, {
            "core.inference.lookup_tables": lookup, "gguf": gguf, "numpy": ModuleType("numpy"),
        }):
            read = runpy.run_path(str(BACKEND / "core/inference/offload_layout.py"))["layout_from_gguf"]
            with tempfile.TemporaryDirectory() as tmp:
                paths = [str(Path(tmp) / f"model-{i:05d}-of-00002.gguf") for i in (1, 2)]
                for p in paths:
                    Path(p).touch()
                for start in paths:
                    calls.clear()
                    result = read(start, all_shards=True)
                    self.assertTrue(result.complete)
                    self.assertEqual(result.arch, "llama")
                    self.assertEqual(result.tensor_bytes, 368)
                    self.assertEqual([b.index for b in result.blocks], [0, 1])
                    self.assertEqual(len(calls), 2)
                    self.assertCountEqual(calls, paths)
                Path(paths[1]).unlink()
                self.assertFalse(read(paths[0], all_shards=True).complete)

    def test_rag_overlap_equivalence(self):
        namespace = source_defs("core/rag/chunking.py", {"_Piece", "_merge"},
                                globals_={"dataclass": dataclass})
        piece, merge = namespace["_Piece"], namespace["_merge"]

        def reference(pieces, starts, maximum, overlap):
            chunks, buf, positions, tokens = [], [], [], 0

            def flush():
                raw = "".join(p.text for p in buf)
                if raw.strip():
                    chunks.append((raw.strip(), positions[0] + len(raw) - len(raw.lstrip()),
                                   positions[0] + len(raw.rstrip())))

            for p, start in zip(pieces, starts):
                if buf and tokens + p.token_count > maximum:
                    flush()
                    budget = min(overlap, max(0, maximum - p.token_count))
                    carry, carry_pos, used = [], [], 0
                    for prev, pos in zip(reversed(buf), reversed(positions)):
                        if used + prev.token_count > budget:
                            break
                        carry.insert(0, prev)
                        carry_pos.insert(0, pos)
                        used += prev.token_count
                    buf, positions, tokens = carry, carry_pos, used
                buf.append(p)
                positions.append(start)
                tokens += p.token_count
            if buf:
                flush()
            return chunks

        rng = random.Random(680)
        for _ in range(1000):
            pieces = [piece(rng.choice(["x", " y", " ", "\n", "abc"]), rng.randrange(0, 8))
                      for _ in range(rng.randrange(0, 250))]
            starts, pos = [], 0
            for p in pieces:
                starts.append(pos)
                pos += len(p.text)
            maximum, overlap = rng.randrange(1, 40), rng.randrange(0, 50)
            self.assertEqual(merge(pieces, starts, maximum, overlap),
                             reference(pieces, starts, maximum, overlap))

    def test_stream_fragmentation_equivalence(self):
        http = ModuleType("core.inference.http_stream")
        http.__dict__.update(runpy.run_path(str(BACKEND / "core/inference/http_stream.py")))
        with patch.dict(sys.modules, {"core.inference.http_stream": http}):
            decoder = runpy.run_path(str(BACKEND / "core/inference/sse.py"))["SSELineDecoder"]

        class Response:
            def __init__(self, pieces):
                self.pieces = pieces
                self.closed = False

            async def aiter_text(self):
                try:
                    for piece in self.pieces:
                        yield piece
                finally:
                    self.closed = True

        def expected(text):
            if not text:
                return []
            lines = re.split(r"\r\n|\r|\n", text)
            if lines[-1] == "":
                lines.pop()
            return lines

        async def check():
            rng = random.Random(7735)
            alphabet = ["a", "data: ", "\r", "\n", "\u2028", "\u2029", "\x85", "😺"]
            for _ in range(1500):
                text = "".join(rng.choices(alphabet, k=rng.randrange(0, 120)))
                points = sorted([0, len(text)] + [
                    rng.randrange(len(text) + 1) for _ in range(30)
                ])
                response = Response([text[a:b] for a, b in zip(points, points[1:])])
                self.assertEqual([line async for line in http.aiter_sse_lines(response)],
                                 expected(text))
                self.assertTrue(response.closed)

                # Byte splits can bisect UTF-8, CRLF, or a leading BOM.
                wire = ("\ufeff" + text).encode("utf-8")
                points = sorted([0, len(wire)] + [
                    rng.randrange(len(wire) + 1) for _ in range(30)
                ])
                parsed, lines = decoder(), []
                for a, b in zip(points, points[1:]):
                    piece = rng.choice([bytes, bytearray, memoryview])(wire[a:b])
                    lines.extend(parsed.feed(piece))
                lines.extend(parsed.flush())
                self.assertEqual(lines, expected(text))

            # Large fragmented JSON/tool-call line, followed by a split CRLF and empty line.
            pieces = ["data: "] + ["x" * 256] * 4096 + ["\r", "\n", "\r", "\n"]
            response = Response(pieces)
            want = ["data: " + "x" * (256 * 4096), ""]
            self.assertEqual([line async for line in http.aiter_sse_lines(response)], want)
            parsed, lines = decoder(), []
            for piece in pieces:
                lines.extend(parsed.feed(piece))
            lines.extend(parsed.flush())
            self.assertEqual(lines, want)

            response = Response(["data: first\n\n", "data: second\n\n"])
            iterator = http.aiter_sse_lines(response)
            self.assertEqual(await iterator.__anext__(), "data: first")
            await iterator.aclose()
            self.assertTrue(response.closed)

        asyncio.run(check())


class RagCorrectnessTests(unittest.TestCase):
    def test_transient_scopes_never_schedule_vector_caches(self):
        ns = source_defs("core/rag/vec_cache.py", {"enabled_for_scope", "_budget"},
                         globals_=dict(os=os, enabled=lambda: True))
        self.assertFalse(ns["enabled_for_scope"]("research_scrape_" + "a" * 32))
        self.assertTrue(ns["enabled_for_scope"]("persistent-knowledge-base"))
        for value in ("inf", "-inf", "nan", "bad"):
            with patch.dict(os.environ, {"RAG_VEC_CACHE_MB": value}):
                self.assertEqual(ns["_budget"](), 0)

    def test_vector_cutoff_ties_choose_lowest_rowids(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        top = source_defs("core/rag/vec_cache.py", {"_top"})["_top"]
        rng = random.Random(680)
        for _ in range(250):
            n = rng.randrange(1, 150)
            scores = np.array([rng.randrange(4) for _ in range(n)], dtype=np.float32)
            rowids = np.array(rng.sample(range(1, 10000), n), dtype=np.int64)
            k = rng.randrange(n + 1)
            expected = sorted(range(n), key=lambda i: (-scores[i], rowids[i]))[:k]
            actual = top(np, scores, k, rowids)
            self.assertCountEqual(actual.tolist(), expected)


class AudioStartupTests(unittest.TestCase):
    def test_setup_and_callback_failures_release_audio_resources(self):
        import json
        import shutil
        from contextlib import nullcontext

        class Unavailable(RuntimeError):
            pass

        reservations, directories, stopped, adopted = [], [], [], []
        class Reservation:
            closed = False
            def close(self):
                if self.closed:
                    raise AssertionError("reservation closed twice")
                self.closed = True

        def reserve():
            reservation = Reservation()
            reservations.append(reservation)
            return reservation, 12345

        with tempfile.TemporaryDirectory() as root:
            def mkdtemp(**kwargs):
                directory = tempfile.mkdtemp(dir=root, **kwargs)
                directories.append(Path(directory))
                return directory

            process = SimpleNamespace(pid=42, stdout=None)
            class Server:
                def __init__(self, process, port, model, model_id, backend, config_dir):
                    self.config_dir = config_dir
                def stop(self):
                    stopped.append(self.config_dir)
                    shutil.rmtree(self.config_dir)

            ns = source_defs("core/inference/audio_cpp_server.py", {"_start_once"},
                             class_name="AudioCppServer",
                             globals_=dict(Path=Path, tempfile=SimpleNamespace(mkdtemp=mkdtemp),
                                           json=json, os=os, shutil=shutil, nullcontext=nullcontext,
                                           AudioCppModel=object, Optional=Optional,
                                           Any=__import__("typing").Any, threading=__import__("threading"),
                                           _reserve_free_port=reserve, _cpu_threads=lambda: 1,
                                           _NO_CUDA_GRAPH_FAMILIES=set(), logger=SimpleNamespace(info=lambda *a: None),
                                           is_process_shutting_down=lambda: False, child_env=lambda _: {},
                                           child_popen_kwargs=lambda: {}, personal_vulkan=lambda: True,
                                           subprocess=SimpleNamespace(PIPE=-1, STDOUT=-2, DEVNULL=-3,
                                                                      Popen=lambda *a, **k: process),
                                           adopt_pid=adopted.append, AudioCppUnavailableError=Unavailable))
            start = ns["_start_once"]
            model = SimpleNamespace(family="test", id="test")
            with self.assertRaises(TypeError):
                start(Server, model, "model", "cpu", "binary", {"bad": object()}, None, None)
            self.assertTrue(reservations[-1].closed)
            self.assertFalse(directories[-1].exists())
            self.assertFalse(adopted)

            def fail_callback(child):
                self.assertIs(child, process)
                raise RuntimeError("callback failed")

            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                start(Server, model, "model", "cpu", "binary", {}, None, fail_callback)
            self.assertTrue(reservations[-1].closed)
            self.assertEqual(adopted, [42])
            self.assertEqual(stopped, [directories[-1]])
            self.assertFalse(directories[-1].exists())


class AdditionalCorrectnessTests(unittest.TestCase):
    def test_download_speed_invalid_numbers(self):
        ns = source_defs("utils/download_speed_limit.py",
                         {"normalize_speed_limit_mbps", "set_download_speed_limit_mbps",
                          "speed_limit_bytes_per_second"},
                         globals_=dict(math=math, Any=__import__("typing").Any, Optional=Optional))
        for value in (False, True, 10 ** 1000, 1e308, float("nan"), float("inf")):
            self.assertIsNone(ns["normalize_speed_limit_mbps"](value))
            with self.assertRaises(ValueError):
                ns["set_download_speed_limit_mbps"](value)
        self.assertIsNone(ns["speed_limit_bytes_per_second"]("1e308"))
        self.assertEqual(ns["speed_limit_bytes_per_second"]("8"), 1000000)

    def test_web_cleanup_rolls_back_each_failed_delete(self):
        calls = []
        class Conn:
            in_transaction = True
            def rollback(self):
                calls.append("rollback")
            def close(self):
                calls.append("close")
                raise OSError("close failed")
        def bulk(conn, ids):
            calls.append(("bulk", ids))
            raise RuntimeError("bulk failed")
        def single(conn, doc_id):
            calls.append(doc_id)
            if doc_id == "a":
                raise RuntimeError("partial delete")
        ns = source_defs("core/rag/web_rank.py", {"_cleanup_web_scope"},
                         globals_=dict(store=SimpleNamespace(delete_documents=bulk, delete_document=single),
                                       logger=SimpleNamespace(warning=lambda *a, **k: None)))
        ns["_cleanup_web_scope"](Conn(), ["a", "b"])
        self.assertEqual(calls, ["rollback", ("bulk", ["a", "b"]), "rollback",
                                 "a", "rollback", "b", "close"])
        class Broken(Conn):
            def rollback(self):
                raise OSError("rollback failed")
        calls.clear()
        ns["_cleanup_web_scope"](Broken(), ["a"])
        self.assertEqual(calls, ["close"])

    def test_model_listing_cache_includes_recursive_setting(self):
        path = BACKEND / "core/inference/model_cache_search.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "search_other_locations")
        # Postponed annotations mirror the source module.
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        listings = {}
        rows = {}
        scans = []
        recursive = False
        def put(memo, key, value):
            memo[key] = value
        def scan(path, recurse):
            scans.append(recurse)
            return ("nested",) if recurse else ()
        ns = dict(account_scope=lambda: "owner", _location_rows_memo=rows,
                  _location_listings_memo=listings, _memo_get=lambda m, k: m.get(k),
                  _memo_put=put, _known_cache_rows=lambda: (),
                  _custom_folder_rows=lambda: (("folder", Path("/folder"), recursive, "folder"),),
                  _build_custom_folder_listing=scan, _bounded=lambda raw, fn: fn())
        mod = ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[]))
        exec(compile(mod, str(path), "exec"), ns)
        search = ns["search_other_locations"]
        self.assertIsNone(search(iter, lambda x: x))
        recursive = True
        rows.clear()
        self.assertEqual(search(iter, lambda x: x), "nested")
        self.assertEqual(scans, [False, True])

    def test_cancelled_image_follower_frees_batch_capacity(self):
        import secrets
        path = BACKEND / "core/inference/image_request_batching.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        from dataclasses import field
        ns = dict(__name__=__name__, asyncio=asyncio, os=os, secrets=secrets, dataclass=dataclass,
                  field=field, SEED_MASK=(1 << 32) - 1, BATCH_MAX_DEFAULT=3, POLL_S=0.001,
                  _OPEN={}, logger=SimpleNamespace(info=lambda *a: None))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                     str(path), "exec"), ns)
        async def check():
            busy = True
            rendered = []
            async def render(prompts, seeds):
                rendered.extend(prompts)
                return {"images": prompts, "seeds": seeds}
            async def submit(prompt):
                return await ns["generate"]("key", prompt, 1, None,
                                            busy=lambda: busy, run_batch=render)
            leader = asyncio.create_task(submit("leader"))
            await asyncio.sleep(0)
            follower = asyncio.create_task(submit("cancelled"))
            await asyncio.sleep(0)
            follower.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await follower
            self.assertEqual(next(iter(ns["_OPEN"].values())).images, 1)
            next_follower = asyncio.create_task(submit("next"))
            await asyncio.sleep(0)
            busy = False
            first, second = await asyncio.wait_for(asyncio.gather(leader, next_follower), 2)
            self.assertEqual(rendered, ["leader", "next"])
            self.assertEqual(first["images"], ["leader"])
            self.assertEqual(second["images"], ["next"])
            self.assertFalse(ns["_OPEN"])
        with patch.dict(os.environ, {"UNSLOTH_STUDIO_IMAGE_BATCH_MAX": "3"}):
            asyncio.run(check())


class SevenAreaRegressionTests(unittest.TestCase):
    def test_audio_stale_unload_does_not_stop_new_model(self):
        import threading
        ns = source_defs("core/inference/audio_cpp_backend.py", {"unload_model"},
                         class_name="AudioCppBackend")
        stopped = []
        backend = SimpleNamespace(models={"new": {}}, active_model_name="new", _model=object(),
                                  _server_lock=threading.RLock(), _stop_server_locked=lambda: stopped.append(True))
        self.assertTrue(ns["unload_model"](backend, "old"))
        self.assertEqual(backend.active_model_name, "new")
        self.assertEqual(stopped, [])
        self.assertTrue(ns["unload_model"](backend, "new"))
        self.assertIsNone(backend.active_model_name)
        self.assertEqual(stopped, [True])

    def test_audio_exit_between_poll_and_signal_is_reaped(self):
        import shutil
        import subprocess
        calls = []
        class Proc:
            pid = 123
            def poll(self):
                return None
            def terminate(self):
                raise ProcessLookupError()
            def wait(self, timeout):
                calls.append(timeout)
        ns = source_defs("core/inference/audio_cpp_server.py", {"stop"},
                         class_name="AudioCppServer",
                         globals_=dict(shutil=shutil, subprocess=subprocess,
                                       forget_pid=lambda pid: calls.append(pid)))
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "config"
            folder.mkdir()
            ns["stop"](SimpleNamespace(process=Proc(), _config_dir=folder))
            self.assertFalse(folder.exists())
        self.assertEqual(calls, [10, 123])

    def test_repeat_kv_stash_does_not_delete_live_manifest(self):
        import threading
        entry = {"gguf": "model", "slots": []}
        old = {"gguf": "model", "slots": [{"filename": "old"}]}
        saved = [old, entry]
        deleted = []
        ns = source_defs("core/inference/llama_keepwarm.py", {"stash_unload_kv"},
                         globals_=dict(_lock=threading.Lock(), _unload_kv=saved,
                                       _trim_unload_kv=lambda *a, **k: None,
                                       _delete_resume_files=deleted.append))
        ns["stash_unload_kv"](entry)
        self.assertEqual(len(saved), 1)
        self.assertIs(saved[0], entry)
        self.assertEqual(deleted, [old])
        ns["stash_unload_kv"](entry)
        self.assertEqual(deleted, [old])

    def test_manifest_rejects_non_regular_files_even_with_unknown_size(self):
        import stat
        ns = source_defs("hub/utils/download_manifest.py", {"verify_against_disk"},
                         globals_=dict(Manifest=object, VerifyResult=SimpleNamespace,
                                       Path=Path, stat=stat, expected_path_is_safe=lambda _: True))
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / "dir.gguf").mkdir()
            (base / "weights").write_bytes(b"123")
            (base / "alias.gguf").symlink_to(base / "weights")
            (base / "truncated.gguf").write_bytes(b"x")
            m = SimpleNamespace(expected_files=[
                SimpleNamespace(path="dir.gguf", size=0),
                SimpleNamespace(path="alias.gguf", size=3),
                SimpleNamespace(path="truncated.gguf", size=3)])
            result = ns["verify_against_disk"](m, base)
            self.assertFalse(result.ok)
            self.assertEqual(result.missing, ("dir.gguf",))
            self.assertEqual(result.size_mismatched, ("truncated.gguf",))

    def test_corrupted_download_heartbeat_never_raises(self):
        ns = source_defs("hub/utils/download_heartbeat.py", {"read"},
                         globals_=dict(Path=Path, Optional=Optional))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "heartbeat"
            for malformed in (b"\xff", "²".encode(), b"9" * 5000, b"-1", b"bad"):
                path.write_bytes(malformed)
                self.assertIsNone(ns["read"](str(path)))
            path.write_bytes(b" 123\n")
            self.assertEqual(ns["read"](str(path)), 123)


class LifecycleTests(unittest.TestCase):
    def test_response_close_error_still_closes_lines(self):
        ns = runpy.run_path(str(BACKEND / "core/inference/http_stream.py"))
        class Response:
            closed = False
            async def aiter_text(self):
                try:
                    yield "data: first\n"
                    yield "data: second\n"
                finally:
                    self.closed = True
            async def aclose(self):
                raise OSError("close failed")
        async def check():
            response = Response()
            lines = ns["closing_response_lines"](response)
            self.assertEqual(await lines.__anext__(), "data: first")
            with self.assertRaisesRegex(OSError, "close failed"):
                await lines.aclose()
            self.assertTrue(response.closed)
        asyncio.run(check())

    def test_task_cancellation_drains_pending_provider_read(self):
        import contextlib
        import threading
        from collections.abc import AsyncIterator
        path = BACKEND / "core/inference/external_tool_transport.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        watcher = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == "_await_cancel")
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "OAICompatTransport")
        cancel = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)
                      and n.name == "_cancellable")
        cancel.decorator_list = []
        ns = dict(asyncio=asyncio, contextlib=contextlib, threading=threading,
                  AsyncIterator=AsyncIterator, _CANCEL_POLL_S=0.001)
        exec(compile(ast.Module(body=[watcher, cancel], type_ignores=[]), str(path), "exec"), ns)

        async def check():
            waiting = asyncio.Event()
            released = asyncio.Event()
            async def provider():
                try:
                    waiting.set()
                    await asyncio.Future()
                    yield "unreachable"
                finally:
                    await asyncio.sleep(0)
                    released.set()
            async def consume():
                async for _ in ns["_cancellable"](provider(), threading.Event()):
                    pass
            task = asyncio.create_task(consume())
            await waiting.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(released.is_set())
        asyncio.run(check())

    def test_late_tool_output_is_not_retained_after_completion(self):
        import queue
        loggers = ModuleType("loggers")
        loggers.get_logger = lambda name: SimpleNamespace()
        controller = ModuleType("core.inference.tool_loop_controller")
        controller.redact_studio_credentials = lambda text: text
        original_queue = queue.Queue
        enqueued = []
        callbacks = []

        class RecordingQueue(original_queue):
            def put(self, item, *args, **kwargs):
                enqueued.append(item)
                return super().put(item, *args, **kwargs)

        def invoke(callback):
            callbacks.append(callback)
            callback("first output!")
            return "result"

        with patch.dict(sys.modules, {"loggers": loggers,
                                      "core.inference.tool_loop_controller": controller}):
            ns = runpy.run_path(str(BACKEND / "core/inference/tool_stream_exec.py"))
            with patch.object(queue, "Queue", RecordingQueue):
                events = list(ns["stream_tool_execution"](invoke, tool_name="test"))
                self.assertEqual("".join(e.get("text", "") for e in events), "first output!")
                before = len(enqueued)
                callbacks[0]("late text")
                callbacks[0]({"browser": "late action"})
                self.assertEqual(len(enqueued), before)

    def test_remote_embedding_indices_are_complete_and_ordered(self):
        external = ModuleType("core.inference.external_provider")
        external._append_provider_path = lambda base, path: base + "/" + path
        client = SimpleNamespace(base_url="http://example.invalid",
                                 set_header_context=lambda *_: None,
                                 _auth_headers=lambda: {})
        ns = source_defs("core/rag/embed_connection.py",
                         {"ConnectionEmbeddingBackend", "ConnectionEmbedderError"},
                         globals_=dict(_ATTEMPTS=1,
                                       _RETRY_STATUSES=frozenset(),
                                       _client_for=lambda _: (client, {"id": "test"}, "model")))
        backend = ns["ConnectionEmbeddingBackend"].__new__(ns["ConnectionEmbeddingBackend"])
        backend._attempts = 1

        def post(data):
            backend._http = SimpleNamespace(post=lambda *a, **k:
                SimpleNamespace(status_code=200, json=lambda: {"data": data}))
            return backend._post("model", ["a", "b"])

        with patch.dict(sys.modules, {"core.inference.external_provider": external}):
            self.assertEqual(post([{"index": 1, "embedding": [2]},
                                   {"index": 0, "embedding": [1]}]), [[1], [2]])
            self.assertEqual(post([{"embedding": [1]}, {"embedding": [2]}]), [[1], [2]])
            for indices in ((0, 0), (0, 2), (0, None), (False, 1)):
                with self.assertRaises(ns["ConnectionEmbedderError"]):
                    post([{"index": i, "embedding": [n]} for n, i in enumerate(indices)])

    def test_hash_cache_rejects_replaced_file_with_same_size_and_mtime(self):
        import hashlib
        ns = runpy.run_path(str(BACKEND / "core/inference/cached_weight_sha.py"))
        digest = ns["file_sha256"]
        state = digest.__globals__
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            record = Path(tmp) / "hashes.json"
            state["_store_path"] = lambda: record
            path.write_bytes(b"aaaa")
            original = path.stat()
            self.assertEqual(digest(str(path)), hashlib.sha256(b"aaaa").hexdigest())
            state["_local"].clear()
            state["_persisted"] = None
            self.assertEqual(digest(str(path)), hashlib.sha256(b"aaaa").hexdigest())
            replacement = Path(tmp) / "replacement.bin"
            replacement.write_bytes(b"bbbb")
            os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
            replacement.replace(path)
            self.assertEqual(digest(str(path)), hashlib.sha256(b"bbbb").hexdigest())
            state["_MAX_ENTRIES"] = 2
            for index in range(4):
                other = Path(tmp) / str(index)
                other.write_bytes(bytes([index]))
                self.assertIsNotNone(digest(str(other)))
            self.assertLessEqual(len(state["_local"]), 2)
            self.assertLessEqual(len(state["_persisted"]), 2)


class CacheAndBatchOptimizationTests(unittest.TestCase):
    def test_hub_metadata_cache_bounded_and_reuses_recent_hits(self):
        ns = runpy.run_path(str(BACKEND / "core/inference/cached_weight_sha.py"))
        lookup = ns["hub_sha256s"]
        state = lookup.__globals__
        state["_MAX_ENTRIES"] = 3
        state["_offline"] = lambda: False
        requests = []
        hub = ModuleType("huggingface_hub")
        def paths(repo, names, **kwargs):
            requests.append(tuple(names))
            return [SimpleNamespace(path=name, lfs={"sha256": "a" * 64})
                    for name in names if name != "missing"]
        hub.HfApi = lambda: SimpleNamespace(get_paths_info=paths)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            revision = "b" * 40
            self.assertEqual(lookup("a/b", ["a", "b", "missing"], revision=revision),
                             {"a": "a" * 64, "b": "a" * 64})
            lookup("a/b", ["a", "missing"], revision=revision)
            lookup("a/b", ["c"], revision=revision)
            self.assertEqual(len(state["_hub"]), 3)
            lookup("a/b", ["a", "missing"], revision=revision)
            self.assertEqual(requests, [("a", "b", "missing"), ("c",)])
            lookup("a/b", ["b"], revision=revision)
            self.assertEqual(requests[-1], ("b",))
            self.assertLessEqual(len(state["_hub"]), 3)

    def test_weight_hash_buffer_boundaries(self):
        import hashlib
        ns = runpy.run_path(str(BACKEND / "core/inference/cached_weight_sha.py"))
        digest = ns["file_sha256"]
        state = digest.__globals__
        state["_CHUNK"] = 7
        state["_store_path"] = lambda: None
        with tempfile.TemporaryDirectory() as tmp:
            for length in (0, 1, 7, 8, 14, 15, 1003):
                path = Path(tmp) / str(length)
                data = bytes(i % 251 for i in range(length))
                path.write_bytes(data)
                self.assertEqual(digest(str(path)), hashlib.sha256(data).hexdigest())

    def test_location_cache_preserves_order_and_evicts_only_oldest(self):
        import threading
        import time
        loggers = ModuleType("loggers")
        loggers.get_logger = lambda *_: SimpleNamespace()
        with patch.dict(sys.modules, {"loggers": loggers}):
            ns = runpy.run_path(str(BACKEND / "core/inference/model_cache_search.py"))
        cls = ns["LocationListing"]
        a, b, c = (Path("/" + name) for name in ("a", "b", "c"))
        listing = cls({}, {"one": (c, a, b, a), "two": (a, c), "three": (a, b)})
        self.assertEqual(listing.dirs_holding("ONE"), [c, a, b, a])
        self.assertEqual(listing.dirs_holding("one", "two", "three"), [a, a])
        self.assertEqual(listing.dirs_holding("one", "missing"), [])
        put, get = ns["_memo_put"], ns["_memo_get"]
        put.__globals__["_LOCATION_MEMO_MAX"] = 2
        put.__globals__["_LOCATION_TTL_SECONDS"] = 100
        memo = {}
        put(memo, "a", 1)
        put(memo, "b", 2)
        self.assertEqual(get(memo, "a"), 1)
        put(memo, "c", 3)
        self.assertEqual(set(memo), {"a", "c"})

    def test_tool_decode_filters_ids_without_changing_order(self):
        ns = runpy.run_path(str(BACKEND / "core/inference/native_tool_tokens.py"))
        decode = ns["_decode_with_token_sets"]
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: (list(ids), kwargs["skip_special_tokens"]))
        self.assertEqual(decode(tokenizer, iter([1, 2, 3, 2, 4]), frozenset({2, 3}), frozenset({3})),
                         ([1, 3, 4], False))
        self.assertEqual(decode(tokenizer, 3, frozenset({2, 3}), frozenset({3})),
                         ([3], False))
        self.assertEqual(decode(tokenizer, [1, 2], frozenset(), frozenset()), ([1, 2], True))

    def test_web_rank_batches_complete_pages_and_keeps_vector_identity(self):
        import hashlib
        import uuid
        encodes, indexed, cleaned = [], [], []
        def encode(texts, **kwargs):
            encodes.append(list(texts))
            return [("vector:" + text) for text in texts], "space-" + str(len(encodes))
        docs = {}
        def create(conn, **kwargs):
            doc_id = str(len(docs))
            docs[doc_id] = kwargs
            return doc_id
        def add(conn, scope, doc, chunks, vectors):
            indexed.append((docs[doc]["filename"], docs[doc]["embedding_model"],
                            [c.text for c in chunks], list(vectors)))
        ns = source_defs("core/rag/web_rank.py", {"retrieve_web_chunks", "_index_page_batch"},
                         globals_=dict(hashlib=hashlib, uuid=uuid,
                           config=SimpleNamespace(EMBED_BATCH=4, CHUNK_OVERLAP=0,
                                                  effective_embedding_model=lambda: "test"),
                           embeddings=SimpleNamespace(token_counter=lambda m: len,
                                                      chunk_tokens=lambda m: 16,
                                                      encode_with_identity=encode),
                           rag_db=SimpleNamespace(RAG_AVAILABLE=True, get_connection=lambda: object()),
                           Page=SimpleNamespace,
                           chunk_pages=lambda pages, **kw: [SimpleNamespace(text=t)
                                                            for t in pages[0].text.split()],
                           store=SimpleNamespace(create_document=create, add_chunks=add),
                           retrieval=SimpleNamespace(retrieve_hybrid=lambda *a, **kw: []),
                           logger=SimpleNamespace(warning=lambda *a, **kw: None),
                           _cleanup_web_scope=lambda conn, ids: cleaned.extend(ids)))
        pages = [{"title": str(i), "text": body}
                 for i, body in enumerate(["a b", "c d", "e", "f g h i j", "k"])]
        self.assertEqual(ns["retrieve_web_chunks"](pages, "q", top_n=3, min_score=0), ("", []))
        self.assertEqual(encodes, [["a", "b", "c", "d"], ["e"], ["f", "g", "h", "i", "j"], ["k"]])
        self.assertEqual([row[0] for row in indexed], ["0", "1", "2", "3", "4"])
        self.assertEqual([row[1] for row in indexed], ["space-1", "space-1", "space-2", "space-3", "space-4"])
        for _, _, chunks, vectors in indexed:
            self.assertEqual(vectors, ["vector:" + chunk for chunk in chunks])
        self.assertEqual(cleaned, ["0", "1", "2", "3", "4"])


class ResearchImportTests(unittest.TestCase):
    def test_recorded_sources_and_stable_snapshot(self):
        policy = runpy.run_path(str(BACKEND / "core/inference/web_access_policy.py"))
        module = ModuleType("core.inference.web_access_policy")
        module.check_url_access = policy["check_url_access"]
        module.normalize_website_policy = policy["normalize_website_policy"]
        with patch.dict(sys.modules, {"core.inference.web_access_policy": module}):
            ns = runpy.run_path(str(BACKEND / "core/rag/research_source_import.py"))
        run = {"sources": [{"url": "https://example.org/paper#section", "title": "Paper"}],
               "config": {"websitePolicy": {"allowedDomains": ["example.org"]}}}
        resolve = ns["resolve_source"]
        url, title, access = resolve(run, run["sources"][0]["url"],
                                     {"allowedDomains": ["example.org"]})
        self.assertEqual(url, "https://example.org/paper")
        self.assertEqual(title, "Paper")
        self.assertEqual(access["allowedDomains"], ["example.org"])
        with self.assertRaises(ValueError):
            resolve(run, "https://example.org/not-recorded", {})
        with self.assertRaises(ValueError):
            resolve(run, run["sources"][0]["url"], {"blockedDomains": ["example.org"]})
        with self.assertRaises(ValueError):
            resolve(run, run["sources"][0]["url"], {"allowedDomains": ["other.org"]})
        snapshot = ns["snapshot"]
        name, content = snapshot(url, title, "original paper text")
        self.assertTrue(name.endswith(".md"))
        self.assertIn(url, content)
        self.assertIn("original paper text", content)
        self.assertEqual(snapshot(url, title, "original paper text"), (name, content))
        with self.assertRaises(ValueError):
            snapshot(url, title, "   ")


class ResearchSynthesisValidationTests(unittest.TestCase):
    def test_rejects_tool_requests_and_recovers_without_rejecting_reports(self):
        import json
        ns = source_defs("core/research_runs.py",
                         {"_usable_synthesis_report", "_synthesis_needs_recovery"},
                         globals_=dict(re=re, json=json))
        validate = ns["_usable_synthesis_report"]
        needs = ns["_synthesis_needs_recovery"]
        malformed = [
            '<function name="fetch_content"><param name="url">Unsloth Docs | Unsloth Documentation</param></function>',
            '<tool_call>{"name":"web_search","arguments":{"query":"test"}}</tool_call>',
            '<|tool_call_start|>fetch_content',
            '{"action":"fetch","url":"https://example.test"}',
            '{"name":"fetch_content","arguments":{"url":"https://example.test"}}',
            '[{"type":"function","function":{"name":"fetch_content","arguments":"{}"}}]',
            '~~~json\n{"action":"search","query":"test"}\n~~~',
            '## Findings\n\n<function=fetch_content>',
        ]
        for text in malformed:
            self.assertEqual(validate(text, "stop"), "", text)
            self.assertTrue(needs(text, "stop"), text)
        valid = '## Findings\n\nThe agent uses tools.\n\n~~~xml\n<function name="fetch_content"/>\n~~~'
        self.assertEqual(validate(valid, "stop"), valid)
        self.assertFalse(needs(valid, "stop"))
        self.assertTrue(needs(valid, "length"))
        self.assertTrue(needs(valid, "tool_calls"))
        self.assertTrue(needs(valid, "function_call"))
        self.assertEqual(validate("  "), "")
        self.assertEqual(validate("## Results\n\nThe API accepts a tool name."), "## Results\n\nThe API accepts a tool name.")


if __name__ == "__main__":
    unittest.main()
