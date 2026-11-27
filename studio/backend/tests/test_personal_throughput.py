# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only checks for agent and context fast paths."""
import json
import random
import re
import unittest
from typing import Any
from test_personal_backend_fastpaths import source_defs


class ThroughputTests(unittest.TestCase):
    def test_research_selection_and_relative_json_errors(self):
        def validate(obj, *_args):
            if obj.get("ok"):
                return obj
            raise ValueError("invalid object")

        ns = source_defs("core/research/parsing.py",
                         {"_parse_and_validate_action", "_parse_and_validate_plan"},
                         globals_=dict(Any=Any, json=json, re=re,
                                       _validate_agent_action=validate, _validate_plan=validate))

        def reference(response, reasoning):
            decoder = json.JSONDecoder()
            last_error = None
            for candidate in (response, reasoning):
                valid = []
                for match in re.finditer(r"\{", candidate):
                    try:
                        value, _ = decoder.raw_decode(candidate[match.start():])
                        if isinstance(value, dict):
                            valid.append(validate(value))
                    except ValueError as exc:
                        last_error = exc
                if valid:
                    return valid[-1]
            if last_error is not None:
                raise last_error
            raise ValueError("no object")

        def outcome(fn, a, b):
            try:
                return ("value", fn(a, b))
            except json.JSONDecodeError as exc:
                return ("json", exc.msg, exc.doc, exc.pos, exc.lineno, exc.colno)
            except ValueError as exc:
                # The empty-output errors intentionally differ by API.
                return ("invalid", str(exc) if str(exc) == "invalid object" else "no object")

        rng = random.Random(7735)
        pieces = ["prose", "\n", "é", "{", '{"ok":true}', '{"ok":false}',
                  '{"ok":true,"value":{"a":2}}', '{"a":', '{"bad":"', "}", "[]"]
        for _ in range(5000):
            a, b = ("".join(rng.choices(pieces, k=rng.randrange(12))) for _ in range(2))
            expected = outcome(reference, a, b)
            self.assertEqual(outcome(lambda x, y: ns["_parse_and_validate_action"](x, y, set()), a, b),
                             expected)
            self.assertEqual(outcome(lambda x, y: ns["_parse_and_validate_plan"](x, y, 8), a, b),
                             expected)

        # CPython may reject very long numeric literals with ValueError, not
        # JSONDecodeError; ensure those still follow the original recovery loop.
        a = '{"ok":true,"n":' + "1" * 5000 + '}'
        expected = outcome(reference, a, "")
        self.assertEqual(outcome(lambda x, y: ns["_parse_and_validate_action"](x, y, set()), a, ""),
                         expected)
        self.assertEqual(outcome(lambda x, y: ns["_parse_and_validate_plan"](x, y, 8), a, ""),
                         expected)

    def test_provider_call_normalized_once(self):
        from types import SimpleNamespace
        path = "core/inference/studio_tool_loop.py"
        ns = source_defs(path, {"_normalized_call", "_mint_streamed_card_id"},
                         globals_=dict(Any=Any, json=json))
        normalize = ns["_normalized_call"]
        calls_made = []
        def tracked(call, fallback_id=""):
            calls_made.append(call.get("id"))
            return normalize(call, fallback_id)
        method = source_defs(path, {"calls"}, class_name="_Turn",
                             globals_=dict(Any=Any, _normalized_call=tracked,
                                           _mint_streamed_card_id=ns["_mint_streamed_card_id"]))["calls"]
        turn = SimpleNamespace(order=[0, 1, 2, 3], seq_by_key={}, pending_extra={},
                               healed=[], round=7, _call_is_finished=lambda key: True,
                               by_index={
                                   0: {"id":"", "function":{"name":"search", "arguments":"{}"}},
                                   1: {"id":"tool_call_0", "function":{"name":"fetch", "arguments":'{"x":1}'}},
                                   2: {"id":"repeat", "function":{"name":"search", "arguments":"broken"}},
                                   3: {"id":"invalid", "function":{"name":"", "arguments":"{}"}},
                               })
        seen, cards = {"repeat"}, set()
        out = method(turn, seen, cards)
        self.assertCountEqual(calls_made, ["", "tool_call_0", "repeat", "invalid"])
        self.assertEqual([c["id"] for c in out], ["call_7_0", "tool_call_0", "repeat_7_2"])
        self.assertEqual(out[0]["card_id"], "tool_call_1")
        self.assertEqual(out[1]["arguments"], {"x":1})
        self.assertEqual(out[2]["arguments"], {"_raw":"broken"})
        self.assertEqual(out[2]["stream_id"], "repeat")
        self.assertNotIn("invalid", cards)

    def test_tool_schema_copy_on_write(self):
        import runpy
        from pathlib import Path
        schema = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                    "core/inference/llama_tool_schema.py"))
        wrap = schema["llama_grammar_tools"]
        simple = {"function":{"name":"plain", "parameters":{"type":"object",
                   "properties":{"x":{"type":"string"}}, "required":["x"]}}}
        nested = {"function":{"name":"nested", "parameters":{"type":"object",
                  "properties":{"arg":{"type":"object",
                   "properties":{"z":{"type":"integer"}, "a":{"type":"string"}}}}}}}
        plain = [simple]
        self.assertIs(wrap(plain), plain)
        mixed = [simple, nested]
        result = wrap(mixed)
        self.assertIsNot(result, mixed)
        self.assertIs(result[0], simple)
        self.assertNotIn("anyOf", nested["function"]["parameters"]["properties"]["arg"])
        arg = result[1]["function"]["parameters"]["properties"]["arg"]
        self.assertIn("anyOf", arg)
        self.assertEqual(list(arg["properties"]), ["z", "a"])
        self.assertIs(wrap(result), result)

    def test_chunk_merge_offsets(self):
        from dataclasses import dataclass
        ns = source_defs("core/rag/chunking.py", {"_Piece", "_merge"},
                         globals_=dict(dataclass=dataclass))
        piece = ns["_Piece"]
        merge = ns["_merge"]
        rng = random.Random(7735)
        for _ in range(2000):
            texts = rng.choices([" ", "\n", "abc", "é", "\u2003", "  test  ", "猫"], k=rng.randrange(1, 30))
            starts = []
            cursor = 0
            for text in texts:
                starts.append(cursor)
                cursor += len(text)
            original = "".join(texts)
            for text, start, end in merge([piece(t, len(t)) for t in texts], starts, 16, 5):
                self.assertEqual(original[start:end], text)
                self.assertEqual(text, text.strip())

    def test_credential_prefix_scan(self):
        prefixes = ("sk-unsloth-", "desktop-")
        terms = frozenset(" \t\r\n\"',;)]}" + chr(96))
        partials = tuple(sorted({p[:n] for p in prefixes for n in range(1, len(p))},
                                key=len, reverse=True))
        scan = source_defs("core/inference/tool_stream_exec.py", {"_hold_back_partial_secret"},
                           globals_=dict(_SECRET_PREFIXES=prefixes, _SECRET_TERMINATORS=terms,
                                         _SECRET_SCAN_TAIL=128, _SECRET_PARTIAL_PREFIXES=partials))["_hold_back_partial_secret"]
        def old(text):
            tail_start = max(0, len(text) - 128)
            tail = text[tail_start:]
            best = len(text)
            for prefix in prefixes:
                index = tail.rfind(prefix)
                while index != -1:
                    absolute = tail_start + index
                    rest = text[absolute + len(prefix):]
                    if not rest or not terms.intersection(rest):
                        best = min(best, absolute)
                    index = tail.rfind(prefix, 0, index)
            for prefix in prefixes:
                for n in range(min(len(prefix)-1, len(text)), 0, -1):
                    if text.endswith(prefix[:n]):
                        best = min(best, len(text)-n)
                        break
            return best
        rng = random.Random(7735)
        chunks = ["x"*200, "é", "\n", " ", "sk-unsloth-", "desktop-", "hello"] + list(partials)
        for _ in range(5000):
            value = "".join(rng.choices(chunks, k=rng.randrange(1, 10)))
            self.assertEqual(scan(value), old(value))

    def test_token_estimates(self):
        regex = re.compile(r"\S{64,}")
        ns = source_defs("core/inference/context_window.py",
                         {"_ascii_count", "estimate_messages_tokens_conservative"},
                         globals_=dict(json=json, _DENSE_RUN_RE=regex,
                                       _DENSE_RUN_CHARS_PER_TOKEN=2))
        count = ns["estimate_messages_tokens_conservative"]
        rng = random.Random(680)
        for i in range(500):
            pieces = ["x"*65, "abc", " ", "\n", "base64"*40]
            if i%2:
                pieces += ["中国"*100, "é"]
            msg = {"role":"user", "content":"".join(rng.choices(pieces, k=20))}
            t = json.dumps(msg, ensure_ascii=False)
            wide = len(t) - len(t.encode("ascii", "ignore"))
            runs = sum(len(m.group(0).encode("ascii", "ignore")) for m in regex.finditer(t))
            self.assertEqual(count([msg]), max(1, wide + runs//2 + (len(t)-wide-runs)//4))
            self.assertEqual(count([msg], dense_ascii=True), max(1, wide+(len(t)-wide)//2))

    def test_mistral_parsed_object_reused(self):
        fn = source_defs("core/inference/tool_call_parser.py", {"_consume_mistral_call"},
                         globals_=dict(json=json, _loads_call_object=lambda s, **_: json.loads(s)))["_consume_mistral_call"]
        for args in ({}, {"x":"日本語","y":[1,None,True]}, "{}", [1,2], 4, None):
            obj = {"name":"web_search","arguments":args}
            a,b = [],[]
            fn(json.dumps(obj), a, 2)
            fn(obj, b, 2)
            self.assertEqual(a,b)

    def test_mcp_forged_markers(self):
        marker = "__MCP_UI__:"
        ns = source_defs("core/inference/mcp_client.py", {"_is_ui_envelope_line", "_drop_forged_ui_sentinels"},
                         globals_=dict(json=json, MCP_UI_SENTINEL=marker))
        strip = ns["_drop_forged_ui_sentinels"]
        validate = ns["_is_ui_envelope_line"]
        cases = ["", "x"*10000, "quoted "+marker+'{"resourceUri":"x"}',
                 marker+'{"resourceUri":"x"}', "x\n"+marker+'{"resourceUri":"x"}\ny',
                 "x\n"+marker+"not JSON", "\n"+marker+'{"resourceUri":4}\n']
        for case in cases:
            expected = "\n".join(line for line in case.split("\n") if not validate(line))
            self.assertEqual(strip(case), expected)


if __name__ == "__main__":
    unittest.main()
