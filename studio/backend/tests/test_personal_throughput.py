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
