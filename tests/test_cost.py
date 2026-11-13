"""Offline parity and historical-log migration checks; no model calls."""

import ast
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from MetaWeft.experiment.cost import PRICING, refresh_cost_estimate, summarize_cost, usage_cost
from MetaWeft.experiment.recovery import accounting, write_json


ROOT = Path(__file__).resolve().parents[2]


def call(identifier="paid", *, usage=None, model="qwen3.8-flash", hit=False):
    return {"id": identifier, "input": {"model": model},
            "output": {"usage": usage},
            "metadata": {"phase": "execution", "extra": {"cache": {"hit": hit}}}}


USAGE = {"prompt_tokens": 1000000, "completion_tokens": 100000,
         "prompt_tokens_details": {"cached_tokens": 250000}}


class CostTests(unittest.TestCase):
    def test_matches_parent_formula_for_chat_embeddings_aliases_and_invalid_usage(self):
        parent = ROOT / "metaskill-appworld/baselines/reme2f37a15/benchmark/appworld/execution_accounting.py"
        definitions = [node for node in ast.parse(parent.read_text()).body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                       and node.name in {"AccountingError", "decimal", "usage_cost"}]
        namespace = {"Decimal": Decimal, "InvalidOperation": InvalidOperation}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(parent), "exec"), namespace)
        cases = [
            ("qwen3.8-flash", USAGE, "chat"),
            ("qwen3.8-max-0902", USAGE, "chat"),
            ("qwen3.8-flash", {"input_tokens": 10, "output_tokens": 2,
                               "input_tokens_details": {"cached_tokens": 4}}, "chat"),
            ("text-embedding-3-large", {"prompt_tokens": 1000}, "embedding"),
            ("qwen3.8-flash", None, "chat"),
            ("qwen3.8-flash", {"prompt_tokens": 10}, "chat"),
            ("qwen3.8-flash", {**USAGE, "prompt_tokens": -1}, "chat"),
            ("qwen3.8-flash", {**USAGE, "prompt_tokens": True}, "chat"),
            ("qwen3.8-flash", {**USAGE, "prompt_tokens": 1}, "chat"),
        ]
        for model, usage, kind in cases:
            with self.subTest(model=model, usage=usage):
                self.assertEqual(usage_cost(model, usage, request_kind=kind),
                                 namespace["usage_cost"](model, usage, PRICING, request_kind=kind))
        self.assertEqual(usage_cost("qwen3.8-flash", USAGE), Decimal("0.716"))
        self.assertEqual(usage_cost("qwen3.8-max-0902", {"prompt_tokens": 1000, "completion_tokens": 100}),
                         Decimal("0.01248"))
        self.assertIsNone(usage_cost("unpriced-model", USAGE))

    def test_missing_usage_cache_replays_and_duplicate_physical_ids(self):
        paid = call(usage=USAGE)
        paid["metadata"]["status"] = "error"  # Reported usage is chargeable even on a failed response.
        result = summarize_cost([paid, paid, call("hit", usage=USAGE, hit=True),
                                 call("unknown"), call("other", usage=USAGE, model="other")])
        self.assertEqual(result["cost_usd"], 0.716)
        self.assertEqual(result["cost_status"], "partial")
        self.assertEqual(result["priced_calls"], 1)
        self.assertEqual(result["unpriced_calls"], 2)
        self.assertEqual(result["cache_hits"], 1)
        self.assertEqual(result["duplicate_calls"], 1)
        self.assertIsNone(summarize_cost([call()])["cost_usd"])
        self.assertIsNone(summarize_cost([])["cost_usd"])
        self.assertEqual(summarize_cost([call(hit=True)])["cost_usd"], 0)
        with self.assertRaisesRegex(ValueError, "Conflicting cost evidence"):
            summarize_cost([paid, call(usage={"prompt_tokens": 1, "completion_tokens": 1})])

    def test_sidecar_preserves_evidence_includes_initialization_and_refreshes_changed_logs(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_json(output / "accounting.json", {"cost_usd": None, "coverage": "original"})
            original = (output / "accounting.json").read_bytes()
            path = output / "tasks/task/logs/lm_calls.jsonl"
            path.parent.mkdir(parents=True)
            encoded = json.dumps(call(usage=USAGE)) + "\n"
            path.write_text(encoded)
            archive = output / "recovery_history/parent/tasks/task/logs/lm_calls.jsonl"
            archive.parent.mkdir(parents=True)
            archive.write_text(encoded)
            setup = output / "baseline_artifacts/reme/initialization_lm_calls.jsonl"
            setup.parent.mkdir(parents=True)
            embedding = call("embedding", model="text-embedding-3-large", usage={"prompt_tokens": 1000})
            embedding["input"]["input"] = ["setup"]
            setup.write_text(json.dumps(embedding) + "\n")
            result = refresh_cost_estimate(output)
            self.assertEqual(result["cost_usd"], 0.71613)
            self.assertEqual(result["duplicate_calls"], 1)
            self.assertEqual(result["priced_calls"], 2)
            saved_time = (output / "cost_estimate.json").stat().st_mtime_ns
            self.assertEqual(refresh_cost_estimate(output), result)
            self.assertEqual((output / "cost_estimate.json").stat().st_mtime_ns, saved_time)
            with path.open("a") as stream:
                stream.write('{"id":')
            self.assertEqual(refresh_cost_estimate(output)["cost_status"], "partial")
            path.write_text(encoded + json.dumps(call("failed")) + "\n")
            self.assertEqual(refresh_cost_estimate(output)["unpriced_calls"], 1)
            self.assertEqual((output / "accounting.json").read_bytes(), original)
            self.assertEqual(archive.read_text(), encoded)

    def test_accounting_preserves_inherited_and_current_costs(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_json(output / "run_manifest.json", {"reused_task_ids": ["old"]})
            for task, record in [("old", call("old", usage=USAGE)),
                                 ("new", call("new", usage=USAGE)),
                                 ("cached", call("cache", usage=USAGE, hit=True))]:
                path = output / f"tasks/{task}/logs/lm_calls.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(record) + "\n")
            result = accounting(output)
            self.assertEqual(result["cost_usd"], 1.432)
            self.assertEqual(result["current_attempt"]["cost_usd"], 0.716)
            self.assertEqual(result["inherited"]["cost_usd"], 0.716)
            self.assertEqual(result["recorded_calls"], 2)
            self.assertEqual(result["cache_hits"], 1)

    def test_thinking_summary_survives_a_truncated_log_rejected_by_accounting(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            record = call(usage=USAGE)
            record["input"].update(messages=[{"role": "user", "content": "test"}],
                                   extra_body={"enable_thinking": False})
            record["output"]["choices"] = [{"message": {"content": "answer", "reasoning_content": "work"}}]
            path = output / "tasks/task/logs/lm_calls.jsonl"
            path.parent.mkdir(parents=True)
            raw = json.dumps(record) + '\n{"id":'
            path.write_text(raw)
            with self.assertRaises(json.JSONDecodeError):
                accounting(output)
            thinking = json.loads((output / "thinking_summary.json").read_text())
            self.assertEqual(thinking["status"], "partial")
            self.assertEqual(thinking["total_llm_calls"], 1)
            self.assertEqual(thinking["thinking_despite_no_thinking_llm_calls"], 1)
            self.assertEqual(path.read_text(), raw)


if __name__ == "__main__":
    unittest.main()
