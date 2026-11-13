"""Offline thinking-control evidence and phase-sidecar checks; no provider calls."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from MetaWeft.experiment.thinking import classify, refresh_thinking_summary, summarize_thinking


def call(identifier="one", *, flag=False, reasoning="", tokens=0):
    return {"id": identifier,
            "input": {"model": "example", "messages": [], "extra_body": {"enable_thinking": flag}},
            "output": {"choices": [{"message": {"content": "42", "reasoning_content": reasoning}}],
                       "usage": {"completion_tokens_details": {"reasoning_tokens": tokens}}},
            "metadata": {"phase": "execution"}}


class ThinkingTests(unittest.TestCase):
    def test_actual_request_controls_and_extra_body_precedence(self):
        base = call()
        base["input"]["enable_thinking"] = True
        self.assertEqual(classify(base)["requested_thinking"], "off")
        for controls in ({"reasoning_effort": "none"}, {"thinking": {"type": "disabled"}},
                         {"no_thinking": True}, {"no_think": True}, {"thinking": False},
                         {"reasoning": {"effort": "none"}}, {"reasoning": {"enabled": False}}):
            with self.subTest(controls=controls):
                base["input"] = {"messages": [], **controls}
                self.assertEqual(classify(base)["requested_thinking"], "off")
        for controls in ({}, {"enable_thinking": "false"}, {"enable_thinking": 0}):
            base["input"] = {"messages": [], **controls}
            self.assertEqual(classify(base)["requested_thinking"], "unspecified")
        base["input"] = {"messages": [], "enable_thinking": False, "reasoning_effort": "high"}
        self.assertEqual(classify(base)["requested_thinking"], "conflicting")

    def test_reasoning_aliases_and_zero_counter_contradictions(self):
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = call()
            value["output"]["choices"][0]["message"][key] = "actual reasoning"
            result = summarize_thinking([value])
            self.assertEqual(result["thinking_despite_no_thinking_llm_calls"], 1)
            self.assertEqual(result["reasoning_text_with_zero_tokens"], 1)
        value = call()
        value["output"]["choices"][0]["message"]["content"] = [
            {"type": "thinking", "thinking": "internal steps"}, {"type": "text", "text": "42"}]
        self.assertEqual(classify(value)["evidence"], "observed_thinking")
        value["output"] = {"content": [{"type": "thinking", "thinking": "steps"}]}
        self.assertEqual(classify(value)["reasoning_chars"], 5)

    def test_positive_token_aliases_are_not_added_together(self):
        value = call(tokens=0)
        value["output"]["usage"]["output_tokens_details"] = {"reasoning_tokens": 12}
        value["output"]["usage"]["thinking_tokens"] = 12
        self.assertEqual(classify(value)["reasoning_tokens"], 12)
        self.assertEqual(classify(value)["evidence"], "observed_thinking")
        value["output"] = {"usageMetadata": {"thoughtsTokenCount": 8}}
        self.assertEqual(classify(value)["evidence"], "observed_thinking")
        value["output"] = {"choices": [{"message": {"content": "42"}}],
                           "usage": {"reasoning_tokens": True, "thinking_tokens": -1}}
        self.assertIsNone(classify(value)["reasoning_tokens"])

    def test_only_protocol_think_blocks_count_not_quoted_examples(self):
        value = call()
        message = value["output"]["choices"][0]["message"]
        for text in ("<think>work</think>42", " \n<THINK>unfinished", "<think></think><think>work</think>"):
            message["content"] = text
            self.assertTrue(classify(value)["think_tag"])
        for text in ("<think> </think>42", "Here is <think>an example</think>",
                     "```xml\n<think>an example</think>\n```", "`<think>quoted</think>`"):
            message["content"] = text
            self.assertFalse(classify(value)["think_tag"])

    def test_missing_usage_and_failed_responses_stay_unknown(self):
        no_usage = call("no_usage", tokens=None)
        failure = call("failed")
        failure["output"] = {"error_type": "APITimeoutError"}
        malformed = call("malformed")
        malformed["output"] = ["unexpected"]
        result = summarize_thinking([no_usage, failure, malformed])
        self.assertEqual(result["total_llm_calls"], 3)
        self.assertEqual(result["no_thinking_llm_calls"], 3)
        self.assertEqual(result["no_thinking_unknown_reasoning_calls"], 1)
        self.assertEqual(result["no_thinking_no_response_calls"], 2)
        self.assertEqual(result["no_thinking_zero_reasoning_calls"], 0)

    def test_deduplicates_physical_calls_excludes_embeddings_and_replays(self):
        first = call("first", reasoning="work")
        retry = call("retry", reasoning="more")
        retry["metadata"]["logical_call_id"] = first["metadata"]["logical_call_id"] = "logical"
        embedding = {"id": "embed", "input": {"model": "embedding", "input": ["text"]}}
        replay = call("replay", reasoning="cached")
        replay["metadata"]["extra"] = {"cache": {"hit": True}}
        result = summarize_thinking([first, first, retry, embedding, replay, call("on", flag=True)])
        self.assertEqual((result["total_llm_calls"], result["no_thinking_llm_calls"],
                          result["thinking_despite_no_thinking_llm_calls"]), (3, 2, 2))
        self.assertEqual(result["duplicate_records_excluded"], 1)
        self.assertEqual(result["embedding_calls_excluded"], 1)
        self.assertEqual(result["cache_replay_calls_excluded"], 1)
        conflicting = deepcopy(first)
        conflicting["output"]["choices"][0]["message"]["content"] = "different"
        result = summarize_thinking([first, conflicting, {}, [], {"id": "unknown"}])
        self.assertEqual(result["total_llm_calls"], 1)
        self.assertEqual(result["conflicting_call_ids"], 1)
        self.assertEqual(result["malformed_records"], 2)
        self.assertEqual(result["unclassified_records"], 1)
        self.assertEqual(len(result["coverage_gaps"]), 4)

    def test_phase_sidecar_covers_setup_failed_tasks_and_archive_without_mutating_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "run_manifest.json"
            manifest.write_text('{"status":"failed"}')
            logs = {
                "tasks/failed/logs/lm_calls.jsonl": call("task", reasoning="work"),
                "setup_lm_calls.jsonl": call("setup"),
                "baseline_artifacts/reme/initialization_lm_calls.jsonl": call("initial"),
                "recovery_history/old/tasks/failed/logs/lm_calls.jsonl": call("task", reasoning="work"),
                "recovery_history/old/setup_lm_calls.jsonl": call("old_setup"),
            }
            for relative, record in logs.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record) + "\n")
            original = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            result = refresh_thinking_summary(root)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["total_llm_calls"], 4)
            self.assertEqual(result["thinking_despite_no_thinking_llm_calls"], 1)
            self.assertEqual(result["duplicate_records_excluded"], 1)
            stamp = (root / "thinking_summary.json").stat().st_mtime_ns
            self.assertEqual(refresh_thinking_summary(root), result)
            self.assertEqual((root / "thinking_summary.json").stat().st_mtime_ns, stamp)
            self.assertTrue(all(Path(path).read_bytes() == data for path, data in original.items()))
            task_path = root / "tasks/failed/logs/lm_calls.jsonl"
            with task_path.open("a") as stream:
                stream.write('{"id":')
            changed = refresh_thinking_summary(root)
            self.assertEqual(changed["status"], "partial")
            self.assertEqual(changed["total_llm_calls"], 4)
            self.assertIn("Incomplete log tail", str(changed["coverage_gaps"]))
            task_path.write_text(json.dumps(logs["tasks/failed/logs/lm_calls.jsonl"]) + '\ninvalid\n' + json.dumps(call("new")) + '\n')
            changed = refresh_thinking_summary(root)
            self.assertEqual(changed["total_llm_calls"], 5)
            self.assertIn("Malformed JSON record", str(changed["coverage_gaps"]))

    def test_missing_logs_are_unavailable_not_proven_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = refresh_thinking_summary(Path(temporary))
            self.assertEqual(result["status"], "unavailable")
            self.assertTrue(result["coverage_gaps"])

    def test_task_coverage_gaps_and_manifest_changes_invalidate_cached_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tasks/one/logs/lm_calls.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(call()) + "\n")
            manifest = root / "run_manifest.json"
            manifest.write_text(json.dumps({"completed_task_ids": ["one"]}))
            self.assertEqual(refresh_thinking_summary(root)["status"], "complete")
            manifest.write_text(json.dumps({"completed_task_ids": ["one", "missing"], "current_task_id": "current"}))
            result = refresh_thinking_summary(root)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["total_llm_calls"], 1)
            self.assertEqual({gap["path"] for gap in result["coverage_gaps"]}, {
                "tasks/missing/logs/lm_calls.jsonl", "tasks/current/logs/lm_calls.jsonl"})

    def test_unavailable_initial_file_metadata_does_not_abort_known_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inaccessible = root / "tasks/inaccessible/logs/lm_calls.jsonl"
            available = root / "tasks/available/logs/lm_calls.jsonl"
            for path in (inaccessible, available):
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(call(path.parent.parent.name)) + "\n")
            manifest = root / "run_manifest.json"
            manifest.write_text('{}')
            stat = Path.stat

            def unreadable(path, *args, **kwargs):
                if path in (inaccessible, manifest):
                    raise PermissionError("temporarily unavailable")
                return stat(path, *args, **kwargs)

            with patch.object(Path, "stat", unreadable):
                result = refresh_thinking_summary(root)
            self.assertEqual(result["total_llm_calls"], 1)
            self.assertEqual(result["status"], "partial")
            self.assertEqual({gap["path"] for gap in result["coverage_gaps"]}, {
                "tasks/inaccessible/logs/lm_calls.jsonl", "run_manifest.json"})


if __name__ == "__main__":
    unittest.main()
