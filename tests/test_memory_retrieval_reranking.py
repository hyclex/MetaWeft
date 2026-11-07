"""Offline stage contracts, including exact request and recovery boundaries."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import unittest

from MetaWeft.service.llm.client import InvalidResponseError, LLMUnavailableError
from MetaWeft.src.memory_retrieval.reranking import (
    RankingFailure, SCHEMA_CORRECTION, parse_ranking_response, parse_scores,
    rerank_indices, run_stage,
)


def records(count):
    return [{"id": f"memory-{index}", "when_to_use": "page café", "content": "Read each page",
             "source_refs": [f"private/{index}"], "label": True}
            for index in range(1, count + 1)]


def stage(method="listwise"):
    return {
        "id": "select" if method == "listwise" else "grade", "method": method,
        "prompt": "procedure_listwise20_v1" if method == "listwise" else "utility_scores_v1",
        "model": "qwen3.8-max-0902" if method == "listwise" else "qwen3.8-flash",
        "fields": ["when_to_use", "content"],
        "output_limit": 20 if method == "listwise" else "all",
        "small_input_policy": "bypass" if method == "listwise" else "run",
        "generation": {"temperature": 0, "seed": 0, "enable_thinking": False,
                       "thinking_budget": None, "max_tokens": 512, "max_completion_tokens": None},
        "response": {"accept_wrapped": True, "repair_duplicates": method == "listwise",
                     "schema_retries": 1, "fallback_to_input": True,
                     "fallback_on_empty_response": True},
    }


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.chat_model = "caller-model"
        self.max_retries = 2
        self.retry_delay_seconds = 60
        self.cache_mode = "no_cache"
        self.cache_path = "unchanged.sqlite3"

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), **deepcopy(kwargs)})
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        text = output if isinstance(output, str) else json.dumps(output)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def unavailable(cause, operation="chat"):
    error = LLMUnavailableError(operation=operation, phase="retrieval", logical_call_id="logical-1",
                                attempts=3, cause=cause)
    error.__cause__ = cause
    return error


class ResponseParserTests(unittest.TestCase):
    def test_complete_wrappers_only_and_saved_normalization_order(self):
        ranking = list(range(1, 21))
        for text in (json.dumps(ranking), json.dumps([{"index": value} for value in ranking]),
                     "```json\n" + json.dumps(ranking) + "\n```",
                     "```\n" + json.dumps({"ranking": ranking}) + "\n```"):
            with self.subTest(text=text):
                value, normalized = parse_ranking_response(text, accept_wrapped=True)
                self.assertTrue(normalized)
                self.assertEqual(rerank_indices(value, 20), (ranking, False))
        for text in ("answer: " + json.dumps(ranking), "```python\n[]\n```",
                     "```json\n[]\n``` trailing", '{"ranking": [], "ranking": []}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_ranking_response(text, accept_wrapped=True)
        with self.assertRaises(ValueError):
            parse_ranking_response("```json\n{}\n```", accept_wrapped=False)
        self.assertEqual(parse_ranking_response("[[1, 80], [2, 90]]", "scores", True),
                         ({"scores": [[1, 80], [2, 90]]}, True))

    def test_duplicate_repair_never_accepts_wrong_counts_types_or_bounds(self):
        repeated = {"ranking": [4, 4, *range(5, 23)]}
        self.assertEqual(rerank_indices(repeated, 25, True), ([4, *range(5, 23), 1], True))
        with self.assertRaises(ValueError):
            rerank_indices(repeated, 25, False)
        good = list(range(1, 21))
        for value in ({"ranking": good[:-1]}, {"ranking": good + [21]},
                      {"ranking": [True, *good[1:]]}, {"ranking": [1.0, *good[1:]]},
                      {"ranking": [0, *good[1:]]}, {"ranking": [26, *good[1:]]},
                      {"ranking": good, "extra": []}, good):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rerank_indices(value, 25, True)

    def test_grade_coverage_integer_types_and_input_order_ties(self):
        self.assertEqual(parse_scores({"scores": [[3, 80], [2, 100], [1, 80]]}, 3), [2, 1, 3])
        bad = ({"scores": [[1, True]]}, {"scores": [[True, 70]]},
               {"scores": [[1, 1.0]]}, {"scores": [[1.0, 70]]},
               {"scores": [[1, -1]]}, {"scores": [[1, 101]]},
               {"scores": [[0, 80]]}, {"scores": [[2, 80]]},
               {"scores": [[1, 80, 90]]}, {"scores": []},
               {"scores": [[1, 80]], "extra": 0})
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_scores(value, 1)
        with self.assertRaises(ValueError):
            parse_scores({"scores": [[1, 80], [1, 90]]}, 2)


class StageTests(unittest.TestCase):
    def run_stage(self, outputs, *, count=80, config=None, callback=None, input_records=None):
        self.llm = FakeLLM(outputs)
        self.snapshots = []
        self.config = config or stage()
        self.input_records = input_records or records(count)
        return run_stage("  Original task café  ", self.input_records, llm=self.llm,
                         stage=self.config, prompt="frozen prompt\n",
                         record_stage=callback or self.snapshots.append)

    def test_listwise_payload_mapping_and_client_isolation(self):
        original = records(80)
        config = stage()
        output, trace = self.run_stage([{"ranking": list(range(80, 60, -1))}],
                                       config=config, input_records=original)
        call = self.llm.calls[0]
        payload = json.loads(call["messages"][1]["content"])
        self.assertEqual(list(payload), ["task", "return_count", "memories"])
        self.assertEqual(payload["task"], "  Original task café  ")
        self.assertEqual(payload["return_count"], 20)
        self.assertEqual(len(payload["memories"]), 80)
        self.assertEqual(payload["memories"][0], {"index": 1, "when_to_use": "page café",
                                                "content": "Read each page"})
        self.assertNotIn("\\u00e9", call["messages"][1]["content"])
        self.assertEqual([item["id"] for item in output], [f"memory-{i}" for i in range(80, 60, -1)])
        self.assertEqual(trace["ranking"], list(range(80, 60, -1)))
        self.assertEqual(trace["local_indices"][79], {"index": 80, "id": "memory-80"})
        self.assertEqual(call["component"], "retrieval.reranking.select.attempt_0")
        self.assertEqual(call["phase"], "retrieval")
        self.assertEqual(call["model"], "qwen3.8-max-0902")
        self.assertEqual(call["extra_body"], {"enable_thinking": False})
        self.assertEqual(call["max_tokens"], 512)
        self.assertNotIn("max_completion_tokens", call)
        self.assertEqual(self.llm.chat_model, "caller-model")
        self.assertEqual(self.llm.cache_path, "unchanged.sqlite3")
        self.assertEqual(config, stage())
        self.assertEqual(original, records(80))
        self.assertEqual(self.snapshots[0]["attempts"][0]["status"], "running")
        self.assertEqual(self.snapshots[-1]["status"], "completed")

    def test_scores_preserve_all_candidates_and_payload_count(self):
        output, trace = self.run_stage([{"scores": [[i, i] for i in range(1, 81)]}],
                                       config=stage("scores"))
        self.assertEqual(len(output), 80)
        self.assertEqual(output[0]["id"], "memory-80")
        payload = json.loads(self.llm.calls[0]["messages"][1]["content"])
        self.assertEqual(list(payload), ["task", "memories", "score_count"])
        self.assertEqual(payload["score_count"], 80)
        self.assertEqual(len(trace["grades"]), 80)

    def test_cascade_renumbers_shortlist_and_ties_keep_selector_order(self):
        shortlist, _ = self.run_stage([{"ranking": list(range(225, 205, -1))}], count=225)
        config = stage("scores")
        config["generation"].update(enable_thinking=True, thinking_budget=2048,
                                     max_tokens=None, max_completion_tokens=6144)
        output, trace = self.run_stage([{"scores": [[i, 80] for i in range(20, 0, -1)]}],
                                       config=config, input_records=shortlist)
        self.assertEqual(output, shortlist)
        self.assertEqual(trace["local_indices"][0], {"index": 1, "id": "memory-225"})
        call = self.llm.calls[0]
        self.assertEqual(call["max_completion_tokens"], 6144)
        self.assertNotIn("max_tokens", call)
        self.assertEqual(call["extra_body"], {"enable_thinking": True, "thinking_budget": 2048})
        self.assertEqual(self.llm.chat_model, "caller-model")

    def test_small_listwise_inputs_skip_but_grading_still_runs(self):
        for count in (1, 19):
            with self.subTest(count=count):
                output, trace = self.run_stage([], count=count)
                self.assertEqual(output, records(count))
                self.assertEqual(trace["status"], "skipped")
                self.assertEqual(trace["skip"]["input_count"], count)
                self.assertEqual(trace["skip"]["reason"], "insufficient_for_fixed20")
                self.assertEqual(trace["attempts"], [])
                self.assertEqual(self.llm.calls, [])
                output, trace = self.run_stage([{"scores": [[i, 80] for i in range(1, count + 1)]}],
                                               count=count, config=stage("scores"))
                self.assertEqual(len(self.llm.calls), 1)
                self.assertEqual(trace["status"], "completed")

    def test_one_exact_schema_correction_and_no_client_mutation(self):
        output, trace = self.run_stage(["not JSON", {"ranking": list(range(1, 21))}])
        self.assertEqual(len(output), 20)
        self.assertEqual(len(self.llm.calls), 2)
        self.assertEqual(trace["schema_retries"], 1)
        self.assertEqual(self.llm.calls[1]["messages"], [
            *self.llm.calls[0]["messages"], {"role": "assistant", "content": "not JSON"},
            {"role": "user", "content": SCHEMA_CORRECTION}])
        self.assertEqual(self.llm.calls[1]["component"], "retrieval.reranking.select.attempt_1")
        self.assertEqual(trace["attempts"][0]["status"], "schema_error")
        self.assertEqual(trace["attempts"][0]["response_sha256"],
                         hashlib.sha256(b"not JSON").hexdigest())
        self.assertEqual((self.llm.max_retries, self.llm.retry_delay_seconds), (2, 60))

    def test_schema_fallback_retains_stage_input_and_following_stage_can_run(self):
        for method, expected in (("listwise", 20), ("scores", 80)):
            with self.subTest(method=method):
                output, trace = self.run_stage(["invalid", "still invalid"], config=stage(method))
                self.assertEqual(output, records(80)[:expected])
                self.assertEqual(trace["status"], "completed_with_fallback")
                self.assertEqual(trace["fallback"]["policy"], "fallback_to_input")
                self.assertEqual(len(self.llm.calls), 2)
                self.assertEqual(self.snapshots[-1]["fallback"], trace["fallback"])
        shortlist, _ = self.run_stage(["invalid", "invalid"])
        output, trace = self.run_stage([{"scores": [[i, i] for i in range(1, 21)]}],
                                       config=stage("scores"), input_records=shortlist)
        self.assertEqual(output[0]["id"], "memory-20")
        self.assertEqual(trace["status"], "completed")

    def test_nonfinite_json_uses_schema_correction_before_artifact_serialization(self):
        for number in ("NaN", "Infinity", "-Infinity", "1e9999"):
            snapshots = []
            def record(trace):
                json.dumps(trace, allow_nan=False)
                snapshots.append(trace)
            with self.subTest(number=number):
                output, trace = self.run_stage(
                    ['{"scores": [[1, ' + number + ']]}', {"scores": [[1, 80]]}],
                    count=1, config=stage("scores"), callback=record)
                self.assertEqual(output, records(1))
                self.assertEqual(len(self.llm.calls), 2)
                self.assertEqual(trace["status"], "completed")
                self.assertEqual(trace["attempts"][0]["status"], "schema_error")
                self.assertIsNone(trace["attempts"][0]["parsed"])
                self.assertEqual(snapshots[-1], trace)

    def test_strict_schema_exhaustion_raises_ranking_failure(self):
        config = stage()
        config["response"].update(schema_retries=0, fallback_to_input=False,
                                   fallback_on_empty_response=False, accept_wrapped=False)
        with self.assertRaises(RankingFailure):
            self.run_stage(["invalid"], config=config)
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(self.snapshots[-1]["status"], "failed")

    def test_empty_fallback_exact_typed_exhaustion_evidence(self):
        error = unavailable(InvalidResponseError("Chat response contains no text"))
        output, trace = self.run_stage([error])
        self.assertEqual(output, records(80)[:20])
        self.assertEqual(trace["status"], "completed_with_fallback")
        self.assertEqual(trace["fallback"]["logical_call_id"], "logical-1")
        self.assertEqual(trace["fallback"]["attempts"], 3)
        self.assertEqual(trace["fallback"]["phase"], "retrieval")
        self.assertEqual(len(self.llm.calls), 1)

    def test_other_provider_and_programming_failures_never_fallback_or_correct(self):
        class DerivedInvalid(InvalidResponseError):
            pass
        cases = [unavailable(ValueError("Chat response contains no text")),
                 unavailable(DerivedInvalid("Chat response contains no text")),
                 unavailable(InvalidResponseError("other invalid response")),
                 unavailable(InvalidResponseError("Chat response contains no text"), "embedding"),
                 RuntimeError("cache or logger error"), ValueError("programming error")]
        for error in cases:
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.run_stage([error])
            self.assertEqual(len(self.llm.calls), 1)
            self.assertEqual(self.snapshots[-1]["status"], "failed")
            self.assertEqual(self.llm.chat_model, "caller-model")
        for flag in ("fallback_to_input", "fallback_on_empty_response"):
            config = stage()
            config["response"][flag] = False
            with self.subTest(flag=flag), self.assertRaises(LLMUnavailableError):
                self.run_stage([unavailable(InvalidResponseError("Chat response contains no text"))],
                               config=config)
            self.assertEqual(len(self.llm.calls), 1)

    def test_artifact_failure_does_not_cause_another_paid_call(self):
        for fail_at, expected_calls in ((1, 0), (2, 1)):
            count = 0
            def callback(trace):
                nonlocal count
                count += 1
                if count == fail_at:
                    raise ValueError("artifact unavailable")
            with self.subTest(fail_at=fail_at), self.assertRaisesRegex(ValueError, "artifact"):
                self.run_stage([{"ranking": list(range(1, 21))}], callback=callback)
            self.assertEqual(len(self.llm.calls), expected_calls)
        def reject_fallback(trace):
            if trace["status"] == "completed_with_fallback":
                raise OSError("fallback evidence unavailable")
        with self.assertRaises(OSError):
            self.run_stage(["invalid", "invalid"], callback=reject_fallback)
        self.assertEqual(len(self.llm.calls), 2)


if __name__ == "__main__":
    unittest.main()
