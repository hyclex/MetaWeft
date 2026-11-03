"""Provider-free checks for the base variant's research-policy boundaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from MetaWeft.src.base import BasePipeline, memory_index_body


TASK = {"task_id": "example_1", "instruction": "Find all pages of results",
        "supervisor": {"first_name": "Test"}, "app_descriptions": {"example": "Example app"}}
EVIDENCE = {"task_spec": TASK, "trace": [{"step": 1, "code": "print('observed')", "observation": "observed"}],
            "reward": 0.5}
CANDIDATE = {"when_to_use": "When listing paginated results", "content": "Continue until the page is empty"}


def decision(operation, target=None, **kwargs):
    return {"operation": operation, "target_id": target, "when_to_use": None,
            "content": None, "reason": "Supported by observations", **kwargs}


class FakeStore:
    def __init__(self, records=()):
        self.records = {record["id"]: record for record in records}
        self.queries = []
        self.writes = []
        self.fail_on_write = None

    def query(self, body):
        self.queries.append(body)
        return {"hits": {"hits": [{"_id": key, "_source": value}
                                   for key, value in list(self.records.items())[:body["size"]]]}}

    def put(self, identifier, document):
        if len(self.writes) + 1 == self.fail_on_write:
            error = RuntimeError("Write outcome is uncertain")
            error.uncertain = True
            raise error
        self.writes.append(identifier)
        self.records[identifier] = document
        return document


class FakeLLM:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.embeddings = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        content = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def embed(self, texts, **kwargs):
        self.embeddings.append((texts, kwargs))
        return [[1.0, 0.0] for _ in texts]


class BaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts/task1"
        self.store = FakeStore()
        self.pipeline = BasePipeline(self.store, self.root)

    def existing(self, identifier="old", refs=None):
        ref = "artifacts/old/evidence.json"
        path = self.root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(EVIDENCE))
        record = {"id": identifier, **CANDIDATE, "source_refs": [ref] if refs is None else refs,
                  "embedding": [1.0, 0.0]}
        self.store.records[identifier] = record
        return record

    def test_empty_memory_makes_no_model_calls(self):
        llm = FakeLLM()
        result = self.pipeline.prepare_context(TASK, llm=llm, artifacts=self.artifacts)
        self.assertEqual(result.status, "empty_memory")
        self.assertEqual((llm.calls, llm.embeddings), ([], []))
        self.assertEqual(result.guidance, "")

    def test_fewer_than_five_memories_adapt_once(self):
        self.existing()
        llm = FakeLLM([{"guidance": "Inspect pagination"}])
        result = self.pipeline.prepare_context(TASK, llm=llm, artifacts=self.artifacts)
        self.assertEqual(result.guidance, "Inspect pagination")
        self.assertEqual(result.retrieved_ids, ["old"])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(self.store.queries[-1]["knn"]["k"], 5)
        self.assertEqual(self.store.queries[-1]["knn"]["num_candidates"], 10)
        self.assertEqual(llm.calls[0]["seed"], 0)
        self.assertEqual(llm.calls[0]["extra_body"], {"enable_thinking": False})

    def test_irrelevant_memories_allow_empty_guidance(self):
        self.existing()
        result = self.pipeline.prepare_context(TASK, llm=FakeLLM([{"guidance": ""}]), artifacts=self.artifacts)
        self.assertFalse(result.degraded)
        self.assertEqual(result.guidance, "")

    def test_invalid_adaptation_is_visible_without_repair(self):
        self.existing()
        llm = FakeLLM(["not JSON"])
        result = self.pipeline.prepare_context(TASK, llm=llm, artifacts=self.artifacts)
        self.assertTrue(result.degraded)
        self.assertEqual(result.guidance, "")
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(json.loads((self.artifacts / "adaptation.json").read_text())["degraded"])

    def test_retrieval_failure_continues_without_guidance(self):
        with patch.object(self.store, "query", side_effect=RuntimeError("store unavailable")):
            result = self.pipeline.prepare_context(TASK, llm=FakeLLM(), artifacts=self.artifacts)
        self.assertTrue(result.degraded)
        self.assertIn("store unavailable", result.error)

    def test_failed_task_can_add_grounded_memory_and_excludes_evaluator_fields(self):
        evidence = {**EVIDENCE, "report": "SECRET ASSERTION", "solution": "SECRET SOLUTION",
                    "task_spec": {**TASK, "required_apis": "SECRET REQUIRED APIS"},
                    "trace": [{**EVIDENCE["trace"][0], "evaluator": "SECRET TRACE FIELD"}]}
        llm = FakeLLM([{"candidates": [CANDIDATE]}, decision("add")])
        result = self.pipeline.post_task(evidence, llm=llm, artifacts=self.artifacts)
        self.assertFalse(result.degraded)
        self.assertEqual(len(self.store.records), 1)
        self.assertNotIn("SECRET", json.dumps(llm.calls))
        self.assertNotIn("SECRET", (self.artifacts / "evidence.json").read_text())
        record = next(iter(self.store.records.values()))
        self.assertEqual(record["source_refs"], ["artifacts/task1/evidence.json"])
        self.assertEqual(llm.embeddings[0][0], [CANDIDATE["when_to_use"] + "\n\n" + CANDIDATE["content"]])

    def test_semantic_redundancy_skips_without_writing(self):
        before = self.existing()
        llm = FakeLLM([{"candidates": [CANDIDATE]}, decision("skip", "old")])
        result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
        self.assertEqual(result.decisions[0]["operation"], "skip")
        self.assertEqual(self.store.records, {"old": before})
        self.assertEqual(self.store.writes, [])

    def test_revision_preserves_source_history_and_full_evidence(self):
        before = self.existing()
        llm = FakeLLM([{"candidates": [CANDIDATE]}, decision(
            "revise", "old", when_to_use="When querying paginated lists", content="Read pages until empty")])
        result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
        after = self.store.records["old"]
        self.assertEqual(after["source_refs"], ["artifacts/old/evidence.json", "artifacts/task1/evidence.json"])
        self.assertEqual(result.decisions[0]["before"], before)
        self.assertEqual(after["content"], "Read pages until empty")
        payload = json.loads(llm.calls[-1]["messages"][-1]["content"])
        self.assertEqual(len(payload["evidence"]), 2)
        self.assertEqual(payload["evidence"]["artifacts/old/evidence.json"]["trace"], EVIDENCE["trace"])

    def test_unresolved_conflict_defers_without_writing(self):
        before = self.existing()
        llm = FakeLLM([{"candidates": [CANDIDATE]}, decision("defer", "old")])
        result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
        self.assertFalse(result.degraded)
        self.assertEqual(result.decisions[0]["operation"], "defer")
        self.assertEqual(self.store.records["old"], before)

    def test_revision_retains_evidence_consulted_from_other_neighbors(self):
        self.existing()
        other_path = self.root / "artifacts/other/evidence.json"
        other_path.parent.mkdir(parents=True)
        other_path.write_text(json.dumps(EVIDENCE))
        self.existing("other", refs=["artifacts/other/evidence.json"])
        llm = FakeLLM([{"candidates": [CANDIDATE]}, decision(
            "revise", "old", when_to_use="When listing", content="Combined supported procedure")])
        self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
        self.assertEqual(set(self.store.records["old"]["source_refs"]),
                         {"artifacts/old/evidence.json", "artifacts/other/evidence.json",
                          "artifacts/task1/evidence.json"})

    def test_missing_or_external_evidence_defers_before_consolidation_call(self):
        for refs in (["missing.json"], ["../outside.json"], []):
            with self.subTest(refs=refs):
                self.existing(refs=refs)
                llm = FakeLLM([{"candidates": [CANDIDATE]}])
                result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
                self.assertEqual(result.decisions[0]["operation"], "defer")
                self.assertTrue(result.decisions[0]["missing_evidence"])
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(self.store.writes, [])

    def test_sequential_candidates_see_earlier_updates_and_deduplicate_evidence(self):
        llm = FakeLLM([{"candidates": [CANDIDATE, CANDIDATE]}, decision("add")])
        original_chat = llm.chat

        def chat(messages, **kwargs):
            if kwargs["phase"] == "consolidation" and self.store.records:
                llm.responses.append(decision("skip", next(iter(self.store.records))))
            return original_chat(messages, **kwargs)

        llm.chat = chat
        result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
        self.assertEqual([item["operation"] for item in result.decisions], ["add", "skip"])
        last = json.loads(llm.calls[-1]["messages"][-1]["content"])
        self.assertEqual(len(last["neighbors"]), 1)
        self.assertEqual(len(last["evidence"]), 1)
        next_result = self.pipeline.prepare_context(TASK, llm=FakeLLM([{"guidance": "Use new memory"}]),
                                                    artifacts=self.root / "artifacts/task2")
        self.assertEqual(len(next_result.retrieved_ids), 1)

    def test_update_failure_retains_confirmed_writes_and_skips_remaining(self):
        self.store.fail_on_write = 2
        llm = FakeLLM([{"candidates": [CANDIDATE] * 3}, decision("add"), decision("add")])
        result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
        self.assertTrue(result.degraded)
        self.assertEqual(len(self.store.records), 1)
        self.assertTrue(result.decisions[1]["uncertain_write"])
        self.assertEqual(result.decisions[2]["operation"], "not_processed")
        self.assertEqual(len(llm.calls), 3)

    def test_invalid_consolidation_target_is_degraded(self):
        llm = FakeLLM([{"candidates": [CANDIDATE]}, decision("revise", {"bad": "id"})])
        result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
        self.assertTrue(result.degraded)
        self.assertEqual(self.store.writes, [])

    def test_reflection_accepts_zero_candidates_and_rejects_more_than_three(self):
        for candidates, degraded in (([], False), ([CANDIDATE] * 4, True)):
            with self.subTest(candidates=candidates):
                llm = FakeLLM([{"candidates": candidates}])
                result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
                self.assertEqual(result.degraded, degraded)
                self.assertEqual(llm.embeddings, [])

    def test_canonical_logging_and_artifact_failures_are_fatal(self):
        class LLMLoggingError(RuntimeError):
            pass

        llm = FakeLLM([LLMLoggingError("log unavailable")])
        with self.assertRaises(LLMLoggingError):
            self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)
        with patch("MetaWeft.src.base.pipeline._write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.pipeline.prepare_context(TASK, llm=FakeLLM(), artifacts=self.artifacts)

    def test_runs_have_separate_state(self):
        self.existing()
        other = BasePipeline(FakeStore(), self.root / "other")
        result = other.prepare_context(TASK, llm=FakeLLM(), artifacts=self.root / "other/task")
        self.assertEqual(result.status, "empty_memory")
        self.assertEqual(len(self.store.records), 1)

    def test_mapping_keeps_vectors_and_unquantized_cosine(self):
        mapping = memory_index_body()
        self.assertFalse(mapping["settings"]["index.mapping.exclude_source_vectors"])
        vector = mapping["mappings"]["properties"]["embedding"]
        self.assertEqual((vector["dims"], vector["similarity"], vector["index_options"]["type"]),
                         (1024, "cosine", "hnsw"))


if __name__ == "__main__":
    unittest.main()
