"""Provider-free checks for the base variant's research-policy boundaries."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from MetaWeft.src.base import BasePipeline, memory_index_body
from MetaWeft.service.llm import LLMUnavailableError
from MetaWeft.service.elasticsearch import StoreSearchError
from MetaWeft.src.memory_retrieval import load_preset


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
        return {"hits": {"hits": [{"_id": key, "_source": value, "_score": 1.0}
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
        with patch.object(self.store, "query", side_effect=StoreSearchError("store unavailable")):
            result = self.pipeline.prepare_context(TASK, llm=FakeLLM(), artifacts=self.artifacts)
        self.assertTrue(result.degraded)
        self.assertIn("store unavailable", result.error)

    def test_unexpected_retrieval_error_is_fatal(self):
        with patch.object(self.store, "query", side_effect=RuntimeError("programming error")):
            with self.assertRaisesRegex(RuntimeError, "programming error"):
                self.pipeline.prepare_context(TASK, llm=FakeLLM(), artifacts=self.artifacts)

    def test_shared_retrieval_preserves_exact_query_and_legacy_artifact(self):
        self.existing("first")
        self.existing("second")
        instruction = "  Keep original\nwhitespace  "
        llm = FakeLLM([{"guidance": "use both"}])
        result = self.pipeline.prepare_context({**TASK, "instruction": instruction}, llm=llm,
                                               artifacts=self.artifacts)
        self.assertEqual(result.retrieved_ids, ["first", "second"])
        self.assertEqual(llm.embeddings[0][0], [instruction])
        self.assertEqual(self.store.queries, [
            {"size": 1, "query": {"match_all": {}}},
            {"size": 5, "knn": {"field": "embedding", "query_vector": [1.0, 0.0],
                                  "k": 5, "num_candidates": 10}},
        ])
        artifact = json.loads((self.artifacts / "retrieval.json").read_text())
        self.assertEqual(artifact["status"], "completed")
        self.assertIsNone(artifact["error"])
        self.assertEqual([record["id"] for record in artifact["records"]], ["first", "second"])
        self.assertNotIn("embedding", artifact["records"][0])
        self.assertEqual(artifact["trace"]["meta"]["name"], "base")

    def test_resolved_retrieval_limits_do_not_change_consolidation(self):
        self.existing()
        pipeline = BasePipeline(self.store, self.root, retrieval_k=2,
                                retrieval_meta=load_preset("base", num_candidates=7))
        pipeline.prepare_context(TASK, llm=FakeLLM([{"guidance": ""}]), artifacts=self.artifacts)
        self.assertEqual(self.store.queries[-1]["knn"]["k"], 2)
        self.assertEqual(self.store.queries[-1]["knn"]["num_candidates"], 7)
        pipeline._neighbors([1.0, 0.0])
        self.assertEqual(self.store.queries[-1]["knn"]["k"], 5)
        self.assertEqual(self.store.queries[-1]["knn"]["num_candidates"], 10)

    def test_disabled_retrieval_makes_no_search_or_model_calls(self):
        pipeline = BasePipeline(self.store, self.root, retrieval_k=0)
        llm = FakeLLM()
        result = pipeline.prepare_context(TASK, llm=llm, artifacts=self.artifacts)
        self.assertEqual(result.status, "disabled")
        self.assertEqual((self.store.queries, llm.calls, llm.embeddings), ([], [], []))

    def test_live_base_rejects_expanded_retrieval_before_search(self):
        for name in ("setting_1", "setting_2", "setting_3", "setting_4"):
            with self.subTest(preset=name), self.assertRaisesRegex(ValueError, "retrieval-only runner"):
                BasePipeline(self.store, self.root, retrieval_meta=load_preset(name))
        self.assertEqual(self.store.queries, [])
        renamed = load_preset("base")
        renamed["name"] = "descriptive-label-only"
        BasePipeline(self.store, self.root, retrieval_meta=renamed)

    def test_live_memory_without_source_refs_degrades_before_adaptation(self):
        record = self.existing()
        del record["source_refs"]
        llm = FakeLLM()
        result = self.pipeline.prepare_context(TASK, llm=llm, artifacts=self.artifacts)
        self.assertTrue(result.degraded)
        self.assertEqual((result.retrieved_ids, result.guidance, llm.calls), ([], "", []))
        artifact = json.loads((self.artifacts / "retrieval.json").read_text())
        self.assertEqual(artifact["records"], [])
        self.assertEqual(artifact["trace"]["status"], "degraded")
        self.assertIsInstance(artifact["error"], str)

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

    def test_oversized_consolidation_defers_each_candidate_without_dropping_evidence(self):
        before = self.existing()
        historical = self.root / before["source_refs"][0]
        large_evidence = {**EVIDENCE, "trace": [{"step": 1, "observation": "界" * 310_000}]}
        historical.write_text(json.dumps(large_evidence, ensure_ascii=False))
        original = historical.read_bytes()
        llm = FakeLLM([{"candidates": [CANDIDATE, CANDIDATE]}])

        result = self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)

        self.assertEqual([event.get("operation") for event in result.decisions], ["defer", "defer"])
        self.assertFalse(result.degraded)
        self.assertTrue(all("900000" in event["reason"] and "UTF-8" in event["reason"]
                            for event in result.decisions))
        self.assertEqual([call["phase"] for call in llm.calls], ["reflection"])
        self.assertEqual(self.store.writes, [])
        self.assertEqual(self.store.records, {"old": before})
        self.assertEqual(historical.read_bytes(), original)
        self.assertEqual(json.loads((self.artifacts / "evidence.json").read_text()), EVIDENCE)
        self.assertEqual(json.loads((self.artifacts / "updates.json").read_text()), result.to_dict())

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

    def test_provider_exhaustion_escapes_every_memory_stage(self):
        self.existing()
        for phase in ("retrieval", "adaptation", "reflection", "consolidation"):
            error = LLMUnavailableError(operation="chat", phase=phase, logical_call_id="call", attempts=6,
                                        cause=TimeoutError("provider unavailable"))
            llm = FakeLLM([{"candidates": [CANDIDATE]}, error] if phase == "consolidation" else [error])
            with self.subTest(phase=phase), self.assertRaises(LLMUnavailableError):
                if phase == "retrieval":
                    with patch.object(llm, "embed", side_effect=error):
                        self.pipeline.prepare_context(TASK, llm=llm, artifacts=self.artifacts)
                elif phase == "adaptation":
                    self.pipeline.prepare_context(TASK, llm=llm, artifacts=self.artifacts)
                else:
                    self.pipeline.post_task(EVIDENCE, llm=llm, artifacts=self.artifacts)

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
