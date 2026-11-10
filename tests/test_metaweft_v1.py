"""Provider-free acceptance for V1 preparation, policy, failures and provenance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from MetaWeft.service.elasticsearch import StoreSearchError, StoreWriteError
from MetaWeft.service.llm import LLMUnavailableError
from MetaWeft.src.ace_adapt_reme_task import V1Pipeline, empty_playbook, memory_index_body
from MetaWeft.src.ace_adapt_reme_task.common import PromptOverflow, write_json
from MetaWeft.src.ace_adapt_reme_task.config import TASK_SCHEMA, ADAPTATION_SCHEMA, PROMPTS
from MetaWeft.src.memory_retrieval import load_preset, search
from MetaWeft.src.memory_retrieval.candidate_search import prepare_exact_bank


TASK = {"task_id": "task1", "instruction": "Inspect every page", "supervisor": {"first_name": "Test"},
        "app_descriptions": {"example": "Example app"}}
EVIDENCE = {"task_spec": TASK, "trace": [{"step": 1, "code": "print('page')", "observation": "page"}], "reward": 0}
CANDIDATE = {"name": "Read all pages", "when_to_use": "When listing pages", "content": "Continue until empty"}
VECTOR = [1.0] + [0.0] * 1023


def decision(operation, target=None, **kwargs):
    return {"operation": operation, "target_id": target, "name": None, "when_to_use": None,
            "content": None, "reason": "Supported by observations", **kwargs}


def edit(content="Validate assumptions before applying retrieved procedures"):
    return {"operation": "ADD", "target_id": None, "heading": "verification", "content": content,
            "evidence": ["/adaptation/guide", "/trace/0/observation"], "reason": "Guide and observation support this lesson"}


class Store:
    def __init__(self):
        self.records, self.queries, self.writes = {}, [], []

    def query(self, body):
        self.queries.append(deepcopy(body))
        return {"hits": {"hits": [{"_id": key, "_source": deepcopy(value), "_score": 1.0}
                                   for key, value in list(self.records.items())[:body["size"]]]}}

    def put(self, identifier, document):
        self.records[identifier] = deepcopy(document)
        self.writes.append(identifier)
        return deepcopy(document)

    def export(self):
        return [{"id": key, "document": deepcopy(value)} for key, value in self.records.items()]


class LLM:
    max_retries = 5
    retry_delay_seconds = 60
    dimensions = 1024
    chat_model = "qwen3.8-flash"

    def __init__(self, responses=(), call_refs=None):
        self.responses, self.calls, self.embeddings = list(responses), [], []
        self.call_refs = call_refs

    def chat(self, messages, **controls):
        self.calls.append({"messages": deepcopy(messages), **controls})
        if self.call_refs is not None:
            self.call_refs.setdefault(controls["component"], []).append({"logical_call_id": str(len(self.calls))})
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=value if isinstance(value, str) else json.dumps(value)))])

    def embed(self, texts, **controls):
        self.embeddings.append((list(texts), controls))
        return [list(VECTOR) for _ in texts]


class V1Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts/task1"
        self.store = Store()
        self.pipeline = V1Pipeline(self.store, self.root)
        self.commit_pair()

    def commit_pair(self, epoch=0):
        refs = []
        for role, schema in (("task", TASK_SCHEMA), ("adaptation", ADAPTATION_SCHEMA)):
            path = self.root / f"{role}-{epoch}.json"
            snapshot = {"role": role, "schema": schema, "stream_id": "run:test-normal", "state_epoch": epoch,
                        "status": "verified"}
            snapshot.update({"records": self.store.export()} if role == "task" else {"playbook": self.pipeline.playbook})
            write_json(path, snapshot)
            refs.append({"role": role, "schema": schema, "stream_id": "run:test-normal", "epoch": epoch,
                         "snapshot": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.pipeline.set_state_refs(*refs)

    def existing(self, identifier="old", refs=None):
        source = self.root / "historical/evidence.json"
        write_json(source, EVIDENCE)
        record = {"id": identifier, **CANDIDATE, "source_refs": ["historical/evidence.json"] if refs is None else refs,
                  "embedding": list(VECTOR)}
        self.store.records[identifier] = record
        return deepcopy(record)

    def prepare(self, guide="Inspect prerequisites", *, llm=None):
        return self.pipeline.prepare_context(TASK, llm=llm or LLM([{"guidance": guide}]), artifacts=self.artifacts)

    def update(self, responses, *, prepared=None, evidence=None, llm=None):
        return self.pipeline.post_task(evidence or EVIDENCE, prepared=prepared or self.prepare(),
                                       llm=llm or LLM(responses), artifacts=self.artifacts)

    def test_both_empty_still_adapts_once_and_preserves_exact_event(self):
        refs = {}
        llm = LLM([{"guidance": ""}], refs)
        prepared = self.pipeline.prepare_context(TASK, llm=llm, artifacts=self.artifacts, call_refs=refs)
        self.assertFalse(prepared.degraded)
        self.assertEqual(prepared.guidance, "")
        self.assertEqual(llm.embeddings, [])
        self.assertEqual(len(llm.calls), 1)
        event = json.loads((self.artifacts / "adaptation_event.json").read_text())
        self.assertEqual(event["messages"], llm.calls[0]["messages"])
        self.assertEqual(event["input"]["retrieved_memories"], [])
        self.assertEqual(event["call_references"], [{"logical_call_id": "1"}])
        self.assertEqual(event["status"], "completed")

    def test_nonempty_playbook_without_task_memory_reaches_both_calls(self):
        prepared = self.prepare()
        result = self.update([{"candidates": []}, {"base_revision": 0, "edits": [edit()]}], prepared=prepared)
        self.assertFalse(result.degraded)
        self.commit_pair(1)
        self.artifacts = self.root / "artifacts/task2"
        llm = LLM([{"guidance": "Use a supported check"}, {"candidates": []}, {"base_revision": 1, "edits": []}])
        prepared = self.prepare(llm=llm)
        self.update([], prepared=prepared, llm=llm)
        first = json.loads(llm.calls[0]["messages"][1]["content"])
        last = json.loads(llm.calls[-1]["messages"][1]["content"])
        self.assertEqual(first["playbook"], last["prior_playbook"])
        self.assertIn("[adapt-000001]", first["playbook"])
        self.assertEqual(first["retrieved_memories"], [])
        self.assertEqual(llm.embeddings, [])

    def test_small_pool_skips_ranking_and_preserves_names(self):
        self.existing()
        llm = LLM([{"guidance": "Inspect"}])
        prepared = self.prepare(llm=llm)
        self.assertEqual(prepared.task_prepared.payload[0]["name"], CANDIDATE["name"])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.embeddings[0][0], [TASK["instruction"]])
        self.assertEqual(self.store.queries[-1]["knn"]["k"], 20)
        self.assertEqual(self.store.queries[-1]["knn"]["num_candidates"], 40)

    def test_twenty_candidates_rank_then_limit_ten_without_export(self):
        for index in range(25):
            self.existing(f"m{index:02d}")
        llm = LLM([{"ranking": list(range(20, 0, -1))}, {"guidance": "Ranked"}])
        with patch.object(self.store, "export", side_effect=AssertionError("Unexpected bank export")):
            prepared = self.prepare(llm=llm)
        self.assertEqual([record["id"] for record in prepared.task_prepared.payload],
                         [f"m{index:02d}" for index in range(19, 9, -1)])
        rank_input = json.loads(llm.calls[0]["messages"][1]["content"])
        self.assertNotIn('"name"', json.dumps(rank_input))
        self.assertEqual(llm.calls[0]["max_tokens"], 512)

    def test_preflight_rejects_missing_retrieval_zero_k_and_changed_preset(self):
        for kwargs in ({"retrieval_k": 0}, {"retrieval_meta": load_preset("base")}, {"retrieval_k": 5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                V1Pipeline(self.store, self.root, **kwargs)
        with self.assertRaises(ValueError):
            V1Pipeline(None, self.root)
        llm = LLM()
        llm.max_retries = 1
        with self.assertRaisesRegex(ValueError, "provider_policy"):
            self.prepare(llm=llm)
        self.assertEqual(llm.calls, [])

    def test_degraded_retrieval_still_adapts_and_separates_adapter_status(self):
        llm = LLM([{"guidance": "Check assumptions"}])
        with patch.object(self.store, "query", side_effect=StoreSearchError("search failed")):
            prepared = self.prepare(llm=llm)
        self.assertTrue(prepared.degraded)
        self.assertEqual(prepared.adaptation_event["status"], "completed")
        self.assertEqual(prepared.task_prepared.payload, [])
        self.assertEqual(len(llm.calls), 1)

    def test_invalid_adapter_has_no_repair_and_cannot_ground_edits(self):
        llm = LLM(['{"guidance":"one","guidance":"two"}'])
        prepared = self.prepare(llm=llm)
        self.assertTrue(prepared.degraded)
        self.assertEqual(prepared.guidance, "")
        self.assertEqual(len(llm.calls), 1)
        result = self.update([{"candidates": []}, {"base_revision": 0, "edits": [edit()]}], prepared=prepared)
        self.assertTrue(result.adaptation.degraded)
        self.assertEqual(self.pipeline.playbook, empty_playbook())

    def test_failed_adapter_allows_empty_edits(self):
        prepared = self.prepare(llm=LLM(["bad JSON"]))
        result = self.update([{"candidates": []}, {"base_revision": 0, "edits": []}], prepared=prepared)
        self.assertFalse(result.degraded)

    def test_rewards_do_not_gate_either_branch(self):
        for reward in (0, 0.5, 1):
            with self.subTest(reward=reward):
                self.artifacts = self.root / f"artifacts/reward-{reward}"
                llm = LLM([{"candidates": []}, {"base_revision": 0, "edits": []}])
                result = self.update([], evidence={**EVIDENCE, "reward": reward}, llm=llm)
                self.assertFalse(result.degraded)
                self.assertEqual([call["phase"] for call in llm.calls], ["task_extraction", "adaptation_update"])

    def test_frozen_evidence_does_not_expose_task_writes_to_adaptation(self):
        llm = LLM([{"candidates": [{**CANDIDATE, "content": "NEW_WRITE_SENTINEL"}]}, decision("add"),
                   {"base_revision": 0, "edits": []}])
        result = self.update([], llm=llm)
        self.assertFalse(result.degraded)
        self.assertEqual(len(self.store.writes), 1)
        self.assertNotIn("NEW_WRITE_SENTINEL", json.dumps(llm.calls[-1]))
        self.assertEqual(json.loads(llm.calls[-1]["messages"][1]["content"])["retrieved_memories"], [])

    def test_mutated_prepared_and_snapshot_are_rejected(self):
        prepared = self.prepare()
        prepared.adaptation_prepared.payload["state"]["revision"] = 9
        with self.assertRaisesRegex(ValueError, "changed"):
            self.update([], prepared=prepared)
        self.commit_pair()
        prepared = self.prepare()
        (self.root / "task-0.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.update([], prepared=prepared)

    def test_immutable_evidence_is_not_overwritten(self):
        prepared = self.prepare()
        write_json(self.artifacts / "evidence.json", {**EVIDENCE, "reward": 1})
        before = (self.artifacts / "evidence.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Immutable"):
            self.update([], prepared=prepared)
        self.assertEqual((self.artifacts / "evidence.json").read_bytes(), before)

    def test_feedback_allowlist_and_invalid_scalar(self):
        evidence = {**EVIDENCE, "report": "EVALUATOR_SENTINEL", "task_spec": {
            **TASK, "private": "EVALUATOR_SENTINEL", "supervisor": {**TASK["supervisor"], "private": "EVALUATOR_SENTINEL"}},
            "trace": [{**EVIDENCE["trace"][0], "evaluation": "EVALUATOR_SENTINEL"}]}
        llm = LLM([{"candidates": []}, {"base_revision": 0, "edits": []}])
        self.update([], evidence=evidence, llm=llm)
        self.assertNotIn("EVALUATOR_SENTINEL", json.dumps(llm.calls))
        self.assertNotIn("EVALUATOR_SENTINEL", (self.artifacts / "episode_evidence.json").read_text())
        for reward in (True, float("nan"), -1, 2):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                self.update([], evidence={**EVIDENCE, "reward": reward})

    def test_task_revision_preserves_id_refs_and_embedding_input(self):
        before = self.existing()
        llm = LLM([{"candidates": [CANDIDATE]}, decision("revise", "old", name="Revised title",
                   when_to_use="When listing", content="Check every page"), {"base_revision": 0, "edits": []}])
        result = self.update([], llm=llm)
        self.assertFalse(result.degraded)
        updated = self.store.records["old"]
        self.assertEqual(updated["id"], "old")
        self.assertEqual(updated["name"], "Revised title")
        self.assertEqual(updated["source_refs"], ["historical/evidence.json", "artifacts/task1/evidence.json"])
        self.assertEqual(result.task.decisions[0]["before"], before)
        self.assertEqual(llm.embeddings[-1][0], ["When listing\n\nCheck every page"])

    def test_sequential_candidates_observe_confirmed_writes(self):
        llm = LLM([{"candidates": [CANDIDATE, CANDIDATE]}, decision("add")])
        original = llm.chat

        def chat(messages, **controls):
            if controls["phase"] == "task_consolidation" and self.store.records:
                llm.responses.append(decision("skip", next(iter(self.store.records))))
            if controls["phase"] == "adaptation_update":
                llm.responses.append({"base_revision": 0, "edits": []})
            return original(messages, **controls)

        llm.chat = chat
        result = self.update([], llm=llm)
        self.assertEqual([row["operation"] for row in result.task.decisions], ["add", "skip"])
        self.assertEqual(len(self.store.writes), 1)

    def test_missing_evidence_defers_without_consolidation(self):
        self.existing(refs=["missing.json"])
        llm = LLM([{"candidates": [CANDIDATE]}, {"base_revision": 0, "edits": []}])
        result = self.update([], llm=llm)
        self.assertEqual(result.task.decisions[0]["operation"], "defer")
        self.assertEqual([call["phase"] for call in llm.calls], ["task_extraction", "adaptation_update"])

    def test_invalid_extraction_rejects_whole_candidate_list_but_runs_other_branch(self):
        for candidates in ([CANDIDATE] * 4, [CANDIDATE, {**CANDIDATE, "extra": "invalid"}]):
            with self.subTest(candidates=candidates):
                result = self.update([{"candidates": candidates}, {"base_revision": 0, "edits": []}])
                self.assertTrue(result.task.degraded)
                self.assertFalse(result.adaptation.degraded)
                self.assertEqual(self.store.writes, [])

    def test_invalid_edit_preserves_confirmed_task_updates(self):
        result = self.update([{"candidates": [CANDIDATE]}, decision("add"),
                              {"base_revision": 0, "edits": [{**edit(), "heading": "invalid"}]}])
        self.assertFalse(result.task.degraded)
        self.assertTrue(result.adaptation.degraded)
        self.assertEqual(len(self.store.writes), 1)
        self.assertEqual(self.pipeline.playbook, empty_playbook())

    def test_uncertain_write_and_programming_error_are_fatal(self):
        for error in (StoreWriteError("put", "id"), RuntimeError("programming error")):
            with self.subTest(error=error):
                prepared = self.prepare()
                with patch.object(self.store, "put", side_effect=error), self.assertRaises(type(error)):
                    self.update([{"candidates": [CANDIDATE]}, decision("add")], prepared=prepared)
                status = json.loads((self.artifacts / "pair_update_status.json").read_text())
                self.assertEqual(status["adaptation"]["status"], "not_run")
                self.assertEqual(status["task"]["status"], "failed")

    def test_provider_failure_is_fatal_and_request_survives(self):
        error = LLMUnavailableError(operation="chat", phase="adaptation", logical_call_id="id", attempts=6,
                                    cause=TimeoutError("provider unavailable"))
        with self.assertRaises(LLMUnavailableError):
            self.prepare(llm=LLM([error]))
        event = json.loads((self.artifacts / "adaptation_event.json").read_text())
        self.assertEqual(event["status"], "failed")
        self.assertTrue(event["messages"])

    def test_adapter_overflow_stops_before_model_without_trimming(self):
        self.pipeline.framework.adapter.max_bytes = 10
        llm = LLM()
        with self.assertRaises(PromptOverflow):
            self.prepare(llm=llm)
        self.assertEqual(llm.calls, [])
        event = json.loads((self.artifacts / "adaptation_event.json").read_text())
        self.assertEqual(event["status"], "budget_limited")
        self.assertIn("## Verification", event["input"]["playbook"])

    def test_update_overflow_defers_each_branch(self):
        prepared = self.prepare()
        self.pipeline.task_channel.max_bytes = 10
        self.pipeline.adaptation_channel.max_bytes = 10
        llm = LLM()
        result = self.update([], prepared=prepared, llm=llm)
        self.assertTrue(result.task.degraded)
        self.assertTrue(result.adaptation.degraded)
        self.assertEqual(llm.calls, [])

    def test_consolidation_overflow_keeps_sources_and_continues_candidates(self):
        self.existing()
        write_json(self.root / "historical/evidence.json", {**EVIDENCE, "trace": [{"observation": "界" * 310000}]})
        llm = LLM([{"candidates": [CANDIDATE] * 2}, {"base_revision": 0, "edits": []}])
        result = self.update([], llm=llm)
        self.assertTrue(result.task.degraded)
        self.assertEqual([row["operation"] for row in result.task.decisions], ["defer", "defer"])
        self.assertTrue(all(row["call"]["prompt_bytes"] > 900000 for row in result.task.decisions))
        self.assertFalse(result.adaptation.degraded)

    def test_named_and_legacy_shared_search_exact_hashes_and_probe_agreement(self):
        record = self.existing()
        named, _ = search(self.store, VECTOR, representation="combined", k=5, num_candidates=10)
        self.assertEqual(named[0]["name"], CANDIDATE["name"])
        bank = prepare_exact_bank(self.store, representations=["combined"], dimensions=1024)
        self.assertEqual(bank.records["old"]["name"], CANDIDATE["name"])
        with self.assertRaisesRegex(ValueError, "probe source field: name"):
            prepare_exact_bank(self.store, representations=["combined"], dimensions=1024,
                               probe_id="old", probe_source={**record, "name": "different"})
        del self.store.records["old"]["name"]
        legacy, _ = search(self.store, VECTOR, representation="combined", k=5, num_candidates=10)
        self.assertNotIn("name", legacy[0])
        other = prepare_exact_bank(self.store, representations=["combined"], dimensions=1024)
        self.assertNotEqual(bank.observed["exact_snapshot_sha256"], other.observed["exact_snapshot_sha256"])

    def test_mapping_and_prompt_source_bytes(self):
        body = memory_index_body()
        self.assertEqual(body["mappings"]["properties"]["name"], {"type": "text"})
        self.assertFalse(body["settings"]["index.mapping.exclude_source_vectors"])
        lineage = json.loads((PROMPTS / "lineage.json").read_text())
        for row in lineage["parents"]:
            self.assertEqual(hashlib.sha256((PROMPTS / row["captured_path"]).read_bytes()).hexdigest(), row["sha256"])
        self.assertEqual((PROMPTS / "execution_base.txt").read_bytes(),
                         (PROMPTS / "parents/MetaWeft/src/base/prompts/v1/execution_base.txt").read_bytes())


if __name__ == "__main__":
    unittest.main()
