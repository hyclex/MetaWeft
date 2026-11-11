"""Provider-free V2 preparation, channel isolation, continuity and provenance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from MetaWeft.src.ace_adapt_reme_task.common import write_json
from MetaWeft.src.ace_adapt_reme_task_v2 import V2Pipeline
from MetaWeft.src.ace_adapt_reme_task_v2.config import ADAPTATION_SCHEMA, TASK_SCHEMA, PROMPTS, v2_settings
from MetaWeft.tests.test_adaptation_memory_v2 import edit
from MetaWeft.tests.test_metaweft_v1 import CANDIDATE, EVIDENCE, LLM, Store, TASK, VECTOR, decision


class V2Tests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts = self.root / "artifacts/task1"
        self.store = Store()
        self.pipeline = V2Pipeline(self.store, self.root)
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

    def prepare(self, *, llm=None):
        return self.pipeline.prepare_context(TASK, llm=llm or LLM([{"guidance": "Inspect prerequisites"}]),
                                             artifacts=self.artifacts)

    def update(self, responses, *, prepared=None, reward=1, llm=None):
        return self.pipeline.post_task({**EVIDENCE, "reward": reward}, prepared=prepared or self.prepare(),
                                       llm=llm or LLM(responses), artifacts=self.artifacts)

    def test_cold_start_calls_adapter_once_and_records_v2_component(self):
        llm = LLM([{"guidance": ""}])
        prepared = self.prepare(llm=llm)
        self.assertFalse(prepared.degraded)
        self.assertEqual(prepared.guidance, "")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["component"], "metaweft_v2.adaptation")
        self.assertEqual(prepared.adaptation_event["input"]["retrieved_memories"], [])
        self.assertEqual(prepared.adaptation_event["input"]["playbook"], {
            "memory_selection": [], "task_specific_pruning": [], "exploration_guidance": []})

    def test_retrieved_metadata_stays_in_artifacts_but_adapter_gets_only_content(self):
        record = {"id": "MEMORY_ID_SENTINEL", "name": "MEMORY_NAME_SENTINEL",
                  "when_to_use": "APPLICABILITY_SENTINEL", "content": "Check prerequisites",
                  "source_refs": ["PROVENANCE_SENTINEL.json"], "embedding": list(VECTOR)}
        self.store.records[record["id"]] = record
        prepared = self.prepare()
        request = json.dumps(prepared.adaptation_event["messages"])
        self.assertEqual(prepared.adaptation_event["input"]["retrieved_memories"], [record["content"]])
        for sentinel in (record["id"], record["name"], record["when_to_use"], record["source_refs"][0]):
            self.assertNotIn(sentinel, request)
        self.assertEqual(prepared.task_prepared.payload[0]["source_refs"], record["source_refs"])
        saved = json.loads((self.artifacts / "retrieval.json").read_text())
        self.assertEqual(saved["records"][0]["id"], record["id"])

    def test_successful_update_reaches_next_task_without_playbook_ids(self):
        self.update([{"candidates": []}, {"base_revision": 0, "edits": [edit()]}])
        self.commit_pair(1)
        self.artifacts = self.root / "artifacts/task2"
        llm = LLM([{"guidance": "Discover the missing detail"}, {"candidates": []},
                   {"base_revision": 1, "edits": []}])
        prepared = self.prepare(llm=llm)
        self.update([], prepared=prepared, llm=llm)
        payload = prepared.adaptation_event["input"]
        self.assertEqual(payload["playbook"]["exploration_guidance"], [edit()["content"]])
        self.assertNotIn("adapt-000001", json.dumps(payload))
        self.assertNotIn("source_refs", json.dumps(payload))
        update_request = json.loads(llm.calls[-1]["messages"][1]["content"])
        self.assertEqual(update_request["edit_targets"], {
            "adapt-000001": "/adaptation/input/playbook/exploration_guidance/0"})
        self.assertEqual(self.pipeline.playbook["revision"], 1)

    def test_reward_gate_does_not_disable_task_learning_or_change_task_prompt(self):
        llm = LLM([{"candidates": [CANDIDATE]}, decision("add")])
        result = self.update([], reward=0.5, llm=llm)
        self.assertEqual(result.adaptation.status, "skipped")
        self.assertFalse(result.degraded)
        self.assertEqual(len(self.store.records), 1)
        self.assertEqual([call["phase"] for call in llm.calls], ["task_extraction", "task_consolidation"])
        self.assertTrue(all(call["component"].startswith("metaweft_v2.") for call in llm.calls))
        self.assertEqual(llm.embeddings[0][1]["component"], "metaweft_v2.task_embedding")
        parent_prompt = PROMPTS.parents[2] / "ace_adapt_reme_task/prompts/v1/task_extraction.txt"
        self.assertEqual(llm.calls[0]["messages"][0]["content"], parent_prompt.read_text())

    def test_success_with_invalid_adapter_skips_only_adaptation_update(self):
        prepared = self.prepare(llm=LLM(["invalid JSON"]))
        llm = LLM([{"candidates": []}])
        result = self.update([], prepared=prepared, llm=llm)
        self.assertTrue(prepared.degraded)
        self.assertEqual(result.adaptation.status, "skipped")
        self.assertFalse(result.adaptation.degraded)
        self.assertEqual(len(llm.calls), 1)
        event = json.loads((self.artifacts / "adaptation_updates.json").read_text())
        self.assertEqual(event["reason"], "adapter_not_completed")

    def test_trace_and_evaluator_data_are_retained_or_filtered_at_correct_boundaries(self):
        prepared = self.prepare()
        evidence = {**deepcopy(EVIDENCE), "reward": 1, "evaluation": "PRIVATE_EVALUATION_SENTINEL"}
        evidence["trace"][0]["observation"] = "TRAJECTORY_OBSERVATION_SENTINEL"
        evidence["trace"][0]["private_assertion"] = "PRIVATE_ASSERTION_SENTINEL"
        llm = LLM([{"candidates": []}, {"base_revision": 0, "edits": []}])
        self.pipeline.post_task(evidence, prepared=prepared, llm=llm, artifacts=self.artifacts)
        extraction = json.loads(llm.calls[0]["messages"][1]["content"])
        adaptation = json.loads(llm.calls[1]["messages"][1]["content"])
        self.assertIn("TRAJECTORY_OBSERVATION_SENTINEL", json.dumps(extraction))
        for marker in ("TRAJECTORY_OBSERVATION_SENTINEL", "PRIVATE_EVALUATION_SENTINEL", "PRIVATE_ASSERTION_SENTINEL"):
            self.assertNotIn(marker, json.dumps(adaptation))
        preserved = (self.artifacts / "episode_evidence.json").read_text()
        self.assertIn("TRAJECTORY_OBSERVATION_SENTINEL", preserved)
        self.assertNotIn("PRIVATE_EVALUATION_SENTINEL", preserved)


class ProvenanceTests(unittest.TestCase):
    def test_unchanged_prompts_parent_captures_and_settings(self):
        parent = PROMPTS.parents[2] / "ace_adapt_reme_task/prompts/v1"
        for name in ("task_extraction", "task_consolidation", "execution_base"):
            self.assertEqual((PROMPTS / f"{name}.txt").read_bytes(), (parent / f"{name}.txt").read_bytes())
        lineage = json.loads((PROMPTS / "lineage.json").read_text())
        for record in lineage["parents"]:
            captured = PROMPTS / record["captured_path"]
            self.assertEqual(hashlib.sha256(captured.read_bytes()).hexdigest(), record["sha256"])
            dependency = record.get("shared_working_dependency")
            if dependency:
                self.assertEqual(hashlib.sha256((PROMPTS / dependency["captured_path"]).read_bytes()).hexdigest(),
                                 dependency["sha256"])
        settings = v2_settings()
        self.assertEqual(settings["retrieval_k"], 10)
        self.assertEqual(settings["retrieval_preset"], "efficiency_1")
        self.assertEqual(settings["components"]["adaptation_memory"], ADAPTATION_SCHEMA)
        self.assertEqual(settings["pair_schema"], 2)
        self.assertEqual(settings["adaptation_success_reward"], 1)
        self.assertIn("v2_prompt_manifest", settings)
        self.assertNotIn("v1_prompt_manifest", settings)


if __name__ == "__main__":
    unittest.main()
