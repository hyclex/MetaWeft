"""Offline V2 worker, eligibility and paired checkpoint checks with the real recorder and fake provider."""

from copy import deepcopy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from MetaWeft.experiment.appworld import worker
from MetaWeft.experiment.recovery import (
    commit_checkpoint, read_json, snapshot_adaptation_v2, snapshot_task_v2, write_json,
)
from MetaWeft.service.llm import ModelClient
from MetaWeft.src.ace_adapt_reme_task_v2 import V2Pipeline, empty_playbook, memory_index_body
from MetaWeft.src.memory_retrieval import load_preset
from MetaWeft.tests import test_integration as legacy


class Store:
    index = "offline-v2"

    def __init__(self):
        self.records = {}

    def export(self):
        return [{"id": key, "document": deepcopy(value)} for key, value in sorted(self.records.items())]

    def query(self, body):
        return {"hits": {"hits": [{"_id": key, "_source": deepcopy(value), "_score": 1.0}
                                  for key, value in list(self.records.items())[:body["size"]]]}}

    def put(self, identifier, document):
        self.records[identifier] = deepcopy(document)
        return deepcopy(document)


class V2WorkerTests(unittest.TestCase):
    make_world = legacy.AdapterTests.make_world
    evaluate = legacy.AdapterTests.evaluate

    def setUp(self):
        legacy.AdapterTests.setUp(self)
        self.plan.update(setting="metaweft_v2", run_id="offline-origin", recovery_version=2)
        self.plan["settings"].update(embedding_dimensions=1024, retrieval_k=10,
                                     retrieval_meta=load_preset("efficiency_1"))
        self.phase.update(split="test-normal", experiment_name="split")
        self.store = Store()
        self.pipeline = V2Pipeline(self.store, self.root)
        write_json(self.artifacts.parent / "memory_mapping.json", memory_index_body())
        snapshot_task_v2(self.store, self.artifacts.parent / "memory_initial.json", self.plan, self.phase, 0)
        snapshot_adaptation_v2(empty_playbook(), self.artifacts.parent / "adaptation_memory_initial.json",
                               self.plan, self.phase, 0)
        commit_checkpoint(self.plan, self.phase, [])
        self.requests, self.providers = [], []
        self.invalid_extraction = False
        self.invalid_edit = False
        self.invalid_adaptation = False

    def make_model(self, **kwargs):
        active = {}
        recorder = kwargs.pop("recorder")

        def record(operation, request, invoke, **context):
            active.update(context)
            self.requests.append((context["phase"], deepcopy(request)))
            return recorder(operation, request, invoke, **context)

        def chat(**request):
            phase = active["phase"]
            if phase == "adaptation":
                value = ({"invalid": "guide"} if self.invalid_adaptation else
                         {"guidance": "Check the observed output before reporting completion."})
            elif phase == "task_extraction":
                value = {"invalid": []} if self.invalid_extraction else {"candidates": [{
                    "name": "Inspect observed results", "when_to_use": "When completing the example",
                    "content": "Read the observed output before reporting completion.",
                }]}
            elif phase == "task_consolidation":
                value = {"operation": "add", "target_id": None, "name": None,
                         "when_to_use": None, "content": None, "reason": "New observed procedure"}
            elif phase == "adaptation_update":
                value = {"base_revision": 0, "edits": [{
                    "operation": "ADD", "target_id": None, "heading": "exploration_guidance",
                    "content": "Include an observable completion check when composing guidance.",
                    "evidence": ["/adaptation/output/guidance", "/adaptation/input/task_spec/instruction"],
                    "reason": "The adapter input omitted a result and the guide requested a check.",
                }]}
                if self.invalid_edit:
                    value["edits"][0]["evidence"] = ["/reward"]
            else:
                value = None
            content = json.dumps(value) if value is not None else "```python\nprint('observed')\n```"
            result = legacy.response(content)
            result.model_dump = lambda: {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
            return result

        provider = Mock()
        provider.chat.completions.create.side_effect = chat
        vector = [0.1] * 1024
        provider.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=vector)],
            model_dump=lambda: {"data": [{"index": 0, "embedding": vector}],
                                "usage": {"prompt_tokens": 2, "total_tokens": 2}},
        )
        self.providers.append(provider)
        return ModelClient(client=provider, recorder=record, sleep=Mock(), **kwargs)

    def run_worker(self):
        adapter = worker.MetaWeftAdapter(self.plan, self.phase, self.manifest, self.pipeline, self.store)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-for-offline-tests"}), \
             patch.object(worker, "ModelClient", side_effect=self.make_model), \
             patch.object(worker.CachedDBHandler, "reset"):
            return adapter.run_task(self.context, self.services)

    def test_worker_commits_both_memories_and_accounts_all_new_phases(self):
        result = self.run_worker().metadata
        checkpoint = read_json(self.output / "checkpoint.json")
        self.assertEqual(checkpoint["state_epoch"], 1)
        self.assertEqual(checkpoint["completed_task_ids"], ["fake_1"])
        self.assertEqual(len(self.store.records), 1)
        self.assertEqual(self.pipeline.playbook["revision"], 1)
        self.assertFalse(result["degraded"])
        self.assertIn("task", self.manifest["memory_branch_results"]["fake_1"])
        self.assertEqual(self.manifest["adaptation_memory"]["entries"], 1)
        self.assertNotIn("SECRET", json.dumps(self.requests))
        update = next(request for phase, request in self.requests if phase == "adaptation_update")
        payload = json.loads(update["messages"][1]["content"])
        self.assertEqual(set(payload), {"adaptation", "base_revision", "edit_targets"})
        self.assertEqual(set(payload["adaptation"]), {"input", "output"})
        self.assertNotIn("trace", payload)
        self.assertNotIn("reward", payload)
        snapshot = read_json(self.output / checkpoint["memories"]["adaptation"]["snapshot"])
        self.assertEqual(snapshot["schema"], "metaweft-adaptation-memory-v2")
        phases = worker.accounting(self.output)["recorded_calls_by_phase"]
        for phase in ("adaptation", "execution", "task_extraction", "task_consolidation", "adaptation_update"):
            self.assertGreaterEqual(phases[phase], 1)
        self.assertNotIn("retrieval", phases)
        self.assertIsNone(worker.accounting(self.output)["cost_usd"])
        before = len(self.requests)
        repeated = self.run_worker().metadata
        self.assertEqual(repeated, result)
        self.assertEqual(len(self.requests), before)
        for provider in self.providers:
            provider.close.assert_called_once()

    def test_degraded_task_branch_still_commits_valid_adaptation_update(self):
        self.invalid_extraction = True
        result = self.run_worker().metadata
        self.assertEqual(self.store.records, {})
        self.assertEqual(self.pipeline.playbook["revision"], 1)
        self.assertTrue(result["stages"]["update"]["task"]["degraded"])
        self.assertFalse(result["stages"]["update"]["adaptation"]["degraded"])
        self.assertEqual(read_json(self.output / "checkpoint.json")["state_epoch"], 1)

    def test_invalid_playbook_batch_keeps_confirmed_task_write(self):
        self.invalid_edit = True
        result = self.run_worker().metadata
        self.assertEqual(len(self.store.records), 1)
        self.assertEqual(self.pipeline.playbook, empty_playbook())
        self.assertFalse(result["stages"]["update"]["task"]["degraded"])
        self.assertTrue(result["stages"]["update"]["adaptation"]["degraded"])
        self.assertEqual(read_json(self.output / "checkpoint.json")["state_epoch"], 1)

    def test_required_snapshot_failure_keeps_previous_pair_and_task_position(self):
        with patch.object(worker, "snapshot_adaptation_v2", side_effect=OSError("snapshot failed")):
            with self.assertRaisesRegex(OSError, "snapshot failed"):
                self.run_worker()
        self.assertEqual(len(self.store.records), 1)
        self.assertEqual(read_json(self.output / "checkpoint.json")["state_epoch"], 0)
        self.assertEqual(self.manifest["completed_task_ids"], [])
        worker.final_pair_summary(self.output, self.artifacts.parent, {"evidence_gaps": []})
        self.assertEqual(read_json(self.artifacts.parent / "memory_final.json")["records"], [])
        self.assertEqual(read_json(self.artifacts.parent / "adaptation_memory_final.json")["playbook"],
                         empty_playbook())

    def assert_adaptation_skipped(self):
        result = self.run_worker().metadata
        self.assertNotIn("adaptation_update", [phase for phase, _ in self.requests])
        self.assertEqual(self.pipeline.playbook, empty_playbook())
        self.assertEqual(len(self.store.records), 1)
        self.assertFalse(result["stages"]["update"]["adaptation"]["degraded"])
        self.assertEqual(result["stages"]["update"]["adaptation"]["status"], "skipped")
        self.assertTrue((self.artifacts / "adaptation_updates.json").is_file())
        self.assertEqual(read_json(self.output / "checkpoint.json")["state_epoch"], 1)
        return result

    def test_partial_reward_skips_playbook_even_after_environment_completion(self):
        self.passed = 0.5
        result = self.assert_adaptation_skipped()
        self.assertEqual(result["reward"], 0.5)
        self.assertTrue(result["task_completed"])

    def test_failed_task_keeps_task_memory_learning_and_skips_playbook(self):
        self.passed = 0
        result = self.assert_adaptation_skipped()
        self.assertEqual(result["reward"], 0)

    def test_success_with_invalid_adapter_output_skips_playbook_update(self):
        self.invalid_adaptation = True
        result = self.assert_adaptation_skipped()
        self.assertEqual(result["reward"], 1)
        self.assertTrue(result["degraded"])

    def test_evaluation_failure_prevents_both_updates_and_commit(self):
        self.evaluator_error = RuntimeError("evaluation infrastructure failure")
        with self.assertRaisesRegex(RuntimeError, "evaluation infrastructure failure"):
            self.run_worker()
        self.assertEqual([phase for phase, _ in self.requests], ["adaptation", "execution"])
        self.assertEqual(read_json(self.output / "checkpoint.json")["state_epoch"], 0)
        self.assertEqual(self.store.records, {})


if __name__ == "__main__":
    unittest.main()
