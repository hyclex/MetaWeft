"""Offline pair commit, corruption and interruption checks; no provider or ES calls."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from MetaWeft.experiment import recovery
from MetaWeft.src.ace_adapt_reme_task.adaptation_memory import empty_playbook
from MetaWeft.src.base.pipeline import memory_index_body


class Store:
    def __init__(self, records=()):
        self.records = {record["id"]: deepcopy(record["document"]) for record in records}

    def export(self):
        return [{"id": key, "document": deepcopy(value)} for key, value in sorted(self.records.items())]

    def put(self, identifier, document):
        self.records[identifier] = deepcopy(document)


class PairRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "parent"
        self.output = self.root / "experiments/outputs/test-normal"
        self.artifacts = self.output / "baseline_artifacts/metaweft"
        self.phase = {"split": "test-normal", "experiment_name": "test-normal", "dataset": "fixture",
                      "ordered_task_ids": ["a_1", "b_1"], "source_task_count": 2, "partial": False,
                      "dataset_sha256": "dataset", "selected_dataset_sha256": "selected",
                      "output_directory": str(self.output), "completed_task_ids": []}
        self.plan = {"setting": "metaweft_v1", "setting_id": "metaweft_v1-fixture",
                     "variant_id": "metaweft-ace-adapt-reme-task-v1", "scenario": "cold-start",
                     "run_id": "original", "run_directory": str(self.root), "dataset_version": "fixture",
                     "provider": {"chat_base_url": "https://provider.invalid"},
                     "recovery_version": 2, "selection": {}, "settings": {"embedding_dimensions": 2},
                     "phases": [self.phase], "status": "failed", "data_source": str(self.root.parent / "data")}
        for name in ("variant.json", "source_provenance.json", "data_provenance.json", "runtime.json"):
            recovery.write_json(self.root / name, {})
        (self.root / "appworld-requirements.txt").write_text("fixture\n")
        for name in ("source/MetaWeft", "source/project/experiments", "inputs", "data/datasets"):
            (self.root / name).mkdir(parents=True)
        Path(self.plan["data_source"]).mkdir()
        recovery.seal_run(self.plan)
        mapping = memory_index_body(2)
        mapping["mappings"]["properties"]["name"] = {"type": "text"}
        recovery.write_json(self.artifacts / "memory_mapping.json", mapping)
        self.store, self.playbook = Store(), empty_playbook()
        recovery.snapshot_task_v2(self.store, self.artifacts / "memory_initial.json", self.plan, self.phase, 0)
        recovery.snapshot_adaptation_v2(self.playbook, self.artifacts / "adaptation_memory_initial.json",
                                        self.plan, self.phase, 0)
        self.initial = recovery.commit_checkpoint(self.plan, self.phase, [])

    def task_artifacts(self, task="a_1"):
        for directory in ("dbs", "checkpoints", "logs", "evaluation", "version", "misc"):
            (self.output / "tasks" / task / directory).mkdir(parents=True)
        (self.output / "tasks" / task / "logs/lm_calls.jsonl").write_text("")
        (self.output / "tasks" / task / "evaluation/report.md").write_text("private evaluation")
        for name in recovery.V1_TASK_ARTIFACTS:
            if name not in ("memory_after.json", "adaptation_memory_after.json"):
                recovery.write_json(self.artifacts / task / name, {})
        recovery.write_json(self.artifacts / task / "evidence.json",
                            {"task_spec": {"task_id": task}, "trace": [{"observation": "observed"}], "reward": 0})
        recovery.write_json(self.artifacts / task / "adaptation_updates.json", {"applied": [{"operation": "ADD"}]})
        recovery.write_json(self.artifacts / task / "result.json",
                            {"degraded": [], "stages": {"update": {"task": {"status": "completed"},
                                                                   "adaptation": {"status": "completed"}}}})

    def changed_pair(self):
        self.task_artifacts()
        prefix = "experiments/outputs/test-normal/baseline_artifacts/metaweft/a_1/"
        self.store.put("memory-1", {"id": "memory-1", "name": "Remember a procedure", "when_to_use": "When relevant",
                                    "content": "Use the observed procedure", "embedding": [0.25, -0.5],
                                    "source_refs": [prefix + "evidence.json"]})
        self.playbook = {"schema_version": 1, "revision": 1, "next_entry_number": 2, "entries": [
            {"id": "adapt-000001", "heading": "verification", "content": "Check assumptions in the guide",
             "source_refs": [prefix + "adaptation_updates.json#/applied/0"]}]}

    def snapshots(self, epoch=1, task="a_1"):
        recovery.snapshot_task_v2(self.store, self.artifacts / task / "memory_after.json",
                                 self.plan, self.phase, epoch)
        recovery.snapshot_adaptation_v2(self.playbook, self.artifacts / task / "adaptation_memory_after.json",
                                       self.plan, self.phase, epoch)

    def commit(self):
        self.snapshots()
        return recovery.commit_checkpoint(self.plan, self.phase, ["a_1"], previous=self.initial)

    def test_empty_pair_and_noop_epoch_have_exact_refs_and_restoration(self):
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), self.initial)
        self.assertEqual(self.initial["state_epoch"], 0)
        self.assertNotIn("memory_file", self.initial)
        references = recovery.state_refs(self.plan, self.phase, self.initial)
        for role, reference in references.items():
            self.assertEqual(reference["stream_id"], "original:test-normal")
            self.assertEqual(reference["epoch"], 0)
            self.assertEqual(reference["role"], role)
            self.assertEqual(reference["sha256"], recovery.sha256(self.root / reference["snapshot"]))
        self.task_artifacts()
        checkpoint = self.commit()
        self.assertEqual(checkpoint["state_epoch"], 1)
        snapshot = recovery.read_json(self.output / checkpoint["memories"]["adaptation"]["snapshot"])
        self.assertEqual(recovery.restore_adaptation(snapshot), empty_playbook())
        self.assertEqual(snapshot["playbook"]["revision"], 0)

    def test_nonempty_pair_preserves_vectors_ids_and_both_sources(self):
        self.changed_pair()
        checkpoint = self.commit()
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), checkpoint)
        restored = Store()
        recovery.restore_memory(restored, recovery.read_json(self.output / checkpoint["memories"]["task"]["snapshot"]))
        self.assertEqual(restored.export(), self.store.export())
        snapshot = recovery.read_json(self.output / checkpoint["memories"]["adaptation"]["snapshot"])
        self.assertEqual(recovery.restore_adaptation(snapshot), self.playbook)
        for name in ("evidence.json", "adaptation_updates.json"):
            self.assertIn(f"baseline_artifacts/metaweft/a_1/{name}", checkpoint["files"])

    def test_failed_provisional_writes_and_snapshots_never_advance_checkpoint(self):
        self.changed_pair()
        # An uncertain ES mutation or an in-memory playbook draft has no commit authority.
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), self.initial)
        with patch.object(self.store, "export", side_effect=RuntimeError("failed export")):
            with self.assertRaisesRegex(RuntimeError, "failed export"):
                self.snapshots()
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), self.initial)
        original_write = recovery.write_json
        for suffix in ("memory_after.json", "adaptation_memory_after.json"):
            def interrupted(path, value):
                if Path(path).name == suffix:
                    raise OSError("snapshot interrupted")
                original_write(path, value)
            with self.subTest(snapshot=suffix), patch.object(recovery, "write_json", side_effect=interrupted):
                with self.assertRaisesRegex(OSError, "snapshot interrupted"):
                    self.snapshots()
            self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), self.initial)
        self.snapshots()
        before = (self.output / "checkpoint.json").read_bytes()
        with patch.object(recovery.os, "replace", side_effect=OSError("publication interrupted")):
            with self.assertRaisesRegex(OSError, "publication interrupted"):
                recovery.commit_checkpoint(self.plan, self.phase, ["a_1"], previous=self.initial)
        self.assertEqual((self.output / "checkpoint.json").read_bytes(), before)
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), self.initial)

    def test_committed_checkpoint_overrides_a_stale_manifest_and_clone_preserves_pair(self):
        self.changed_pair()
        checkpoint = self.commit()
        recovery.write_json(self.output / "run_manifest.json", {"completed_task_ids": []})
        # Failed next-task drafts and calls are archived, never restored as the next pair.
        self.task_artifacts("b_1")
        child_root = self.root.parent / "child"
        child = recovery.recovery_plan(self.plan, self.root, [checkpoint], root=child_root,
                                       tag="child", command=["offline"])
        recovery.clone_run(self.root, child, [checkpoint])
        phase = child["phases"][0]
        self.assertEqual(phase["completed_task_ids"], ["a_1"])
        self.assertEqual(phase["remaining_task_ids"], ["b_1"])
        self.assertEqual(set(phase["memory_branch_results"]), {"a_1"})
        self.assertEqual(phase["adaptation_memory"]["revision"], 1)
        output = Path(phase["output_directory"])
        self.assertFalse((output / "tasks/b_1").exists())
        self.assertTrue((output / "recovery_history/original/tasks/b_1").is_dir())
        copied = recovery.load_checkpoint(child, phase, child_root)
        self.assertEqual(copied["memories"], checkpoint["memories"])
        self.assertEqual(recovery.stream_id(child, phase), "original:test-normal")
        original = recovery.read_json(self.output / checkpoint["memories"]["adaptation"]["snapshot"])
        restored = recovery.read_json(output / copied["memories"]["adaptation"]["snapshot"])
        self.assertEqual(restored, original)

    def test_first_task_recovery_keeps_empty_pair_and_archives_partial_artifacts(self):
        self.changed_pair()
        self.snapshots()
        child_root = self.root.parent / "empty-child"
        child = recovery.recovery_plan(self.plan, self.root, [self.initial], root=child_root,
                                       tag="empty-child", command=["offline"])
        recovery.clone_run(self.root, child, [self.initial])
        phase = child["phases"][0]
        output = Path(phase["output_directory"])
        checkpoint = recovery.load_checkpoint(child, phase, child_root)
        self.assertEqual(checkpoint["completed_task_ids"], [])
        self.assertEqual(checkpoint["next_task_id"], "a_1")
        self.assertFalse((output / "baseline_artifacts/metaweft/a_1").exists())
        self.assertTrue((output / "recovery_history/original/baseline_artifacts/metaweft/a_1").is_dir())
        restored = Store()
        recovery.restore_memory(restored, recovery.read_json(output / checkpoint["memories"]["task"]["snapshot"]))
        self.assertEqual(restored.export(), [])
        self.assertEqual(recovery.restore_adaptation(recovery.read_json(
            output / checkpoint["memories"]["adaptation"]["snapshot"])), empty_playbook())
        self.assertEqual(recovery.accounting(output)["current_attempt"]["recorded_calls"], 0)

    def test_incomplete_artifacts_and_hash_failures_preserve_prior_pair(self):
        self.changed_pair()
        self.snapshots()
        required = self.artifacts / "a_1/episode_evidence.json"
        required.unlink()
        with self.assertRaisesRegex(ValueError, "required task artifact"):
            recovery.commit_checkpoint(self.plan, self.phase, ["a_1"], previous=self.initial)
        recovery.write_json(required, {})
        with patch.object(recovery, "file_hashes", side_effect=OSError("hash failure")):
            with self.assertRaisesRegex(OSError, "hash failure"):
                recovery.commit_checkpoint(self.plan, self.phase, ["a_1"], previous=self.initial)
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), self.initial)

    def test_a_new_commit_cannot_rehash_changed_historical_source_evidence(self):
        self.changed_pair()
        checkpoint = self.commit()
        self.task_artifacts("b_1")
        self.snapshots(epoch=2, task="b_1")
        path = self.artifacts / "a_1/evidence.json"
        evidence = recovery.read_json(path)
        evidence["reward"] = 0.5
        recovery.write_json(path, evidence)
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            recovery.commit_checkpoint(self.plan, self.phase, ["a_1", "b_1"], previous=checkpoint)
        self.assertEqual(recovery.read_json(self.output / "checkpoint.json"), checkpoint)

    def test_corrupt_pair_is_rejected_even_if_file_hash_is_recomputed(self):
        self.changed_pair()
        checkpoint = self.commit()
        name = checkpoint["memories"]["adaptation"]["snapshot"]
        path = self.output / name
        original = recovery.read_json(path)
        mutations = [
            lambda value: value.update(stream_id="other:test-normal"),
            lambda value: value.update(state_epoch=0),
            lambda value: value.update(state_epoch=True),
            lambda value: value.update(schema="unknown"),
            lambda value: value.update(role="task"),
            lambda value: value["playbook"].update(revision=2),
            lambda value: value["playbook"].update(next_entry_number=1),
            lambda value: value["playbook"]["entries"].append(deepcopy(value["playbook"]["entries"][0])),
            lambda value: value["playbook"]["entries"][0].update(source_refs=[
                "experiments/outputs/test-normal/baseline_artifacts/metaweft/a_1/adaptation_updates.json#/applied/9"]),
        ]
        for index, mutate in enumerate(mutations):
            value = deepcopy(original)
            mutate(value)
            recovery.write_json(path, value)
            changed = deepcopy(checkpoint)
            changed["files"][name] = recovery.sha256(path)
            recovery.write_json(self.output / "checkpoint.json", changed)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                recovery.load_checkpoint(self.plan, self.phase, self.root)
        recovery.write_json(path, original)
        recovery.write_json(self.output / "checkpoint.json", checkpoint)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            recovery.load_checkpoint(self.plan, self.phase, self.root)

    def test_wrong_boundary_format_sources_and_incomplete_vectors_cannot_commit(self):
        self.changed_pair()
        self.snapshots()
        for mutation in (lambda value: value["records"][0]["document"].update(embedding=[0.25]),
                         lambda value: value["records"][0]["document"].update(name=""),
                         lambda value: value["records"][0]["document"].update(id="different"),
                         lambda value: value["records"][0]["document"].update(source_refs=["../outside.json"])):
            self.snapshots()
            path = self.artifacts / "a_1/memory_after.json"
            value = recovery.read_json(path)
            mutation(value)
            recovery.write_json(path, value)
            with self.assertRaises(ValueError):
                recovery.commit_checkpoint(self.plan, self.phase, ["a_1"], previous=self.initial)
        self.snapshots()
        checkpoint = recovery.commit_checkpoint(self.plan, self.phase, ["a_1"], previous=self.initial)
        for modify in (lambda value: value.update(state_epoch=2),
                       lambda value: value.update(version=1),
                       lambda value: value["memories"]["adaptation"].update(
                           snapshot="baseline_artifacts/metaweft/b_1/adaptation_memory_after.json")):
            changed = deepcopy(checkpoint)
            modify(changed)
            recovery.write_json(self.output / "checkpoint.json", changed)
            with self.assertRaises(ValueError):
                recovery.load_checkpoint(self.plan, self.phase, self.root)
        self.assertEqual(recovery.recovery_version("base"), 1)
        self.assertEqual(recovery.recovery_version("direct"), 1)
        self.assertEqual(recovery.recovery_version("efficiency_1"), 1)
        with self.assertRaisesRegex(ValueError, "format"):
            recovery.load_checkpoint({**self.plan, "recovery_version": 1}, self.phase, self.root)


if __name__ == "__main__":
    unittest.main()
