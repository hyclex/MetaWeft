"""Offline V2 schema selection and paired recovery; checkpoint format stays 2."""

from copy import deepcopy
from pathlib import Path
import unittest

from MetaWeft.experiment import recovery
from MetaWeft.src.ace_adapt_reme_task_v2 import empty_playbook
from MetaWeft.tests import test_recovery_v2 as parent


class V2RecoveryTests(unittest.TestCase):
    task_artifacts = parent.PairRecoveryTests.task_artifacts
    snapshots = parent.PairRecoveryTests.snapshots
    commit = parent.PairRecoveryTests.commit

    def setUp(self):
        parent.PairRecoveryTests.setUp(self)
        self.plan.update(setting="metaweft_v2", setting_id="metaweft_v2-fixture",
                         variant_id="metaweft-ace-adapt-reme-task-v2")
        recovery.seal_run(self.plan)
        self.playbook = empty_playbook()
        recovery.snapshot_task_v2(self.store, self.artifacts / "memory_initial.json", self.plan, self.phase, 0)
        recovery.snapshot_adaptation_v2(self.playbook, self.artifacts / "adaptation_memory_initial.json",
                                       self.plan, self.phase, 0)
        self.initial = recovery.commit_checkpoint(self.plan, self.phase, [])

    def changed_pair(self):
        parent.PairRecoveryTests.changed_pair(self)
        self.playbook["schema_version"] = 2
        self.playbook["entries"][0].update(heading="exploration_guidance",
                                            content="Ask for missing task facts before specializing a memory.")

    def test_v2_pair_uses_distinct_schema_and_preserves_nonempty_playbook(self):
        self.assertEqual(recovery.recovery_version("metaweft_v2"), 2)
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), self.initial)
        self.assertEqual(recovery.state_refs(self.plan, self.phase, self.initial)["adaptation"]["schema"],
                         "metaweft-adaptation-memory-v2")
        self.changed_pair()
        checkpoint = self.commit()
        snapshot = recovery.read_json(self.output / checkpoint["memories"]["adaptation"]["snapshot"])
        self.assertEqual(snapshot["schema"], "metaweft-adaptation-memory-v2")
        self.assertEqual(recovery.restore_adaptation(snapshot), self.playbook)
        self.assertNotIn("adapt-000001", recovery.render_adaptation(self.playbook, "metaweft_v2"))
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), checkpoint)

    def test_clone_keeps_v2_headings_and_completed_task_pair(self):
        self.changed_pair()
        checkpoint = self.commit()
        self.task_artifacts("b_1")
        root = self.root.parent / "v2-child"
        child = recovery.recovery_plan(self.plan, self.root, [checkpoint], root=root,
                                       tag="child", command=["offline"])
        recovery.clone_run(self.root, child, [checkpoint])
        phase = child["phases"][0]
        self.assertEqual(phase["remaining_task_ids"], ["b_1"])
        self.assertEqual(phase["adaptation_memory"]["revision"], 1)
        self.assertEqual(set(phase["memory_branch_results"]), {"a_1"})
        restored = recovery.load_checkpoint(child, phase, root)
        snapshot = recovery.read_json(Path(phase["output_directory"]) / restored["memories"]["adaptation"]["snapshot"])
        self.assertEqual(recovery.restore_adaptation(snapshot), self.playbook)
        self.assertTrue((Path(phase["output_directory"]) / "recovery_history/original/tasks/b_1").is_dir())

    def test_wrong_variant_schema_or_heading_is_rejected(self):
        self.changed_pair()
        checkpoint = self.commit()
        name = checkpoint["memories"]["adaptation"]["snapshot"]
        path = self.output / name
        original = recovery.read_json(path)
        for mutate in (lambda value: value.update(schema="metaweft-adaptation-memory-v1"),
                       lambda value: value["playbook"].update(schema_version=1),
                       lambda value: value["playbook"]["entries"][0].update(heading="verification")):
            changed = deepcopy(original)
            mutate(changed)
            recovery.write_json(path, changed)
            rehashed = deepcopy(checkpoint)
            rehashed["files"][name] = recovery.sha256(path)
            recovery.write_json(self.output / "checkpoint.json", rehashed)
            with self.assertRaises(ValueError):
                recovery.load_checkpoint(self.plan, self.phase, self.root)
        with self.assertRaises(ValueError):
            recovery.restore_adaptation({**original, "schema": "metaweft-adaptation-memory-v1"})

    def test_skipped_playbook_update_still_advances_pair_epoch(self):
        self.changed_pair()
        checkpoint = self.commit()
        self.task_artifacts("b_1")
        recovery.write_json(self.artifacts / "b_1/adaptation_updates.json", {"status": "skipped", "applied": []})
        self.snapshots(epoch=2, task="b_1")
        final = recovery.commit_checkpoint(self.plan, self.phase, ["a_1", "b_1"], previous=checkpoint)
        self.assertEqual(final["state_epoch"], 2)
        snapshot = recovery.read_json(self.output / final["memories"]["adaptation"]["snapshot"])
        self.assertEqual(snapshot["playbook"]["revision"], 1)
        self.assertEqual(recovery.load_checkpoint(self.plan, self.phase, self.root), final)


if __name__ == "__main__":
    unittest.main()
