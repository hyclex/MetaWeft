"""Offline report regression checks using an isolated raw-log workspace."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WORKSPACE = Path(__file__).resolve().parents[2]


class CollectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        (self.workspace / "tool").mkdir()
        (self.workspace / "raw_logs").mkdir()
        shutil.copy2(WORKSPACE / "tool/collect_appworld.sh", self.workspace / "tool")

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def add_run(self, baseline, *, setting="base", status="completed", scenario=None,
                phase_diagnostics=None, evaluation=True):
        run = self.workspace / "raw_logs" / f"{baseline}-{setting}" / "experiment/output/run"
        # Saved absolute paths deliberately differ from the discovered location.
        original = Path("/original/workspace/raw_logs") / f"{baseline}-{setting}/experiment/output/run"
        relative_output = Path("experiments/outputs/test")
        output = run / relative_output
        phase = {
            "experiment_name": "test", "output_directory": str(original / relative_output),
            "dataset": "quick2h_test_normal_v2", "dataset_version": "2h", "split": "test-normal",
            "ordered_task_ids": ["one_1", "two_1"], "source_task_count": 18,
            "partial": True, "status": status,
        }
        if baseline == "ace":
            phase["adapt"] = True
        invocation = {
            "baseline": baseline, "run_id": f"{baseline}-run", "run_directory": str(original),
            "status": status, "setting": setting, "note": "fixture",
            "dataset_version": "2h",
            "scenario": scenario or ("online" if baseline == "ace" else "cold-start"),
            "phases": [phase],
        }
        self.write_json(run / "run_manifest.json", invocation)
        if phase_diagnostics is not None:
            self.write_json(output / "run_manifest.json", {**phase, **phase_diagnostics})
        if evaluation:
            self.write_json(output / "evaluations/quick2h_test_normal_v2.json", {
                "aggregate": {"task_goal_completion": 50.0, "scenario_goal_completion": 25.0},
            })
        return run, output

    def collect(self):
        result = subprocess.run(
            ["bash", "tool/collect_appworld.sh"], cwd=self.workspace,
            capture_output=True, text=True, check=True,
        )
        return (self.workspace / "res/appworld_2h.md").read_text(encoding="utf-8"), result.stdout

    def test_separate_table_keeps_legacy_rows_and_uses_complete_phase_diagnostics(self):
        self.add_run("reme")
        self.add_run("ace")
        self.add_run("metaweft", phase_diagnostics={
            "completed_task_ids": ["one_1", "two_1"],
            "degraded_task_ids": ["two_1"],
            "degraded_reasons": {"two_1": ["adaptation failed"]},
        })
        report, stdout = self.collect()
        self.assertIn("**MetaWeft**", report)
        self.assertIn("MetaWeft: 1", stdout)
        for baseline, scenario in (("reme", "cold start"), ("ace", "cold start")):
            relative = f"../raw_logs/{baseline}-base/experiment/output/run"
            expected = (
                f"| [{baseline}-run/test]({relative}/run_manifest.json) | fixture | {scenario} | "
                "appworld | 2h (v2) | test-normal | "
                f"[50.0% / 25.0%]({relative}/experiments/outputs/test/evaluations/quick2h_test_normal_v2.json) | "
                "Complete; subset: 2 tasks |"
            )
            self.assertIn(expected, report)
        self.assertIn("| fixture | base | cold start | appworld |", report)
        self.assertIn("| 2 / 2 / 18 tasks; subset | Complete; degraded 1/2 tasks |", report)
        self.assertIn("../raw_logs/metaweft-base/experiment/output/run/experiments/outputs/test/run_manifest.json", report)
        self.assertNotIn("/original/workspace", report)

    def test_running_phase_has_progress_degradation_and_missing_scores(self):
        self.add_run("metaweft", status="planned",
                     phase_diagnostics={
                         "status": "running", "started_at": "2020-01-01T00:00:00Z",
                         "completed_task_ids": ["one_1"], "degraded_task_ids": ["one_1"],
                         "degraded_reasons": {"one_1": ["adaptation provider failed"]},
                     }, evaluation=False)
        report, _ = self.collect()
        self.assertIn("| fixture | base | cold start | appworld |", report)
        self.assertIn("| N/A / N/A | 1 / 2 / 18 tasks; subset | Running; degraded 1/2 tasks; elapsed ", report)
        self.assertIn("; ETA ~", report)
        self.assertIn("no aggregate evaluation saved", report)
        self.assertIn("`N/A` means no aggregate result was saved, rather than a zero score.", report)

    def test_direct_setting_needs_no_ace_adaptation_fields(self):
        self.add_run("metaweft", setting="direct", scenario="direct", phase_diagnostics={
            "completed_task_ids": ["one_1", "two_1"], "degraded_task_ids": [],
            "degraded_reasons": {},
        })
        report, _ = self.collect()
        self.assertIn("| fixture | direct | direct | appworld |", report)
        self.assertIn("| 2 / 2 / 18 tasks; subset | Complete; degraded 0/2 tasks |", report)

    def test_missing_diagnostics_stay_unknown_and_use_invocation_link(self):
        self.add_run("metaweft", evaluation=False)
        report, _ = self.collect()
        self.assertIn("| N/A / 2 / 18 tasks; subset | Complete; degraded N/A/2 tasks", report)
        self.assertIn("[metaweft-run/test](../raw_logs/metaweft-base/experiment/output/run/run_manifest.json)", report)

    def test_completed_phase_keeps_final_snapshot_evidence_gap_visible(self):
        self.add_run("metaweft", phase_diagnostics={
            "completed_task_ids": ["one_1", "two_1"], "degraded_task_ids": [],
            "degraded_reasons": {}, "final_memory_status": "unavailable",
            "evidence_gaps": [{"stage": "final_memory_snapshot", "reason": "store unavailable"}],
        })
        report, _ = self.collect()
        self.assertIn("Complete; degraded 0/2 tasks; evidence gaps 1 |", report)

    def test_recovery_is_labeled_and_inherited_tasks_do_not_drive_eta(self):
        self.add_run("metaweft", status="running", phase_diagnostics={
            "status": "running", "started_at": "2020-01-01T00:00:00Z",
            "completed_task_ids": ["one_1"], "reused_task_ids": ["one_1"],
            "degraded_task_ids": [], "recovery": {"parent_run_id": "failed-parent"},
        }, evaluation=False)
        report, _ = self.collect()
        self.assertIn("recovered from failed-parent; reused 1 tasks", report)
        self.assertIn("ETA estimating (no new tasks completed)", report)


if __name__ == "__main__":
    unittest.main()
