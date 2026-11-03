"""Read-only launch selection and runtime dependency checks."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "MetaWeft/experiment/appworld"
spec = importlib.util.spec_from_file_location("metaweft_launch", ENTRY / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LaunchTests(unittest.TestCase):
    def test_same_two_tasks_and_unique_run_identity(self):
        selected = []
        for setting in ("base", "direct"):
            args = launch.parse_args(setting, ["--dataset", "2h", "--split", "test-normal", "--limit", "2"])
            plan = launch.make_plan(setting, args)
            selected.append(plan["phases"][0]["ordered_task_ids"])
            self.assertEqual(plan["phases"][0]["source_task_count"], 21)
            self.assertTrue(plan["partial"])
            self.assertNotEqual(plan["run_id"], launch.make_plan(setting, args)["run_id"])
        self.assertEqual(selected, [["ff58e36_1", "ff58e36_2"]] * 2)

    def test_test_all_keeps_separate_ordered_phases(self):
        args = launch.parse_args("base", ["--dataset", "2h", "--limit", "1"])
        phases = launch.make_plan("base", args)["phases"]
        self.assertEqual([p["split"] for p in phases], ["test-normal", "test-challenge"])
        self.assertNotEqual(phases[0]["output_directory"], phases[1]["output_directory"])

    def test_dry_run_needs_no_runtime_or_credentials_and_writes_nothing(self):
        raw = ROOT / "raw_logs"
        before = sorted(str(p) for p in raw.glob("metaweft-*/experiment/output/*"))
        environment = {**os.environ, "APPWORLD_RUNTIME_PROFILE": "/nonexistent-metaweft-profile"}
        environment.pop("OPENAI_API_KEY", None)
        completed = subprocess.run(
            [sys.executable, "-B", str(ENTRY / "direct.py"), "--dataset", "2h",
             "--split", "test-normal", "--limit", "2", "--dry-run"],
            cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(completed.stdout)["setting"], "direct")
        self.assertEqual(before, sorted(str(p) for p in raw.glob("metaweft-*/experiment/output/*")))

    def test_direct_runtime_ignores_es_and_service_interpreter_overrides(self):
        with patch.dict(os.environ, {"REME_ES_BINARY": "/nonexistent/es", "REME_PYTHON": "/nonexistent/python"}):
            runtime = launch.load_runtime(required_assets=("appworld_python", "installed_source", "data_root"))
        self.assertTrue(Path(runtime["appworld_python"]).is_file())

    def test_invalid_task_and_limit_are_rejected(self):
        for flags in (["--limit", "0"], ["--task-id", "not-a-task"]):
            args = launch.parse_args("base", ["--dataset", "2h", "--split", "test-normal", *flags])
            with self.assertRaises(ValueError):
                launch.make_plan("base", args)


if __name__ == "__main__":
    unittest.main()
