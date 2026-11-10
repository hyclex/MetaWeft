"""Offline V1 setting dispatch, provenance and legacy-default checks."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from MetaWeft.experiment.appworld import launch
from MetaWeft.experiment.recovery import read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "MetaWeft/experiment/appworld"


class V1LaunchTests(unittest.TestCase):
    def plan(self, *flags):
        args = launch.parse_args("metaweft_v1", ["--dataset", "2h", "--limit", "2", *flags])
        return launch.make_plan("metaweft_v1", args), args

    def test_v1_resolves_separate_composition_without_preset_fallback(self):
        with patch.object(launch, "load_preset", side_effect=AssertionError("V1 used launcher preset fallback")):
            plan, _ = self.plan()
        settings = plan["settings"]
        self.assertEqual(plan["setting"], "metaweft_v1")
        self.assertEqual(plan["variant_id"], "metaweft-ace-adapt-reme-task-v1")
        self.assertEqual(plan["recovery_version"], 2)
        self.assertEqual(settings["retrieval_meta"], launch.load_preset("efficiency_1"))
        self.assertEqual(settings["retrieval_k"], 10)
        self.assertEqual(settings["retrieval_meta"]["search"]["depth"], 20)
        self.assertEqual(settings["retrieval_meta"]["reranking"]["stages"][0]["output_limit"], 20)
        self.assertEqual(settings["full_prompt_max_bytes"], 900000)
        self.assertEqual(settings["max_interactions"], 40)
        self.assertEqual(settings["execution_temperature"], 0.9)
        self.assertEqual(settings["memory_temperature"], 0)
        self.assertEqual(settings["seed"], 0)
        self.assertEqual(settings["appworld_seed"], 100)
        self.assertEqual(plan["phases"][0]["ordered_task_ids"], ["ff58e36_1", "ff58e36_2"])
        self.assertEqual([p["split"] for p in plan["phases"]], ["test-normal", "test-challenge"])
        self.assertNotEqual(plan["phases"][0]["output_directory"], plan["phases"][1]["output_directory"])
        self.assertNotEqual(plan["run_id"], self.plan()[0]["run_id"])
        for setting in ("base", "direct", "efficiency_1"):
            old = launch.make_plan(setting, launch.parse_args("base", ["--dataset", "2h", "--limit", "2"]))
            self.assertEqual(old["settings"]["retrieval_k"], 5)
            self.assertEqual(old["recovery_version"], 1)
            self.assertNotEqual(Path(plan["run_directory"]).parent, Path(old["run_directory"]).parent)

    def test_shell_dry_run_is_credential_free_and_has_no_writes(self):
        raw = ROOT / "raw_logs"
        before = sorted(str(p) for p in raw.glob("metaweft-*/experiment/output/*"))
        environment = {**os.environ, "APPWORLD_RUNTIME_PROFILE": "/nonexistent-metaweft-profile"}
        environment.pop("OPENAI_API_KEY", None)
        environment.pop("OPENAI_BASE_URL", None)
        result = subprocess.run(
            ["bash", str(ENTRY / "run.sh"), "metaweft_v1", "--dataset", "2h",
             "--split", "test-normal", "--limit", "2", "--cache-mode", "no_cache", "--dry-run"],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["settings"]["cache_mode"], "no_cache")
        self.assertEqual(plan["settings"]["retrieval_k"], 10)
        self.assertEqual(plan["selection"]["setting"], "metaweft_v1")
        self.assertEqual(before, sorted(str(p) for p in raw.glob("metaweft-*/experiment/output/*")))

    def test_entrypoint_descriptor_and_recovery_selection(self):
        required = {"variant_id", "parent_variant", "parent_source", "hypothesis", "intended_differences",
                    "memory_structure", "retrieval", "adaptation", "execution_context", "evidence_update",
                    "feedback_access", "memory_lifetime"}
        descriptor = read_json(ENTRY / "metaweft_v1.variant.json")
        self.assertTrue(required <= descriptor.keys())
        self.assertIn("metaweft_v1.variant.json", (ENTRY / "metaweft_v1.py").read_text())
        args = launch.parse_args("metaweft_v1", ["--dataset", "2h", "--recover"])
        self.assertEqual(args.recover, "latest")
        self.assertEqual(launch.selection("metaweft_v1", args)["setting"], "metaweft_v1")

    def test_prepare_captures_v1_descriptor_and_actual_source(self):
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {
                "OPENAI_API_KEY": "offline-key", "OPENAI_BASE_URL": "https://provider.invalid/v1"}):
            workspace = Path(temporary)
            with patch.object(launch, "WORKSPACE", workspace):
                plan, args = self.plan("--split", "test-normal")
            root = Path(plan["run_directory"])
            root.mkdir(parents=True)
            repo = workspace / "project"
            data = repo / "data"
            data.mkdir(parents=True)
            (data / "version.txt").write_text("offline")
            (data / "datasets").mkdir()
            dataset = plan["phases"][0]["dataset"]
            (data / "datasets" / f"{dataset}.txt").write_text("ff58e36_1\nff58e36_2\n")
            for task_id in plan["phases"][0]["ordered_task_ids"]:
                write_json(data / "tasks" / task_id / "task.json", {"task_id": task_id})
            args.env_file = workspace / "missing.env"

            def snapshot(target, _setting, _runtime):
                project = target / "source/project"
                project.mkdir(parents=True)
                write_json(project / "captured.json", {"offline": True})
                write_json(target / "source_provenance.json", {})
                return project, None

            with patch.object(launch, "REPO", repo), \
                 patch.object(launch, "find_data", return_value=data), \
                 patch.object(launch, "snapshot_source", side_effect=snapshot), \
                 patch.object(launch.subprocess, "run", return_value=SimpleNamespace(stdout="offline==1\n")):
                launch.prepare(plan, args, {"appworld_python": sys.executable, "es_binary": "/offline/es"})
            descriptor = read_json(root / "variant.json")
            self.assertEqual(descriptor["variant_id"], "metaweft-ace-adapt-reme-task-v1")
            self.assertEqual(descriptor, read_json(ENTRY / "metaweft_v1.variant.json"))
            self.assertTrue(plan["recovery_ready"])
            self.assertEqual(read_json(root / "resolved_settings.json")["recovery_version"], 2)
            captured = root / "source/MetaWeft/src/ace_adapt_reme_task/prompts/v1"
            self.assertTrue((captured / "lineage.json").is_file())
            provenance = read_json(root / "source_provenance.json")["metaweft_files"]
            self.assertIn("experiment/appworld/metaweft_v1.variant.json", provenance)
            self.assertIn("src/ace_adapt_reme_task/prompts/v1/adaptation.txt", provenance)
            self.assertTrue(any(name.startswith("src/ace_adapt_reme_task/prompts/v1/parents/")
                                for name in provenance))


if __name__ == "__main__":
    unittest.main()
