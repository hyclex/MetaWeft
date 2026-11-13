"""Offline failure/recovery through the real worker, runner, pipeline and logger.

Only the provider, AppWorld world/evaluator, and Elasticsearch transport are fake.
All run directories, captured sources and immutable input fixtures are temporary.
"""

from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from appworld_experiments.code.common import appworld_runner as runner
from appworld_experiments.code.common.episodes import EpisodeService
from appworld_experiments.code.common.evaluator import TaskEvaluator
from appworld_experiments.code.common.output_contract import TASK_DIRECTORIES, TASK_FILES
from appworld_experiments.code.common.task_loader import TaskRef

from MetaWeft.experiment import recovery
from MetaWeft.experiment.appworld import launch, worker
from MetaWeft.service.llm import LLMCacheError, LLMUnavailableError, ModelClient


ROOT = Path(__file__).resolve().parents[2]
TASKS = ["a_1", "b_1", "c_1"]


class MemoryStore:
    def __init__(self):
        self.records = {}
        self.index = "offline-memory"

    def create(self, mapping):
        self.mapping = deepcopy(mapping)

    def put(self, identifier, document):
        self.records[identifier] = deepcopy(document)
        return document

    def query(self, body):
        return {"hits": {"hits": [{"_id": key, "_source": value, "_score": 1.0}
                                   for key, value in self.records.items()][:body["size"]]}}

    def export(self):
        return [{"id": key, "document": deepcopy(value)} for key, value in sorted(self.records.items())]

    def close(self):
        pass


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.data = self.workspace / "data"
        recovery.write_json(self.data / "version.txt", "offline")
        for task in TASKS:
            recovery.write_json(self.data / "tasks" / task / "task.json", {"id": task})
        profile = self.workspace / "runtime"
        recovery.write_json(profile / "lock.json", {})
        recovery.write_json(profile / "assets.json", {})
        (profile / "installed").mkdir()
        (profile / "elasticsearch").touch()
        self.runtime = {
            "profile_directory": str(profile), "appworld_python": sys.executable,
            "installed_source": str(profile / "installed"), "data_root": str(self.data),
            "es_binary": str(profile / "elasticsearch"),
            "lock_sha256": {"lock.json": recovery.sha256(profile / "lock.json")},
            "asset_manifest": str(profile / "assets.json"),
            "asset_manifest_sha256": recovery.sha256(profile / "assets.json"),
        }
        recovery.write_json(profile / "runtime.json", self.runtime)

    def make_run(self, setting="direct", *, splits=("test-normal",), marker=False, tag="parent"):
        family = self.workspace / "raw_logs" / f"metaweft-{setting}-2h/experiment/output"
        root = family / tag
        root.mkdir(parents=True)
        shutil.copytree(ROOT / "MetaWeft", root / "source/MetaWeft",
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "tests"))
        recovery.write_json(root / "source/project/captured.json", {"original": True})
        (root / "source/project/experiments").mkdir()
        (root / "source/imports").mkdir()
        (root / "source/imports/appworld_experiments").symlink_to(root / "source/project/experiments")
        if marker:
            (root / "source/MetaWeft/experiment/appworld/worker.py").write_text(
                "import json\nfrom pathlib import Path\n"
                "from MetaWeft.service.llm import client\n"
                "Path('executed.json').write_text(json.dumps({'marker': 'captured', 'client': client.__file__}))\n")
        recovery.write_json(root / "source_provenance.json", {
            "metaweft_files": recovery.file_hashes(root / "source/MetaWeft", [root / "source/MetaWeft"]),
            "files": recovery.file_hashes(root / "source/project", [root / "source/project"]),
        })
        recovery.write_json(root / "data_provenance.json", {
            "source": str(self.data), "version_sha256": recovery.sha256(self.data / "version.txt"),
            "shared_files": {}, "task_files": recovery.file_hashes(self.data, [self.data / "tasks"]),
        })
        recovery.write_json(root / "runtime.json", self.runtime)
        recovery.write_json(root / "variant.json", {"setting": setting})
        (root / "appworld-requirements.txt").write_text("offline-test==1\n")
        phases = []
        for split in splits:
            dataset = "fake_" + split
            for prefix in ("inputs", "data"):
                path = root / prefix / "datasets" / (dataset + ".txt")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(TASKS) + "\n")
            phases.append({
                "split": split, "dataset": dataset, "experiment_name": split, "status": "planned",
                "ordered_task_ids": list(TASKS), "task_ids": list(TASKS), "partial": False,
                "source_task_count": len(TASKS), "output_directory": str(root / "experiments/outputs" / split),
                "dataset_sha256": recovery.sha256(root / "inputs/datasets" / (dataset + ".txt")),
                "selected_dataset_sha256": recovery.sha256(root / "data/datasets" / (dataset + ".txt")),
            })
        split = splits[0] if len(splits) == 1 else "test-all"
        plan = {
            "baseline": "metaweft", "setting": setting, "setting_id": f"metaweft-{setting}-2h",
            "variant_id": f"metaweft-{setting}-v1", "dataset_version": "2h", "split": split,
            "scenario": "direct" if setting == "direct" else "cold-start", "note": "offline",
            "run_directory": str(root), "run_id": f"{setting}-{tag}", "run_tag": tag,
            "created_at": "2026-01-01T00:00:00+00:00", "status": "failed", "phases": phases,
            "source_framework": str(root / "source/MetaWeft"), "source_project": str(root / "source/project"),
            "provider": {"chat_base_url": "https://provider.invalid/v1"}, "runtime": self.runtime,
            "data_source": str(self.data), "es_url": None, "es_binary": self.runtime["es_binary"],
            "settings": {**launch.SETTINGS, "max_retries": 1, "retry_delay_seconds": 0,
                         "cache_mode": "no_cache", "cache_path": str(self.workspace / "llm.sqlite3"),
                         "cache_ttl_seconds": 1800,
                         "embedding_dimensions": 2, "max_interactions": 1},
            "recovery_version": recovery.RECOVERY_VERSION,
            "selection": {"setting": setting, "dataset": "2h", "split": split, "limit": None, "task_id": None},
        }
        if setting not in ("base", "direct"):
            plan["settings"]["retrieval_meta"] = launch.load_preset(setting)
            plan["note"] = launch.retrieval_note(plan["settings"]) + "; offline"
        recovery.seal_run(plan)
        recovery.write_json(root / "run_manifest.json", plan)
        recovery.write_json(root / "resolved_settings.json", plan)
        return plan

    def run_worker(self, plan, index=0, *, fail_task=None, fail_phase="execution", fail_evaluation=False,
                   returned_reasoning=""):
        phase = plan["phases"][index]
        output = Path(phase["output_directory"])
        refs = tuple(TaskRef(task, number, 0) for number, task in enumerate(TASKS))
        calls, opened, stores = [], [], []
        task = SimpleNamespace(instruction="Complete the offline task", app_descriptions={"example": "Example app"},
                               supervisor=SimpleNamespace(first_name="Test", last_name="User", email="test@example.org",
                                                          phone_number="555-0100"))

        def world_factory(task_id, **kwargs):
            opened.append(task_id)
            primary = output / "tasks" / task_id
            if primary.exists():
                shutil.rmtree(primary)
            for directory in TASK_DIRECTORIES:
                (primary / directory).mkdir(parents=True)
            for name in TASK_FILES:
                if not name.startswith("evaluation/"):
                    (primary / name).touch()
            recovery.write_json(primary / "dbs/state.json", {"task": task_id})
            return SimpleNamespace(execute=lambda code: "ok", task_completed=lambda: True, close=lambda: None)

        def evaluate_task(task_id, **kwargs):
            directory = output / "tasks" / task_id / "evaluation"
            (directory / "report.md").write_text("Evaluation-only assertion")
            (directory / "version.txt").write_text("offline")
            return SimpleNamespace(pass_count=1, fail_count=0), "Evaluation-only assertion"

        def evaluate_dataset(**kwargs):
            if fail_evaluation:
                raise RuntimeError("aggregate evaluation interrupted")
            for task_id in TASKS:
                (output / "tasks" / task_id / "evaluation/report.md").write_text("Regenerated official report")
            recovery.write_json(output / "evaluations" / (phase["dataset"] + ".json"),
                                {"aggregate": {"task_goal_completion": 100}, "individual": {task: {} for task in TASKS}})
            (output / "evaluations" / (phase["dataset"] + ".txt")).write_text("offline evaluation")

        def make_store(*args):
            store = MemoryStore()
            stores.append(store)
            return store

        def make_model(**kwargs):
            task_id = recovery.read_json(output / "run_manifest.json")["current_task_id"]
            original_recorder = kwargs.pop("recorder")
            active = {}
            successful_decisions = 0

            def record(operation, request, invoke, **context):
                active.update(context)
                calls.append((task_id, context["phase"]))
                return original_recorder(operation, request, invoke, **context)

            def chat(**request):
                nonlocal successful_decisions
                stage = active["phase"]
                if task_id == fail_task and stage == fail_phase and (stage != "consolidation" or successful_decisions):
                    raise TimeoutError("injected provider outage")
                if stage == "execution":
                    content = "```python\nprint('ok')\n```"
                elif stage == "adaptation":
                    content = json.dumps({"guidance": "Use the previous procedure"})
                elif stage == "reflection":
                    candidate = {"when_to_use": "For the example", "content": "Use observed procedure"}
                    content = json.dumps({"candidates": [candidate] * (2 if task_id == "b_1" else 1)})
                elif stage == "retrieval":
                    payload = json.loads(request["messages"][1]["content"])
                    count = len(payload["memories"])
                    content = json.dumps({"scores": [[i, 100] for i in range(1, count + 1)]})
                else:
                    successful_decisions += 1
                    content = json.dumps({"operation": "add", "target_id": None, "when_to_use": None,
                                          "content": None, "reason": "Supported by observed evidence"})
                raw = {"choices": [{"message": {"content": content, "reasoning_content": returned_reasoning}}],
                       "usage": {"total_tokens": 3}}
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                                       model_dump=lambda: raw)

            def embed(**request):
                if task_id == fail_task and active["phase"] == fail_phase == "retrieval":
                    raise TimeoutError("injected provider outage")
                raw = {"data": [{"index": 0, "embedding": [0.1, 0.2]}], "usage": {"total_tokens": 2}}
                return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])], model_dump=lambda: raw)

            provider = Mock()
            provider.chat.completions.create.side_effect = chat
            provider.embeddings.create.side_effect = embed
            return ModelClient(client=provider, recorder=record, sleep=Mock(), **kwargs)

        root = Path(plan["run_directory"])
        recovery.write_json(root / "run_manifest.json", plan)
        error = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"OPENAI_API_KEY": "fake"}))
            stack.enter_context(patch.object(worker.sys, "argv", ["worker", str(root / "run_manifest.json"), str(index)]))
            stack.enter_context(patch.object(worker.signal, "signal"))
            stack.enter_context(patch.object(worker, "load_task_stream", return_value=refs))
            stack.enter_context(patch.object(runner, "load_task_stream", return_value=refs))
            stack.enter_context(patch.object(runner, "path_store", SimpleNamespace(experiment_outputs=str(output.parent))))
            stack.enter_context(patch.object(runner.Task, "load", return_value=task))
            stack.enter_context(patch.object(runner, "EpisodeService", side_effect=lambda **kw: EpisodeService(
                world_factory=world_factory, **kw)))
            stack.enter_context(patch.object(runner, "TaskEvaluator", side_effect=lambda **kw: TaskEvaluator(
                evaluate_task_fn=evaluate_task, **kw)))
            stack.enter_context(patch.object(runner, "evaluate_dataset", side_effect=evaluate_dataset))
            stack.enter_context(patch.object(worker, "ModelClient", side_effect=make_model))
            stack.enter_context(patch.object(worker.CachedDBHandler, "reset"))
            stack.enter_context(patch("reme_runtime.elasticsearch", return_value=nullcontext("http://fake-store")))
            store_factory = stack.enter_context(patch("MetaWeft.service.elasticsearch.ElasticsearchStore", side_effect=make_store))
            try:
                worker.main()
            except (LLMUnavailableError, RuntimeError) as caught:
                error = caught
            if plan["setting"] == "direct":
                store_factory.assert_not_called()
        phase.update(recovery.read_json(output / "run_manifest.json"))
        plan["status"] = "failed" if error else "completed" if all(p["status"] == "completed" for p in plan["phases"]) else "running"
        recovery.write_json(root / "run_manifest.json", plan)
        return error, calls, opened, stores

    def child(self, parent, tag="child"):
        parent_root = Path(parent["run_directory"])
        checkpoints = recovery.validate_parent(parent, parent_root)
        root = parent_root.parent / tag
        root.mkdir()
        child = recovery.recovery_plan(parent, parent_root, checkpoints, root=root, tag=tag, command=["offline"])
        recovery.clone_run(parent_root, child, checkpoints)
        recovery.write_json(root / "run_manifest.json", child)
        return child

    def test_settings_resume_prefix_and_restore_only_committed_memory(self):
        for setting in ("direct", "base", "transferability_3", "efficiency_1"):
            with self.subTest(setting=setting):
                parent = self.make_run(setting)
                error, calls, opened, stores = self.run_worker(
                    parent, fail_task="b_1", fail_phase="execution" if setting == "direct" else "consolidation")
                self.assertIsInstance(error, LLMUnavailableError)
                self.assertEqual(opened, ["a_1", "b_1"])
                self.assertNotIn("c_1", [task for task, stage in calls])
                self.assertEqual(parent["phases"][0]["completed_task_ids"], ["a_1"])
                parent_root = Path(parent["run_directory"])
                before = recovery.file_hashes(parent_root, [parent_root])
                parent_cost = recovery.read_json(Path(parent["phases"][0]["output_directory"]) / "accounting.json")
                child = self.child(parent)
                error, calls, opened, restored = self.run_worker(child)
                self.assertIsNone(error)
                self.assertEqual(opened, ["b_1", "c_1"])
                self.assertNotIn("a_1", [task for task, stage in calls])
                self.assertEqual(child["phases"][0]["completed_task_ids"], TASKS)
                self.assertEqual(before, recovery.file_hashes(parent_root, [parent_root]))
                output = Path(child["phases"][0]["output_directory"])
                cost = recovery.read_json(output / "accounting.json")
                self.assertEqual(cost["inherited"]["recorded_calls"], parent_cost["recorded_calls"])
                self.assertEqual(cost["recorded_calls"], parent_cost["recorded_calls"] + cost["current_attempt"]["recorded_calls"])
                if setting != "direct":
                    self.assertEqual(len(stores[0].records), 2)  # A and the unfinished B update.
                    original = recovery.read_json(Path(parent["phases"][0]["output_directory"]) /
                                                  "baseline_artifacts/metaweft/a_1/memory_after.json")
                    self.assertEqual(recovery.read_json(output / "baseline_artifacts/metaweft/memory_restored.json"), original)
                    partial_ids = set(stores[0].records) - {item["id"] for item in original["records"]}
                    self.assertFalse(partial_ids & set(restored[0].records))
                    self.assertEqual(child["phases"][0]["degraded_task_ids"], [])
                    for record in restored[0].records.values():
                        for reference in record["source_refs"]:
                            self.assertTrue((Path(child["run_directory"]) / reference).is_file())

    def test_final_thinking_logging_for_completed_and_failed_workers(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                plan = self.make_run(tag="thinking-failed" if failed else "thinking-completed")
                error, _, _, _ = self.run_worker(plan, fail_task="b_1" if failed else None,
                                                returned_reasoning="Provider ignored no-thinking")
                self.assertEqual(error is not None, failed)
                output = Path(plan["phases"][0]["output_directory"])
                thinking = recovery.read_json(output / "thinking_summary.json")
                logged = recovery.read_json(output / "accounting.json")
                self.assertEqual(thinking["total_llm_calls"], 3)
                self.assertEqual(thinking["no_thinking_llm_calls"], 3)
                self.assertEqual(thinking["thinking_despite_no_thinking_llm_calls"], 1 if failed else 3)
                for key in ("total_llm_calls", "no_thinking_llm_calls", "thinking_despite_no_thinking_llm_calls"):
                    self.assertEqual(logged[key], thinking[key])
                self.assertEqual(logged["thinking_summary_path"], "thinking_summary.json")
                self.assertEqual(logged["thinking_summary_status"], thinking["status"])
                self.assertEqual(recovery.read_json(output / "run_manifest.json")["status"],
                                 "failed" if failed else "completed")

    def test_preset_recovery_selects_only_its_family_and_keeps_saved_controls_and_notes(self):
        parent = self.make_run("transferability_3")
        other = self.make_run("efficiency_1")
        self.run_worker(parent, fail_task="b_1", fail_phase="retrieval")
        self.assertEqual(parent["phases"][0]["failure"]["attempts"], 3)
        root = Path(parent["run_directory"])
        with self.assertRaisesRegex(ValueError, "selection"):
            recovery.select_parent(root.parent, parent["selection"], other["run_directory"])
        with self.assertRaisesRegex(ValueError, "No compatible"):
            recovery.select_parent(root.parent, other["selection"], "latest")
        args = launch.parse_args("retrieval", ["--preset", "transferability_3", "--dataset", "2h",
                                               "--split", "test-normal", "--recover", "--note", "new annotation",
                                               "--env-file", "/nonexistent"])
        before = recovery.file_hashes(self.workspace, [self.workspace])
        with patch.object(launch, "WORKSPACE", self.workspace), patch.dict(os.environ, {}, clear=True), \
             patch.object(launch, "load_preset", side_effect=AssertionError("must use saved preset")):
            plan, selected_root, checkpoints = launch.inspect_recovery(args.preset, args)
        self.assertEqual(selected_root, root)
        self.assertEqual(plan["settings"], parent["settings"])
        self.assertEqual(checkpoints[0]["completed_task_ids"], ["a_1"])
        self.assertEqual(plan["note"], launch.retrieval_note(parent["settings"]) + "; new annotation")
        self.assertEqual(before, recovery.file_hashes(self.workspace, [self.workspace]))

    def test_explicit_source_repair_preserves_parent_prefix_and_future_recovery(self):
        from tool import recover_metaweft_consolidation_limit as repair

        parent = self.make_run("efficiency_1", splits=("test-normal", "test-challenge"))
        self.assertIsNone(self.run_worker(parent, index=0)[0])
        self.run_worker(parent, index=1, fail_task="b_1", fail_phase="consolidation")
        parent_root = Path(parent["run_directory"])
        before = recovery.file_hashes(parent_root, [parent_root])
        child = self.child(parent)
        child_root = Path(child["run_directory"])
        pipeline = child_root / "source/MetaWeft" / repair.PIPELINE
        replacement = self.workspace / "repaired_pipeline.py"
        replacement.write_bytes(pipeline.read_bytes() + b"\n# Offline source-repair fixture.\n")
        checkpoints = [recovery.read_json(Path(p["output_directory"]) / "checkpoint.json")
                       for p in child["phases"]]

        with patch.object(repair, "ORIGINAL_PIPELINE_SHA256", recovery.sha256(pipeline)), \
                patch.object(repair, "REPAIRED_PIPELINE_SHA256", recovery.sha256(replacement)), \
                patch.object(repair, "REPAIRED_SOURCE", replacement):
            repair.apply_repair(child)

        self.assertNotEqual(child["recovery_identity"], parent["recovery_identity"])
        self.assertEqual(child["settings"], parent["settings"])
        self.assertEqual(pipeline.read_bytes(), replacement.read_bytes())
        self.assertIn("mixed protocol", child["note"])
        self.assertEqual(child["source_repair"]["boundaries"][1]["first_repaired_task_id"], "b_1")
        for phase, original in zip(child["phases"], checkpoints):
            checkpoint = recovery.load_checkpoint(child, phase, child_root)
            self.assertEqual(checkpoint["files"], original["files"])
            self.assertEqual(checkpoint["completed_task_ids"], original["completed_task_ids"])
            self.assertEqual(checkpoint["inherited_identity_before_source_repair"], parent["recovery_identity"])

        error, _, opened, _ = self.run_worker(child, index=1, fail_task="c_1")
        self.assertIsInstance(error, LLMUnavailableError)
        self.assertEqual(opened, ["b_1", "c_1"])
        grandchild = self.child(child, tag="grandchild")
        self.assertEqual(grandchild["phases"][1]["completed_task_ids"], ["a_1", "b_1"])
        self.assertEqual(grandchild["recovery_identity"], child["recovery_identity"])
        self.assertEqual((Path(grandchild["run_directory"]) / "source/MetaWeft" / repair.PIPELINE).read_bytes(),
                         replacement.read_bytes())
        self.assertEqual(before, recovery.file_hashes(parent_root, [parent_root]))

    def test_repeated_recovery_keeps_all_attempt_costs_and_latest_prefix(self):
        parent = self.make_run("base")
        self.run_worker(parent, fail_task="b_1")
        child = self.child(parent)
        self.run_worker(child, fail_task="c_1")
        root, selected, checkpoints = recovery.select_parent(Path(parent["run_directory"]).parent, parent["selection"], "latest")
        self.assertEqual(root, Path(child["run_directory"]))
        self.assertEqual(checkpoints[0]["completed_task_ids"], ["a_1", "b_1"])
        grandchild = self.child(child, "grandchild")
        error, calls, opened, _ = self.run_worker(grandchild)
        self.assertIsNone(error)
        self.assertEqual(opened, ["c_1"])
        self.assertEqual(grandchild["recovery"]["generation"], 2)
        cost = recovery.read_json(Path(grandchild["phases"][0]["output_directory"]) / "accounting.json")
        prior_cost = recovery.read_json(Path(child["phases"][0]["output_directory"]) / "accounting.json")
        self.assertEqual(cost["inherited"]["recorded_calls"], prior_cost["recorded_calls"])
        with self.assertRaisesRegex(ValueError, "active or completed"):
            recovery.select_parent(root.parent, parent["selection"], "latest")

    def test_retrieval_adaptation_and_reflection_outages_stop_before_the_next_task(self):
        for phase in ("retrieval", "adaptation", "reflection"):
            with self.subTest(phase=phase):
                parent = self.make_run("base", tag=phase)
                error, calls, opened, _ = self.run_worker(parent, fail_task="b_1", fail_phase=phase)
                self.assertIsInstance(error, LLMUnavailableError)
                self.assertEqual(parent["phases"][0]["completed_task_ids"], ["a_1"])
                self.assertNotIn("c_1", [task for task, stage in calls])
                self.assertEqual(parent["phases"][0]["failure"]["phase"], phase)
                if phase in ("retrieval", "adaptation"):
                    self.assertEqual(opened, ["a_1"])

    def test_recovery_selection_lock_excludes_another_process(self):
        directory = self.workspace / "output"
        code = (
            "import fcntl, sys\n"
            "with open(sys.argv[1], 'a') as stream:\n"
            "    try:\n"
            "        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "    except BlockingIOError:\n"
            "        sys.exit(23)\n"
        )
        with recovery.recovery_lock(directory):
            result = subprocess.run([sys.executable, "-B", "-c", code, str(directory / ".recovery.lock")])
        self.assertEqual(result.returncode, 23)

    def test_latest_can_select_an_independent_failure_while_another_lineage_runs(self):
        parent = self.make_run(tag="first")
        self.child(parent)
        independent = self.make_run(tag="independent")
        independent["created_at"] = "2025-01-01T00:00:00+00:00"
        recovery.write_json(Path(independent["run_directory"]) / "run_manifest.json", independent)
        root, selected, _ = recovery.select_parent(Path(parent["run_directory"]).parent, parent["selection"], "latest")
        self.assertEqual(selected["run_id"], independent["run_id"])

    def test_failure_before_first_task_and_completed_split_reuse_through_launcher(self):
        parent = self.make_run("base", splits=("test-normal", "test-challenge"))
        self.assertIsNone(self.run_worker(parent)[0])
        self.assertIsInstance(self.run_worker(parent, 1, fail_task="a_1")[0], LLMUnavailableError)
        self.assertEqual(parent["phases"][1]["completed_task_ids"], [])
        invoked, child_plans = [], []

        def prepare(plan, source, checkpoints, args):
            recovery.clone_run(source, plan, checkpoints)
            child_plans.append(plan)

        def run_phases(plan):
            for index, phase in enumerate(plan["phases"]):
                if phase["status"] != "completed":
                    invoked.append(index)
                    self.assertIsNone(self.run_worker(plan, index)[0])

        with patch.object(launch, "WORKSPACE", self.workspace), \
             patch.object(launch, "prepare_recovery", side_effect=prepare), \
             patch.object(launch, "run_phases", side_effect=run_phases), \
             patch.object(launch.signal, "signal"), patch.dict(os.environ, {}, clear=True):
            launch.main("base", ["--dataset", "2h", "--recover", "--env-file", "/nonexistent"])
        self.assertEqual(invoked, [1])
        child = child_plans[0]
        self.assertEqual(child["status"], "completed")
        cost = recovery.read_json(Path(child["phases"][0]["output_directory"]) / "accounting.json")
        self.assertEqual(cost["current_attempt"]["recorded_calls"], 0)
        parent_thinking = recovery.read_json(Path(parent["phases"][0]["output_directory"]) / "thinking_summary.json")
        child_thinking = recovery.read_json(Path(child["phases"][0]["output_directory"]) / "thinking_summary.json")
        self.assertGreater(child_thinking["total_llm_calls"], 0)
        self.assertEqual(child_thinking["total_llm_calls"], parent_thinking["total_llm_calls"])
        self.assertEqual(cost["total_llm_calls"], child_thinking["total_llm_calls"])
        initial = recovery.read_json(Path(child["phases"][1]["output_directory"]) /
                                     "baseline_artifacts/metaweft/memory_restored.json")
        self.assertEqual(initial["records"], [])

    def test_finalization_only_recovery_makes_no_model_calls(self):
        parent = self.make_run()
        self.assertIsInstance(self.run_worker(parent, fail_evaluation=True)[0], RuntimeError)
        self.assertEqual(parent["phases"][0]["completed_task_ids"], TASKS)
        child = self.child(parent)
        error, calls, opened, _ = self.run_worker(child)
        self.assertIsNone(error)
        self.assertEqual((calls, opened), ([], []))
        self.assertEqual(child["status"], "completed")

    def test_pending_cache_restore_failure_stops_before_model_calls_and_preserves_checkpoint(self):
        parent = self.make_run()
        self.run_worker(parent, fail_task="b_1")
        child = self.child(parent)
        failure = LLMCacheError(operation="write", phase="recovery", key="pending", attempts=2,
                                cause=OSError("cache unavailable"))
        with patch.object(worker, "restore_pending_responses", side_effect=failure) as restore:
            error, calls, opened, _ = self.run_worker(child)
        self.assertIs(error, failure)
        self.assertEqual((calls, opened), ([], []))
        output = Path(child["phases"][0]["output_directory"])
        restore.assert_called_once_with(output, cache_path=child["settings"]["cache_path"],
                                        max_retries=1, retry_delay_seconds=0)
        self.assertEqual(recovery.read_json(output / "checkpoint.json")["completed_task_ids"], ["a_1"])
        recorded = recovery.read_json(output / "failure.json")
        self.assertEqual(recorded["kind"], "cache_unavailable")
        self.assertEqual(recorded["cache_key"], "pending")

    def test_recovery_rejects_cache_overrides_and_preserves_saved_policy(self):
        parent = self.make_run()
        for option, value in (("--cache-mode", "timed_cache"), ("--cache-path", str(self.workspace / "other.sqlite3"))):
            args = launch.parse_args("direct", ["--dataset", "2h", option, value])
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "conflicts with the saved run"):
                launch.check_recovery_options(parent, args)
        child = self.child(parent)
        self.assertEqual(child["settings"], parent["settings"])

    def test_checkpoint_remains_valid_when_manifest_write_was_interrupted(self):
        parent = self.make_run()
        self.run_worker(parent, fail_task="b_1")
        phase = parent["phases"][0]
        manifest = recovery.read_json(Path(phase["output_directory"]) / "run_manifest.json")
        manifest["completed_task_ids"] = []
        recovery.write_json(Path(phase["output_directory"]) / "run_manifest.json", manifest)
        phase["completed_task_ids"] = []
        child = self.child(parent)
        self.assertEqual(child["phases"][0]["completed_task_ids"], ["a_1"])
        path = Path(phase["output_directory"]) / "checkpoint.json"
        before = path.read_bytes()
        with patch.object(recovery.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                recovery.write_json(path, {"partial": "never published"})
        self.assertEqual(path.read_bytes(), before)

    def test_recovery_rejects_changed_configuration_artifacts_and_task_order(self):
        parent = self.make_run()
        self.run_worker(parent, fail_task="b_1")
        root = Path(parent["run_directory"])
        changed = deepcopy(parent)
        changed["settings"]["max_retries"] += 1
        with self.assertRaisesRegex(ValueError, "configuration"):
            recovery.validate_parent(changed, root)
        extra_source = root / "source/MetaWeft/unrecorded.py"
        extra_source.touch()
        with self.assertRaisesRegex(ValueError, "inventory"):
            recovery.validate_parent(parent, root)
        extra_source.unlink()
        empty_directory = Path(parent["phases"][0]["output_directory"]) / "tasks/a_1/checkpoints"
        empty_directory.rmdir()
        with self.assertRaisesRegex(ValueError, "Missing committed task directory"):
            recovery.validate_parent(parent, root)
        empty_directory.mkdir()
        for path in (root / "source/MetaWeft/service/llm/retry.json",
                     Path(parent["phases"][0]["output_directory"]) / "tasks/a_1/logs/lm_calls.jsonl",
                     self.data / "tasks/a_1/task.json", Path(self.runtime["profile_directory"]) / "lock.json"):
            with self.subTest(path=path):
                previous = path.read_bytes()
                path.write_text("changed")
                with self.assertRaises(ValueError):
                    recovery.validate_parent(parent, root)
                path.write_bytes(previous)
        path = Path(parent["phases"][0]["output_directory"]) / "checkpoint.json"
        checkpoint = recovery.read_json(path)
        checkpoint["completed_task_ids"] = ["b_1"]
        recovery.write_json(path, checkpoint)
        with self.assertRaisesRegex(ValueError, "task order"):
            recovery.validate_parent(parent, root)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing checkpoint"):
            recovery.validate_parent(parent, root)

    def test_legacy_conflicting_selection_endpoint_and_active_lineage_are_rejected(self):
        parent = self.make_run()
        root = Path(parent["run_directory"])
        args = launch.parse_args("direct", ["--dataset", "2h", "--split", "test-normal", "--env-file", "/nonexistent"])
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://different.invalid"}):
            with self.assertRaisesRegex(ValueError, "endpoint"):
                launch.check_recovery_options(parent, args)
        with self.assertRaisesRegex(ValueError, "selection"):
            recovery.select_parent(root.parent, {**parent["selection"], "limit": 1}, str(root))
        legacy = {**parent, "recovery_version": 0}
        recovery.write_json(root / "run_manifest.json", legacy)
        with self.assertRaisesRegex(ValueError, "predates"):
            recovery.select_parent(root.parent, parent["selection"], str(root))
        recovery.write_json(root / "run_manifest.json", parent)
        child = self.child(parent)
        with self.assertRaisesRegex(ValueError, "active"):
            recovery.select_parent(root.parent, parent["selection"], "latest")

    def test_dry_run_uses_saved_settings_and_writes_nothing(self):
        parent = self.make_run()
        self.run_worker(parent, fail_task="b_1")
        before = recovery.file_hashes(self.workspace, [self.workspace])
        stream = io.StringIO()
        with patch.object(launch, "WORKSPACE", self.workspace), \
             patch.object(launch, "load_runtime", side_effect=AssertionError("must use saved runtime")), \
             patch.object(launch, "run_phases", side_effect=AssertionError("dry run")), \
             patch.dict(os.environ, {}, clear=True), redirect_stdout(stream):
            launch.main("direct", ["--dataset", "2h", "--split", "test-normal", "--recover", "--dry-run",
                                   "--env-file", "/nonexistent"])
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan["settings"], parent["settings"])
        self.assertEqual(plan["phases"][0]["remaining_task_ids"], ["b_1", "c_1"])
        self.assertEqual(before, recovery.file_hashes(self.workspace, [self.workspace]))
        self.assertFalse((Path(parent["run_directory"]).parent / ".recovery.lock").exists())

    def test_worker_subprocess_imports_captured_source_and_rebased_links(self):
        parent = self.make_run(marker=True)
        child = self.child(parent)
        root = Path(child["run_directory"])
        # A different current checkout is irrelevant to the worker's import path.
        with patch.object(launch, "FRAMEWORK", self.workspace / "different-checkout"), patch.dict(os.environ):
            launch.worker_environment(child)
            launch.run_phases(child)
        executed = recovery.read_json(root / "executed.json")
        self.assertEqual(executed["marker"], "captured")
        self.assertTrue(Path(executed["client"]).is_relative_to(root / "source/MetaWeft"))
        self.assertEqual((root / "source/imports/appworld_experiments").resolve(), root / "source/project/experiments")


if __name__ == "__main__":
    unittest.main()
