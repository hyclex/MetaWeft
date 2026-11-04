"""Offline integration checks using fake worlds and provider transports."""

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import RateLimitError

from appworld_experiments.code.common.episodes import EpisodeService
from appworld_experiments.code.common.lm_logger import task_lm_logger
from MetaWeft.experiment.appworld import executor, worker
from MetaWeft.experiment.recovery import commit_checkpoint, write_json
from MetaWeft.service.llm import LLMUnavailableError, ModelClient
from MetaWeft.src.memory_retrieval import load_preset
from MetaWeft.src.base import BasePipeline


ROOT = Path(__file__).resolve().parents[2]
TASK_SPEC = {
    "task_id": "fake_1", "instruction": "Complete the example task",
    "supervisor": {"first_name": "Test", "last_name": "User", "email": "test@example.com",
                   "phone_number": "555-0100"},
    "app_descriptions": {"example": "Example application"},
}


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeModel:
    def __init__(self, events=None):
        self.calls = []
        self.events = events if events is not None else []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), **kwargs})
        return response("```python\nprint('observed')\n```")

    def embed(self, *args, **kwargs):
        raise AssertionError("Direct execution must not call embeddings")

    def close(self):
        self.events.append("model_close")


class ExecutorTests(unittest.TestCase):
    def test_parser_matches_parent_complete_partial_multiple_and_raw(self):
        source = ROOT / "metaskill-appworld/baselines/reme2f37a15/benchmark/appworld/appworld_react_agent.py"
        tree = ast.parse(source.read_text())
        parent = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "extract_code_and_fix_content")
        namespace = {"re": re}
        exec(compile(ast.Module(body=[parent], type_ignores=[]), str(source), "exec"), namespace)
        cases = [
            "```python\nprint(1)\n```", "before\n```python\nprint(1)",
            "```python\nprint(1)\n```\n```python\nprint(2)\n```",
            "print(3)", "```python\n", "```python\n```", "```text\nraw\n```",
        ]
        for text in cases:
            with self.subTest(text=text):
                original, _ = namespace["extract_code_and_fix_content"](None, text)
                self.assertEqual(executor.extract_code(text), original)

    def test_guidance_is_added_once_and_stays_fixed_during_execution(self):
        model = FakeModel()
        environment = Mock()
        environment.execute.return_value = "observed\n"
        environment.task_completed.side_effect = [False, True]
        trace = executor.execute(
            TASK_SPEC, "UNIQUE-GUIDANCE", llm=model, environment=environment,
            prompt="Task instructions for {{ supervisor.first_name }}", max_interactions=4,
        )
        self.assertEqual(len(trace), 2)
        for call in model.calls:
            self.assertEqual(json.dumps(call["messages"]).count("UNIQUE-GUIDANCE"), 1)
            self.assertEqual(call["messages"][0]["role"], "system")
        self.assertEqual(model.calls[0]["messages"][:2], model.calls[1]["messages"][:2])
        self.assertEqual(model.calls[1]["messages"][-1]["content"], "Output:\n```\nobserved\n```\n\n")
        self.assertEqual(environment.execute.call_count, 2)


class CanonicalRecorderTests(unittest.TestCase):
    def test_retries_have_distinct_physical_ids_with_usage_and_accounting(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs/split"
            logger = task_lm_logger(output.parent, "split", "metaweft", "fake_1")
            provider = Mock()
            error = RateLimitError("retry", response=httpx.Response(
                429, request=httpx.Request("POST", "https://provider.invalid/v1")), body={})
            completion = {"choices": [{"message": {"content": "ok"}}],
                          "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}
            value = response("ok")
            value.model_dump = lambda: completion
            provider.chat.completions.create.side_effect = [error, value]
            model = ModelClient(client=provider, chat_model="test", recorder=worker.physical_recorder(logger),
                                sleep=Mock())
            self.assertIs(model.chat([], phase="adaptation", component="base"), value)
            rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["id"] for row in rows}), 2)
            self.assertEqual(len({row["metadata"]["logical_call_id"] for row in rows}), 1)
            self.assertEqual([row["metadata"]["status"] for row in rows], ["error", "success"])
            self.assertEqual([row["metadata"]["attempt"] for row in rows], [1, 2])
            self.assertEqual(rows[1]["output"]["usage"], completion["usage"])
            accounting = worker.accounting(output)
            self.assertEqual(accounting["recorded_calls_by_phase"], {"adaptation": 2})
            self.assertEqual(accounting["reported_tokens"]["total_tokens"], 7)
            self.assertEqual(accounting["missing_usage_calls"], 1)
            self.assertIsNone(accounting["cost_usd"])

    def test_embedding_recorder_keeps_usage_and_hashes_without_full_vectors(self):
        with TemporaryDirectory() as temporary:
            logger = task_lm_logger(Path(temporary) / "outputs", "split", "metaweft", "fake_1")
            raw = {"model": "embed", "usage": {"prompt_tokens": 2, "total_tokens": 2},
                   "data": [{"index": 0, "embedding": [0.1, 0.2]}]}
            value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
                                    model_dump=lambda: raw)
            provider = Mock()
            provider.embeddings.create.return_value = value
            model = ModelClient(client=provider, chat_model="test", dimensions=2,
                                recorder=worker.physical_recorder(logger))
            self.assertEqual(model.embed(["text"], phase="retrieval", component="embedding"), [[0.1, 0.2]])
            row = json.loads(logger.path.read_text())
            self.assertEqual(row["output"]["usage"], raw["usage"])
            self.assertEqual(row["output"]["vector_count"], 1)
            self.assertEqual(len(row["output"]["vector_hashes"]), 1)
            self.assertNotIn("data", row["output"])

    def test_retrieval_trace_links_embedding_retries_to_canonical_calls(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = task_lm_logger(root / "outputs", "split", "metaweft", "fake_1")
            refs = {}
            vector = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
                                     model_dump=lambda: {"data": [{"index": 0, "embedding": [0.1, 0.2]}],
                                                         "usage": {"total_tokens": 2}})
            adapted = response('{"guidance": "Use the memory"}')
            adapted.model_dump = lambda: {"choices": [{"message": {"content": adapted.choices[0].message.content}}]}
            provider = Mock()
            provider.embeddings.create.side_effect = [TimeoutError("retry embedding"), vector]
            provider.chat.completions.create.return_value = adapted
            model = ModelClient(client=provider, chat_model="test", dimensions=2, sleep=Mock(),
                                recorder=worker.physical_recorder(logger, refs))
            store = Mock()
            store.query.return_value = {"hits": {"hits": [{"_id": "memory", "_score": 1.0,
                "_source": {"id": "memory", "when_to_use": "For the example", "content": "Follow the example",
                            "source_refs": [], "embedding": [0.1, 0.2]}}]}}
            artifacts = root / "artifacts"
            pipeline = BasePipeline(store, root)
            result = pipeline.prepare_context(TASK_SPEC, llm=model, artifacts=artifacts,
                                               trace_context={"bank": {"index": "offline"},
                                                              "model_settings": {"chat_model": "test"}, "prompts": {}},
                                               call_refs=refs)
            self.assertEqual(result.guidance, "Use the memory")
            trace = json.loads((artifacts / "retrieval.json").read_text())["trace"]
            rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
            embedding_rows = [row for row in rows if row["metadata"]["phase"] == "retrieval"]
            calls = trace["searches"][0]["embedding_calls"]
            self.assertEqual([call["physical_call_id"] for call in calls], [row["id"] for row in embedding_rows])
            self.assertEqual([call["attempt"] for call in calls], [1, 2])
            self.assertEqual(len({call["logical_call_id"] for call in calls}), 1)
            self.assertEqual(trace["bank"], {"index": "offline"})
            self.assertEqual(trace["model_settings"]["chat_model"], "test")


class EpisodeTests(unittest.TestCase):
    def test_episode_reset_preserves_pre_execution_llm_log_and_disables_gt(self):
        with TemporaryDirectory() as temporary:
            primary = Path(temporary) / "outputs/split/tasks/fake_1"
            canonical = primary / "logs/lm_calls.jsonl"
            canonical.parent.mkdir(parents=True)
            previous = b'{"id":"adaptation-before-world"}\n'
            canonical.write_bytes(previous)
            world = Mock()

            def factory(**kwargs):
                self.assertFalse(kwargs["load_ground_truth"])
                shutil.rmtree(primary)
                return world

            episodes = EpisodeService(
                task_id="fake_1", experiment_name="split", primary_task_dir=primary,
                appworld_kwargs={"load_ground_truth": False}, world_factory=factory,
            )
            with episodes.open(kind="primary"):
                self.assertEqual(canonical.read_bytes(), previous)
                self.assertTrue(episodes.is_open)
            world.close.assert_called_once()
            self.assertFalse(episodes.is_open)
            self.assertEqual(canonical.read_bytes(), previous)

    def test_initialization_failure_also_preserves_pre_execution_log(self):
        with TemporaryDirectory() as temporary:
            primary = Path(temporary) / "tasks/fake_1"
            canonical = primary / "logs/lm_calls.jsonl"
            canonical.parent.mkdir(parents=True)
            canonical.write_text("previous\n")

            def factory(**kwargs):
                shutil.rmtree(primary)
                raise RuntimeError("world initialization failed")

            episodes = EpisodeService(task_id="fake_1", experiment_name="split", primary_task_dir=primary,
                                      world_factory=factory)
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                with episodes.open():
                    pass
            self.assertEqual(canonical.read_text(), "previous\n")
            self.assertFalse(episodes.is_open)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "outputs/split"
        self.primary = self.output / "tasks/fake_1"
        self.artifacts = self.output / "baseline_artifacts/metaweft/fake_1"
        self.artifacts.mkdir(parents=True)
        self.events = []
        task = SimpleNamespace(instruction=TASK_SPEC["instruction"],
                               supervisor=SimpleNamespace(**TASK_SPEC["supervisor"]),
                               app_descriptions=TASK_SPEC["app_descriptions"],
                               ground_truth="SECRET SOLUTION")
        self.context = SimpleNamespace(task_id="fake_1", task=task, artifacts_dir=self.artifacts,
                                       primary_task_dir=self.primary)
        self.plan = {"setting": "direct", "source_framework": str(ROOT / "MetaWeft"),
                     "run_directory": str(self.root), "recovery_identity": "test-identity",
                     "provider": {"chat_base_url": "https://provider.invalid/v1"},
                     "settings": {"chat_model": "test", "embedding_model": "embed",
                                  "embedding_dimensions": 2, "max_interactions": 2,
                                  "max_retries": 0, "retry_delay_seconds": 0,
                                  "chat_timeout": 120, "embedding_timeout": 30}}
        self.phase = {"output_directory": str(self.output), "ordered_task_ids": ["fake_1"]}
        self.manifest = {"completed_task_ids": [], "degraded_task_ids": [], "degraded_reasons": {}}
        self.model = FakeModel(self.events)
        self.logger = task_lm_logger(self.output.parent, "split", "metaweft", "fake_1")
        self.episodes = EpisodeService(
            task_id="fake_1", experiment_name="split", primary_task_dir=self.primary,
            appworld_kwargs={"load_ground_truth": False}, world_factory=self.make_world,
        )
        self.evaluator = Mock()
        self.evaluator.evaluate_task.side_effect = self.evaluate
        self.services = SimpleNamespace(episodes=self.episodes, environment=self.episodes,
                                        completion=self.episodes, evaluator=self.evaluator, llm=self.logger)
        self.passed = 1
        self.evaluator_error = None

    def make_world(self, **kwargs):
        self.assertFalse(kwargs["load_ground_truth"])
        self.events.append("open")
        shutil.rmtree(self.primary)
        for directory in ("dbs", "checkpoints", "logs", "version"):
            (self.primary / directory).mkdir(parents=True)
        for filename in ("logs/api_calls.jsonl", "logs/environment_io.md", "version/code.txt", "version/data.txt"):
            (self.primary / filename).write_text("")
        return SimpleNamespace(execute=lambda code: "observed\n", task_completed=lambda: True,
                               close=lambda: self.events.append("close"))

    def evaluate(self, *, suppress_errors):
        self.events.append("evaluate")
        self.assertTrue(suppress_errors)
        self.assertFalse(self.episodes.is_open)
        if self.evaluator_error:
            raise self.evaluator_error
        directory = self.primary / "evaluation"
        directory.mkdir()
        (directory / "report.md").write_text("SECRET EVALUATOR ASSERTION")
        (directory / "version.txt").write_text("test")
        return SimpleNamespace(tracker=SimpleNamespace(pass_count=self.passed, fail_count=1-self.passed),
                               report="SECRET EVALUATOR ASSERTION")

    def run_adapter(self, pipeline=None, store=None):
        if not (self.output / "checkpoint.json").is_file():
            write_json(self.artifacts.parent / "memory_initial.json", {
                "status": "verified" if self.plan["setting"] == "base" else "disabled", "records": []})
            if self.plan["setting"] == "base":
                write_json(self.artifacts.parent / "memory_mapping.json", {})
            commit_checkpoint(self.plan, self.phase, [])
        adapter = worker.MetaWeftAdapter(self.plan, self.phase, self.manifest, pipeline, store)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-for-offline-tests"}), \
             patch.object(worker, "ModelClient", return_value=self.model), \
             patch.object(worker.CachedDBHandler, "reset", side_effect=lambda: self.events.append("cache_reset")):
            return adapter.run_task(self.context, self.services)

    def test_direct_runs_only_execution_then_post_close_evaluation(self):
        result = self.run_adapter()
        self.assertEqual([call["phase"] for call in self.model.calls], ["execution"])
        self.assertEqual(result.metadata["reward"], 1)
        self.assertEqual(result.metadata["stages"]["context"]["status"], "disabled")
        self.assertEqual(result.metadata["stages"]["update"]["status"], "disabled")
        self.assertEqual(self.events, ["open", "close", "evaluate", "cache_reset", "model_close"])
        self.assertNotIn("SECRET", json.dumps(self.model.calls))
        self.assertNotIn("SECRET", (self.artifacts / "evidence.json").read_text())
        self.assertEqual(self.manifest["completed_task_ids"], ["fake_1"])

    def test_failed_reward_reaches_reflection_without_evaluator_details(self):
        self.plan["setting"] = "base"
        self.passed = 0
        prepared = SimpleNamespace(guidance="", degraded=False, to_dict=lambda: {"status": "empty_memory"})
        updated = SimpleNamespace(degraded=False, to_dict=lambda: {"status": "completed"})
        pipeline = Mock()
        pipeline.retrieval_meta = load_preset("base")
        pipeline.prepare_context.return_value = prepared
        pipeline.post_task.return_value = updated
        store = Mock()
        store.export.return_value = []
        result = self.run_adapter(pipeline, store)
        self.assertEqual(result.metadata["reward"], 0)
        evidence = pipeline.post_task.call_args.args[0]
        self.assertEqual(evidence["reward"], 0)
        self.assertEqual(set(evidence), {"task_spec", "trace", "reward"})
        self.assertNotIn("SECRET", json.dumps(evidence))
        self.assertLess(self.events.index("close"), self.events.index("evaluate"))
        self.assertEqual(self.manifest["completed_task_ids"], ["fake_1"])

    def test_evaluator_infrastructure_error_is_fatal_and_cleans_cache(self):
        self.evaluator_error = RuntimeError("evaluator infrastructure failure")
        with self.assertRaisesRegex(RuntimeError, "infrastructure failure"):
            self.run_adapter()
        self.assertEqual(self.events, ["open", "close", "evaluate", "cache_reset", "model_close"])
        self.assertEqual(self.manifest["completed_task_ids"], [])

    def test_reused_task_never_constructs_a_model_or_reopens_the_world(self):
        first = self.run_adapter()
        self.events.clear()
        with patch.object(worker, "ModelClient", side_effect=AssertionError("must not call")):
            adapter = worker.MetaWeftAdapter(self.plan, self.phase, self.manifest)
            second = adapter.run_task(self.context, self.services)
        self.assertEqual(first.metadata, second.metadata)
        self.assertEqual(self.events, [])

    def test_required_memory_snapshot_failure_does_not_commit_task(self):
        self.plan["setting"] = "base"
        stage = SimpleNamespace(guidance="", degraded=False, to_dict=lambda: {})
        pipeline = Mock(prepare_context=Mock(return_value=stage), post_task=Mock(return_value=stage))
        pipeline.retrieval_meta = load_preset("base")
        store = Mock(export=Mock(side_effect=RuntimeError("store unavailable")))
        with self.assertRaisesRegex(RuntimeError, "Cannot checkpoint"):
            self.run_adapter(pipeline, store)
        checkpoint = json.loads((self.output / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["completed_task_ids"], [])
        self.assertEqual(self.manifest["completed_task_ids"], [])


class WorkerMainTests(unittest.TestCase):
    def test_direct_main_passes_no_gt_and_never_constructs_memory_services(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "experiments/outputs/split"
            phase = {"output_directory": str(output), "dataset": "fake_dataset", "split": "split",
                     "experiment_name": "split", "ordered_task_ids": ["fake_1"],
                     "source_task_count": 2, "partial": True}
            plan = {"phases": [phase], "setting": "direct", "variant_id": "metaweft-direct-v1",
                    "run_id": "test", "scenario": "direct", "dataset_version": "2h", "note": "offline",
                    "run_directory": str(root), "recovery_identity": "test-identity",
                    "settings": {"appworld_seed": 100}, "source_framework": str(ROOT / "MetaWeft")}
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))

            def run_dataset(**kwargs):
                self.assertFalse(kwargs["appworld_kwargs"]["load_ground_truth"])
                self.assertTrue(kwargs["suppress_evaluation_errors"])
                self.assertIsNone(kwargs["adapter"].pipeline)
                self.assertIsNone(kwargs["adapter"].store)
                kwargs["adapter"].manifest["completed_task_ids"] = ["fake_1"]
                primary = output / "tasks/fake_1"
                primary.mkdir(parents=True)
                (primary / "evaluation").mkdir()
                (primary / "evaluation/report.md").write_text("official fake task report")
                write_json(output / "baseline_artifacts/metaweft/fake_1/result.json", {"degraded": []})
                directory = output / "evaluations"
                directory.mkdir()
                (directory / "fake_dataset.json").write_text(json.dumps({"aggregate": {}, "individual": {"fake_1": {}}}))
                (directory / "fake_dataset.txt").write_text("official fake output")

            with patch.object(worker.sys, "argv", ["worker.py", str(plan_path), "0"]), \
                 patch.object(worker.signal, "signal"), \
                 patch.object(worker, "load_task_stream", return_value=[SimpleNamespace(task_id="fake_1")]), \
                 patch.object(worker, "run_dataset", side_effect=run_dataset), \
                 patch.object(worker.CachedDBHandler, "reset"), \
                 patch.object(worker, "ModelClient") as model, \
                 patch("MetaWeft.service.elasticsearch.ElasticsearchStore") as store:
                worker.main()
                model.assert_not_called()
                store.assert_not_called()
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            initial = json.loads((output / "baseline_artifacts/metaweft/memory_initial.json").read_text())
            self.assertEqual(initial, {"status": "disabled", "records": []})


class ResponseLoggingTests(unittest.TestCase):
    def test_empty_text_retains_billed_tokens_before_successful_retry(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs/split"
            logger = task_lm_logger(output.parent, "split", "metaweft", "fake_1")
            empty, valid = response(""), response("ok")
            empty.model_dump = lambda: {"choices": [{"message": {"content": ""}}],
                                       "usage": {"prompt_tokens": 21192, "completion_tokens": 0, "total_tokens": 21192}}
            valid.model_dump = lambda: {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}}
            provider = Mock()
            provider.chat.completions.create.side_effect = [empty, valid]
            model = ModelClient(client=provider, chat_model="test", max_retries=1, retry_delay_seconds=0,
                                recorder=worker.physical_recorder(logger))
            self.assertIs(model.chat([], phase="execution", component="react"), valid)
            rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
            self.assertEqual([row["metadata"]["attempt"] for row in rows], [1, 2])
            self.assertEqual(len({row["metadata"]["logical_call_id"] for row in rows}), 1)
            self.assertEqual(worker.accounting(output)["reported_tokens"]["total_tokens"], 21195)

    def test_malformed_embedding_envelopes_are_recorded_before_retry(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs/split"
            logger = task_lm_logger(output.parent, "split", "metaweft", "fake_1")
            invalid = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=42)],
                                      model_dump=lambda: {"data": [{"embedding": 42}], "usage": {"total_tokens": 2}})
            valid = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])],
                                    model_dump=lambda: {"data": [{"embedding": [0.1, 0.2]}], "usage": {"total_tokens": 2}})
            provider = Mock()
            provider.embeddings.create.side_effect = [invalid, valid]
            model = ModelClient(client=provider, chat_model="test", dimensions=2, max_retries=1,
                                retry_delay_seconds=0, recorder=worker.physical_recorder(logger))
            self.assertEqual(model.embed(["a"], phase="retrieval", component="base"), [[0.1, 0.2]])
            rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
            self.assertTrue(rows[0]["output"]["invalid_embedding_response"])
            self.assertEqual(worker.accounting(output)["reported_tokens"]["total_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
