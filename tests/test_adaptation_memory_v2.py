"""Offline coverage of successful adapter-event selection and atomic V2 edits."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from MetaWeft.src.ace_adapt_reme_task.common import SchemaError
from MetaWeft.src.ace_adapt_reme_task_v2.adaptation_memory import (
    AdaptationMemory, apply_edits, empty_playbook, playbook_content, render_playbook, validate_playbook,
)
from MetaWeft.src.ace_adapt_reme_task_v2.evidence import adaptation_projection
from MetaWeft.src.framework import EpisodeEvidence, StateRef


def episode(state=None, *, reward=1, status="completed", guide="Discover missing details before applying a procedure."):
    state = empty_playbook() if state is None else state
    payload = {"task_spec": {"task_id": "task1", "instruction": "Discover the required details"},
               "trace": [{"step": 1, "code": "TRACE_CODE_SENTINEL", "observation": "TRACE_OBSERVATION_SENTINEL"}],
               "reward": reward, "prior_playbook": {"state": deepcopy(state), "rendered": render_playbook(state)},
               "adaptation": {"status": status, "guide": guide, "raw_output": "RAW_OUTPUT_SENTINEL",
                              "messages": [{"role": "system", "content": "DUPLICATE_MESSAGE_SENTINEL"}],
                              "input": {"task_spec": {"task_id": "task1", "instruction": "Discover the required details"},
                                        "retrieved_memories": ["Check prerequisites before acting"],
                                        "playbook": playbook_content(state),
                                        "preparation_status": {"task": "completed", "adaptation": "completed"}}}}
    return EpisodeEvidence(payload, "task1/episode_evidence.json")


def edit(content="When required details are missing, turn them into discovery goals."):
    return {"operation": "ADD", "target_id": None, "heading": "exploration_guidance", "content": content,
            "evidence": ["/adaptation/output/guidance", "/adaptation/input/task_spec/instruction"],
            "reason": "The guide expresses missing task details as discovery goals."}


def apply(state, edits, *, projection=None, **kwargs):
    return apply_edits(state, {"base_revision": state["revision"], "edits": edits},
                       adaptation_projection(episode(state)) if projection is None else projection,
                       source_ref="task1/adaptation_updates.json", adaptation_template="system", **kwargs)


class PlaybookTests(unittest.TestCase):
    def test_new_schema_and_deterministic_content_only_rendering(self):
        state, _ = apply(empty_playbook(), [edit("Second heading rule\nContinued"),
                                         {**edit("Select by applicability"), "heading": "memory_selection"}])
        state["entries"].reverse()
        rendered = render_playbook(state)
        self.assertLess(rendered.index("Select by applicability"), rendered.index("Second heading rule"))
        self.assertIn("Second heading rule\n  Continued", rendered)
        self.assertEqual(playbook_content(state)["exploration_guidance"], ["Second heading rule\nContinued"])
        for hidden in ("adapt-000001", "source_refs", "adaptation_updates.json", "next_entry_number"):
            self.assertNotIn(hidden, rendered)
            self.assertNotIn(hidden, json.dumps(playbook_content(state)))
        for change in ({"schema_version": 1}, {"revision": True}, {"next_entry_number": 9}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_playbook({**state, **change})
        invalid = deepcopy(state)
        invalid["entries"][0]["heading"] = "verification"
        with self.assertRaises(ValueError):
            validate_playbook(invalid)

    def test_projection_preserves_exact_input_once_with_target_locations(self):
        state, _ = apply(empty_playbook(), [edit(), {**edit("Filter irrelevant memories"), "heading": "memory_selection"}])
        value = episode(state)
        projected = adaptation_projection(value)
        self.assertEqual(projected["adaptation"]["input"], value.payload["adaptation"]["input"])
        self.assertEqual(projected["adaptation"]["output"], {"guidance": value.payload["adaptation"]["guide"]})
        self.assertEqual(projected["edit_targets"], {
            "adapt-000002": "/adaptation/input/playbook/memory_selection/0",
            "adapt-000001": "/adaptation/input/playbook/exploration_guidance/0"})
        serialized = json.dumps(projected)
        for excluded in ("trace", "reward", "RAW_OUTPUT_SENTINEL", "DUPLICATE_MESSAGE_SENTINEL", "source_refs"):
            self.assertNotIn(excluded, serialized)

    def test_mixed_edits_preserve_history_ids_and_untargeted_content(self):
        state, _ = apply(empty_playbook(), [edit("Keep exactly\nthis line."), edit("Original")])
        prior = deepcopy(state)
        update = {**edit("Improved conditional rule"), "operation": "UPDATE", "target_id": "adapt-000002",
                  "heading": "task_specific_pruning"}
        result, applied = apply(state, [update, edit("New rule")])
        self.assertEqual(state, prior)
        self.assertEqual(result["revision"], 2)
        self.assertEqual(result["next_entry_number"], 4)
        self.assertEqual(result["entries"][0], prior["entries"][0])
        self.assertEqual(result["entries"][1]["id"], "adapt-000002")
        self.assertEqual(applied[0]["before"], prior["entries"][1])
        self.assertEqual(result["entries"][1]["source_refs"],
                         ["task1/adaptation_updates.json#/applied/1", "task1/adaptation_updates.json#/applied/0"])

    def test_only_output_and_substantive_input_can_ground_edits(self):
        projection = adaptation_projection(episode())
        invalid = ["/trace/0/observation", "/reward", "/base_revision", "/edit_targets",
                   "/adaptation/input/task_spec/task_id", "/adaptation/input/preparation_status/task",
                   "/adaptation/input/task_spec", "/adaptation/input/retrieved_memories",
                   "/adaptation/input/retrieved_memories/99", "/adaptation/input/retrieved_memories/00",
                   "/adaptation/input/task_spec/~2", "/adaptation/input"]
        for pointer in invalid:
            with self.subTest(pointer=pointer), self.assertRaises(SchemaError):
                apply(empty_playbook(), [{**edit(), "evidence": ["/adaptation/output/guidance", pointer]}],
                      projection=projection)
        for pointers in (["/adaptation/output/guidance"], ["/adaptation/input/task_spec/instruction"],
                         ["/adaptation/output/guidance", "/adaptation/output/guidance"]):
            with self.subTest(pointers=pointers), self.assertRaises(SchemaError):
                apply(empty_playbook(), [{**edit(), "evidence": pointers}])
        state, _ = apply(empty_playbook(), [{**edit(), "evidence": ["/adaptation/output/guidance",
                                                                  "/adaptation/input/retrieved_memories/0"]}])
        self.assertEqual(state["revision"], 1)

    def test_invalid_batch_is_atomic_and_stale_revision_is_rejected(self):
        state, _ = apply(empty_playbook(), [edit()])
        prior = deepcopy(state)
        update = {**edit(), "operation": "UPDATE", "target_id": "adapt-000001"}
        invalid_edits = [{**edit(), "operation": "REMOVE"}, {**edit(), "heading": "verification"},
                         {**edit(), "target_id": "adapt-000001"}, {**edit(), "content": " "},
                         {**update, "target_id": "adapt-000009"}]
        for invalid in invalid_edits:
            with self.subTest(edit=invalid), self.assertRaises(SchemaError):
                apply(state, [edit("Valid prefix"), invalid])
            self.assertEqual(state, prior)
        with self.assertRaises(SchemaError):
            apply(state, [update, update])
        with self.assertRaises(SchemaError):
            apply_edits(state, {"base_revision": 0, "edits": []}, adaptation_projection(episode(state)),
                        source_ref="event.json", adaptation_template="system")

    def test_empty_guidance_noop_and_serialized_playbook_budget(self):
        projection = adaptation_projection(episode(guide=""))
        state, _ = apply(empty_playbook(), [edit()], projection=projection)
        unchanged, applied = apply(state, [])
        self.assertEqual(unchanged, state)
        self.assertEqual(applied, [])
        quoted = edit('"' * 300)
        # JSON escaping, rather than the shorter text rendering, determines this guard.
        with self.assertRaisesRegex(SchemaError, "byte guard"):
            apply(empty_playbook(), [quoted], max_bytes=600)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts = self.root / "task1"
        self.prior = StateRef("adaptation", "metaweft-adaptation-memory-v2", "stream", 0, "initial.json", "hash")
        self.templates = {"adaptation": "fixed adapter", "adaptation_update": "fixed updater"}
        self.calls = []

    def services(self, response=None, failure=None):
        def chat(messages, **kwargs):
            self.calls.append((deepcopy(messages), kwargs))
            if failure:
                raise failure
            raw = response if isinstance(response, str) else json.dumps(response)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
        return {"llm": SimpleNamespace(chat=chat), "component_prefix": "metaweft_v2"}

    def test_failed_partial_and_degraded_adapter_skip_without_call(self):
        for reward, status, reason in ((0, "completed", "task_not_successful"),
                                       (0.9, "completed", "task_not_successful"),
                                       (1, "degraded", "adapter_not_completed")):
            with self.subTest(reward=reward, status=status):
                memory = AdaptationMemory(self.root, self.templates)
                result = memory.update(episode(reward=reward, status=status), self.prior,
                                       services=self.services(), artifacts=self.artifacts)
                event = json.loads((self.artifacts / "adaptation_updates.json").read_text())
                self.assertEqual(result.status, "skipped")
                self.assertFalse(result.degraded)
                self.assertEqual(result.selected_evidence_refs, [])
                self.assertEqual(event["reason"], reason)
                self.assertEqual(event["result"], empty_playbook())
                self.assertEqual(memory.playbook, empty_playbook())
                self.assertNotIn("messages", event)
        self.assertEqual(self.calls, [])

    def test_success_update_contains_no_trace_or_reward_and_records_exact_request(self):
        memory = AdaptationMemory(self.root, self.templates)
        result = memory.update(episode(), self.prior,
                               services=self.services({"base_revision": 0, "edits": [edit()]}), artifacts=self.artifacts)
        self.assertFalse(result.degraded)
        self.assertEqual(memory.playbook["revision"], 1)
        self.assertEqual(len(self.calls), 1)
        event = json.loads((self.artifacts / "adaptation_updates.json").read_text())
        self.assertEqual(event["messages"], self.calls[0][0])
        self.assertEqual(self.calls[0][1]["component"], "metaweft_v2.adaptation_update")
        for excluded in ("TRACE_", "reward", "raw_output", "DUPLICATE_MESSAGE"):
            self.assertNotIn(excluded, json.dumps(event["input"]))

    def test_invalid_output_and_overflow_preserve_prior_and_fatal_failure_propagates(self):
        for maximum, response in ((900000, "invalid JSON"), (10, {"base_revision": 0, "edits": []})):
            memory = AdaptationMemory(self.root, self.templates, max_bytes=maximum)
            result = memory.update(episode(), self.prior, services=self.services(response), artifacts=self.artifacts)
            self.assertTrue(result.degraded)
            self.assertEqual(memory.playbook, empty_playbook())
        self.assertEqual(len(self.calls), 1)
        memory = AdaptationMemory(self.root, self.templates)
        with self.assertRaisesRegex(RuntimeError, "provider"):
            memory.update(episode(), self.prior, services=self.services(failure=RuntimeError("provider")),
                          artifacts=self.artifacts)
        event = json.loads((self.artifacts / "adaptation_updates.json").read_text())
        self.assertEqual(event["status"], "failed")
        self.assertEqual(memory.playbook, empty_playbook())


if __name__ == "__main__":
    unittest.main()
