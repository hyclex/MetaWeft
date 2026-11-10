"""Offline acceptance for complete playbook access and atomic supported edits."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from MetaWeft.src.ace_adapt_reme_task.adaptation_memory import (
    AdaptationMemory, apply_edits, empty_playbook, render_playbook, validate_playbook,
)
from MetaWeft.src.ace_adapt_reme_task.common import SchemaError
from MetaWeft.src.ace_adapt_reme_task.evidence import validate_source_reference
from MetaWeft.src.framework import EpisodeEvidence, StateRef


def projection(guide="Check retrieved assumptions before using them."):
    return {
        "task_spec": {"task_id": "task-1", "instruction": "Complete the task"},
        "trace": [{"step": 1, "code": "inspect()", "observation": "The assumed identifier was absent."}],
        "reward": 0,
        "retrieved_memories": [],
        "prior_playbook": render_playbook(empty_playbook()),
        "base_revision": 0,
        "adaptation": {"status": "completed", "guide": guide, "messages": [], "raw_output": "{}"},
    }


def add_edit(content="Make discovery explicit when retrieved knowledge assumes an identifier."):
    return {"operation": "ADD", "target_id": None, "heading": "verification", "content": content,
            "evidence": ["/adaptation/guide", "/trace/0/observation"],
            "reason": "The guide relied on an identifier that the observation showed was absent."}


def apply(state, edits, input_projection=None, **options):
    return apply_edits(state, {"base_revision": state["revision"], "edits": edits},
                       projection() if input_projection is None else input_projection,
                       source_ref="task-1/adaptation_updates.json", adaptation_template="fixed system", **options)


class PlaybookTests(unittest.TestCase):
    def test_empty_initialization_and_complete_deterministic_rendering(self):
        self.assertEqual(render_playbook(empty_playbook()),
                         "## Selection\n\n## Specialization\n\n## Composition\n\n## Verification\n")
        first, second, third = add_edit("First line\nSecond line"), add_edit("Another rule"), add_edit("Select useful knowledge")
        third["heading"] = "selection"
        state, _ = apply(empty_playbook(), [first, second, third])
        state["entries"].reverse()
        rendered = render_playbook(state)
        self.assertLess(rendered.index("[adapt-000003]"), rendered.index("[adapt-000001]"))
        self.assertLess(rendered.index("[adapt-000001]"), rendered.index("[adapt-000002]"))
        self.assertIn("[adapt-000001] First line\n    Second line", rendered)
        for storage_field in ("source_refs", "revision", "schema_version", "next_entry_number", "adaptation_updates.json"):
            self.assertNotIn(storage_field, rendered)
        self.assertEqual(rendered, render_playbook(state))

    def test_playbook_schema_has_only_content_knowledge_and_valid_bookkeeping(self):
        state, _ = apply(empty_playbook(), [add_edit()])
        for key, value in (("when_to_use", "condition"), ("name", "title"), ("embedding", [1]), ("score", 0.5)):
            invalid = deepcopy(state)
            invalid["entries"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_playbook(invalid)
        changes = [lambda s: s.update(revision=0), lambda s: s.update(revision=True),
                   lambda s: s.update(schema_version=2), lambda s: s.update(next_entry_number=3),
                   lambda s: s["entries"].append(deepcopy(s["entries"][0])),
                   lambda s: s["entries"][0].update(content="  "),
                   lambda s: s["entries"][0].update(id="adapt-000000"),
                   lambda s: s["entries"][0].update(source_refs=[])]
        for index, change in enumerate(changes):
            invalid = deepcopy(state)
            change(invalid)
            with self.subTest(change=index), self.assertRaises(ValueError):
                validate_playbook(invalid)
        validated = validate_playbook(state)
        validated["entries"][0]["content"] = "changed copy"
        self.assertNotEqual(state, validated)

    def test_mixed_updates_preserve_ids_unrelated_bytes_and_history(self):
        state, _ = apply(empty_playbook(), [add_edit("Keep me exactly.\nSpacing stays."), add_edit("Original rule")])
        before = deepcopy(state)
        update = {**add_edit("Improved rule"), "operation": "UPDATE", "target_id": "adapt-000002", "heading": "selection"}
        result, applied = apply(state, [update, add_edit("New independent rule")])
        self.assertEqual(state, before)
        self.assertEqual(result["revision"], 2)
        self.assertEqual(result["next_entry_number"], 4)
        self.assertEqual(result["entries"][0], state["entries"][0])
        self.assertEqual(result["entries"][1]["id"], "adapt-000002")
        self.assertEqual(result["entries"][1]["heading"], "selection")
        self.assertEqual(result["entries"][1]["source_refs"],
                         ["task-1/adaptation_updates.json#/applied/1", "task-1/adaptation_updates.json#/applied/0"])
        self.assertEqual(result["entries"][2]["id"], "adapt-000003")
        self.assertEqual(applied[0]["before"], state["entries"][1])
        self.assertEqual(applied[0]["after"], result["entries"][1])
        self.assertEqual(applied[1]["before"], None)

    def test_invalid_batch_never_applies_valid_prefix_or_allocates_ids(self):
        state, _ = apply(empty_playbook(), [add_edit()])
        before = deepcopy(state)
        update = {**add_edit(), "operation": "UPDATE", "target_id": "adapt-000001"}
        invalid_edits = [
            {**add_edit(), "operation": "REMOVE"}, {**add_edit(), "target_id": "adapt-000001"},
            {**update, "target_id": "adapt-000999"}, {**add_edit(), "heading": "other"},
            {**add_edit(), "when_to_use": "for this task"}, {**add_edit(), "reason": ""},
            {**add_edit(), "content": "  "}, {**add_edit(), "evidence": ["/reward"]},
            {**add_edit(), "evidence": ["/adaptation/guide", "/trace/9/observation"]},
            {**add_edit(), "evidence": ["/adaptation/guide", "/trace/00/observation"]},
            {**add_edit(), "evidence": ["/adaptation/guide", "/trace/0/observation", "/private"]},
            {**add_edit(), "evidence": ["/adaptation/guide", "/trace/0/observation", "/trace/0/~2"]},
        ]
        for index, invalid in enumerate(invalid_edits):
            with self.subTest(edit=index), self.assertRaises(SchemaError):
                apply(state, [add_edit("Valid prefix"), invalid])
            self.assertEqual(state, before)
        for edits in ([update, update], [add_edit(), {**update, "target_id": "adapt-000002"}]):
            with self.assertRaises(SchemaError):
                apply(state, edits)
            self.assertEqual(state, before)
        result, _ = apply(state, [add_edit("Next valid edit")])
        self.assertEqual(result["entries"][-1]["id"], "adapt-000002")

    def test_stale_revision_extra_outer_keys_and_failed_adapter_are_rejected(self):
        state, _ = apply(empty_playbook(), [add_edit()])
        for proposal in ({"base_revision": 0, "edits": []},
                         {"base_revision": True, "edits": []},
                         {"base_revision": 1, "edits": [], "playbook": "replacement"}):
            with self.assertRaises(SchemaError):
                apply_edits(state, proposal, projection(), source_ref="event.json", adaptation_template="system")
        input_projection = projection()
        input_projection["adaptation"]["status"] = "degraded"
        with self.assertRaises(SchemaError):
            apply(state, [add_edit()], input_projection)
        unchanged, edits = apply(state, [], input_projection)
        self.assertEqual(unchanged, state)
        self.assertEqual(edits, [])

    def test_empty_guidance_valid_noop_and_full_size_guard(self):
        state, _ = apply(empty_playbook(), [add_edit()], projection(guide=""))
        unchanged, edits = apply(state, [])
        self.assertEqual(unchanged, state)
        self.assertEqual(edits, [])
        prior = deepcopy(state)
        with self.assertRaisesRegex(SchemaError, "byte guard"):
            apply(state, [add_edit("é" * 1000)], max_bytes=1000)
        self.assertEqual(state, prior)

    def test_source_reference_splits_fragment_and_checks_path_boundary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = root / "adaptation_updates.json"
            event.write_text(json.dumps({"applied": [{"after": {"content": "rule"}}]}))
            self.assertEqual(validate_source_reference(root, "adaptation_updates.json#/applied/0/after/content"), "rule")
            for reference in ("../adaptation_updates.json#/applied/0", str(event) + "#/applied/0",
                              "adaptation_updates.json#/applied/8", "adaptation_updates.json#/applied/~2"):
                with self.subTest(reference=reference), self.assertRaises(ValueError):
                    validate_source_reference(root, reference)


class AdaptationUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts = self.root / "task-1"
        self.prior = StateRef("adaptation", "playbook-v1", "stream", 0, "initial.json", "hash")
        self.templates = {"adaptation": "fixed adapter", "adaptation_update": "fixed update"}
        self.calls = []

    def evidence(self, state):
        value = projection()
        value["prior_playbook"] = {"state": deepcopy(state), "rendered": render_playbook(state)}
        value.pop("base_revision")
        return EpisodeEvidence(value, "task-1/episode_evidence.json")

    def services(self, raw=None, failure=None):
        references = {}
        def chat(messages, **kwargs):
            self.calls.append((deepcopy(messages), kwargs))
            references.setdefault(kwargs["component"], []).append({"logical_call_id": "call-1", "attempt": 0})
            if failure:
                raise failure
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
        return {"llm": SimpleNamespace(chat=chat, chat_model="offline"),
                "call_refs": references}

    def test_full_prepared_playbook_and_exact_update_request_are_preserved(self):
        state, _ = apply(empty_playbook(), [add_edit("Rule one\ncontinued"), add_edit("Rule two")])
        memory = AdaptationMemory(self.root, self.templates, state)
        prepared = memory.prepare({}, self.prior, services={}, artifacts=self.artifacts)
        self.assertEqual(prepared.payload["rendered"], render_playbook(state))
        raw = json.dumps({"base_revision": 1, "edits": []})
        result = memory.update(self.evidence(state), self.prior, services=self.services(raw), artifacts=self.artifacts)
        event = json.loads((self.artifacts / "adaptation_updates.json").read_text())
        self.assertEqual(event["messages"], self.calls[0][0])
        self.assertEqual(event["input"]["prior_playbook"], prepared.payload["rendered"])
        self.assertEqual(event["messages"][0]["content"], self.templates["adaptation_update"])
        self.assertEqual(event["raw_output"], raw)
        self.assertEqual(event["call_references"][0]["logical_call_id"], "call-1")
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(result.degraded)
        self.assertEqual(memory.playbook, state)

    def test_invalid_model_json_is_degraded_without_repair_or_state_change(self):
        memory = AdaptationMemory(self.root, self.templates)
        result = memory.update(self.evidence(memory.playbook), self.prior,
                               services=self.services("invalid JSON"), artifacts=self.artifacts)
        self.assertTrue(result.degraded)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(memory.playbook, empty_playbook())
        event = json.loads((self.artifacts / "adaptation_updates.json").read_text())
        self.assertEqual(event["raw_output"], "invalid JSON")
        self.assertEqual(event["applied"], [])
        self.assertFalse(event["validation"]["valid"])

    def test_prompt_overflow_preserves_full_input_without_call_or_truncation(self):
        memory = AdaptationMemory(self.root, self.templates, max_bytes=10)
        result = memory.update(self.evidence(memory.playbook), self.prior,
                               services=self.services("unused"), artifacts=self.artifacts)
        self.assertTrue(result.degraded)
        self.assertEqual(self.calls, [])
        self.assertEqual(memory.playbook, empty_playbook())
        event = json.loads((self.artifacts / "adaptation_updates.json").read_text())
        self.assertEqual(event["input"]["prior_playbook"], render_playbook(empty_playbook()))
        self.assertGreater(event["prompt_bytes"], 10)
        self.assertEqual(event["status"], "budget_limited")

    def test_unexpected_call_failure_is_fatal_and_preserves_event(self):
        memory = AdaptationMemory(self.root, self.templates)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            memory.update(self.evidence(memory.playbook), self.prior,
                          services=self.services(failure=RuntimeError("unexpected")), artifacts=self.artifacts)
        event = json.loads((self.artifacts / "adaptation_updates.json").read_text())
        self.assertEqual(event["status"], "failed")
        self.assertEqual(memory.playbook, empty_playbook())
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
