"""Generic two-memory contracts and frozen episode scheduling, without services."""

from copy import deepcopy
from dataclasses import replace
import unittest

from MetaWeft.src.framework import (
    BranchUpdateResult, DualMemoryPipeline, EpisodeEvidence, PreparedMemory, StateRef,
)


def prior_pair():
    return {role: StateRef(role, f"opaque-{role}-schema", "stream-1", 2, f"{role}.json", "content-hash")
            for role in ("task", "adaptation")}


class Channel:
    def __init__(self, role, events, *, degraded=False, failure=None):
        self.role, self.events = role, events
        self.degraded, self.failure = degraded, failure
        self.payload = {"representation": f"opaque-{role}", "nested": ["prior"]}
        self.seen = []

    def prepare(self, task_spec, state, **kwargs):
        self.events.append((self.role, "prepare", deepcopy(task_spec)))
        task_spec["nested"][0] = self.role
        return PreparedMemory(self.role, state, self.payload,
                              "degraded" if self.degraded else "completed",
                              f"{self.role} preparation failed" if self.degraded else None)

    def select_evidence(self, episode):
        self.events.append((self.role, "select", episode.sha256))
        return episode

    def update(self, episode, state, **kwargs):
        self.events.append((self.role, "update", episode.sha256))
        self.seen.append(deepcopy(episode.payload))
        episode.payload["new_memory"] = self.role
        if self.failure:
            raise self.failure
        return BranchUpdateResult(self.role, status="degraded" if self.degraded else "completed",
                                  degraded=self.degraded, error="ordinary failure" if self.degraded else None,
                                  before=state.to_dict(), after={"new_memory": self.role})


class Adapter:
    def __init__(self, events):
        self.events, self.calls = events, []

    def adapt(self, task_spec, task, adaptation, **kwargs):
        self.events.append(("adapter", "adapt"))
        self.calls.append(deepcopy((task_spec, task, adaptation)))
        task.payload["nested"][0] = "adapter mutation"
        adaptation.payload["nested"].append("adapter mutation")
        return {"status": "completed", "guide": "fixed guide"}, "episode/adaptation_event.json"


class FrameworkTests(unittest.TestCase):
    def make(self, *, task_degraded=False, adaptation_degraded=False, failure=None):
        events = []
        task = Channel("task", events, degraded=task_degraded, failure=failure)
        adaptation = Channel("adaptation", events, degraded=adaptation_degraded)
        adapter = Adapter(events)
        return DualMemoryPipeline(task, adaptation, adapter), events

    def evidence(self, pair):
        return EpisodeEvidence({"prior_pair": {role: state.to_dict() for role, state in pair.items()},
                                "prepared": {"task": ["before-task"], "adaptation": ["before-adaptation"]},
                                "reward": 0}, "episode.json")

    def test_opaque_payloads_and_exact_preparation_order_are_isolated(self):
        pipeline, events = self.make()
        task = {"task_id": "task-1", "nested": ["original"]}
        prepared = pipeline.prepare(task, prior_pair(), services={}, artifacts="unused")
        self.assertEqual([event[:2] for event in events],
                         [("task", "prepare"), ("adaptation", "prepare"), ("adapter", "adapt")])
        self.assertEqual(events[0][2], task)
        self.assertEqual(events[1][2], task)
        self.assertEqual(prepared.task_prepared.payload, pipeline.task_channel.payload)
        self.assertEqual(prepared.adaptation_prepared.payload, pipeline.adaptation_channel.payload)
        self.assertEqual(prepared.to_dict()["guidance"], "fixed guide")
        self.assertNotIn("retrieved_ids", prepared.to_dict())
        task["nested"].append("caller mutation")
        pipeline.task_channel.payload["nested"].append("store mutation")
        prepared.verify()
        self.assertEqual(prepared.task_spec["nested"], ["original"])
        self.assertEqual(prepared.task_prepared.payload["nested"], ["prior"])
        prepared.task_prepared.payload["nested"].append("direct tampering")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepared.verify()

    def test_both_preparation_statuses_contribute_to_context_degradation(self):
        for role in ("task", "adaptation"):
            with self.subTest(role=role):
                pipeline, _ = self.make(**{f"{role}_degraded": True})
                prepared = pipeline.prepare({"nested": ["original"]}, prior_pair(), services={}, artifacts="unused")
                self.assertTrue(prepared.degraded)
                self.assertIn(role, prepared.error)
                self.assertEqual(len(pipeline.adapter.calls), 1)

    def test_prior_pair_and_prepared_role_mismatches_fail_before_adaptation(self):
        variants = []
        for field, value in (("stream_id", "different"), ("epoch", 3), ("epoch", True), ("role", "task")):
            pair = prior_pair()
            pair["adaptation"] = replace(pair["adaptation"], **{field: value})
            variants.append(pair)
        variants.append({"task": prior_pair()["task"]})
        for pair in variants:
            pipeline, events = self.make()
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                pipeline.prepare({"nested": []}, pair, services={}, artifacts="unused")
            self.assertEqual(events, [])
        pipeline, _ = self.make()
        pipeline.task_channel.role = "adaptation"
        with self.assertRaisesRegex(ValueError, "role"):
            pipeline.prepare({"nested": ["original"]}, prior_pair(), services={}, artifacts="unused")
        self.assertEqual(pipeline.adapter.calls, [])

    def test_independent_selectors_and_updates_receive_identical_frozen_episode(self):
        pipeline, events = self.make(task_degraded=True)
        pair = prior_pair()
        evidence = self.evidence(pair)
        before = deepcopy(evidence.payload)
        failures = []
        result = pipeline.update(evidence, pair, services={}, artifacts="unused", record_failure=lambda *args: failures.append(args))
        self.assertEqual([event[:2] for event in events],
                         [("task", "select"), ("task", "update"), ("adaptation", "select"), ("adaptation", "update")])
        self.assertEqual(pipeline.task_channel.seen, [before])
        self.assertEqual(pipeline.adaptation_channel.seen, [before])
        self.assertEqual({event[2] for event in events}, {evidence.sha256})
        self.assertTrue(result.task.degraded)
        self.assertFalse(result.adaptation.degraded)
        self.assertTrue(result.degraded)
        self.assertEqual(failures, [])
        evidence.verify()
        self.assertEqual(evidence.payload, before)
        # The selected episode is also independent when its consumers are scheduled in reverse.
        selected = {channel.role: channel.select_evidence(deepcopy(evidence))
                    for channel in (pipeline.task_channel, pipeline.adaptation_channel)}
        for channel in (pipeline.adaptation_channel, pipeline.task_channel):
            channel.update(selected[channel.role], pair[channel.role])
            self.assertEqual(channel.seen[-1], before)

    def test_fatal_update_stops_other_branch_and_keeps_failure_role(self):
        pipeline, events = self.make(failure=RuntimeError("fatal mutation"))
        failures, pair = [], prior_pair()
        evidence = self.evidence(pair)
        with self.assertRaisesRegex(RuntimeError, "fatal mutation"):
            pipeline.update(evidence, pair, services={}, artifacts="unused",
                            record_failure=lambda *args: failures.append(args))
        self.assertEqual([event[:2] for event in events], [("task", "select"), ("task", "update")])
        result, role, error = failures[0]
        self.assertEqual(role, "task")
        self.assertEqual(result.adaptation.status, "not_run")
        self.assertIsInstance(error, RuntimeError)
        evidence.verify()

    def test_updates_reject_another_pair_or_mutated_episode_without_stage_work(self):
        pair = prior_pair()
        for corrupt in ("pair", "evidence"):
            pipeline, events = self.make()
            evidence = self.evidence(pair)
            supplied = deepcopy(pair)
            if corrupt == "pair":
                supplied = {role: replace(ref, epoch=3) for role, ref in supplied.items()}
            else:
                evidence.payload["reward"] = 1
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                pipeline.update(evidence, supplied, services={}, artifacts="unused", record_failure=lambda *_: None)
            self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
