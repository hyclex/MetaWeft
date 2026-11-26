"""Offline checks for the promoted search policies and their native recovery."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from MetaWeft.experiment import recovery
from MetaWeft.experiment.appworld import launch
from MetaWeft.src.ace_adapt_reme_task_v2 import V2Pipeline
from MetaWeft.src.search_designs.client import SearchClient, OptionalOverflow
from MetaWeft.src.search_designs.observation_dict import ObservationDictionary, compact_request
from MetaWeft.src.search_designs.settings import CAPS, NOTES
from MetaWeft.tests import test_recovery_metaweft_v2 as paired


class SearchDesignTests(unittest.TestCase):
    def test_constructors_and_paired_restore(self):
        for design, pipeline in (("contract_check", V2Pipeline), ("observation_dict", ObservationDictionary)):
            for seed in (0, 1):
                args = launch.parse_args(design, ["--dataset", "8h", "--seed", str(seed)])
                plan = launch.make_plan(design, args)
                self.assertEqual([len(p["task_ids"]) for p in plan["phases"]], [57, 138])
                meta = plan["settings"]["retrieval_meta"]
                self.assertEqual(meta["reranking"]["stages"][0]["generation"]["seed"], seed)
                pipeline(Mock(), Path.cwd(), retrieval_meta=meta)
                self.assertIn(NOTES[design], plan["note"])
            fixture = paired.V2RecoveryTests()
            try:
                fixture.setUp()
                fixture.plan.update(setting=design, variant_id=plan["variant_id"])
                recovery.seal_run(fixture.plan)
                fixture.initial = recovery.commit_checkpoint(fixture.plan, fixture.phase, [])
                fixture.changed_pair()
                committed = fixture.commit()
                child_root = fixture.root.parent / "child"
                child = recovery.recovery_plan(fixture.plan, fixture.root, [committed], root=child_root,
                                               tag="child", command=["offline"])
                recovery.clone_run(fixture.root, child, [committed])
                phase = child["phases"][0]
                self.assertEqual(phase["remaining_task_ids"], ["b_1"])
                restored = recovery.load_checkpoint(child, phase, child_root)
                self.assertEqual(restored["state_epoch"], 1)
                self.assertEqual(restored["memories"], committed["memories"])
            finally:
                fixture.doCleanups()

    def test_round_trip_and_threshold(self):
        yes, short = "é" * 128, "é" * 127
        payload = {"evidence": {"a": {"trace": [{"observation": yes}, {"observation": short}]},
                                "b": {"trace": [{"observation": short}, {"observation": yes}]}}}
        template, encoded = compact_request("base", payload)
        self.assertNotEqual(template, "base")
        self.assertEqual(encoded["observation_table"], {"r0": yes})
        decoded = deepcopy(encoded)
        table = decoded.pop("observation_table")
        for source in decoded["evidence"].values():
            for row in source["trace"]:
                if isinstance(row["observation"], dict):
                    row["observation"] = table[row["observation"]["ref"]]
        self.assertEqual(decoded, payload)
        single = {"evidence": {"a": payload["evidence"]["a"]}}
        self.assertEqual(compact_request("base", single), ("base", single))

    def test_wire_controls_and_overflow_before_dispatch(self):
        with TemporaryDirectory() as tmp:
            emit = Mock(return_value=SimpleNamespace(usage=None))
            client = SearchClient(seed=1, recorder=emit, client=Mock(), chat_model="qwen3.8-flash",
                                  cache_path=Path(tmp) / "cache.sqlite3")
            with patch.object(client, "_invoke", return_value=None) as invoke:
                for phase, cap in CAPS.items():
                    options = {"max_tokens": 512} if phase == "retrieval" else {}
                    client.chat([], component="fixture", phase=phase, temperature=0.9, seed=0, **options)
                    request = invoke.call_args.args[1]
                    self.assertEqual((request["seed"], request["temperature"], request["max_tokens"]),
                                     (1, 0, options.get("max_tokens", cap)))
                    self.assertFalse(request["extra_body"]["enable_thinking"])
            for operation, phase, maximum, ceiling, error in (
                ("chat", "execution", 8192, 65536, RuntimeError),
                ("chat", "task_consolidation", 4096, 65536, OptionalOverflow),
                ("embedding", "retrieval", 0, 8192, RuntimeError),
            ):
                request = {"max_tokens": maximum}
                with patch("MetaWeft.src.search_designs.client.input_tokens",
                           side_effect=[ceiling - maximum, ceiling - maximum + 1]):
                    client.recorder(operation, request, Mock(), phase=phase)
                    with self.assertRaises(error):
                        client.recorder(operation, request, Mock(), phase=phase)
            self.assertEqual(emit.call_count, 3)
