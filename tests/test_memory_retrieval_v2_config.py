"""Offline contracts for explicit v2 presets, controls and their provenance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from MetaWeft.src.memory_retrieval import (
    PRESETS, load_preset, prompt_manifest, required_representations, validate_meta,
)
from MetaWeft.src.memory_retrieval import config


NEW_PRESETS = tuple(f"{category}_{rank}" for category in
                    ("performance", "efficiency", "transferability") for rank in (1, 2, 3))


class V2ConfigurationTests(unittest.TestCase):
    def test_defaults_and_exact_overrides_keep_independent_depths(self):
        depths = (225, 225, 80, 20, 80, 80, 80, 80, 225)
        for name, depth in zip(NEW_PRESETS, depths):
            with self.subTest(name=name):
                self.assertIn(name, PRESETS)
                meta = load_preset(name)
                self.assertEqual(meta["search"]["backend"], "hnsw")
                self.assertEqual(meta["search"]["depth"], depth)
                self.assertEqual(meta["search"]["num_candidates"], 2 * depth)
                self.assertIsNone(meta["search"]["max_bank_size"])
                self.assertEqual(meta["fusion"], {"method": "rrf", "rank_constant": 20,
                                                 "candidate_limit": depth})
                self.assertEqual(required_representations(meta), ["combined"])
                for k in (0, 5, 20):
                    self.assertEqual(validate_meta(meta, k=k), meta)
                expected = deepcopy(meta)
                expected["search"].update(backend="exact", num_candidates=None)
                self.assertEqual(load_preset(name, backend="exact"), expected)
                self.assertEqual(load_preset(name, num_candidates=1000)["search"]["depth"], depth)
        control = load_preset("exact_base")
        self.assertEqual(control["search"]["backend"], "exact")
        self.assertEqual(control["search"]["depth"], "all")
        self.assertEqual(control["search"]["max_bank_size"], 225)
        self.assertEqual(control["fusion"]["candidate_limit"], "all")
        self.assertEqual(control["reranking"]["stages"], [])

    def test_resolution_rejects_incompatible_overrides(self):
        for name, kwargs in (("base", {"backend": "hnsw"}),
                             ("performance_1", {"backend": "other"}),
                             ("performance_1", {"backend": "exact", "num_candidates": 450}),
                             ("performance_3", {"num_candidates": 40}),
                             ("exact_base", {"backend": "hnsw"}),
                             ("exact_base", {"num_candidates": 450})):
            with self.subTest(name=name, kwargs=kwargs), self.assertRaises(ValueError):
                load_preset(name, **kwargs)
        self.assertEqual(load_preset("base", num_candidates=40)["search"]["num_candidates"], 40)
        self.assertEqual(load_preset()["schema_version"], 1)

    def test_finite_exact_preset_switches_to_resolved_hnsw_count(self):
        meta = load_preset("performance_3", backend="exact")
        with TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / "presets").mkdir()
            (package / "presets/performance_3.json").write_text(json.dumps(meta))
            (package / "prompts").symlink_to(config.PACKAGE / "prompts", target_is_directory=True)
            with patch.object(config, "PACKAGE", package):
                resolved = load_preset("performance_3", backend="hnsw")
                self.assertEqual(resolved["search"]["num_candidates"], 160)
                self.assertEqual(resolved["reranking"], meta["reranking"])
                self.assertEqual(load_preset("performance_3", backend="hnsw", num_candidates=320)
                                 ["search"]["num_candidates"], 320)

    def test_category_pairs_are_identical_and_there_are_six_families(self):
        behavior = lambda meta: {key: value for key, value in meta.items()
                                 if key not in ("name", "description")}
        for backend in ("hnsw", "exact"):
            options = {"backend": backend}
            if backend == "hnsw":
                options["num_candidates"] = 1000
            configs = {name: behavior(load_preset(name, **options)) for name in NEW_PRESETS}
            self.assertEqual(len({json.dumps(meta, sort_keys=True) for meta in configs.values()}), 6)
            for left, right in (("performance_3", "transferability_1"),
                                ("efficiency_3", "transferability_2"),
                                ("performance_1", "transferability_3")):
                self.assertEqual(configs[left], configs[right])

    def test_historical_model_and_response_profiles_remain_distinct(self):
        flash20 = load_preset("efficiency_1")
        self.assertEqual(flash20["provider_policy"], {"max_retries": 5, "retry_delay_seconds": 60})
        self.assertEqual(flash20["reranking"]["stages"][0]["response"], {
            "accept_wrapped": False, "repair_duplicates": True, "schema_retries": 0,
            "fallback_to_input": False, "fallback_on_empty_response": False})
        max80 = load_preset("efficiency_2")
        max_stage = max80["reranking"]["stages"][0]
        self.assertEqual(max_stage["model"], "qwen3.8-max-0902")
        self.assertEqual(max_stage["generation"]["max_tokens"], 512)
        self.assertEqual(max_stage["response"]["schema_retries"], 1)
        self.assertTrue(max_stage["response"]["fallback_to_input"])
        self.assertFalse(max_stage["response"]["fallback_on_empty_response"])
        self.assertEqual(max80["provider_policy"]["max_retries"], 2)
        select, grade = load_preset("performance_1")["reranking"]["stages"]
        self.assertTrue(select["response"]["fallback_on_empty_response"])
        self.assertEqual(grade["generation"], {"temperature": 0, "seed": 0,
            "enable_thinking": True, "thinking_budget": 2048,
            "max_tokens": None, "max_completion_tokens": 6144})
        self.assertFalse(grade["response"]["repair_duplicates"])
        self.assertEqual(load_preset("performance_2")["reranking"]["stages"][1]
                         ["generation"]["max_tokens"], 1024)
        self.assertEqual(load_preset("performance_3")["reranking"]["stages"][0]
                         ["generation"]["max_tokens"], 2048)

    def test_invalid_schema_fields_and_boundaries(self):
        cases = [
            (("unknown",), True), (("schema_version",), True),
            (("search", "backend"), "dense"), (("search", "depth"), True),
            (("search", "depth"), 0), (("search", "depth"), "all"),
            (("search", "num_candidates"), True), (("search", "num_candidates"), 224),
            (("search", "num_candidates"), 10001), (("search", "max_bank_size"), 225),
            (("search", "routes", 0, "branch"), "missing"),
            (("fusion", "candidate_limit"), "all"), (("fusion", "candidate_limit"), 0),
            (("fusion", "rank_constant"), True), (("fusion", "method"), "weighted"),
            (("provider_policy", "max_retries"), True),
            (("provider_policy", "max_retries"), -1),
            (("provider_policy", "retry_delay_seconds"), float("nan")),
            (("provider_policy", "retry_delay_seconds"), True),
            (("provider_policy", "retry_delay_seconds"), -1),
            (("reranking", "stages", 0, "fields"), []),
            (("reranking", "stages", 0, "fields"), ["id"]),
            (("reranking", "stages", 0, "fields"), ["content", "content"]),
            (("reranking", "stages", 0, "fields"), [{}]),
            (("reranking", "stages", 0, "id"), "grade"),
            (("reranking", "stages", 0, "model"), " "),
            (("reranking", "stages", 0, "output_limit"), 5),
            (("reranking", "stages", 0, "small_input_policy"), "run"),
            (("reranking", "stages", 0, "prompt"), "general_v1"),
            (("reranking", "stages", 0, "response", "schema_retries"), True),
            (("reranking", "stages", 0, "response", "schema_retries"), 2),
            (("reranking", "stages", 0, "response", "accept_wrapped"), 1),
            (("reranking", "stages", 0, "response", "fallback_to_input"), False),
            (("reranking", "stages", 1, "response", "repair_duplicates"), True),
            (("reranking", "stages", 1, "output_limit"), 20),
            (("reranking", "stages", 0, "generation", "temperature"), float("inf")),
            (("reranking", "stages", 0, "generation", "temperature"), True),
            (("reranking", "stages", 0, "generation", "seed"), False),
            (("reranking", "stages", 0, "generation", "max_tokens"), 0),
            (("reranking", "stages", 0, "generation", "thinking_budget"), 2048),
            (("reranking", "stages", 1, "generation", "thinking_budget"), 6144),
            (("reranking", "stages", 1, "generation", "max_tokens"), 2048),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                meta = load_preset("performance_1")
                target = meta
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    validate_meta(meta, k=20)
        meta = load_preset("performance_1")
        for key in ("search", "fusion", "reranking", "provider_policy"):
            bad = deepcopy(meta)
            bad[key]["unknown"] = 1
            with self.subTest(extra=key), self.assertRaises(ValueError):
                validate_meta(bad)
        for stages in (None, list(reversed(meta["reranking"]["stages"])),
                       meta["reranking"]["stages"] * 2):
            bad = deepcopy(meta)
            bad["reranking"]["stages"] = stages
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                validate_meta(bad)
        for k in (True, -1, 1.5, 21):
            with self.subTest(k=k), self.assertRaises(ValueError):
                validate_meta(meta, k=k)
        for group, field in (("search", "depth"), ("fusion", "candidate_limit")):
            bad = deepcopy(meta)
            bad[group][field] = 4
            with self.assertRaises(ValueError):
                validate_meta(bad, k=5)
        for capacity in (None, 0, True):
            bad = load_preset("exact_base")
            bad["search"]["max_bank_size"] = capacity
            with self.assertRaises(ValueError):
                validate_meta(bad)

    def test_general_v2_branches_preserve_existing_generation_contract(self):
        meta = load_preset("performance_3")
        expanded = load_preset("setting_4")
        meta["query_generation"] = expanded["query_generation"]
        meta["search"]["routes"] = expanded["search"]["routes"]
        self.assertEqual(required_representations(meta), ["combined", "applicability"])
        manifest = prompt_manifest(meta)
        self.assertEqual(set(manifest), {"general_v1", "applicability_v1", "utility_scores_v1"})
        meta["query_generation"]["branches"][0]["prompt"] = "utility_scores_v1"
        with self.assertRaises(ValueError):
            validate_meta(meta)

    def test_prompt_identity_descriptors_and_mutation_isolation(self):
        manifest = prompt_manifest(load_preset("performance_1"))
        expected = {
            "procedure_listwise20_v1": "7902048bc7106a5460d55325d4bc558ebcffa06087da87058e1bc3ab76271e80",
            "utility_scores_v1": "b49f4a42df8c4aadca04d4826c11f28e71701019a267d7d4c5ff49740f3152f6",
        }
        self.assertEqual(set(manifest), set(expected))
        for identifier, digest in expected.items():
            entry = manifest[identifier]
            self.assertEqual(entry["kind"], "reranking")
            self.assertEqual(entry["sha256"], digest)
            self.assertEqual(hashlib.sha256(entry["text"].encode()).hexdigest(), digest)
        for name in (*NEW_PRESETS, "exact_base"):
            descriptor = json.loads((config.PACKAGE / "presets" / f"{name}.variant.json").read_text())
            self.assertTrue({"variant_id", "parent_variant", "parent_source", "hypothesis",
                "intended_differences", "memory_structure", "retrieval", "adaptation",
                "execution_context", "evidence_update", "feedback_access", "memory_lifetime",
                "selection_category", "selection_rank", "shared_design"}.issubset(descriptor))
        meta = load_preset("performance_1")
        validated = validate_meta(meta)
        validated["reranking"]["stages"][0]["model"] = "changed"
        self.assertEqual(meta, load_preset("performance_1"))
        self.assertNotEqual(meta, validated)


if __name__ == "__main__":
    unittest.main()
