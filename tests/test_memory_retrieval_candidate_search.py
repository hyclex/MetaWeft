"""Offline exact-bank integrity, numerical ordering, and export boundaries."""

from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import requests

from MetaWeft.service.elasticsearch import ElasticsearchStore, StoreSearchError
from MetaWeft.src.memory_retrieval.candidate_search import exact_search, prepare_exact_bank
from MetaWeft.src.memory_retrieval import load_preset, retrieve


def entry(identifier, vector=None):
    return {"id": identifier, "document": {
        "id": identifier, "when_to_use": "When traversing pages", "content": "Read every page",
        "source_refs": ["opaque/source"], "embedding": vector if vector is not None else [1.0, 0.0],
        "when_to_use_embedding": [0.0, 1.0], "content_embedding": [1.0, 1.0],
    }}


class ExactCandidateTests(unittest.TestCase):
    def prepare(self, entries=None, **kwargs):
        self.store = Mock()
        self.store.export.return_value = entries if entries is not None else [entry("a")]
        return prepare_exact_bank(self.store, representations=kwargs.pop("representations", ["combined"]),
                                  dimensions=kwargs.pop("dimensions", 2), **kwargs)

    def test_reference_float64_order_scores_and_id_ties(self):
        rows = [entry("z", [2.0, 0.0]), entry("a", [7.0, 0.0]),
                entry("c", [0.0, 3.0]), entry("b", [-4.0, 0.0])]
        bank = self.prepare(rows)
        records, candidates = exact_search(bank, [1.0, 0.0], representation="combined", depth="all")
        self.assertEqual([row["id"] for row in records], ["a", "z", "c", "b"])
        self.assertEqual([row["score"] for row in candidates], [1.0, 1.0, 0.0, -1.0])
        self.assertEqual([row["rank"] for row in candidates], [1, 2, 3, 4])
        self.assertEqual(bank.matrices["combined"].dtype.name, "float64")
        self.assertEqual(set(records[0]), {"id", "when_to_use", "content", "source_refs"})
        self.store.export.assert_called_once_with()
        self.store.query.assert_not_called()

    def test_complete_order_matches_reference_numerics(self):
        import numpy as np

        generator = np.random.default_rng(19)
        vectors = generator.normal(size=(226, 17))
        query = generator.normal(size=17)
        ids = [f"m{index:04d}" for index in range(len(vectors))]
        rows = [entry(identifier, vector.tolist()) for identifier, vector in zip(ids, vectors)]
        bank = self.prepare(list(reversed(rows)), dimensions=17)
        _, candidates = exact_search(bank, query.tolist(), representation="combined", depth="all")
        scores = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
        expected = sorted(zip(ids, map(float, scores)), key=lambda item: (-item[1], item[0]))
        self.assertEqual([row["id"] for row in candidates], [item[0] for item in expected])
        for actual, (_, expected_score) in zip(candidates, expected):
            self.assertTrue(math.isclose(actual["score"], expected_score, rel_tol=0, abs_tol=1e-12))

    def test_bank_sizes_and_finite_limits_do_not_become_capacity_guards(self):
        for size in (0, 1, 19, 20, 79, 80, 225, 226, 1100):
            with self.subTest(size=size):
                bank = self.prepare([entry(f"m{index:04d}") for index in range(size)])
                self.assertEqual(bank.observed["exported_count"], size)
                for depth in (20, 80, 225, "all"):
                    records, candidates = exact_search(bank, [1.0, 0.0], representation="combined", depth=depth)
                    self.assertEqual(len(records), size if depth == "all" else min(size, depth))
                    self.assertEqual(len(candidates), len(records))

    def test_all_bank_capacity_rejects_226_before_any_query(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.prepare([entry(f"m{index}") for index in range(226)], max_bank_size=225)
        self.store.query.assert_not_called()
        self.assertEqual(self.prepare([entry(f"m{index}") for index in range(225)],
                                      max_bank_size=225).observed["exported_count"], 225)

    def test_routes_reuse_one_export_without_substitution(self):
        bank = self.prepare([entry("a", [1.0, 0.0]), entry("b", [0.0, 1.0])],
                            representations=["combined", "applicability", "content"])
        self.assertEqual(exact_search(bank, [0.0, 1.0], representation="combined", depth=20)[0][0]["id"], "b")
        for name in ("applicability", "content"):
            self.assertEqual(exact_search(bank, [0.0, 1.0], representation=name, depth=20)[0][0]["id"], "a")
        self.store.export.assert_called_once_with()
        with self.assertRaisesRegex(ValueError, "representation"):
            exact_search(bank, [1.0, 0.0], representation="unknown", depth=20)

    def test_call_local_snapshot_and_returned_records_are_independent(self):
        rows = [entry("a")]
        bank = self.prepare(rows)
        initial_hash = bank.observed["exact_snapshot_sha256"]
        rows[0]["document"]["embedding"][:] = [0.0, 1.0]
        rows[0]["document"]["source_refs"].append("changed")
        records, candidates = exact_search(bank, [1.0, 0.0], representation="combined", depth=20)
        self.assertEqual(candidates[0]["score"], 1.0)
        self.assertEqual(records[0]["source_refs"], ["opaque/source"])
        records[0]["source_refs"].append("consumer mutation")
        self.assertEqual(bank.records["a"]["source_refs"], ["opaque/source"])
        later = prepare_exact_bank(self.store, representations=["combined"], dimensions=2)
        self.assertNotEqual(initial_hash, later.observed["exact_snapshot_sha256"])
        self.assertEqual(self.store.export.call_count, 2)

    def test_snapshot_hash_is_sorted_and_covers_selected_evidence(self):
        rows = [entry("β"), entry("a", [0.0, 1.0])]
        original = self.prepare(rows).observed["exact_snapshot_sha256"]
        self.assertEqual(original, self.prepare(list(reversed(rows))).observed["exact_snapshot_sha256"])
        for key, value in (("content", "Changed content"), ("when_to_use", "Changed condition"),
                           ("source_refs", []), ("embedding", [0.5, 0.5])):
            with self.subTest(field=key):
                changed = deepcopy(rows)
                changed[0]["document"][key] = value
                self.assertNotEqual(original, self.prepare(changed).observed["exact_snapshot_sha256"])
        rows[0]["document"]["content_embedding"] = [0.2, 0.8]
        self.assertEqual(original, self.prepare(rows).observed["exact_snapshot_sha256"])
        self.assertNotEqual(original, self.prepare(rows, representations=["combined", "content"])
                            .observed["exact_snapshot_sha256"])

    def test_invalid_export_records_fail_as_integrity_errors(self):
        invalid = [None, {}, [entry("a"), entry("a")], [{"id": "a"}],
                   [{"id": "", "document": {}}], [{"id": True, "document": {}}]]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                store = Mock(export=Mock(return_value=value))
                prepare_exact_bank(store, representations=["combined"], dimensions=2)
        for key, value in (("id", "different"), ("when_to_use", " "), ("content", None),
                           ("source_refs", "text"), ("source_refs", [1])):
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                row = entry("a")
                row["document"][key] = value
                self.prepare([row])

    def test_missing_and_invalid_vectors_are_never_repaired(self):
        invalid = (None, [], [1.0], [1.0, 0.0, 0.0], [True, 0.0], ["1", 0.0],
                   [float("nan"), 0.0], [float("inf"), 0.0], [0.0, 0.0], [1e308, 1e308])
        for vector in invalid:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                row = entry("a")
                row["document"]["embedding"] = vector
                self.prepare([row])
        row = entry("a")
        del row["document"]["when_to_use_embedding"]
        with self.assertRaises(ValueError):
            self.prepare([row], representations=["combined", "applicability"])

    def test_query_vectors_and_depths_are_validated(self):
        bank = self.prepare()
        for vector in ([0.0, 0.0], [1.0], [True, 0.0], [float("nan"), 0.0], [1e308, 1e308]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                exact_search(bank, vector, representation="combined", depth=20)
        for depth in (True, 0, -1, 1.5, "20", None):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                exact_search(bank, [1.0, 0.0], representation="combined", depth=depth)

    def test_probe_export_contradictions_propagate(self):
        for rows in ([], [entry("other")]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "probe"):
                self.prepare(rows, probe_id="a")
        self.assertEqual(self.prepare(probe_id="a").ids, ["a"])

    def test_probe_source_checks_selected_text_vectors_and_references(self):
        for field, value in (("when_to_use", "Changed condition"), ("content", "Changed procedure"),
                             ("embedding", [0.0, 1.0]), ("source_refs", ["changed/evidence"])):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "probe source field"):
                self.prepare(probe_id="a", probe_source={field: value})
        for source in ([], "text", {"id": "different"}, {"content": None},
                       {"source_refs": [1]}, {"embedding": [True, 0.0]}):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "probe"):
                self.prepare(probe_id="a", probe_source=source)
        self.assertEqual(self.prepare(probe_id="a", probe_source=entry("a")["document"]).ids, ["a"])
        self.assertEqual(self.prepare(probe_id="a", probe_source={}).ids, ["a"])
        self.assertEqual(self.prepare(probe_id="a", probe_source={"content_embedding": [0.0, 0.0]}).ids, ["a"])

    def test_known_transport_error_normalized_without_hiding_integrity_failures(self):
        for original in (requests.ConnectionError("offline"), requests.Timeout("timeout")):
            store = Mock(export=Mock(side_effect=original))
            with self.assertRaises(StoreSearchError) as caught:
                prepare_exact_bank(store, representations=["combined"], dimensions=2)
            self.assertIs(caught.exception.__cause__, original)
        for original in (ValueError("incomplete export"), KeyError("malformed page"),
                         RuntimeError("programming failure"), StoreSearchError("partial shards")):
            store = Mock(export=Mock(side_effect=original))
            with self.assertRaises(type(original)) as caught:
                prepare_exact_bank(store, representations=["combined"], dimensions=2)
            self.assertIs(caught.exception, original)

    def test_real_export_contract_rejects_incomplete_pages(self):
        for total, scroll in ((2, "scroll-id"), (1, None)):
            with self.subTest(total=total, scroll=scroll):
                store = ElasticsearchStore("http://example.invalid", "test")
                first = {"hits": {"total": {"relation": "eq", "value": total}, "hits": [
                    {"_id": "a", "_source": entry("a")["document"]}]}}
                if scroll:
                    first["_scroll_id"] = scroll
                store._request = Mock(side_effect=[first, {"hits": {"hits": []}}, {}])
                with self.assertRaisesRegex(ValueError, "incomplete|exact total"):
                    prepare_exact_bank(store, representations=["combined"], dimensions=2)

    def test_real_export_rejects_missing_or_nonexact_totals_and_cleans_scroll(self):
        for total in (None, True, -1, 1.0, "1", {}, {"relation": "gte", "value": 1},
                      {"relation": "eq", "value": True}, {"relation": "eq", "value": -1}):
            with self.subTest(total=total):
                store = ElasticsearchStore("http://example.invalid", "test")
                first = {"_scroll_id": "opened-scroll", "hits": {"hits": []}}
                if total is not None:
                    first["hits"]["total"] = total
                store._request = Mock(side_effect=[first, {}])
                with self.assertRaisesRegex(ValueError, "exact nonnegative total"):
                    prepare_exact_bank(store, representations=["combined"], dimensions=2)
                self.assertEqual(store._request.call_args.args, ("DELETE", "/_search/scroll"))
                self.assertEqual(store._request.call_count, 2)

    def test_real_export_accepts_integer_and_object_exact_totals(self):
        for total in (0, {"relation": "eq", "value": 0}):
            with self.subTest(total=total):
                store = ElasticsearchStore("http://example.invalid", "test")
                store._request = Mock(return_value={"hits": {"total": total, "hits": []}})
                self.assertEqual(prepare_exact_bank(store, representations=["combined"], dimensions=2).ids, [])


class ExactRetrievalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [entry(f"m{index:04d}") for index in range(225)]
        self.store = Mock()
        self.store.query.side_effect = lambda body: {"hits": {"hits": [] if not self.rows else [
            {"_id": self.rows[0]["id"], "_source": self.rows[0]["document"], "_score": 1.0}]}}
        self.store.export.side_effect = lambda: deepcopy(self.rows)
        self.llm = Mock(dimensions=2, max_retries=5, retry_delay_seconds=60)
        self.llm.embed.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
        self.llm.chat.side_effect = AssertionError("Exact control makes no chat calls")
        self.traces = []

    def run_case(self, *, k=20, meta=None):
        return retrieve("Original task", k, store=self.store, llm=self.llm,
                        meta=meta or load_preset("exact_base"), record_trace=self.traces.append)

    def test_exact_control_exports_before_embedding_and_keeps_observed_evidence(self):
        events = []
        self.store.export.side_effect = lambda: (events.append("export"), deepcopy(self.rows))[1]
        self.llm.embed.side_effect = lambda texts, **kwargs: (events.append("embedding"), [[1.0, 0.0]])[1]
        result = self.run_case(k=5)
        self.assertEqual(events, ["export", "embedding"])
        self.assertEqual(result.status, "completed")
        self.assertEqual([row["id"] for row in result.records], [f"m{index:04d}" for index in range(5)])
        observed = self.traces[-1]["bank"]["observed"]
        self.assertEqual(observed["exported_count"], 225)
        self.assertEqual(len(observed["exact_snapshot_sha256"]), 64)
        self.assertTrue(all(trace["bank"]["observed"] == observed for trace in self.traces
                            if "bank_export" in trace["progress"]["completed_stages"]))
        self.assertEqual(len(self.traces[-1]["searches"][0]["candidates"]), 225)
        self.assertEqual(len(self.traces[-1]["fusion"]["ranking"]), 225)
        self.assertEqual(len(self.traces[-1]["ranking"]), 5)
        self.store.query.assert_called_once_with({"size": 1, "query": {"match_all": {}}})

    def test_capacity_and_integrity_fail_before_paid_calls(self):
        self.rows.append(entry("m0225"))
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.run_case()
        self.llm.embed.assert_not_called()
        self.llm.chat.assert_not_called()
        self.assertEqual(self.traces[-1]["status"], "failed")
        self.assertEqual(self.traces[-1]["error"]["stage"], "bank_export")
        self.rows = [entry("a", [0.0, 0.0])]
        with self.assertRaisesRegex(ValueError, "nonzero"):
            self.run_case()
        self.llm.embed.assert_not_called()

    def test_missing_probe_id_fails_before_export_or_paid_calls(self):
        self.store.query.side_effect = None
        self.store.query.return_value = {"hits": {"hits": [{"_source": self.rows[0]["document"]}]}}
        with self.assertRaisesRegex(ValueError, "probe"):
            self.run_case()
        self.store.export.assert_not_called()
        self.llm.embed.assert_not_called()
        self.llm.chat.assert_not_called()

    def test_probe_source_contradictions_fail_before_paid_calls(self):
        for source in ({"content": "Changed text"}, {"embedding": [0.0, 1.0]}, "invalid source"):
            with self.subTest(source=source):
                self.store.query.side_effect = None
                self.store.query.return_value = {"hits": {"hits": [
                    {"_id": self.rows[0]["id"], "_source": source}]}}
                with self.assertRaisesRegex(ValueError, "probe"):
                    self.run_case()
                self.llm.embed.assert_not_called()
                self.llm.chat.assert_not_called()

    def test_exact_finite_depth_accepts_larger_bank_and_truncates_last(self):
        self.rows.append(entry("m0225"))
        meta = load_preset("exact_base")
        meta["search"].update(depth=80, max_bank_size=None)
        meta["fusion"]["candidate_limit"] = 80
        result = self.run_case(k=5, meta=meta)
        self.assertEqual(len(result.records), 5)
        self.assertEqual(self.traces[-1]["bank"]["observed"]["exported_count"], 226)
        self.assertEqual(len(self.traces[-1]["searches"][0]["candidates"]), 80)
        self.assertEqual(len(self.traces[-1]["fusion"]["ranking"]), 80)

    def test_export_transport_degrades_but_integrity_failure_propagates(self):
        self.store.export.side_effect = requests.Timeout("offline")
        result = self.run_case()
        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.error["type"], "StoreSearchError")
        self.llm.embed.assert_not_called()
        self.store.export.side_effect = ValueError("incomplete snapshot")
        with self.assertRaisesRegex(ValueError, "incomplete snapshot"):
            self.run_case()
        self.assertEqual(self.traces[-1]["status"], "failed")

    def test_empty_and_disabled_skip_export_and_paid_calls(self):
        for count, k, status in ((0, 20, "empty_memory"), (225, 0, "disabled")):
            with self.subTest(count=count, k=k):
                self.rows = [entry(f"m{i}") for i in range(count)]
                result = self.run_case(k=k)
                self.assertEqual(result.status, status)
        self.store.export.assert_not_called()
        self.llm.embed.assert_not_called()
        self.llm.chat.assert_not_called()

    def test_live_callback_preserves_exact_hash_with_bank_descriptor(self):
        from MetaWeft.src.base.pipeline import BasePipeline

        class Adapter:
            def chat(self, *args, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content=json.dumps({"guidance": "Use every page"})))])

        with TemporaryDirectory() as directory:
            pipeline = BasePipeline(self.store, directory, retrieval_meta=load_preset("exact_base"))
            result = pipeline.prepare_context({"instruction": "Original task"}, llm=Adapter(),
                                              retrieval_llm=self.llm, artifacts=directory,
                                              trace_context={"bank": {"pool_size": 225}})
            self.assertEqual(result.status, "completed")
            trace = json.loads((Path(directory) / "retrieval.json").read_text())["trace"]
            self.assertEqual(trace["bank"]["descriptor"], {"pool_size": 225})
            self.assertEqual(trace["bank"]["observed"]["exported_count"], 225)
            self.assertEqual(len(trace["bank"]["observed"]["exact_snapshot_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
