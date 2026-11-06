"""Offline contracts for the shared retrieval component and prepared-bank helpers."""

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
from MetaWeft.service.llm import LLMUnavailableError, ModelClient
from MetaWeft.src.base import memory_index_body as base_mapping
from MetaWeft.src.memory_retrieval import (
    PRESETS, REPRESENTATIONS, fuse, load_preset, memory_index_body,
    prompt_manifest, required_representations, retrieve, validate_meta,
)


def response(value):
    content = value if isinstance(value, str) else json.dumps(value)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def hit(identifier, score=0.7):
    return {"_id": identifier, "_score": score, "_source": {
        "id": identifier, "when_to_use": "When paging through a list", "content": "Read each page",
        "source_refs": ["opaque/evidence.json"], "embedding": [1.0, 0.0],
    }}


class FakeStore:
    def __init__(self, lists=None, *, empty=False, events=None):
        self.lists = deepcopy(lists) if lists is not None else [[hit("a"), hit("b")]]
        self.empty = empty
        self.requests = []
        self.events = events if events is not None else []

    def query(self, body):
        self.requests.append(deepcopy(body))
        if "knn" not in body:
            self.events.append("probe")
            return {"hits": {"hits": [] if self.empty else [hit("a")]}}
        self.events.append("search")
        hits = self.lists.pop(0) if len(self.lists) > 1 else self.lists[0]
        if isinstance(hits, BaseException):
            raise hits
        return {"hits": {"hits": hits}}


class FakeLLM:
    def __init__(self, generated=(), *, events=None):
        self.generated = list(generated)
        self.chats, self.embeddings = [], []
        self.events = events if events is not None else []

    def chat(self, messages, **controls):
        self.events.append("generation")
        self.chats.append({"messages": deepcopy(messages), **controls})
        value = self.generated.pop(0)
        if isinstance(value, BaseException):
            raise value
        return response(value)

    def embed(self, texts, **controls):
        self.events.append("embedding")
        self.embeddings.append({"texts": list(texts), **controls})
        return [[float(index + 1), 0.0] for index in range(len(texts))]


class RetrievalTests(unittest.TestCase):
    def run_retrieval(self, *, preset="base", task="Read all pages", k=5, store=None, llm=None, meta=None):
        self.traces = []
        self.store = store if store is not None else FakeStore()
        self.llm = llm if llm is not None else FakeLLM()
        self.result = retrieve(task, k, store=self.store, llm=self.llm,
                               meta=meta if meta is not None else load_preset(preset),
                               record_trace=self.traces.append)
        return self.result

    def test_baseline_exact_text_vector_request_order_and_no_vectors_in_results(self):
        task = "  Read\n all pages  "
        result = self.run_retrieval(task=task, store=FakeStore([[hit("z", 0.8), hit("a", 0.8)]]))
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.llm.chats, [])
        self.assertEqual(self.llm.embeddings[0]["texts"], [task])
        self.assertEqual(self.store.requests, [
            {"size": 1, "query": {"match_all": {}}},
            {"size": 5, "knn": {"field": "embedding", "query_vector": [1.0, 0.0],
                                 "k": 5, "num_candidates": 10}},
        ])
        self.assertEqual([record["id"] for record in result.records], ["z", "a"])
        self.assertEqual(set(result.records[0]), {"id", "when_to_use", "content", "source_refs"})
        self.assertEqual(self.traces[-1]["queries"][0]["text"], task)
        self.assertEqual(self.traces[-1]["ranking"][0]["score"], 1 / 61)

    def test_disabled_and_empty_bank_do_not_generate_embed_or_rank_search(self):
        for k, empty, expected in ((0, False, "disabled"), (5, True, "empty_memory")):
            with self.subTest(status=expected):
                result = self.run_retrieval(k=k, preset="setting_4", store=FakeStore(empty=empty))
                self.assertEqual(result.status, expected)
                self.assertEqual(result.records, [])
                self.assertIsNone(result.error)
                self.assertEqual((self.llm.chats, self.llm.embeddings), ([], []))
                self.assertEqual(len(self.store.requests), 0 if k == 0 else 1)
                self.assertEqual(self.traces[-1]["searches"], [])

    def test_setting_four_generates_first_then_batches_five_texts_and_searches_six_lists(self):
        events = []
        llm = FakeLLM([{"queries": ["general one", "general two"]},
                       {"queries": ["when one", "when two"]}], events=events)
        self.run_retrieval(preset="setting_4", store=FakeStore(events=events), llm=llm)
        self.assertEqual(events, ["probe", "generation", "generation", "embedding", *["search"] * 6])
        self.assertEqual(llm.embeddings[0]["texts"],
                         ["Read all pages", "general one", "general two", "when one", "when two"])
        searches = self.store.requests[1:]
        self.assertEqual([item["knn"]["field"] for item in searches],
                         ["embedding"] * 3 + ["when_to_use_embedding"] * 3)
        self.assertEqual([item["knn"]["query_vector"][0] for item in searches], [1, 2, 3, 1, 4, 5])
        trace = self.traces[-1]
        self.assertEqual(trace["progress"]["completed_searches"], 6)
        self.assertEqual([search["embedding_index"] for search in trace["searches"]], [0, 1, 2, 0, 3, 4])
        self.assertEqual(trace["embedding"]["inputs"][0]["query_ids"], ["general.q0", "applicability.q0"])
        self.assertEqual(trace["ranking"][0]["score"], math.fsum([1 / 61] * 6))
        self.assertEqual(len(trace["ranking"][0]["contributions"]), 6)
        self.assertTrue(any(len(snapshot["generations"]) == 1
                            and snapshot["generations"][0]["status"] == "completed"
                            and not snapshot["searches"] for snapshot in self.traces))

    def test_settings_one_and_three_share_generation_controls_but_not_field(self):
        requests = []
        for preset, field in (("setting_1", "embedding"), ("setting_2", "when_to_use_embedding"),
                              ("setting_3", "content_embedding")):
            self.run_retrieval(preset=preset, llm=FakeLLM([{"queries": ["one", "two"]}]))
            self.assertEqual([item["knn"]["field"] for item in self.store.requests[1:]], [field] * 3)
            requests.append(self.llm.chats[0])
        self.assertEqual(requests[0], requests[2])
        self.assertNotEqual(requests[0]["messages"][0], requests[1]["messages"][0])

    def test_prompt_payload_contains_only_task_and_count(self):
        self.run_retrieval(preset="setting_1", llm=FakeLLM([{"queries": ["one", "two"]}]))
        request = self.llm.chats[0]
        self.assertEqual(json.loads(request["messages"][1]["content"]),
                         {"task": "Read all pages", "generated_count": 2})
        self.assertEqual({key: request[key] for key in ("temperature", "seed", "extra_body")},
                         {"temperature": 0.0, "seed": 0, "extra_body": {"enable_thinking": False}})

    def test_generated_count_one_and_three_are_supported(self):
        for count in (1, 3):
            meta = load_preset("setting_1")
            meta["query_generation"]["branches"][0]["generated_count"] = count
            self.run_retrieval(meta=meta, llm=FakeLLM([{"queries": [f"query {i}" for i in range(count)]}]))
            self.assertEqual(len(self.store.requests), count + 2)

    def test_optimized_presets_change_only_system_prompt(self):
        requests, searches = [], []
        for preset in ("setting_2", "setting_5", "setting_6"):
            self.run_retrieval(preset=preset, llm=FakeLLM([{"queries": ["one", "two"]}]))
            request = deepcopy(self.llm.chats[0])
            prompt = request["messages"].pop(0)["content"]
            self.assertEqual(prompt, next(iter(prompt_manifest(load_preset(preset)).values()))["text"])
            self.assertEqual(json.loads(request["messages"][0]["content"]),
                             {"task": "Read all pages", "generated_count": 2})
            requests.append(request)
            searches.append(self.store.requests)
            self.assertEqual(self.llm.embeddings[0]["texts"], ["Read all pages", "one", "two"])
        self.assertEqual(requests, [requests[0]] * 3)
        self.assertEqual(searches, [searches[0]] * 3)

    def test_setting_six_is_exact_setting_five_plus_three_examples(self):
        fifth = next(iter(prompt_manifest(load_preset("setting_5")).values()))["text"]
        sixth = next(iter(prompt_manifest(load_preset("setting_6")).values()))["text"]
        self.assertTrue(sixth.startswith(fifth + "\nDemonstrations"))
        self.assertNotIn("\nInput:", fifth)
        self.assertEqual(sixth[len(fifth):].count("\nInput:"), 3)
        self.assertEqual(sixth[len(fifth):].count("\nOutput:"), 3)

    def test_deduplication_preserves_original_case_and_does_not_refill(self):
        meta = load_preset("setting_1")
        meta["query_generation"]["branches"][0]["generated_count"] = 3
        task = "  Read\nall pages  "
        self.run_retrieval(task=task, meta=meta,
                           llm=FakeLLM([{"queries": [" Read all pages ", "read all pages", "read  all pages"]}]))
        self.assertEqual(self.llm.embeddings[0]["texts"], [task, "read all pages"])
        self.assertEqual(len(self.llm.chats), 1)
        self.assertEqual([item["duplicate_of"] for item in self.traces[-1]["queries"]],
                         [None, "general.q0", None, "general.q2"])

    def test_same_query_representation_is_searched_once_across_branches(self):
        meta = load_preset("setting_4")
        meta["search"]["routes"][1]["representation"] = "combined"
        self.run_retrieval(meta=meta, llm=FakeLLM([
            {"queries": ["same text", "first unique"]},
            {"queries": ["same  text", "second unique"]},
        ]))
        self.assertEqual(len(self.store.requests), 5)
        self.assertEqual(self.llm.embeddings[0]["texts"],
                         ["Read all pages", "same text", "first unique", "second unique"])
        skipped = self.traces[-1]["progress"]["skipped_searches"]
        self.assertEqual([item["query_id"] for item in skipped], ["applicability.q0", "applicability.q1"])

    def test_small_and_empty_ranked_lists_are_not_padded(self):
        for values in ([hit("a")], []):
            result = self.run_retrieval(store=FakeStore([values]))
            self.assertEqual(result.status, "completed")
            self.assertEqual(len(result.records), len(values))

    def test_optional_source_references_are_not_required_or_dereferenced(self):
        candidate = hit("a")
        del candidate["_source"]["source_refs"]
        result = self.run_retrieval(store=FakeStore([[candidate]]))
        self.assertEqual(result.status, "completed")
        self.assertNotIn("source_refs", result.records[0])

    def test_invalid_generated_json_schema_count_and_text_stop_without_repair(self):
        values = ["not json", "```json\n{}\n```", [], {"queries": ["one"]},
                  {"queries": ["one", "two"], "extra": True}, {"queries": ["one", " "]},
                  {"queries": ["one", 2]}, {"queries": "one"},
                  '{"queries": ["one", "two"], "queries": ["three", "four"]}']
        for value in values:
            with self.subTest(value=value):
                result = self.run_retrieval(preset="setting_1", llm=FakeLLM([value]))
                self.assertEqual(result.status, "degraded")
                self.assertEqual(result.error["stage"], "query_generation")
                self.assertEqual(result.records, [])
                self.assertEqual(len(self.llm.chats), 1)
                self.assertEqual(self.llm.embeddings, [])
                self.assertEqual(len(self.store.requests), 1)

    def test_failure_in_second_branch_preserves_first_and_never_embeds(self):
        result = self.run_retrieval(preset="setting_4", llm=FakeLLM([
            {"queries": ["one", "two"]}, {"queries": []},
        ]))
        self.assertEqual(result.status, "degraded")
        self.assertEqual(self.traces[-1]["progress"]["completed_branches"], ["general"])
        self.assertEqual(len(self.traces[-1]["queries"]), 3)
        self.assertEqual(self.llm.embeddings, [])

    def test_bad_search_hits_are_not_votes(self):
        invalid = [[hit("a"), hit("a")], [None]]
        for change in ({"_id": ""}, {"_score": None}, {"_score": float("nan")},
                       {"_score": True}, {"_source": {}},
                       {"_source": {**hit("a")["_source"], "id": "other"}},
                       {"_source": {**hit("a")["_source"], "source_refs": "not-list"}}):
            invalid.append([{**hit("a"), **change}])
        for values in invalid:
            with self.subTest(hits=values):
                result = self.run_retrieval(store=FakeStore([values]))
                self.assertEqual(result.status, "degraded")
                self.assertEqual(result.error["stage"], "search")
                self.assertEqual(result.records, [])
                self.assertEqual(self.traces[-1]["ranking"], [])

    def test_failed_later_search_preserves_prior_lists_but_returns_no_records(self):
        self.run_retrieval(preset="setting_1", llm=FakeLLM([{"queries": ["one", "two"]}]),
                           store=FakeStore([[hit("a")], StoreSearchError("search unavailable")]))
        self.assertEqual(self.result.status, "degraded")
        self.assertEqual(self.result.records, [])
        self.assertEqual(self.traces[-1]["progress"]["completed_searches"], 1)
        self.assertEqual(self.traces[-1]["searches"][0]["candidates"][0]["id"], "a")
        self.assertEqual(self.traces[-1]["searches"][-1]["status"], "failed")
        self.assertEqual(len(self.store.requests), 3)

    def test_malformed_probe_is_degraded_without_model_calls(self):
        for invalid in ({}, {"hits": {"hits": "wrong"}}, {"timed_out": True, "hits": {"hits": []}}):
            with self.subTest(response=invalid):
                store = Mock(query=Mock(return_value=invalid))
                self.run_retrieval(store=store)
                self.assertEqual(self.result.status, "degraded")
                self.assertEqual(self.result.error["stage"], "empty_bank_probe")
                self.assertEqual(self.llm.embeddings, [])

    def test_provider_exhaustion_propagates_with_partial_trace(self):
        failure = LLMUnavailableError(operation="chat", phase="retrieval", logical_call_id="call",
                                      attempts=6, cause=RuntimeError("provider down"))
        llm = FakeLLM([{"queries": ["one", "two"]}, failure])
        traces = []
        with self.assertRaises(LLMUnavailableError) as raised:
            retrieve("task", 5, store=FakeStore(), llm=llm, meta=load_preset("setting_4"),
                     record_trace=traces.append)
        self.assertIs(raised.exception, failure)
        self.assertEqual(traces[-1]["status"], "failed")
        self.assertEqual(traces[-1]["generations"][-1]["component"], "retrieval.query_generation.applicability")
        self.assertEqual(traces[-1]["generations"][-1]["status"], "failed")
        self.assertEqual(len(traces[-1]["queries"]), 3)

    def test_embedding_exhaustion_trace_retains_batch_inputs(self):
        failure = LLMUnavailableError(operation="embedding", phase="retrieval", logical_call_id="call",
                                      attempts=6, cause=RuntimeError("provider down"))
        llm = FakeLLM()
        llm.embed = Mock(side_effect=failure)
        traces = []
        with self.assertRaises(LLMUnavailableError):
            retrieve("task", 5, store=FakeStore(), llm=llm, meta=load_preset(), record_trace=traces.append)
        self.assertEqual(traces[-1]["embedding"]["inputs"], [{"index": 0, "text": "task", "query_ids": ["identity.q0"]}])
        self.assertEqual(traces[-1]["embedding"]["status"], "failed")

    def test_artifact_errors_propagate_unchanged_without_extra_work(self):
        for failure in (OSError("disk full"), StoreSearchError("recorder error resembles store error")):
            traces, store, llm = [], FakeStore(), FakeLLM()
            def record(trace):
                traces.append(trace)
                if trace["progress"]["completed_searches"]:
                    raise failure
            with self.subTest(error=type(failure).__name__), self.assertRaises(type(failure)) as raised:
                retrieve("task", 5, store=store, llm=llm, meta=load_preset(), record_trace=record)
            self.assertIs(raised.exception, failure)
            self.assertEqual(len(store.requests), 2)
            self.assertEqual(traces[-1]["ranking"], [])

    def test_programming_errors_are_fatal_and_traced(self):
        store, traces = Mock(query=Mock(side_effect=RuntimeError("unexpected bug"))), []
        with self.assertRaisesRegex(RuntimeError, "unexpected bug"):
            retrieve("task", 5, store=store, llm=FakeLLM(), meta=load_preset(), record_trace=traces.append)
        self.assertEqual(traces[-1]["status"], "failed")

    def test_recorder_error_that_resembles_storage_failure_is_fatal(self):
        llm, traces = FakeLLM(), []
        llm.embed = Mock(side_effect=StoreSearchError("recorder failure"))
        with self.assertRaises(StoreSearchError):
            retrieve("task", 5, store=FakeStore(), llm=llm, meta=load_preset(), record_trace=traces.append)
        self.assertEqual(traces[-1]["status"], "failed")

    def test_transient_provider_failure_retries_and_keeps_logical_component_identity(self):
        cache_directory = TemporaryDirectory()
        self.addCleanup(cache_directory.cleanup)
        attempts, sleep = [], Mock()
        provider = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=[
                RuntimeError("transient"), response({"queries": ["one", "two"]}),
            ]))),
            embeddings=SimpleNamespace(create=Mock(return_value=SimpleNamespace(data=[
                SimpleNamespace(index=i, embedding=[float(i + 1), 0.0]) for i in range(3)
            ]))),
        )
        def record(operation, request, invoke, **context):
            attempts.append({"operation": operation, **context})
            return invoke()
        llm = ModelClient(client=provider, chat_model="test", dimensions=2, recorder=record,
                          cache_mode="no_cache", cache_path=Path(cache_directory.name) / "llm.sqlite3",
                          max_retries=1, retry_delay_seconds=0, sleep=sleep)
        result = self.run_retrieval(preset="setting_1", llm=llm)
        self.assertEqual(result.status, "completed")
        self.assertEqual([item["attempt"] for item in attempts], [1, 2, 1])
        self.assertEqual(attempts[0]["logical_call_id"], attempts[1]["logical_call_id"])
        self.assertEqual(attempts[0]["component"], "retrieval.query_generation.general")
        self.assertEqual(attempts[-1]["component"], "retrieval.embedding.batch_0")
        sleep.assert_called_once_with(0)


class ConfigurationAndFusionTests(unittest.TestCase):
    def test_presets_and_representation_metadata(self):
        from evaluations.appworld_mem_retrieval.tool.common import EMBEDDING_INPUTS
        for name in PRESETS:
            meta = load_preset(name, num_candidates=40)
            self.assertEqual(validate_meta(meta, k=20), meta)
        for definition in REPRESENTATIONS.values():
            self.assertEqual(definition["input"], EMBEDDING_INPUTS[definition["dataset_field"]])
        self.assertEqual(required_representations(load_preset("setting_4")), ["combined", "applicability"])
        self.assertEqual(memory_index_body(), base_mapping())
        fields = memory_index_body(2, representations=["applicability"])["mappings"]["properties"]
        self.assertNotIn("embedding", fields)
        self.assertEqual(fields["when_to_use_embedding"]["dims"], 2)

    def test_prompt_manifest_has_exact_bytes_and_hashes(self):
        import hashlib
        first = prompt_manifest(load_preset("setting_1"))
        third = prompt_manifest(load_preset("setting_3"))
        self.assertEqual(first, third)
        self.assertEqual(prompt_manifest(load_preset()), {})
        for definition in prompt_manifest(load_preset("setting_4")).values():
            self.assertEqual(definition["sha256"], hashlib.sha256(definition["text"].encode()).hexdigest())

    def test_invalid_arguments_and_meta_fail_before_model_or_store_calls(self):
        invalid = []
        for key, value in (("schema_version", True), ("name", ""), ("description", ""), ("unknown", 1)):
            meta = load_preset()
            meta[key] = value
            invalid.append(meta)
        changes = [
            ("query_generation", "branches", []),
            ("query_generation", "branches", [{"id": "x", "method": "unknown"}]),
            ("query_generation", "branches", [{"id": "identity", "method": "identity", "prompt": "general_v1"}]),
            ("query_generation", "branches", [{"id": "identity", "method": "identity"}] * 2),
            ("search", "num_candidates", True), ("search", "num_candidates", 10001),
            ("search", "routes", [{"branch": "missing", "representation": "combined"}]),
            ("search", "routes", [{"branch": "identity", "representation": "unknown"}]),
            ("search", "routes", [{"branch": "identity", "representation": "combined"}] * 2),
            ("fusion", "method", "other"), ("fusion", "rank_constant", 0),
        ]
        for section, key, value in changes:
            meta = load_preset()
            meta[section][key] = value
            invalid.append(meta)
        for key, value in (("generated_count", 0), ("generated_count", True), ("generated_count", 4),
                           ("prompt", "missing"), ("prompt", [])):
            meta = load_preset("setting_1")
            meta["query_generation"]["branches"][0][key] = value
            invalid.append(meta)
        meta = load_preset("setting_4")
        meta["search"]["routes"] = meta["search"]["routes"][:1]
        invalid.append(meta)
        for meta in invalid:
            store, llm, callback = Mock(), Mock(), Mock()
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                retrieve("task", 5, store=store, llm=llm, meta=meta, record_trace=callback)
            store.query.assert_not_called()
            llm.embed.assert_not_called()
            llm.chat.assert_not_called()
            callback.assert_not_called()
        for task, k in (("", 5), (" ", 5), (None, 5), ("task", True), ("task", -1),
                        ("task", 1.5), ("task", 11), ("task", None), ("task", "5")):
            with self.subTest(task=task, k=k), self.assertRaises(ValueError):
                retrieve(task, k, store=Mock(), llm=Mock(), meta=load_preset(), record_trace=Mock())

    def test_configuration_name_does_not_select_behavior_or_share_mutable_state(self):
        meta = load_preset("setting_2")
        meta["name"] = "base"
        self.assertEqual(required_representations(meta), ["applicability"])
        meta["search"]["num_candidates"] = 99
        self.assertEqual(load_preset("setting_2")["search"]["num_candidates"], 10)

    def test_rrf_hand_calculation_ties_and_unique_final_ids(self):
        searches = [{"id": "one", "candidates": [{"id": "b", "rank": 1}, {"id": "a", "rank": 2},
                                                   {"id": "c", "rank": 3}]},
                    {"id": "two", "candidates": [{"id": "a", "rank": 1}, {"id": "b", "rank": 2}]}]
        ranked = fuse(searches, k=3)
        self.assertEqual([item["id"] for item in ranked], ["a", "b", "c"])
        self.assertEqual(ranked[0]["score"], math.fsum([1 / 62, 1 / 61]))
        self.assertEqual(ranked[1]["score"], ranked[0]["score"])
        self.assertEqual(ranked[2]["score"], 1 / 63)
        self.assertEqual(fuse(searches, k=1), ranked[:1])
        self.assertEqual(fuse([], k=5), [])

    def test_rrf_best_rank_precedes_lexicographic_tie(self):
        # With constant=1: (1/2 + 1/8) and (1/4 + 1/4 + 1/8) tie exactly.
        searches = [{"id": "one", "candidates": [{"id": "z", "rank": 1}, {"id": "a", "rank": 3}]},
                    {"id": "two", "candidates": [{"id": "a", "rank": 3}, {"id": "z", "rank": 7}]},
                    {"id": "three", "candidates": [{"id": "a", "rank": 7}]}]
        ranked = fuse(searches, k=2, rank_constant=1)
        self.assertEqual(ranked[0]["score"], ranked[1]["score"])
        self.assertEqual([item["id"] for item in ranked], ["z", "a"])

    def test_search_service_wraps_transport_invalid_json_and_incomplete_responses(self):
        session = Mock()
        store = ElasticsearchStore("http://localhost:9200", "test", session=session)
        for failure in (requests.Timeout("timeout"), requests.HTTPError("http error")):
            session.request.side_effect = failure
            with self.assertRaises(StoreSearchError):
                store.query({"size": 1})
        session.request.side_effect = None
        session.request.return_value.json.side_effect = ValueError("invalid JSON")
        with self.assertRaises(StoreSearchError):
            store.query({"size": 1})
        session.request.return_value.json.side_effect = None
        for value in ([], {"_shards": []}, {"timed_out": True}, {"_shards": {"failed": 1}}):
            session.request.return_value.json.return_value = value
            with self.assertRaises(StoreSearchError):
                store.query({"size": 1})


if __name__ == "__main__":
    unittest.main()
