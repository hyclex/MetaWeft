"""Offline integration boundaries for candidate selection and ranking."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from MetaWeft.src.base.pipeline import BasePipeline
from MetaWeft.src.memory_retrieval import load_preset, retrieve


class Bank:
    def __init__(self, count):
        self.hits = [{"_id": f"m{i:04d}", "_score": 1.0, "_source": {
            "id": f"m{i:04d}", "when_to_use": "Read pages", "content": f"Procedure {i}",
            "source_refs": ["evidence.json"], "embedding": [1.0, 0.0],
        }} for i in range(count)]
        self.requests = []

    def query(self, body):
        self.requests.append(deepcopy(body))
        return {"hits": {"hits": self.hits[:body["size"]]}}

    def export(self):
        raise AssertionError("HNSW must not export the bank")


class Ranker:
    dimensions = 2
    chat_model = "unchanged-adaptation-model"
    max_retries = 2
    retry_delay_seconds = 60

    def __init__(self, *, malformed=None):
        self.chats, self.embeddings = [], []
        self.malformed = malformed

    def embed(self, texts, **controls):
        self.embeddings.append((list(texts), controls))
        return [[1.0, 0.0] for _ in texts]

    def chat(self, messages, **controls):
        self.chats.append({"messages": deepcopy(messages), **controls})
        payload = json.loads(messages[1]["content"])
        count = len(payload["memories"])
        if self.malformed == controls["component"].split(".")[2]:
            value = {"wrong": []}
        elif "return_count" in payload:
            value = {"ranking": list(range(count, count - 20, -1))}
        else:
            value = {"scores": [[i, i % 101] for i in range(1, count + 1)]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])


class RetrievalV2Tests(unittest.TestCase):
    def run_case(self, preset, count=225, k=5, *, llm=None):
        meta = load_preset(preset)
        self.llm = llm or Ranker()
        self.llm.max_retries = meta["provider_policy"]["max_retries"]
        self.store, self.traces = Bank(count), []
        result = retrieve("Task bytes\n  preserved", k, store=self.store, llm=self.llm,
                          meta=meta, record_trace=self.traces.append)
        return result

    def test_final_limit_does_not_reduce_grading_input(self):
        result = self.run_case("performance_3", k=5)
        self.assertEqual(len(result.records), 5)
        self.assertEqual(self.store.requests[1]["knn"]["k"], 80)
        self.assertEqual(self.store.requests[1]["knn"]["num_candidates"], 160)
        payload = json.loads(self.llm.chats[0]["messages"][1]["content"])
        self.assertEqual(len(payload["memories"]), 80)
        self.assertEqual(payload["task"], "Task bytes\n  preserved")
        self.assertEqual([row["id"] for row in self.traces[-1]["ranking"]],
                         [row["id"] for row in result.records])
        self.assertEqual(len(self.traces[-1]["fusion"]["ranking"]), 80)

    def test_cascade_uses_two_local_index_spaces(self):
        result = self.run_case("performance_1")
        stages = self.traces[-1]["reranking"]
        self.assertEqual([stage["input_count"] for stage in stages], [225, 20])
        self.assertEqual(stages[1]["input_ids"], stages[0]["output_ids"])
        self.assertEqual([row["id"] for row in result.records],
                         [f"m{i:04d}" for i in range(205, 210)])
        second = json.loads(self.llm.chats[1]["messages"][1]["content"])
        self.assertEqual([row["index"] for row in second["memories"]], list(range(1, 21)))
        self.assertEqual(self.llm.chat_model, "unchanged-adaptation-model")
        self.assertEqual(self.llm.chats[1]["max_completion_tokens"], 6144)
        self.assertNotIn("max_tokens", self.llm.chats[1])

    def test_equivalent_categories_make_identical_requests(self):
        for first, second in (("performance_3", "transferability_1"),
                              ("efficiency_3", "transferability_2"),
                              ("performance_1", "transferability_3")):
            with self.subTest(pair=(first, second)):
                result = self.run_case(first)
                requests = deepcopy(self.llm.chats)
                other = self.run_case(second)
                self.assertEqual(result.records, other.records)
                self.assertEqual(requests, self.llm.chats)

    def test_small_banks_bypass_listwise_but_keep_grader(self):
        for count in (1, 19):
            result = self.run_case("performance_1", count=count, k=20)
            self.assertEqual(len(result.records), count)
            self.assertEqual(self.traces[-1]["reranking"][0]["status"], "skipped")
            self.assertEqual(len(self.llm.chats), 1)
            self.assertEqual(self.store.requests[1]["knn"]["k"], 225)

    def test_empty_and_disabled_avoid_paid_calls(self):
        for count, k, status in ((0, 20, "empty_memory"), (225, 0, "disabled")):
            result = self.run_case("performance_1", count=count, k=k)
            self.assertEqual(result.status, status)
            self.assertEqual(self.llm.chats, [])
            self.assertEqual(self.llm.embeddings, [])
            self.assertEqual(len(self.store.requests), int(k != 0))

    def test_bounded_hnsw_accepts_larger_banks(self):
        for count in (226, 1000):
            result = self.run_case("performance_1", count=count, k=20)
            self.assertEqual(len(result.records), 20)
            self.assertEqual(self.traces[-1]["reranking"][0]["input_count"], 225)
            self.assertEqual(len(self.store.requests), 2)
            self.assertNotIn("exact_snapshot_sha256", self.traces[-1]["bank"]["observed"])

    def test_fallback_remains_visible_after_successful_grader(self):
        result = self.run_case("performance_1", llm=Ranker(malformed="select"))
        self.assertEqual(result.status, "completed_with_fallback")
        self.assertEqual(result.error["type"], "RankingFallback")
        self.assertEqual(self.traces[-1]["reranking"][1]["status"], "completed")
        self.assertEqual(len(self.llm.chats), 3)
        self.assertTrue(any(trace["reranking"] and
                            trace["reranking"][0].get("status") == "completed_with_fallback"
                            and len(trace["reranking"]) == 1 for trace in self.traces))

    def test_provider_mismatch_rejected_before_work(self):
        store, llm, traces = Bank(20), Ranker(), []
        llm.max_retries = 5
        with self.assertRaisesRegex(ValueError, "provider_policy"):
            retrieve("task", 5, store=store, llm=llm, meta=load_preset("performance_1"),
                     record_trace=traces.append)
        self.assertEqual((store.requests, llm.chats, llm.embeddings, traces), ([], [], [], []))

    def test_live_fallback_adapts_with_separate_client(self):
        class Adapter:
            def chat(self, messages, **controls):
                self.controls = controls
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content=json.dumps({"guidance": "Use every page"})))])

        with TemporaryDirectory() as directory:
            pipeline = BasePipeline(Bank(80), directory, retrieval_meta=load_preset("performance_3"))
            adapter = Adapter()
            ranker = Ranker(malformed="grade")
            result = pipeline.prepare_context({"instruction": "task"}, llm=adapter,
                                              retrieval_llm=ranker, artifacts=directory,
                                              trace_context={"bank": {"pool_size": 80}})
            self.assertEqual(result.status, "completed_with_fallback")
            self.assertTrue(result.degraded)
            self.assertEqual(result.guidance, "Use every page")
            saved = json.loads((Path(directory) / "retrieval.json").read_text())
            self.assertEqual(saved["trace"]["bank"]["descriptor"], {"pool_size": 80})
            self.assertIn("observed", saved["trace"]["bank"])
            self.assertEqual(adapter.controls["phase"], "adaptation")


if __name__ == "__main__":
    unittest.main()
