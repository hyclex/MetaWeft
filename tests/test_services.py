"""Offline service checks; provider and Elasticsearch transports are fakes."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import requests
from openai import BadRequestError, RateLimitError

from MetaWeft.service.elasticsearch import ElasticsearchStore, StoreWriteError
from MetaWeft.service.llm import LLMUnavailableError, ModelClient


def provider_error(kind, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://provider.invalid/v1"))
    return kind("test provider error", response=response, body={})


class RecordingCalls:
    def __init__(self):
        self.calls = []

    def __call__(self, operation, request, invoke, **context):
        record = {"operation": operation, "request": request, **context}
        self.calls.append(record)
        try:
            record["response"] = invoke()
            return record["response"]
        except Exception as error:
            record["error"] = error
            raise


class ModelTests(unittest.TestCase):
    def setUp(self):
        cache_directory = TemporaryDirectory()
        self.addCleanup(cache_directory.cleanup)
        self.provider = Mock()
        self.recorder = RecordingCalls()
        self.sleep = Mock()
        self.client = ModelClient(
            client=self.provider, chat_model="chat-test", embedding_model="embed-test",
            dimensions=2, recorder=self.recorder, sleep=self.sleep,
            cache_mode="no_cache", cache_path=Path(cache_directory.name) / "llm.sqlite3",
        )

    def test_retry_attempts_share_identity_and_preserve_usage(self):
        result = SimpleNamespace(usage={"total_tokens": 17},
                                 choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        self.provider.chat.completions.create.side_effect = [
            provider_error(RateLimitError, 429), provider_error(RateLimitError, 429), result,
        ]
        self.assertIs(self.client.chat(
            [{"role": "user", "content": "test"}], phase="adapt", component="model", step=2,
        ), result)
        self.assertEqual([call["attempt"] for call in self.recorder.calls], [1, 2, 3])
        self.assertEqual(len({call["logical_call_id"] for call in self.recorder.calls}), 1)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [60, 60])
        self.assertEqual(self.recorder.calls[-1]["response"].usage, {"total_tokens": 17})
        self.assertEqual(self.provider.chat.completions.create.call_args.kwargs["timeout"], 120)

    def test_exhaustion_has_six_attempts_and_preserves_cause(self):
        self.provider.chat.completions.create.side_effect = provider_error(RateLimitError, 429)
        with self.assertRaises(LLMUnavailableError) as caught:
            self.client.chat([], phase="execute", component="model")
        self.assertEqual(self.provider.chat.completions.create.call_count, 6)
        self.assertEqual(len(self.recorder.calls), 6)
        self.assertEqual(self.sleep.call_count, 5)
        self.assertIsInstance(caught.exception.__cause__, RateLimitError)
        self.assertEqual(caught.exception.attempts, 6)
        self.assertEqual(caught.exception.phase, "execute")

    def test_http_errors_retry_but_recorder_errors_do_not(self):
        self.provider.chat.completions.create.side_effect = provider_error(BadRequestError, 400)
        with self.assertRaises(LLMUnavailableError):
            self.client.chat([], phase="execute", component="model")
        self.assertEqual(self.provider.chat.completions.create.call_count, 6)
        self.sleep.reset_mock()
        self.client.recorder = Mock(side_effect=RuntimeError("canonical log unavailable"))
        with self.assertRaisesRegex(RuntimeError, "canonical log unavailable"):
            self.client.chat([], phase="execute", component="model")
        self.assertEqual(self.client.recorder.call_count, 1)
        self.sleep.assert_not_called()

    def test_embedding_order_and_usage_are_preserved(self):
        result = SimpleNamespace(data=[
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ], usage={"total_tokens": 8})
        self.provider.embeddings.create.return_value = result
        self.assertEqual(self.client.embed(["a", "b"], phase="retrieve", component="embedding"),
                         [[0.1, 0.2], [0.3, 0.4]])
        self.assertIs(self.recorder.calls[0]["response"], result)
        self.assertEqual(self.provider.embeddings.create.call_args.kwargs["timeout"], 30)

    def test_embedding_validation_is_after_logging_and_is_retried(self):
        self.provider.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[1.0])], usage={"total_tokens": 2},
        )
        with self.assertRaisesRegex(LLMUnavailableError, "dimensions"):
            self.client.embed(["a"], phase="retrieve", component="embedding")
        self.assertEqual(len(self.recorder.calls), 6)
        self.assertIn("response", self.recorder.calls[0])
        self.assertEqual(self.provider.embeddings.create.call_count, 6)

    def test_empty_and_malformed_chat_responses_retry(self):
        success = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))])
        bad = [None, SimpleNamespace(choices=[]), SimpleNamespace(choices=[None])]
        bad += [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
                for text in (None, "", " \n", 12)]
        for response in bad:
            with self.subTest(response=response):
                self.provider.chat.completions.create.side_effect = [response, success]
                self.assertIs(self.client.chat([], phase="execution", component="model"), success)
                self.assertIs(self.recorder.calls[-2]["response"], response)

    def test_connection_and_decoding_exceptions_retry(self):
        for error in (TimeoutError("timeout"), ConnectionError("unreachable"), ValueError("bad JSON")):
            with self.subTest(error=error):
                self.provider.embeddings.create.side_effect = [error, SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])])]
                self.assertEqual(self.client.embed(["a"], phase="retrieval", component="model"), [[0.1, 0.2]])

    def test_zero_budget_and_custom_delay(self):
        self.client.max_retries = 0
        self.provider.chat.completions.create.return_value = None
        with self.assertRaises(LLMUnavailableError):
            self.client.chat([], phase="execution", component="model")
        self.sleep.assert_not_called()
        self.provider.chat.completions.create.reset_mock()
        self.client.max_retries, self.client.retry_delay_seconds = 2, 0.25
        with self.assertRaises(LLMUnavailableError):
            self.client.chat([], phase="execution", component="model")
        self.assertEqual(self.provider.chat.completions.create.call_count, 3)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [0.25, 0.25])

    def test_invalid_retry_configuration_and_cancellation_do_not_call_again(self):
        for retries, delay in ((-1, 0), (True, 0), (1.5, 0), (1, -1), (1, float("inf")), (1, True)):
            with self.subTest(retries=retries, delay=delay), self.assertRaises(ValueError):
                ModelClient(client=self.provider, chat_model="test", max_retries=retries, retry_delay_seconds=delay)
        for config in ({"dimensions": 0}, {"chat_timeout": -1}, {"embedding_timeout": float("nan")},
                       {"embedding_model": ""}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ModelClient(client=self.provider, chat_model="test", **config)
        self.provider.chat.completions.create.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.client.chat([], phase="execution", component="model")
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)
        self.sleep.assert_not_called()

    def test_recorder_api_error_is_not_confused_with_provider_error(self):
        self.client.recorder = Mock(side_effect=provider_error(RateLimitError, 429))
        with self.assertRaises(RateLimitError):
            self.client.chat([], phase="execution", component="model")
        self.provider.chat.completions.create.assert_not_called()
        self.sleep.assert_not_called()

    def test_empty_embedding_batch_has_no_provider_call(self):
        self.assertEqual(self.client.embed([], phase="retrieve", component="embedding"), [])
        self.provider.embeddings.create.assert_not_called()


def es_response(body, status=200):
    response = Mock(status_code=status)
    response.json.return_value = body
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    return response


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        self.store = ElasticsearchStore("http://localhost:9200", "stream-1", session=self.session)

    def test_mapping_and_query_are_caller_owned(self):
        self.session.request.return_value = es_response({"acknowledged": True})
        mapping = {"mappings": {"properties": {"neighbors": {"type": "keyword"}}}}
        self.store.create(mapping)
        self.assertEqual(self.session.request.call_args.kwargs["json"], mapping)
        query = {"query": {"match": {"custom_field": "value"}}, "size": 7}
        self.store.query(query)
        self.assertEqual(self.session.request.call_args.kwargs["json"], query)

    def test_write_confirms_full_source_and_requests_refresh(self):
        document = {"name": "memory", "neighbors": ["m2"], "vector": [0.1, 0.2]}
        self.session.request.side_effect = [
            es_response({"result": "created"}), es_response({"_source": document}),
        ]
        self.assertEqual(self.store.put("m1", document), document)
        self.assertEqual(self.session.request.call_args_list[0].kwargs["params"],
                         {"refresh": "wait_for"})

    def test_timed_out_but_committed_write_is_reconciled(self):
        document = {"content": "complete"}
        self.session.request.side_effect = [
            requests.Timeout("write timed out"), es_response({"_source": document}), es_response({}),
        ]
        self.assertEqual(self.store.put("m1", document), document)
        self.assertEqual(self.session.request.call_count, 3)
        self.assertTrue(self.session.request.call_args.args[1].endswith("/_refresh"))

    def test_unconfirmed_write_is_explicitly_uncertain(self):
        self.session.request.side_effect = [
            requests.Timeout("write timed out"), es_response({"found": False}, 404),
        ]
        with self.assertRaises(StoreWriteError) as raised:
            self.store.put("m1", {"content": "new"})
        self.assertTrue(raised.exception.uncertain)
        self.assertEqual(raised.exception.document_id, "m1")

    def test_missing_index_is_not_mistaken_for_missing_document(self):
        self.session.request.return_value = es_response({"error": {"type": "index_not_found_exception"}}, 404)
        with self.assertRaises(requests.HTTPError):
            self.store.get("m1")

    def test_delete_checks_absence(self):
        self.session.request.side_effect = [es_response({"result": "deleted"}),
                                            es_response({"found": False}, 404)]
        self.assertIsNone(self.store.delete("m1"))
        self.assertEqual(self.session.request.call_count, 2)

    def test_export_reads_all_pages_and_preserves_source(self):
        self.session.request.side_effect = [
            es_response({"_scroll_id": "s1", "hits": {"total": {"value": 2, "relation": "eq"}, "hits": [
                {"_id": "b", "_source": {"vector": [0.1, 0.2]}},
            ]}}),
            es_response({"_scroll_id": "s2", "hits": {"hits": [
                {"_id": "a", "_source": {"extra": {"graph": ["b"]}}},
            ]}}),
            es_response({"_scroll_id": "s2", "hits": {"hits": []}}),
            es_response({"succeeded": True}),
        ]
        self.assertEqual(self.store.export(page_size=1), [
            {"id": "a", "document": {"extra": {"graph": ["b"]}}},
            {"id": "b", "document": {"vector": [0.1, 0.2]}},
        ])
        self.assertEqual(self.session.request.call_count, 4)
        self.assertEqual(self.session.request.call_args.kwargs["json"], {"scroll_id": ["s2"]})

    def test_partial_export_fails_and_cleans_up_scroll(self):
        self.session.request.side_effect = [
            es_response({"_scroll_id": "s1", "hits": {"total": {"value": 1, "relation": "eq"},
                                                      "hits": [{"_id": "a", "_source": {}}]}}),
            requests.Timeout("next page unavailable"), es_response({"succeeded": True}),
        ]
        with self.assertRaises(requests.Timeout):
            self.store.export()
        self.assertEqual(self.session.request.call_args.args[0], "DELETE")

    def test_incomplete_search_is_not_a_valid_result(self):
        self.session.request.return_value = es_response({"timed_out": True, "hits": {"hits": []}})
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.store.query({"query": {"match_all": {}}})

    def test_export_checks_exact_total(self):
        self.session.request.side_effect = [
            es_response({"_scroll_id": "s1", "hits": {
                "total": {"value": 2, "relation": "eq"},
                "hits": [{"_id": "a", "_source": {}}],
            }}),
            es_response({"_scroll_id": "s1", "hits": {"hits": []}}),
            es_response({"succeeded": True}),
        ]
        with self.assertRaisesRegex(ValueError, "exact total"):
            self.store.export()


if __name__ == "__main__":
    unittest.main()
