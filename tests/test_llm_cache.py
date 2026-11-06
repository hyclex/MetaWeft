"""Offline response-cache behavior; every cache and pending artifact is temporary."""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from MetaWeft.service.llm import LLMCacheError, LLMUnavailableError, ModelClient
from MetaWeft.service.llm.cache import (
    CACHE_VERSION, ResponseCache, request_key, restore_pending_responses, save_pending,
)
from MetaWeft.service.llm.config import cache_settings


def completion(text="hello"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                           provider_extra={"preserved": True})


class CacheTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "llm.sqlite3"
        self.provider = Mock()
        self.provider.chat.completions.create.return_value = completion()
        self.records = []
        self.sleep = Mock()

    def record(self, operation, request, invoke, **context):
        self.records.append({"operation": operation, **context})
        return invoke()

    def client(self, **kwargs):
        settings = {"client": self.provider, "base_url": "https://provider.invalid/v1",
                    "chat_model": "chat", "embedding_model": "embed", "dimensions": 2,
                    "cache_mode": "cache", "cache_path": self.path, "recorder": self.record,
                    "max_retries": 0, "retry_delay_seconds": 0, "sleep": self.sleep}
        settings.update(kwargs)
        return ModelClient(**settings)

    def chat(self, client, messages=None, **kwargs):
        return client.chat(messages if messages is not None else [{"role": "user", "content": "hi"}],
                           phase="test", component="unit", **kwargs)

    def test_default_no_cache_always_calls_and_populates(self):
        with patch.dict(os.environ, {}, clear=True):
            client = self.client(cache_mode=None)
        self.assertEqual(client.cache_mode, "no_cache")
        self.chat(client)
        self.provider.chat.completions.create.return_value = completion("new")
        self.chat(client)
        self.assertEqual(self.provider.chat.completions.create.call_count, 2)
        result = self.chat(self.client())
        self.assertEqual(result.choices[0].message.content, "new")
        self.assertEqual(self.provider.chat.completions.create.call_count, 2)

    def test_config_precedence_and_workspace_location(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = cache_settings(workspace=self.root)
        self.assertEqual(settings, {"cache_mode": "no_cache", "cache_ttl_seconds": 1800,
                         "cache_path": str(self.root / "raw_logs/.cache/metaweft/llm.sqlite3")})
        with patch.dict(os.environ, {"METAWEFT_LLM_CACHE_MODE": "timed_cache",
                                    "METAWEFT_LLM_CACHE_PATH": "custom.sqlite3"}):
            self.assertEqual(cache_settings(workspace=self.root)["cache_mode"], "timed_cache")
            settings = cache_settings("no_cache", "explicit.sqlite3", workspace=self.root)
            self.assertEqual(settings["cache_path"], str(self.root / "explicit.sqlite3"))
            self.assertEqual(settings["cache_mode"], "no_cache")
        with self.assertRaises(ValueError):
            self.client(cache_mode="unknown")
        self.provider.chat.completions.create.assert_not_called()

    def test_chat_roundtrip_preserves_response_and_mutation_isolation(self):
        first = self.chat(self.client())
        first.choices[0].message.content = "mutated"
        replay = self.chat(self.client())
        self.assertEqual(replay.choices[0].message.content, "hello")
        self.assertEqual(replay.usage.total_tokens, 5)
        self.assertEqual(replay.provider_extra, {"preserved": True})
        replay.choices[0].message.content = "another mutation"
        self.assertEqual(self.chat(self.client()).choices[0].message.content, "hello")
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)
        self.assertEqual([row["attempt"] for row in self.records], [1, 0, 0])
        self.assertEqual(len({row["logical_call_id"] for row in self.records}), 3)
        self.assertEqual(self.records[1]["cache"]["source_logical_call_id"], self.records[0]["logical_call_id"])

    def test_exact_request_parameters_and_endpoint_separation(self):
        client = self.client()
        self.chat(client, extra_body={"a": 1, "b": 2})
        client.chat([{"content": "hi", "role": "user"}], phase="different", component="different", step=8,
                    extra_body={"b": 2, "a": 1})
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)
        for kwargs in ({"temperature": 0.8}, {"seed": 3}, {"top_p": 0.2},
                       {"response_format": {"type": "json_object"}}, {"extra_body": {"a": 2}}):
            self.chat(client, **kwargs)
        self.chat(client, [{"role": "user", "content": "hi "}])
        self.chat(self.client(chat_model="different"))
        self.chat(self.client(base_url="https://other.invalid/v1"))
        self.assertEqual(self.provider.chat.completions.create.call_count, 9)

    def test_key_keeps_order_and_prompt_fields_but_excludes_credentials(self):
        key = lambda request: request_key("chat", "https://provider.invalid/v1", request)
        self.assertNotEqual(key({"messages": ["a", "b"]}), key({"messages": ["b", "a"]}))
        self.assertNotEqual(key({"messages": [{"password": "a"}]}), key({"messages": [{"password": "b"}]}))
        self.assertEqual(key({"api_key": "first", "timeout": 10}), key({"api_key": "second", "timeout": 20}))
        self.assertEqual(key({"extra_headers": {"Authorization": "a"}}),
                         key({"extra_headers": {"Authorization": "b"}}))

    def test_sdk_iterables_and_omitted_arguments_share_the_dispatched_request(self):
        from openai import NOT_GIVEN, omit

        client = self.client()
        messages = ({"role": "user", "content": "hi"} for _ in range(1))
        self.chat(client, messages, seed=omit, top_p=NOT_GIVEN)
        sent = self.provider.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["messages"], [{"role": "user", "content": "hi"}])
        self.assertNotIn("seed", sent)
        self.assertNotIn("top_p", sent)
        self.chat(client)
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)

    def test_timed_cache_exact_expiry_and_unlimited_cache_age(self):
        client = self.client(cache_mode="timed_cache")
        with patch("MetaWeft.service.llm.client.time.time", return_value=100):
            self.chat(client)
        with patch("MetaWeft.service.llm.client.time.time", return_value=1899.999):
            self.chat(client)
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)
        with patch("MetaWeft.service.llm.client.time.time", return_value=1900):
            self.chat(client)
        self.assertEqual(self.provider.chat.completions.create.call_count, 2)
        with patch("MetaWeft.service.llm.client.time.time", return_value=9999999):
            self.chat(self.client())
        self.assertEqual(self.provider.chat.completions.create.call_count, 2)

    def test_failed_refresh_retains_old_response_without_returning_it(self):
        with patch("MetaWeft.service.llm.client.time.time", return_value=100):
            self.chat(self.client())
        self.provider.chat.completions.create.side_effect = TimeoutError("offline")
        with patch("MetaWeft.service.llm.client.time.time", return_value=2000), self.assertRaises(LLMUnavailableError):
            self.chat(self.client(cache_mode="timed_cache"))
        self.assertEqual(self.chat(self.client()).choices[0].message.content, "hello")
        self.assertEqual(self.provider.chat.completions.create.call_count, 2)

    def test_embedding_roundtrip_order_dimensions_and_input_order(self):
        self.provider.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=1, embedding=[0.3, 0.4]), SimpleNamespace(index=0, embedding=[0.1, 0.2])],
            usage={"total_tokens": 8})
        self.assertEqual(self.client().embed(["a", "b"], phase="test", component="embed"), [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(self.client().embed(["a", "b"], phase="another", component="other"), [[0.1, 0.2], [0.3, 0.4]])
        self.client().embed(["b", "a"], phase="test", component="embed")
        self.assertEqual(self.provider.embeddings.create.call_count, 2)
        self.provider.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])
        self.client(dimensions=1).embed(["a"], phase="test", component="embed")
        self.assertEqual(self.provider.embeddings.create.call_count, 3)

    def test_failures_and_invalid_responses_are_not_cached(self):
        client = self.client()
        self.provider.chat.completions.create.side_effect = [TimeoutError("failed"), completion(" "), completion()]
        for _ in range(2):
            with self.assertRaises(LLMUnavailableError):
                self.chat(client)
        self.chat(client)
        self.chat(client)
        self.assertEqual(self.provider.chat.completions.create.call_count, 3)

    def test_transient_read_retries_before_calling_provider(self):
        client = self.client(max_retries=2, retry_delay_seconds=0.25)
        with patch.object(client.cache, "get", side_effect=[sqlite3.OperationalError("busy"), None]) as read:
            self.chat(client)
        self.assertEqual(read.call_count, 2)
        self.sleep.assert_called_once_with(0.25)
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)

    def test_corrupt_database_stops_before_provider(self):
        self.path.write_bytes(b"not a SQLite database")
        with self.assertRaises(LLMCacheError) as caught:
            self.chat(self.client(max_retries=2))
        self.assertEqual(caught.exception.operation, "read")
        self.assertEqual(caught.exception.attempts, 3)
        self.assertEqual(self.sleep.call_count, 2)
        self.provider.chat.completions.create.assert_not_called()

    def test_transient_write_retries_only_storage(self):
        client = self.client(max_retries=2)
        with patch.object(client.cache, "put", side_effect=[sqlite3.OperationalError("busy"), None]) as write:
            self.chat(client)
        self.assertEqual(write.call_count, 2)
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)

    def test_exhausted_write_preserves_response_and_recovery_reuses_it(self):
        client = self.client(max_retries=1)
        with patch("MetaWeft.service.llm.client.time.time", return_value=100), \
             patch.object(client.cache, "put", side_effect=sqlite3.OperationalError("unavailable")), \
             self.assertRaises(LLMCacheError) as caught:
            self.chat(client)
        error = caught.exception
        self.assertEqual(error.operation, "write")
        pending = Path(error.pending_response_path)
        self.assertEqual(json.loads(pending.read_text())["created_at"], 100)
        self.assertEqual(self.chat(self.client()).choices[0].message.content, "hello")
        self.assertTrue(pending.exists())
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)

    def test_pending_embedding_keeps_full_vectors(self):
        client = self.client()
        self.provider.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])], usage={"total_tokens": 2})
        with patch.object(client.cache, "put", side_effect=OSError("disk")), self.assertRaises(LLMCacheError):
            client.embed(["a"], phase="embedding", component="test")
        self.assertEqual(self.client().embed(["a"], phase="embedding", component="test"), [[0.1, 0.2]])
        self.assertEqual(self.provider.embeddings.create.call_count, 1)

    def test_custom_pending_directory_is_restored_automatically(self):
        client = self.client(cache_recovery_dir=self.root / "custom-pending")
        with patch.object(client.cache, "put", side_effect=OSError("database")), self.assertRaises(LLMCacheError):
            self.chat(client)
        self.chat(self.client(cache_recovery_dir=self.root / "custom-pending"))
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)

    def test_unreadable_pending_directory_retries_and_stops_before_provider(self):
        pending = self.root / "custom-pending"
        pending.mkdir()
        client = self.client(cache_recovery_dir=pending, max_retries=2)
        with patch("MetaWeft.service.llm.cache.os.scandir", side_effect=PermissionError("unreadable")), \
             self.assertRaises(LLMCacheError) as caught:
            self.chat(client)
        self.assertEqual(caught.exception.operation, "discover")
        self.assertEqual(caught.exception.attempts, 3)
        self.assertEqual(self.sleep.call_count, 2)
        self.provider.chat.completions.create.assert_not_called()

    def test_recovery_respects_no_cache_and_timed_cache(self):
        for mode in ("no_cache", "timed_cache"):
            with self.subTest(mode=mode):
                client = self.client(cache_path=self.root / mode / "llm.sqlite3")
                with patch("MetaWeft.service.llm.client.time.time", return_value=100), \
                     patch.object(client.cache, "put", side_effect=OSError("disk")), self.assertRaises(LLMCacheError):
                    self.chat(client)
                calls = self.provider.chat.completions.create.call_count
                with patch("MetaWeft.service.llm.client.time.time", return_value=2000):
                    self.chat(self.client(cache_path=client.cache_path, cache_mode=mode))
                self.assertEqual(self.provider.chat.completions.create.call_count, calls + 1)

    def test_failed_pending_save_reports_that_response_was_not_preserved(self):
        client = self.client()
        with patch.object(client.cache, "put", side_effect=OSError("database unavailable")), \
             patch("MetaWeft.service.llm.client.save_pending", side_effect=OSError("artifacts unavailable")), \
             self.assertRaisesRegex(LLMCacheError, "could not be preserved") as caught:
            self.chat(client)
        self.assertIsNone(caught.exception.pending_response_path)
        self.assertIn("artifacts unavailable", caught.exception.preservation_error)
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)

    def test_pending_restore_never_overwrites_newer_entry(self):
        client = self.client()
        self.chat(client)
        key = self.records[0]["cache"]["key"]
        current = client.cache.get(key)
        old = {**current, "created_at": current["created_at"] - 10, "response": {"old": True}}
        save_pending(self.root / "archived/llm_cache_pending", old)
        for _ in range(2):
            self.assertEqual(restore_pending_responses(self.root, cache_path=self.path,
                             max_retries=0, retry_delay_seconds=0), 1)
        self.assertEqual(client.cache.get(key), current)

    def test_hit_recorder_failure_does_not_call_provider(self):
        self.chat(self.client())
        client = self.client(recorder=Mock(side_effect=RuntimeError("logging failed")), max_retries=2)
        with self.assertRaisesRegex(RuntimeError, "logging failed"):
            self.chat(client)
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)
        self.sleep.assert_not_called()

    def test_concurrent_process_writes_and_cross_process_reads(self):
        script = """import json, sys
from MetaWeft.service.llm.cache import ResponseCache
ResponseCache(sys.argv[1]).put(json.loads(sys.argv[2]))
"""
        entry = {"version": CACHE_VERSION, "operation": "chat", "key": "a" * 64,
                 "created_at": 1, "source_logical_call_id": "source", "response": {"value": "complete"}}
        processes = [subprocess.Popen([sys.executable, "-B", "-c", script, str(self.path),
                                       json.dumps({**entry, "created_at": stamp})],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                     for stamp in range(1, 5)]
        for process in processes:
            _, error = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, error.decode())
        self.assertEqual(ResponseCache(self.path).get(entry["key"])["created_at"], 4)


if __name__ == "__main__":
    unittest.main()
