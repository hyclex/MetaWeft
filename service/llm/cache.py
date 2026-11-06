"""Persistent exact-response storage, independent of experiment policy and logging."""

from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from types import SimpleNamespace

from .config import validate_retry


CACHE_VERSION = 1


class LLMCacheError(RuntimeError):
    """Cache retries exhausted; preserve the task boundary and stop the stream."""

    def __init__(self, *, operation, phase, key, attempts, cause, pending_response_path=None):
        super().__init__(f"LLM cache {operation} failed in {phase} after {attempts} attempts: "
                         f"{type(cause).__name__}: {cause}")
        self.operation, self.phase, self.key = operation, phase, key
        self.attempts, self.cause = attempts, cause
        self.pending_response_path = pending_response_path
        self.preservation_error = None


def plain(value):
    """JSON response data, including SDK extras and lightweight injected fakes."""
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if isinstance(value, SimpleNamespace):
        return plain(vars(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def request_key(operation, endpoint, request):
    # Transport credentials are not response inputs. Do not strip fields inside
    # messages or extra_body: those may be actual prompt/model parameters.
    body = {key: value for key, value in request.items()
            if key not in {"api_key", "authorization", "timeout", "max_retries", "retry_delay_seconds"}}
    if isinstance(body.get("extra_headers"), dict):
        body["extra_headers"] = {key: value for key, value in body["extra_headers"].items()
                                 if key.lower() not in {"authorization", "api-key", "x-api-key"}}
    payload = {"version": CACHE_VERSION, "operation": operation,
               "endpoint": endpoint.rstrip("/"), "request": plain(body)}
    return hashlib.sha256(encoded(payload).encode("utf-8")).hexdigest()


def validate_entry(entry):
    if (not isinstance(entry, dict) or entry.get("version") != CACHE_VERSION
            or entry.get("operation") not in {"chat", "embedding"}
            or not isinstance(entry.get("key"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["key"])
            or type(entry.get("created_at")) not in (int, float)
            or not math.isfinite(entry["created_at"]) or entry["created_at"] < 0
            or not isinstance(entry.get("source_logical_call_id"), str)
            or not entry["source_logical_call_id"] or not isinstance(entry.get("response"), dict)):
        raise ValueError("Invalid LLM cache entry")
    return entry


def restore_response(entry):
    from openai.types.chat import ChatCompletion
    from openai.types import CreateEmbeddingResponse

    response_type = ChatCompletion if entry["operation"] == "chat" else CreateEmbeddingResponse
    # The existing validators deliberately accept partial provider envelopes.
    return response_type.model_construct(**entry["response"])


def retry_cache(action, *, operation, phase, key, max_retries, retry_delay_seconds, sleep=time.sleep):
    for attempt in range(1, max_retries + 2):
        try:
            return action()
        except Exception as error:
            if attempt == max_retries + 1:
                raise LLMCacheError(operation=operation, phase=phase, key=key,
                                    attempts=attempt, cause=error) from error
        sleep(retry_delay_seconds)


class ResponseCache:
    def __init__(self, path):
        self.path = Path(path)

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS responses "
                               "(key TEXT PRIMARY KEY, created_at REAL NOT NULL, entry TEXT NOT NULL)")
        except BaseException:
            connection.close()
            raise
        return connection

    def get(self, key):
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT entry FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        entry = validate_entry(json.loads(row[0]))
        if entry["key"] != key:
            raise ValueError("LLM cache entry key does not match its database row")
        return entry

    def put(self, entry):
        validate_entry(entry)
        payload = encoded(entry)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO responses (key, created_at, entry) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET created_at=excluded.created_at, entry=excluded.entry "
                "WHERE excluded.created_at > responses.created_at",
                (entry["key"], entry["created_at"], payload),
            )


def save_pending(directory, entry):
    """Publish a durable full response separately from an unavailable database."""
    validate_entry(entry)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entry['key']}_{hashlib.sha256(encoded(entry).encode()).hexdigest()}.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded(entry) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        with_directory = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(with_directory)
        finally:
            os.close(with_directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)


def restore_pending_responses(root, *, cache_path, max_retries, retry_delay_seconds,
                              sleep=time.sleep, include_root=False):
    """Import pending artifacts without changing their evidence or original age."""
    retry = validate_retry(max_retries, retry_delay_seconds)
    store = ResponseCache(cache_path)
    root = Path(root)
    def discover():
        try:
            root.stat()
        except FileNotFoundError:
            return []
        paths = []
        def inaccessible(error):
            raise error
        # glob/rglob suppress access errors, which could silently lose a paid
        # response during recovery. Walk explicitly within the retry boundary.
        for directory, _, filenames in os.walk(root, onerror=inaccessible):
            directory = Path(directory)
            if directory.name == "llm_cache_pending" or (include_root and directory == root):
                paths.extend(directory / name for name in filenames if name.endswith(".json"))
        return sorted(paths)
    paths = retry_cache(discover, operation="discover", phase="recovery", key=None, sleep=sleep, **retry)
    for path in paths:
        def restore():
            entry = validate_entry(json.loads(path.read_text(encoding="utf-8")))
            store.put(entry)
        try:
            retry_cache(restore, operation="restore", phase="recovery", key=None, sleep=sleep, **retry)
        except LLMCacheError as error:
            error.pending_response_path = str(path)
            raise
    return len(paths)
