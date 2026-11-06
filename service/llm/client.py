"""Small synchronous provider client; workflow and prompt policy live in variants."""

import math
from pathlib import Path
import time
import uuid

from .cache import (
    CACHE_VERSION, LLMCacheError, ResponseCache, plain, request_key,
    restore_pending_responses, restore_response, retry_cache, save_pending,
)
from .config import CACHE_TTL_SECONDS, cache_settings, retry_defaults, validate_retry


class LLMUnavailableError(RuntimeError):
    """The physical attempt budget was exhausted; stop the task stream."""

    def __init__(self, *, operation, phase, logical_call_id, attempts, cause):
        super().__init__(f"{operation} unavailable in {phase} after {attempts} attempts: "
                         f"{type(cause).__name__}: {cause}")
        self.operation = operation
        self.phase = phase
        self.logical_call_id = logical_call_id
        self.attempts = attempts


class InvalidResponseError(ValueError):
    """A recorded provider response cannot be consumed by this service."""


class ModelClient:
    def __init__(
        self, *, api_key=None, base_url=None, chat_model,
        embedding_model="text-embedding-3-large", dimensions=1024,
        recorder=None, chat_timeout=120, embedding_timeout=30,
        max_retries=None, retry_delay_seconds=None,
        cache_mode=None, cache_path=None, cache_recovery_dir=None,
        client=None, sleep=time.sleep,
    ):
        defaults = retry_defaults() if max_retries is None or retry_delay_seconds is None else {}
        retry = validate_retry(
            defaults["max_retries"] if max_retries is None else max_retries,
            defaults["retry_delay_seconds"] if retry_delay_seconds is None else retry_delay_seconds,
        )
        cache = cache_settings(cache_mode, cache_path)
        for name, value in (("chat_model", chat_model), ("embedding_model", embedding_model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        for name, value in (("chat_timeout", chat_timeout), ("embedding_timeout", embedding_timeout)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.client = client
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.recorder = recorder
        self.chat_timeout = chat_timeout
        self.embedding_timeout = embedding_timeout
        self.sleep = sleep
        self.max_retries = retry["max_retries"]
        self.retry_delay_seconds = retry["retry_delay_seconds"]
        self.cache_mode = cache["cache_mode"]
        self.cache_path = Path(cache["cache_path"])
        self.cache = ResponseCache(self.cache_path)
        self.cache_recovery_dir = (Path(cache_recovery_dir) if cache_recovery_dir is not None
                                   else self.cache_path.parent / "llm_cache_pending")
        endpoint = base_url if base_url is not None else getattr(client, "base_url", None)
        # Real SDK clients expose httpx.URL; fake transports may omit the URL.
        self.endpoint = str(endpoint) if endpoint is not None else "injected-client"
        self._pending_restored = False

    def _cache_action(self, action, *, operation, phase, key):
        return retry_cache(action, operation=operation, phase=phase, key=key,
                           max_retries=self.max_retries, retry_delay_seconds=self.retry_delay_seconds,
                           sleep=self.sleep)

    def _store_response(self, response, *, operation, phase, key, logical_call_id):
        entry = {"version": CACHE_VERSION, "key": key, "operation": operation,
                 "created_at": time.time(), "source_logical_call_id": logical_call_id,
                 "response": plain(response)}
        try:
            self._cache_action(lambda: self.cache.put(entry), operation="write", phase=phase, key=key)
        except LLMCacheError as error:
            try:
                error.pending_response_path = self._cache_action(
                    lambda: save_pending(self.cache_recovery_dir, entry),
                    operation="preserve", phase=phase, key=key,
                )
            except LLMCacheError as preservation_error:
                error.preservation_error = str(preservation_error)
                error.args = (f"{error}; response could not be preserved: {preservation_error}",)
            raise

    def _invoke(self, operation, request, invoke, validate, *, phase, component, step):
        logical_call_id = uuid.uuid4().hex
        key = request_key(operation, self.endpoint, request)
        if not self._pending_restored:
            restore_pending_responses(
                self.cache_recovery_dir, cache_path=self.cache_path,
                max_retries=self.max_retries, retry_delay_seconds=self.retry_delay_seconds, sleep=self.sleep,
                include_root=True,
            )
            self._pending_restored = True
        cache_context = {"hit": False, "mode": self.cache_mode, "key": key}
        if self.cache_mode != "no_cache":
            def lookup():
                entry = self.cache.get(key)
                if entry is None:
                    return None
                if entry["operation"] != operation:
                    raise ValueError("LLM cache operation does not match the request")
                response = restore_response(entry)
                validate(response)
                if self.cache_mode == "timed_cache" and time.time() - entry["created_at"] >= CACHE_TTL_SECONDS:
                    return None
                return entry, response

            cached = self._cache_action(lookup, operation="read", phase=phase, key=key)
            if cached is not None:
                entry, response = cached
                cache_context.update(hit=True, created_at=entry["created_at"],
                                     source_logical_call_id=entry["source_logical_call_id"])
                if self.recorder is not None:
                    response = self.recorder(
                        operation, request, lambda: response, phase=phase, component=component,
                        logical_call_id=logical_call_id, attempt=0, step=step, cache=cache_context,
                    )
                return validate(response)
        for attempt in range(1, self.max_retries + 2):
            provider_error = None

            def physical_call():
                nonlocal provider_error
                try:
                    return invoke()
                except Exception as error:
                    provider_error = error
                    raise

            try:
                if self.recorder is None:
                    response = physical_call()
                else:
                    response = self.recorder(
                        operation, request, physical_call, phase=phase, component=component,
                        logical_call_id=logical_call_id, attempt=attempt, step=step,
                        cache=cache_context,
                    )
            except Exception as error:
                # A recorder failure must never cause a second provider call, even
                # when its exception happens to resemble a transport failure.
                if error is not provider_error:
                    raise
                failure = error
            else:
                try:
                    result = validate(response)
                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                    failure = InvalidResponseError(str(error))
                else:
                    # Cache persistence must never enter the provider retry loop.
                    self._store_response(response, operation=operation, phase=phase, key=key,
                                         logical_call_id=logical_call_id)
                    return result
            if attempt == self.max_retries + 1:
                raise LLMUnavailableError(
                    operation=operation, phase=phase, logical_call_id=logical_call_id,
                    attempts=attempt, cause=failure,
                ) from failure
            self.sleep(self.retry_delay_seconds)

    @staticmethod
    def _chat_response(response):
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise InvalidResponseError("Chat response contains no text")
        return response

    def chat(self, messages, *, phase, component, step=None, temperature=0.0, **kwargs):
        """Return the complete provider response, including content and usage."""
        from openai import NotGiven, Omit
        from openai._utils import maybe_transform
        from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

        if kwargs.get("stream"):
            raise ValueError("ModelClient.chat requires a non-streaming response")
        request = {
            "model": self.chat_model, "messages": messages,
            "temperature": temperature, **kwargs,
        }
        # Apply the SDK's own normalization once so iterables are not consumed
        # by hashing before dispatch, and omission sentinels never enter JSON.
        request = maybe_transform({key: value for key, value in request.items()
                                   if not isinstance(value, (NotGiven, Omit))},
                                  CompletionCreateParamsNonStreaming)
        return self._invoke(
            "chat", request,
            lambda: self.client.chat.completions.create(**request, timeout=self.chat_timeout),
            self._chat_response,
            phase=phase, component=component, step=step,
        )

    def embed(self, texts, *, phase, component, step=None):
        """Embed a batch in input order; reject malformed or altered dimensions."""
        if not texts:
            return []
        request = {
            "model": self.embedding_model, "input": list(texts),
            "dimensions": self.dimensions, "encoding_format": "float",
        }
        return self._invoke(
            "embedding", request,
            lambda: self.client.embeddings.create(**request, timeout=self.embedding_timeout),
            lambda response: self._embedding_response(response, len(texts)),
            phase=phase, component=component, step=step,
        )

    def _embedding_response(self, response, count):
        vectors = [None] * count
        for item in response.data:
            if type(item.index) is not int or not 0 <= item.index < len(vectors):
                raise ValueError("Embedding response contains an invalid input index")
            if vectors[item.index] is not None:
                raise ValueError("Embedding response contains a duplicate input index")
            vector = list(item.embedding)
            if len(vector) != self.dimensions:
                raise ValueError("Embedding response dimensions differ from the configured dimensions")
            if not all(isinstance(value, (float, int)) and not isinstance(value, bool)
                       and math.isfinite(value) for value in vector):
                raise ValueError("Embedding response contains a nonfinite or nonnumeric value")
            vectors[item.index] = vector
        if any(vector is None for vector in vectors):
            raise ValueError("Embedding response is missing an input")
        return vectors

    def close(self):
        self.client.close()
