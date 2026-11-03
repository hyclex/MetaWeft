"""Small synchronous provider client; workflow and prompt policy live in variants."""

import math
import time
import uuid

from .config import retry_defaults, validate_retry


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
        client=None, sleep=time.sleep,
    ):
        defaults = retry_defaults() if max_retries is None or retry_delay_seconds is None else {}
        retry = validate_retry(
            defaults["max_retries"] if max_retries is None else max_retries,
            defaults["retry_delay_seconds"] if retry_delay_seconds is None else retry_delay_seconds,
        )
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

    def _invoke(self, operation, request, invoke, validate, *, phase, component, step):
        logical_call_id = uuid.uuid4().hex
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
                    )
            except Exception as error:
                # A recorder failure must never cause a second provider call, even
                # when its exception happens to resemble a transport failure.
                if error is not provider_error:
                    raise
                failure = error
            else:
                try:
                    return validate(response)
                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                    failure = InvalidResponseError(str(error))
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
        if kwargs.get("stream"):
            raise ValueError("ModelClient.chat requires a non-streaming response")
        request = {
            "model": self.chat_model, "messages": messages,
            "temperature": temperature, **kwargs,
        }
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
