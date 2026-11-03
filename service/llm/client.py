"""Small synchronous provider client; workflow and prompt policy live in variants."""

import math
import time
import uuid


class ModelClient:
    def __init__(
        self, *, api_key=None, base_url=None, chat_model,
        embedding_model="text-embedding-3-large", dimensions=1024,
        recorder=None, chat_timeout=120, embedding_timeout=30,
        client=None, sleep=time.sleep,
    ):
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

    @staticmethod
    def _transient(error):
        from openai import APIConnectionError, APIStatusError

        return isinstance(error, APIConnectionError) or (
            isinstance(error, APIStatusError)
            and (error.status_code in (408, 409, 429) or error.status_code >= 500)
        )

    def _invoke(self, operation, request, invoke, *, phase, component, step):
        logical_call_id = uuid.uuid4().hex
        for attempt in range(1, 4):
            try:
                if self.recorder is None:
                    return invoke()
                return self.recorder(
                    operation, request, invoke, phase=phase, component=component,
                    logical_call_id=logical_call_id, attempt=attempt, step=step,
                )
            except Exception as error:
                if not self._transient(error) or attempt == 3:
                    raise
                self.sleep((1, 5)[attempt - 1])

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
        response = self._invoke(
            "embedding", request,
            lambda: self.client.embeddings.create(**request, timeout=self.embedding_timeout),
            phase=phase, component=component, step=step,
        )
        vectors = [None] * len(texts)
        for item in response.data:
            if not isinstance(item.index, int) or not 0 <= item.index < len(vectors):
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
