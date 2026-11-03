# LLM service API

`ModelClient` makes synchronous calls to OpenAI-compatible chat and embedding endpoints.
The service depends on `openai`; use the prepared AppWorld interpreter.

```python
from MetaWeft.service.llm import ModelClient

client = ModelClient(
    api_key=provider_key, base_url=provider_url, chat_model="qwen3.8-flash",
    embedding_model="text-embedding-3-large", dimensions=1024, recorder=record_call,
)
response = client.chat(messages, phase="adaptation", component="base")
guidance = response.choices[0].message.content
vectors = client.embed(["applicability and procedure"], phase="retrieval", component="base")
client.close()
```

- `chat(messages, *, phase, component, step=None, temperature=0.0, **kwargs)` returns
  the provider response, including usage. Additional arguments go to the provider;
  streaming is unsupported. Default timeout is 120 seconds per physical attempt.
- `embed(texts, *, phase, component, step=None)` returns vectors in input order.
  Empty input makes no call. Default timeout is 30 seconds. Response indices,
  dimensions, and finite numeric values are validated; inputs/vectors are never
  silently truncated or padded.
- Both operations use `retry.json`: `max_retries=5` and
  `retry_delay_seconds=60` by default (one initial call plus five retries).
  `ModelClient(..., max_retries=..., retry_delay_seconds=...)` overrides these
  defaults; launchers resolve and preserve them once per run. Zero retries and
  zero delay are supported. SDK retries remain disabled.
- Exceptions from provider calls, missing/empty chat text, and malformed embedding
  responses are retried. Every physical response is recorded before validation,
  including usage reported for an empty response. Nonempty text is returned
  without JSON/schema repair. Logging failures and cancellation are not retried.
- Exhaustion raises `LLMUnavailableError` with `operation`, `phase`,
  `logical_call_id`, `attempts`, and the last error as `__cause__`. Experiment
  pipelines must propagate it so the worker can stop at its saved task boundary.
- `recorder(operation, request, invoke, *, phase, component, logical_call_id,
  attempt, step)` must call `invoke()` once, preserve its response/error, and return
  the response. Each physical attempt invokes the recorder separately. Operation
  is `chat` or `embedding`. The caller supplies canonical task ownership and any
  accounting; create a client with the next task's recorder for each task.
- A supplied `client` is a dependency-injection hook for offline tests; it must
  itself have SDK retries disabled. `close()` closes that client too.

See [DESIGNS.md](DESIGNS.md) for boundaries and rationale.

Observed MetaWeft failure cases include embedding HTTP 429 rate limits and a
chat response with empty text but 21,192 reported input tokens. The service uses
the broader provider-call boundary above rather than an allowlist of status codes.
It does not detect a provider silently substituting another model.
