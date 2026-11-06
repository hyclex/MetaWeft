# LLM service API

`ModelClient` makes synchronous calls to OpenAI-compatible chat and embedding endpoints.
The service depends on `openai`; use the prepared AppWorld interpreter.
Both operations persist successful responses. The default `no_cache` mode makes
fresh provider calls; choose `cache` for repeated design tests that should reuse
identical requests.

```python
from MetaWeft.service.llm import ModelClient

client = ModelClient(
    api_key=provider_key, base_url=provider_url, chat_model="qwen3.8-flash",
    embedding_model="text-embedding-3-large", dimensions=1024, recorder=record_call,
    cache_mode="cache",  # Opt in to reuse across repeated tests.
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
  attempt, step, cache=None)` must call `invoke()` once, preserve its response/error,
  and return the response. Each physical attempt invokes the recorder separately. Operation
  is `chat` or `embedding`. The caller supplies canonical task ownership and any
  accounting; create a client with the next task's recorder for each task.
  Cache hits use `attempt=0`; provider attempts start at one. The `cache` dictionary
  contains `hit`, `mode`, and `key`, plus `created_at` (Unix seconds) and
  `source_logical_call_id` when known. Integrated recorders preserve it under
  canonical `metadata.extra.cache`; retrieval call references retain it too.
- A supplied `client` is a dependency-injection hook for offline tests; it must
  itself have SDK retries disabled. `close()` closes that client too.

## Response cache

| Mode | Behavior |
| --- | --- |
| `no_cache` (default) | Always call the provider and store each successful response for future reuse. |
| `cache` | Reuse a matching successful response regardless of age; otherwise call and store. |
| `timed_cache` | Reuse a matching response only while it is younger than 1,800 seconds; otherwise call and replace it. |

Set `ModelClient(..., cache_mode=..., cache_path=...)`, or use
`METAWEFT_LLM_CACHE_MODE` and `METAWEFT_LLM_CACHE_PATH`. Explicit constructor
arguments take precedence over environment variables, then defaults apply.
The default database is `raw_logs/.cache/metaweft/llm.sqlite3` under the workspace
root. Launch commands from that root. Experiment launchers preserve the resolved
mode and absolute path in run settings, and recovery uses those saved settings.

Keys match the operation, provider endpoint, and complete request, including
model, exact ordered messages/inputs, generation parameters, and embedding
dimensions. Dictionary key order is immaterial; text and list order are not
normalized. Task IDs, stage attribution, retry controls, and timeouts do not
partition identical provider requests. Credentials are not stored in the cache.

The cache stores complete JSON responses, including embedding vectors and original
usage, only after service validation succeeds. It does not repair or reject
nonempty chat text based on a variant's JSON schema. Reads preserve the response
contract and validate cached responses again. Cache hits remain visible in call
logs but incur no new provider tokens or charges; original usage is retained as
evidence, separate from incurred usage. Reusing a stochastic request reuses its
previous sample, so use `no_cache` for fresh repetitions.

Cache read/write failures retry with the configured `max_retries` and
`retry_delay_seconds`, then raise `LLMCacheError`. They do not silently bypass
the cache or trigger another provider call. A successful response awaiting a
cache write is retained in `cache_recovery_dir`, defaulting to an
`llm_cache_pending` directory beside the database. Experiment integrations set a
task/query-owned directory instead. Recovery restores pending writes before provider
calls, preserving their original timestamps. The selected mode still applies:
`no_cache` makes fresh calls when unfinished work restarts, and `timed_cache`
does not gain a new lifetime from recovery. Logging and cache failures must
propagate past memory fallback.
`LLMCacheError` reports `operation`, `phase`, `key`, `attempts`, `cause`, and
`pending_response_path`; `preservation_error` also identifies a failure to save
the pending response.

The SQLite store is shared across runs; mutable task state, memory, and outputs
remain isolated. Concurrent cache misses can make duplicate provider calls.
Offline tests should use a temporary cache path so they cannot consume or change
the workspace cache.

See [DESIGNS.md](DESIGNS.md) for boundaries and rationale.

Observed MetaWeft failure cases include embedding HTTP 429 rate limits and a
chat response with empty text but 21,192 reported input tokens. The service uses
the broader provider-call boundary above rather than an allowlist of status codes.
It does not detect a provider silently substituting another model.
