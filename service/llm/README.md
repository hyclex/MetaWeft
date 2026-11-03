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
- Both operations try transient provider failures at most three times, waiting
  one then five seconds. SDK retries are disabled. Other failures propagate.
- `recorder(operation, request, invoke, *, phase, component, logical_call_id,
  attempt, step)` must call `invoke()` once, preserve its response/error, and return
  the response. Each physical attempt invokes the recorder separately. Operation
  is `chat` or `embedding`. The caller supplies canonical task ownership and any
  accounting; create a client with the next task's recorder for each task.
- A supplied `client` is a dependency-injection hook for offline tests; it must
  itself have SDK retries disabled. `close()` closes that client too.

See [DESIGNS.md](DESIGNS.md) for boundaries and rationale.
