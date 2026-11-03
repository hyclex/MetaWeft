# LLM service design

This is new minimal transport code using the existing runtime's OpenAI-compatible
SDK. It does not inherit ReMe's model base classes, caches, prompt parsing, or
research policy. The existing shared AppWorld `common/lm_logger.py` supplies the
canonical recorder through the experiment layer; this package imports no AppWorld
code and owns no output paths.

One logical call has one ID and at most three separately recorded physical
attempts. Retryable failures are SDK connection errors and HTTP 408/409/429/5xx.
Disabling internal SDK retries makes attribution unambiguous. Logging failures and
structured-output errors do not trigger provider retries. Provider responses are
recorded before the embedding service validates them; variants separately parse
chat output after it has been recorded.

Chat and embedding configuration remain separate. The caller records resolved
model names, URLs, dimensions, timeouts, and prompts without secrets. No response
cache is implemented; embedding text selection and batching belong to variants.
The initial API is synchronous because the v1 stream is sequential. It does not
predefine future retrieval or adaptation workflows.
