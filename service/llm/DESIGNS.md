# LLM service design

This is new minimal transport code using the existing runtime's OpenAI-compatible
SDK. It does not inherit ReMe's model base classes, caches, prompt parsing, or
research policy. The existing shared AppWorld `common/lm_logger.py` supplies the
canonical recorder through the experiment layer; this package imports no AppWorld
code and owns no output paths.

One logical call has one ID and, by default, at most six separately recorded
physical attempts, separated by 60-second waits. `retry.json` owns the configurable
defaults; saved run settings own the effective budget on recovery. SDK retries
are disabled. All exceptions originating from the provider invocation, empty chat
text, and malformed response envelopes/vectors are retryable. Recorder failures,
service configuration errors, cancellation, and variant JSON/schema errors are
outside that boundary. Recording happens before response validation so rejected
responses retain their reported usage. The embedding recorder also handles malformed
vectors without logging full vector payloads. Transport success in a raw log does
not imply that response validation succeeded.

Exhaustion raises `LLMUnavailableError` with call attribution and the original
cause. The experiment layer propagates it past memory fallback, preserves failure
evidence, and resumes only from a committed task checkpoint. No provider-model
identity detection or automatic structured-output repair is added.

Chat and embedding configuration remain separate. The caller records resolved
model names, URLs, dimensions, timeouts, and prompts without secrets. No response
cache is implemented; embedding text selection and batching belong to variants.
The initial API is synchronous because the v1 stream is sequential. It does not
predefine future retrieval or adaptation workflows.
