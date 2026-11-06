# LLM service design

This is new minimal transport code using the existing runtime's OpenAI-compatible
SDK. It does not inherit ReMe's model base classes, caches, prompt parsing, or
research policy. The existing shared AppWorld `common/lm_logger.py` supplies the
canonical recorder through the experiment layer; this package imports no AppWorld
code. Callers supply task-owned recording and pending-cache recovery paths.

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
model names, URLs, dimensions, timeouts, prompts, and cache settings without secrets.
Embedding text selection and batching belong to variants.
The initial API is synchronous because the v1 stream is sequential. It does not
predefine future retrieval or adaptation workflows.

## Persistent response reuse

The parent is the existing synchronous client and its retry/recording contracts.
The intended change is to reduce repeated provider work when the operation,
endpoint, and complete request match. Offline comparisons verify identical
responses with fewer physical calls; they make no model-quality claim.

Both chat and embedding successes populate a shared SQLite store as JSON after
service validation. `no_cache` is the default and always calls the provider;
`cache` opts into reuse regardless of age; `timed_cache` requires an age below
1,800 seconds. The timestamp is the original successful response's cache timestamp,
not its last access or recovery time. Cache keys preserve exact text, list order,
and all request parameters while ignoring dictionary order and caller attribution.
No semantic matching or partial embedding-batch reuse is performed.

Explicit client cache settings override environment settings. Launchers resolve
the mode and absolute database path once and preserve them for recovery. The
default `raw_logs/.cache/metaweft/llm.sqlite3` is outside source snapshots and
invocation-local runtime caches. Sharing this response store is an explicit
exception to run-state isolation; it does not share procedural memory, checkpoints,
or mutable benchmark state. Short transactions protect writes, but simultaneous
misses may issue duplicate provider calls.

Cache I/O has its own bounded retries using the existing retry budget and delay.
Exhaustion raises `LLMCacheError`, which is fatal in every stage; a cache failure
never becomes a provider retry or an unrecorded bypass. Callers provide
`cache_recovery_dir` so full successful responses awaiting persistence survive in
task/query `llm_cache_pending` artifacts. Recovery replays pending writes before
provider calls without changing their timestamps or the selected mode. A restart
in `no_cache` still calls the provider; timed entries can expire during downtime.

Canonical logs distinguish hits from physical attempts and retain original
response usage without charging it again. Replay preserves SDK response fields
and embedding order. Cache hits therefore reuse stochastic samples as well as
deterministic responses; matched experiments must record the mode and use
`no_cache` when fresh samples are required. Service-valid nonempty text remains
cacheable even if a variant later rejects its JSON. No historical log import,
automatic pruning, or cross-request semantic reuse is added.
