# MetaWeft design decisions

## Research boundary

The framework fixes reusable service operations and benchmark bookkeeping while
leaving experimental policy in versioned `src/` implementations. Its design space
covers memory representation, retrieval, task-conditioned adaptation, execution
context, and evidence selection plus post-update. The v1 reference choices below
are defaults for `base`, not restrictions on later variants.

Services accept caller-owned content, mappings, queries, model settings, and
logging hooks. They do not decide which experience is useful, how evidence is
sampled, which memories should be revised, or when a task stream advances. Keep
new graph relationships, multiple indexes, mixed retrieval, query decomposition,
iterative adaptation, and evidence batching in a new variant when required by an
experiment. V1 supplies extension points rather than implementing these features.

## Reference policy

1. **Memory:** a flat Elasticsearch collection with stable IDs, `when_to_use`,
   procedural `content`, and references to preserved supporting evidence. Embed
   `when_to_use` together with `content`. Store the configured vector alongside the
   record. Preserve source vectors and mappings in memory snapshots; exports must
   traverse the complete collection rather than the first search page.
2. **Retrieval:** embed the task instruction and retrieve up to five neighbors
   once before execution. Use all available records when fewer than five exist.
   An empty store skips memory guidance. Keep retrieved IDs, content, and outcomes
   in task artifacts.
3. **Adaptation:** one LLM pass turns retrieved records and the current task into
   execution guidance; using no memory is allowed. Guidance enters the initial
   executor context and remains fixed throughout the task.
4. **Execution:** retain the ReMe-derived AppWorld ReAct interaction pattern,
   benchmark loading and stopping behavior, with the declared v1 execution limits.
   The `direct` setting bypasses all memory stages and uses its own executor prompt.
5. **Evidence and reflection:** preserve the task specification, full execution
   trace, and official post-task scalar reward. One reflection pass proposes at
   most three candidates from success or failure. Prompts distinguish observed
   procedures from inferred explanations and untested fixes.
6. **Consolidation:** process candidates in order against the current store,
   including earlier confirmed writes from the same task. Retrieve up to five
   semantic neighbors and consult their referenced evidence as well as the current
   task's evidence. Choose add, skip, revise, or defer. A revision needs supporting
   evidence; differing applicability conditions are not automatically a conflict.
   Deferred candidates and reasons remain in the update record. There is no
   utility-based eviction.

The five-neighbor bound may miss overlaps elsewhere in memory. Full referenced
evidence is supplied without silently dropping trace content to fit a prompt;
provider context-limit failures follow the visible failure policy. This is a
deliberate v1 limitation, not a promise that memory grows indefinitely without
additional design work.

## Controls and failure policy

The initial setting uses `qwen3.8-flash`, execution temperature `0.9`, memory-stage
temperature `0`, seed `0`, thinking disabled, at most 40 executor interactions, and
one primary trial per task. AppWorld's environment random seed remains `100`,
distinct from the model seed. Embeddings use `text-embedding-3-large` at 1024
dimensions. Retrieval and candidate-neighbor search each return at most five
records. Reflection returns at most three candidates.

Each logical provider call defaults to an initial attempt plus five retries,
with a 60-second delay. `service/llm/retry.json` supplies configurable defaults;
the resolved run settings preserve the actual budget. SDK-internal retries are
disabled. Chat attempts time
out after 120 seconds and embedding attempts after 30 seconds. Invalid structured
output receives no additional model repair call. These limits are not inherited
from ReMe's much larger retry loop. Record the settings and every physical attempt
so comparisons include retry and memory costs.

Successful chat and embedding responses populate a shared exact-request cache.
The default `no_cache` mode makes fresh calls; `cache` opts into unlimited-age
reuse and `timed_cache` accepts responses younger than 30 minutes. Preserve the
resolved mode/path with every run and match modes when comparing designs. Hits
reuse the previous sample even at nonzero temperature and are recorded separately
from physical provider usage. Cache I/O retries use the same budget and delay;
exhaustion raises fatal `LLMCacheError`, preserving pending successful responses
for recovery without repeating the paid call during a cache-write retry. See
the [LLM service design](service/llm/DESIGNS.md#persistent-response-reuse).

Provider-call exceptions, empty chat responses, and invalid embedding responses
are retried. Exhaustion propagates `LLMUnavailableError` from every stage and stops
the stream without committing the unfinished task. Recovery restores memory from
the preceding completed task, excluding partial updates from the unfinished task.
Other retrieval/adaptation failures retain the existing visible fallback to no
guidance. Non-provider reflection/consolidation failures retain confirmed writes
and skip remaining updates; invalid nonempty JSON receives no repair call. Normal
empty-memory skips and deliberate skip/defer decisions remain distinct from failed
stages. A completed stream may still contain explicitly degraded tasks.

Missing canonical logs, failed provenance/artifact writes, invalid benchmark
execution or evaluation, and interrupted runs are infrastructure failures. Do not
hide them behind memory fallback. Store operations reconcile uncertain writes
before treating them as confirmed. Initial memory must be verified empty or the
run fails. Task-boundary snapshots are required for checkpoint commits; a failure
stops the stream while retaining its last verified checkpoint. Optional final
snapshot failures remain visible evidence gaps and never imply empty memory.

## Feedback, lifetime, and provenance

The executor sees task information, API descriptions, environment observations,
and adapted guidance. Adaptation sees the task and retrieved records. Reflection
and consolidation may additionally see preserved traces and scalar rewards.
Official assertion text, private evaluation data, and reference solutions remain
outside these inputs. Evidence references resolve inside preserved run artifacts;
they are not permission to open arbitrary files.

Independent invocations and test splits start with empty memory. Tasks within one
stream run in manifest order and observe completed prior updates. V1 has no
training phase, arbitrary warm-start import, task parallelism, mid-task memory
refresh, automatic eviction, or general design-search controller.

Recovery-format v1 adds completed-task continuation for newly created runs. Each
attempt has a unique directory and retains its failed parent unchanged. Original
source, settings, task order and benchmark/runtime identity are verified and
reused. This reliability extension is based on the current `base`/`direct` v1
variants; its hypothesis is that provider outages can be tolerated without losing
committed task progress or altering memory continuity. Offline comparisons use
the same stream with injected outages and assert restored state, call coverage,
and task order. They are not performance or model-identity claims.

The executor and initial executor prompt derive from the sibling bundled ReMe
AppWorld implementation. Memory policy is new: explicit adaptation, failure-aware
reflection, and semantic revision/defer replace native summary/addition/utility
management. Parent paths and revisions are recorded in adjacent descriptors;
actual copied working source is preserved for every run because a parent commit
alone cannot recover local changes. The sibling AppWorld runtime, benchmark source,
shared runner/logger, and lifecycle helpers remain explicit dependencies.

Future changes should state their parent variant and hypothesis, update the
descriptor and relevant versioned prompts, preserve matched comparison controls,
and test the changed mechanism. Consult
[AppWorld integration decisions](experiment/appworld/DESIGNS.md) before changing
launching, logging, evaluation, or source snapshots.
