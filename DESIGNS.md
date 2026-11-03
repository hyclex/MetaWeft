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

Each logical provider call permits at most three physical transient attempts,
with 1- and 5-second waits. SDK-internal retries are disabled. Chat attempts time
out after 120 seconds and embedding attempts after 30 seconds. Invalid structured
output receives no additional model repair call. These limits are not inherited
from ReMe's much larger retry loop. Record the settings and every physical attempt
so comparisons include retry and memory costs.

An exhausted retrieval or adaptation failure is recorded and execution proceeds
without memory guidance. Reflection or consolidation failure preserves confirmed
earlier writes, records the failure, and skips the remaining update work for that
task. Normal empty-memory skips and deliberate skip/defer decisions are distinct
from failed stages. A finished stream can therefore be `completed` with degraded
tasks; degradation is not silently converted into a successful memory operation.

Missing canonical logs, failed provenance/artifact writes, invalid benchmark
execution or evaluation, and interrupted runs are infrastructure failures. Do not
hide them behind memory fallback. Store operations reconcile uncertain writes
before treating them as confirmed. Initial memory must be verified empty or the
run fails. A later unavailable memory snapshot is recorded as an evidence gap;
task snapshot failures also mark the task degraded. The stream may continue, but
its manifest and report must expose this limitation rather than claim verified
memory state.

## Feedback, lifetime, and provenance

The executor sees task information, API descriptions, environment observations,
and adapted guidance. Adaptation sees the task and retrieved records. Reflection
and consolidation may additionally see preserved traces and scalar rewards.
Official assertion text, private evaluation data, and reference solutions remain
outside these inputs. Evidence references resolve inside preserved run artifacts;
they are not permission to open arbitrary files.

Independent invocations and test splits start with empty memory. Tasks within one
stream run in manifest order and observe completed prior updates. V1 has no
training phase, warm-start import, resume path, task parallelism, mid-task memory
refresh, automatic eviction, or general design-search controller.

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
