# AppWorld integration decisions

## Ownership and lineage

The entrypoints select `base`, `direct`, or a compatible retrieval preset through
`retrieval.py --preset NAME`. The launcher owns fresh and recovered run
identity, task selection, source snapshots, runtime isolation, and phase
processes. The worker owns benchmark integration and finalization. Variant code
owns memory policy; generic services own provider and store operations.

The shared retrieval entrypoint preserves base's adaptation, executor prompt,
memory updates and cold-start lifetime. V2 retrieval uses a separate client with
its preset's exact provider retry policy; ordinary stages keep their configured
policy. All clients share canonical logging and the resolved response cache.
The selected preset becomes the setting and run family, separating automatic
recovery across designs. The composed run descriptor embeds the selected preset's
configuration and retrieval-only descriptor as provenance, while the adjacent
`retrieval.variant.json` defines the end-to-end lifecycle. Base remains an alias
of its existing entrypoint and descriptor.

The executor interaction pattern and initial executor prompt derive from
`metaskill-appworld/baselines/reme2f37a15/benchmark/appworld/appworld_react_agent.py`
and `prompt.py`. The bundled upstream revision is
`2f37a159b72a04ac1885a7db7f1a663a833e7791`; the inspected sibling checkout revision
is `2bd51dabc0e84c2d6aa9f9db66e630a78f58ab94`, with local integration changes.
These identifiers describe lineage, not an assertion that the working tree is
clean. Every run's recoverable snapshots and file hashes describe what executed.

The baseline replaces ReMe memory orchestration with the new retrieve/adapt/
reflect/consolidate policy and uses the declared 40-interaction and bounded-retry
controls. The direct setting shares execution mechanics and initially matched
instructions, with a separate prompt file to make future prompt experiments
explicit. Do not claim exact parity with the native ReMe actor or its original
memory update rules.

## Benchmark lifecycle

Reuse official AppWorld task loading and evaluation and the verified shared
runner, episode service, and call logger. Task order is resolved before execution
and rechecked against the selected dataset in the run. Input task objects exclude
ground truth. Official evaluation reports are preserved separately; only the
scalar post-task reward enters the memory policy.

AppWorld initialization removes its task output directory. Opening execution
through the shared episode service preserves earlier retrieval/adaptation calls
in the canonical `lm_calls.jsonl`. Close the world, evaluate its saved outputs,
and release evaluator database caches before starting the next task. Finish or
explicitly skip post-update work and persist its outcomes before marking a task
completed and advancing the stream.

Direct mode must not initialize memory services or require Elasticsearch assets.
Its official evaluation and native outputs remain identical in structure to base
mode. Ordinary unsuccessful task outcomes are valid benchmark observations;
missing outputs or failed evaluation are infrastructure errors.

## Isolation and recoverability

Each invocation lives under `raw_logs/metaweft-<setting>-<dataset>/experiment/output/`
with a timestamp and random suffix. Each test split gets its own process, memory
index, mutable databases, and state. Runtime cache and temporary paths are invocation
local. The explicit shared LLM response store is an exception; it holds completed
responses, never mutable benchmark state. Read-only runtime/data assets may be shared. Recovery creates another
isolated invocation linked to the failed parent; it never edits that parent.

Copy MetaWeft and the relevant current sibling source before executing workers;
resolve imports to those copies rather than the live source tree. Preserve
prompts, descriptors, dependency records, source/data hashes, original and selected
task manifests, and actual resolved settings. Shared immutable benchmark data are
identified explicitly. Store mappings and full vectors accompany complete memory
snapshots, allowing the learned state to be understood without a live ES process.

Recovery-format v1 is implemented by `experiment/recovery.py`. Every future
launcher must use its recovery argument, immutable identity/provenance checks,
selection lock, and completed-task checkpoint contract. Fresh runs seal their
resolved identity after source/data/runtime capture. Recovery selects a failed
compatible new-format run, rejects active/completed lineages, validates its
artifacts, and executes the captured worker and variant code. Current checkout
defaults do not override saved settings. No legacy-log migration is inferred.

Commit `checkpoint.json` atomically after native output validation, official
evaluation, post-update work and a verified complete memory export. It records
the contiguous completed prefix, next task, memory reference/mapping and artifact
hashes. The checkpoint is authoritative if a subsequent manifest write failed.
Task output files precede checkpoint publication; unavailable required memory
snapshots prevent a commit. Initial checkpoints support failure in the first task.

Recovery copies committed tasks and their relative evidence references. A fresh
store restores IDs, full documents and vectors and verifies the resulting export.
Incomplete-task outputs and partial updates are archived separately and never
used as restored state. The adapter bypasses completed tasks before model calls
or memory mutations, while the shared runner retains the full dataset for final
evaluation. Completed phases are copied; unstarted splits still begin empty.
Source import links are rebuilt inside the new directory, and runtime caches/ES
state are newly allocated. Resolved LLM cache mode/path remain those of the parent.
Pending full responses in task `llm_cache_pending` artifacts are restored before
provider calls, preserving original timestamps. The selected mode still governs
restarted calls; `no_cache` never reuses an unfinished task's prior response.
Repeated recovery carries all earlier failure evidence.

## Logging, failure, and collection

Use one task-owned canonical LLM log. The recorder attributes every physical chat
or embedding attempt to a stage/component and logical call, retains failed
attempts, and fails closed when logging fails. Account for adaptation, execution,
reflection, consolidation, embeddings, and retries; missing usage and unknown
pricing must remain visible. Do not substitute custom traces for native AppWorld
logs and output databases.

The same canonical logger records cache hits with `attempt=0` and
`metadata.extra.cache`; physical attempts start at one. Hits retain original
response usage as evidence but are excluded from incurred token/cost totals and
physical-call counts. The default `no_cache` mode stores successful responses
while making fresh calls. Opt-in `cache` and `timed_cache` reuse matching samples,
so experiments must retain and compare their resolved cache settings.

Exhausted provider calls stop the stream from every stage and preserve a structured
failure record. Other memory-stage fallback produces explicit stage results and degradation metadata.
Cache I/O retries use the saved retry budget and delay; exhausted `LLMCacheError`
is fatal in every stage and never triggers another provider call as a write retry.
A phase may finish with `status=completed` and nonempty `degraded_task_ids`;
`degraded_reasons` records why. Confirmed prior candidate writes survive a later
candidate failure, while remaining work is recorded as skipped. Missing logging,
provenance, benchmark outputs, or official evaluation is fatal. Initial empty
memory verification is also mandatory. Task checkpoint snapshot failures stop
the stream; a failed optional final
snapshot is preserved in phase `evidence_gaps` without rewriting completion as
failure or presenting unavailable memory as an empty verified snapshot.

Accounting deduplicates physical-call IDs across copied task logs and archived
attempts. Top-level totals are cumulative; `current_attempt` and `inherited`
subtotals distinguish newly incurred usage. Malformed response envelopes count as
unknown usage when no numeric usage is reported. Collector rows label recovery
lineage and reused tasks; ETA uses newly completed tasks in the current attempt.

The invocation manifest uses `baseline=metaweft` and a `setting` of `base`, `direct`,
or the selected retrieval preset, with
phases retaining the established `ordered_task_ids`, `source_task_count`,
`partial`, output path, status, and progress fields. `tool/collect_appworld.sh`
reads phase diagnostics even after completion and resolves moved run paths against
the discovered invocation root. Its separate MetaWeft table reports setting,
official scores, completed/selected/source coverage, degradation, and nonempty
evidence-gap counts. V2 preset notes are generated from resolved settings and
optionally followed by a user annotation; recovery descriptions use saved settings.
ACE/ReMe
rows keep their previous interpretation.

Offline tests should verify retry attribution, native log preservation, feedback
boundaries, candidate operations, failure/skipping behavior, task sequencing,
split isolation, no-memory bypass, snapshots, and collector compatibility. Live
acceptance uses the same two Quick2h normal tasks for both settings, preserves
actual task outcomes, and verifies complete run artifacts. Two successful launches
alone do not establish that every consolidation operation occurred or that memory
improved performance.
