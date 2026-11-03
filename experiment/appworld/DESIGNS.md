# AppWorld integration decisions

## Ownership and lineage

The entrypoints only select `base` or `direct`. The launcher owns fresh run
identity, task selection, source snapshots, runtime isolation, and phase
processes. The worker owns benchmark integration and finalization. Variant code
owns memory policy; generic services own provider and store operations.

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
index, mutable databases, and state. Cache and temporary paths are invocation
local. Read-only runtime/data assets may be shared. There is no resume path.

Copy MetaWeft and the relevant current sibling source before executing workers;
resolve imports to those copies rather than the live source tree. Preserve
prompts, descriptors, dependency records, source/data hashes, original and selected
task manifests, and actual resolved settings. Shared immutable benchmark data are
identified explicitly. Store mappings and full vectors accompany complete memory
snapshots, allowing the learned state to be understood without a live ES process.

## Logging, failure, and collection

Use one task-owned canonical LLM log. The recorder attributes every physical chat
or embedding attempt to a stage/component and logical call, retains failed
attempts, and fails closed when logging fails. Account for adaptation, execution,
reflection, consolidation, embeddings, and retries; missing usage and unknown
pricing must remain visible. Do not substitute custom traces for native AppWorld
logs and output databases.

Memory-stage fallback produces explicit stage results and degradation metadata.
A phase may finish with `status=completed` and nonempty `degraded_task_ids`;
`degraded_reasons` records why. Confirmed prior candidate writes survive a later
candidate failure, while remaining work is recorded as skipped. Missing logging,
provenance, benchmark outputs, or official evaluation is fatal. Initial empty
memory verification is also mandatory. Later unavailable snapshots remain
explicit evidence gaps: task snapshot failures degrade the task; a failed final
snapshot is preserved in phase `evidence_gaps` without rewriting completion as
failure or presenting unavailable memory as an empty verified snapshot.

The invocation manifest uses `baseline=metaweft` and `setting=base|direct`, with
phases retaining the established `ordered_task_ids`, `source_task_count`,
`partial`, output path, status, and progress fields. `tool/collect_appworld.sh`
reads phase diagnostics even after completion and resolves moved run paths against
the discovered invocation root. Its separate MetaWeft table reports setting,
official scores, completed/selected/source coverage, degradation, and nonempty
evidence-gap counts. ACE/ReMe
rows keep their previous interpretation.

Offline tests should verify retry attribution, native log preservation, feedback
boundaries, candidate operations, failure/skipping behavior, task sequencing,
split isolation, no-memory bypass, snapshots, and collector compatibility. Live
acceptance uses the same two Quick2h normal tasks for both settings, preserves
actual task outcomes, and verifies complete run artifacts. Two successful launches
alone do not establish that every consolidation operation occurred or that memory
improved performance.
