# V1 validation — 2026-11-02

Both settings completed real AppWorld runs on `ff58e36_1`, then `ff58e36_2`, the
first two tasks in `quick2h_test_normal_v2`. These are **2/21-task smoke tests**,
with one ordered stream per run. They establish operation of the implementation,
not a performance improvement or complete-scenario result.

## Live evidence

| Setting / run | Task rewards | Execution steps | Degraded tasks | Physical model calls | Reported total tokens |
| --- | --- | --- | --- | --- | --- |
| [Direct](../../../raw_logs/metaweft-direct-2h/experiment/output/20261102T201435Z_test-normal_21bde701/run_manifest.json) | 1.0, 1.0 | 17, 18 | 0/2 | 35 | 238,538 |
| [Base, clarified consolidation prompt](../../../raw_logs/metaweft-base-2h/experiment/output/20261102T201823Z_test-normal_79e887bc/run_manifest.json) | 1.0, 1.0 | 17, 16 | 0/2 | 46 | 253,641 |
| [Initial base smoke, retained](../../../raw_logs/metaweft-base-2h/experiment/output/20261102T201435Z_test-normal_421f0b9d/run_manifest.json) | 1.0, 1.0 | 19, 20 | 2/2 | 47 | 334,700 |

All three streams completed official evaluation. Their selected-task TGC and SGC
were 100%; subset SGC is not a complete-scenario measurement. Each run has zero
missing-usage calls, and provider dollar prices remain unverified (`cost_usd:
null`). Token totals include all invoked stages and the initial degraded run is
retained in the [generated report](../../../res/raw_collect.md).

The final base run started with zero memories. Task one added one memory and
skipped a redundant candidate. Task two retrieved that memory, produced 568
characters of fixed initial guidance, then revised the memory and skipped another
candidate. The final snapshot contains one record, its embedding, and supporting
references to both tasks. Its 46 calls comprise 33 execution calls, one retrieval
embedding, one adaptation call, two reflection calls, and nine consolidation
chat/embedding calls. No memory stage was degraded.

Direct execution made only its 35 execution calls. Initial/final memory artifacts
explicitly record disabled empty state; there is no Elasticsearch index, embedding,
adaptation, reflection, or consolidation call.

The initial base smoke exposed a prompt-contract error: some Add decisions echoed
candidate text in fields required to be null. The framework recorded both tasks
as degraded, skipped remaining candidates after failure, and retained one
confirmed write from task two. Task one had made no write, so this run did not
exercise next-task adaptation. The consolidation prompt was clarified with
operation-specific examples; the schema, no-repair policy, and failure behavior
were unchanged. The second base invocation started fresh and used a new source
snapshot. No historical run was edited or replaced.

## Offline and artifact checks

The prepared AppWorld interpreter passed **54 offline tests**: 18 base-policy,
16 service, 10 integration, five launch, and five collector tests. Coverage includes
Add/Skip/Revise/Defer, failed-task learning, missing evidence, malformed responses,
retained partial updates, uncertain writes, provider retries, source-vector export,
feedback isolation, evaluator cleanup, pre-execution log preservation, direct-mode
service bypass, and unchanged ACE/ReMe report rows.

Native task outputs, exact evaluation coverage, dataset hashes, source hashes,
memory snapshots/references, canonical call phases/usage, and collection were
checked against the saved artifacts. Snapshot bootstrap resolution was checked
without rerunning tasks: benchmark and MetaWeft modules resolve to captured source.
Ground-truth loading is explicitly disabled in the captured worker/runner source.

Live tasks both succeeded, so failed-task learning and Defer were established
by deterministic offline cases, not these live trajectories. Provider retry and
missing-usage handling were likewise tested offline. Full experiments, resume,
parallel task execution, and other benchmarks were not run.

## Retry and recovery validation (2026-11-03)

The reliability extension passes 78 offline tests in the prepared AppWorld
interpreter. These include the real MetaWeft worker, memory pipeline, shared task
runner, episode service and canonical logger with fake provider, world/evaluator
and storage transports. Tests inject outages in every LLM stage, including after
a confirmed partial memory update. They verify completed-task reuse, exact
restoration of checkpoint documents/vectors, exclusion of partial updates,
independent split resets, repeated recovery, finalization-only recovery, atomic
checkpoint publication, source/runtime/data integrity checks, and unchanged
parent artifacts. A subprocess test confirms imports resolve to copied saved
source and rebuilt import links. No provider or Elasticsearch process is started.

Canonical-log tests preserve reported tokens from empty text and malformed
embedding responses. Recovery accounting includes inherited failed calls without
double counting copied calls. Collector tests cover lineage labels and ETA based
on newly completed work. Existing collector fixtures were aligned with the
current dataset-specific report files and required `dataset_version` field.

Both actual shell entrypoints pass two-task dry runs with five retries and a
60-second delay recorded in settings, without creating run directories. Recovery
dry runs use temporary new-format fixtures and make no writes or provider calls.
Historical live runs above retain their original protocol and artifacts; they
were not migrated or rerun. This extension has not been exercised in a paid live
experiment and does not detect provider model substitution.

## Shared retrieval entrypoint validation (2026-11-06)

The new `retrieval --preset NAME` entrypoint completed one fresh `no_cache` run
per requested preset on `ff58e36_1`, then `ff58e36_2`. Both runs completed **2/21**
Quick2h normal tasks, with 100% selected-task TGC and SGC, zero degraded tasks,
and zero recorded evidence gaps. Subset SGC is not a complete-scenario result;
these smoke tests establish integration, not improved end-to-end performance.

| Preset / saved run | Task rewards | Execution steps | Memory counts: initial / task 1 / task 2 | Physical calls | Reported total tokens |
| --- | --- | --- | --- | --- | --- |
| [transferability_3](../../../raw_logs/metaweft-transferability_3-2h/experiment/output/20261106T220741Z_test-normal_0e3a18de/run_manifest.json) | 1.0, 1.0 | 20, 21 | 0 / 2 / 4 | 54 | 324,666 |
| [efficiency_1](../../../raw_logs/metaweft-efficiency_1-2h/experiment/output/20261106T220745Z_test-normal_226cf521/run_manifest.json) | 1.0, 1.0 | 17, 19 | 0 / 1 / 1 | 49 | 286,176 |

On task two, transferability retrieved and graded two learned memories, then
produced 799 characters of adapted guidance. Efficiency retrieved one learned
memory and produced 651 characters of guidance. Both listwise selectors correctly
bypassed inputs smaller than 20. Thus the live runs exercised empty-bank behavior,
updates, nonempty retrieval, adaptation and transferability's grader; they did
not exercise the Max shortlist or efficiency's Flash listwise call.

Accounting includes all stages, with no cache hits or inherited attempts:

| Preset | Retrieval | Adaptation | Execution | Reflection | Consolidation |
| --- | --- | --- | --- | --- | --- |
| transferability_3 | 2 | 1 | 41 | 2 | 8 |
| efficiency_1 | 1 | 1 | 36 | 2 | 9 |

Counts are physical chat/embedding calls. Both accounting records report
`missing_usage_calls=0`; dollar cost remains unknown because provider rates were
not verified. Full records are preserved for
[transferability](../../../raw_logs/metaweft-transferability_3-2h/experiment/output/20261106T220741Z_test-normal_0e3a18de/experiments/outputs/test-normal/accounting.json)
and [efficiency](../../../raw_logs/metaweft-efficiency_1-2h/experiment/output/20261106T220745Z_test-normal_226cf521/experiments/outputs/test-normal/accounting.json).

The prepared runtime passed **211 offline framework tests**. New cases cover
all 11 compatible preset selections, unsupported-preset rejection, unchanged base
alias controls, actual worker ranking with synthetic 20/225-memory banks,
separate retrieval retries, call attribution, fallback degradation, fatal outages,
client cleanup, exact-control capacity, committed-memory recovery and preset
selection boundaries. Collector fixtures verify automatic/custom notes without
changing the report schema. All 54 documented background commands passed shell
syntax checks; all 18 fresh-launch combinations passed dry runs without creating
run directories or making provider calls.

Post-run checks verified each run's 530 captured framework/sibling source files,
sealed identity, composed descriptor and unchanged preset configuration, final
checkpoint hashes, exact evaluation coverage, full memory vectors and source
references, prior-memory hashes in retrieval traces, canonical ranking-call
references and accounting totals. The [generated Quick2h report](../../../res/appworld_2h.md)
is collected from these raw artifacts and retains their saved design notes.
No full benchmark or live recovery run was performed; recovery behavior was
verified with offline failure/restart fixtures.
