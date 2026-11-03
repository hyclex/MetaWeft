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
