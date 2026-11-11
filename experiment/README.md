# MetaWeft experiment commands

Run all commands from the workspace root. `base` enables the reference memory
pipeline; `direct` disables retrieval and post-update. `retrieval --preset NAME`
selects a compatible retrieval design within the same cold-start pipeline.
`metaweft_v1` enables separate ReMe-like task memory and an ACE-derived adaptation
playbook, using `efficiency_1` retrieval with final k=10 and paired checkpoints.
It is its own entrypoint and takes no `--preset` option. See its
[design descriptor](appworld/metaweft_v1.variant.json).
`metaweft_v2` is a separate variant with content-only adapter memory inputs and
success-gated playbook learning from adapter inputs and outputs. It retains V1
retrieval, execution controls and paired checkpoints; see its
[design descriptor](appworld/metaweft_v2.variant.json).
All launchers start new
isolated runs by default. `--recover` continues a failed new-format run in a new
linked directory. See [AppWorld usage](appworld/README.md) for setup and options and
[integration decisions](appworld/DESIGNS.md) before modifying the launchers.

The shared retrieval entrypoint supports `base`, `performance_1..3`,
`efficiency_1..3`, `transferability_1..3`, and `exact_base`. The background catalog
below covers `base`, `direct`, `metaweft_v1`, and `metaweft_v2` alongside
`transferability_3` and `efficiency_1`. All background examples use `test-all`
for `2h`, `8h`, and `full` datasets.
Each command below is a separate launch; choose the combination you need.
`test-all` runs both test splits in parallel with independent memory initialization;
tasks within each split remain sequential.

Both requested presets pass at most five memories to the unchanged base adaptation
and execution/update pipeline. Automatic run notes describe the resolved retrieval
design in the generated `res/` tables; `--note` appends a user annotation. Recovery
preserves the saved design and optionally replaces the user annotation.

| Preset | Retrieval design |
| --- | --- |
| `transferability_3` | HNSW depth 225 / 450 ANN candidates; Max selects 20, reasoning Flash grades them; top 5 to adaptation. |
| `efficiency_1` | HNSW depth 20 / 40 ANN candidates; nonthinking Flash listwise ranking; top 5 to adaptation. |

`exact_base` retains its 225-memory capacity and fails if evolving memory exceeds
it. Expanded query/representation presets `setting_1..6` remain retrieval-only.

LLM calls default to `no_cache` (fresh calls that still populate the shared
response store). For repeated design tests, add `--cache-mode cache`; use
`--cache-mode timed_cache` for a 30-minute lifetime. The default database is
`raw_logs/.cache/metaweft/llm.sqlite3`, overridable with `--cache-path PATH`.
For example, from the workspace root:

```bash
bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --limit 2 --cache-mode cache
```

This is a live run on misses. Reused responses are logged separately from incurred
provider usage. Recovery preserves the original cache settings; use `no_cache`
for fresh stochastic repetitions. See the [cache contract](../service/llm/README.md#response-cache).

## Inspect and check

```bash
# First two tasks per split (four total); no writes or model calls.
bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 2h --split test-all --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 2h --split test-all --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-all --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-all --limit 2 --dry-run

# Existing runtime readiness; no benchmark run.
bash metaskill-appworld/run/setup_runtime.sh --check

# Offline framework checks in the prepared AppWorld Python environment.
metaweft_python=$(python3 -B metaskill-appworld/run/runtime_paths.py appworld_python)
"$metaweft_python" -B -m unittest discover -s MetaWeft/tests -v
```

## Live smoke validation

These commands make real provider calls and write raw artifacts. They select the
first two Quick2h tasks from each split in existing manifest order, four tasks in
total. Memory resets between splits; within each split, the second task can use
memory learned from the first. Run completion does not require every task to be solved.

```bash
bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --limit 2 --cache-mode no_cache --note "v1 two-per-split smoke"
bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all --limit 2 --cache-mode no_cache --note "v1 two-per-split smoke"
bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 2h --split test-all --limit 2 --cache-mode no_cache --note "retrieval two-per-split smoke"
bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 2h --split test-all --limit 2 --cache-mode no_cache --note "retrieval two-per-split smoke"
```

For the dual-memory V1 smoke, select one split to run exactly two examples:

```bash
bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-normal --limit 2 --cache-mode no_cache --note "dual-memory V1 two-task smoke"
```

Both memories start empty and persist from the first task to the second. The
[V1 validation record](appworld/VALIDATION_V1.md) reports the completed smoke,
including rejected playbook updates and the requested test-log cleanup.

For V2, the normal-split smoke selects exactly `ff58e36_1`, then `ff58e36_2`
(2/21 Quick2h normal tasks). Both memories begin empty. Only official reward 1.0
allows an adaptation update; partial reward and an environment completion signal
alone do not qualify. The updater reads the adapter input/output pair and minimal
edit targets, while full trajectories remain available to task-memory learning.

```bash
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-normal --limit 2 --dry-run
# Live provider calls; use a disposable response store when smoke logs will be removed.
metaweft_smoke_cache=$(mktemp -d /tmp/metaweft-v2-smoke.XXXXXX)
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-normal --limit 2 --cache-mode no_cache --cache-path "$metaweft_smoke_cache/llm.sqlite3" --note "V2 two-task smoke; disposable logs"
```

Inspect the printed unique run directory and record task outcomes, update/skip
coverage, paired checkpoints and all-stage accounting before cleanup. When test-log
cleanup is requested, remove only that smoke run directory, its disposable cache
directory and temporary test logs. Retain the validation summary and leave shared
runtime assets, the shared response store and historical runs intact. Two tasks
may not exercise a successful playbook edit or nonempty-playbook transfer; report
actual coverage and use offline fixtures for missing branches.

The [V2 validation record](appworld/VALIDATION_V2.md) reports 319 passing offline
tests, both live tasks fully solved, valid playbook ADD/UPDATE and memory transfer,
all-stage usage, coverage limits, and the requested test-log cleanup.

Inspect official evaluations, memory-stage outcomes, degradation, complete task
logs, and accounting before recording validation results. A cold-start first task
alone cannot test nonempty retrieval or adaptation. Offline fixtures cover the
operation branches that two live tasks may not naturally reach. Listwise ranking
bypasses banks smaller than 20, so these two-task-per-split runs cannot establish its live
coverage. Transferability's grader still runs on a nonempty smaller bank.

The [validation record](appworld/VALIDATION.md#shared-retrieval-entrypoint-validation-2026-11-06)
records the earlier two-task normal-split runs, their observed stage coverage, and accounting.
Their raw smoke-run artifacts were removed during requested test-log cleanup.

## Retrieval-only validation

The retrieval-only benchmark supports `base`, `setting_1..6`, all nine v2 category
presets, and `exact_base` through one configurable entrypoint. It reads an existing
frozen pool and never executes AppWorld tasks or updates memory. The expanded
`setting_1..6` designs are not supported by the live AppWorld wrapper.

```bash
# Offline checks; no provider calls.
python3 -B -m unittest discover -s evaluations/appworld_mem_retrieval/tests -v

# Inspect a two-query selection without writes or provider calls.
python3 -B evaluations/appworld_mem_retrieval/scenarios/base.py --preset setting_4 --limit 2 --dry-run

# Live retrieval only: two queries per preset against the full prepared pool.
for preset in base setting_1 setting_2 setting_3 setting_4; do
  python3 -B evaluations/appworld_mem_retrieval/scenarios/base.py --preset "$preset" --limit 2
done
```

The two-query matrix verifies operability, not improved retrieval quality. Keep
all runs and inspect coverage, traces, and canonical calls. See the
[retrieval benchmark README](../../evaluations/appworld_mem_retrieval/README.md)
for representations, metrics, checkpoint recovery, and output details. Its
`--recover RUN_DIRECTORY` resumes retrieval queries; the AppWorld recovery
commands below resume full task streams.

## Longer runs and collection

Each command is a separate optional experiment; choose the combination you need.
`test-all` runs normal and challenge as separate streams, resetting memory for each.
`--limit N` applies separately to each selected split. Fresh commands explicitly
use `no_cache` for fresh provider samples.

### Fresh runs

#### 2h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --cache-mode no_cache >> raw_logs/metaweft-base-2h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all --cache-mode no_cache >> raw_logs/metaweft-direct-2h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 2h --split test-all --cache-mode no_cache >> raw_logs/metaweft-transferability_3-2h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 2h --split test-all --cache-mode no_cache >> raw_logs/metaweft-efficiency_1-2h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-all --cache-mode no_cache >> raw_logs/metaweft-metaweft_v1-2h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-all --cache-mode no_cache >> raw_logs/metaweft-metaweft_v2-2h-test-all-fresh-launch.log 2>&1 < /dev/null &
```

#### 8h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-all --cache-mode no_cache >> raw_logs/metaweft-base-8h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-all --cache-mode no_cache >> raw_logs/metaweft-direct-8h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 8h --split test-all --cache-mode no_cache >> raw_logs/metaweft-transferability_3-8h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 8h --split test-all --cache-mode no_cache >> raw_logs/metaweft-efficiency_1-8h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 8h --split test-all --cache-mode no_cache >> raw_logs/metaweft-metaweft_v1-8h-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 8h --split test-all --cache-mode no_cache >> raw_logs/metaweft-metaweft_v2-8h-test-all-fresh-launch.log 2>&1 < /dev/null &
```

#### full

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-all --cache-mode no_cache >> raw_logs/metaweft-base-full-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-all --cache-mode no_cache >> raw_logs/metaweft-direct-full-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset full --split test-all --cache-mode no_cache >> raw_logs/metaweft-transferability_3-full-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset full --split test-all --cache-mode no_cache >> raw_logs/metaweft-efficiency_1-full-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset full --split test-all --cache-mode no_cache >> raw_logs/metaweft-metaweft_v1-full-test-all-fresh-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset full --split test-all --cache-mode no_cache >> raw_logs/metaweft-metaweft_v2-full-test-all-fresh-launch.log 2>&1 < /dev/null &
```

These commands continue after the terminal closes. Each combination has a
separate launcher log that retains startup failures and earlier launch attempts.
Every experiment also saves `console.log` inside its unique run directory.

### Collect results

The collector generates all three dataset reports from actual run artifacts:

```bash
nohup bash tool/collect_appworld.sh >> raw_logs/metaweft-collection-launch.log 2>&1 < /dev/null &
```

The reports (`res/appworld_2h.md`, `res/appworrld_8h.md`, and
`res/appworkd_full.md`) show the selected setting and saved note (automatic design
descriptions plus any custom annotation for v2 presets),
official TGC/SGC, completed/selected/source coverage, status, degradation, evidence
gaps, and recovery lineage. Existing ACE and ReMe tables remain separate.
No performance or successful validation claim follows from these example commands.

## Recovery

Repeat the original entrypoint/preset, dataset, split, and any `--limit`/`--task-id` options
and add `--recover`. The examples below select full splits; append the original
subset options when recovering a smoke run. Inspect a saved run without writes:

```bash
bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 2h --split test-all --limit 2 --recover --dry-run
```

Recovery selects only the same setting, uses captured source/settings and committed
memory when enabled, and creates a new linked run directory. A different setting
cannot recover the parent. Cache policy and retrieval retry controls come from the saved run.
For `metaweft_v1` and `metaweft_v2`, recovery restores both memories from the same
committed epoch, using the saved variant's playbook schema. V1 and V2 cannot recover
each other's runs.

### Recover the latest compatible failed run

#### 2h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --recover >> raw_logs/metaweft-base-2h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all --recover >> raw_logs/metaweft-direct-2h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 2h --split test-all --recover >> raw_logs/metaweft-transferability_3-2h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 2h --split test-all --recover >> raw_logs/metaweft-efficiency_1-2h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-all --recover >> raw_logs/metaweft-metaweft_v1-2h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-all --recover >> raw_logs/metaweft-metaweft_v2-2h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
```

#### 8h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-all --recover >> raw_logs/metaweft-base-8h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-all --recover >> raw_logs/metaweft-direct-8h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 8h --split test-all --recover >> raw_logs/metaweft-transferability_3-8h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 8h --split test-all --recover >> raw_logs/metaweft-efficiency_1-8h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 8h --split test-all --recover >> raw_logs/metaweft-metaweft_v1-8h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 8h --split test-all --recover >> raw_logs/metaweft-metaweft_v2-8h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
```

#### full

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-all --recover >> raw_logs/metaweft-base-full-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-all --recover >> raw_logs/metaweft-direct-full-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset full --split test-all --recover >> raw_logs/metaweft-transferability_3-full-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset full --split test-all --recover >> raw_logs/metaweft-efficiency_1-full-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset full --split test-all --recover >> raw_logs/metaweft-metaweft_v1-full-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset full --split test-all --recover >> raw_logs/metaweft-metaweft_v2-full-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
```

### Recover a specific failed run

Replace each placeholder directory with the failed run matching the command's
setting, dataset, split, and task selection.

#### 2h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --recover "/path/to/failed/base-2h-test-all-run" >> raw_logs/metaweft-base-2h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all --recover "/path/to/failed/direct-2h-test-all-run" >> raw_logs/metaweft-direct-2h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 2h --split test-all --recover "/path/to/failed/transferability_3-2h-test-all-run" >> raw_logs/metaweft-transferability_3-2h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 2h --split test-all --recover "/path/to/failed/efficiency_1-2h-test-all-run" >> raw_logs/metaweft-efficiency_1-2h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-all --recover "/path/to/failed/metaweft_v1-2h-test-all-run" >> raw_logs/metaweft-metaweft_v1-2h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-all --recover "/path/to/failed/metaweft_v2-2h-test-all-run" >> raw_logs/metaweft-metaweft_v2-2h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
```

#### 8h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-all --recover "/path/to/failed/base-8h-test-all-run" >> raw_logs/metaweft-base-8h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-all --recover "/path/to/failed/direct-8h-test-all-run" >> raw_logs/metaweft-direct-8h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset 8h --split test-all --recover "/path/to/failed/transferability_3-8h-test-all-run" >> raw_logs/metaweft-transferability_3-8h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 8h --split test-all --recover "/path/to/failed/efficiency_1-8h-test-all-run" >> raw_logs/metaweft-efficiency_1-8h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 8h --split test-all --recover "/path/to/failed/metaweft_v1-8h-test-all-run" >> raw_logs/metaweft-metaweft_v1-8h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 8h --split test-all --recover "/path/to/failed/metaweft_v2-8h-test-all-run" >> raw_logs/metaweft-metaweft_v2-8h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
```

#### full

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-all --recover "/path/to/failed/base-full-test-all-run" >> raw_logs/metaweft-base-full-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-all --recover "/path/to/failed/direct-full-test-all-run" >> raw_logs/metaweft-direct-full-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset transferability_3 --dataset full --split test-all --recover "/path/to/failed/transferability_3-full-test-all-run" >> raw_logs/metaweft-transferability_3-full-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset full --split test-all --recover "/path/to/failed/efficiency_1-full-test-all-run" >> raw_logs/metaweft-efficiency_1-full-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset full --split test-all --recover "/path/to/failed/metaweft_v1-full-test-all-run" >> raw_logs/metaweft-metaweft_v1-full-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset full --split test-all --recover "/path/to/failed/metaweft_v2-full-test-all-run" >> raw_logs/metaweft-metaweft_v2-full-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
```

Recovery reuses committed tasks and restarts the unfinished task from the prior
checkpoint's memory. The failed parent remains unchanged, including failed calls
and partial-update evidence. Runs predating the checkpoint format are unsupported.
See [recovery details](appworld/README.md#recovery).
