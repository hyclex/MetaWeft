# MetaWeft experiment commands

Run all commands from the workspace root. `base` enables the reference memory
pipeline; `direct` disables retrieval and post-update. Both launchers start new
isolated runs by default. `--recover` continues a failed new-format run in a new
linked directory. See [AppWorld usage](appworld/README.md) for setup and options and
[integration decisions](appworld/DESIGNS.md) before modifying the launchers.

The supported settings are `base` (cold-start memory) and `direct`. Each supports
`test-normal`, `test-challenge`, and `test-all` for `2h`, `8h`, and `full` datasets.
Each command below is a separate launch; choose the combination you need.
`test-all` includes both test splits with independent memory initialization.

## Inspect and check

```bash
# Exact two-task selection; no writes or model calls.
bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-normal --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-normal --limit 2 --dry-run

# Existing runtime readiness; no benchmark run.
bash metaskill-appworld/run/setup_runtime.sh --check

# Offline framework checks in the prepared AppWorld Python environment.
metaweft_python=$(python3 -B metaskill-appworld/run/runtime_paths.py appworld_python)
"$metaweft_python" -B -m unittest discover -s MetaWeft/tests -v
```

## Live two-task validation

These commands make real provider calls and write raw artifacts. They select the
same first two Quick2h normal tasks in existing manifest order. The second task
can exercise memory learned from the first; a successful run does not require
both benchmark tasks to be solved.

```bash
bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-normal --limit 2 --note "v1 two-task smoke"
bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-normal --limit 2 --note "v1 two-task smoke"
```

Inspect official evaluations, memory-stage outcomes, degradation, complete task
logs, and accounting before recording validation results. A cold-start first task
alone cannot test nonempty retrieval or adaptation. Offline fixtures cover the
operation branches that two live tasks may not naturally reach.

## Recovery

Repeat the original selection options and add `--recover`. Inspect the selected
parent, saved configuration, reused tasks, and remaining work first if useful:

```bash
bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --recover --dry-run
bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --recover
# A specific parent can replace automatic latest-failure selection.
bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-all --recover /path/to/failed/run
```

For background recovery with `nohup`, keep the original dataset, split, and any
`--limit`/`--task-id` options. All supported dataset, setting, and split combinations
are listed below.

### Recover the latest compatible failed run

#### 2h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-normal --recover >> raw_logs/metaweft-base-2h-test-normal-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-challenge --recover >> raw_logs/metaweft-base-2h-test-challenge-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --recover >> raw_logs/metaweft-base-2h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-normal --recover >> raw_logs/metaweft-direct-2h-test-normal-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-challenge --recover >> raw_logs/metaweft-direct-2h-test-challenge-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all --recover >> raw_logs/metaweft-direct-2h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
```

#### 8h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-normal --recover >> raw_logs/metaweft-base-8h-test-normal-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-challenge --recover >> raw_logs/metaweft-base-8h-test-challenge-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-all --recover >> raw_logs/metaweft-base-8h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-normal --recover >> raw_logs/metaweft-direct-8h-test-normal-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-challenge --recover >> raw_logs/metaweft-direct-8h-test-challenge-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-all --recover >> raw_logs/metaweft-direct-8h-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
```

#### full

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-normal --recover >> raw_logs/metaweft-base-full-test-normal-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-challenge --recover >> raw_logs/metaweft-base-full-test-challenge-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-all --recover >> raw_logs/metaweft-base-full-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-normal --recover >> raw_logs/metaweft-direct-full-test-normal-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-challenge --recover >> raw_logs/metaweft-direct-full-test-challenge-recovery-latest-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-all --recover >> raw_logs/metaweft-direct-full-test-all-recovery-latest-launch.log 2>&1 < /dev/null &
```

### Recover a specific failed run

Replace each placeholder directory with the failed run matching that command's
setting, dataset, and split.

#### 2h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-normal --recover "/path/to/failed/base-2h-test-normal-run" >> raw_logs/metaweft-base-2h-test-normal-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-challenge --recover "/path/to/failed/base-2h-test-challenge-run" >> raw_logs/metaweft-base-2h-test-challenge-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all --recover "/path/to/failed/base-2h-test-all-run" >> raw_logs/metaweft-base-2h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-normal --recover "/path/to/failed/direct-2h-test-normal-run" >> raw_logs/metaweft-direct-2h-test-normal-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-challenge --recover "/path/to/failed/direct-2h-test-challenge-run" >> raw_logs/metaweft-direct-2h-test-challenge-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all --recover "/path/to/failed/direct-2h-test-all-run" >> raw_logs/metaweft-direct-2h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
```

#### 8h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-normal --recover "/path/to/failed/base-8h-test-normal-run" >> raw_logs/metaweft-base-8h-test-normal-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-challenge --recover "/path/to/failed/base-8h-test-challenge-run" >> raw_logs/metaweft-base-8h-test-challenge-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-all --recover "/path/to/failed/base-8h-test-all-run" >> raw_logs/metaweft-base-8h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-normal --recover "/path/to/failed/direct-8h-test-normal-run" >> raw_logs/metaweft-direct-8h-test-normal-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-challenge --recover "/path/to/failed/direct-8h-test-challenge-run" >> raw_logs/metaweft-direct-8h-test-challenge-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-all --recover "/path/to/failed/direct-8h-test-all-run" >> raw_logs/metaweft-direct-8h-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
```

#### full

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-normal --recover "/path/to/failed/base-full-test-normal-run" >> raw_logs/metaweft-base-full-test-normal-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-challenge --recover "/path/to/failed/base-full-test-challenge-run" >> raw_logs/metaweft-base-full-test-challenge-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-all --recover "/path/to/failed/base-full-test-all-run" >> raw_logs/metaweft-base-full-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-normal --recover "/path/to/failed/direct-full-test-normal-run" >> raw_logs/metaweft-direct-full-test-normal-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-challenge --recover "/path/to/failed/direct-full-test-challenge-run" >> raw_logs/metaweft-direct-full-test-challenge-recovery-specific-launch.log 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-all --recover "/path/to/failed/direct-full-test-all-run" >> raw_logs/metaweft-direct-full-test-all-recovery-specific-launch.log 2>&1 < /dev/null &
```

These commands continue after the terminal closes. Each combination has a separate
launcher log that captures startup and selection errors, including failures before
a new run directory is created. Each recovered run also saves its own `console.log`
in its unique run directory. `>>` retains earlier launch attempts.

Recovery reuses committed tasks and restarts the unfinished task with the prior
checkpoint's memory. It executes saved source and settings, including the retry
budget, and preserves the parent unchanged. Runs created before the checkpoint
format are not supported. See [recovery details](appworld/README.md#recovery).

## Longer runs and collection

The following are optional experiment examples, not part of the two-task smoke
test. `test-all` runs normal and challenge as separate streams with independent
memory initialization. `--limit N` applies separately to each selected split.

### Fresh runs

#### 2h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-normal > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-challenge > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-normal > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-challenge > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all > /dev/null 2>&1 &
```

#### 8h

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-normal > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-challenge > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 8h --split test-all > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-normal > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-challenge > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 8h --split test-all > /dev/null 2>&1 &
```

#### full

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-normal > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-challenge > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset full --split test-all > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-normal > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-challenge > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset full --split test-all > /dev/null 2>&1 &
```

### Collect results

The collector generates reports for all three dataset settings in one invocation.

```bash
nohup bash tool/collect_appworld.sh > /dev/null 2>&1 &
```

These commands continue after the terminal closes. Each experiment saves its
console output in its unique run directory's `console.log`.

The generated dataset reports (`res/appworld_2h.md`, `res/appworrld_8h.md`, and
`res/appworkd_full.md`) contain separate MetaWeft tables showing
setting, official TGC/SGC, completed/selected/source task counts, status,
degradation, and evidence gaps. Existing ACE and ReMe tables remain separate. No score or successful
live-validation claim is implied by these example commands.
