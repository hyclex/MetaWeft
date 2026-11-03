# MetaWeft experiment commands

Run all commands from the workspace root. `base` enables the reference memory
pipeline; `direct` disables retrieval and post-update. Both launchers start new
isolated runs. See [AppWorld usage](appworld/README.md) for setup and options and
[integration decisions](appworld/DESIGNS.md) before modifying the launchers.

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

## Longer runs and collection

The following are optional experiment examples, not part of the two-task smoke
test. `test-all` runs normal and challenge as separate streams with independent
memory initialization. `--limit N` applies separately to each selected split.

```bash
nohup bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-all > /dev/null 2>&1 &
nohup bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-all > /dev/null 2>&1 &

# Replace --dataset 2h with --dataset 8h or --dataset full for those manifests.
# Regenerate the report once from preserved raw artifacts, in the background.
nohup bash tool/collect_appworld.sh > /dev/null 2>&1 &
```

These commands continue after the terminal closes. Each experiment saves its
console output in its unique run directory's `console.log`.

The generated `res/raw_collect.md` contains a separate MetaWeft table showing
setting, official TGC/SGC, completed/selected/source task counts, status,
degradation, and evidence gaps. Existing ACE and ReMe tables remain separate. No score or successful
live-validation claim is implied by these example commands.
