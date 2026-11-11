# Dual-memory V2 validation — 2026-11-11

Implemented the separate [V2 entrypoint](metaweft_v2.py),
[variant descriptor](metaweft_v2.variant.json), and
[policy and prompt lineage](../../src/ace_adapt_reme_task_v2/DESIGNS.md).
This validates the requested mechanism; it is not a matched performance comparison
against V1 or evidence of generalization beyond the selected tasks.

## Offline checks

The prepared runtime check passed for `origin-py312-59b185a7b247`.
The complete offline suite passed **319 tests in 80.851 seconds**, including 33
new V2 policy, launcher, worker and recovery tests. `git diff --check` and shell
syntax checks passed. No offline test made a billable model call.

Coverage includes content-only adapter requests; reward-zero/partial/full-success
gating; invalid-adapter skips; trajectory/evaluator exclusion; exact frozen
adapter input/output; deterministic edit targets; valid and invalid evidence
pointers; atomic ADD/UPDATE; empty guidance and edit plans; JSON-encoded playbook
size limits; memory transfer; V2 schema isolation and paired recovery; source
capture; accounting; and unchanged V1/legacy behavior.

```bash
bash metaskill-appworld/run/setup_runtime.sh --check
metaweft_python=$(python3 -B metaskill-appworld/run/runtime_paths.py appworld_python)
"$metaweft_python" -B -m unittest discover -s MetaWeft/tests -v
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-normal --limit 2 --cache-mode no_cache --dry-run
```

## Live two-task smoke

Run identity: `metaweft-metaweft_v2-2h-20261111T222731Z_test-normal_d6dd0957`.
The user authorized the configured third-party API calls. The run used
`https://az.gptplus5.com/v1`, fresh `no_cache` calls and a disposable response store.
Both memories started empty, with tasks executed sequentially in manifest order.
V1 controls remained unchanged: `qwen3.8-flash`, execution temperature 0.9,
memory temperature 0, model seed 0, environment seed 100, 40 interactions,
`text-embedding-3-large`/1024, and `efficiency_1` retrieval with final k=10.

```bash
smoke_cache=$(mktemp -d /tmp/metaweft-v2-smoke.XXXXXX)
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 \
  --dataset 2h --split test-normal --limit 2 \
  --cache-mode no_cache --cache-path "$smoke_cache/llm.sqlite3" \
  --note "V2 two-task smoke; disposable logs"
```

| Task | Official reward | Interactions | Retrieved items | Input playbook entries | Task-memory operations | Playbook update |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `ff58e36_1` | 1.0 | 22 | 0 | 0 | 3 add | Valid ADD; revision 1, one entry. |
| `ff58e36_2` | 1.0 | 27 | 3 | 1 | 1 revise, 1 skip | Valid UPDATE; revision 2, one entry. |

The process exited successfully and completed 2/2 selected tasks, covering 2/21
Quick2h normal tasks. Official task-goal completion was 100%; scenario-goal
completion was 100% **for the selected tasks only**, not the complete dataset.
No degraded tasks, infrastructure failures, or evidence gaps were reported.

Both adapter outputs were schema-valid. Task two consumed exactly the three
task-memory content strings and complete playbook rule committed by task one,
without memory bookkeeping. Each updater request matched the saved adapter
input and validated guidance, plus revision/edit targeting only; neither contained
execution trajectories, scalar reward, or duplicate context. The second UPDATE
used the correct ID-to-input mapping and preserved both source references.

All 518 captured framework source hashes and 21 V2 prompt-manifest hashes were
verified. The captured descriptor matched the source descriptor, and the executor
and task-memory prompt copies were byte-identical to V1. The finalized format-2
checkpoint and its referenced hashes verified at epoch 2, with three task-memory
items and one V2 playbook entry at revision 2.

The learned rule occupies exploration guidance and remains conditioned on
relationship-based connection tasks. It describes discovery and verification,
but also contains domain-specific procedural details. These observations do not
establish that task-memory drift is eliminated. Memory-selection/pruning entries,
failure/partial-success skips, listwise ranking, and interrupted recovery were
not exercised live; offline fixtures cover those branches. No extra benchmark
tasks, baseline runs, or prompt search were performed.

## Usage

| Phase/component | Physical provider calls |
| --- | ---: |
| Retrieval embedding | 1 |
| Adapter | 2 |
| Execution | 49 |
| Task extraction | 2 |
| Task consolidation embedding | 6 |
| Task consolidation chat | 5 |
| Adaptation update | 2 |
| Total | 67 |

All 67 calls returned reported usage: 60 chat calls and seven embedding calls.
There were zero response-cache hits, inherited attempts, missing-usage calls, or
recorded provider failures. Normalizing usage aliases once per call gives
524,328 input tokens and 29,366 output tokens, 553,694 total. Provider-reported
cached input was 378,112 tokens; this is distinct from response-cache replay.
The existing estimator reported **USD 0.1868161**, using the repository's recorded
historical rates and provider usage. This is an estimate, not a verified invoice.
All workflow phases are included.

## Cleanup and evidence limits

After inspection, the unique smoke-run directory (including native logs, memory
snapshots, source copy and isolated Elasticsearch state), disposable response
cache, and temporary validation/offline logs were removed as requested. All 21
pre-existing run paths and the shared runtime, benchmark assets and response
store remained present. The validation summary, implementation, prompt lineage
and offline tests remain available. Raw smoke evidence was intentionally not
archived; the run identity above is not a durable artifact link.
