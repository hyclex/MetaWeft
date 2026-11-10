# Dual-memory V1 validation — 2026-11-10

Implemented [the requested plan](../../../docs/implementation_plan/ACE_adapt_ReMe_task_design.md)
as `metaweft_v1`, with [the adjacent descriptor](metaweft_v1.variant.json) and
[policy decisions/source lineage](../../src/ace_adapt_reme_task/DESIGNS.md).
This is mechanism validation, not evidence that learned adaptation improves reward.

## Offline checks

The prepared runtime check passed for `origin-py312-59b185a7b247`.
All 281 current offline tests passed: the complete suite at that point contained
254 tests (58.157 seconds), followed by 25 new policy tests and two optional-name
integrity tests. Subsequent focused suites passed against the final changes,
including 51 retrieval/policy tests and 46 policy/framework/playbook/launcher tests.
`git diff --check` passed.

Coverage includes opaque framework payloads; frozen prior inputs and separate
selectors; both-empty adaptation; full-playbook rendering; named task memory;
twenty-candidate ranking followed by final k=10; unchanged embedding/ranking text;
add/skip/revise/defer consolidation; atomic ADD/UPDATE validation; invalid schemas,
evidence pointers and prompt overflows; provider/uncertain-write failures; paired
checkpoint interruption/corruption/restoration; completed-task skipping; canonical
accounting; and compatibility with existing settings and format-1 recovery.
Tests use fake providers and temporary directories, with no billable model calls.

Both dry-runs selected `ff58e36_1`, then `ff58e36_2` from `2h/test-normal`.
V1 resolved the unchanged `efficiency_1` preset with final k=10 and recovery
format 2. Existing `retrieval --preset efficiency_1` retained k=5 and format 1.

```bash
bash metaskill-appworld/run/setup_runtime.sh --check
metaweft_python=$(python3 -B metaskill-appworld/run/runtime_paths.py appworld_python)
"$metaweft_python" -B -m unittest discover -s MetaWeft/tests -v
bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-normal --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset 2h --split test-normal --limit 2 --dry-run
```

## Live two-example smoke

Run identity: `metaweft-metaweft_v1-2h-20261110T175524Z_test-normal_8881ac4e`.
The user explicitly approved sending synthetic benchmark context to the configured
provider `https://az.gptplus5.com/v1`. Both tasks ran sequentially from an empty
task index and empty four-heading playbook. Model controls remained unchanged:
`qwen3.8-flash`, execution temperature 0.9, memory temperature 0, seed 0,
40 interactions, and `text-embedding-3-large` with 1024 dimensions.
Calls used `no_cache`; a disposable cache path changed storage location only.

```bash
bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-normal --limit 2 --cache-mode no_cache --cache-path /tmp/metaweft-v1-smoke-7e749be1/llm.sqlite3 --note 'dual-memory V1 two-task smoke; disposable logs'
```

| Task | Official scalar reward | Interactions | Retrieved items | Task-memory operations | Playbook update |
| --- | ---: | ---: | ---: | --- | --- |
| `ff58e36_1` | 1.0 | 16 | 0 | 2 add | Rejected: explanatory prose and fenced JSON instead of one JSON object. |
| `ff58e36_2` | 0.8333333333333334 (5/6) | 13 | 2 | 1 revise, 1 skip | Rejected: both proposed edits omitted the required `/adaptation/guide` evidence pointer. |

The stream completed 2/2 selected tasks (2/21 source tasks), with 1/2 fully solved:
task-goal completion 50%; subset scenario-goal completion 0%. Both tasks emitted
the environment completion signal, which did not guarantee evaluator success.
Both adapter calls completed with schema-valid guidance, including the initial
both-empty call. Task two consumed the two memories learned from task one.

Both playbook branches were explicitly degraded, applied zero edits, and retained
revision 0 with zero entries. Confirmed task-memory changes still committed as
specified. Both states reached epoch 2; the finalized format-2 checkpoint and all
referenced file hashes were verified after execution. No infrastructure failure
or snapshot evidence gap was reported. No model-format repair calls were made.

This smoke did not exercise successful playbook learning or full-pool ranking.
Task two's two-item pool bypassed ranking as specified. Valid ADD/UPDATE, nonempty
playbook transfer, and ranking are covered by offline fixtures. The two rejected
updates are retained here as negative observations; they do not establish a
benefit from adaptation memory. No extra tasks or prompt search were run.

## Usage and failures

| Phase/component | Physical provider attempts |
| --- | ---: |
| Retrieval embedding | 1 |
| Adapter | 2 |
| Execution | 29 |
| Task extraction | 2 |
| Task consolidation embedding | 6 |
| Task consolidation chat | 4 |
| Adaptation update | 2 |
| Total | 46 |

There were 45 successful provider responses and one embedding rate-limit (HTTP
429) attempt, recovered through the configured retry policy. Cache hits: 0.
Inherited attempts: 0. Known reported usage across all stages was 271,700 input
tokens and 4,560 output tokens, 276,260 total. The failed attempt had unknown
usage; these known totals are not a claim of complete billed usage. Dollar cost
remains unknown because provider rates are unverified. Invalid structured outputs
were still successful, potentially billable provider responses and are included.

## Cleanup and evidence limits

Per the user's explicit cleanup request, the run directory (including native
logs, snapshots, isolated Elasticsearch state, and source copy), the disposable
response-cache directory, and the temporary offline-suite log were removed after
inspection. No historical run or shared runtime/benchmark asset was removed.
This document retains the validation summary; raw smoke evidence is intentionally
not archived and its former paths are not presented as durable evidence links.
The source implementation, prompt parent bytes, descriptors, and offline tests
remain available for reproduction.
