# MetaWeft on AppWorld

Use `run.sh` from the workspace root. It dispatches to thin `base.py`,
`direct.py`, `retrieval.py`, and `metaweft_v1.py` entrypoints, each referencing its adjacent variant descriptor. See the
[command catalog](../README.md) for runnable check, smoke-test, and collection
examples.

## Setup

MetaWeft reuses the sibling `metaskill-appworld` runtime profile and downloaded
benchmark assets. Prepare that runtime if it is not already available:

```bash
bash metaskill-appworld/run/setup_runtime.sh
bash metaskill-appworld/run/setup_runtime.sh --check
```

Runtime setup can download packages, data, and Elasticsearch. Fresh live launch requires
`OPENAI_API_KEY` and `OPENAI_BASE_URL`, inherited from the environment or loaded
from `raw_logs/.runtime/providers/metaskill-appworld-origin.env`. Inherited values
take precedence. Use `--env-file` for another provider file; secrets are not copied
into manifests. Preserve the declared model settings when comparing variants.

`base` and the retrieval presets start an isolated local Elasticsearch instance for each phase. `direct`
requires only the AppWorld runtime/data assets and does not start Elasticsearch,
construct a memory store, or make embedding calls. Both use the existing prepared
AppWorld Python interpreter, selected automatically by the launcher.

## Options

| Option | Meaning |
| --- | --- |
| Positional entrypoint | `base`, `direct`, `retrieval`, or `metaweft_v1`. |
| `--preset NAME` | Required for `retrieval`: `base`, `performance_1..3`, `efficiency_1..3`, `transferability_1..3`, or `exact_base`. The preset becomes the run setting. |
| `--dataset` | Required: `2h`, `8h`, or `full`, using the sibling dataset mapping. |
| `--split` | `test-normal`, `test-challenge`, or `test-all` (default). |
| `--limit N` | First N tasks per selected split; records partial coverage. |
| `--task-id ID` | One task from a single selected split; mutually exclusive with `--limit`. |
| `--dry-run` | Print the exact plan without writes, setup, or provider calls. |
| `--recover [latest\|RUN_DIRECTORY]` | Recover a compatible failed new-format run in a new linked directory; bare `--recover` selects `latest`. |
| `--env-file PATH` | Provider dotenv file; inherited environment wins. |
| `--cache-mode MODE` | `no_cache`, `cache`, or `timed_cache`; explicit option overrides `METAWEFT_LLM_CACHE_MODE`, then defaults to `no_cache`. |
| `--cache-path PATH` | Shared response database; overrides `METAWEFT_LLM_CACHE_PATH`, then defaults to `raw_logs/.cache/metaweft/llm.sqlite3`. |
| `--data-root PATH` | Existing downloaded AppWorld data directory accepted by the shared runtime rules. |
| `--note TEXT` | Recorded annotation; v2 presets prepend an automatic design description. Recovery preserves the saved description and replaces the annotation when supplied. Does not change variant identity. |

All settings use 40 executor interactions, one primary trial, `qwen3.8-flash` at
temperature `0.9`, model seed `0`, and thinking disabled. AppWorld's environment
random seed is `100`. Base memory calls use
temperature `0`; embeddings use `text-embedding-3-large`, 1024 dimensions. Provider
calls default to one initial attempt plus five retries with a 60-second delay;
edit `service/llm/retry.json` to configure new runs. Invalid nonempty model JSON is not
repaired by an extra LLM call. Refer to [design decisions](../../DESIGNS.md) for
the stage-specific behavior and failure policy.

Chat and embedding calls default to `no_cache`: call the provider and store
successful responses. Select `cache` for exact-request reuse, or `timed_cache`
for responses younger than 1,800 seconds. Run commands from the workspace root.
Resolved cache settings are captured with the run and preserved on recovery;
explicit conflicting recovery options are rejected. Cache hits reuse prior
samples and are excluded from incurred provider usage; choose `no_cache` when a
comparison requires fresh samples. See the [LLM cache API](../../service/llm/README.md#response-cache).

With `test-all`, the launchers run normal and challenge splits concurrently in
separate workers, each with its own memory, outputs, runtime caches, and temporary
paths. Tasks within each split remain sequential. The launcher waits for both
splits and reports failure if either fails; recovery skips completed splits.
Recovery from committed task boundaries is supported. The launchers do not provide
training, arbitrary warm-start memory, within-task resume, repetitions within an
invocation, or parallel tasks within a split. Every invocation gets a unique
output directory; prior runs are not overwritten.

Base pre-task retrieval runs through the shared
[retrieval framework](../../src/memory_retrieval/README.md) using its `base`
configuration. `resolved_settings.json` captures `settings.retrieval_meta` and
the retrieval limit consumed by the pipeline. The original task searches only
the combined vector; one-list RRF preserves its order. Consolidation keeps its
separate five-neighbor search and existing memory updates.

`retrieval --preset NAME` changes only pre-task retrieval. All nine v2 category
presets use unchanged candidate/ranking controls and return up to five memories
to the base adaptation pipeline. A separate retrieval client uses the selected
provider policy: two retries for `transferability_3`, five for `efficiency_1`, both
with 60-second delays. Execution, adaptation and updates retain ordinary retry
settings. Traces record both policies and canonical references to ranking calls.
`retrieval --preset base` is an alias of the existing base setting.

Cold-start memory evolves between tasks and resets per split. Listwise ranking
bypasses banks below 20 memories; a configured grader still runs. `exact_base`
uses exact whole-bank search and fails before paid retrieval calls above its
declared capacity of 225; it does not evict memories or silently increase capacity.
The expanded `setting_1..6` designs remain available only through the
[retrieval benchmark](../../../evaluations/appworld_mem_retrieval/README.md).

## Dual-memory V1

`metaweft_v1` uses [the dual-memory design](../../src/ace_adapt_reme_task/DESIGNS.md)
and [its descriptor](metaweft_v1.variant.json). It starts with empty task memory
and an empty four-heading adaptation playbook. The unchanged `efficiency_1`
retrieval preset returns up to ten task memories; baseline retrieval still
returns five. One adapter call receives those records and the complete playbook,
including when both are empty, and supplies a fixed guide to the existing executor.
After evaluation, independent task and adaptation updates receive the frozen
episode and scalar reward, without private evaluator reports or solutions.
Task items have descriptive names; playbook bullets contain content under
selection, specialization, composition, or verification, with ADD/UPDATE edits.

```bash
bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-normal --limit 2 --dry-run
# Live provider calls, two sequential tasks with an isolated empty memory pair:
bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-normal --limit 2 --cache-mode no_cache
```

V1 writes format-2 checkpoints that commit both verified memory snapshots and the
task position together. Playbook revision changes only for an applied edit batch;
the pair epoch advances for every completed task, including no-op updates.
Recovery restores both states from the same epoch into a new isolated attempt.
Existing settings retain format-1 checkpoints. Invalid update schemas are visible
branch degradation; uncertain writes, required snapshot failures, and exhausted
provider/cache calls stop the stream without committing the task.
V1 does not expose warm starts, retrieval tuning, within-task adaptation, or
playbook pruning. See [V1 validation](VALIDATION_V1.md) for tested coverage.

## Recovery

Repeat the original entrypoint/preset, dataset, split, and `--limit`/`--task-id` selection
with `--recover`. `--recover latest` selects the most recently failed compatible
run; an explicit directory selects that parent. `--dry-run` reports the saved
configuration, lineage, reused task IDs, and remaining IDs without writes or
provider calls. Completed or active recovery lineages and pre-checkpoint runs
are rejected. The `base.py`, `direct.py`, `retrieval.py`, and `run.sh` entrypoints share this API.
The [experiment command catalog](../README.md#recovery) lists `nohup` recovery
commands for `transferability_3` and `efficiency_1`, all three dataset sizes, and every test split, using
either the latest compatible failure or a specific parent directory. Its
[fresh-run examples](../README.md#fresh-runs) cover the same combinations.

Recovery uses the original captured source, prompts, model and cache settings, retry policy,
task order, immutable benchmark inputs, and runtime. Edits to the current checkout
do not change the resumed experiment. Conflicting selection, data, provider URL,
or runtime options are rejected. Credentials may be rotated; API keys are still
loaded from the environment/provider file and are never copied to manifests.
`--note` may annotate the new attempt without changing its experimental identity.

Each split commits `checkpoint.json` after evaluation, post-task updates, native
output validation, and a verified memory export. A task with an unsuccessful
benchmark score can still be committed. Exhausted provider calls in any stage
stop the stream; exhausted cache I/O raises `LLMCacheError` and also stops it.
Cache errors retry with the saved retry budget and delay, without another provider
call. A failed required checkpoint export also stops the stream. Recovery
restores a fresh memory index from the last committed snapshot, skips completed
tasks and phases, and restarts the unfinished task. Partial updates made by that
unfinished task remain in failure evidence and are excluded from restored memory.
Independent splits retain separate memory initialization.

The new run records its parent, origin, generation, and reused tasks. Completed
outputs and evidence are copied; incomplete task attempts are archived under
`recovery_history/`. Imports point into the new copy of the saved source. Runtime
caches, temporary files, AppWorld databases for restarted tasks, and Elasticsearch
state are isolated; the completed-response cache remains shared. Pending full
responses in task `llm_cache_pending` artifacts are restored to that cache before
provider calls, retaining their original timestamps. This does not resume task
execution midstream: `no_cache` still makes fresh calls for restarted work, and
`timed_cache` still enforces expiration. Final aggregate evaluation covers the
original full selection.
Accounting reports cumulative distinct physical calls plus `current_attempt` and
`inherited` subtotals, including failed attempts and unknown usage. Cache hits are
counted separately and do not add incurred tokens or charges.

### Recovery with the consolidation size repair

Ordinary recovery executes the parent's captured code. For the diagnosed
consolidation input-length failure in the original pipeline, the explicitly
scoped [repair utility](../../../tool/recover_metaweft_consolidation_limit.py)
applies the pinned 900,000-byte consolidation guard to a new recovery copy:

```bash
python3 -B tool/recover_metaweft_consolidation_limit.py --preset efficiency_1 --dataset 8h --split test-all --recover /path/to/failed/run --dry-run
# Remove --dry-run to launch; this makes live provider calls.
```

The guard defers oversized memory updates without truncating raw evidence.
The utility validates the original parent, records the exact source diff and
task boundary, and binds the child's checkpoints to the repaired source identity.
Inherited tasks retain their original behavior; reports label this continuation
as a mixed-protocol repair. Parent evidence remains unchanged. See its
[design descriptor](../../../tool/recover_metaweft_consolidation_limit.variant.json).
If the repaired run later fails, use ordinary `run.sh ... --recover` against that
child so its saved repair is preserved; do not reapply this one-time utility.

## Outputs and interpretation

```text
raw_logs/metaweft-<setting>-<2h|8h|full>/experiment/output/<UTC>_<split>_<id>/
  run_manifest.json
  variant.json
  resolved_settings.json
  runtime.json
  source_provenance.json
  data_provenance.json
  console.log
  source/MetaWeft/
  source/project/
  experiments/outputs/<test-normal|test-challenge>/
    run_manifest.json
    tasks/<task-id>/{dbs,checkpoints,logs,evaluation,version,misc}/
    evaluations/<dataset>.json
    evaluations/<dataset>.txt
    baseline_artifacts/metaweft/
    accounting.json
```

Task `logs/lm_calls.jsonl` is the canonical provider-call record for all stages,
including embeddings and failed/retried attempts. It also records cache hits with
`attempt=0` and `metadata.extra.cache`; physical attempts start at one. Native AppWorld traces, API
calls, databases, and official task reports retain the established layout.
Memory-stage artifacts preserve retrieval, guidance, allowed evidence, update
decisions, and memory evolution. The run snapshots source, prompts, descriptor,
resolved controls, task order, and runtime identity.

Each task's `retrieval.json` retains its top-level `records`, `status`, and
string-or-null `error`. Its versioned `trace` adds resolved retrieval metadata,
bank/model/prompt references, query and search details, raw search scores, RRF
contributions, timing, and partial progress on failure. Trace call references
identify the corresponding calls in canonical `lm_calls.jsonl`, including retries
and explicitly marked cache hits. Vectors remain in memory snapshots, not retrieval traces.

`completed` means the requested stream and evaluation finished. It does not mean
all tasks were solved or all memory stages succeeded. Check `degraded_task_ids`
and `degraded_reasons` and follow their task artifacts. Missing usage is unknown;
unverified provider prices leave dollar cost unknown. Subset SGC covers only
selected variants and must not be presented as full-split SGC.

Fresh initial memory must be verified empty. Unavailable snapshots are recorded
as missing evidence, never as empty memory. A failed per-task checkpoint snapshot
prevents that task from being committed and stops the stream. An optional final
snapshot failure remains in phase `evidence_gaps`; the already verified task
checkpoint remains available. The collector displays gaps and recovery lineage.

Run `bash tool/collect_appworld.sh` to regenerate the separate MetaWeft table in
`res/appworld_2h.md`, `res/appworrld_8h.md`, and `res/appworkd_full.md`. The tables link to phase manifests and official aggregate
evaluation files. Consult [integration decisions](DESIGNS.md) before changing
these contracts.

See [v1 validation](VALIDATION.md) for the completed two-task smoke runs, the
retained initial degraded run, and tested limitations.
