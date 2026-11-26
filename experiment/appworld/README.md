# MetaWeft on AppWorld

Use `run.sh` from the workspace root. It dispatches to thin `base.py`,
`direct.py`, `retrieval.py`, `efficiency_1_emb3072.py`, `metaweft_v1.py`, `metaweft_v2.py`, `contract_check.py`, and `observation_dict.py` entrypoints, each referencing its adjacent variant descriptor. See the
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
| Positional entrypoint | `base`, `direct`, `retrieval`, `efficiency_1_emb3072`, `metaweft_v1`, `metaweft_v2`, `contract_check`, or `observation_dict`. |
| `--preset NAME` | Required for `retrieval`: `base`, `performance_1..3`, `efficiency_1..3`, `transferability_1..3`, or `exact_base`. The preset becomes the run setting. |
| `--dataset` | Required. Choose `2h`, `8h`, `full`, `shuffled_v1_2h`, `shuffled_v1_8h`, or `shuffled_v1_full`. |
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

The original settings use 40 executor interactions, one primary trial, `qwen3.8-flash` at
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

## Shuffled test order

All launchers share the frozen `shuffled_v1` test indexes with ACE and ReMe.
The dataset version identifies the run family and its recovery selection.
`--limit N` selects the first N tasks of the chosen order for each split.

```bash
bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset shuffled_v1_8h --dry-run
smoke_cache=$(mktemp -d /tmp/metaweft-shuffle-smoke.XXXXXX)
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset shuffled_v1_8h --split test-normal --limit 2 --cache-mode no_cache --cache-path "$smoke_cache/llm.sqlite3"
nohup bash MetaWeft/experiment/appworld/run.sh retrieval --preset efficiency_1 --dataset shuffled_v1_8h --cache-mode no_cache > /dev/null 2>&1 < /dev/null &
nohup bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset shuffled_v1_8h --cache-mode no_cache > /dev/null 2>&1 < /dev/null &
```

For the shared three-split concurrency cap, use the
[8h campaign launcher](../../../tool/READEME.md#shuffled-appworld-8h-campaign).

## Native embedding size

`efficiency_1_emb3072` uses the unchanged `efficiency_1` retrieval preset with
3072-dimensional `text-embedding-3-large` vectors, the model's
[native size](https://developers.openai.com/api/docs/guides/embeddings).
Its [descriptor](efficiency_1_emb3072.variant.json) records the comparison with
the 1024-dimensional parent. Keep 40 interactions, top-five adaptation, existing
retrieval/provider controls, and independently empty memory per split.
The automatic note appends `emb3072` to the retrieval description, including on
recovery with a replacement user annotation. Existing entries retain 1024.

```bash
bash MetaWeft/experiment/appworld/run.sh efficiency_1_emb3072 --dataset 8h --split test-all --cache-mode no_cache --dry-run
nohup bash MetaWeft/experiment/appworld/run.sh efficiency_1_emb3072 --dataset 8h --split test-all --cache-mode no_cache > /dev/null 2>&1 < /dev/null &
# Recover only a failed invocation from this variant's own family:
bash MetaWeft/experiment/appworld/run.sh efficiency_1_emb3072 --dataset 8h --split test-all --recover --dry-run
```

Runs live under `raw_logs/metaweft-efficiency_1_emb3072-8h/experiment/output/<run_tag>/`.
`8h/test-all` selects 57 normal and 138 challenge tasks, with parallel split
workers and sequential tasks within each split; it is not a wall-clock timeout.
Recovery retains the saved source, settings, preset, dimensions and committed
memory. It cannot select the 1024-dimensional `efficiency_1` family.

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

## Dual-memory V2

`metaweft_v2` derives from V1 and has its own [descriptor](metaweft_v2.variant.json)
and [policy implementation](../../src/ace_adapt_reme_task_v2/DESIGNS.md). It keeps
V1 task retrieval, task-memory learning, executor and model controls. The adapter
receives retrieved memory content and playbook content without IDs, references or
other stored metadata. Its playbook organizes reusable adaptation rules under
memory selection, task-specific pruning and exploration guidance.

Only official scalar reward exactly 1.0 and a valid completed adaptation call
allow a playbook update. Other outcomes explicitly skip that branch without a
provider call. The updater examines the saved adapter input and validated guidance,
plus the revision and lookup needed to target edits. Execution trajectories and
evaluator details are excluded from that prompt; they remain in raw evidence,
and task-memory learning still uses the established success/failure protocol.

```bash
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-normal --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-normal --limit 2 --cache-mode no_cache
```

The two tasks are `ff58e36_1` and `ff58e36_2`, run sequentially with an initially
empty memory pair. V2 retains checkpoint format 2 with a distinct adaptation
schema, restoring both memories and their epoch together. Recovery requires the
same setting; V1 and V2 do not migrate or share playbooks. See the
[command catalog](../README.md) for disposable smoke storage and cleanup, and
fresh/background/recovery examples for all dataset sizes.

## Search designs

**Contract Check** adds the frozen task-contract and completion reminder to the executor prompt. **Observation Dictionary** stores long, identical observations once per task-consolidation request. The repaired three-arm confirmation recorded 39/72 and 37/72 solves respectively, against 34/72 for the parent. See the [matched results](../../../auto_explore/consolidated_20261119/02_repaired__research__confirmation_analysis.md).

Both derive from dual-memory V2. Their search protocol uses temperature zero throughout, requested model seed 0 or 1 (`--seed`, default 0), AppWorld seed 100, a 65,536-token chat limit including reserved output and an 8,192-token embedding batch limit. Stage output caps are 8,192 for execution and adaptation, 512 for ranking and 4,096 for memory updates. Each new run captures the tokenizer files, descriptor and implementation. The seed and design appear in the result note. [Contract Check descriptor](contract_check.variant.json) · [Observation Dictionary descriptor](observation_dict.variant.json)

Run from the workspace root. Use an explicit failed run directory with `--recover` to restore its committed task and playbook pair. The chat tokenizer comes from the search’s [publisher-tokenizer validation](../../../auto_explore/consolidated_20261119/01_initial__research__context_counter_review.md). The embedding guard uses the captured cl100k_base vocabulary. Each run verifies both local files against the frozen hashes.

```bash
bash MetaWeft/experiment/appworld/run.sh contract_check --dataset 8h --split test-all --seed 0 --dry-run
bash MetaWeft/experiment/appworld/run.sh observation_dict --dataset 8h --split test-all --seed 1 --dry-run
python3 -B tool/tmp/run_search_designs_8h.py --dry-run
```

## Recovery

Repeat the original entrypoint/preset, dataset, split, and `--limit`/`--task-id` selection
with `--recover`. `--recover latest` selects the most recently failed compatible
run; an explicit directory selects that parent. `--dry-run` reports the saved
configuration, lineage, reused task IDs, and remaining IDs without writes or
provider calls. Completed or active recovery lineages and pre-checkpoint runs
are rejected. The `base.py`, `direct.py`, `retrieval.py`, `metaweft_v1.py`, `metaweft_v2.py`, and
`run.sh` entrypoints share this API.
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
    cost_estimate.json                 # generated by the collector/backfill
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
cost estimates use explicit historical rates, not verified billing. Subset SGC covers only
selected variants and must not be presented as full-split SGC.

Fresh initial memory must be verified empty. Unavailable snapshots are recorded
as missing evidence, never as empty memory. A failed per-task checkpoint snapshot
prevents that task from being committed and stops the stream. An optional final
snapshot failure remains in phase `evidence_gaps`; the already verified task
checkpoint remains available. The collector displays gaps and recovery lineage.

Run `bash tool/collect_appworld.sh` to regenerate the separate MetaWeft table in
`res/appworld_2h.md`, `res/appworld_8h.md`, and `res/appworkd_full.md`. The tables link to phase manifests and official aggregate
evaluation files and estimated costs. Consult [integration decisions](DESIGNS.md) before changing
these contracts.

The [cost estimator](../cost.py) migrates the per-million-token formula and usage
validation from `metaskill-appworld`'s ReMe `execution_accounting.usage_cost`.
Qwen rates are the existing October 13 registry values: $0.64 uncached input,
$0.08 cached input, and $2.16 output per million tokens. Embedding rates retain
the October 28 assumptions in the [archived cost script](../../../res/tool_provenance_2026-11-24/tool/get_cost.sh)
of $0.13 input, $0.00845 cached input, and no output charge. The historical registry lacks `qwen3.8-max-0902`;
its November 10 public provider ratios supply $9.60 input, approximately $0.80
cached input, and $28.80 output per million tokens, using the registry's same
ratio conversion. The exact returned ratios are preserved with that model's
rates. These are pinned comparison assumptions. New run snapshots preserve the
estimator and its rates.

New `accounting.json` records include `cost_usd`, `cost_status`, `priced_calls`,
`unpriced_calls`, and the full `pricing_assumptions`, including cumulative,
current-attempt and inherited subtotals. A partial estimate sums only priced
calls; an entirely unpriced stream remains `null`. Response-cache replays have
no new cost, while provider-cached input receives its recorded discount. Failed
responses with usable token usage are included.

Final accounting also verifies thinking controls from the actual saved chat
requests and responses, on both successful and failed phases. The shared
[detector](../thinking.py) writes `thinking_summary.json` and adds
`total_llm_calls`, `no_thinking_llm_calls`, and
`thinking_despite_no_thinking_llm_calls` to new accounting records. Counts include
distinct physical chat attempts across stages and recovery history; embeddings
and response-cache replays are excluded. Nonempty reasoning text/blocks or
positive reasoning-token counters indicate observable thinking, including when
the provider reports zero tokens alongside reasoning text. Missing responses,
unknown controls, and missing telemetry remain explicit in the summary.

Run `python3 -B tool/backfill_thinking_summaries.py` from the workspace root to
add derived summaries to existing completed, failed, or running phases. Existing
raw calls, accounting, manifests, checkpoints, and captured sources stay intact.
The periodic collector refreshes these summaries as logs change, including
workers executing older source snapshots, and reports the three counts as
`Think leak / Off / Calls` immediately before Degradation. Recovery rows include
inherited calls, so parent and recovery counts must not be summed together.

For existing ACE, ReMe and MetaWeft runs, execute
`python3 -B tool/backfill_cost_estimates.py` from the workspace root. It creates
derived `cost_estimate.json` files without modifying original accounting, traces,
checkpoints or captured sources. Collection refreshes these files when source
logs or the estimator change, including for workers executing older snapshots.
Each estimate preserves its source-file signatures, estimator hash, rates and
coverage. Initialization and archived attempts are included, with physical IDs
deduplicated. Recovery rows include inherited spend: do not sum them with their
parent rows as independent runs.

See [v1 validation](VALIDATION.md) for the completed two-task smoke runs, the
retained initial degraded run, and tested limitations.
