# MetaWeft on AppWorld

Use `run.sh` from the workspace root. It dispatches to thin `base.py` and
`direct.py` entrypoints, each referencing its adjacent variant descriptor. See the
[command catalog](../README.md) for runnable check, smoke-test, and collection
examples.

## Setup

MetaWeft reuses the sibling `metaskill-appworld` runtime profile and downloaded
benchmark assets. Prepare that runtime if it is not already available:

```bash
bash metaskill-appworld/run/setup_runtime.sh
bash metaskill-appworld/run/setup_runtime.sh --check
```

Runtime setup can download packages, data, and Elasticsearch. Live launch requires
`OPENAI_API_KEY` and `OPENAI_BASE_URL`, inherited from the environment or loaded
from `raw_logs/.runtime/providers/metaskill-appworld-origin.env`. Inherited values
take precedence. Use `--env-file` for another provider file; secrets are not copied
into manifests. Preserve the declared model settings when comparing variants.

`base` starts an isolated local Elasticsearch instance for each phase. `direct`
requires only the AppWorld runtime/data assets and does not start Elasticsearch,
construct a memory store, or make embedding calls. Both use the existing prepared
AppWorld Python interpreter, selected automatically by the launcher.

## Options

| Option | Meaning |
| --- | --- |
| Positional setting | `base` or `direct`. |
| `--dataset` | Required: `2h`, `8h`, or `full`, using the sibling dataset mapping. |
| `--split` | `test-normal`, `test-challenge`, or `test-all` (default). |
| `--limit N` | First N tasks per selected split; records partial coverage. |
| `--task-id ID` | One task from a single selected split; mutually exclusive with `--limit`. |
| `--dry-run` | Print the exact plan without writes, setup, or provider calls. |
| `--env-file PATH` | Provider dotenv file; inherited environment wins. |
| `--data-root PATH` | Existing downloaded AppWorld data directory accepted by the shared runtime rules. |
| `--note TEXT` | Recorded annotation; it does not change the variant identity. |

Both settings use 40 executor interactions, one primary trial, `qwen3.8-flash` at
temperature `0.9`, model seed `0`, and thinking disabled. AppWorld's environment
random seed is `100`. Base memory calls use
temperature `0`; embeddings use `text-embedding-3-large`, 1024 dimensions. Provider
calls have at most three physical transient attempts; invalid model JSON is not
repaired by an extra LLM call. Refer to [design decisions](../../DESIGNS.md) for
the stage-specific behavior and failure policy.

V1 supports sequential fresh test streams. It does not provide training,
warm-start memory, resume, repetitions within an invocation, or parallel task
execution. Each invocation gets a unique output directory; prior runs are not
overwritten.

## Outputs and interpretation

```text
raw_logs/metaweft-<base|direct>-<2h|8h|full>/experiment/output/<UTC>_<split>_<id>/
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
including embeddings and failed/retried attempts. Native AppWorld traces, API
calls, databases, and official task reports retain the established layout.
Memory-stage artifacts preserve retrieval, guidance, allowed evidence, update
decisions, and memory evolution. The run snapshots source, prompts, descriptor,
resolved controls, task order, and runtime identity.

`completed` means the requested stream and evaluation finished. It does not mean
all tasks were solved or all memory stages succeeded. Check `degraded_task_ids`
and `degraded_reasons` and follow their task artifacts. Missing usage is unknown;
unverified provider prices leave dollar cost unknown. Subset SGC covers only
selected variants and must not be presented as full-split SGC.

Initial memory must be verified empty. Later snapshot failures are recorded as
unavailable evidence, not empty or verified memory. Per-task failures mark that
task degraded; phase-level `evidence_gaps` retain a failed final snapshot even if
all tasks completed. The collector displays those gap counts.

Run `bash tool/collect_appworld.sh` to regenerate the separate MetaWeft table in
`res/raw_collect.md`. The table links to phase manifests and official aggregate
evaluation files. Consult [integration decisions](DESIGNS.md) before changing
these contracts.

See [v1 validation](VALIDATION.md) for the completed two-task smoke runs, the
retained initial degraded run, and tested limitations.
