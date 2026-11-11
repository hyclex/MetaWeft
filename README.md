# MetaWeft

MetaWeft is a research framework for solving ordered task streams with evolving
procedural memory. It separates reusable provider/storage tools from experimental
choices about memory structure, retrieval, adaptation, execution context, evidence,
and updates. The first reference variant is `src/base`; `direct` provides a matched
execution setting with memory retrieval and post-update disabled.

Read the workspace [research workflow](../docs/Research_Workflow.md),
[blueprint](../docs/blueprint_MetaWeft.md), and this framework's
[design decisions](DESIGNS.md) before changing a variant. The
[proposal](../refs/Meta_skill_Library.pdf) motivates the research; its exploratory
ideas are not additional implementation requirements.

## Structure

| Location | Responsibility |
| --- | --- |
| `service/llm/` | Provider calls, embeddings, persistent response caching, bounded retries, and an injected call recorder. |
| `service/elasticsearch/` | Storage, retrieval primitives, confirmed writes, and complete exports. |
| `src/base/` | The reference memory representation and retrieve/adapt/reflect/consolidate policy. |
| `src/framework/` | Explicit prepared inputs, episode evidence, and independent memory update contracts. |
| `src/ace_adapt_reme_task/` | Dual-memory V1: ReMe-like task knowledge, a full ACE-derived adaptation playbook, and one fixed guide. |
| `src/ace_adapt_reme_task_v2/` | Dual-memory V2: content-only adapter inputs and playbook learning from successful adapter input/output examples. |
| `src/memory_retrieval/` | Shared query generation, HNSW/exact search, RRF fusion, optional ranking, versioned presets, and retrieval traces. |
| `src/base/prompts/` | Explicit, versioned prompts owned by the baseline. |
| `experiment/appworld/` | Thin setting entrypoints, AppWorld execution integration, run lifecycle, and adjacent variant descriptors. |
| `tests/` | Offline behavior and reporting checks. |

Service READMEs document their APIs; service `DESIGNS.md` files explain constraints
for future modifications. Policies and prompts belong to their variants. New
variants may replace their own representations and control flow while reusing
services and benchmark contracts; v1 does not introduce a plugin registry or a
declarative workflow engine.

## First experiment

Run commands from the workspace root containing `MetaWeft/`,
`metaskill-appworld/`, and `raw_logs/`.

```bash
# Inspect the exact task selection without writes or provider calls.
bash MetaWeft/experiment/appworld/run.sh base --dataset 2h --split test-normal --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh direct --dataset 2h --split test-normal --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh metaweft_v1 --dataset 2h --split test-normal --limit 2 --dry-run
bash MetaWeft/experiment/appworld/run.sh metaweft_v2 --dataset 2h --split test-normal --limit 2 --dry-run
```

See the [experiment command catalog](experiment/README.md) for offline checks,
live two-task examples, and collection. The
[AppWorld README](experiment/appworld/README.md) documents setup, options, output
paths, and interpretation.

Both settings use the existing AppWorld runtime, data, loader, official evaluator,
and canonical LLM logger from the sibling checkout. Every invocation preserves
the relevant working source and resolved configuration and executes workers from
that snapshot. This framework is not independent of that checkout. Mutable state,
caches, logs, and outputs belong under `raw_logs/`, never under source directories.

LLM calls default to `no_cache`: make fresh calls and store successful responses.
For repeated design tests, set `METAWEFT_LLM_CACHE_MODE=cache` to reuse exact
requests, or `timed_cache` to reuse responses younger than 30 minutes. Chat and
embeddings share `raw_logs/.cache/metaweft/llm.sqlite3`; override the database with
`METAWEFT_LLM_CACHE_PATH`. Launchers preserve the resolved settings for recovery.
Only the response store is shared; task state and procedural memory remain
isolated. See the [cache API](service/llm/README.md#response-cache) for precedence,
accounting, and cache-failure recovery.

Provider calls default to five retries after the initial attempt, waiting 60
seconds between attempts. Configure new runs in `service/llm/retry.json`.
Exhausted provider failures stop the stream at its last completed-task checkpoint.
Repeat the original launch selection with `--recover` to continue in a new linked
directory using the saved source, settings and memory. Add `--dry-run` to inspect
the recovery plan. Only runs created with the new checkpoint format are supported;
see [recovery usage](experiment/appworld/README.md#recovery).

## Reference comparison

Base's pre-execution retrieval uses the shared
[retrieval framework](src/memory_retrieval/README.md) with the `base` preset: the
original task searches the combined memory vector, and single-list RRF preserves
the search order. Its resolved retrieval configuration is saved with each run.
Consolidation retains its separate existing neighbor search and update policy.

The framework also provides `setting_1` through `setting_6` for query expansion
and representation comparisons, plus nine v2 category presets for candidate
selection and utility ranking and a separate `exact_base` control. The shared
`run.sh retrieval --preset NAME` entrypoint supports `base`, all nine v2 presets,
and `exact_base` with the same evolving cold-start memory and final retrieval
limit of five. Preset-specific settings and automatic notes distinguish reports
and recovery families. `exact_base` retains its 225-memory capacity and fails
above it. The expanded `setting_1..6` designs remain retrieval-only. Use the
[retrieval benchmark runner](../evaluations/appworld_mem_retrieval/README.md) for
frozen-bank evaluation; see the [command catalog](experiment/README.md) for
end-to-end launches of `transferability_3` and `efficiency_1`.

`base` starts empty for each test split, retrieves up to five memories, adapts them
once, executes with fixed guidance, reflects on the trace and scalar reward, and
consolidates at most three candidates sequentially. It supports add, skip, revise,
and defer decisions. It learns from observed successes and failures without
accessing reference solutions or evaluator assertions.

`direct` uses the same execution controls and a separate editable executor prompt.
It creates no memory index and makes no embedding, adaptation, reflection, or
consolidation calls. Its initial executor instructions match the base setting;
future prompt edits are experimental changes and must be recorded.

This is a system comparison with a ReMe-derived executor, not an exact ReMe
reproduction or an adaptation-only ablation. Completion is separate from task
success. Degraded memory stages remain visible in task artifacts, manifests, and
the MetaWeft report table. Live validation results must be established from actual
run artifacts; no performance claim follows from the framework's existence.

The [v1 validation record](experiment/appworld/VALIDATION.md) links the two-task
live runs, the retained degraded attempt, and offline check coverage.

`metaweft_v2` is a separate child of `metaweft_v1`. Its adapter receives task-memory
content and the complete playbook's rule content without memory bookkeeping.
The playbook covers memory selection, task-specific pruning, and exploration
guidance. Only full official success permits playbook learning, using the frozen
adapter input and output without execution trajectories. Task-memory learning
still uses both successful and unsuccessful episodes. See the
[V2 policy](src/ace_adapt_reme_task_v2/DESIGNS.md) and
[variant descriptor](experiment/appworld/metaweft_v2.variant.json).
