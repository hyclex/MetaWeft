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
| `service/llm/` | Provider calls, embeddings, bounded retries, and an injected call recorder. |
| `service/elasticsearch/` | Storage, retrieval primitives, confirmed writes, and complete exports. |
| `src/base/` | The reference memory representation and retrieve/adapt/reflect/consolidate policy. |
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

## Reference comparison

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
