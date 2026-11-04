# Memory retrieval

This package implements one read-only retrieval call over one prepared memory
bank. It provides query generation, Elasticsearch vector search, and reciprocal
rank fusion (RRF). Bank preparation, adaptation, memory updates, checkpointing,
and run orchestration remain caller responsibilities. See the
[implementation handoff](../../../docs/implementation_plan/memory_retrieval_framework.md)
for the detailed retrieval contract; the current delivery integrates only the
base preset into the live memory pipeline.

## API and preparation

```python
from MetaWeft.src.memory_retrieval import load_preset, retrieve

result = retrieve(
    task_description, 20,
    store=prepared_store,
    llm=model_client,
    meta=load_preset("setting_4", num_candidates=40),
    record_trace=persist_trace,
)
```

`task_description` must be nonempty text; `k` must be a nonnegative integer.
`result` contains `records`, `status`, and `error`. Records contain canonical IDs,
`when_to_use`, `content`, and opaque `source_refs` when present; vectors are omitted.
Statuses are `completed`, `empty_memory`, `disabled` (`k=0`), or `degraded`.
Errors are null or an object containing `stage`, `type`, and `message`.

The caller supplies an existing [ModelClient](../../service/llm/README.md) and one
prepared [ElasticsearchStore](../../service/elasticsearch/README.md), with matching
embedding provider, model, and dimensions. Retrieval never repairs a bank or
embeds its memories. Use `required_representations(meta)` and
`memory_index_body(dimensions, representations=...)` when preparing a fresh index.
The mapping uses cosine/HNSW and preserves source vectors.

| Representation | Cached dataset field | Elasticsearch field | Embedding input |
| --- | --- | --- | --- |
| `combined` | `ReMe_emb` | `embedding` | `when_to_use + "\n\n" + content` |
| `applicability` | `when_to_use_emb` | `when_to_use_embedding` | `when_to_use` |
| `content` | `content_emb` | `content_embedding` | `content` |

The exported `REPRESENTATIONS` definitions use schema-v1 input descriptions.
The frozen evaluator validates cached inputs, exact pool coverage, checksums,
provider/model/dimensions, and finite vectors before indexing.

## Presets and behavior

| Preset | Generated queries per branch | Search representation | Normal ranked searches |
| --- | --- | --- | --- |
| `base` | None; exact original task | Combined | 1 |
| `setting_1` | Two general procedural queries | Combined | 3 |
| `setting_2` | Two applicability queries | Applicability | 3 |
| `setting_3` | Same prompt/controls as Setting 1 | Content | 3 |
| `setting_4` | Two general and two applicability queries | Each branch's corresponding field | 6 |

Every branch includes the exact original task first. Generated strings are
trimmed; whitespace-normalized, case-sensitive duplicates are recorded and
removed without refill calls. Duplicate `(normalized query, representation)`
searches are also removed. These rules can reduce the normal search counts above.
Setting 4 generates both branches before any embedding/search, searches each
branch only against its configured representation, and fuses all lists together.

Distinct searched texts are embedded in one batch. Identical text, including
Setting 4's original task, reuses its vector within the call. Each search returns
up to `k` records with `size=k`, `knn.k=k`, and configured `num_candidates`.
Require `k <= num_candidates <= 10000`. RRF sums `1 / (60 + rank)` over lists
using one-based ranks, then sorts by descending score, best rank, and memory ID.
Return up to `k` unique records; a single list retains its original search order.

JSON presets contain `schema_version`, `name`, `description`, `query_generation`,
`search`, and `fusion`. Branches explicitly choose `identity` or `llm`, while
routes select branches and representations. Labels do not choose behavior.
Unknown keys/methods, unreferenced branches, duplicate routes, invalid counts,
or missing prompt definitions fail validation before calls. LLM branches permit
one to three generated queries, default two. Their strict response contains
only `{"queries": [...]}` with the requested count of nonempty strings.

Live `BasePipeline` remains restricted to identity generation against combined
vectors, with its existing `k=5` and candidate count 10 defaults. Expanded presets
are available through this API and the
[single frozen retrieval runner](../../../evaluations/appworld_mem_retrieval/README.md),
whose defaults are `k=20` and candidate count 40. Expanded live vector maintenance
and setting entrypoints are deferred until a setting is selected for deployment.

## Traces and failures

`record_trace` receives an independent JSON-serializable trace snapshot before
work and after generation branches, embedding, each search, and fusion. It also
receives partial progress before provider/programming errors propagate.
Snapshots contain version, task, limits, resolved meta, queries, generation and
embedding operations, candidate ranks/raw scores, final RRF scores/contributions,
timing, and completed-stage information. The empty-bank probe is separate from
ranked search lists. There are no vectors in the trace.

Stable components identify generation as `retrieval.query_generation.<branch>`
and embedding as `retrieval.embedding.batch_0`. Search entries retain the batch
input index; embedding inputs retain query IDs. The caller's canonical recorder
maps these components to logical IDs, physical attempt IDs, and attempt numbers.
The trace callback attaches those references and captured bank/model/prompt
metadata. `prompt_manifest(meta)` exposes selected prompt bytes as UTF-8 text,
package-relative paths, and SHA-256 hashes for run snapshots.

`k=0` makes no store/model calls. An empty bank makes only the small `match_all`
probe and returns `empty_memory`. Invalid generated JSON/schema or failed search
responses return `degraded` with no records; no repair, fallback, search retries,
or partial-branch fusion is performed. Live base continues its existing degraded
task policy; the frozen evaluator stops and retains the failure.

Provider retries remain owned by `ModelClient`. Exhaustion raises
`LLMUnavailableError`; logger/artifact failures and unexpected programming errors
are fatal. A trace callback failure propagates unchanged and never causes another
model call. Recovery is managed by the caller at task/query boundaries, not here.

## Offline checks

Run from the workspace root; these use fake providers and stores:

```bash
python3 -B -m unittest discover -s MetaWeft/tests -p test_memory_retrieval.py -v
```

The retrieval benchmark README documents provider-calling two-query smoke runs
and recovery. Those runs establish operability, not retrieval superiority.
