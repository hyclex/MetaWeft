# Memory retrieval

This package implements one read-only retrieval call over one prepared memory
bank. It provides query generation, HNSW or exact candidate search, reciprocal
rank fusion (RRF), and optional listwise selection or utility grading. Bank
preparation, adaptation, memory updates, checkpointing, and run orchestration
remain caller responsibilities. The [v2 implementation contract](../../../docs/implementation_plan/memory_retrieval_framework_revise_01.md)
extends the [original contract](../../../docs/implementation_plan/memory_retrieval_framework.md).
The live AppWorld [shared entrypoint](../../experiment/appworld/README.md) supports
`base`, all nine v2 category presets, and `exact_base` through
`run.sh retrieval --preset NAME`. Expanded `setting_1..6` designs remain available
through the shared API and retrieval-only benchmark.

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
Statuses are `completed`, `empty_memory`, `disabled` (`k=0`), `degraded`, or
v2's `completed_with_fallback`. Errors are null or an object containing `stage`,
`type`, and `message`. A completed fallback returns usable memories and summarizes
the first fallback with type `RankingFallback`; the full trace retains all causes.

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
provider/model/dimensions, and finite vectors before indexing. The caller must
prevent concurrent bank writes during retrieval. Exact search also validates its
complete exported records and vectors; it cannot infer stale text/vector
associations from vector values alone.

## Existing v1 presets

| Preset | Generated queries per branch | Search representation | Normal ranked searches |
| --- | --- | --- | --- |
| `base` | None; exact original task | Combined | 1 |
| `setting_1` | Two general procedural queries | Combined | 3 |
| `setting_2` | Two applicability queries | Applicability | 3 |
| `setting_3` | Same prompt/controls as Setting 1 | Content | 3 |
| `setting_4` | Two general and two applicability queries | Each branch's corresponding field | 6 |
| `setting_5` | Two queries from a training-derived applicability prompt | Applicability | 3 |
| `setting_6` | Same prompt as Setting 5 plus three training demonstrations | Applicability | 3 |

Settings 5–6 inherit Setting 2's controls and retrieval mechanics. Their prompt
authoring evidence and family-disjoint 9/107 query split are recorded in the
[retrieval benchmark](../../../evaluations/appworld_mem_retrieval/README.md#training-derived-prompts-settings-5-and-6).
The split affects evaluation query selection only; the full memory bank remains
unchanged. See the benchmark's saved run artifacts for evaluation coverage.

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

The live default remains identity generation against combined vectors with
`k=5` and candidate count 10. The v1 frozen benchmark retains `k=20` and candidate
count 40. Existing JSON, provider requests, trace schema 1 and failure policies
remain supported; historical runs do not need schema migration.

## V2 presets and independent limits

V2 separates returned search depth, fused candidate depth, ranking-stage output
limit, and final `k`. HNSW's `num_candidates` controls its internal search budget.
For example, `k=5` still sends 80 candidates to utility grading, or 225 candidates
to the cascade selector and 20 to its grader. Final truncation happens last.

All nine category presets use the exact original task and combined vectors,
with RRF constant 20. Their default HNSW depths remain bounded on larger banks.

| Preset | Search/fused depth | ANN count | Ranking |
| --- | --- | --- | --- |
| `performance_1` | 225 | 450 | Max selects 20, then reasoning Flash grades 20. |
| `performance_2` | 225 | 450 | Max selects 20, then nonthinking Flash grades 20. |
| `performance_3` | 80 | 160 | Flash grades all candidates. |
| `efficiency_1` | 20 | 40 | Flash selects/orders 20. |
| `efficiency_2` | 80 | 160 | Max selects 20. |
| `efficiency_3` | 80 | 160 | Flash selects 20. |
| `transferability_1` | 80 | 160 | Same complete configuration as `performance_3`. |
| `transferability_2` | 80 | 160 | Same complete configuration as `efficiency_3`. |
| `transferability_3` | 225 | 450 | Same complete configuration as `performance_1`. |
| `exact_base` | All, capacity 225 | None | Exact cosine control without ranking. |

The nine names represent six ranking configurations. Category assignments come
from historical exact-search evidence; they do not establish HNSW performance or
cross-benchmark transfer. Each JSON has an adjacent `.variant.json` recording its
parent, selection rationale and shared design. The separate `exact_base` control
is outside the nine category settings.

```python
meta = load_preset("performance_1")                         # HNSW depth 225, count 450
meta = load_preset("performance_1", backend="exact")        # Exact top 225
meta = load_preset("efficiency_2", num_candidates=320)       # HNSW depth 80, count 320
meta = load_preset("exact_base")                            # Exact all, capacity 225
```

Exact overrides preserve finite candidate depths and clear inherited ANN counts;
an explicit count with exact search is an error. Exact search exports the full
bank once per call, computes float64 cosine, and breaks score ties by memory ID.
HNSW retains Elasticsearch order and performs only the empty-bank probe and
configured searches. Neither backend silently switches to the other. Exact finite
depths allow banks above 225; only explicit exact/all configurations enforce their
declared bank capacity before model calls. HNSW cannot use an `all` depth.

V2 adds `reranking` and `provider_policy` to the top-level schema. Its `search`
declares backend, depth, ANN count or null, maximum bank size or null, and routes;
`fusion` adds `candidate_limit`. Each ordered ranking stage declares its method,
prompt, model, fields, output limit, small-input policy, generation controls and
response policy. Permitted sequences are empty, listwise, scores, or listwise
then scores. Unknown fields and incompatible depths fail validation before I/O.
Names and descriptions never select runtime behavior.

Listwise selection requests 20 unique local indices. A nonempty input below 20
deliberately skips this stage and retains its order; a following grader still
runs. Utility grading requires an integer 0–100 for every input, preserves
membership and resolves equal grades in input order. Cascade stages renumber
their own inputs independently. Ranking prompts contain only the task, local
indices, and configured memory text; they exclude canonical IDs, references,
labels and upstream reasoning. No semantic deduplication or procedure rewriting
is performed. [Prompt provenance](prompts/v2/provenance.json) identifies the copied
source bytes and correction message.

Flash is `qwen3.8-flash`; Max is `qwen3.8-max-0902`. Calls use explicit model
overrides without changing the supplied client. `efficiency_1` and `exact_base`
require five provider retries; other v2 presets require two, all with 60-second
delays. Construct the retrieval `ModelClient` with the preset's `provider_policy`;
a mismatch fails preflight. Cache mode/path remain caller settings, with
`no_cache` as the default. Reasoning Flash requests a 2048-token thinking budget
and 6144-token total completion cap; the total cap bounds output accounting.

`BasePipeline.prepare_context(..., retrieval_llm=None)` accepts a separate
retrieval client while retaining the existing client for adaptation and updates.
Explicit v2 injection supports identity generation over one combined route.
Valid fallback memories are adapted while `ContextResult.degraded` remains true.
The live requirement for `source_refs` and consolidation's neighbor search remain
unchanged. The AppWorld wrapper supplies this separate client for v2 presets,
preserves their exact provider policies, and records them with the retrieval trace.
Its final `k=5` does not change candidate/ranking depth. `exact_base` retains its
225-memory capacity even as end-to-end memory evolves.

## Traces and failures

`record_trace` receives an independent JSON-serializable trace snapshot before
work and after generation branches, embedding, each search, and fusion. It also
receives partial progress before provider/programming errors propagate.
Snapshots contain version, task, limits, resolved meta, queries, generation and
embedding operations, candidate ranks/raw scores, final RRF scores/contributions,
timing, and completed-stage information. V2 also emits snapshots before external
operations and after ranking decisions. Its trace schema 2 separates full fused
candidates, each stage's input/output mapping and grades, and `ranking`, which is
the final returned prefix. `bank.descriptor` holds caller metadata; exact search
adds the exported count and snapshot hash under `bank.observed`. HNSW does not
manufacture an exact-bank hash. The empty-bank probe and exact export are separate
from ranked search lists. There are no vectors in the trace.

Stable components identify generation as `retrieval.query_generation.<branch>`
and embedding as `retrieval.embedding.batch_0`. Search entries retain the batch
input index; embedding inputs retain query IDs. The caller's canonical recorder
maps these components to logical IDs, physical attempt IDs, and attempt numbers.
The trace callback attaches those references and captured bank/model/prompt
metadata. `prompt_manifest(meta)` exposes selected prompt bytes as UTF-8 text,
package-relative paths, and SHA-256 hashes for run snapshots. Ranking prompt
entries additionally identify `kind="reranking"`. Ranking logical components
are `retrieval.reranking.<stage>.attempt_0` and, if enabled, `attempt_1` for schema
correction. Canonical physical attempt numbers remain separate. Callers attach
references and preserve/hash the response evidence required for recovery.

`k=0` makes no store/model calls. An empty bank makes only the small `match_all`
probe and returns `empty_memory`. Invalid generated JSON/schema or failed search
responses return `degraded` with no records. Query generation and candidate
search receive no schema repair, partial-branch fusion, or input fallback.
Live base continues its existing degraded task policy; the frozen evaluator stops
and retains unsuccessful retrieval.

Ranking stages follow their explicit response policies. The strict Flash 20
setting permits duplicate repair but no wrapper normalization, schema correction
or fallback. The 80-candidate listwise settings permit recognized whole-response
wrappers, bounded duplicate repair, one schema correction and input-order
fallback. Utility grading and validated cascades additionally permit the exact
exhausted-empty-chat exception. Scores never receive duplicate repair. A fallback
uses that stage's input prefix; a later successful stage cannot erase it.

Provider retries remain owned by `ModelClient`. Exhaustion raises
`LLMUnavailableError`, except when the stage explicitly permits the chained
`InvalidResponseError("Chat response contains no text")` fallback. Other provider
exhaustion, logger/artifact errors and programming or bank-integrity failures are
fatal. A trace callback failure propagates unchanged and never causes another
model call. Recovery belongs to the caller at task/query boundaries; no partial
ranking-stage resume is provided. Reports include fallback query coverage and
separate cache-hit source usage from incurred physical-call usage.

## Offline checks

Run from the workspace root; these use fake providers and stores:

```bash
retrieval_python=$(python3 -B metaskill-appworld/run/runtime_paths.py appworld_python)
"$retrieval_python" -B -m unittest discover -s MetaWeft/tests -p 'test_memory_retrieval*.py' -v
```

The retrieval benchmark README documents provider-calling two-query smoke runs
and recovery. Those runs establish operability, not retrieval superiority.
