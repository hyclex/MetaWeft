# ACE adaptation and ReMe task memory, V1

The hypothesis is that learned adaptation rules improve task guidance enough to
justify their update cost. This implementation establishes the mechanism; it does
not establish a performance benefit. The implementation parent is
`src/base/pipeline.py` at MetaWeft revision
`598aa4f29f456c245a46130af2289dba4589818d`, with the exact inspected source captured
under `prompts/v1/parents/MetaWeft/`. The existing `efficiency_1` live setting is
the operational comparison. Its k=5, unnamed task records, and one-memory update
policy differ, so an unchanged comparison is a system comparison.

`pipeline.V1Pipeline` composes ordinary task-memory, playbook and adapter objects
through `src/framework`. Benchmark loading, execution, evaluation and checkpoint
publication remain in `experiment/appworld`; no second harness or plugin registry
is introduced. See the adjacent AppWorld `metaweft_v1.variant.json` and workspace
implementation plan for the complete protocol.

Task memory is a per-stream Elasticsearch collection of stable ID, descriptive
name, applicability, content, source references and a 1024-dimensional combined
vector. Names do not enter embedding or ranking text. The unmodified resolved
`efficiency_1` retrieval pipeline uses HNSW depth 20, ANN candidates 40, RRF 20,
listwise ranking when at least 20 candidates are available, and final k=10.
Consolidation independently uses five neighbors and ten ANN candidates. Shared
retrieval carries optional names for compatibility; nameless legacy records keep
their previous representation. V1 newly stored records require names.

Adaptation memory is strict JSON with a revision, monotonic ID counter and content
bullets under selection, specialization, composition and verification headings.
IDs and source references are bookkeeping. It has no applicability, title,
embedding, ranking, utility counters or retrieval. The deterministic renderer
retains all content and indents multiline continuations. Every task receives the
complete prior playbook, including empty headings on cold start.

Exactly one logical adapter call combines the whitelisted task, retrieved task
records and full rendered playbook with a fixed system prompt. Both-empty input
still invokes adaptation. Only its guide enters the existing executor and remains
fixed for all interactions. The executor template is a byte-identical copy of the
base template. `prompts/v1/lineage.json` captures parent revisions, actual file
hashes, recoverable source bytes, and transformations. The ACE generator's
playbook-use design becomes guide construction; demonstrations and prior
reflection inputs are omitted. ACE reflection and curation become one update
call, with no evaluator details or detailed reasoning transcript. UPDATE is a
declared extension to the local ADD-only curator. This is not an exact ACE
reproduction.

Both separate selectors choose the current episode at rewards zero, partial or
one. Task extraction sees its task, trace, scalar reward, frozen retrieved records
and guide; it returns zero to three named candidates. Sequential consolidation
preserves the parent's add/skip/revise/defer behavior and complete deduplicated
source evidence. Revisions retain IDs and the union of consulted references;
missing evidence defers a candidate. Later candidates can see earlier confirmed
writes within this branch. No task-memory deletion or eviction is added.

Playbook update receives the same prior rendered playbook, actual adapter input
and output, retrieved records, guide, trace and scalar reward. It never re-reads
the task store after updates. It proposes ADD/UPDATE deltas only. Code validates
the whole batch against the prior revision before applying anything, allocates
IDs, retains before-images and source history, and increments revision once for
a nonempty valid batch. Every edit cites a valid completed guide and an actual
recorded observation through JSON pointers into the preserved input projection.
A failed adapter permits only an empty edit plan. Structural validation verifies
grounding references; it cannot prove causal attribution by the model.

The facade independently copies prepared inputs, hashes them, verifies committed
snapshot references and persists immutable permitted evidence before either
branch runs. Explicit allowlists exclude evaluator objects, reports, assertions
and reference solutions. Only finite official scalar rewards in [0, 1] are valid.
The AppWorld adapter rejects zero-test evaluation before updates.

The complete system and user contents are guarded at 900000 UTF-8 bytes. This is
a conservative byte guard, not a model-token guarantee. Adapter overflow stops
before execution. Extraction or update overflow degrades that branch without
trimming; consolidation overflow defers the candidate. Oversized resulting
playbooks are rejected. No compaction, pruning or extra repair call is hidden in
this limit. Raw inputs and failures are preserved before calls or writes.

Invalid completed model JSON is a visible recoverable schema failure. Task
retrieval degradation still permits full-playbook adaptation. Ordinary neighbor
search failure retains earlier confirmed writes, leaves remaining candidates
unprocessed, and allows the playbook branch to run. Provider/cache exhaustion,
unexpected exceptions, canonical logging or artifact failure, and uncertain
store writes propagate and prevent commit. Fatal branch failures record the other
branch as not run. Pair publication and recovery are owned by checkpoint format 2;
format 1 remains supported for existing settings.

Both memories start empty for each independent run/split and persist in task
order. A pair epoch equals the completed-task count; playbook revision advances
only on a nonempty edit batch. Failed mutable stores are never continued in place.
Recovery restores a verified pair into a new store/output directory while keeping
the original logical stream identity and completed prefix. There is no warm-start
CLI, mid-task resume, task parallelism, historical selector, automatic design
search or within-task re-adaptation.

Offline acceptance lives in `tests/test_metaweft_v1.py`,
`tests/test_adaptation_memory.py`, `tests/test_framework.py`, and the paired
recovery/worker/launcher tests. The scoped two-task live smoke and cleanup outcome
are recorded in `experiment/appworld/VALIDATION_V1.md`; completion is distinct
from benchmark success, and unobserved ranking/edit branches require offline
fixtures rather than additional live tasks.
