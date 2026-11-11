# Content-only adaptation and successful-event playbook learning, V2

The hypothesis is that removing memory bookkeeping from adaptation input and
learning adaptation rules from successful input-to-guide transformations reduces
context waste and task-knowledge drift. The parent and comparison is
`metaweft_v1`, using the same models, task order, empty initialization, retrieval
and task-memory update policy. These combined changes are a system comparison;
a two-task smoke validates operation, not a performance improvement.

`V2Pipeline` subclasses the V1 facade through explicit adapter, adaptation channel,
schema, prompt-path and component-prefix hooks. It reuses V1 TaskMemory, the
shared framework, retrieval and provider services, and the same AppWorld harness.
The parent is MetaWeft revision `1abb3419b94106b9c413575d72f372206af86d53`.
`prompts/v1/lineage.json` records recoverable parent bytes and separately captures
the local shared compatibility hooks. Task extraction, consolidation and executor
prompts are byte-identical copies. Run snapshots preserve the executing source,
resolved prompts and settings. This package is not standalone.

Task memory keeps V1 named records, stable IDs, applicability, content, source
references and embeddings. Retrieval remains efficiency_1 with final k=10.
Both memory preparation artifacts and immutable episode evidence preserve all
stored fields. Only the adapter input changes: retrieved memories become a list
of content strings, and the complete playbook becomes a dictionary of content
lists under memory_selection, task_specific_pruning and exploration_guidance.
Entries are ordered by stable ID within each heading, but IDs and references do
not appear in the adapter request. Only one resulting guidance string enters the
executor and remains fixed throughout the task.

The playbook uses schema 2 with the same ID, revision, source-reference and atomic
ADD/UPDATE mechanics as V1. Its rules describe how to filter retrieval noise,
prune and specialize useful content, or identify missing information and direct
exploration. They should not become another store of concrete task procedures.
The complete playbook remains available even when retrieval is empty, and the
adapter still runs when both memories are empty. Empty guidance is valid.

Task-memory learning still uses the task, trace and official scalar reward for
all outcomes. Adaptation learning is independently eligible only when the reward
equals one and the adapter returned a valid completed result. The reward is the
official passed-test fraction, not the executor completion flag. Other outcomes
write a skipped update artifact with unchanged state, no model call and no
degradation solely due to the skip.

The adaptation updater receives the frozen adapter input once and its validated
guidance output. No trajectory, reward, evaluation detail, reference solution or
duplicate raw messages enter that request. The prior revision and a stable-ID
to input-location map are the only extra operational metadata, allowing targeted
UPDATE without duplicating playbook content. Every proposed edit must cite the
output guidance and a substantive string in the adapter input through valid JSON
pointers. Metadata or whole containers cannot ground edits. Cold-start evidence
may cite the task itself. A successful task does not establish which adaptation
decision caused success; prompts require support in the input-to-guide relation.

All edits validate before any state changes. A nonempty valid batch advances the
revision once, retains history, and adds source references to applied edits.
Invalid JSON/schema and oversized proposals degrade only the memory branch;
provider, cache, artifact and uncertain-write failures remain fatal. The full
adapter and updater requests use the parent's byte guard without truncation.
The resulting-playbook guard counts its actual JSON encoding. Unchanged task
learning failure policies and source-evidence rules remain in force.

Both memories start empty independently per run/split, persist in sequential task
order, and publish together at completed-task boundaries. Checkpoint format 2 is
retained; adaptation schema `metaweft-adaptation-memory-v2` prevents interchange
with V1 playbooks. No historical-evidence selection, deletion, warm-start, new
retrieval policy, or within-task re-adaptation is introduced.

Offline acceptance is in `tests/test_metaweft_v2.py` and
`tests/test_adaptation_memory_v2.py`, alongside shared launcher, worker and
paired-recovery tests. Live validation is exactly two normal-split tasks; the
experiment validation record reports observed coverage and requested log cleanup.
