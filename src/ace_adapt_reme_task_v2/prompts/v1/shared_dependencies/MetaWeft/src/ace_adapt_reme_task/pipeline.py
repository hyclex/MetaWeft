"""Concrete V1 facade; AppWorld retains execution and pair-commit authority."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from MetaWeft.src.framework import DualMemoryPipeline, EpisodeEvidence, StateRef
from .adapter import Adapter
from .adaptation_memory import AdaptationMemory
from .common import FULL_PROMPT_MAX_BYTES, write_json
from .config import ADAPTATION_SCHEMA, TASK_SCHEMA, PROMPTS
from .evidence import evidence_view, reference_path, task_view
from .task_memory import TaskMemory


class V1Pipeline:
    prompt_directory = PROMPTS
    adapter_class = Adapter
    adaptation_memory_class = AdaptationMemory
    adaptation_schema = ADAPTATION_SCHEMA
    component_prefix = "metaweft_v1"

    def __init__(self, store, run_root, prompts=None, *, retrieval_k=10, retrieval_meta=None,
                 playbook=None, task_state_ref=None, adaptation_state_ref=None,
                 full_prompt_max_bytes=FULL_PROMPT_MAX_BYTES):
        self.run_root = Path(run_root).resolve()
        self.prompts = Path(prompts) if prompts else self.prompt_directory
        self.templates = {name: (self.prompts / f"{name}.txt").read_text() for name in
                          ("adaptation", "task_extraction", "task_consolidation", "adaptation_update")}
        self.task_channel = TaskMemory(store, self.run_root, self.templates, retrieval_k=retrieval_k,
                                       retrieval_meta=retrieval_meta, max_bytes=full_prompt_max_bytes)
        self.adaptation_channel = self.adaptation_memory_class(self.run_root, self.templates, playbook,
                                                               max_bytes=full_prompt_max_bytes)
        adapter = self.adapter_class(self.run_root, self.templates["adaptation"], max_bytes=full_prompt_max_bytes)
        self.framework = DualMemoryPipeline(self.task_channel, self.adaptation_channel, adapter)
        self.store, self.retrieval_meta, self.retrieval_k = store, self.task_channel.retrieval_meta, retrieval_k
        self.prior_pair = None
        if task_state_ref is not None or adaptation_state_ref is not None:
            self.set_state_refs(task_state_ref, adaptation_state_ref)

    @property
    def playbook(self):
        return deepcopy(self.adaptation_channel.playbook)

    def set_state_refs(self, task, adaptation):
        pair = {"task": task if isinstance(task, StateRef) else StateRef(**task),
                "adaptation": adaptation if isinstance(adaptation, StateRef) else StateRef(**adaptation)}
        if pair["task"].stream_id != pair["adaptation"].stream_id or pair["task"].epoch != pair["adaptation"].epoch:
            raise ValueError("Prior memory pair must share stream and epoch")
        for role, schema in (("task", TASK_SCHEMA), ("adaptation", self.adaptation_schema)):
            ref = pair[role]
            if (ref.role != role or ref.schema != schema or type(ref.epoch) is not int or ref.epoch < 0
                    or not isinstance(ref.stream_id, str) or not ref.stream_id):
                raise ValueError("Invalid V1 state reference")
        self.prior_pair = pair
        self._verify_pair()

    def _verify_pair(self):
        if self.prior_pair is None:
            raise ValueError("V1 requires a verified committed pair before preparation")
        for role, ref in self.prior_pair.items():
            path = reference_path(self.run_root, ref.snapshot)
            if hashlib.sha256(path.read_bytes()).hexdigest() != ref.sha256:
                raise ValueError("Prior state snapshot hash changed")
            snapshot = json.loads(path.read_text())
            if any(snapshot.get(key) != expected for key, expected in
                   (("role", role), ("schema", ref.schema), ("stream_id", ref.stream_id),
                    ("state_epoch", ref.epoch), ("status", "verified"))):
                raise ValueError("Prior state snapshot differs from its reference")
            if role == "adaptation" and snapshot["playbook"] != self.playbook:
                raise ValueError("Active playbook differs from committed snapshot")

    def prepare_context(self, task_spec, *, llm, artifacts, trace_context=None, call_refs=None, retrieval_llm=None):
        self._verify_pair()
        services = {"llm": llm, "retrieval_llm": retrieval_llm, "call_refs": call_refs if call_refs is not None else {},
                    "trace_context": trace_context or {}, "component_prefix": self.component_prefix}
        return self.framework.prepare(task_view(task_spec), self.prior_pair, services=services, artifacts=artifacts)

    def post_task(self, evidence, *, prepared, llm, artifacts, call_refs=None):
        prepared.verify()
        self._verify_pair()
        current = evidence_view(evidence)
        if current["task_spec"] != prepared.task_spec:
            raise ValueError("Episode evidence task differs from prepared task")
        if prepared.prior_pair != {key: ref.to_dict() for key, ref in self.prior_pair.items()}:
            raise ValueError("Episode prior pair differs from committed state")
        artifacts = Path(artifacts)
        evidence_path = artifacts / "evidence.json"
        if evidence_path.exists():
            if json.loads(evidence_path.read_text()) != current:
                raise ValueError("Immutable task evidence already exists with different content")
        else:
            write_json(evidence_path, current)
        event_path = reference_path(self.run_root, prepared.adaptation_event_ref)
        if json.loads(event_path.read_text()) != prepared.adaptation_event:
            raise ValueError("Preserved adaptation event differs from prepared event")
        path = artifacts / "episode_evidence.json"
        reference = str(path.resolve().relative_to(self.run_root))
        payload = {"schema_version": 1, **current, "prior_pair": deepcopy(prepared.prior_pair),
                   "prepared_inputs_sha256": prepared.inputs_sha256,
                   "task_evidence_ref": str(evidence_path.resolve().relative_to(self.run_root)),
                   "retrieved_memories": deepcopy(prepared.task_prepared.payload),
                   "retrieval_status": prepared.task_prepared.status,
                   "task_preparation": prepared.task_prepared.to_dict(),
                   "adaptation_preparation": prepared.adaptation_prepared.to_dict(),
                   "prior_playbook": deepcopy(prepared.adaptation_prepared.payload),
                   "adaptation": deepcopy(prepared.adaptation_event),
                   "adaptation_event_ref": prepared.adaptation_event_ref}
        write_json(path, payload)
        episode = EpisodeEvidence(payload, reference)

        def record_failure(result, active, error):
            value = result.to_dict()
            value[active].update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(artifacts / "pair_update_status.json", value)

        result = self.framework.update(episode, self.prior_pair,
                                       services={"llm": llm, "call_refs": call_refs if call_refs is not None else {},
                                                 "component_prefix": self.component_prefix},
                                       artifacts=artifacts, record_failure=record_failure)
        write_json(artifacts / "pair_update_status.json", result.to_dict())
        return result
