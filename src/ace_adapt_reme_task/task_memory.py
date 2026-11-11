"""Bounded derivation of base extraction/consolidation; retrieval stays shared.

Parent: src/base/pipeline.py, captured with hashes under prompts/v1/parents.
Names and frozen adapter context are V1 additions; uncertain writes are fatal.
"""

from copy import deepcopy
import math
from pathlib import Path
import time
import uuid

from MetaWeft.service.elasticsearch import StoreSearchError
from MetaWeft.src.base.pipeline import memory_index_body as parent_index_body
from MetaWeft.src.framework import PreparedMemory, BranchUpdateResult
from MetaWeft.src.memory_retrieval import load_preset, retrieve, validate_meta
from .common import FULL_PROMPT_MAX_BYTES, SchemaError, PromptOverflow, call_model, text, write_json
from .evidence import evidence_view, memory_view, reference_path, select_task_evidence, task_projection


def memory_index_body(dimensions=1024):
    body = parent_index_body(dimensions)
    body["mappings"]["properties"]["name"] = {"type": "text"}
    return body


def validate_record(record):
    if not isinstance(record, dict):
        raise SchemaError("Task memory must be an object")
    for key in ("id", "name", "when_to_use", "content"):
        text(record.get(key), key)
    if (not isinstance(record.get("source_refs"), list) or not record["source_refs"]
            or any(not isinstance(ref, str) or not ref.strip() or "#" in ref for ref in record["source_refs"])):
        raise SchemaError("Task memory requires fragment-free source references")
    return memory_view(record)


class TaskMemory:
    def __init__(self, store, run_root, templates, *, retrieval_k=10, retrieval_meta=None,
                 max_bytes=FULL_PROMPT_MAX_BYTES):
        if store is None or not callable(getattr(store, "query", None)):
            raise ValueError("V1 requires a task retrieval store")
        self.retrieval_meta = deepcopy(retrieval_meta) if retrieval_meta is not None else load_preset("efficiency_1")
        validate_meta(self.retrieval_meta, k=retrieval_k)
        if retrieval_k != 10 or self.retrieval_meta != load_preset("efficiency_1"):
            raise ValueError("V1 requires the unchanged efficiency_1 preset with k=10")
        self.store, self.run_root, self.templates = store, Path(run_root).resolve(), templates
        self.retrieval_k, self.max_bytes = retrieval_k, max_bytes

    def prepare(self, task_spec, prior_state, *, services, artifacts):
        path = Path(artifacts) / "retrieval.json"
        saved_trace = {}

        def record_trace(trace):
            nonlocal saved_trace
            saved_trace = deepcopy(trace)
            context = deepcopy(services.get("trace_context", {}))
            if "bank" in context:
                saved_trace["bank"]["descriptor"] = context.pop("bank")
            saved_trace.update(context)
            saved_trace["prior_state"] = prior_state.to_dict()
            refs = deepcopy(services.get("call_refs", {}))
            saved_trace["call_references"] = refs
            for query in saved_trace.get("queries", []):
                query["generation_calls"] = refs.get(query.get("generation_component"), [])
            for search in saved_trace.get("searches", []):
                search["embedding_calls"] = refs.get(search.get("embedding_component"), [])
            for generation in saved_trace.get("generations", []):
                generation["calls"] = refs.get(generation.get("component"), [])
            if saved_trace.get("embedding"):
                saved_trace["embedding"]["calls"] = refs.get(saved_trace["embedding"].get("component"), [])
            for stage in saved_trace.get("reranking", []):
                for attempt in stage.get("attempts", []):
                    attempt["calls"] = refs.get(attempt.get("component"), [])
            write_json(path, {"records": [], "status": saved_trace["status"],
                              "error": saved_trace.get("error"), "trace": saved_trace})

        retrieval = retrieve(task_spec["instruction"], self.retrieval_k, store=self.store,
                             llm=services.get("retrieval_llm") or services["llm"],
                             meta=self.retrieval_meta, record_trace=record_trace)
        records = []
        error = retrieval.error["message"] if retrieval.error else None
        status = retrieval.status
        try:
            if status != "degraded":
                records = [validate_record(record) for record in retrieval.records]
        except SchemaError as failure:
            records, status, error = [], "degraded", str(failure)
            saved_trace.update(status=status, error={"stage": "record_validation", "message": error,
                                                     "type": "SchemaError"})
        write_json(path, {"records": records, "status": status, "error": error, "trace": saved_trace})
        return PreparedMemory("task", prior_state, records, status, error,
                              (str(path.resolve().relative_to(self.run_root)),))

    select_evidence = staticmethod(select_task_evidence)

    def _embed(self, llm, value, event, services):
        component = f"{services.get('component_prefix', 'metaweft_v1')}.task_embedding"
        refs = services.get("call_refs", {})
        start = len(refs.get(component, []))
        try:
            vectors = llm.embed([value], phase="task_consolidation", component=component)
        finally:
            event.setdefault("embedding_calls", []).extend(deepcopy(refs.get(component, [])[start:]))
        dimensions = getattr(llm, "dimensions", 1024)
        if (not isinstance(vectors, list) or len(vectors) != 1 or not isinstance(vectors[0], list)
                or len(vectors[0]) != dimensions
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in vectors[0])):
            raise SchemaError("Embedding service returned an invalid vector")
        return vectors[0]

    def _neighbors(self, vector):
        response = self.store.query({"size": 5, "knn": {"field": "embedding", "query_vector": vector,
                                                        "k": 5, "num_candidates": 10}})
        try:
            records = [{**hit["_source"], "id": hit["_id"]} for hit in response["hits"]["hits"]]
            for record in records:
                for key in ("id", "name", "when_to_use", "content"):
                    text(record.get(key), key)
                record.setdefault("source_refs", [])
        except (KeyError, TypeError) as error:
            raise SchemaError("Memory search returned invalid records") from error
        return records

    def _source_evidence(self, records, current_ref, current):
        import json

        sources, missing = {current_ref: evidence_view(current)}, []
        for record in records:
            if not isinstance(record["source_refs"], list) or not record["source_refs"]:
                missing.append(f"memory:{record['id']}:no_source_refs")
                continue
            for reference in record["source_refs"]:
                if not isinstance(reference, str):
                    missing.append(f"memory:{record['id']}:invalid_source_ref")
                    continue
                if reference in sources:
                    continue
                try:
                    if "#" in reference:
                        raise ValueError("Task source references cannot contain fragments")
                    sources[reference] = evidence_view(json.loads(reference_path(self.run_root, reference).read_text()))
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    missing.append(reference)
        return sources, sorted(set(missing))

    @staticmethod
    def _decision(value, neighbors):
        if (set(value) != {"operation", "target_id", "name", "when_to_use", "content", "reason"}
                or not isinstance(value["operation"], str) or value["operation"] not in {"add", "skip", "revise", "defer"}):
            raise SchemaError("Consolidation returned an invalid decision schema")
        text(value["reason"], "reason")
        target, operation = value["target_id"], value["operation"]
        identifiers = {record["id"] for record in neighbors}
        if target is not None and not isinstance(target, str):
            raise SchemaError("Consolidation target_id must be a string or null")
        if operation in {"skip", "revise"} and target not in identifiers:
            raise SchemaError("Consolidation target must identify a neighbor")
        if operation == "add" and target is not None:
            raise SchemaError("Add must have null target")
        if operation == "defer" and target is not None and target not in identifiers:
            raise SchemaError("Defer target must be null or identify a neighbor")
        for field in ("name", "when_to_use", "content"):
            if operation == "revise":
                text(value[field], field)
            elif value[field] is not None:
                raise SchemaError("Only revise may supply replacement text")
        return value

    def update(self, episode, prior_state, *, services, artifacts):
        started = time.monotonic()
        artifacts = Path(artifacts)
        reflection_path, updates_path = artifacts / "task_reflection.json", artifacts / "task_updates.json"
        reflection_ref = str(reflection_path.resolve().relative_to(self.run_root))
        updates_ref = str(updates_path.resolve().relative_to(self.run_root))
        current = task_projection(episode)
        current_ref = episode.payload["task_evidence_ref"]
        result = BranchUpdateResult("task", selected_evidence_refs=[episode.reference],
                                    proposed_refs=[reflection_ref], applied_refs=[updates_ref], before=prior_state.to_dict())
        reflection = {"candidates": []}
        candidates = []
        try:
            proposal = call_model("task_extraction", self.templates["task_extraction"], current,
                                  services=services, event=reflection,
                                  save=lambda: write_json(reflection_path, reflection), max_bytes=self.max_bytes)
            if set(proposal) != {"candidates"} or not isinstance(proposal["candidates"], list) or len(proposal["candidates"]) > 3:
                raise SchemaError("Task extraction requires zero to three candidates")
            for candidate in proposal["candidates"]:
                if not isinstance(candidate, dict) or set(candidate) != {"name", "when_to_use", "content"}:
                    raise SchemaError("Task candidate requires exactly name, when_to_use and content")
                candidates.append({"id": uuid.uuid4().hex, **{key: text(candidate[key], key) for key in candidate},
                                   "source_refs": [current_ref]})
            reflection.update(candidates=candidates, status="completed")
        except (SchemaError, PromptOverflow) as error:
            candidates = []
            result.status, result.degraded, result.error = "degraded", True, str(error)
            reflection.update(candidates=[], status="budget_limited" if isinstance(error, PromptOverflow) else "degraded",
                              error=str(error))
        write_json(reflection_path, reflection)

        def save_updates():
            value = result.to_dict()
            value["duration_seconds"] = time.monotonic() - started
            write_json(updates_path, value)

        save_updates()
        for index, candidate in enumerate(candidates):
            event = {"candidate": candidate, "before": None, "after": None, "status": "pending",
                     "write_confirmation": None, "embedding_calls": []}
            result.decisions.append(event)
            save_updates()
            try:
                vector = self._embed(services["llm"], candidate["when_to_use"] + "\n\n" + candidate["content"], event, services)
                event["candidate_embedding"] = vector
                neighbors = self._neighbors(vector)
                sources, missing = self._source_evidence(neighbors, current_ref, current)
                event.update(neighbors=[memory_view(record) for record in neighbors],
                             evidence_refs=list(sources), consulted_evidence=deepcopy(sources), missing_evidence=missing)
                if missing:
                    event.update(operation="defer", status="completed", reason="Referenced evidence is missing or invalid")
                    save_updates()
                    continue
                model_event = event["call"] = {}
                try:
                    value = call_model("task_consolidation", self.templates["task_consolidation"],
                                       {"candidate": candidate, "neighbors": event["neighbors"], "evidence": sources},
                                       services=services, event=model_event, save=save_updates, max_bytes=self.max_bytes)
                except PromptOverflow as error:
                    event.update(operation="defer", status="budget_limited", reason=str(error))
                    result.status, result.degraded, result.error = "degraded", True, str(error)
                    save_updates()
                    continue
                decision = self._decision(value, neighbors)
                event.update(decision)
                document = None
                if decision["operation"] == "add":
                    document = {**candidate, "embedding": vector}
                elif decision["operation"] == "revise":
                    before = next(record for record in neighbors if record["id"] == decision["target_id"])
                    event["before"] = deepcopy(before)
                    revised_vector = self._embed(services["llm"], decision["when_to_use"] + "\n\n" + decision["content"], event, services)
                    document = {**before, **{key: decision[key] for key in ("name", "when_to_use", "content")},
                                "source_refs": list(dict.fromkeys([*before["source_refs"], *sources])), "embedding": revised_vector}
                if document is not None:
                    event.update(proposed_document=deepcopy(document), write_confirmation="pending")
                    save_updates()
                    event["after"] = self.store.put(document["id"], document)
                    if event["after"] != document:
                        from MetaWeft.service.elasticsearch import StoreWriteError
                        raise StoreWriteError("put", document["id"])
                    event["write_confirmation"] = "confirmed"
                event["status"] = "completed"
            except (SchemaError, StoreSearchError) as error:
                event.update(status="degraded", error=str(error))
                result.status, result.degraded, result.error = "degraded", True, str(error)
                result.decisions.extend({"candidate": remaining, "operation": "not_processed",
                                         "reason": "Earlier candidate failed; confirmed writes are retained"}
                                        for remaining in candidates[index + 1:])
                save_updates()
                break
            except Exception as error:
                event.update(status="failed", error=f"{type(error).__name__}: {error}",
                             uncertain_write=bool(getattr(error, "uncertain", False)))
                save_updates()
                raise
            save_updates()
        result.after = {"draft": updates_ref, "confirmed_writes": sum(event.get("write_confirmation") == "confirmed"
                                                                       for event in result.decisions)}
        save_updates()
        return result
