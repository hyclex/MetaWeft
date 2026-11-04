"""Variant-owned retrieval, adaptation, evidence and consolidation policy."""

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import json
import math
from pathlib import Path
import uuid

from MetaWeft.service.llm import LLMUnavailableError
from MetaWeft.src.memory_retrieval import load_preset, retrieve, validate_meta


def memory_index_body(dimensions=1024):
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.mapping.exclude_source_vectors": False,
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "when_to_use": {"type": "text"},
                "content": {"type": "text"},
                "source_refs": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector", "dims": dimensions, "index": True,
                    "similarity": "cosine", "index_options": {"type": "hnsw"},
                },
            },
        },
    }


@dataclass
class ContextResult:
    guidance: str = ""
    status: str = "completed"
    degraded: bool = False
    error: str | None = None
    retrieved_ids: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class UpdateResult:
    status: str = "completed"
    degraded: bool = False
    decisions: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class StageFailure(Exception):
    def __init__(self, message, *, uncertain=False):
        super().__init__(message)
        self.uncertain = uncertain


def _call(function, *args, **kwargs):
    """Only provider/storage calls enter this recoverable-error boundary."""
    try:
        return function(*args, **kwargs)
    except LLMUnavailableError:
        raise
    except Exception as error:
        if any(base.__name__ == "LLMLoggingError" for base in type(error).__mro__):
            raise
        raise StageFailure(
            f"{type(error).__name__}: {error}", uncertain=getattr(error, "uncertain", False),
        ) from error


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _object(response):
    try:
        value = json.loads(response.choices[0].message.content)
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise StageFailure("Model response must be a JSON object; no repair was attempted") from error
    if not isinstance(value, dict):
        raise StageFailure("Model response must be a JSON object")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise StageFailure(f"{name} must be a nonempty string")
    return value


def _memory_view(record):
    return {key: record[key] for key in ("id", "when_to_use", "content", "source_refs")}


def _evidence_view(evidence):
    """Construct prompt evidence explicitly; never forward evaluator objects/reports."""
    task = evidence["task_spec"]
    reward = evidence["reward"]
    if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Evidence reward must be an official scalar in [0, 1]")
    return {
        "task_spec": {key: task[key] for key in
                      ("task_id", "instruction", "supervisor", "app_descriptions") if key in task},
        "trace": [{key: step[key] for key in ("step", "code", "observation") if key in step}
                  for step in evidence["trace"]],
        "reward": reward,
    }


class BasePipeline:
    def __init__(self, store, run_root, prompts=None, *, retrieval_k=5, retrieval_meta=None):
        self.store = store
        self.run_root = Path(run_root).resolve()
        self.prompts = Path(prompts) if prompts else Path(__file__).parent / "prompts/v1"
        self.templates = {name: (self.prompts / f"{name}.txt").read_text()
                          for name in ("adaptation", "reflection", "consolidation")}
        self.retrieval_k = retrieval_k
        self.retrieval_meta = deepcopy(retrieval_meta) if retrieval_meta is not None else load_preset("base")
        validate_meta(self.retrieval_meta, k=self.retrieval_k)
        base_meta = load_preset("base")
        if (self.retrieval_meta["query_generation"] != base_meta["query_generation"]
                or self.retrieval_meta["search"]["routes"] != base_meta["search"]["routes"]):
            raise ValueError("Live base supports only identity retrieval over combined memory; "
                             "evaluate expanded presets with the retrieval-only runner")

    def _chat(self, llm, name, payload):
        response = _call(
            llm.chat,
            [{"role": "system", "content": self.templates[name]},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}],
            phase=name, component=f"base.{name}", temperature=0.0,
            seed=0, extra_body={"enable_thinking": False},
        )
        return _object(response)

    def _embed(self, llm, text, phase):
        vectors = _call(llm.embed, [text], phase=phase, component="base.embedding")
        if len(vectors) != 1:
            raise StageFailure("Embedding service returned the wrong vector count")
        return vectors[0]

    def _neighbors(self, vector):
        response = _call(self.store.query, {
            "size": 5,
            "knn": {"field": "embedding", "query_vector": vector, "k": 5, "num_candidates": 10},
        })
        try:
            records = [{**hit["_source"], "id": hit["_id"]} for hit in response["hits"]["hits"]]
            for record in records:
                _memory_view(record)
            return records
        except (KeyError, TypeError) as error:
            raise StageFailure("Memory search returned invalid records") from error

    def prepare_context(self, task_spec, *, llm, artifacts, trace_context=None, call_refs=None):
        artifacts = Path(artifacts)
        saved_trace = {}

        def record_trace(trace):
            nonlocal saved_trace
            saved_trace = deepcopy(trace)
            saved_trace.update(deepcopy(trace_context or {}))
            references = deepcopy(call_refs or {})
            saved_trace["call_references"] = references
            for query in saved_trace.get("queries", []):
                query["generation_calls"] = references.get(query.get("generation_component"), [])
            for search in saved_trace.get("searches", []):
                search["embedding_calls"] = references.get(search.get("embedding_component"), [])
            for generation in saved_trace.get("generations", []):
                generation["calls"] = references.get(generation.get("component"), [])
            if saved_trace.get("embedding"):
                saved_trace["embedding"]["calls"] = references.get(saved_trace["embedding"].get("component"), [])
            error = saved_trace.get("error")
            _write(artifacts / "retrieval.json", {
                "records": [], "status": saved_trace["status"],
                "error": error["message"] if error else None, "trace": saved_trace,
            })

        retrieval = retrieve(task_spec["instruction"], self.retrieval_k, store=self.store, llm=llm,
                             meta=self.retrieval_meta, record_trace=record_trace)
        if any("source_refs" not in record for record in retrieval.records):
            # The shared API permits memories without evidence references, but
            # live base preserves its stronger existing memory schema.
            retrieval.records = []
            retrieval.status = "degraded"
            retrieval.error = {"stage": "search", "type": "StageFailure",
                               "message": "Memory search returned invalid records: missing source_refs"}
            record_trace({**saved_trace, "status": retrieval.status, "error": retrieval.error})
        retrieved = retrieval.records
        result = ContextResult(status=retrieval.status, degraded=retrieval.status == "degraded",
                               error=retrieval.error["message"] if retrieval.error else None,
                               retrieved_ids=[record["id"] for record in retrieved])
        _write(artifacts / "retrieval.json", {"records": [_memory_view(r) for r in retrieved],
                                             "status": result.status, "error": result.error,
                                             "trace": saved_trace})
        if retrieved and not result.degraded:
            try:
                value = self._chat(llm, "adaptation", {
                    "task_spec": {key: task_spec[key] for key in
                                  ("task_id", "instruction", "supervisor", "app_descriptions") if key in task_spec},
                    "memories": [_memory_view(record) for record in retrieved],
                })
                if set(value) != {"guidance"} or not isinstance(value["guidance"], str):
                    raise StageFailure("Adaptation requires exactly one string field: guidance")
                result.guidance = value["guidance"]
            except StageFailure as error:
                result.status, result.degraded, result.error = "degraded", True, str(error)
        _write(artifacts / "adaptation.json", result.to_dict())
        return result

    def _source_evidence(self, records, current_ref, current_evidence):
        evidence = {current_ref: current_evidence}
        missing = []
        for record in records:
            refs = record["source_refs"]
            if not isinstance(refs, list) or not refs:
                missing.append(f"memory:{record['id']}:no_source_refs")
                continue
            for ref in refs:
                if not isinstance(ref, str):
                    missing.append(f"memory:{record['id']}:invalid_source_ref")
                    continue
                path = (self.run_root / ref).resolve()
                if Path(ref).is_absolute() or not path.is_relative_to(self.run_root):
                    missing.append(ref)
                    continue
                if ref in evidence:
                    continue
                try:
                    evidence[ref] = _evidence_view(json.loads(path.read_text()))
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    missing.append(ref)
        return evidence, sorted(set(missing))

    def _decision(self, value, neighbors):
        expected = {"operation", "target_id", "when_to_use", "content", "reason"}
        if (set(value) != expected or not isinstance(value.get("operation"), str)
                or value["operation"] not in {"add", "skip", "revise", "defer"}):
            raise StageFailure("Consolidation returned an invalid decision schema")
        _text(value["reason"], "reason")
        operation, target = value["operation"], value["target_id"]
        if target is not None and not isinstance(target, str):
            raise StageFailure("Consolidation target_id must be a string or null")
        ids = {record["id"] for record in neighbors}
        if operation in {"revise", "skip"} and target not in ids:
            raise StageFailure("Consolidation target must be one of the retrieved memories")
        if operation == "add" and target is not None:
            raise StageFailure("An add decision cannot target an existing memory")
        if operation == "defer" and target is not None and target not in ids:
            raise StageFailure("A defer target must be null or a retrieved memory")
        if operation == "revise":
            _text(value["when_to_use"], "when_to_use")
            _text(value["content"], "content")
        elif value["when_to_use"] is not None or value["content"] is not None:
            raise StageFailure("Only revise may supply replacement memory content")
        return value

    def post_task(self, evidence, *, llm, artifacts):
        artifacts = Path(artifacts)
        current = _evidence_view(evidence)
        evidence_path = artifacts / "evidence.json"
        current_ref = str(evidence_path.resolve().relative_to(self.run_root))
        _write(evidence_path, current)
        result = UpdateResult()
        candidates = []
        try:
            reflected = self._chat(llm, "reflection", {"evidence": current})
            raw = reflected.get("candidates")
            if set(reflected) != {"candidates"} or not isinstance(raw, list) or len(raw) > 3:
                raise StageFailure("Reflection requires a candidates list containing zero to three items")
            for candidate in raw:
                if not isinstance(candidate, dict) or set(candidate) != {"when_to_use", "content"}:
                    raise StageFailure("Each candidate requires when_to_use and content")
                candidates.append({
                    "id": uuid.uuid4().hex,
                    "when_to_use": _text(candidate["when_to_use"], "when_to_use"),
                    "content": _text(candidate["content"], "content"),
                    "source_refs": [current_ref],
                })
        except StageFailure as error:
            result.status, result.degraded, result.error = "degraded", True, str(error)
            candidates = []
        _write(artifacts / "reflection.json", {"candidates": candidates,
                                              "status": result.status, "error": result.error})
        for index, candidate in enumerate(candidates):
            event = {"candidate": candidate, "before": None, "after": None}
            try:
                candidate_vector = self._embed(
                    llm, candidate["when_to_use"] + "\n\n" + candidate["content"], "consolidation")
                neighbors = self._neighbors(candidate_vector)
            except StageFailure as error:
                failure = error
            else:
                sources, missing = self._source_evidence(neighbors, current_ref, current)
                event.update(neighbors=[_memory_view(record) for record in neighbors],
                             evidence_refs=list(sources), missing_evidence=missing)
                if missing:
                    event.update(operation="defer", reason="Referenced evidence is missing or invalid")
                    result.decisions.append(event)
                    _write(artifacts / "updates.json", result.to_dict())
                    continue
                try:
                    decision = self._decision(self._chat(llm, "consolidation", {
                        "candidate": candidate, "neighbors": event["neighbors"], "evidence": sources,
                    }), neighbors)
                    event.update(decision)
                    operation = decision["operation"]
                    if operation == "add":
                        document = {**candidate, "embedding": candidate_vector}
                        event["after"] = _call(self.store.put, candidate["id"], document)
                    elif operation == "revise":
                        before = next(record for record in neighbors if record["id"] == decision["target_id"])
                        event["before"] = before
                        vector = self._embed(llm, decision["when_to_use"] + "\n\n" + decision["content"],
                                             "consolidation")
                        document = {
                            **before, "when_to_use": decision["when_to_use"], "content": decision["content"],
                            "source_refs": list(dict.fromkeys([*before["source_refs"], *sources])),
                            "embedding": vector,
                        }
                        event["after"] = _call(self.store.put, before["id"], document)
                    result.decisions.append(event)
                    _write(artifacts / "updates.json", result.to_dict())
                    continue
                except StageFailure as error:
                    failure = error
            event.update(status="failed", error=str(failure), uncertain_write=failure.uncertain)
            result.decisions.append(event)
            result.status, result.degraded, result.error = "degraded", True, str(failure)
            result.decisions.extend({"candidate": remaining, "operation": "not_processed",
                                     "reason": "An earlier candidate failed; confirmed updates are retained"}
                                    for remaining in candidates[index + 1:])
            break
        _write(artifacts / "updates.json", result.to_dict())
        return result
