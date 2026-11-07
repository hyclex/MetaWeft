"""Read-only retrieval with query generation, candidate fusion and optional ranking."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import time

from MetaWeft.service.elasticsearch import StoreSearchError
from .config import REPRESENTATIONS, prompt_manifest, validate_meta
from .reranking import RankingFailure, run_stage


@dataclass
class RetrievalResult:
    records: list[dict]
    status: str
    error: dict | None = None


class RetrievalFailure(ValueError):
    """Expected invalid generation/search data, without provider repair calls."""


def _normalized(text):
    return " ".join(text.split())


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object key")
        value[key] = item
    return value


def _query(store, body):
    try:
        return store.query(body)
    except StoreSearchError as error:
        raise RetrievalFailure(str(error)) from error


def generate_queries(task, branch, *, llm, prompt=None):
    """Return original-first queries, retaining duplicates as trace decisions."""
    generated = []
    component = None
    if branch["method"] == "llm":
        component = f"retrieval.query_generation.{branch['id']}"
        response = llm.chat(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": json.dumps(
                 {"task": task, "generated_count": branch["generated_count"]},
                 ensure_ascii=False, allow_nan=False)}],
            phase="retrieval", component=component, temperature=0.0,
            seed=0, extra_body={"enable_thinking": False},
        )
        try:
            value = json.loads(response.choices[0].message.content, object_pairs_hook=_json_object)
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise RetrievalFailure("Query generation requires a JSON object; no repair was attempted") from error
        if (not isinstance(value, dict) or set(value) != {"queries"}
                or not isinstance(value["queries"], list)
                or len(value["queries"]) != branch["generated_count"]
                or any(not isinstance(text, str) or not text.strip() for text in value["queries"])):
            raise RetrievalFailure("Query generation requires exactly the requested number of nonempty queries")
        generated = [text.strip() for text in value["queries"]]
    queries, seen = [], {}
    for index, text in enumerate([task, *generated]):
        identifier = f"{branch['id']}.q{index}"
        normalized = _normalized(text)
        duplicate = seen.get(normalized)
        queries.append({
            "id": identifier, "branch_id": branch["id"], "original": index == 0,
            "text": text, "normalized": normalized, "duplicate_of": duplicate,
            "generation_component": component if index else None,
        })
        if duplicate is None:
            seen[normalized] = identifier
    return queries


def _hits(response):
    if not isinstance(response, dict):
        raise RetrievalFailure("Search response must be an object")
    shards = response.get("_shards", {})
    if not isinstance(shards, dict) or response.get("timed_out") or shards.get("failed", 0):
        raise RetrievalFailure("Search response is incomplete")
    hits = response.get("hits")
    if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
        raise RetrievalFailure("Search response requires a hits list")
    return hits["hits"]


def search(store, vector, *, representation, k, num_candidates):
    """Return records and raw-score candidates in the backend's original order."""
    response = _query(store, {
        "size": k,
        "knn": {"field": REPRESENTATIONS[representation]["field"],
                "query_vector": vector, "k": k, "num_candidates": num_candidates},
    })
    hits = _hits(response)
    if len(hits) > k:
        raise RetrievalFailure("Search returned more than k memories")
    records, candidates, seen = [], [], set()
    for rank, hit in enumerate(hits, 1):
        if not isinstance(hit, dict):
            raise RetrievalFailure("Search hits must be objects")
        identifier, source, score = hit.get("_id"), hit.get("_source"), hit.get("_score")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise RetrievalFailure("Search returned a missing or duplicate canonical memory ID")
        if not isinstance(source, dict):
            raise RetrievalFailure("Search returned an invalid memory source")
        if "id" in source and source["id"] != identifier:
            raise RetrievalFailure("Search source ID differs from its canonical memory ID")
        if any(not isinstance(source.get(key), str) or not source[key].strip()
               for key in ("when_to_use", "content")):
            raise RetrievalFailure("Search memories require when_to_use and content strings")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise RetrievalFailure("Search memories require finite numeric scores")
        record = {"id": identifier, "when_to_use": source["when_to_use"], "content": source["content"]}
        if "source_refs" in source:
            refs = source["source_refs"]
            if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                raise RetrievalFailure("Memory source_refs must be a list of opaque strings")
            record["source_refs"] = list(refs)
        records.append(record)
        candidates.append({"id": identifier, "rank": rank, "score": score})
        seen.add(identifier)
    return records, candidates


def fuse(searches, *, k, rank_constant=60):
    """Fuse distinct completed lists once; retain contributions for auditability."""
    contributions = {}
    for result in searches:
        for candidate in result["candidates"]:
            contributions.setdefault(candidate["id"], []).append(
                {"search_id": result["id"], "rank": candidate["rank"]})
    ranked = [{
        "id": identifier,
        "score": math.fsum(1.0 / (rank_constant + item["rank"]) for item in items),
        "best_rank": min(item["rank"] for item in items), "contributions": items,
    } for identifier, items in contributions.items()]
    ranked.sort(key=lambda item: (-item["score"], item["best_rank"], item["id"]))
    return ranked[:k]


def retrieve(task, k, *, store, llm, meta, record_trace):
    """Retrieve from one prepared bank without modifying it or consulting labels."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a nonempty string")
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")
    meta = validate_meta(meta, k=k)
    if not callable(record_trace):
        raise ValueError("record_trace must be callable")
    version = meta["schema_version"]
    if version == 2:
        for field, expected in meta["provider_policy"].items():
            if getattr(llm, field, None) != expected:
                raise ValueError(f"Retrieval client {field} does not match provider_policy")
    prompts = prompt_manifest(meta)
    started = time.monotonic()
    trace = {
        "schema_version": version, "status": "running", "error": None,
        "task": task, "k": k, "meta": meta,
        "bank": None, "model_settings": None, "prompts": prompts,
        "queries": [], "generations": [], "embedding": None, "searches": [], "ranking": [],
        "progress": {"stage": "preflight", "completed_stages": [], "completed_branches": [],
                     "completed_searches": 0, "empty_bank_probe": None, "skipped_searches": []},
        "timing": {"elapsed_seconds": 0.0},
    }
    if version == 2:
        trace.update(bank={"descriptor": None, "observed": {}}, fusion=None, reranking=[])
    recording = False

    def emit():
        nonlocal recording
        trace["timing"]["elapsed_seconds"] = time.monotonic() - started
        recording = True
        record_trace(deepcopy(trace))
        recording = False

    def finish(status, records=None, error=None):
        trace.update(status=status, error=error)
        emit()
        return RetrievalResult(records or [], status, error)

    def mark_failure(error):
        original = error.__cause__ if isinstance(error.__cause__, StoreSearchError) else error
        detail = {"stage": trace["progress"]["stage"],
                  "type": type(original).__name__, "message": str(error)}
        for entry in [*trace["generations"], *trace["searches"], *trace.get("reranking", []), trace["embedding"],
                      trace["progress"]["empty_bank_probe"]]:
            if entry is not None and entry.get("status") == "running":
                entry.update(status="failed", error=detail)
        return detail

    try:
        emit()
        if k == 0:
            return finish("disabled")
        trace["progress"]["stage"] = "empty_bank_probe"
        probe = {"status": "running", "empty": None}
        trace["progress"]["empty_bank_probe"] = probe
        if version == 2:
            emit()
        probe_started = time.monotonic()
        hits = _hits(_query(store, {"size": 1, "query": {"match_all": {}}}))
        probe.update(status="completed", empty=not hits,
                     elapsed_seconds=time.monotonic() - probe_started)
        trace["progress"]["completed_stages"].append("empty_bank_probe")
        if not hits:
            return finish("empty_memory")
        emit()

        exact_bank = None
        if version == 2 and meta["search"]["backend"] == "exact":
            from .candidate_search import prepare_exact_bank

            trace["progress"]["stage"] = "bank_export"
            emit()
            probe_id = hits[0].get("_id") if isinstance(hits[0], dict) else None
            if not isinstance(probe_id, str) or not probe_id.strip():
                raise ValueError("Exact bank probe requires a canonical memory ID")
            try:
                exact_bank = prepare_exact_bank(
                    store, representations=list(dict.fromkeys(
                        route["representation"] for route in meta["search"]["routes"])),
                    dimensions=llm.dimensions, max_bank_size=meta["search"]["max_bank_size"],
                    probe_id=probe_id, probe_source=hits[0].get("_source"),
                )
            except StoreSearchError as error:
                raise RetrievalFailure(str(error)) from error
            trace["bank"]["observed"] = exact_bank.observed
            trace["progress"]["completed_stages"].append("bank_export")
            emit()

        trace["progress"]["stage"] = "query_generation"
        stage_started = time.monotonic()
        for branch in meta["query_generation"]["branches"]:
            generation = {
                "branch_id": branch["id"], "method": branch["method"], "status": "running",
                "component": f"retrieval.query_generation.{branch['id']}" if branch["method"] == "llm" else None,
                "prompt": branch.get("prompt"),
            }
            trace["generations"].append(generation)
            emit()
            queries = generate_queries(task, branch, llm=llm,
                                       prompt=prompts[branch["prompt"]]["text"] if branch["method"] == "llm" else None)
            trace["queries"].extend(queries)
            generation["status"] = "completed"
            trace["progress"]["completed_branches"].append(branch["id"])
            emit()
        trace["timing"]["query_generation_seconds"] = time.monotonic() - stage_started
        trace["progress"]["completed_stages"].append("query_generation")

        scheduled, seen = [], {}
        for route_index, route in enumerate(meta["search"]["routes"]):
            for query in trace["queries"]:
                if query["branch_id"] != route["branch"] or query["duplicate_of"] is not None:
                    continue
                key = (query["normalized"], route["representation"])
                identifier = f"route_{route_index}.{query['id']}"
                if key in seen:
                    trace["progress"]["skipped_searches"].append({
                        "id": identifier, "query_id": query["id"], "route_id": f"route_{route_index}",
                        "representation": route["representation"], "duplicate_of": seen[key],
                    })
                    continue
                seen[key] = identifier
                scheduled.append((identifier, route_index, route, query))

        trace["progress"]["stage"] = "embedding"
        texts = list(dict.fromkeys(query["text"] for _, _, _, query in scheduled))
        text_indices = {text: index for index, text in enumerate(texts)}
        component = "retrieval.embedding.batch_0"
        trace["embedding"] = {
            "component": component, "status": "running", "texts": texts,
            "inputs": [{"index": index, "text": text,
                        "query_ids": [query["id"] for query in trace["queries"]
                                      if query["duplicate_of"] is None and query["text"] == text]}
                       for index, text in enumerate(texts)],
        }
        emit()
        stage_started = time.monotonic()
        vectors = llm.embed(texts, phase="retrieval", component=component)
        # ModelClient validates values, dimensions, and input order; a nonconforming
        # injected client is a programming error, not a recoverable retrieval.
        if len(vectors) != len(texts):
            raise ValueError("Embedding client violated its batch length contract")
        trace["embedding"]["status"] = "completed"
        trace["timing"]["embedding_seconds"] = time.monotonic() - stage_started
        trace["progress"]["completed_stages"].append("embedding")
        emit()

        trace["progress"]["stage"] = "search"
        stage_started = time.monotonic()
        records_by_id = {}
        search_depth = meta["search"]["depth"] if version == 2 else k
        if search_depth == "all":
            search_depth = len(exact_bank.records)
        for identifier, route_index, route, query in scheduled:
            embedding_index = text_indices[query["text"]]
            result = {
                "id": identifier, "route_id": f"route_{route_index}", "query_id": query["id"],
                "branch_id": route["branch"], "representation": route["representation"],
                "field": REPRESENTATIONS[route["representation"]]["field"],
                "parameters": {"size": search_depth, "k": search_depth,
                               "num_candidates": meta["search"]["num_candidates"]},
                "embedding_component": component, "embedding_index": embedding_index,
                "status": "running", "candidates": [],
            }
            if version == 2:
                result.update(backend=meta["search"]["backend"], requested_depth=meta["search"]["depth"],
                              effective_depth=search_depth)
            trace["searches"].append(result)
            if version == 2:
                emit()
            search_started = time.monotonic()
            if exact_bank is not None:
                from .candidate_search import exact_search

                records, candidates = exact_search(
                    exact_bank, vectors[embedding_index], representation=route["representation"],
                    depth=search_depth,
                )
            else:
                records, candidates = search(
                    store, vectors[embedding_index], representation=route["representation"],
                    k=search_depth, num_candidates=meta["search"]["num_candidates"],
                )
            result.update(status="completed", candidates=candidates,
                          elapsed_seconds=time.monotonic() - search_started)
            if version == 2:
                result["effective_depth"] = len(candidates)
            for record in records:
                records_by_id.setdefault(record["id"], record)
            trace["progress"]["completed_searches"] += 1
            emit()
        trace["timing"]["search_seconds"] = time.monotonic() - stage_started
        trace["progress"]["completed_stages"].append("search")

        trace["progress"]["stage"] = "fusion"
        stage_started = time.monotonic()
        fusion_limit = meta["fusion"]["candidate_limit"] if version == 2 else k
        if fusion_limit == "all":
            fusion_limit = len(exact_bank.records)
        fused = fuse(trace["searches"], k=fusion_limit, rank_constant=meta["fusion"]["rank_constant"])
        trace["ranking"] = fused
        trace["timing"]["fusion_seconds"] = time.monotonic() - stage_started
        trace["progress"]["completed_stages"].append("fusion")
        records = [records_by_id[item["id"]] for item in fused]
        if version == 1:
            return finish("completed", records)
        trace["fusion"] = {"candidate_limit": meta["fusion"]["candidate_limit"], "ranking": fused}
        trace["ranking"] = []
        emit()
        fallback = None
        for stage in meta["reranking"]["stages"]:
            trace["progress"]["stage"] = f"reranking.{stage['id']}"
            position = len(trace["reranking"])
            trace["reranking"].append({})

            def record_stage(value):
                trace["reranking"][position] = value
                emit()

            records, stage_trace = run_stage(task, records, llm=llm, stage=stage,
                                             prompt=prompts[stage["prompt"]]["text"],
                                             record_stage=record_stage)
            if stage_trace["status"] == "completed_with_fallback" and fallback is None:
                fallback = {"stage": trace["progress"]["stage"], "type": "RankingFallback",
                            "message": stage_trace["error"]["message"]}
            trace["progress"]["completed_stages"].append(trace["progress"]["stage"])
        records = records[:k]
        trace["ranking"] = [{"id": record["id"], "rank": index + 1}
                            for index, record in enumerate(records)]
        trace["progress"]["stage"] = "completed"
        return finish("completed_with_fallback" if fallback else "completed", records, fallback)
    except (RetrievalFailure, RankingFailure) as error:
        if recording:
            raise
        return finish("degraded", error=mark_failure(error))
    except BaseException as error:
        if not recording:
            trace.update(status="failed", error=mark_failure(error))
            emit()
        raise
