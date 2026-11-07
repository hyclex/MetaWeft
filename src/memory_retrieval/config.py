"""Versioned retrieval configuration, prompts, and prepared-bank representations."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


PACKAGE = Path(__file__).parent
PRESETS = ("base", "setting_1", "setting_2", "setting_3", "setting_4", "setting_5", "setting_6",
           "performance_1", "performance_2", "performance_3",
           "efficiency_1", "efficiency_2", "efficiency_3",
           "transferability_1", "transferability_2", "transferability_3", "exact_base")
PROMPTS = {"general_v1": "prompts/v1/general.txt",
           "applicability_v1": "prompts/v1/applicability.txt",
           "applicability_optimized_v1": "prompts/v1/applicability_optimized.txt",
           "applicability_fewshot_v1": "prompts/v1/applicability_fewshot.txt"}
RANKING_PROMPTS = {"procedure_listwise20_v1": "prompts/v2/procedure_listwise20.txt",
                   "utility_scores_v1": "prompts/v2/utility_scores.txt"}
REPRESENTATIONS = {
    "combined": {"dataset_field": "ReMe_emb", "field": "embedding",
                 "input": 'when_to_use + "\\n\\n" + content'},
    "applicability": {"dataset_field": "when_to_use_emb", "field": "when_to_use_embedding",
                      "input": "when_to_use"},
    "content": {"dataset_field": "content_emb", "field": "content_embedding", "input": "content"},
}


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} requires exactly: {', '.join(sorted(expected))}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def validate_meta(meta, *, k=None):
    """Return an independent validated configuration; labels never select behavior."""
    if isinstance(meta, dict) and type(meta.get("schema_version")) is int and meta["schema_version"] == 2:
        return _validate_v2(meta, k=k)
    _keys(meta, {"schema_version", "name", "description", "query_generation", "search", "fusion"}, "meta")
    if type(meta["schema_version"]) is not int or meta["schema_version"] != 1:
        raise ValueError("Unsupported retrieval schema_version")
    _text(meta["name"], "name")
    _text(meta["description"], "description")
    _keys(meta["query_generation"], {"branches"}, "query_generation")
    branches = meta["query_generation"]["branches"]
    if not isinstance(branches, list) or not branches:
        raise ValueError("query_generation.branches must be a nonempty list")
    identifiers = set()
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("Each generation branch must be an object")
        method = branch.get("method")
        if method == "identity":
            _keys(branch, {"id", "method"}, "identity branch")
        elif method == "llm":
            _keys(branch, {"id", "method", "prompt", "generated_count"}, "llm branch")
            if not isinstance(branch["prompt"], str) or branch["prompt"] not in PROMPTS:
                raise ValueError("Unknown generation prompt")
            if not (PACKAGE / PROMPTS[branch["prompt"]]).is_file():
                raise ValueError("Required generation prompt is missing")
            if type(branch["generated_count"]) is not int or not 1 <= branch["generated_count"] <= 3:
                raise ValueError("generated_count must be an integer from 1 to 3")
        else:
            raise ValueError("Unknown query-generation method")
        _text(branch["id"], "branch id")
        if branch["id"] in identifiers:
            raise ValueError("Generation branch IDs must be unique")
        identifiers.add(branch["id"])
    _keys(meta["search"], {"num_candidates", "routes"}, "search")
    candidates = meta["search"]["num_candidates"]
    if type(candidates) is not int or not 1 <= candidates <= 10000:
        raise ValueError("num_candidates must be an integer from 1 to 10000")
    if k is not None:
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if k > candidates:
            raise ValueError("num_candidates must be at least k")
    routes = meta["search"]["routes"]
    if not isinstance(routes, list) or not routes:
        raise ValueError("search.routes must be a nonempty list")
    route_keys = set()
    referenced = set()
    for route in routes:
        _keys(route, {"branch", "representation"}, "search route")
        if not isinstance(route["branch"], str) or route["branch"] not in identifiers:
            raise ValueError("Search route refers to an unknown generation branch")
        if not isinstance(route["representation"], str) or route["representation"] not in REPRESENTATIONS:
            raise ValueError("Unknown memory representation")
        key = (route["branch"], route["representation"])
        if key in route_keys:
            raise ValueError("Duplicate search route")
        route_keys.add(key)
        referenced.add(route["branch"])
    if referenced != identifiers:
        raise ValueError("Every generation branch must have a search route")
    _keys(meta["fusion"], {"method", "rank_constant"}, "fusion")
    if meta["fusion"]["method"] != "rrf":
        raise ValueError("Unknown fusion method")
    constant = meta["fusion"]["rank_constant"]
    if type(constant) is not int or constant < 1:
        raise ValueError("rank_constant must be a positive integer")
    return deepcopy(meta)


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _depth(value, name, k):
    if value == "all":
        return
    _positive(value, name)
    if k is not None and value < k:
        raise ValueError(f"{name} must be at least k")


def _validate_v2(meta, *, k):
    _keys(meta, {"schema_version", "name", "description", "query_generation", "search",
                 "fusion", "reranking", "provider_policy"}, "meta")
    if k is not None and (type(k) is not int or k < 0):
        raise ValueError("k must be a nonnegative integer")
    search = meta["search"]
    _keys(search, {"backend", "depth", "num_candidates", "max_bank_size", "routes"}, "search")
    if search["backend"] not in ("hnsw", "exact"):
        raise ValueError("Unknown candidate search backend")
    _depth(search["depth"], "search.depth", k)
    if search["depth"] == "all":
        if search["backend"] != "exact":
            raise ValueError("All-depth search requires exact search")
        _positive(search["max_bank_size"], "search.max_bank_size")
    elif search["max_bank_size"] is not None:
        raise ValueError("Finite search depth requires max_bank_size=null")
    if search["backend"] == "hnsw":
        candidates = search["num_candidates"]
        if type(candidates) is not int or not search["depth"] <= candidates <= 10000:
            raise ValueError("HNSW requires depth <= num_candidates <= 10000")
    elif search["num_candidates"] is not None:
        raise ValueError("Exact search requires num_candidates=null")
    fusion = meta["fusion"]
    _keys(fusion, {"method", "rank_constant", "candidate_limit"}, "fusion")
    _depth(fusion["candidate_limit"], "fusion.candidate_limit", k)
    if fusion["candidate_limit"] == "all" and search["depth"] != "all":
        raise ValueError("All-candidate fusion requires all-depth search")
    # Reuse the unchanged v1 branch, route, representation and RRF contracts.
    validate_meta({"schema_version": 1, "name": meta["name"], "description": meta["description"],
                   "query_generation": meta["query_generation"],
                   "search": {"num_candidates": 10000, "routes": search["routes"]},
                   "fusion": {"method": fusion["method"], "rank_constant": fusion["rank_constant"]}})
    _keys(meta["reranking"], {"stages"}, "reranking")
    stages = meta["reranking"]["stages"]
    if not isinstance(stages, list) or len(stages) > 2:
        raise ValueError("reranking.stages must contain zero to two stages")
    identifiers, methods = set(), []
    for stage in stages:
        _keys(stage, {"id", "method", "prompt", "model", "fields", "output_limit",
                      "small_input_policy", "generation", "response"}, "ranking stage")
        _text(stage["id"], "ranking stage id")
        _text(stage["model"], "ranking model")
        if stage["id"] in identifiers:
            raise ValueError("Ranking stage IDs must be unique")
        identifiers.add(stage["id"])
        methods.append(stage["method"])
        fields = stage["fields"]
        if (not isinstance(fields, list) or not fields
                or any(field not in ("when_to_use", "content") for field in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError("Ranking fields must be an ordered unique subset of when_to_use and content")
        response = stage["response"]
        _keys(response, {"accept_wrapped", "repair_duplicates", "schema_retries",
                         "fallback_to_input", "fallback_on_empty_response"}, "ranking response")
        for field in ("accept_wrapped", "repair_duplicates", "fallback_to_input", "fallback_on_empty_response"):
            if type(response[field]) is not bool:
                raise ValueError(f"response.{field} must be boolean")
        if type(response["schema_retries"]) is not int or response["schema_retries"] not in (0, 1):
            raise ValueError("schema_retries must be 0 or 1")
        if response["fallback_on_empty_response"] and not response["fallback_to_input"]:
            raise ValueError("Empty-response fallback requires input fallback")
        if stage["method"] == "listwise":
            if (stage["prompt"] != "procedure_listwise20_v1"
                    or type(stage["output_limit"]) is not int or stage["output_limit"] != 20
                    or stage["small_input_policy"] != "bypass"):
                raise ValueError("Listwise requires the fixed-20 prompt, output_limit=20 and bypass policy")
        elif stage["method"] == "scores":
            if (stage["prompt"] != "utility_scores_v1" or stage["output_limit"] != "all"
                    or stage["small_input_policy"] != "run" or response["repair_duplicates"]):
                raise ValueError("Scores requires the utility prompt, all outputs, run policy and no duplicate repair")
        else:
            raise ValueError("Unknown ranking method")
        _depth(stage["output_limit"], "ranking output_limit", k)
        if not (PACKAGE / RANKING_PROMPTS[stage["prompt"]]).is_file():
            raise ValueError("Required ranking prompt is missing")
        generation = stage["generation"]
        _keys(generation, {"temperature", "seed", "enable_thinking", "thinking_budget",
                           "max_tokens", "max_completion_tokens"}, "ranking generation")
        temperature = generation["temperature"]
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if type(generation["seed"]) is not int or type(generation["enable_thinking"]) is not bool:
            raise ValueError("seed must be an integer and enable_thinking must be boolean")
        if generation["enable_thinking"]:
            _positive(generation["thinking_budget"], "thinking_budget")
            _positive(generation["max_completion_tokens"], "max_completion_tokens")
            if (generation["thinking_budget"] >= generation["max_completion_tokens"]
                    or generation["max_tokens"] is not None):
                raise ValueError("Thinking requires budget below completion cap and max_tokens=null")
        else:
            _positive(generation["max_tokens"], "max_tokens")
            if generation["thinking_budget"] is not None or generation["max_completion_tokens"] is not None:
                raise ValueError("Nonthinking requires null thinking budget and completion cap")
    if methods not in ([], ["listwise"], ["scores"], ["listwise", "scores"]):
        raise ValueError("Unsupported ranking stage sequence")
    policy = meta["provider_policy"]
    _keys(policy, {"max_retries", "retry_delay_seconds"}, "provider_policy")
    if type(policy["max_retries"]) is not int or policy["max_retries"] < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    delay = policy["retry_delay_seconds"]
    if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
        raise ValueError("retry_delay_seconds must be finite and nonnegative")
    return deepcopy(meta)


def load_preset(name="base", *, backend=None, num_candidates=None):
    if not isinstance(name, str) or name not in PRESETS:
        raise ValueError(f"Unknown retrieval preset: {name}")
    meta = json.loads((PACKAGE / "presets" / f"{name}.json").read_text())
    if meta["schema_version"] == 1:
        if backend is not None:
            raise ValueError("Backend override is only supported for schema v2")
    else:
        search = meta["search"]
        if backend is not None:
            search["backend"] = backend
        if search["backend"] == "exact":
            if num_candidates is not None:
                raise ValueError("Explicit num_candidates is invalid for exact search")
            search["num_candidates"] = None
        elif search["backend"] == "hnsw" and search["num_candidates"] is None:
            if type(search["depth"]) is not int:
                raise ValueError("HNSW requires finite search depth")
            search["num_candidates"] = min(10000, 2 * search["depth"])
    if num_candidates is not None:
        meta["search"]["num_candidates"] = num_candidates
    return validate_meta(meta)


def required_representations(meta):
    meta = validate_meta(meta)
    return list(dict.fromkeys(route["representation"] for route in meta["search"]["routes"]))


def prompt_manifest(meta):
    """Capture the exact package-owned prompts selected by a configuration."""
    meta = validate_meta(meta)
    result = {}
    for branch in meta["query_generation"]["branches"]:
        if branch["method"] == "llm":
            identifier = branch["prompt"]
            raw = (PACKAGE / PROMPTS[identifier]).read_bytes()
            result[identifier] = {"path": PROMPTS[identifier],
                                  "sha256": hashlib.sha256(raw).hexdigest(),
                                  "text": raw.decode("utf-8")}
    for stage in meta.get("reranking", {}).get("stages", []):
        identifier = stage["prompt"]
        raw = (PACKAGE / RANKING_PROMPTS[identifier]).read_bytes()
        result[identifier] = {"path": RANKING_PROMPTS[identifier], "kind": "reranking",
                              "sha256": hashlib.sha256(raw).hexdigest(),
                              "text": raw.decode("utf-8")}
    return result


def memory_index_body(dimensions=1024, *, representations=("combined",)):
    """Prepare a fresh mapping; this helper never embeds or writes memories."""
    if type(dimensions) is not int or dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")
    representations = list(representations)
    if not representations or any(not isinstance(key, str) or key not in REPRESENTATIONS
                                  for key in representations):
        raise ValueError("Mapping requires supported representations")
    if len(set(representations)) != len(representations):
        raise ValueError("Mapping representations must be unique")
    properties = {
        "id": {"type": "keyword"}, "when_to_use": {"type": "text"},
        "content": {"type": "text"}, "source_refs": {"type": "keyword"},
    }
    for representation in representations:
        properties[REPRESENTATIONS[representation]["field"]] = {
            "type": "dense_vector", "dims": dimensions, "index": True,
            "similarity": "cosine", "index_options": {"type": "hnsw"},
        }
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0,
                     "index.mapping.exclude_source_vectors": False},
        "mappings": {"properties": properties},
    }
