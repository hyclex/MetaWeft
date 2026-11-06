"""Versioned retrieval configuration, prompts, and prepared-bank representations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).parent
PRESETS = ("base", "setting_1", "setting_2", "setting_3", "setting_4", "setting_5", "setting_6")
PROMPTS = {"general_v1": "prompts/v1/general.txt",
           "applicability_v1": "prompts/v1/applicability.txt",
           "applicability_optimized_v1": "prompts/v1/applicability_optimized.txt",
           "applicability_fewshot_v1": "prompts/v1/applicability_fewshot.txt"}
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


def load_preset(name="base", *, num_candidates=None):
    if not isinstance(name, str) or name not in PRESETS:
        raise ValueError(f"Unknown retrieval preset: {name}")
    meta = json.loads((PACKAGE / "presets" / f"{name}.json").read_text())
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
