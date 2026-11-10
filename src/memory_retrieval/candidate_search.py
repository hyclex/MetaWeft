"""Call-local exact candidate search over a fully exported prepared bank."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

import requests

from MetaWeft.service.elasticsearch import StoreSearchError
from .config import REPRESENTATIONS


@dataclass
class ExactBank:
    ids: list[str]
    records: dict
    matrices: dict
    dimensions: int
    observed: dict


def _vector(value, dimensions, label):
    if (not isinstance(value, list) or len(value) != dimensions
            or any(type(component) not in (int, float) or not math.isfinite(component)
                   for component in value)):
        raise ValueError(f"{label} requires {dimensions} finite numeric components")
    return value


def prepare_exact_bank(store, *, representations, dimensions, max_bank_size=None,
                       probe_id=None, probe_source=None):
    """Export once and validate before any query generation or embedding calls.

    The store's complete-export contract checks pagination and exact totals.
    Caller preparation owns embedding provenance and the read-only interval.
    """
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("Exact bank dimensions must be a positive integer")
    representations = list(representations)
    if (not representations or any(not isinstance(name, str) or name not in REPRESENTATIONS
                                   for name in representations)
            or len(set(representations)) != len(representations)):
        raise ValueError("Exact bank requires unique supported representations")
    if max_bank_size is not None and (type(max_bank_size) is not int or max_bank_size < 1):
        raise ValueError("Exact bank capacity must be a positive integer")
    if probe_id is not None and (not isinstance(probe_id, str) or not probe_id.strip()):
        raise ValueError("Exact bank probe ID must be a canonical nonempty string")
    if probe_source is not None:
        if not isinstance(probe_source, dict) or probe_id is None:
            raise ValueError("Exact bank probe source requires an object and canonical ID")
        if "id" in probe_source and probe_source["id"] != probe_id:
            raise ValueError("Exact bank probe source ID differs from its canonical ID")
        for field in ("when_to_use", "content", "name"):
            if field in probe_source and (not isinstance(probe_source[field], str) or not probe_source[field].strip()):
                raise ValueError("Exact bank probe source text must be a nonempty string")
        if "source_refs" in probe_source:
            refs = probe_source["source_refs"]
            if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                raise ValueError("Exact bank probe source_refs must be a list of opaque strings")
        for name in representations:
            field = REPRESENTATIONS[name]["field"]
            if field in probe_source:
                _vector(probe_source[field], dimensions, f"Exact bank probe/{field}")
    try:
        exported = store.export()
    except requests.RequestException as error:
        raise StoreSearchError(f"Elasticsearch export failed: {type(error).__name__}: {error}") from error
    if not isinstance(exported, list):
        raise ValueError("Exact bank export must be a complete list")
    if max_bank_size is not None and len(exported) > max_bank_size:
        raise ValueError(f"Exact bank exceeds its capacity of {max_bank_size}")

    records, vectors, snapshot = {}, {name: {} for name in representations}, []
    for entry in exported:
        if not isinstance(entry, dict) or set(entry) != {"id", "document"}:
            raise ValueError("Exact bank export requires id/document entries")
        identifier, source = entry["id"], entry["document"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in records:
            raise ValueError("Exact bank requires unique canonical memory IDs")
        if not isinstance(source, dict) or ("id" in source and source["id"] != identifier):
            raise ValueError("Exact bank source must agree with its canonical memory ID")
        if any(not isinstance(source.get(key), str) or not source[key].strip()
               for key in ("when_to_use", "content")):
            raise ValueError("Exact bank memories require when_to_use and content strings")
        record = {"id": identifier, "when_to_use": source["when_to_use"], "content": source["content"]}
        if "name" in source:
            if not isinstance(source["name"], str) or not source["name"].strip():
                raise ValueError("Exact bank name must be a nonempty string when present")
            record["name"] = source["name"]
        if "source_refs" in source:
            refs = source["source_refs"]
            if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                raise ValueError("Exact bank source_refs must be a list of opaque strings")
            record["source_refs"] = list(refs)
        records[identifier] = record
        observed_record = deepcopy(record)
        for name in representations:
            field = REPRESENTATIONS[name]["field"]
            vector = list(_vector(source.get(field), dimensions, f"Exact bank {identifier}/{field}"))
            vectors[name][identifier] = vector
            observed_record[field] = vector
        if identifier == probe_id and probe_source is not None:
            if ("name" in probe_source) != ("name" in observed_record):
                raise ValueError("Exact bank export contradicts the probe source field: name")
            selected = ("id", "when_to_use", "content", "name", "source_refs",
                        *(REPRESENTATIONS[name]["field"] for name in representations))
            for field in selected:
                if field in probe_source and probe_source[field] != observed_record.get(field):
                    raise ValueError(f"Exact bank export contradicts the probe source field: {field}")
        snapshot.append(observed_record)
    if probe_id is not None and probe_id not in records:
        raise ValueError("Exact bank export contradicts the nonempty-bank probe")

    # Keep NumPy out of the HNSW/v1 import path.
    import numpy as np

    identifiers = sorted(records)
    matrices = {}
    for name in representations:
        matrix = np.asarray([vectors[name][identifier] for identifier in identifiers], dtype=np.float64)
        matrix = matrix.reshape((len(identifiers), dimensions))
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("Exact bank vectors require finite nonzero norms")
        matrices[name] = matrix
    snapshot.sort(key=lambda row: row["id"])
    encoded = json.dumps({"records": snapshot,
                          "representations": {name: REPRESENTATIONS[name] for name in representations}},
                         sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    observed = {"exported_count": len(identifiers),
                "exact_snapshot_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}
    return ExactBank(identifiers, records, matrices, dimensions, observed)


def exact_search(bank, vector, *, representation, depth):
    """Return raw float64 cosine scores, resolving equal scores by canonical ID."""
    if representation not in bank.matrices:
        raise ValueError("Exact search representation was not prepared")
    if depth != "all" and (type(depth) is not int or depth < 1):
        raise ValueError("Exact search depth must be a positive integer or all")
    _vector(vector, bank.dimensions, "Exact query vector")
    import numpy as np

    query = np.asarray(vector, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        query_norm = np.linalg.norm(query)
    if query_norm == 0 or not np.isfinite(query_norm):
        raise ValueError("Exact query vector requires a finite nonzero norm")
    matrix = bank.matrices[representation]
    # Match the exploration's float64 operation order, without prenormalizing.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        norms = np.linalg.norm(matrix, axis=1) * query_norm
        scores = matrix @ query / norms
    if not np.isfinite(scores).all():
        raise ValueError("Exact cosine produced nonfinite scores")
    ranked = sorted(zip(bank.ids, map(float, scores)), key=lambda item: (-item[1], item[0]))
    if depth != "all":
        ranked = ranked[:depth]
    records = [deepcopy(bank.records[identifier]) for identifier, _ in ranked]
    candidates = [{"id": identifier, "rank": rank, "score": score}
                  for rank, (identifier, score) in enumerate(ranked, 1)]
    return records, candidates
