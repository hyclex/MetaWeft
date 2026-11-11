"""Offline estimates migrated from metaskill-appworld's execution_accounting.usage_cost.

The parent is baselines/reme2f37a15/benchmark/appworld/execution_accounting.py.
Keep its per-million-token formula, aliases, validation and embedding semantics.
Rates are historical workspace assumptions, not live prices or provider invoices.
"""

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path


PRICING = {
    "currency": "USD",
    "unit": "per_1m_tokens",
    "provider": "az.gptplus5.com",
    "pricing_source": "https://az.gptplus5.com/api/pricing",
    "formula": "((input - cached) * input_rate + cached * cached_rate + output * output_rate) / 1000000",
    "models": {
        "qwen3.8-flash": {"input": "0.64", "cached": "0.08", "output": "2.16",
                          "recorded_on": "2026-10-13",
                          "source": "metaskill-appworld/experiments/code/ace/openai_cost_registry.json"},
        "qwen3.8-max-0902": {"input": "9.6", "cached": "0.7999999999968", "output": "28.8",
                             "recorded_on": "2026-11-10",
                             "source": "https://az.gptplus5.com/api/pricing",
                             "provider_ratios": {"model_ratio": "4.8", "completion_ratio": "3",
                                                 "cache_ratio": "0.083333333333"},
                             "conversion": "input = 2 * model_ratio; cached/output = input * respective ratio"},
        "text-embedding-3-large": {"input": "0.13", "cached": "0.00845", "output": "0",
                                   "recorded_on": "2026-10-28", "source": "tool/get_cost.sh"},
    },
    "note": "Fixed historical rates; reported-usage estimates, not verified billed amounts. "
            "Partial estimates exclude unpriced calls; response-cache replays incur no new cost.",
}


def usage_cost(model, usage, data=PRICING, *, request_kind="chat"):
    """Return the parent's Decimal estimate, or None for unusable usage/model."""
    if not isinstance(usage, dict) or model not in data["models"]:
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    if request_kind == "embedding" and completion is None:
        completion = 0
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else None
    if any(type(value) is not int or value < 0 for value in (prompt, completion, cached)) or cached > prompt:
        return None
    spec = data["models"][model]
    return ((prompt - cached) * Decimal(spec["input"]) + cached * Decimal(spec["cached"])
            + completion * Decimal(spec["output"])) / Decimal(1_000_000)


def summarize_cost(calls):
    """Price distinct physical calls, including failed responses with usable usage."""
    seen, reasons = {}, Counter()
    total, priced, cache_hits, duplicates = Decimal(0), 0, 0, 0
    for call in calls:
        request = call.get("input") or {}
        output = call.get("output")
        output = output if isinstance(output, dict) else {}
        model = request.get("model") or output.get("model")
        usage = output.get("usage")
        kind = "embedding" if "input" in request or "vector_count" in output else "chat"
        hit = call.get("metadata", {}).get("extra", {}).get("cache", {}).get("hit", False)
        signature = (model, kind, hit, json.dumps(usage, sort_keys=True))
        identifier = call["id"]
        if identifier in seen:
            if seen[identifier] != signature:
                raise ValueError(f"Conflicting cost evidence for physical call {identifier}")
            duplicates += 1
            continue
        seen[identifier] = signature
        if hit:
            cache_hits += 1
            continue
        cost = usage_cost(model, usage, request_kind=kind)
        if cost is None:
            reasons["unknown_model" if model not in PRICING["models"] else "missing_or_invalid_usage"] += 1
        else:
            total += cost
            priced += 1
    known = priced > 0 or cache_hits > 0
    return {
        "cost_usd": float(total) if known else None,
        "cost_status": ("partial" if reasons else "complete") if known else "unavailable",
        "priced_calls": priced, "unpriced_calls": sum(reasons.values()),
        "unpriced_reasons": dict(reasons), "cache_hits": cache_hits,
        "duplicate_calls": duplicates,
    }


def refresh_cost_estimate(output):
    """Refresh a derived sidecar without changing canonical evidence or saved accounting.

    File signatures avoid rereading unchanged traces on each periodic collection.
    Historical workers can still finalize their own accounting independently.
    """
    output = Path(output)
    paths = sorted({*output.glob("tasks/*/logs/lm_calls.jsonl"),
                    *output.glob("baseline_artifacts/*/*lm_calls.jsonl"),
                    *output.glob("setup_lm_calls.jsonl"),
                    *output.glob("recovery_history/**/tasks/*/logs/lm_calls.jsonl"),
                    *output.glob("recovery_history/**/baseline_artifacts/*/*lm_calls.jsonl")})
    sources = {}
    for path in paths:
        stat = path.stat()
        sources[str(path.relative_to(output))] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    estimator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = output / "cost_estimate.json"
    if destination.is_file():
        saved = json.loads(destination.read_text())
        if saved.get("sources") == sources and saved.get("estimator_sha256") == estimator_sha256:
            return saved
    incomplete = []

    def calls():
        for path in paths:
            with path.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        if line.endswith("\n"):
                            raise
                        incomplete.append(str(path.relative_to(output)))
                        break

    result = summarize_cost(calls())
    if incomplete and result["cost_status"] == "complete":
        result["cost_status"] = "partial"
    result.update(
        pricing_assumptions=PRICING, estimator_sha256=estimator_sha256,
        generated_at=datetime.now(timezone.utc).isoformat(), sources=sources, incomplete_logs=incomplete,
        coverage="Distinct recorded physical calls across tasks, initialization and archived recovery attempts; "
                 "cache replays excluded. Snapshot of saved evidence; unrecorded calls remain unknown.",
    )
    if output.is_dir():
        from MetaWeft.experiment.recovery import write_json
        write_json(destination, result)
    return result
