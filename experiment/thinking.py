"""Offline evidence of thinking-control compliance in saved physical chat calls.

Visible reasoning or positive reasoning-token telemetry establishes observation;
absent telemetry never establishes that no hidden computation occurred.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


def mapping(value):
    return value if isinstance(value, dict) else {}


def requested_thinking(request):
    # The SDK merges extra_body into the top-level wire request.
    effective = {**request, **mapping(request.get("extra_body"))}
    controls = []
    for key in ("enable_thinking", "thinking"):
        value = effective.get(key)
        if type(value) is bool:
            controls.append(value)
    for key in ("no_thinking", "no_think"):
        value = effective.get(key)
        if type(value) is bool:
            controls.append(not value)
    for options in (mapping(effective.get("thinking")), mapping(effective.get("reasoning"))):
        if options.get("type") in ("enabled", "disabled"):
            controls.append(options["type"] == "enabled")
        if type(options.get("enabled")) is bool:
            controls.append(options["enabled"])
    for effort in (effective.get("reasoning_effort"), mapping(effective.get("reasoning")).get("effort")):
        if effort == "none":
            controls.append(False)
        elif effort in ("minimal", "low", "medium", "high", "xhigh"):
            controls.append(True)
    if not controls:
        return "unspecified"
    if len(set(controls)) != 1:
        return "conflicting"
    return "on" if controls[0] else "off"


def text_parts(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [part for item in value for part in text_parts(item)]
    if isinstance(value, dict):
        return [part for key in ("text", "thinking", "reasoning", "content", "summary")
                for part in text_parts(value.get(key))]
    return []


def has_think_tag(text):
    # A leading protocol block counts. Quoted examples in prose/fences do not.
    while match := re.match(r"\s*<think>(.*?)(?:</think>|$)", text, re.DOTALL | re.IGNORECASE):
        if match[1].strip():
            return True
        text = text[match.end():]
    return False


def classify(call):
    """Return observable chat evidence, preserving the workspace audit's API."""
    request = mapping(call.get("input"))
    if "messages" not in request:
        return None
    raw_output = call.get("output")
    output, metadata = mapping(raw_output), mapping(call.get("metadata"))
    choices = output.get("choices") if isinstance(output.get("choices"), list) else []
    choices = [choice for choice in choices if isinstance(choice, dict)]
    messages = [mapping(choice.get("message")) for choice in choices]
    if not choices and isinstance(output.get("content"), (str, list)):
        messages = [output]
    reasoning = []
    for message in [output, *(message for message in messages if message is not output)]:
        for key in ("reasoning_content", "reasoning", "thinking", "reasoning_details"):
            reasoning.extend(text_parts(message.get(key)))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
                    reasoning.extend(text_parts(block))
    reasoning_chars = sum(len(text) for text in reasoning if text.strip())
    contents = ["".join(text_parts(message.get("content"))) for message in messages]
    if isinstance(raw_output, str) and metadata.get("status") != "error":
        contents.append(raw_output)
    tagged = any(has_think_tag(text) for text in contents)
    usage = mapping(output.get("usage"))
    counters = []
    for container in (usage, mapping(usage.get("completion_tokens_details")),
                      mapping(usage.get("output_tokens_details")), mapping(output.get("usageMetadata"))):
        for key in ("reasoning_tokens", "thinking_tokens", "thoughts_token_count", "thoughtsTokenCount"):
            value = container.get(key)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                counters.append(value)
    tokens = max(counters) if counters else None  # Aliases are not additive.
    observed = reasoning_chars > 0 or tagged or (tokens is not None and tokens > 0)
    has_response = bool(choices or any(contents) or observed)
    if observed:
        evidence = "observed_thinking"
    elif not has_response:
        evidence = "no_response"
    elif tokens == 0:
        evidence = "zero_reported_no_visible_reasoning"
    else:
        evidence = "no_visible_reasoning_tokens_unknown"
    return {
        "id": call.get("id"), "logical_call_id": metadata.get("logical_call_id"),
        "task_id": metadata.get("root_task_id"), "phase": metadata.get("phase"),
        "component": metadata.get("component"), "attempt": metadata.get("attempt"),
        "started_at": metadata.get("started_at"), "model": request.get("model"),
        "returned_model": output.get("model"), "requested_thinking": requested_thinking(request),
        "cache_hit": mapping(mapping(metadata.get("extra")).get("cache")).get("hit") is True,
        "has_response": has_response, "evidence": evidence,
        "reasoning_chars": reasoning_chars, "think_tag": tagged, "reasoning_tokens": tokens,
        "usage_source": usage.get("usage_source"),
        "finish_reasons": [choice.get("finish_reason") for choice in choices],
        "empty_content": has_response and not any(text.strip() for text in contents),
        "error_type": output.get("error_type"),
    }


def summarize_thinking(calls):
    """Count distinct recorded physical attempts; retain missing-evidence counts."""
    names = ("total_llm_calls", "no_thinking_llm_calls", "thinking_despite_no_thinking_llm_calls",
             "embedding_calls_excluded", "cache_replay_calls_excluded", "duplicate_records_excluded",
             "unclassified_records", "malformed_records", "conflicting_call_ids",
             "unspecified_control_llm_calls", "conflicting_control_llm_calls",
             "no_response_llm_calls", "unknown_reasoning_llm_calls",
             "no_thinking_no_response_calls", "no_thinking_unknown_reasoning_calls",
             "no_thinking_zero_reasoning_calls", "no_thinking_positive_reasoning_token_calls",
             "no_thinking_visible_reasoning_calls", "reasoning_text_with_zero_tokens")
    counts = Counter({name: 0 for name in names})
    seen, gaps = {}, []
    for index, call in enumerate(calls, 1):
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
            counts["malformed_records"] += 1
            gaps.append({"record": index, "reason": "Missing or invalid physical call ID"})
            continue
        signature = hashlib.sha256(json.dumps(call, sort_keys=True).encode()).hexdigest()
        identifier = call["id"]
        if identifier in seen:
            if seen[identifier] != signature:
                counts["conflicting_call_ids"] += 1
                gaps.append({"call_id": identifier, "reason": "Conflicting physical call records; counted first record only"})
            else:
                counts["duplicate_records_excluded"] += 1
            continue
        seen[identifier] = signature
        request = mapping(call.get("input"))
        row = classify(call)
        if row is None:
            if "input" in request:
                counts["embedding_calls_excluded"] += 1
            else:
                counts["unclassified_records"] += 1
                gaps.append({"call_id": identifier, "reason": "Request has neither chat messages nor embedding input"})
            continue
        if row["cache_hit"]:
            counts["cache_replay_calls_excluded"] += 1
            continue
        counts["total_llm_calls"] += 1
        requested, evidence = row["requested_thinking"], row["evidence"]
        if requested in ("unspecified", "conflicting"):
            counts[f"{requested}_control_llm_calls"] += 1
        counts["no_response_llm_calls"] += evidence == "no_response"
        counts["unknown_reasoning_llm_calls"] += evidence == "no_visible_reasoning_tokens_unknown"
        if requested != "off":
            continue
        counts["no_thinking_llm_calls"] += 1
        counts["thinking_despite_no_thinking_llm_calls"] += evidence == "observed_thinking"
        counts["no_thinking_no_response_calls"] += evidence == "no_response"
        counts["no_thinking_unknown_reasoning_calls"] += evidence == "no_visible_reasoning_tokens_unknown"
        counts["no_thinking_zero_reasoning_calls"] += evidence == "zero_reported_no_visible_reasoning"
        counts["no_thinking_positive_reasoning_token_calls"] += (row["reasoning_tokens"] or 0) > 0
        counts["no_thinking_visible_reasoning_calls"] += row["reasoning_chars"] > 0 or row["think_tag"]
        counts["reasoning_text_with_zero_tokens"] += (row["reasoning_chars"] > 0 or row["think_tag"]) and row["reasoning_tokens"] == 0
    return {**counts, "coverage_gaps": gaps}


def refresh_thinking_summary(output):
    """Refresh a derived phase sidecar without modifying canonical run evidence."""
    output = Path(output)
    patterns = ("tasks/*/logs/lm_calls.jsonl", "setup_lm_calls.jsonl",
                "baseline_artifacts/*lm_calls.jsonl", "baseline_artifacts/*/*lm_calls.jsonl",
                "recovery_history/**/lm_calls.jsonl", "recovery_history/**/*_lm_calls.jsonl")
    paths = sorted({path for pattern in patterns for path in output.glob(pattern)})
    sources, gaps = {}, []
    for path in paths:
        relative = str(path.relative_to(output))
        try:
            stat = path.stat()
            sources[relative] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError as error:
            gaps.append({"path": relative, "reason": f"Cannot inspect log: {type(error).__name__}"})
    manifest_path = output / "run_manifest.json"
    manifest_source = None
    try:
        stat = manifest_path.stat()
        manifest_source = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except FileNotFoundError:
        pass
    except OSError as error:
        gaps.append({"path": "run_manifest.json", "reason": f"Cannot inspect task coverage: {type(error).__name__}"})
    detector_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = output / "thinking_summary.json"
    if destination.is_file():
        try:
            saved = json.loads(destination.read_text())
        except (OSError, json.JSONDecodeError):
            saved = {}
        if (not gaps and isinstance(saved, dict) and saved.get("sources") == sources
                and saved.get("manifest_source") == manifest_source
                and saved.get("detector_sha256") == detector_sha256):
            return saved
    if manifest_source is not None:
        try:
            manifest = mapping(json.loads(manifest_path.read_text()))
            expected = set(manifest.get("completed_task_ids", []))
            if manifest.get("current_task_id"):
                expected.add(manifest["current_task_id"])
            for task in sorted(expected):
                relative = f"tasks/{task}/logs/lm_calls.jsonl"
                if relative not in sources:
                    gaps.append({"path": relative, "reason": "Expected task call log is missing"})
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            gaps.append({"path": "run_manifest.json", "reason": f"Cannot read task coverage: {type(error).__name__}"})

    def calls():
        for path in paths:
            relative = str(path.relative_to(output))
            if relative not in sources:
                continue
            try:
                with path.open(encoding="utf-8") as stream:
                    for number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            gaps.append({"path": relative, "line": number,
                                         "reason": "Malformed JSON record" if line.endswith("\n") else "Incomplete log tail"})
            except (OSError, UnicodeError) as error:
                gaps.append({"path": relative, "reason": f"Cannot read log: {type(error).__name__}"})
            try:
                stat = path.stat()
                if sources[relative] != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                    gaps.append({"path": relative, "reason": "Log changed during collection"})
            except OSError as error:
                gaps.append({"path": relative, "reason": f"Log unavailable after reading: {type(error).__name__}"})

    result = summarize_thinking(calls())
    result["coverage_gaps"].extend(gaps)
    if not paths:
        result["coverage_gaps"].append({"reason": "No canonical LLM call logs found"})
    result.update(
        status="unavailable" if not sources else "partial" if result["coverage_gaps"] else "complete",
        generated_at=datetime.now(timezone.utc).isoformat(), detector_sha256=detector_sha256,
        sources=sources, manifest_source=manifest_source,
        coverage="Distinct recorded physical chat attempts across tasks, initialization and archived recovery attempts. "
                 "Retries count separately; embeddings and response-cache replays are excluded. "
                 "Explicit disabling controls only; conflicting controls are unknown. Observable thinking means "
                 "positive reasoning tokens, nonempty reasoning fields/blocks, or a leading nonempty think block. "
                 "Missing telemetry and no response are unknown, not proof of no hidden thinking. "
                 "Recovery phases include inherited calls; do not sum parent and recovery rows as independent traffic.",
    )
    if output.is_dir():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                             prefix=".thinking_summary.", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(result, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return result
