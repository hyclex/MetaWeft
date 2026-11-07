"""Bounded procedural-memory selection and grading over existing records."""

from copy import deepcopy
import hashlib
import json
import time

from MetaWeft.service.llm.client import InvalidResponseError, LLMUnavailableError


# Frozen correction text from memory_retrieval_01's captured runner/live.py.
SCHEMA_CORRECTION = (
    "The previous response did not satisfy the output schema. "
    "Use the ORIGINAL task and candidate memories above. Return only the requested JSON "
    "object with valid unique local indices, the exact required count, and no other fields. "
    "Do not echo or invent an input task or candidate list."
)


class RankingFailure(ValueError):
    """Nonempty ranking output exhausted its configured schema recovery."""


def _json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object key")
        value[key] = item
    return value


def parse_ranking_response(text, key="ranking", accept_wrapped=False):
    """Normalize only the historical whole-response wrappers, then parse JSON."""
    normalized = False
    text = text.strip()
    if accept_wrapped and text.startswith("```") and text.endswith("```"):
        first, _, rest = text.partition("\n")
        if first not in {"```", "```json"}:
            raise ValueError("Unsupported response code-fence language")
        text, normalized = rest[:-3].strip(), True
    value = json.loads(text, object_pairs_hook=_json_object)
    # Python accepts NaN/Infinity and overflowing floats; traces must remain JSON.
    json.dumps(value, allow_nan=False)
    if accept_wrapped and isinstance(value, list):
        if key == "ranking" and all(isinstance(item, dict) and set(item) == {"index"}
                                    for item in value):
            value = [item["index"] for item in value]
        value, normalized = {key: value}, True
    return value, normalized


def rerank_indices(value, candidate_count, repair_duplicates=False):
    """Validate twenty local indices; optionally fill duplicate slots in input order."""
    indices = value.get("ranking") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or set(value) != {"ranking"}
            or not isinstance(indices, list) or len(indices) != 20
            or any(type(index) is not int or not 1 <= index <= candidate_count
                   for index in indices)):
        raise ValueError("Reranker must return exactly20 valid local integer IDs")
    unique = list(dict.fromkeys(indices))
    repaired = len(unique) != len(indices)
    if repaired and not repair_duplicates:
        raise ValueError("Reranker returned duplicate local indices")
    if repaired:
        unique += [index for index in range(1, candidate_count + 1)
                   if index not in unique][:20 - len(unique)]
    return unique, repaired


def parse_scores(value, candidate_count):
    """Require every index once; rank integer utility grades with stable ties."""
    if (not isinstance(value, dict) or set(value) != {"scores"}
            or not isinstance(value["scores"], list)
            or len(value["scores"]) != candidate_count):
        raise ValueError("Require exactly one score pair per candidate")
    scores = {}
    for pair in value["scores"]:
        if (not isinstance(pair, list) or len(pair) != 2
                or type(pair[0]) is not int or not 1 <= pair[0] <= candidate_count
                or type(pair[1]) is not int or not 0 <= pair[1] <= 100):
            raise ValueError("Each pair must contain a valid integer index and score 0..100")
        index, score = pair
        if index in scores:
            raise ValueError("Duplicate candidate index")
        scores[index] = score
    return sorted(scores, key=lambda index: (-scores[index], index))


def _error_detail(error):
    detail = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, LLMUnavailableError):
        detail.update(operation=error.operation, phase=error.phase,
                      logical_call_id=error.logical_call_id, attempts=error.attempts)
        if error.__cause__ is not None:
            detail["cause"] = {"type": type(error.__cause__).__name__,
                               "message": str(error.__cause__)}
    return detail


def _empty_fallback(error, policy):
    cause = error.__cause__
    return (policy["fallback_on_empty_response"] and policy["fallback_to_input"]
            and error.operation == "chat" and type(cause) is InvalidResponseError
            and str(cause) == "Chat response contains no text")


def run_stage(task, records, *, llm, stage, prompt, record_stage):
    """Run one validated v2 stage; callbacks own artifacts and canonical call refs.

    Provider and artifact errors propagate. Only declared schema recovery or the
    exact exhausted-empty-response exception can yield a usable input fallback.
    Neither records, stage configuration, nor supplied client settings are changed.
    """
    started = time.monotonic()
    count = len(records)
    limit = count if stage["output_limit"] == "all" else stage["output_limit"]
    policy, generation = stage["response"], stage["generation"]
    controls = {"model": stage["model"], "temperature": generation["temperature"],
                "seed": generation["seed"],
                "extra_body": {"enable_thinking": generation["enable_thinking"]}}
    if generation["enable_thinking"]:
        controls["extra_body"]["thinking_budget"] = generation["thinking_budget"]
        controls["max_completion_tokens"] = generation["max_completion_tokens"]
    else:
        controls["max_tokens"] = generation["max_tokens"]
    trace = {
        "id": stage["id"], "method": stage["method"], "model": stage["model"],
        "prompt": stage["prompt"], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generation": deepcopy(generation), "controls": deepcopy(controls),
        "response_policy": deepcopy(policy), "fields": list(stage["fields"]),
        "input_ids": [record["id"] for record in records], "input_count": count,
        "local_indices": [{"index": index, "id": record["id"]}
                          for index, record in enumerate(records, 1)],
        "output_limit": stage["output_limit"], "output_ids": [], "output_count": 0,
        "ranking": [], "grades": None, "attempts": [], "schema_retries": 0,
        "envelope_normalized": False, "duplicate_repair": False,
        "fallback": None, "skip": None, "error": None, "status": "running",
        "timing": {"elapsed_seconds": 0.0},
    }

    def emit():
        trace["timing"]["elapsed_seconds"] = time.monotonic() - started
        record_stage(deepcopy(trace))

    def finish(indices, status, *, fallback=None):
        output = [records[index - 1] for index in indices]
        trace.update(status=status, ranking=indices,
                     output_ids=[record["id"] for record in output], output_count=len(output),
                     fallback=fallback, error=fallback)
        emit()
        return output, trace

    def fail(error):
        trace.update(status="failed", error=_error_detail(error))
        emit()

    if stage["method"] == "listwise" and count < 20:
        trace["skip"] = {"reason": "insufficient_for_fixed20", "input_count": count,
                         "required_count": 20, "policy": stage["small_input_policy"]}
        return finish(list(range(1, count + 1)), "skipped")

    documents = [{"index": index, **{field: record[field] for field in stage["fields"]}}
                 for index, record in enumerate(records, 1)]
    key = "scores" if stage["method"] == "scores" else "ranking"
    payload = ({"task": task, "memories": documents, "score_count": count} if key == "scores"
               else {"task": task, "return_count": 20, "memories": documents})
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    for schema_attempt in range(policy["schema_retries"] + 1):
        component = f"retrieval.reranking.{stage['id']}.attempt_{schema_attempt}"
        attempt = {"component": component, "schema_attempt": schema_attempt,
                   "status": "running", "response_sha256": None, "parsed": None,
                   "envelope_normalized": False, "duplicate_repair": False, "error": None}
        trace["attempts"].append(attempt)
        trace["schema_retries"] = schema_attempt
        emit()
        try:
            response = llm.chat(messages, phase="retrieval", component=component, **controls)
        except LLMUnavailableError as error:
            attempt.update(status="failed", error=_error_detail(error))
            if _empty_fallback(error, policy):
                fallback = {**_error_detail(error), "policy": "fallback_on_empty_response",
                            "component": component, "schema_attempt": schema_attempt}
                return finish(list(range(1, min(count, limit) + 1)),
                              "completed_with_fallback", fallback=fallback)
            fail(error)
            raise
        except Exception as error:
            attempt.update(status="failed", error=_error_detail(error))
            fail(error)
            raise
        # Invalid client response shapes are programming/service errors, not schema repairs.
        text = response.choices[0].message.content
        attempt["response_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        try:
            value, normalized = parse_ranking_response(text, key, policy["accept_wrapped"])
            attempt.update(parsed=value, envelope_normalized=normalized)
            trace["envelope_normalized"] |= normalized
            if key == "scores":
                indices, repaired = parse_scores(value, count), False
                trace["grades"] = value["scores"]
            else:
                indices, repaired = rerank_indices(value, count, policy["repair_duplicates"])
        except (TypeError, ValueError) as error:
            attempt.update(status="schema_error", error=_error_detail(error))
            if schema_attempt == policy["schema_retries"]:
                if policy["fallback_to_input"]:
                    fallback = {**_error_detail(error), "policy": "fallback_to_input",
                                "component": component, "schema_attempt": schema_attempt}
                    return finish(list(range(1, min(count, limit) + 1)),
                                  "completed_with_fallback", fallback=fallback)
                failure = RankingFailure(str(error))
                fail(failure)
                raise failure from error
            emit()
            messages = [*messages, {"role": "assistant", "content": text},
                        {"role": "user", "content": SCHEMA_CORRECTION}]
            continue
        attempt.update(status="completed", duplicate_repair=repaired)
        trace["duplicate_repair"] |= repaired
        return finish(indices[:limit], "completed")
