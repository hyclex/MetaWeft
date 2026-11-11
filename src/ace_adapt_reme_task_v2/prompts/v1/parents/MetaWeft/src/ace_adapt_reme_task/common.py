"""Strict JSON/model-call helpers shared only by the V1 memory stages."""

from copy import deepcopy
import hashlib
import json
import time

FULL_PROMPT_MAX_BYTES = 900_000


def write_json(path, value):
    # Reuse the durable artifact primitive without a package-import cycle.
    from MetaWeft.experiment.recovery import write_json as durable_write

    durable_write(path, value)


class SchemaError(ValueError):
    """A completed model response failed the declared output schema."""


class PromptOverflow(ValueError):
    """The complete preserved request exceeds the recorded byte guard."""


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} must be a nonempty string")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_object(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (TypeError, ValueError) as error:
        raise SchemaError("Model response must be one JSON object; no repair was attempted") from error
    if not isinstance(value, dict):
        raise SchemaError("Model response must be one JSON object")
    return value


def sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def call_model(name, template, payload, *, services, event, save, max_bytes=FULL_PROMPT_MAX_BYTES):
    messages = [{"role": "system", "content": template},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
    component = f"metaweft_v1.{name}"
    started = time.monotonic()
    prior_calls = len(services.get("call_refs", {}).get(component, []))
    llm = services["llm"]
    settings = {"temperature": 0.0, "seed": 0, "extra_body": {"enable_thinking": False}}
    event.update(messages=deepcopy(messages), input=deepcopy(payload), status="pending", raw_output=None,
                 prompt_sha256=sha_text(template),
                 prompt_bytes=sum(len(message["content"].encode("utf-8")) for message in messages),
                 full_prompt_max_bytes=max_bytes,
                 model_settings={**settings, "model": getattr(llm, "chat_model", None),
                                 **{key: getattr(llm, key, None) for key in
                                    ("max_retries", "retry_delay_seconds", "chat_timeout", "cache_mode", "cache_path")}},
                 call_references=[])
    if event["model_settings"]["cache_path"] is not None:
        event["model_settings"]["cache_path"] = str(event["model_settings"]["cache_path"])
    save()
    if event["prompt_bytes"] > max_bytes:
        event.update(status="budget_limited", error=f"Full {name} prompt is {event['prompt_bytes']} UTF-8 bytes, "
                     f"exceeding the {max_bytes}-byte safety limit; no input was truncated")
        event["duration_seconds"] = time.monotonic() - started
        save()
        raise PromptOverflow(event["error"])
    try:
        response = llm.chat(messages, phase=name, component=component, **settings)
        try:
            event["raw_output"] = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise SchemaError("Model response has no returned content") from error
        value = parse_object(event["raw_output"])
        event.update(status="returned", proposal=deepcopy(value))
        return value
    except Exception as error:
        event.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        event["duration_seconds"] = time.monotonic() - started
        event["call_references"] = deepcopy(services.get("call_refs", {}).get(component, [])[prior_calls:])
        save()
