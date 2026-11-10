"""Allowlisted evidence and bounded JSON-pointer resolution for V1."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re


def task_view(task):
    result = deepcopy({key: task[key] for key in
                       ("task_id", "instruction", "app_descriptions") if key in task})
    if "supervisor" in task:
        result["supervisor"] = {key: task["supervisor"][key] for key in
                                ("first_name", "last_name", "email", "phone_number") if key in task["supervisor"]}
    return result


def evidence_view(evidence):
    reward = evidence["reward"]
    if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Evidence reward must be an official scalar in [0, 1]")
    trace = []
    for step in evidence["trace"]:
        clean = {key: step[key] for key in ("step", "code", "observation") if key in step}
        if any(not isinstance(clean[key], str) for key in ("code", "observation") if key in clean):
            raise ValueError("Trajectory code and observation must be text")
        if "step" in clean and (type(clean["step"]) is not int or clean["step"] < 0):
            raise ValueError("Trajectory step must be a nonnegative integer")
        trace.append(clean)
    return {"task_spec": task_view(evidence["task_spec"]), "trace": trace, "reward": reward}


def memory_view(record):
    keys = ("id", "name", "when_to_use", "content", "source_refs")
    return deepcopy({key: record[key] for key in keys})


def validate_pointer(value, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Evidence reference must be an absolute JSON pointer")
    current = value
    for token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("Invalid JSON pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", token) and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ValueError(f"Evidence pointer does not resolve: {pointer}")
    return current


def reference_path(run_root, reference):
    root = Path(run_root).resolve()
    path = Path(reference)
    resolved = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root):
        raise ValueError(f"Invalid artifact reference: {reference}")
    return resolved


def validate_source_reference(run_root, reference):
    if not isinstance(reference, str) or not reference:
        raise ValueError("Source reference must be a nonempty string")
    file, separator, pointer = reference.partition("#")
    if not file:
        raise ValueError("Source reference must name a file")
    value = json.loads(reference_path(run_root, file).read_text())
    return validate_pointer(value, pointer) if separator else value


def task_projection(episode):
    value = episode.payload
    return {**evidence_view(value), "retrieved_memories": deepcopy(value["retrieved_memories"]),
            "guide": value["adaptation"]["guide"]}


def adaptation_projection(episode):
    value = episode.payload
    return {**evidence_view(value), "retrieved_memories": deepcopy(value["retrieved_memories"]),
            "prior_playbook": value["prior_playbook"]["rendered"],
            "base_revision": value["prior_playbook"]["state"]["revision"],
            "adaptation": {key: deepcopy(value["adaptation"][key]) for key in
                           ("status", "guide", "messages", "raw_output")}}


def select_task_evidence(episode):
    episode.verify()
    return deepcopy(episode)


def select_adaptation_evidence(episode):
    episode.verify()
    return deepcopy(episode)
