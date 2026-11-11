"""Adaptation-only evidence from a frozen successful adapter event."""

from copy import deepcopy

from MetaWeft.src.ace_adapt_reme_task.evidence import validate_pointer


def adaptation_projection(episode):
    from .adaptation_memory import HEADINGS, playbook_content, validate_playbook

    episode.verify()
    value = episode.payload
    state = validate_playbook(value["prior_playbook"]["state"])
    event = value["adaptation"]
    if event["status"] != "completed" or not isinstance(event["guide"], str):
        raise ValueError("Adaptation update requires a completed valid adapter event")
    if event["input"]["playbook"] != playbook_content(state):
        raise ValueError("Frozen adapter playbook differs from the episode's prior state")
    targets = {}
    for heading in HEADINGS:
        entries = sorted((entry for entry in state["entries"] if entry["heading"] == heading),
                         key=lambda entry: entry["id"])
        for index, entry in enumerate(entries):
            targets[entry["id"]] = f"/adaptation/input/playbook/{heading}/{index}"
    return {"adaptation": {"input": deepcopy(event["input"]), "output": {"guidance": event["guide"]}},
            "base_revision": state["revision"], "edit_targets": targets}


def substantive_input_pointer(projection, pointer):
    value = validate_pointer(projection, pointer)
    if not isinstance(value, str) or not value.strip():
        return False
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
    if tokens[:2] != ["adaptation", "input"]:
        return False
    path = tokens[2:]
    if path[:1] == ["retrieved_memories"]:
        return len(path) == 2
    if path[:1] == ["playbook"]:
        return len(path) == 3
    if path[:2] == ["task_spec", "instruction"]:
        return len(path) == 2
    if path[:2] == ["task_spec", "app_descriptions"]:
        return True
    return (len(path) == 3 and path[:2] == ["task_spec", "supervisor"]
            and path[2] in {"first_name", "last_name", "email", "phone_number"})
