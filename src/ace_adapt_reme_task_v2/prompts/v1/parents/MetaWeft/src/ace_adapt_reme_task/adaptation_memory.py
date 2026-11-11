"""Full content-only playbook preparation and atomic ADD/UPDATE validation."""

from copy import deepcopy
from pathlib import Path
import re
import time

from MetaWeft.src.framework import PreparedMemory, BranchUpdateResult
from .common import FULL_PROMPT_MAX_BYTES, SchemaError, PromptOverflow, call_model, sha_text, text, write_json
from .evidence import adaptation_projection, select_adaptation_evidence, validate_pointer


HEADINGS = ("selection", "specialization", "composition", "verification")


def empty_playbook():
    return {"schema_version": 1, "revision": 0, "next_entry_number": 1, "entries": []}


def validate_playbook(playbook):
    if not isinstance(playbook, dict) or set(playbook) != {"schema_version", "revision", "next_entry_number", "entries"}:
        raise ValueError("Invalid playbook wrapper")
    if type(playbook["schema_version"]) is not int or playbook["schema_version"] != 1:
        raise ValueError("Unsupported playbook schema")
    if type(playbook["revision"]) is not int or playbook["revision"] < 0:
        raise ValueError("Invalid playbook revision")
    if type(playbook["next_entry_number"]) is not int or playbook["next_entry_number"] < 1:
        raise ValueError("Invalid playbook next-entry counter")
    if not isinstance(playbook["entries"], list):
        raise ValueError("Playbook entries must be a list")
    numbers = set()
    for entry in playbook["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"id", "heading", "content", "source_refs"}:
            raise ValueError("Playbook entry requires only id, heading, content and source_refs")
        identifier = entry["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"adapt-[0-9]{6,}", identifier):
            raise ValueError("Invalid playbook entry ID")
        number = int(identifier[6:])
        if number < 1 or identifier != f"adapt-{number:06d}" or number in numbers:
            raise ValueError("Duplicate or noncanonical playbook entry ID")
        numbers.add(number)
        if not isinstance(entry["heading"], str) or entry["heading"] not in HEADINGS:
            raise ValueError("Invalid playbook heading")
        text(entry["content"], "content")
        if (not isinstance(entry["source_refs"], list) or not entry["source_refs"]
                or any(not isinstance(ref, str) or not ref.strip() for ref in entry["source_refs"])
                or len(set(entry["source_refs"])) != len(entry["source_refs"])):
            raise ValueError("Playbook source_refs must be unique nonempty references")
    # No REMOVE exists: every allocated ID remains, and revisions record nonempty batches.
    if numbers != set(range(1, playbook["next_entry_number"])):
        raise ValueError("Playbook counter does not match allocated IDs")
    if bool(numbers) != bool(playbook["revision"]):
        raise ValueError("Playbook revision does not match its empty/nonempty state")
    return deepcopy(playbook)


def render_playbook(playbook):
    state = validate_playbook(playbook)
    sections = []
    for heading in HEADINGS:
        lines = [f"## {heading.title()}"]
        for entry in sorted((row for row in state["entries"] if row["heading"] == heading), key=lambda row: row["id"]):
            lines.append(f"[{entry['id']}] " + entry["content"].replace("\n", "\n    "))
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def apply_edits(prior, proposal, projection, *, source_ref, adaptation_template,
                max_bytes=FULL_PROMPT_MAX_BYTES):
    prior = validate_playbook(prior)
    if (not isinstance(proposal, dict) or set(proposal) != {"base_revision", "edits"}
            or type(proposal["base_revision"]) is not int
            or proposal["base_revision"] != prior["revision"] or not isinstance(proposal["edits"], list)):
        raise SchemaError("Edit plan requires the exact current base_revision and an edits list")
    entries = {entry["id"]: entry for entry in prior["entries"]}
    targeted = set()
    allowed = {"adaptation", "retrieved_memories", "prior_playbook", "base_revision", "task_spec", "trace", "reward"}
    for edit in proposal["edits"]:
        if not isinstance(edit, dict) or set(edit) != {"operation", "target_id", "heading", "content", "evidence", "reason"}:
            raise SchemaError("Invalid playbook edit fields")
        operation, target = edit["operation"], edit["target_id"]
        if not isinstance(operation, str) or operation not in {"ADD", "UPDATE"}:
            raise SchemaError("Only ADD and UPDATE playbook operations are supported")
        if operation == "ADD" and target is not None:
            raise SchemaError("ADD target must be null")
        if operation == "UPDATE":
            if not isinstance(target, str) or target not in entries or target in targeted:
                raise SchemaError("UPDATE target must be an untargeted existing prior ID")
            targeted.add(target)
        if not isinstance(edit["heading"], str) or edit["heading"] not in HEADINGS:
            raise SchemaError("Unknown playbook heading")
        text(edit["content"], "content")
        text(edit["reason"], "reason")
        pointers = edit["evidence"]
        if (not isinstance(pointers, list) or not pointers
                or any(not isinstance(pointer, str) for pointer in pointers) or len(set(pointers)) != len(pointers)):
            raise SchemaError("Edit evidence must be a nonempty set of JSON pointers")
        for pointer in pointers:
            if not pointer.startswith("/") or pointer.split("/")[1] not in allowed:
                raise SchemaError("Edit evidence target is not permitted")
            try:
                validate_pointer(projection, pointer)
            except ValueError as error:
                raise SchemaError(str(error)) from error
        if projection["adaptation"]["status"] != "completed" or "/adaptation/guide" not in pointers:
            raise SchemaError("An edit requires the valid completed adapter guide")
        if not any(re.fullmatch(r"/trace/(0|[1-9][0-9]*)/observation", pointer)
                   and isinstance(validate_pointer(projection, pointer), str) for pointer in pointers):
            raise SchemaError("An edit requires an actual recorded trajectory observation")
    # Nothing mutates until every edit has passed validation against the prior state.
    result, applied = deepcopy(prior), []
    for index, edit in enumerate(proposal["edits"]):
        reference = f"{source_ref}#/applied/{index}"
        if edit["operation"] == "ADD":
            identifier = f"adapt-{result['next_entry_number']:06d}"
            result["next_entry_number"] += 1
            before = None
            after = {"id": identifier, "heading": edit["heading"], "content": edit["content"], "source_refs": [reference]}
            result["entries"].append(after)
        else:
            position = next(i for i, row in enumerate(result["entries"]) if row["id"] == edit["target_id"])
            before = deepcopy(result["entries"][position])
            after = {**before, "heading": edit["heading"], "content": edit["content"],
                     "source_refs": list(dict.fromkeys([*before["source_refs"], reference]))}
            result["entries"][position] = after
        applied.append({"operation": edit["operation"], "before": before, "after": deepcopy(after),
                        "evidence": deepcopy(edit["evidence"]), "reason": edit["reason"]})
    if applied:
        result["revision"] += 1
    result = validate_playbook(result)
    if len((adaptation_template + render_playbook(result)).encode("utf-8")) > max_bytes:
        raise SchemaError("Resulting playbook plus adaptation system prompt exceeds the full-prompt byte guard")
    return result, applied


class AdaptationMemory:
    def __init__(self, run_root, templates, playbook=None, *, max_bytes=FULL_PROMPT_MAX_BYTES):
        self.run_root = Path(run_root).resolve()
        self.templates = templates
        self.playbook = validate_playbook(empty_playbook() if playbook is None else playbook)
        self.max_bytes = max_bytes

    def prepare(self, task_spec, prior_state, *, services, artifacts):
        started = time.monotonic()
        state = validate_playbook(self.playbook)
        rendered = render_playbook(state)
        payload = {"state": state, "rendered": rendered, "rendered_sha256": sha_text(rendered),
                   "ids": [entry["id"] for entry in state["entries"]], "revision": state["revision"],
                   "entry_count": len(state["entries"]), "rendered_bytes": len(rendered.encode("utf-8")),
                   "duration_seconds": time.monotonic() - started}
        path = Path(artifacts) / "adaptation_preparation.json"
        reference = str(path.resolve().relative_to(self.run_root))
        result = PreparedMemory("adaptation", prior_state, payload, trace_refs=(reference,))
        write_json(path, result.to_dict())
        return result

    select_evidence = staticmethod(select_adaptation_evidence)

    def update(self, episode, prior_state, *, services, artifacts):
        projection = adaptation_projection(episode)
        prior = episode.payload["prior_playbook"]["state"]
        if self.playbook != prior:
            raise ValueError("Active playbook changed since preparation")
        path = Path(artifacts) / "adaptation_updates.json"
        reference = str(path.resolve().relative_to(self.run_root))
        result = BranchUpdateResult("adaptation", selected_evidence_refs=[episode.reference],
                                    proposed_refs=[reference], applied_refs=[reference], before=prior_state.to_dict())
        event = {"applied": [], "validation": None, "result": deepcopy(prior)}
        save = lambda: write_json(path, event)
        try:
            proposal = call_model("adaptation_update", self.templates["adaptation_update"], projection,
                                  services=services, event=event, save=save, max_bytes=self.max_bytes)
            updated, applied = apply_edits(prior, proposal, projection, source_ref=reference,
                                           adaptation_template=self.templates["adaptation"], max_bytes=self.max_bytes)
            event.update(status="completed", validation={"valid": True}, applied=applied, result=updated)
            # Preserve the complete draft before it can become active.
            save()
            self.playbook = updated
        except (SchemaError, PromptOverflow) as error:
            result.status, result.degraded, result.error = "degraded", True, str(error)
            event.update(status="budget_limited" if isinstance(error, PromptOverflow) else "degraded",
                         validation={"valid": False, "error": str(error)}, result=deepcopy(prior))
            save()
        result.after = {"draft": reference + "#/result", "revision": self.playbook["revision"],
                        "entry_count": len(self.playbook["entries"]), "applied_count": len(event["applied"]),
                        "rendered_bytes": len(render_playbook(self.playbook).encode("utf-8"))}
        return result
