"""Completed-task checkpoints and immutable-parent recovery for experiment launchers.

New entrypoints must share these arguments, identity checks, and commit boundaries.
No benchmark or provider calls belong in this module.
"""

from contextlib import contextmanager
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


RECOVERY_VERSION = 1
PAIR_RECOVERY_VERSION = 2
TASK_MEMORY_SCHEMA = "metaweft-task-memory-v1"
ADAPTATION_MEMORY_SCHEMA = "metaweft-adaptation-memory-v1"
V1_TASK_ARTIFACTS = (
    "task_spec.json", "retrieval.json", "adaptation_preparation.json", "adaptation_event.json",
    "context.json", "trace.jsonl", "evidence.json", "episode_evidence.json",
    "task_reflection.json", "task_updates.json", "adaptation_updates.json",
    "memory_after.json", "adaptation_memory_after.json", "stages.json", "result.json",
)


def recovery_version(setting):
    return PAIR_RECOVERY_VERSION if setting == "metaweft_v1" else RECOVERY_VERSION


def stream_id(plan, phase):
    origin = plan.get("recovery", {}).get("origin_run_id", plan["run_id"])
    return f"{origin}:{phase['split']}"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Publish a complete, durable JSON file without exposing partial writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid artifact reference: {name}")
    return Path(root) / path


def verify_files(root, files, *, complete=False):
    if complete:
        actual = {str(path.relative_to(root)) for path in Path(root).rglob("*") if path.is_file()}
        if actual != set(files):
            raise ValueError(f"Captured source file inventory changed: {root}")
    for name, expected in files.items():
        path = relative_file(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed recovery artifact: {path}")


def file_hashes(root, paths):
    result = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for item in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if item.is_file():
                with item.open("rb") as stream:
                    os.fsync(stream.fileno())
                result[str(item.relative_to(root))] = sha256(item)
    return result


def add_recovery_argument(parser):
    parser.add_argument("--recover", nargs="?", const="latest", metavar="latest|RUN_DIRECTORY",
                        help="Continue a compatible failed new-format run in a new directory")


def selection(setting, args):
    return {"setting": setting, "dataset": args.dataset, "split": args.split,
            "limit": args.limit, "task_id": args.task_id}


def identity(plan, root):
    record = {
        "version": plan["recovery_version"], "selection": plan["selection"],
        "settings": plan["settings"], "variant_id": plan["variant_id"], "setting": plan["setting"],
        "scenario": plan["scenario"], "data_source": plan["data_source"],
        "provider_url": plan["provider"]["chat_base_url"],
        "phases": [{k: phase[k] for k in ("split", "dataset", "experiment_name", "ordered_task_ids",
                                          "source_task_count", "partial",
                                          "dataset_sha256", "selected_dataset_sha256")}
                   for phase in plan["phases"]],
        "artifacts": {name: sha256(Path(root) / name) for name in (
            "variant.json", "source_provenance.json", "data_provenance.json",
            "runtime.json", "appworld-requirements.txt")},
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def seal_run(plan):
    plan["recovery_identity"] = identity(plan, Path(plan["run_directory"]))
    plan["recovery_ready"] = True


def phase_output(root, phase):
    return relative_file(root, f"experiments/outputs/{phase['experiment_name']}")


def _validate_task_records(records, dimensions):
    if not isinstance(records, list):
        raise ValueError("Task snapshot records must be a list")
    identifiers = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"id", "document"}:
            raise ValueError("Invalid task snapshot record")
        identifier, document = record["id"], record["document"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("Task snapshot contains invalid or duplicate IDs")
        identifiers.add(identifier)
        if not isinstance(document, dict) or set(document) != {
                "id", "name", "when_to_use", "content", "source_refs", "embedding"}:
            raise ValueError("Invalid V1 task memory schema")
        if document["id"] != identifier:
            raise ValueError("Task snapshot document ID differs from its stored ID")
        for key in ("name", "when_to_use", "content"):
            if not isinstance(document[key], str) or not document[key].strip():
                raise ValueError(f"Task memory {key} must be nonempty text")
        references = document["source_refs"]
        if (not isinstance(references, list) or not references
                or any(not isinstance(ref, str) or not ref for ref in references)
                or len(set(references)) != len(references)):
            raise ValueError("Task memory requires unique source references")
        vector = document["embedding"]
        if (not isinstance(vector, list) or len(vector) != dimensions
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
            raise ValueError("Task snapshot must preserve its complete finite embedding vector")
    if records != sorted(records, key=lambda record: record["id"]):
        raise ValueError("Task snapshot records are not canonically ordered")


def _snapshot_wrapper(role, plan, phase, epoch):
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Pair epoch must be a nonnegative integer")
    return {"role": role, "schema": TASK_MEMORY_SCHEMA if role == "task" else ADAPTATION_MEMORY_SCHEMA,
            "stream_id": stream_id(plan, phase), "state_epoch": epoch, "status": "verified"}


def snapshot_task_v2(store, path, plan, phase, epoch, task_id=None):
    """Required snapshots fail closed; a failed export is never an empty memory."""
    snapshot = {**_snapshot_wrapper("task", plan, phase, epoch),
                "records": sorted(store.export(), key=lambda record: record["id"])}
    _validate_task_records(snapshot["records"], plan["settings"]["embedding_dimensions"])
    write_json(path, snapshot)
    return snapshot


def snapshot_adaptation_v2(playbook, path, plan, phase, epoch, task_id=None):
    from MetaWeft.src.ace_adapt_reme_task.adaptation_memory import validate_playbook
    snapshot = {**_snapshot_wrapper("adaptation", plan, phase, epoch),
                "playbook": validate_playbook(playbook)}
    if snapshot["playbook"]["revision"] > epoch:
        raise ValueError("Playbook revision exceeds its completed-task epoch")
    write_json(path, snapshot)
    return snapshot


def restore_adaptation(snapshot):
    from MetaWeft.src.ace_adapt_reme_task.adaptation_memory import validate_playbook
    if (snapshot.get("status") != "verified" or snapshot.get("role") != "adaptation"
            or snapshot.get("schema") != ADAPTATION_MEMORY_SCHEMA):
        raise ValueError("Cannot restore an unverified adaptation snapshot")
    playbook = validate_playbook(snapshot["playbook"])
    if playbook != snapshot["playbook"]:
        raise ValueError("Restored playbook differs from its checkpoint")
    return playbook


def _pair_memories(completed):
    prefix = "baseline_artifacts/metaweft/"
    boundary = f"{completed[-1]}/" if completed else ""
    return {
        "task": {"snapshot": prefix + boundary + ("memory_after.json" if completed else "memory_initial.json"),
                 "mapping": prefix + "memory_mapping.json"},
        "adaptation": {"snapshot": prefix + boundary + ("adaptation_memory_after.json" if completed
                                                       else "adaptation_memory_initial.json")},
    }


def _source_file(root, output, reference, *, adaptation):
    from MetaWeft.src.ace_adapt_reme_task.evidence import validate_source_reference
    if not isinstance(reference, str):
        raise ValueError("Invalid memory source reference")
    name, separator, pointer = reference.partition("#")
    path = relative_file(root, name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_relative_to(output.resolve()):
        raise ValueError("Memory source reference leaves its stream")
    expected = "adaptation_updates.json" if adaptation else "evidence.json"
    if path.name != expected or bool(separator) != adaptation:
        raise ValueError("Memory source reference has the wrong evidence role")
    if adaptation:
        validate_source_reference(root, reference)
        if not pointer.startswith("/applied/"):
            raise ValueError("Playbook source reference must identify an applied edit")
    else:
        evidence = read_json(path)
        reward = evidence.get("reward")
        if (not isinstance(evidence.get("task_spec"), dict) or not isinstance(evidence.get("trace"), list)
                or type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1):
            raise ValueError("Invalid preserved task evidence")
    return path


def _validate_pair(plan, phase, completed, output, memories, files=None):
    if memories != _pair_memories(completed):
        raise ValueError("Checkpoint memories do not match the completed task boundary")
    epoch, sources = len(completed), []
    for role, references in memories.items():
        name = references["snapshot"]
        if files is not None and name not in files:
            raise ValueError("Checkpoint pair snapshot is not hash-verified")
        snapshot = read_json(relative_file(output, name))
        expected = _snapshot_wrapper(role, plan, phase, epoch)
        if any(snapshot.get(key) != value or type(snapshot.get(key)) is not type(value)
               for key, value in expected.items()):
            raise ValueError("Checkpoint pair has invalid role, schema, stream or epoch")
        payload_key = "records" if role == "task" else "playbook"
        if set(snapshot) != {*expected, payload_key}:
            raise ValueError("Invalid pair snapshot wrapper")
        if role == "task":
            _validate_task_records(snapshot["records"], plan["settings"]["embedding_dimensions"])
            items = [record["document"] for record in snapshot["records"]]
        else:
            playbook = restore_adaptation(snapshot)
            if playbook["revision"] > epoch:
                raise ValueError("Playbook revision exceeds its completed-task epoch")
            items = playbook["entries"]
            if epoch == 0 and (playbook["revision"] != 0 or playbook["next_entry_number"] != 1):
                raise ValueError("Initial playbook must have empty initialization counters")
        if epoch == 0 and items:
            raise ValueError("V1 initial memory pair must be empty")
        for item in items:
            for reference in item["source_refs"]:
                path = _source_file(Path(plan["run_directory"]), output, reference,
                                    adaptation=role == "adaptation")
                if files is not None and str(path.relative_to(output.resolve())) not in files:
                    raise ValueError("Memory source reference is not checkpoint-hashed")
                sources.append(path)
    mapping_name = memories["task"]["mapping"]
    if files is not None and mapping_name not in files:
        raise ValueError("Checkpoint omits memory mapping")
    mapping = read_json(relative_file(output, mapping_name))
    properties = mapping.get("mappings", {}).get("properties", {})
    if (properties.get("name") != {"type": "text"}
            or properties.get("embedding", {}).get("dims") != plan["settings"]["embedding_dimensions"]
            or mapping.get("settings", {}).get("index.mapping.exclude_source_vectors") is not False):
        raise ValueError("V1 memory mapping does not preserve names and complete vectors")
    return sources


def state_refs(plan, phase, checkpoint):
    """Return framework StateRef constructor fields for the committed pair."""
    if checkpoint["version"] != PAIR_RECOVERY_VERSION:
        raise ValueError("State references require a paired checkpoint")
    output, root = Path(phase["output_directory"]), Path(plan["run_directory"])
    return {role: {"role": role, "schema": TASK_MEMORY_SCHEMA if role == "task" else ADAPTATION_MEMORY_SCHEMA,
                   "stream_id": stream_id(plan, phase), "epoch": checkpoint["state_epoch"],
                   "snapshot": str((output / reference["snapshot"]).relative_to(root)),
                   "sha256": checkpoint["files"][reference["snapshot"]]}
            for role, reference in checkpoint["memories"].items()}


def _validate_task_artifacts(output, completed, files=None, *, paired=False):
    for task_id in completed:
        for directory in ("dbs", "checkpoints", "logs", "evaluation", "version", "misc"):
            path = output / "tasks" / task_id / directory
            if not path.is_dir():
                raise ValueError(f"Missing committed task directory: {path}")
        required = [f"baseline_artifacts/metaweft/{task_id}/result.json",
                    f"tasks/{task_id}/logs/lm_calls.jsonl", f"tasks/{task_id}/evaluation/report.md"]
        if paired:
            required.extend(f"baseline_artifacts/metaweft/{task_id}/{name}" for name in V1_TASK_ARTIFACTS)
        for name in required:
            if not relative_file(output, name).is_file() or (files is not None and name not in files):
                raise ValueError(f"Checkpoint omits required task artifact: {name}")


def commit_checkpoint(plan, phase, completed, *, previous=None, finalized=False):
    """Commit only after the adapter has validated all outputs and memory."""
    output = Path(phase["output_directory"])
    ordered = phase["ordered_task_ids"]
    if completed != ordered[:len(completed)]:
        raise ValueError("Checkpoint completion must be an ordered task prefix")
    if finalized and completed != ordered:
        raise ValueError("Cannot finalize an incomplete task stream")
    artifacts = output / "baseline_artifacts/metaweft"
    version = recovery_version(plan["setting"])
    if plan.get("recovery_version", RECOVERY_VERSION) != version:
        raise ValueError("Recovery format does not match the saved setting")
    paired = version == PAIR_RECOVERY_VERSION
    if paired:
        memories = _pair_memories(completed)
        sources = _validate_pair(plan, phase, completed, output, memories)
        if previous:
            preserved = {str(path.relative_to(output)) for path in sources}
            preserved.add(memories["task"]["mapping"])
            verify_files(output, {name: digest for name, digest in previous["files"].items() if name in preserved})
        paths = [relative_file(output, item["snapshot"]) for item in memories.values()] + sources
        _validate_task_artifacts(output, completed, paired=True)
    else:
        memory = artifacts / (f"{completed[-1]}/memory_after.json"
                              if completed and plan["setting"] != "direct" else "memory_initial.json")
        snapshot = read_json(memory)
        expected_status = "disabled" if plan["setting"] == "direct" else "verified"
        if snapshot.get("status") != expected_status:
            raise RuntimeError("Cannot commit a checkpoint without verified memory")
        paths = [memory]
    files = dict(previous["files"]) if previous else {}
    if previous is None and (output / "recovery_history").is_dir():
        paths.append(output / "recovery_history")
    if plan["setting"] != "direct":
        paths.append(artifacts / "memory_mapping.json")
    old_count = len(previous["completed_task_ids"]) if previous else 0
    if previous and (previous["version"] != version or previous["identity"] != plan["recovery_identity"]):
        raise ValueError("Previous checkpoint has a different recovery format or identity")
    if previous and previous["completed_task_ids"] != completed[:old_count]:
        raise ValueError("Checkpoint commits cannot remove or reorder completed tasks")
    for task_id in completed[old_count:]:
        paths.extend([output / "tasks" / task_id, artifacts / task_id])
    if finalized:
        paths.extend([output / "evaluations", output / "summary.json"])
        # Official aggregate evaluation regenerates individual task reports.
        paths.extend(output / "tasks" / task_id / "evaluation" for task_id in completed)
    files.update(file_hashes(output, paths))
    checkpoint = {
        "version": version, "identity": plan["recovery_identity"],
        "ordered_task_ids": ordered, "completed_task_ids": list(completed),
        "next_task_id": ordered[len(completed)] if len(completed) < len(ordered) else None,
        "files": files,
        "finalized": finalized,
    }
    if paired:
        checkpoint.update(state_epoch=len(completed), memories=memories)
        _validate_pair(plan, phase, completed, output, memories, files)
    else:
        checkpoint["memory_file"] = str(memory.relative_to(output))
    write_json(output / "checkpoint.json", checkpoint)
    return checkpoint


def load_checkpoint(plan, phase, root):
    output = phase_output(root, phase)
    version = recovery_version(plan["setting"])
    if plan.get("recovery_version", RECOVERY_VERSION) != version:
        raise ValueError("Recovery format does not match the saved setting")
    path = output / "checkpoint.json"
    if not path.is_file():
        # A phase may not yet have initialized its store or started any task.
        manifest = read_json(output / "run_manifest.json") if (output / "run_manifest.json").is_file() else {}
        if manifest.get("completed_task_ids") or any((output / "tasks").glob("*")):
            raise ValueError(f"Missing checkpoint for started tasks: {output}")
        return None
    checkpoint = read_json(path)
    if checkpoint.get("version") != version or checkpoint.get("identity") != plan["recovery_identity"]:
        raise ValueError(f"Incompatible checkpoint: {path}")
    completed, ordered = checkpoint["completed_task_ids"], phase["ordered_task_ids"]
    if (checkpoint["ordered_task_ids"] != ordered or len(completed) > len(ordered)
            or completed != ordered[:len(completed)]
            or checkpoint["next_task_id"] != (ordered[len(completed)] if len(completed) < len(ordered) else None)):
        raise ValueError(f"Checkpoint task order differs: {path}")
    verify_files(output, checkpoint["files"])
    paired = version == PAIR_RECOVERY_VERSION
    if paired:
        if type(checkpoint.get("state_epoch")) is not int or checkpoint["state_epoch"] != len(completed):
            raise ValueError("Checkpoint pair epoch differs from completed task count")
        if "memory_file" in checkpoint:
            raise ValueError("Paired checkpoint must not contain a legacy memory pointer")
        _validate_pair(plan, phase, completed, output, checkpoint.get("memories"), checkpoint["files"])
    else:
        memory_name = checkpoint["memory_file"]
        expected_memory = "baseline_artifacts/metaweft/" + (
            f"{completed[-1]}/memory_after.json" if completed and plan["setting"] != "direct" else "memory_initial.json")
        if memory_name != expected_memory:
            raise ValueError("Checkpoint memory does not match its completed task boundary")
        if memory_name not in checkpoint["files"]:
            raise ValueError(f"Checkpoint memory is not verified: {path}")
        memory = read_json(relative_file(output, memory_name))
        if memory.get("status") != ("disabled" if plan["setting"] == "direct" else "verified"):
            raise ValueError(f"Checkpoint memory unavailable: {path}")
    _validate_task_artifacts(output, completed, checkpoint["files"], paired=paired)
    if plan["setting"] != "direct" and "baseline_artifacts/metaweft/memory_mapping.json" not in checkpoint["files"]:
        raise ValueError("Checkpoint omits memory mapping")
    if checkpoint["finalized"]:
        name = f"evaluations/{phase['dataset']}.json"
        if len(completed) != len(ordered) or name not in checkpoint["files"]:
            raise ValueError("Finalized checkpoint has incomplete coverage")
        if set(read_json(output / name)["individual"]) != set(ordered):
            raise ValueError("Saved evaluation coverage differs from the selected tasks")
    return checkpoint


def restore_memory(store, snapshot):
    if snapshot.get("status") != "verified":
        raise ValueError("Cannot restore unverified memory")
    records = snapshot["records"]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Memory checkpoint contains duplicate IDs")
    for record in records:
        store.put(record["id"], record["document"])
    if store.export() != sorted(records, key=lambda record: record["id"]):
        raise RuntimeError("Restored memory differs from its checkpoint")


def accounting(output):
    """Count physical calls and cache replays, retaining failures and unknown usage."""
    output = Path(output)
    manifest_path = output / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    reused = set(manifest.get("reused_task_ids", []))
    records, inherited = {}, set()
    paths = [*output.glob("tasks/*/logs/lm_calls.jsonl"),
             *output.glob("recovery_history/**/tasks/*/logs/lm_calls.jsonl")]
    for path in paths:
        old = path.relative_to(output).parts[0] == "recovery_history" or path.parts[-3] in reused
        for line in path.read_text().splitlines():
            call = json.loads(line)
            identifier = call["id"]
            if identifier in records and records[identifier] != call:
                raise ValueError(f"Conflicting records for physical call {identifier}")
            records[identifier] = call
            if old:
                inherited.add(identifier)

    def totals(calls):
        phases, tokens, cache_phases = Counter(), Counter(), Counter()
        missing = 0
        for call in calls:
            if call["metadata"].get("extra", {}).get("cache", {}).get("hit", False):
                cache_phases[call["metadata"]["phase"]] += 1
                continue
            phases[call["metadata"]["phase"]] += 1
            usage = call["output"].get("usage") if isinstance(call["output"], dict) else None
            reported = {}
            if isinstance(usage, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
                    value = usage.get(key)
                    if type(value) in (int, float) and math.isfinite(value):
                        reported[key] = value
            if not reported:
                missing += 1
            tokens.update(reported)
        return {"recorded_calls": sum(phases.values()), "recorded_calls_by_phase": dict(phases),
                "cache_hits": sum(cache_phases.values()), "cache_hits_by_phase": dict(cache_phases),
                "reported_tokens": dict(tokens), "missing_usage_calls": missing, "cost_usd": None}

    result = totals(list(records.values()))
    result.update(
        current_attempt=totals([record for key, record in records.items() if key not in inherited]),
        inherited=totals([records[key] for key in inherited]),
        pricing_assumptions="Provider rates unverified; unknown is not zero.",
        coverage="Cumulative distinct physical calls across all stages and recovery attempts; "
                 "cache replays counted separately with no incurred provider usage; no billed preflight.",
        retry_coverage="SDK retries disabled; explicit attempt budget recorded in resolved settings.",
    )
    write_json(output / "accounting.json", result)
    return result


def validate_runtime(runtime):
    profile = Path(runtime["profile_directory"])
    if read_json(profile / "runtime.json") != runtime:
        raise ValueError("Saved runtime profile changed; restore the original runtime")
    verify_files(profile, runtime["lock_sha256"])
    if sha256(runtime["asset_manifest"]) != runtime["asset_manifest_sha256"]:
        raise ValueError("Saved runtime asset identity changed")
    for key in ("appworld_python", "installed_source", "data_root"):
        if not Path(runtime[key]).exists():
            raise ValueError(f"Missing saved runtime asset: {key}")


def validate_parent(plan, root):
    if plan.get("recovery_version") != recovery_version(plan["setting"]) or not plan.get("recovery_ready"):
        raise ValueError("This run predates recovery support or never finished preparation; start a fresh run")
    if plan["status"] != "failed":
        raise ValueError("Recovery requires a failed run; active and completed runs cannot be recovered")
    if identity(plan, root) != plan["recovery_identity"]:
        raise ValueError("Saved configuration or provenance changed")
    if read_json(Path(root) / "runtime.json") != plan["runtime"]:
        raise ValueError("Runtime differs from the saved configuration")
    validate_runtime(plan["runtime"])
    if plan["setting"] != "direct" and not Path(plan["runtime"]["es_binary"]).is_file():
        raise ValueError("Missing saved Elasticsearch binary")
    source = read_json(Path(root) / "source_provenance.json")
    verify_files(Path(root) / "source/MetaWeft", source["metaweft_files"], complete=True)
    verify_files(Path(root) / "source/project", source["files"], complete=True)
    data = read_json(Path(root) / "data_provenance.json")
    if data["source"] != plan["data_source"]:
        raise ValueError("Saved data location differs from its provenance")
    data_root = Path(data["source"])
    verify_files(data_root, {"version.txt": data["version_sha256"], **data["shared_files"], **data["task_files"]})
    for phase in plan["phases"]:
        name = phase["dataset"] + ".txt"
        verify_files(root, {f"inputs/datasets/{name}": phase["dataset_sha256"],
                            f"data/datasets/{name}": phase["selected_dataset_sha256"]})
        actual = (Path(root) / "data/datasets" / name).read_text().splitlines()
        if actual != phase["ordered_task_ids"]:
            raise ValueError("Captured dataset task order changed")
    return [load_checkpoint(plan, phase, root) for phase in plan["phases"]]


def select_parent(output_root, requested, target):
    manifests = []
    for path in Path(output_root).glob("*/run_manifest.json"):
        try:
            manifests.append((path.parent, read_json(path)))
        except (OSError, ValueError):
            continue

    def blocked_reason(plan):
        origin = plan.get("recovery", {}).get("origin_run_id", plan["run_id"])
        for other_root, other in manifests:
            if other["run_id"] == plan["run_id"]:
                continue
            if other.get("recovery", {}).get("origin_run_id", other["run_id"]) != origin:
                continue
            if other.get("status") in ("planned", "running", "completed"):
                return f"Recovery lineage is active or completed: {other_root}"
            if other.get("recovery", {}).get("parent_run_id") == plan["run_id"] and other.get("recovery_ready"):
                return f"Run already has a recovery attempt; recover the latest failed descendant: {other_root}"
        return None

    if target == "latest":
        candidates = [(root, plan) for root, plan in manifests
                      if plan.get("selection") == requested and plan.get("status") == "failed"
                      and plan.get("recovery_version") == recovery_version(plan.get("setting"))
                      and plan.get("recovery_ready")]
        if not candidates:
            raise ValueError("No compatible failed new-format run found; start a fresh run or specify a run directory")
        eligible = [candidate for candidate in candidates if blocked_reason(candidate[1]) is None]
        if not eligible:
            raise ValueError(blocked_reason(candidates[-1][1]))
        root, plan = max(eligible, key=lambda item: (
            item[1].get("finished_at", item[1].get("created_at", "")), item[1]["run_id"]))
    else:
        root = Path(target).expanduser().resolve()
        plan = read_json(root / "run_manifest.json")
    if plan.get("selection") != requested:
        raise ValueError(f"Recovery selection conflicts with the saved run: {plan.get('selection')}")
    reason = blocked_reason(plan)
    if reason:
        raise ValueError(reason)
    return root, plan, validate_parent(plan, root)


@contextmanager
def recovery_lock(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".recovery.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def recovery_plan(parent, parent_root, checkpoints, *, root, tag, command):
    plan = deepcopy(parent)
    plan.update(run_id=f"{parent['setting_id']}-{tag}", run_tag=tag, run_directory=str(root),
                status="planned", created_at=datetime.now(timezone.utc).isoformat(),
                command=command, recovery_ready=False,
                source_project=str(root / "source/project"), source_framework=str(root / "source/MetaWeft"))
    for key in ("error_type", "error", "finished_at", "failure"):
        plan.pop(key, None)
    plan["recovery"] = {
        "parent_run_id": parent["run_id"], "parent_directory": str(parent_root),
        "origin_run_id": parent.get("recovery", {}).get("origin_run_id", parent["run_id"]),
        "generation": parent.get("recovery", {}).get("generation", 0) + 1,
    }
    for phase, checkpoint in zip(plan["phases"], checkpoints):
        completed = checkpoint["completed_task_ids"] if checkpoint else []
        for key in ("error_type", "error", "failure", "started_at", "finished_at", "current_task_id"):
            phase.pop(key, None)
        phase.update(output_directory=str(phase_output(root, phase)), completed_task_ids=list(completed),
                     reused_task_ids=list(completed), remaining_task_ids=phase["ordered_task_ids"][len(completed):],
                     status="completed" if checkpoint and checkpoint["finalized"] else "planned")
    plan.update(degraded_task_ids=[], degraded_reasons={}, evidence_gaps=[])
    return plan


def clone_run(parent_root, plan, checkpoints):
    """Copy evidence, never live databases/caches or an unfinished task into active output."""
    root = Path(plan["run_directory"])
    parent_root = Path(parent_root)
    for name in ("MetaWeft", "project"):
        shutil.copytree(parent_root / "source" / name, root / "source" / name, symlinks=True)
    imports = root / "source/imports"
    imports.mkdir()
    (imports / "appworld_experiments").symlink_to("../project/experiments", target_is_directory=True)
    for name in ("variant.json", "source_provenance.json", "data_provenance.json", "runtime.json",
                 "appworld-requirements.txt"):
        shutil.copy2(parent_root / name, root / name)
    shutil.copytree(parent_root / "inputs", root / "inputs")
    shutil.copytree(parent_root / "data/datasets", root / "data/datasets")
    for item in Path(plan["data_source"]).iterdir():
        if item.name != "datasets":
            (root / "data" / item.name).symlink_to(item, target_is_directory=item.is_dir())
    if (parent_root / "recovery_history").is_dir():
        shutil.copytree(parent_root / "recovery_history", root / "recovery_history")
    history = root / "recovery_history" / plan["recovery"]["parent_run_id"]
    history.mkdir(parents=True)
    for name in ("console.log", "run_manifest.json", "resolved_settings.json"):
        if (parent_root / name).is_file():
            shutil.copy2(parent_root / name, history / name)
    for phase, checkpoint in zip(plan["phases"], checkpoints):
        source = phase_output(parent_root, phase)
        output = Path(phase["output_directory"])
        output.mkdir(parents=True)
        artifacts = output / "baseline_artifacts/metaweft"
        artifacts.mkdir(parents=True)
        if (source / "recovery_history").is_dir():
            shutil.copytree(source / "recovery_history", output / "recovery_history")
        archive = output / "recovery_history" / plan["recovery"]["parent_run_id"]
        archive.mkdir(parents=True)
        completed = set(phase["completed_task_ids"])
        for branch in ("tasks", "baseline_artifacts/metaweft"):
            for item in (source / branch).glob("*"):
                if item.is_dir() and (item.name in phase["ordered_task_ids"] or branch == "tasks"):
                    destination = (output if item.name in completed else archive) / branch / item.name
                    shutil.copytree(item, destination)
        if checkpoint:
            # Initial memory and mapping are immutable across all attempts.
            for name in checkpoint["files"]:
                destination = relative_file(output, name)
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(relative_file(source, name), destination)
            shutil.copy2(source / "checkpoint.json", output / "checkpoint.json")
        for name in ("run_manifest.json", "failure.json", "accounting.json", "runner_config.json",
                     "baseline_artifacts/metaweft/memory_final.json",
                     "baseline_artifacts/metaweft/adaptation_memory_final.json",
                     "baseline_artifacts/metaweft/index.json"):
            if (source / name).is_file():
                destination = archive / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, destination)
        phase["degraded_task_ids"], phase["degraded_reasons"] = [], {}
        phase["evidence_gaps"] = []
        if plan["setting"] == "metaweft_v1":
            phase["memory_branch_results"] = {}
        for task_id in phase["completed_task_ids"]:
            result = read_json(artifacts / task_id / "result.json")
            if result["degraded"]:
                phase["degraded_task_ids"].append(task_id)
                phase["degraded_reasons"][task_id] = result["degraded"]
            if plan["setting"] == "metaweft_v1":
                phase["memory_branch_results"][task_id] = result["stages"]["update"]
        if plan["setting"] == "metaweft_v1" and checkpoint:
            from MetaWeft.src.ace_adapt_reme_task.adaptation_memory import render_playbook
            snapshot = read_json(relative_file(output, checkpoint["memories"]["adaptation"]["snapshot"]))
            playbook = restore_adaptation(snapshot)
            phase["adaptation_memory"] = {"entries": len(playbook["entries"]), "revision": playbook["revision"],
                                          "rendered_bytes": len(render_playbook(playbook).encode("utf-8"))}
        if phase["status"] == "completed" and (source / "run_manifest.json").is_file():
            saved = read_json(source / "run_manifest.json")
            phase["evidence_gaps"] = saved.get("evidence_gaps", [])
            phase["final_memory_status"] = saved.get("final_memory_status")
            if plan["setting"] == "metaweft_v1":
                phase["final_adaptation_memory_status"] = saved.get("final_adaptation_memory_status")
            for name in ("memory_final.json", "adaptation_memory_final.json"):
                if (source / "baseline_artifacts/metaweft" / name).is_file():
                    shutil.copy2(source / "baseline_artifacts/metaweft" / name, artifacts / name)
        write_json(output / "run_manifest.json", {
            **phase, "baseline": "metaweft", "setting": plan["setting"], "run_id": plan["run_id"],
            "recovery": plan["recovery"], "dataset_version": plan["dataset_version"],
        })
        if checkpoint:
            checkpoint = deepcopy(checkpoint)
            checkpoint["files"].update(file_hashes(output, [output / "recovery_history"]))
            write_json(output / "checkpoint.json", checkpoint)
        accounting(output)
    if identity(plan, root) != plan["recovery_identity"]:
        raise ValueError("Recovery changed the saved execution identity")
    seal_run(plan)
