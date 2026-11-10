"""Isolated fresh and recovered launches using captured AppWorld sources."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from threading import Lock
import traceback
import uuid


ENTRY = Path(__file__).resolve().parent
FRAMEWORK = ENTRY.parents[1]
WORKSPACE = FRAMEWORK.parent
REPO = WORKSPACE / "metaskill-appworld"
sys.path.insert(0, str(REPO / "run"))
sys.path.insert(0, str(WORKSPACE))
from runtime_paths import PROVIDER_FILE, isolated_environment, load_runtime
from reproduce import DATASETS, find_data, sha256, snapshot_source, task_ids
from MetaWeft.experiment.recovery import (
    RECOVERY_VERSION, add_recovery_argument, clone_run, recovery_lock, recovery_plan,
    seal_run, select_parent, selection, write_json,
)
from MetaWeft.service.llm.config import cache_settings, retry_defaults
from MetaWeft.src.memory_retrieval import load_preset


SETTINGS = {
    "chat_model": "qwen3.8-flash", "embedding_model": "text-embedding-3-large",
    "embedding_dimensions": 1024, "max_interactions": 40,
    "execution_temperature": 0.9, "memory_temperature": 0.0,
    "seed": 0, "appworld_seed": 100, "enable_thinking": False, "num_trials": 1,
    "retrieval_k": 5, "retrieval_candidates": 10, "reflection_candidates": 3,
    "chat_timeout": 120, "embedding_timeout": 30, "structured_output_repairs": 0,
}
RETRIEVAL_PRESETS = (
    "base", "performance_1", "performance_2", "performance_3",
    "efficiency_1", "efficiency_2", "efficiency_3",
    "transferability_1", "transferability_2", "transferability_3", "exact_base",
)


def parse_args(setting, argv=None):
    parser = argparse.ArgumentParser(description=f"MetaWeft {setting} v1")
    if setting == "retrieval":
        parser.add_argument("--preset", choices=RETRIEVAL_PRESETS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=("test-normal", "test-challenge", "test-all"), default="test-all")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--limit", type=int)
    subset.add_argument("--task-id")
    parser.add_argument("--env-file", type=Path, default=PROVIDER_FILE)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-mode", choices=("no_cache", "cache", "timed_cache"))
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true")
    add_recovery_argument(parser)
    args = parser.parse_args(argv)
    if args.task_id and args.split == "test-all":
        parser.error("--task-id requires a single test split")
    return args


def retrieval_note(settings):
    meta = settings.get("retrieval_meta")
    if not meta or meta["schema_version"] != 2:
        return ""
    search, fusion = meta["search"], meta["fusion"]
    parts = [f"{meta['name']}: {search['backend'].upper()} depth={search['depth']}"]
    if search["num_candidates"] is not None:
        parts.append(f"ANN candidates={search['num_candidates']}")
    if search["max_bank_size"] is not None:
        parts.append(f"bank capacity={search['max_bank_size']}")
    parts.append(f"RRF c={fusion['rank_constant']}, fused limit={fusion['candidate_limit']}")
    for stage in meta["reranking"]["stages"]:
        thinking = "thinking" if stage["generation"]["enable_thinking"] else "no thinking"
        parts.append(f"{stage['model']} {stage['method']} -> {stage['output_limit']} ({thinking})")
    if not meta["reranking"]["stages"]:
        parts.append("no reranking")
    parts.append(f"top {settings['retrieval_k']} to adaptation")
    return "; ".join(parts)


def run_location(setting, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{stamp}_{args.split}_{uuid.uuid4().hex[:8]}"
    setting_id = f"metaweft-{setting}-{args.dataset}"
    root = WORKSPACE / "raw_logs" / setting_id / "experiment/output" / tag
    return root, tag, setting_id


def make_plan(setting, args):
    root, tag, setting_id = run_location(setting, args)
    phases = []
    splits = ("test-normal", "test-challenge") if args.split == "test-all" else (args.split,)
    for split in splits:
        dataset = DATASETS[args.dataset][split]
        selected = task_ids(dataset, args.limit, args.task_id)
        original = task_ids(dataset)
        phases.append({
            "split": split, "dataset": dataset, "task_ids": selected,
            "ordered_task_ids": selected, "source_task_count": len(original),
            "partial": selected != original, "experiment_name": split,
            "output_directory": str(root / "experiments/outputs" / split),
            "adapt": setting != "direct", "status": "planned",
        })
    settings = {**SETTINGS, **retry_defaults(),
                **cache_settings(args.cache_mode, args.cache_path, workspace=WORKSPACE)}
    if setting == "base":
        settings["retrieval_meta"] = load_preset("base", num_candidates=settings["retrieval_candidates"])
    elif setting == "metaweft_v1":
        from MetaWeft.src.ace_adapt_reme_task.config import v1_settings
        settings.update(v1_settings())
    elif setting != "direct":
        settings["retrieval_meta"] = load_preset(setting)
        settings["retrieval_candidates"] = settings["retrieval_meta"]["search"]["num_candidates"]
    return {
        "baseline": "metaweft", "setting": setting, "setting_id": setting_id,
        "variant_id": ("metaweft-ace-adapt-reme-task-v1" if setting == "metaweft_v1"
                       else f"metaweft-{setting}-v1"), "run_id": f"{setting_id}-{tag}",
        "run_tag": tag, "run_directory": str(root), "dataset_version": args.dataset,
        "scenario": "direct" if setting == "direct" else "cold-start", "split": args.split,
        "note": "; ".join(filter(None, [retrieval_note(settings), args.note])),
        "status": "planned", "phases": phases, "phase_execution": "parallel",
        "partial": any(p["partial"] for p in phases), "repetitions": 1,
        "task_order": "manifest order", "memory_reset": "empty independently for each split",
        "settings": settings, "command": [sys.executable, *sys.argv],
        "recovery_version": 2 if setting == "metaweft_v1" else RECOVERY_VERSION,
        "recovery_ready": False,
        "selection": selection(setting, args), "created_at": datetime.now(timezone.utc).isoformat(),
        "degraded_task_ids": [], "degraded_reasons": {}, "evidence_gaps": [],
    }


def prepare(plan, args, runtime):
    root = Path(plan["run_directory"])
    from dotenv import load_dotenv
    if args.env_file.is_file():
        load_dotenv(args.env_file, override=False)
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        raise ValueError("Provide OPENAI_API_KEY and OPENAI_BASE_URL, or --env-file")
    data = find_data(args, runtime)
    if (data / "version.txt").read_bytes() != (REPO / "data/version.txt").read_bytes():
        raise ValueError("AppWorld data version differs from the current checkout")
    project, _ = snapshot_source(root, "metaweft", runtime)
    # Preserve the framework's working source; workers import this copy only.
    framework = root / "source/MetaWeft"
    shutil.copytree(FRAMEWORK, framework, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    provenance = json.loads((root / "source_provenance.json").read_text())
    provenance["metaweft_files"] = {str(p.relative_to(framework)): sha256(p)
                                   for p in sorted(framework.rglob("*")) if p.is_file()}
    provenance["metaweft_source"] = "source/MetaWeft: copied current working files"
    write_json(root / "source_provenance.json", provenance)
    if plan["setting"] in ("base", "direct", "metaweft_v1"):
        shutil.copy2(ENTRY / f"{plan['setting']}.variant.json", root / "variant.json")
    else:
        descriptor = json.loads((framework / "experiment/appworld/retrieval.variant.json").read_text())
        preset = framework / "src/memory_retrieval/presets" / f"{plan['setting']}.variant.json"
        descriptor.update(variant_id=plan["variant_id"], setting=plan["setting"],
                          retrieval_meta=plan["settings"]["retrieval_meta"],
                          retrieval_preset_provenance=json.loads(preset.read_text()))
        write_json(root / "variant.json", descriptor)
    (root / "data/datasets").mkdir(parents=True)
    (root / "inputs/datasets").mkdir(parents=True)
    for item in data.iterdir():
        if item.name != "datasets":
            (root / "data" / item.name).symlink_to(item, target_is_directory=item.is_dir())
    for phase in plan["phases"]:
        name = phase["dataset"] + ".txt"
        shutil.copy2(REPO / "data/datasets" / name, root / "inputs/datasets" / name)
        (root / "data/datasets" / name).write_text("\n".join(phase["task_ids"]) + "\n")
        phase["dataset_sha256"] = sha256(root / "inputs/datasets" / name)
        phase["selected_dataset_sha256"] = sha256(root / "data/datasets" / name)
    selected = dict.fromkeys(t for p in plan["phases"] for t in p["task_ids"])
    for task in selected:
        if not (data / "tasks" / task).is_dir():
            raise FileNotFoundError(f"Missing task {task}")
    write_json(root / "data_provenance.json", {
        "source": str(data), "version_sha256": sha256(data / "version.txt"),
        "inputs_shared_read_only": ["tasks", "base_dbs", "api_docs"],
        "shared_files": {str(p.relative_to(data)): sha256(p) for name in ("base_dbs", "api_docs")
                         for p in sorted((data / name).rglob("*")) if p.is_file()},
        "task_files": {str(p.relative_to(data)): sha256(p) for task in selected
                       for p in sorted((data / "tasks" / task).rglob("*")) if p.is_file()},
    })
    plan.update(source_project=str(project), source_framework=str(framework), runtime=runtime,
                python=sys.executable, data_source=str(data),
                provider={"chat_base_url": os.environ["OPENAI_BASE_URL"],
                          "env_file": str(args.env_file.resolve())})
    if args.env_file.is_file():
        plan["provider"]["env_file_sha256"] = sha256(args.env_file)
    if plan["setting"] != "direct":
        plan.update(es_binary=runtime["es_binary"], es_url=None)
    worker_environment(plan)
    write_json(root / "runtime.json", runtime)
    requirements = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                                 capture_output=True, text=True, cwd=root)
    (root / "appworld-requirements.txt").write_text(requirements.stdout)
    seal_run(plan)
    write_json(root / "resolved_settings.json", plan)


def worker_environment(plan):
    root = Path(plan["run_directory"])
    project = root / "source/project"
    os.environ.update(APPWORLD_ROOT=str(root), APPWORLD_PROJECT_PATH=str(project),
                      APPWORLD_LM_LOGGER_MODULE_PATH=str(project / "experiments/code/common/lm_logger.py"),
                      PYTHONPATH=os.pathsep.join([str(root / "source"), str(project / "src"),
                                                str(root / "source/imports"), str(project / "run")]))


def check_recovery_options(parent, args):
    if args.cache_mode is not None and args.cache_mode != parent["settings"].get("cache_mode"):
        raise ValueError("--cache-mode conflicts with the saved run")
    if args.cache_path is not None:
        cache_path = cache_settings(cache_path=args.cache_path, workspace=WORKSPACE)["cache_path"]
        if cache_path != parent["settings"].get("cache_path"):
            raise ValueError("--cache-path conflicts with the saved run")
    if args.data_root and args.data_root.resolve() != Path(parent["data_source"]).resolve():
        raise ValueError("--data-root conflicts with the saved run")
    runtime = parent["runtime"]
    for variable, key in {"APPWORLD_RUNTIME_PROFILE": "profile_directory", "APPWORLD_PYTHON": "appworld_python",
                          "APPWORLD_INSTALLED_SOURCE": "installed_source", "REME_ES_BINARY": "es_binary"}.items():
        if variable == "REME_ES_BINARY" and parent["setting"] == "direct":
            continue
        if os.environ.get(variable) and os.path.abspath(os.environ[variable]) != runtime[key]:
            raise ValueError(f"{variable} conflicts with the saved runtime")
    url = os.environ.get("OPENAI_BASE_URL")
    if not url and args.env_file.is_file():
        result = subprocess.run(
            [runtime["appworld_python"], "-B", "-c",
             "from dotenv import dotenv_values; import sys; print(dotenv_values(sys.argv[1]).get('OPENAI_BASE_URL') or '')",
             str(args.env_file)], check=True, capture_output=True, text=True,
        )
        url = result.stdout.strip()
    if url and url.rstrip("/") != parent["provider"]["chat_base_url"].rstrip("/"):
        raise ValueError("Provider endpoint conflicts with the saved run; restore the original endpoint")


def inspect_recovery(setting, args):
    root, tag, _ = run_location(setting, args)
    parent_root, parent, checkpoints = select_parent(root.parent, selection(setting, args), args.recover)
    check_recovery_options(parent, args)
    plan = recovery_plan(parent, parent_root, checkpoints, root=root, tag=tag,
                         command=[sys.executable, *sys.argv])
    plan["phase_execution"] = "parallel"
    if args.note:
        plan["note"] = "; ".join(filter(None, [retrieval_note(parent["settings"]), args.note]))
    return plan, parent_root, checkpoints


def prepare_recovery(plan, parent_root, checkpoints, args):
    from dotenv import load_dotenv
    if args.env_file.is_file():
        load_dotenv(args.env_file, override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Provide OPENAI_API_KEY or --env-file for recovery")
    # Check the installed environment as well as the preserved lock records.
    requirements = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                                 capture_output=True, text=True)
    if sorted(requirements.stdout.splitlines()) != sorted((parent_root / "appworld-requirements.txt").read_text().splitlines()):
        raise ValueError("Installed dependencies differ from the saved run")
    clone_run(parent_root, plan, checkpoints)
    plan["provider"]["env_file"] = str(args.env_file.resolve())
    plan["provider"].pop("env_file_sha256", None)
    if args.env_file.is_file():
        plan["provider"]["env_file_sha256"] = sha256(args.env_file)
    worker_environment(plan)
    write_json(Path(plan["run_directory"]) / "resolved_settings.json", plan)


def run_phases(plan):
    root = Path(plan["run_directory"])
    pending = [i for i, phase in enumerate(plan["phases"]) if phase["status"] != "completed"]
    if not pending:
        return
    children, futures = {}, {}
    log_lock = Lock()
    with (root / "console.log").open("a") as log, ThreadPoolExecutor(max_workers=len(pending)) as pool:
        def forward_output(child):
            try:
                for line in child.stdout:
                    with log_lock:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                return child.wait()
            finally:
                child.stdout.close()

        try:
            environments = {}
            for index in pending:
                phase = plan["phases"][index]
                environment = isolated_environment(root / "runtime" / phase["split"])
                environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
                environments[index] = environment
                phase["writable_paths"] = {k: v for k, v in environment.items()
                                           if k.endswith("CACHE_DIR") or k in ("APPWORLD_CACHE", "TMPDIR", "XDG_CACHE_HOME")}
                phase["status"] = "running"
            write_json(root / "run_manifest.json", plan)
            for index in pending:
                command = [sys.executable, "-B", "-m", "MetaWeft.experiment.appworld.worker",
                           str(root / "run_manifest.json"), str(index)]
                child = subprocess.Popen(command, cwd=root, env=environments[index], stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True)
                children[index] = child
                futures[pool.submit(forward_output, child)] = index
            failed = []
            for future in as_completed(futures):
                if future.result():
                    failed.append(plan["phases"][futures[future]]["split"])
            if failed:
                raise RuntimeError(f"Phases failed: {', '.join(failed)}; see {root / 'console.log'}")
        finally:
            for child in children.values():
                if child.poll() is None:
                    child.terminate()
            for child in children.values():
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            for index in pending:
                phase = plan["phases"][index]
                saved = Path(phase["output_directory"]) / "run_manifest.json"
                if saved.is_file():
                    result = json.loads(saved.read_text())
                    for key in ("status", "completed_task_ids", "degraded_task_ids", "degraded_reasons", "evidence_gaps",
                                "memory_branch_results", "adaptation_memory",
                                "started_at", "finished_at", "failure", "current_task_id"):
                        if key in result:
                            phase[key] = result[key]
                else:
                    phase["status"] = "failed"
            write_json(root / "run_manifest.json", plan)


def main(setting, argv=None):
    args = parse_args(setting, argv)
    if setting == "retrieval":
        setting = args.preset
    if args.dry_run:
        plan = inspect_recovery(setting, args)[0] if args.recover else make_plan(setting, args)
        print(json.dumps(plan, indent=2))
        return
    if args.recover:
        with recovery_lock(run_location(setting, args)[0].parent):
            plan, parent_root, checkpoints = inspect_recovery(setting, args)
            runtime = plan["runtime"]
            if os.path.abspath(sys.executable) != runtime["appworld_python"]:
                os.execv(runtime["appworld_python"], [runtime["appworld_python"], "-B", *sys.argv])
            root = Path(plan["run_directory"])
            root.mkdir(parents=True, exist_ok=False)
            write_json(root / "run_manifest.json", plan)
    else:
        plan = make_plan(setting, args)
        assets = ["appworld_python", "installed_source", "data_root"]
        if setting != "direct":
            assets.append("es_binary")
        runtime = load_runtime(required_assets=assets)
        if os.path.abspath(sys.executable) != runtime["appworld_python"]:
            os.execv(runtime["appworld_python"], [runtime["appworld_python"], "-B", *sys.argv])
        Path(plan["run_directory"]).mkdir(parents=True, exist_ok=False)
    root = Path(plan["run_directory"])
    write_json(root / "run_manifest.json", plan)
    print(f"Run: {root}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        environment = isolated_environment(root)
        os.environ.clear()
        os.environ.update(environment)
        plan["writable_paths"] = {k: v for k, v in environment.items()
                                  if k.endswith("CACHE_DIR") or k in ("APPWORLD_CACHE", "TMPDIR", "XDG_CACHE_HOME")}
        if args.recover:
            prepare_recovery(plan, parent_root, checkpoints, args)
        else:
            prepare(plan, args, runtime)
        plan["status"] = "running"
        run_phases(plan)
        plan["status"] = "completed"
    except BaseException as error:
        plan.update(status="failed", error_type=type(error).__name__)
        with (root / "console.log").open("a") as log:
            traceback.print_exc(file=log)
        raise
    finally:
        plan["finished_at"] = datetime.now(timezone.utc).isoformat()
        plan["degraded_task_ids"] = [task for phase in plan["phases"] for task in phase.get("degraded_task_ids", [])]
        plan["degraded_reasons"] = {task: reasons for phase in plan["phases"]
                                    for task, reasons in phase.get("degraded_reasons", {}).items()}
        plan["evidence_gaps"] = [gap for phase in plan["phases"] for gap in phase.get("evidence_gaps", [])]
        if setting == "metaweft_v1":
            plan["memory_branch_results"] = {task: result for phase in plan["phases"]
                                               for task, result in phase.get("memory_branch_results", {}).items()}
            plan["adaptation_memory"] = {phase["split"]: phase["adaptation_memory"]
                                          for phase in plan["phases"] if "adaptation_memory" in phase}
        write_json(root / "run_manifest.json", plan)
