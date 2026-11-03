"""Fresh isolated launches, reusing current AppWorld provenance helpers."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback
import uuid


ENTRY = Path(__file__).resolve().parent
FRAMEWORK = ENTRY.parents[1]
WORKSPACE = FRAMEWORK.parent
REPO = WORKSPACE / "metaskill-appworld"
sys.path.insert(0, str(REPO / "run"))
from runtime_paths import PROVIDER_FILE, isolated_environment, load_runtime
from reproduce import DATASETS, find_data, sha256, snapshot_source, task_ids, write_json


SETTINGS = {
    "chat_model": "qwen3.8-flash", "embedding_model": "text-embedding-3-large",
    "embedding_dimensions": 1024, "max_interactions": 40,
    "execution_temperature": 0.9, "memory_temperature": 0.0,
    "seed": 0, "appworld_seed": 100, "enable_thinking": False, "num_trials": 1,
    "retrieval_k": 5, "retrieval_candidates": 10, "reflection_candidates": 3,
    "provider_attempts": 3, "chat_timeout": 120, "embedding_timeout": 30,
    "retry_waits": [1, 5], "structured_output_repairs": 0,
}


def parse_args(setting, argv=None):
    parser = argparse.ArgumentParser(description=f"MetaWeft {setting} v1")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=("test-normal", "test-challenge", "test-all"), default="test-all")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--limit", type=int)
    subset.add_argument("--task-id")
    parser.add_argument("--env-file", type=Path, default=PROVIDER_FILE)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.task_id and args.split == "test-all":
        parser.error("--task-id requires a single test split")
    return args


def make_plan(setting, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{stamp}_{args.split}_{uuid.uuid4().hex[:8]}"
    setting_id = f"metaweft-{setting}-{args.dataset}"
    root = WORKSPACE / "raw_logs" / setting_id / "experiment/output" / tag
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
            "adapt": setting == "base", "status": "planned",
        })
    return {
        "baseline": "metaweft", "setting": setting, "setting_id": setting_id,
        "variant_id": f"metaweft-{setting}-v1", "run_id": f"{setting_id}-{tag}",
        "run_tag": tag, "run_directory": str(root), "dataset_version": args.dataset,
        "scenario": "cold-start" if setting == "base" else "direct", "split": args.split,
        "note": args.note, "status": "planned", "phases": phases,
        "partial": any(p["partial"] for p in phases), "repetitions": 1,
        "task_order": "manifest order", "memory_reset": "empty independently for each split",
        "settings": SETTINGS, "command": [sys.executable, *sys.argv],
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
    project, imports = snapshot_source(root, "metaweft", runtime)
    # Preserve the framework's working source; workers import this copy only.
    framework = root / "source/MetaWeft"
    shutil.copytree(FRAMEWORK, framework, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    provenance = json.loads((root / "source_provenance.json").read_text())
    provenance["metaweft_files"] = {str(p.relative_to(framework)): sha256(p)
                                   for p in sorted(framework.rglob("*")) if p.is_file()}
    provenance["metaweft_source"] = "source/MetaWeft: copied current working files"
    write_json(root / "source_provenance.json", provenance)
    shutil.copy2(ENTRY / f"{plan['setting']}.variant.json", root / "variant.json")
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
    if plan["setting"] == "base":
        plan.update(es_binary=runtime["es_binary"], es_url=None)
    os.environ.update(APPWORLD_ROOT=str(root), APPWORLD_PROJECT_PATH=str(project),
                      APPWORLD_LM_LOGGER_MODULE_PATH=str(project / "experiments/code/common/lm_logger.py"),
                      PYTHONPATH=os.pathsep.join([str(root / "source"), str(project / "src"),
                                                str(imports), str(project / "run")]))
    write_json(root / "runtime.json", runtime)
    requirements = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                                 capture_output=True, text=True, cwd=root)
    (root / "appworld-requirements.txt").write_text(requirements.stdout)
    write_json(root / "resolved_settings.json", plan)


def run_phase(plan, index):
    root = Path(plan["run_directory"])
    command = [sys.executable, "-B", "-m", "MetaWeft.experiment.appworld.worker",
               str(root / "run_manifest.json"), str(index)]
    with (root / "console.log").open("a") as log:
        child = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
        try:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            if child.wait():
                raise RuntimeError(f"Phase failed; see {root / 'console.log'}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main(setting, argv=None):
    args = parse_args(setting, argv)
    plan = make_plan(setting, args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    assets = ["appworld_python", "installed_source", "data_root"]
    if setting == "base":
        assets.append("es_binary")
    runtime = load_runtime(required_assets=assets)
    if os.path.abspath(sys.executable) != runtime["appworld_python"]:
        os.execv(runtime["appworld_python"], [runtime["appworld_python"], "-B", *sys.argv])
    root = Path(plan["run_directory"])
    root.mkdir(parents=True, exist_ok=False)
    environment = isolated_environment(root)
    os.environ.clear()
    os.environ.update(environment)
    plan["writable_paths"] = {k: v for k, v in environment.items()
                              if k.endswith("CACHE_DIR") or k in ("APPWORLD_CACHE", "TMPDIR", "XDG_CACHE_HOME")}
    write_json(root / "run_manifest.json", plan)
    print(f"Run: {root}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        prepare(plan, args, runtime)
        plan["status"] = "running"
        for index, phase in enumerate(plan["phases"]):
            phase["status"] = "running"
            write_json(root / "run_manifest.json", plan)
            try:
                run_phase(plan, index)
            finally:
                saved = Path(phase["output_directory"]) / "run_manifest.json"
                if saved.is_file():
                    result = json.loads(saved.read_text())
                    for key in ("status", "completed_task_ids", "degraded_task_ids", "degraded_reasons", "evidence_gaps", "started_at", "finished_at"):
                        if key in result:
                            phase[key] = result[key]
                    plan["degraded_task_ids"].extend(result.get("degraded_task_ids", []))
                    plan["degraded_reasons"].update(result.get("degraded_reasons", {}))
                    plan["evidence_gaps"].extend(result.get("evidence_gaps", []))
                else:
                    phase["status"] = "failed"
                write_json(root / "run_manifest.json", plan)
        plan["status"] = "completed"
    except BaseException as error:
        plan.update(status="failed", error_type=type(error).__name__)
        with (root / "console.log").open("a") as log:
            traceback.print_exc(file=log)
        raise
    finally:
        write_json(root / "run_manifest.json", plan)
