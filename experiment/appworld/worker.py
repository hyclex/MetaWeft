"""Execute one split from captured sources, with canonical AppWorld artifacts."""

from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import uuid

from appworld.apps.model_lib import CachedDBHandler
from appworld_experiments.code.common.appworld_runner import run_dataset
from appworld_experiments.code.common.contracts import BaselineTaskResult
from appworld_experiments.code.common.lm_logger import (
    EmbeddingSerializer, LLMCallContext, OpenAISerializer, redact_secrets, task_lm_logger,
)
from appworld_experiments.code.common.output_contract import validate_dataset_output, validate_task_output
from appworld_experiments.code.common.task_loader import load_task_stream

from MetaWeft.experiment.appworld.executor import execute
from MetaWeft.experiment.recovery import (
    accounting, commit_checkpoint, load_checkpoint, read_json, relative_file, restore_memory, write_json,
)
from MetaWeft.service.llm import LLMCacheError, LLMUnavailableError, ModelClient
from MetaWeft.service.llm.cache import restore_pending_responses


class SafeEmbeddingSerializer(EmbeddingSerializer):
    def response(self, response):
        raw = OpenAISerializer().response(response)
        try:
            return super().response(raw)
        except (AttributeError, KeyError, TypeError, ValueError):
            # Malformed vectors still have billable usage. Preserve it without
            # leaking full vector payloads or turning validation into log failure.
            return {"invalid_embedding_response": True,
                    "model": raw.get("model") if isinstance(raw, dict) else None,
                    "usage": raw.get("usage") if isinstance(raw, dict) else None}


def now():
    return datetime.now(timezone.utc).isoformat()


def physical_recorder(logger, call_refs=None):
    def record(operation, request, invoke, *, phase, component, logical_call_id, attempt, step, cache=None):
        physical_call_id = uuid.uuid4().hex
        if call_refs is not None:
            call_refs.setdefault(component, []).append({
                "logical_call_id": logical_call_id, "physical_call_id": physical_call_id, "attempt": attempt,
                **({"cache": cache} if cache is not None else {}),
            })
        return logger.invoke(
            provider_call=invoke, request=request,
            context=LLMCallContext(phase=phase, component=component, logical_call_id=logical_call_id,
                                   physical_call_id=physical_call_id, attempt=attempt, step=step,
                                   episode_kind="primary", metadata={"cache": cache} if cache is not None else {}),
            serializer=SafeEmbeddingSerializer() if operation in ("embed", "embedding", "embeddings") else OpenAISerializer(),
        )
    return record


def snapshot_memory(store, path):
    # File failures are fatal. A storage failure is explicit missing evidence.
    try:
        records = store.export()
    except Exception as error:
        result = {"status": "unavailable", "error_type": type(error).__name__, "error": str(error)}
    else:
        result = {"status": "verified", "records": sorted(records, key=lambda r: r["id"])}
    write_json(path, result)
    return result


class MetaWeftAdapter:
    name = "metaweft"

    def __init__(self, plan, phase, manifest, pipeline=None, store=None):
        self.plan, self.phase, self.manifest = plan, phase, manifest
        self.pipeline, self.store = pipeline, store
        self.output = Path(phase["output_directory"])
        self.prompts = Path(plan["source_framework"]) / "src/base/prompts/v1"
        self.checkpoint = read_json(self.output / "checkpoint.json")

    def retrieval_trace_context(self):
        from MetaWeft.src.memory_retrieval import prompt_manifest
        root = Path(self.plan["run_directory"])
        bank = self.output / "baseline_artifacts/metaweft"
        settings = self.plan["settings"]
        memory_file = self.checkpoint["memory_file"]
        return {
            "bank": {
                "index": self.store.index,
                "mapping": str((bank / "memory_mapping.json").relative_to(root)),
                "memory_checkpoint": str((self.output / memory_file).relative_to(root)),
                "memory_sha256": self.checkpoint["files"][memory_file],
                "checkpoint": str((self.output / "checkpoint.json").relative_to(root)),
                "representations": {"combined": {
                    "field": "embedding", "input": 'when_to_use + "\\n\\n" + content',
                }},
                "embedding_spec": {"provider_base_url": self.plan["provider"]["chat_base_url"],
                                   "model": settings["embedding_model"],
                                   "dimensions": settings["embedding_dimensions"]},
            },
            "model_settings": {**settings, "provider_base_url": self.plan["provider"]["chat_base_url"],
                               "retrieval_provider_policy": self.pipeline.retrieval_meta.get(
                                   "provider_policy", {key: settings[key] for key in
                                                       ("max_retries", "retry_delay_seconds")})},
            "prompts": prompt_manifest(self.pipeline.retrieval_meta),
        }

    def run_task(self, context, services):
        from loguru import logger
        artifacts = context.artifacts_dir
        if context.task_id in self.manifest["completed_task_ids"]:
            return BaselineTaskResult(metadata=read_json(artifacts / "result.json"))
        self.manifest["current_task_id"] = context.task_id
        write_json(self.output / "run_manifest.json", self.manifest)
        spec = {
            "task_id": context.task_id, "instruction": context.task.instruction,
            "supervisor": {key: getattr(context.task.supervisor, key)
                           for key in ("first_name", "last_name", "email", "phone_number")},
            "app_descriptions": dict(context.task.app_descriptions),
        }
        write_json(artifacts / "task_spec.json", spec)
        call_refs = {}
        model_options = dict(api_key=os.environ["OPENAI_API_KEY"], base_url=self.plan["provider"]["chat_base_url"],
                            chat_model=self.plan["settings"]["chat_model"],
                            embedding_model=self.plan["settings"]["embedding_model"],
                            dimensions=self.plan["settings"]["embedding_dimensions"],
                            max_retries=self.plan["settings"]["max_retries"],
                            retry_delay_seconds=self.plan["settings"]["retry_delay_seconds"],
                            chat_timeout=self.plan["settings"]["chat_timeout"],
                            embedding_timeout=self.plan["settings"]["embedding_timeout"],
                            cache_mode=self.plan["settings"]["cache_mode"],
                            cache_path=self.plan["settings"]["cache_path"],
                            cache_recovery_dir=artifacts / "llm_cache_pending",
                            recorder=physical_recorder(services.llm, call_refs))
        model = ModelClient(**model_options)
        clients = ExitStack()
        clients.callback(model.close)
        stages = {}
        degraded = []
        sink = None
        try:
            guidance = ""
            if self.pipeline:
                retrieval_model = model
                if self.pipeline.retrieval_meta["schema_version"] == 2:
                    retrieval_model = ModelClient(**{**model_options,
                                                     **self.pipeline.retrieval_meta["provider_policy"]})
                    clients.callback(retrieval_model.close)
                prepared = self.pipeline.prepare_context(spec, llm=model, artifacts=artifacts,
                                                         trace_context=self.retrieval_trace_context(),
                                                         call_refs=call_refs, retrieval_llm=retrieval_model)
                guidance = prepared.guidance
                stages["context"] = prepared.to_dict()
                if prepared.degraded:
                    degraded.append({"stage": "context", "reason": prepared.error})
            else:
                stages["context"] = {"status": "disabled", "degraded": False, "guidance": ""}
            write_json(artifacts / "context.json", stages["context"])
            with services.episodes.open(kind="primary"):
                sink = logger.add(str(context.primary_task_dir / "logs/loggger.log"))
                logger.info("MetaWeft {} task {} started", self.plan["setting"], context.task_id)
                with (artifacts / "trace.jsonl").open("w") as trace_file:
                    def save_step(event):
                        trace_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                        trace_file.flush()
                    trace = execute(spec, guidance, llm=model, environment=services.environment,
                                    prompt=(self.prompts / ("execution_base.txt" if self.pipeline
                                                            else "execution_direct.txt")).read_text(),
                                    max_interactions=self.plan["settings"]["max_interactions"], record_step=save_step)
                completed = services.completion.task_completed()
            try:
                # Official test failures are scores, not runner exceptions.
                official = services.evaluator.evaluate_task(suppress_errors=True)
                total = official.tracker.pass_count + official.tracker.fail_count
                if total == 0:
                    raise ValueError("Official evaluator returned zero tests")
                reward = official.tracker.pass_count / total
            finally:
                CachedDBHandler.reset()
            evidence = {"task_spec": spec, "trace": trace, "reward": reward}
            write_json(artifacts / "evidence.json", evidence)
            if self.pipeline:
                updated = self.pipeline.post_task(evidence, llm=model, artifacts=artifacts)
                stages["update"] = updated.to_dict()
                if updated.degraded:
                    degraded.append({"stage": "update", "reason": updated.error})
                snapshot = snapshot_memory(self.store, artifacts / "memory_after.json")
                if snapshot["status"] != "verified":
                    raise RuntimeError("Cannot checkpoint task memory: " + snapshot["error"])
            else:
                stages["update"] = {"status": "disabled", "degraded": False}
            write_json(artifacts / "stages.json", stages)
            result = {"task_id": context.task_id, "reward": reward, "task_completed": completed,
                      "interactions": len(trace), "degraded": degraded, "stages": stages}
            write_json(artifacts / "result.json", result)
            misc = context.primary_task_dir / "misc"
            misc.mkdir(exist_ok=True)
            (misc / "cost.txt").write_text("unknown\n")
            validate_task_output(context.primary_task_dir)
            for relative in ("logs/loggger.log", "misc/cost.txt"):
                if not (context.primary_task_dir / relative).is_file():
                    raise FileNotFoundError(relative)
            if sink is not None:
                logger.remove(sink)
                sink = None
            completed_ids = [*self.manifest["completed_task_ids"], context.task_id]
            self.checkpoint = commit_checkpoint(self.plan, self.phase, completed_ids, previous=self.checkpoint)
            self.manifest["completed_task_ids"] = completed_ids
            self.manifest["current_task_id"] = None
            if degraded:
                self.manifest["degraded_task_ids"].append(context.task_id)
                self.manifest["degraded_reasons"][context.task_id] = degraded
            write_json(self.output / "run_manifest.json", self.manifest)
            print(f"Task {context.task_id}: reward={reward:.3f}, steps={len(trace)}, degraded={bool(degraded)}", flush=True)
            return BaselineTaskResult(metadata=result)
        finally:
            try:
                clients.close()
            finally:
                if sink is not None:
                    logger.remove(sink)


def main():
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    plan = json.loads(Path(sys.argv[1]).read_text())
    phase = plan["phases"][int(sys.argv[2])]
    output = Path(phase["output_directory"])
    output.mkdir(parents=True, exist_ok=bool(plan.get("recovery")))
    checkpoint = load_checkpoint(plan, phase, Path(plan["run_directory"]))
    manifest = {**phase, "baseline": "metaweft", "setting": plan["setting"], "variant_id": plan["variant_id"],
                "run_id": plan["run_id"], "scenario": plan["scenario"], "dataset_version": plan["dataset_version"],
                "note": plan["note"], "status": "running", "started_at": now(),
                "completed_task_ids": checkpoint["completed_task_ids"] if checkpoint else [],
                "degraded_task_ids": list(phase.get("degraded_task_ids", [])),
                "degraded_reasons": dict(phase.get("degraded_reasons", {})), "evidence_gaps": [],
                "recovery": plan.get("recovery")}
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "runner_config.json", plan["settings"])
    artifacts = output / "baseline_artifacts/metaweft"
    artifacts.mkdir(parents=True, exist_ok=bool(plan.get("recovery")))
    store = None
    try:
        if plan.get("recovery"):
            restore_pending_responses(
                output, cache_path=plan["settings"]["cache_path"],
                max_retries=plan["settings"]["max_retries"],
                retry_delay_seconds=plan["settings"]["retry_delay_seconds"],
            )
        actual = [ref.task_id for ref in load_task_stream(phase["dataset"])]
        if actual != phase["ordered_task_ids"]:
            raise ValueError("Official loader changed task order")
        if plan["setting"] != "direct":
            from reme_runtime import elasticsearch
            service = elasticsearch(plan, artifacts)
        else:
            service = nullcontext(None)
        with service as url:
            try:
                pipeline = None
                if url is not None:
                    from MetaWeft.service.elasticsearch import ElasticsearchStore
                    from MetaWeft.src.base import BasePipeline, memory_index_body
                    index = "metaweft-" + plan["run_tag"].lower().replace("_", "-") + "-" + phase["split"]
                    store = ElasticsearchStore(url, index)
                    mapping = read_json(artifacts / "memory_mapping.json") if checkpoint else memory_index_body(
                        plan["settings"]["embedding_dimensions"])
                    store.create(mapping)
                    write_json(artifacts / "index.json", {"index": index, "body": mapping})
                    if checkpoint:
                        if store.export():
                            raise RuntimeError("Recovery requires a fresh empty memory index")
                        restore_memory(store, read_json(relative_file(output, checkpoint["memory_file"])))
                        restored = snapshot_memory(store, artifacts / "memory_restored.json")
                        if restored["status"] != "verified":
                            raise RuntimeError("Cannot verify restored memory")
                    else:
                        write_json(artifacts / "memory_mapping.json", mapping)
                        initial = snapshot_memory(store, artifacts / "memory_initial.json")
                        if initial["status"] != "verified" or initial["records"]:
                            raise RuntimeError("Cannot verify empty initial memory")
                    pipeline = BasePipeline(store, Path(plan["run_directory"]),
                                            retrieval_k=plan["settings"]["retrieval_k"],
                                            retrieval_meta=plan["settings"].get("retrieval_meta"))
                elif not checkpoint:
                    write_json(artifacts / "memory_initial.json", {"status": "disabled", "records": []})
                if not checkpoint:
                    checkpoint = commit_checkpoint(plan, phase, [])
                adapter = MetaWeftAdapter(plan, phase, manifest, pipeline, store)
                run_dataset(adapter=adapter, dataset_name=phase["dataset"], experiment_name=phase["experiment_name"],
                            logger_factory=lambda context: task_lm_logger(output.parent, phase["experiment_name"],
                                                                         "metaweft", context.task_id),
                            appworld_kwargs={"load_ground_truth": False, "random_seed": plan["settings"]["appworld_seed"]},
                            suppress_evaluation_errors=True)
            finally:
                CachedDBHandler.reset()
                if store:
                    final = snapshot_memory(store, artifacts / "memory_final.json")
                    manifest["final_memory_status"] = final["status"]
                    if final["status"] != "verified":
                        manifest["evidence_gaps"].append({"stage": "final_memory_snapshot", "reason": final["error"]})
                    store.close()
                else:
                    write_json(artifacts / "memory_final.json", {"status": "disabled", "records": []})
            validate_dataset_output(output.parent, phase["experiment_name"], phase["dataset"])
            scores = json.loads((output / "evaluations" / f"{phase['dataset']}.json").read_text())
            if set(scores["individual"]) != set(phase["ordered_task_ids"]):
                raise RuntimeError("Official evaluation coverage differs from selected tasks")
            write_json(output / "summary.json", {
                "selected_tasks": len(actual), "source_tasks": phase["source_task_count"],
                "partial": phase["partial"], "aggregate": scores["aggregate"],
                "degraded_task_ids": manifest["degraded_task_ids"],
                "scenario_metric_scope": "selected tasks only" if phase["partial"] else "complete dataset",
            })
            commit_checkpoint(plan, phase, manifest["completed_task_ids"],
                              previous=read_json(output / "checkpoint.json"), finalized=True)
        manifest["status"] = "completed"
    except BaseException as error:
        failure = {"error_type": type(error).__name__, "error": str(error),
                   "task_id": manifest.get("current_task_id")}
        if isinstance(error, LLMUnavailableError):
            failure.update(kind="provider_unavailable", phase=error.phase, operation=error.operation,
                           logical_call_id=error.logical_call_id, attempts=error.attempts)
        elif isinstance(error, LLMCacheError):
            failure.update(kind="cache_unavailable", phase=error.phase, operation=error.operation,
                           cache_key=error.key, attempts=error.attempts,
                           pending_response_path=str(error.pending_response_path)
                           if error.pending_response_path is not None else None)
        failure = redact_secrets(failure)
        manifest.update(status="failed", error_type=type(error).__name__, failure=failure)
        write_json(output / "failure.json", failure)
        raise
    finally:
        manifest["finished_at"] = now()
        write_json(output / "run_manifest.json", manifest)
        try:
            accounting(output)
        except BaseException:
            manifest["status"] = "failed"
            write_json(output / "run_manifest.json", manifest)
            raise


if __name__ == "__main__":
    main()
