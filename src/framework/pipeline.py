"""Memory-stage ordering without benchmark, evaluator or checkpoint policy."""

from copy import deepcopy

from .contracts import PreparedEpisode, PairUpdateResult, BranchUpdateResult, StateRef


def validate_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {"task", "adaptation"}:
        raise ValueError("A memory pair requires task and adaptation state references")
    for role, ref in pair.items():
        if (not isinstance(ref, StateRef) or ref.role != role
                or type(ref.epoch) is not int or ref.epoch < 0
                or any(not isinstance(value, str) or not value for value in
                       (ref.schema, ref.stream_id, ref.snapshot, ref.sha256))):
            raise ValueError("Invalid memory-role state reference")
    if pair["task"].stream_id != pair["adaptation"].stream_id or pair["task"].epoch != pair["adaptation"].epoch:
        raise ValueError("Memory pair must share a committed stream and epoch")


class DualMemoryPipeline:
    def __init__(self, task_channel, adaptation_channel, adapter):
        self.task_channel = task_channel
        self.adaptation_channel = adaptation_channel
        self.adapter = adapter

    def prepare(self, task_spec, prior_pair, *, services, artifacts):
        validate_pair(prior_pair)
        task = self.task_channel.prepare(deepcopy(task_spec), prior_pair["task"],
                                         services=services, artifacts=artifacts)
        adaptation = self.adaptation_channel.prepare(deepcopy(task_spec), prior_pair["adaptation"],
                                                      services=services, artifacts=artifacts)
        for role, prepared in (("task", task), ("adaptation", adaptation)):
            if prepared.role != role or prepared.prior != prior_pair[role]:
                raise ValueError("Prepared memory changed its role or prior state reference")
        event, reference = self.adapter.adapt(deepcopy(task_spec), deepcopy(task), deepcopy(adaptation),
                                              services=services, artifacts=artifacts)
        degraded = any(prepared.status in {"degraded", "completed_with_fallback"}
                       for prepared in (task, adaptation)) or event["status"] == "degraded"
        errors = [value for value in (task.error, adaptation.error, event.get("error")) if value]
        return PreparedEpisode(task_spec, {key: ref.to_dict() for key, ref in prior_pair.items()},
                               task, adaptation, event["guide"], event, reference,
                               "degraded" if degraded else "completed", degraded, "; ".join(errors) or None)

    def update(self, episode, prior_pair, *, services, artifacts, record_failure):
        episode.verify()
        validate_pair(prior_pair)
        if episode.payload.get("prior_pair") != {role: ref.to_dict() for role, ref in prior_pair.items()}:
            raise ValueError("Update evidence must identify the prepared committed pair")
        task = BranchUpdateResult("task", status="not_run")
        adaptation = BranchUpdateResult("adaptation", status="not_run")
        active = "task"
        try:
            selected = self.task_channel.select_evidence(deepcopy(episode))
            task = self.task_channel.update(selected, prior_pair["task"], services=services, artifacts=artifacts)
            if task.role != "task":
                raise ValueError("Task updater returned another memory role")
            episode.verify()
            active = "adaptation"
            selected = self.adaptation_channel.select_evidence(deepcopy(episode))
            adaptation = self.adaptation_channel.update(selected, prior_pair["adaptation"],
                                                         services=services, artifacts=artifacts)
            if adaptation.role != "adaptation":
                raise ValueError("Adaptation updater returned another memory role")
            episode.verify()
        except Exception as error:
            # Recording does not convert fatal infrastructure/programming errors to fallback.
            record_failure(PairUpdateResult(task, adaptation), active, error)
            raise
        return PairUpdateResult(task, adaptation)
