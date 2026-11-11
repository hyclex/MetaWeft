"""Expose memory content to the adapter while retaining complete stored evidence."""

from pathlib import Path

from MetaWeft.src.ace_adapt_reme_task.common import FULL_PROMPT_MAX_BYTES, SchemaError, call_model, write_json
from MetaWeft.src.ace_adapt_reme_task.evidence import task_view
from .adaptation_memory import playbook_content


class Adapter:
    def __init__(self, run_root, template, *, max_bytes=FULL_PROMPT_MAX_BYTES):
        self.run_root, self.template, self.max_bytes = Path(run_root).resolve(), template, max_bytes

    def adapt(self, task_spec, task_prepared, adaptation_prepared, *, services, artifacts):
        payload = {"task_spec": task_view(task_spec),
                   "retrieved_memories": [record["content"] for record in task_prepared.payload],
                   "playbook": playbook_content(adaptation_prepared.payload["state"]),
                   "preparation_status": {"task": task_prepared.status, "adaptation": adaptation_prepared.status}}
        event = {"guide": "", "error": None,
                 "prior_pair": {"task": task_prepared.prior.to_dict(), "adaptation": adaptation_prepared.prior.to_dict()},
                 "rendered_playbook_sha256": adaptation_prepared.payload["rendered_sha256"]}
        path = Path(artifacts) / "adaptation_event.json"
        save = lambda: write_json(path, event)
        try:
            value = call_model("adaptation", self.template, payload, services=services, event=event,
                               save=save, max_bytes=self.max_bytes)
            if set(value) != {"guidance"} or not isinstance(value["guidance"], str):
                raise SchemaError("Adaptation requires exactly one string field: guidance")
            event.update(status="completed", guide=value["guidance"])
        except SchemaError as error:
            event.update(status="degraded", error=str(error), guide="")
        event["retrieved_count"] = len(task_prepared.payload)
        event["playbook_entry_count"] = len(adaptation_prepared.payload["state"]["entries"])
        event["guide_bytes"] = len(event["guide"].encode("utf-8"))
        save()
        return event, str(path.resolve().relative_to(self.run_root))
