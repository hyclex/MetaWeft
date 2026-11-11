"""V2 policy differences over the matched dual-memory V1 settings."""

import hashlib
from pathlib import Path

from MetaWeft.src.ace_adapt_reme_task.config import TASK_SCHEMA, v1_settings


ADAPTATION_SCHEMA = "metaweft-adaptation-memory-v2"
PROMPTS = Path(__file__).parent / "prompts/v1"


def v2_settings():
    settings = v1_settings()
    settings["components"].update(
        adaptation_memory=ADAPTATION_SCHEMA,
        adaptation_preparation="full-three-headings-content-only-v2",
        adaptation_selector="successful-completed-adapter-event-v2",
        adaptation_updater="ace-adapter-input-output-deltas-v2",
        adapter="single-fixed-content-only-v2")
    settings.update(
        playbook_renderer="three-headings-no-identifiers-v2",
        evidence_selection="task-current-episode-both-outcomes; adaptation-reward-one-completed-call-only",
        adaptation_success_reward=1,
        adaptation_headings=["memory_selection", "task_specific_pruning", "exploration_guidance"],
        adapter_memory_view="content-only; stored records and provenance retained",
        adaptation_update_view="frozen-adapter-input-output; revision-and-edit-targets-only-metadata")
    settings.pop("v1_prompt_manifest")
    settings["v2_prompt_manifest"] = {
        str(path.relative_to(PROMPTS)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(PROMPTS.rglob("*")) if path.is_file()}
    return settings
