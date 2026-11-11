"""Concrete V1 composition and recorded defaults, without a component registry."""

import hashlib
import json
from pathlib import Path

from MetaWeft.src.memory_retrieval import load_preset, prompt_manifest, validate_meta
from .common import FULL_PROMPT_MAX_BYTES


TASK_SCHEMA = "metaweft-task-memory-v1"
ADAPTATION_SCHEMA = "metaweft-adaptation-memory-v1"
PROMPTS = Path(__file__).parent / "prompts/v1"


def v1_settings():
    retrieval = load_preset("efficiency_1")
    validate_meta(retrieval, k=10)
    provenance = Path(__file__).parent.parent / "memory_retrieval/presets/efficiency_1.variant.json"
    return {
        "components": {"task_memory": TASK_SCHEMA, "adaptation_memory": ADAPTATION_SCHEMA,
                       "task_preparation": "shared-efficiency_1-k10", "adaptation_preparation": "full-playbook-v1",
                       "task_selector": "current-episode-task-v1", "adaptation_selector": "current-episode-adaptation-v1",
                       "task_updater": "reme-named-consolidation-v1", "adaptation_updater": "ace-content-deltas-v1",
                       "adapter": "single-fixed-system-full-playbook-v1"},
        "retrieval_k": 10, "retrieval_meta": retrieval, "retrieval_preset": "efficiency_1",
        "retrieval_prompt_manifest": prompt_manifest(retrieval),
        "retrieval_preset_provenance": {"path": "src/memory_retrieval/presets/efficiency_1.variant.json",
                                         "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
                                         "descriptor": json.loads(provenance.read_text())},
        "retrieval_candidates": 40, "reflection_candidates": 3,
        "consolidation_neighbors": 5, "consolidation_candidates": 10,
        "full_prompt_max_bytes": FULL_PROMPT_MAX_BYTES,
        "pair_schema": 2, "initialization": "empty-task-and-playbook-per-stream",
        "playbook_renderer": "full-four-headings-content-only-v1", "playbook_operations": ["ADD", "UPDATE"],
        "evidence_selection": "independent-current-episode-both-outcomes",
        "execution_guidance": "fixed-once; no raw memories",
        "stage_failure_policy": "invalid-output-degrades; provider-cache-artifact-uncertain-write-fatal",
        "memory_phases": ["adaptation", "task_extraction", "task_consolidation", "adaptation_update"],
        "v1_prompt_manifest": {str(path.relative_to(PROMPTS)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(PROMPTS.rglob("*")) if path.is_file()},
    }
