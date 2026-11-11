"""V2 composition reuses the parent's evidence and pair-integrity facade."""

from MetaWeft.src.ace_adapt_reme_task.pipeline import V1Pipeline
from .adapter import Adapter
from .adaptation_memory import AdaptationMemory
from .config import ADAPTATION_SCHEMA, PROMPTS


class V2Pipeline(V1Pipeline):
    prompt_directory = PROMPTS
    adapter_class = Adapter
    adaptation_memory_class = AdaptationMemory
    adaptation_schema = ADAPTATION_SCHEMA
    component_prefix = "metaweft_v2"
