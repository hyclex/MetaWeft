"""Observation Dictionary uses the lossless consolidation encoding from the repaired R36 search."""

from collections import Counter
from copy import deepcopy

from MetaWeft.src.ace_adapt_reme_task.task_memory import TaskMemory
from MetaWeft.src.ace_adapt_reme_task_v2 import V2Pipeline


ENCODING_INSTRUCTION = ('\n\nIn evidence traces, an observation of the form {"ref":"rN"} represents '
                        'the complete unchanged observation string at observation_table[rN].')


def compact_request(template, payload):
    observations = [row["observation"] for source in payload["evidence"].values()
                    for row in source["trace"] if "observation" in row]
    counts = Counter(value for value in observations if len(value.encode("utf-8")) >= 256)
    repeated = {value for value, count in counts.items() if count >= 2}
    if not repeated:
        return template, payload
    result = deepcopy(payload)
    identifiers, table = {}, {}
    for source in result["evidence"].values():
        for row in source["trace"]:
            if "observation" not in row:
                continue
            observation = row["observation"]
            if observation in repeated:
                if observation not in identifiers:
                    key = f"r{len(identifiers)}"
                    identifiers[observation] = key
                    table[key] = observation
                row["observation"] = {"ref": identifiers[observation]}
    result["observation_table"] = table
    return template + ENCODING_INSTRUCTION, result


class CompactTaskMemory(TaskMemory):
    def consolidation_request(self, candidate, neighbors, sources):
        return compact_request(*super().consolidation_request(candidate, neighbors, sources))


class ObservationDictionary(V2Pipeline):
    task_memory_class = CompactTaskMemory
