"""Small data contracts; concrete payload schemas belong to each variant."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json


def content_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StateRef:
    role: str
    schema: str
    stream_id: str
    epoch: int
    snapshot: str
    sha256: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PreparedMemory:
    role: str
    prior: StateRef
    payload: object
    status: str = "completed"
    error: str | None = None
    trace_refs: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "payload", deepcopy(self.payload))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PreparedEpisode:
    task_spec: dict
    prior_pair: dict
    task_prepared: PreparedMemory
    adaptation_prepared: PreparedMemory
    guidance: str
    adaptation_event: dict
    adaptation_event_ref: str
    status: str = "completed"
    degraded: bool = False
    error: str | None = None
    inputs_sha256: str = field(init=False)

    def __post_init__(self):
        for key in ("task_spec", "prior_pair", "task_prepared", "adaptation_prepared", "adaptation_event"):
            object.__setattr__(self, key, deepcopy(getattr(self, key)))
        object.__setattr__(self, "inputs_sha256", content_hash(self.inputs()))

    def inputs(self):
        return {"task_spec": self.task_spec, "prior_pair": self.prior_pair,
                "task_prepared": self.task_prepared.to_dict(),
                "adaptation_prepared": self.adaptation_prepared.to_dict(),
                "guidance": self.guidance, "adaptation_event": self.adaptation_event}

    def verify(self):
        if content_hash(self.inputs()) != self.inputs_sha256:
            raise ValueError("Prepared episode inputs changed before updates")

    def to_dict(self):
        return {"guidance": self.guidance, "status": self.status, "degraded": self.degraded,
                "error": self.error,
                "prior_pair": deepcopy(self.prior_pair), "inputs_sha256": self.inputs_sha256,
                "adaptation_event_ref": self.adaptation_event_ref}


@dataclass(frozen=True)
class EpisodeEvidence:
    payload: dict
    reference: str
    sha256: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "payload", deepcopy(self.payload))
        object.__setattr__(self, "sha256", content_hash(self.payload))

    def verify(self):
        if content_hash(self.payload) != self.sha256:
            raise ValueError("Episode evidence changed during updates")


@dataclass
class BranchUpdateResult:
    role: str
    status: str = "completed"
    degraded: bool = False
    error: str | None = None
    selected_evidence_refs: list = field(default_factory=list)
    proposed_refs: list = field(default_factory=list)
    applied_refs: list = field(default_factory=list)
    before: object = None
    after: object = None
    decisions: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class PairUpdateResult:
    task: BranchUpdateResult
    adaptation: BranchUpdateResult

    @property
    def degraded(self):
        return self.task.degraded or self.adaptation.degraded

    @property
    def status(self):
        return "degraded" if self.degraded else "completed"

    @property
    def error(self):
        errors = [f"{branch.role}: {branch.error}" for branch in (self.task, self.adaptation) if branch.error]
        return "; ".join(errors) or None

    def to_dict(self):
        # Full operation records live in branch artifacts, not result.json.
        branches = {}
        for branch in (self.task, self.adaptation):
            value = branch.to_dict()
            value.pop("decisions")
            branches[branch.role] = value
        return {"status": self.status, "degraded": self.degraded, "error": self.error, **branches}
