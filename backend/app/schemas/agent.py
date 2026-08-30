from typing import Any, Literal

from pydantic import Field

from app.schemas.common import CamelModel
from app.schemas.imagery import Asset, InputMode, ValidationReport

TaskKind = Literal[
    "vqa",
    "caption",
    "grounding",
    "change",
    "change-vqa",
    "cross-modal",
    "measure",
]

CompareMode = Literal["primary", "secondary", "split", "swipe", "diff"]
LayerId = Literal[
    "flood",
    "water",
    "urban",
    "vegetation",
    "change",
    "grounding",
    "settlements",
]
StepStatus = Literal["pending", "running", "done", "failed"]
ModelStatus = Literal["queued", "running", "complete", "standby", "failed"]


class Box(CamelModel):
    """Percentage coordinates in the 0..100 viewBox the imagery stage draws in."""

    id: str
    label: str
    x: float
    y: float
    w: float
    h: float
    score: float
    kind: str | None = None


class Mask(CamelModel):
    id: str
    label: str
    color: str
    opacity: float
    d: str
    coverage_percent: float | None = None


class Layer(CamelModel):
    id: LayerId
    label: str
    color: str
    active: bool = True


class Metric(CamelModel):
    label: str
    value: str
    hint: str | None = None


class ModelUse(CamelModel):
    name: str
    role: str
    status: ModelStatus = "complete"
    backend: str | None = None


class AgentStep(CamelModel):
    """One row of the auditable execution trace."""

    id: str
    label: str
    detail: str
    status: StepStatus = "pending"
    tool: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int | None = None


class Intent(CamelModel):
    task: TaskKind
    label: str
    specialists: list[str] = Field(default_factory=list)
    warning: str | None = None
    matched_terms: list[str] = Field(default_factory=list)


class QueryResult(CamelModel):
    """Superset of the frontend `QueryResult`; extra keys are ignored by the UI."""

    task: TaskKind
    title: str
    answer: str
    observations: list[str] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    confidence: float = 0.0
    boxes: list[Box] = Field(default_factory=list)
    masks: list[Mask] = Field(default_factory=list)
    layers: list[Layer] = Field(default_factory=list)
    models: list[ModelUse] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)
    compare_default: CompareMode | None = None

    query_id: str = ""
    session_id: str | None = None
    query: str = ""
    mode: InputMode = "single"
    intent: Intent | None = None
    warnings: list[str] = Field(default_factory=list)
    inference_backend: str = "heuristic-baseline"
    assets: list[Asset] = Field(default_factory=list)
    validation: ValidationReport | None = None
    elapsed_ms: int = 0
    created_at: str = ""
