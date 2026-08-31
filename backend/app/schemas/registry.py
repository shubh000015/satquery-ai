from pydantic import Field

from app.schemas.agent import TaskKind
from app.schemas.common import CamelModel


class ModelSpec(CamelModel):
    """A row of the team's model plan, plus whether weights are actually wired."""

    key: str
    requirement: str
    primary_model: str
    fallback_model: str | None = None
    benchmarks: list[str] = Field(default_factory=list)
    purpose: str = ""
    tasks: list[TaskKind] = Field(default_factory=list)
    endpoint_setting: str | None = None
    status: str = "planned"


class ToolSpec(CamelModel):
    name: str
    task: TaskKind
    role: str
    model_key: str
    backend: str
    produces: list[str] = Field(default_factory=list)


class RegistryResponse(CamelModel):
    models: list[ModelSpec]
    tools: list[ToolSpec]
    weights_wired: int
    total: int
