"""Specialist tool contract.

Every tool is a template method: if an inference endpoint is configured for its
registry row it tries that first, and otherwise (or on failure) it runs the local
heuristic baseline and says so in the trace. That is the seam the fine-tuned
models plug into — no caller changes when weights arrive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.agent import registry
from app.core.config import Settings
from app.schemas.agent import Box, CompareMode, Layer, Mask, Metric, TaskKind
from app.schemas.imagery import Asset, InputMode, ValidationReport
from app.services.raster import Scene

BASELINE_BACKEND = "heuristic-baseline"


@dataclass(slots=True)
class ToolContext:
    query: str
    task: TaskKind
    target: str
    mode: InputMode
    assets: list[Asset]
    scenes: list[Scene]
    validation: ValidationReport
    settings: Settings

    @property
    def primary_asset(self) -> Asset:
        return self.assets[0]

    @property
    def primary_scene(self) -> Scene:
        return self.scenes[0]

    @property
    def secondary_asset(self) -> Asset | None:
        return self.assets[1] if len(self.assets) > 1 else None

    @property
    def secondary_scene(self) -> Scene | None:
        return self.scenes[1] if len(self.scenes) > 1 else None

    def optical_index(self) -> int:
        for index, asset in enumerate(self.assets):
            if asset.modality in ("optical", "multispectral"):
                return index
        return 0

    def sar_index(self) -> int | None:
        for index, asset in enumerate(self.assets):
            if asset.modality == "sar":
                return index
        return None


@dataclass(slots=True)
class ToolOutput:
    tool: str
    model: str
    backend: str = BASELINE_BACKEND
    role: str = ""
    title: str = ""
    answer: str = ""
    observations: list[str] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    confidence: float = 0.5
    compare_default: CompareMode | None = None
    params: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Tool(ABC):
    name: str
    role: str
    task: TaskKind
    model_key: str
    produces: list[str] = []

    def backend_label(self, settings: Settings) -> str:
        endpoint = registry.endpoint_for(self.model_key, settings)
        return f"endpoint:{endpoint}" if endpoint else BASELINE_BACKEND

    def run(self, ctx: ToolContext) -> ToolOutput:
        endpoint = registry.endpoint_for(self.model_key, ctx.settings)
        if endpoint:
            try:
                output = self.run_endpoint(ctx, endpoint)
                if output is not None:
                    output.backend = f"endpoint:{endpoint}"
                    output.role = self.role
                    return output
            except NotImplementedError:
                fallback_note = (
                    f"{self.name}: endpoint configured but no adapter implemented yet; "
                    "ran the heuristic baseline."
                )
            except Exception as exc:  # keep the pipeline answerable
                fallback_note = f"{self.name}: endpoint call failed ({exc}); ran the heuristic baseline."
            else:
                fallback_note = f"{self.name}: endpoint returned nothing; ran the heuristic baseline."
            output = self.run_baseline(ctx)
            output.role = self.role
            output.notes.append(fallback_note)
            return output
        output = self.run_baseline(ctx)
        output.role = self.role
        return output

    def run_endpoint(self, ctx: ToolContext, endpoint: str) -> ToolOutput | None:
        """Override once a fine-tuned checkpoint is served for this row."""
        raise NotImplementedError

    @abstractmethod
    def run_baseline(self, ctx: ToolContext) -> ToolOutput:
        """Deterministic local analysis used until weights are wired."""
