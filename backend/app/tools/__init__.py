"""Tool registry and the planner that sequences specialists for a task."""

from __future__ import annotations

from app.agent import registry
from app.schemas.agent import TaskKind
from app.schemas.imagery import InputMode
from app.schemas.registry import ToolSpec
from app.tools.base import Tool, ToolContext, ToolOutput
from app.tools.change import ChangeMaskTool, ChangeVqaTool
from app.tools.fusion import FusionTool, MeasureTool
from app.tools.single_image import CaptionTool, GroundingTool, VqaTool

VQA = VqaTool()
CAPTION = CaptionTool()
GROUNDING = GroundingTool()
CHANGE_MASK = ChangeMaskTool()
CHANGE_VQA = ChangeVqaTool()
FUSION = FusionTool()
MEASURE = MeasureTool()

ALL_TOOLS: tuple[Tool, ...] = (VQA, CAPTION, GROUNDING, CHANGE_MASK, CHANGE_VQA, FUSION, MEASURE)

_CLASS_TARGETS = {"water", "vegetation", "builtup"}
_OBJECT_TARGETS = {"ship", "tank"}


def plan(task: TaskKind, mode: InputMode, target: str) -> list[Tool]:
    """Ordered specialist chain. The first tool owns the answer; the rest add
    evidence, which is what makes the run genuinely multi-tool rather than one
    model doing everything."""
    if task == "vqa":
        chain = [VQA]
        if target in _CLASS_TARGETS or target in _OBJECT_TARGETS:
            chain.append(GROUNDING)  # attach visual evidence to the answer
        return chain
    if task == "caption":
        chain = [CAPTION]
        if target in _CLASS_TARGETS:
            chain.append(GROUNDING)
        return chain
    if task == "grounding":
        return [GROUNDING, CAPTION]  # boxes plus scene context
    if task == "change":
        return [CHANGE_MASK, CHANGE_VQA]
    if task == "change-vqa":
        return [CHANGE_VQA, CHANGE_MASK]
    if task == "cross-modal":
        chain = [FUSION]
        if target in _OBJECT_TARGETS or target == "builtup":
            chain.append(GROUNDING)
        return chain
    if task == "measure":
        return [MEASURE]
    return [VQA]


def tool_specs() -> list[ToolSpec]:
    from app.core.config import get_settings

    settings = get_settings()
    return [
        ToolSpec(
            name=tool.name,
            task=tool.task,
            role=tool.role,
            model_key=tool.model_key,
            backend=tool.backend_label(settings),
            produces=list(tool.produces),
        )
        for tool in ALL_TOOLS
    ]


__all__ = [
    "ALL_TOOLS",
    "CAPTION",
    "CHANGE_MASK",
    "CHANGE_VQA",
    "FUSION",
    "GROUNDING",
    "MEASURE",
    "Tool",
    "ToolContext",
    "ToolOutput",
    "VQA",
    "plan",
    "registry",
    "tool_specs",
]
