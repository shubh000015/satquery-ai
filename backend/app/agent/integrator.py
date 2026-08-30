"""Result integration — combines specialist outputs into one grounded answer.

The first tool in the chain owns the answer; the rest contribute evidence,
metrics and caveats. Confidence is weighted towards the owning tool and then
penalised for anything the validator flagged, so a shaky input cannot produce a
confident-looking response.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.agent import registry
from app.schemas.agent import (
    AgentStep,
    Box,
    Intent,
    Layer,
    Mask,
    Metric,
    ModelUse,
    QueryResult,
)
from app.schemas.imagery import ValidationReport
from app.tools.base import BASELINE_BACKEND, ToolContext, ToolOutput

_MAX_METRICS = 8
_MAX_OBSERVATIONS = 10


def _dedupe_metrics(groups: list[list[Metric]]) -> list[Metric]:
    seen: set[str] = set()
    merged: list[Metric] = []
    for group in groups:
        for metric in group:
            if metric.label in seen:
                continue
            seen.add(metric.label)
            merged.append(metric)
    return merged[:_MAX_METRICS]


def _dedupe_masks(groups: list[list[Mask]]) -> list[Mask]:
    seen: set[str] = set()
    merged: list[Mask] = []
    for group in groups:
        for mask in group:
            if mask.id in seen:
                continue
            seen.add(mask.id)
            merged.append(mask)
    return merged


def _dedupe_boxes(outputs: list[ToolOutput]) -> list[Box]:
    merged: list[Box] = []
    seen: set[str] = set()
    for output in outputs:
        for box in output.boxes:
            box_id = box.id if box.id not in seen else f"{output.model}-{box.id}"
            if box_id in seen:
                continue
            seen.add(box_id)
            merged.append(box.model_copy(update={"id": box_id}))
    return merged


def _dedupe_layers(groups: list[list[Layer]]) -> list[Layer]:
    seen: set[str] = set()
    merged: list[Layer] = []
    for group in groups:
        for layer in group:
            if layer.id in seen:
                continue
            seen.add(layer.id)
            merged.append(layer)
    return merged


def _confidence(primary: ToolOutput, supporting: list[ToolOutput], report: ValidationReport) -> float:
    value = primary.confidence
    if supporting:
        support_mean = sum(o.confidence for o in supporting) / len(supporting)
        value = 0.7 * value + 0.3 * support_mean
    value -= 0.05 * len(report.warnings)
    if report.co_registered is None and report.asset_count > 1:
        value -= 0.03
    return round(max(0.15, min(0.97, value)), 3)


def _models(outputs: list[ToolOutput], settings) -> list[ModelUse]:
    models = [
        ModelUse(
            name="Input Guardian",
            role="format · CRS · co-registration",
            status="complete",
            backend="rule-based",
        )
    ]
    for output in outputs:
        models.append(
            ModelUse(
                name=registry.display_name(output.model, settings),
                role=output.role or output.tool,
                status="complete",
                backend=output.backend,
            )
        )
    # The standby rows advertise the model plan. One checkpoint can cover several
    # requirements (Qwen2.5-VL-7B does grounding and captioning), so group by name:
    # the UI keys these lists by model name and would otherwise see duplicates.
    executed = {o.model for o in outputs}
    used_names = {m.name for m in models}
    standby: dict[str, list[str]] = {}
    for spec in registry.model_specs(settings):
        if spec.status != "planned" or spec.key in executed or spec.key == "controller":
            continue
        if spec.primary_model in used_names:
            continue
        standby.setdefault(spec.primary_model, []).append(spec.requirement.lower())

    for name, requirements in standby.items():
        models.append(
            ModelUse(
                name=name,
                role=f"{' · '.join(requirements)} · awaiting weights",
                status="standby",
                backend="not wired",
            )
        )
    return models


def integrate(
    ctx: ToolContext,
    intent: Intent,
    outputs: list[ToolOutput],
    report: ValidationReport,
    trace: list[AgentStep],
    elapsed_ms: int,
    *,
    query_id: str,
    session_id: str | None,
) -> QueryResult:
    primary = outputs[0]
    supporting = outputs[1:]

    observations = list(primary.observations)
    for output in supporting:
        observations.append(f"{output.tool}: {output.answer}")
        observations.extend(output.observations[:2])

    warnings: list[str] = []
    if intent.warning:
        warnings.append(intent.warning)
    warnings.extend(issue.message for issue in report.warnings)
    for output in outputs:
        warnings.extend(output.notes)

    backends = {output.backend for output in outputs}
    inference_backend = (
        BASELINE_BACKEND if backends == {BASELINE_BACKEND} else ", ".join(sorted(backends))
    )

    compare_default = primary.compare_default
    if compare_default is None and len(ctx.assets) > 1:
        compare_default = "split"

    return QueryResult(
        task=intent.task,
        title=primary.title or intent.label,
        answer=primary.answer,
        observations=observations[:_MAX_OBSERVATIONS],
        metrics=_dedupe_metrics([primary.metrics, *[o.metrics for o in supporting]]),
        confidence=_confidence(primary, supporting, report),
        boxes=_dedupe_boxes(outputs),
        masks=_dedupe_masks([primary.masks, *[o.masks for o in supporting]]),
        layers=_dedupe_layers([primary.layers, *[o.layers for o in supporting]]),
        models=_models(outputs, ctx.settings),
        trace=trace,
        compare_default=compare_default,
        query_id=query_id,
        session_id=session_id,
        query=ctx.query,
        mode=ctx.mode,
        intent=intent,
        warnings=warnings,
        inference_backend=inference_backend,
        assets=ctx.assets,
        validation=report,
        elapsed_ms=elapsed_ms,
        created_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    )
