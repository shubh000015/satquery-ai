"""Bi-temporal specialists: spatial change masks and change VQA."""

from __future__ import annotations

import numpy as np

from app.schemas.agent import Metric
from app.services import analysis
from app.tools import common
from app.tools.base import Tool, ToolContext, ToolOutput
from app.tools.endpoint_client import ask_change, decode_rle

_CHANGE_TARGET_DEFAULT = "builtup"
_SIGNIFICANT_POINTS = 1.0  # percentage points of frame coverage
_SIGNIFICANT_RELATIVE = 0.05


def _pair(ctx: ToolContext):
    before_asset, after_asset = ctx.assets[0], ctx.assets[1]
    before_scene, after_scene = ctx.scenes[0], ctx.scenes[1]
    return before_asset, after_asset, before_scene, after_scene


def _area_of(mask: np.ndarray, asset) -> float | None:
    if asset.meta is None:
        return None
    return analysis.area_km2(mask, asset.meta.gsd_meters, asset.meta.width, asset.meta.height)


class ChangeMaskTool(Tool):
    name = "CD-Mapper"
    role = "bi-temporal spatial change mask"
    task = "change"
    model_key = "change-mask"
    produces = ["mask", "boxes", "metrics"]

    def run_endpoint(self, ctx: ToolContext, endpoint: str) -> ToolOutput | None:
        """ChangeFormer's change map replaces the differencing baseline's mask.

        The served endpoint measures changed fraction and region count from the
        mask itself, so those numbers are model-derived rather than thresholded
        brightness. Directional gain/loss stays local because a binary change
        mask cannot express direction.
        """
        before_asset, after_asset, before_scene, after_scene = _pair(ctx)
        reply = ask_change(endpoint, before_scene, after_scene, ctx.query)
        output = self.run_baseline(ctx)

        output.answer = reply.answer
        output.confidence = max(0.15, min(0.97, reply.confidence))
        output.params["endpoint_model"] = reply.model

        runs = reply.extra.get("changeMaskRle")
        height = reply.extra.get("maskHeight")
        width = reply.extra.get("maskWidth")
        if runs and height and width:
            array = decode_rle(runs, int(height), int(width))
            mask = common.build_mask(
                common.MASK_ID_CHANGE, "Change", common.COLOR_CHANGE, array, opacity=0.42
            )
            if mask:
                output.masks = [mask]
            boxes = common.boxes_from_mask(array, "builtup", limit=12, label_prefix="Change")
            if boxes:
                output.boxes = boxes

        stats = reply.extra.get("changeStats") or {}
        if stats:
            output.metrics.insert(
                0,
                Metric(
                    label="Changed area (model)",
                    value=f"{stats.get('changedPercent', 0):.2f}%",
                    hint=f"{stats.get('regionCount', 0)} region(s) · {reply.model}",
                ),
            )
        output.observations.insert(
            0, f"Change map produced by {reply.model}; area and region count measured from it."
        )
        output.notes.extend(reply.notes)
        return output

    def run_baseline(self, ctx: ToolContext) -> ToolOutput:
        before_asset, after_asset, before_scene, after_scene = _pair(ctx)
        change = analysis.change_mask(before_scene, after_scene)
        gained, lost = analysis.directional_change(before_scene, after_scene, change.mask)

        coverage = analysis.coverage_percent(change.mask)
        area = _area_of(change.mask, after_asset)
        boxes = common.boxes_from_mask(change.mask, "builtup", limit=12, label_prefix="Change")

        mask = common.build_mask(
            common.MASK_ID_CHANGE, "Change", common.COLOR_CHANGE, change.mask, opacity=0.42
        )

        extent = common.fmt_area(area) or common.fmt_percent(coverage)
        answer = (
            f"{extent} of the shared footprint changed between {before_asset.date} and {after_asset.date} "
            f"({common.fmt_percent(coverage)} of the frame). Of the changed pixels, "
            f"{gained * 100:.0f}% brightened and {lost * 100:.0f}% darkened — brightening on optical pairs "
            "usually means new built-up or cleared surface, darkening means vegetation loss, shadow or new water."
        )

        return ToolOutput(
            tool=self.name,
            model=self.model_key,
            title=f"Change: {before_asset.date} → {after_asset.date}",
            answer=answer,
            observations=[
                f"{len(boxes)} distinct change patches passed the minimum-area filter.",
                *change.notes,
                f"Otsu threshold on change magnitude: {change.threshold:.3f} (separability {change.separability:.2f}).",
            ],
            metrics=[
                common.area_metric("Changed area", coverage, area, change.method),
                Metric(label="Change patches", value=str(len(boxes))),
                Metric(label="Brightened", value=f"{gained * 100:.0f}%", hint="of changed pixels"),
                Metric(label="Darkened", value=f"{lost * 100:.0f}%", hint="of changed pixels"),
            ],
            boxes=boxes,
            masks=[m for m in [mask] if m],
            layers=common.layers_for(["change", "settlements", "urban"]),
            confidence=analysis.confidence_from(change.separability, coverage, len(boxes)),
            compare_default="swipe",
            params={
                "method": change.method,
                "threshold": round(change.threshold, 4),
                "t0": before_asset.date,
                "t1": after_asset.date,
            },
            outputs={
                "changed_percent": round(coverage, 2),
                "changed_km2": round(area, 3) if area else None,
                "brightened_fraction": round(gained, 3),
                "patches": len(boxes),
            },
        )


class ChangeVqaTool(Tool):
    """Answers 'has X increased / decreased / stayed the same' over a pair."""

    name = "CDVQA"
    role = "change-based visual question answering"
    task = "change-vqa"
    model_key = "change-vqa"
    produces = ["answer", "metrics", "mask"]

    def run_endpoint(self, ctx: ToolContext, endpoint: str) -> ToolOutput | None:
        """Model localises the change; the per-date class comparison gives direction.

        Deliberately keeps the local verdict as the answer. "Has built-up
        increased or decreased?" needs a signed comparison of the same class at
        two dates, and a binary change mask cannot supply that — the served
        endpoint says as much in its own notes. So the model contributes *where*
        and *how much*, and the class comparison contributes *which way*.
        """
        before_scene = ctx.scenes[0]
        after_scene = ctx.scenes[1] if len(ctx.scenes) > 1 else ctx.scenes[0]
        reply = ask_change(endpoint, before_scene, after_scene, ctx.query)
        output = self.run_baseline(ctx)

        output.confidence = max(output.confidence, min(0.97, reply.confidence))
        output.params["endpoint_model"] = reply.model

        stats = reply.extra.get("changeStats") or {}
        if stats:
            output.metrics.insert(
                0,
                Metric(
                    label="Changed area (model)",
                    value=f"{stats.get('changedPercent', 0):.2f}%",
                    hint=f"{stats.get('regionCount', 0)} region(s) · {reply.model}",
                ),
            )
            where = stats.get("where")
            output.observations.insert(
                0,
                f"{reply.model} measured {stats.get('changedPercent', 0):.2f}% of the frame as changed"
                + (f", concentrated in the {where}." if where else ".")
                + " Direction comes from the per-date class comparison below, because a "
                "binary change mask cannot signal increase versus decrease.",
            )

        runs = reply.extra.get("changeMaskRle")
        height, width = reply.extra.get("maskHeight"), reply.extra.get("maskWidth")
        if runs and height and width:
            mask = common.build_mask(
                common.MASK_ID_CHANGE,
                "Change",
                common.COLOR_CHANGE,
                decode_rle(runs, int(height), int(width)),
                opacity=0.42,
            )
            if mask:
                output.masks = [mask]

        output.notes.extend(reply.notes)
        return output

    def run_baseline(self, ctx: ToolContext) -> ToolOutput:
        before_asset, after_asset, before_scene, after_scene = _pair(ctx)
        target = ctx.target if ctx.target in {"water", "vegetation", "builtup"} else _CHANGE_TARGET_DEFAULT
        label = common.TARGET_LABELS[target]

        before_stats = common.stats_for(before_asset, before_scene)
        after_stats = common.stats_for(after_asset, after_scene)
        before_pct = before_stats.coverage(target)
        after_pct = after_stats.coverage(target)
        delta = after_pct - before_pct
        relative = delta / before_pct if before_pct > 0.5 else (1.0 if delta > 0 else -1.0 if delta < 0 else 0.0)

        significant = abs(delta) >= _SIGNIFICANT_POINTS and abs(relative) >= _SIGNIFICANT_RELATIVE
        if not significant:
            verdict = "remained essentially unchanged"
        elif delta > 0:
            verdict = "increased"
        else:
            verdict = "decreased"

        change = analysis.change_mask(before_scene, after_scene)
        gained, _lost = analysis.directional_change(before_scene, after_scene, change.mask)
        changed_area = _area_of(change.mask, after_asset)

        before_area = common.fmt_area(before_stats.area_km2(target))
        after_area = common.fmt_area(after_stats.area_km2(target))
        extent_phrase = (
            f"{before_area} → {after_area}"
            if before_area and after_area
            else f"{common.fmt_percent(before_pct)} → {common.fmt_percent(after_pct)} of frame"
        )

        answer = (
            f"{label.capitalize()} has {verdict} between {before_asset.date} and {after_asset.date}: "
            f"{extent_phrase} ({delta:+.1f} percentage points of frame coverage, {relative * 100:+.0f}% relative). "
            f"{common.fmt_percent(analysis.coverage_percent(change.mask))} of the footprint changed overall, "
            f"{gained * 100:.0f}% of it brightening."
        )

        mask = common.build_mask(
            common.MASK_ID_CHANGE, "Change", common.COLOR_CHANGE, change.mask, opacity=0.4
        )

        confidence = analysis.confidence_from(
            min(change.separability, max(before_stats.result(target).separability, after_stats.result(target).separability)),
            analysis.coverage_percent(change.mask),
        )
        if not significant:
            # A null result is a real answer, but say so with less certainty.
            confidence = min(confidence, 0.7)

        return ToolOutput(
            tool=self.name,
            model=self.model_key,
            title=f"Change VQA: {label}",
            answer=answer,
            observations=[
                f"{label.capitalize()} at t0 ({before_asset.date}): {common.fmt_percent(before_pct)}.",
                f"{label.capitalize()} at t1 ({after_asset.date}): {common.fmt_percent(after_pct)}.",
                f"Decision rule: ≥{_SIGNIFICANT_POINTS:.1f} percentage points and ≥{_SIGNIFICANT_RELATIVE * 100:.0f}% relative shift counts as a change.",
                *change.notes,
            ],
            metrics=[
                Metric(label=f"{common.nice(label)} t0", value=common.fmt_percent(before_pct), hint=before_asset.date),
                Metric(label=f"{common.nice(label)} t1", value=common.fmt_percent(after_pct), hint=after_asset.date),
                Metric(label="Delta", value=f"{delta:+.1f} pp", hint=f"{relative * 100:+.0f}% relative"),
                common.area_metric(
                    "Changed area", analysis.coverage_percent(change.mask), changed_area, change.method
                ),
            ],
            masks=[m for m in [mask] if m],
            layers=common.layers_for(["change", "urban", "vegetation", "water"]),
            confidence=confidence,
            compare_default="swipe",
            params={
                "target_class": target,
                "t0": before_asset.date,
                "t1": after_asset.date,
                "significance_points": _SIGNIFICANT_POINTS,
            },
            outputs={
                "verdict": verdict,
                "t0_percent": round(before_pct, 2),
                "t1_percent": round(after_pct, 2),
                "delta_points": round(delta, 2),
            },
        )
