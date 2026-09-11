"""Optical–SAR fusion and mensuration specialists."""

from __future__ import annotations

import numpy as np

from app.schemas.agent import Box, Metric
from app.services import analysis
from app.tools import common
from app.tools.base import Tool, ToolContext, ToolOutput
from app.tools.single_image import _endpoint_answer


def _align(*masks: np.ndarray) -> list[np.ndarray]:
    """Crop masks to their common shape. Full reprojection is a rasterio job;
    the validator has already rejected footprints that do not overlap."""
    height = min(m.shape[0] for m in masks)
    width = min(m.shape[1] for m in masks)
    return [m[:height, :width] for m in masks]


def _boxes_touching(boxes: list[Box], mask: np.ndarray) -> int:
    """How many boxes overlap a mask, using percentage coords back onto the grid."""
    if mask.size == 0:
        return 0
    h, w = mask.shape
    count = 0
    for box in boxes:
        y0 = int(np.clip(box.y / 100.0 * h, 0, h - 1))
        y1 = int(np.clip((box.y + box.h) / 100.0 * h, y0 + 1, h))
        x0 = int(np.clip(box.x / 100.0 * w, 0, w - 1))
        x1 = int(np.clip((box.x + box.w) / 100.0 * w, x0 + 1, w))
        if np.any(mask[y0:y1, x0:x1]):
            count += 1
    return count


class FusionTool(Tool):
    """Joint optical + SAR extraction: what each sensor sees, and what only SAR sees."""

    name = "OptSAR-Fusion"
    role = "cross-modal optical–SAR information extraction"
    task = "cross-modal"
    model_key = "fusion"
    produces = ["mask", "metrics", "boxes"]

    def run_endpoint(self, ctx: ToolContext, endpoint: str) -> ToolOutput | None:
        return _endpoint_answer(self, ctx, endpoint, task="cross-modal")

    def run_baseline(self, ctx: ToolContext) -> ToolOutput:
        optical_index = ctx.optical_index()
        sar_index = ctx.sar_index()

        optical_asset = ctx.assets[optical_index]
        optical_stats = common.stats_for(optical_asset, ctx.scenes[optical_index])

        if sar_index is None:
            # Router degradation should prevent this; answer honestly if it happens.
            coverage = optical_stats.coverage("water")
            return ToolOutput(
                tool=self.name,
                model=self.model_key,
                title="Fusion unavailable",
                answer=(
                    "No SAR scene is attached, so cross-modal fusion cannot run. Reporting the optical "
                    f"read alone: water {common.fmt_percent(coverage)}, built-up "
                    f"{common.fmt_percent(optical_stats.coverage('builtup'))}."
                ),
                observations=["Attach a co-registered SAR scene to enable joint extraction."],
                metrics=[common.area_metric("Water (optical only)", coverage, optical_stats.area_km2("water"), optical_stats.water.method)],
                layers=common.layers_for(["water", "urban"]),
                confidence=min(0.5, analysis.confidence_from(optical_stats.water.separability, coverage)),
                params={"sar_available": False},
                outputs={"water_percent": round(coverage, 2)},
            )

        sar_asset = ctx.assets[sar_index]
        sar_stats = common.stats_for(sar_asset, ctx.scenes[sar_index])

        optical_water, sar_water, builtup = _align(
            optical_stats.mask("water"), sar_stats.mask("water"), optical_stats.mask("builtup")
        )
        fused = optical_water | sar_water
        agreement = optical_water & sar_water
        sar_only = sar_water & ~optical_water
        optical_only = optical_water & ~sar_water

        union = float(fused.sum())
        iou = float(agreement.sum()) / union if union > 0 else 0.0

        fused_coverage = analysis.coverage_percent(fused)
        fused_area = (
            analysis.area_km2(fused, sar_asset.meta.gsd_meters, sar_asset.meta.width, sar_asset.meta.height)
            if sar_asset.meta
            else None
        )
        sar_gain_area = (
            analysis.area_km2(sar_only, sar_asset.meta.gsd_meters, sar_asset.meta.width, sar_asset.meta.height)
            if sar_asset.meta
            else None
        )

        boxes = common.boxes_from_mask(builtup, "builtup", limit=10, label_prefix="Settlement")
        exposed = _boxes_touching(boxes, fused)

        masks = []
        fused_mask = common.build_mask(common.MASK_ID_WATER, "Fused water extent", common.COLOR_WATER, fused)
        if fused_mask:
            masks.append(fused_mask)
        agreement_mask = common.build_mask(
            common.MASK_ID_PERMANENT, "Both sensors agree", common.COLOR_PERMANENT, agreement, opacity=0.45
        )
        if agreement_mask:
            masks.append(agreement_mask)

        extent = common.fmt_area(fused_area) or common.fmt_percent(fused_coverage)
        gain = common.fmt_area(sar_gain_area) or common.fmt_percent(analysis.coverage_percent(sar_only))

        answer = (
            f"Fusing {optical_asset.modality} ({optical_asset.sensor}) with SAR ({sar_asset.sensor}) puts the "
            f"water-covered extent at {extent}. The two sensors agree on {iou * 100:.0f}% of that extent "
            f"(IoU). SAR alone contributes {gain} that the optical read misses — the expected signature of "
            "cloud, shadow or thin haze, since radar returns collapse over specular water regardless of "
            f"illumination. Built-up extraction from the optical scene gives "
            f"{common.fmt_percent(analysis.coverage_percent(builtup))} of the frame, and {exposed} of "
            f"{len(boxes)} settlement clusters intersect the fused water mask."
        )

        return ToolOutput(
            tool=self.name,
            model=self.model_key,
            title="Optical–SAR joint extraction",
            answer=answer,
            observations=[
                f"Optical water {common.fmt_percent(analysis.coverage_percent(optical_water))} "
                f"({optical_stats.water.method}); SAR water {common.fmt_percent(analysis.coverage_percent(sar_water))} "
                f"({sar_stats.water.method}).",
                f"SAR-only gain {common.fmt_percent(analysis.coverage_percent(sar_only))}; "
                f"optical-only {common.fmt_percent(analysis.coverage_percent(optical_only))}.",
                f"Cross-sensor agreement (IoU) {iou:.2f} — low agreement usually means cloud in the optical scene "
                "or layover/shadow in the SAR scene.",
                *sar_stats.water.notes,
            ],
            metrics=[
                common.area_metric("Fused water", fused_coverage, fused_area, "optical ∪ SAR"),
                Metric(label="Sensor agreement", value=f"{iou * 100:.0f}%", hint="IoU of the two water masks"),
                common.area_metric(
                    "SAR-only gain", analysis.coverage_percent(sar_only), sar_gain_area, "seen by radar alone"
                ),
                Metric(
                    label="Settlements exposed",
                    value=f"{exposed} / {len(boxes)}",
                    hint="clusters intersecting fused water",
                ),
            ],
            boxes=boxes,
            masks=masks,
            layers=common.layers_for(["flood", "water", "settlements", "urban"]),
            confidence=analysis.confidence_from(
                (optical_stats.water.separability + sar_stats.water.separability) / 2,
                fused_coverage,
                len(boxes),
            ),
            compare_default="swipe",
            params={
                "optical_asset": optical_asset.id,
                "sar_asset": sar_asset.id,
                "optical_method": optical_stats.water.method,
                "sar_method": sar_stats.water.method,
            },
            outputs={
                "fused_water_percent": round(fused_coverage, 2),
                "iou": round(iou, 3),
                "sar_only_percent": round(analysis.coverage_percent(sar_only), 2),
                "settlements_exposed": exposed,
            },
        )


class MeasureTool(Tool):
    """Mensuration straight off the geotransform — no learned model involved."""

    name = "Geometry"
    role = "GSD-scaled area and extent"
    task = "measure"
    model_key = "mensuration"
    produces = ["metrics"]

    def run_baseline(self, ctx: ToolContext) -> ToolOutput:
        asset = ctx.primary_asset
        stats = common.stats_for(asset, ctx.primary_scene)
        target = ctx.target if ctx.target in {"water", "vegetation", "builtup"} else "water"
        label = common.TARGET_LABELS[target]

        mask = stats.mask(target)
        coverage = analysis.coverage_percent(mask)
        area = stats.area_km2(target)
        meta = asset.meta
        components = analysis.label_components(mask)

        scene_area = None
        if meta and meta.gsd_meters:
            scene_area = meta.width * meta.height * meta.gsd_meters**2 / 1_000_000.0

        if area is None:
            answer = (
                f"{label.capitalize()} occupies {common.fmt_percent(coverage)} of the frame, but this input "
                "carries no geotransform, so ground area cannot be reported. Supply GeoTIFF with a CRS and "
                "pixel scale for measurements in km²."
            )
        else:
            largest = components[0].area_fraction * (scene_area or 0) if components and scene_area else None
            answer = (
                f"{label.capitalize()} measures {common.fmt_area(area)} "
                f"({common.fmt_percent(coverage)} of a {common.fmt_area(scene_area)} scene) at "
                f"{meta.gsd_meters:.1f} m GSD."
            )
            if largest:
                answer += f" The largest single patch is about {common.fmt_area(largest)}."

        metrics = [
            common.area_metric(common.nice(label), coverage, area, "thresholded extent"),
            Metric(label="Patches", value=str(len(components))),
        ]
        if scene_area:
            metrics.append(Metric(label="Scene area", value=common.fmt_area(scene_area) or "—", hint=f"{meta.width}×{meta.height} px"))
        if meta and meta.gsd_meters:
            metrics.append(Metric(label="GSD", value=f"{meta.gsd_meters:.2f} m", hint=asset.crs or "no CRS"))

        mask_obj = common.build_mask(
            common.MASK_ID_WATER if target == "water" else common.MASK_ID_BUILTUP,
            f"{common.nice(label)} extent",
            common.COLOR_WATER if target == "water" else common.COLOR_URBAN,
            mask,
            opacity=0.35,
        )

        return ToolOutput(
            tool=self.name,
            model=self.model_key,
            title=f"Mensuration: {label}",
            answer=answer,
            observations=[
                f"Measured from {stats.result(target).method}.",
                (
                    f"Geotransform present ({asset.crs or 'unknown CRS'}), so areas are ground-true."
                    if area is not None
                    else "No geotransform: percentages only."
                ),
            ],
            metrics=metrics,
            masks=[m for m in [mask_obj] if m],
            layers=common.layers_for(["water", "urban", "vegetation"]),
            confidence=analysis.confidence_from(stats.result(target).separability, coverage, len(components)),
            params={"target_class": target, "gsd_meters": meta.gsd_meters if meta else None},
            outputs={
                "coverage_percent": round(coverage, 2),
                "area_km2": round(area, 4) if area else None,
                "patches": len(components),
            },
        )
