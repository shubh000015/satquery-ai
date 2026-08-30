"""Shared land-cover statistics and evidence builders for the specialist tools."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from app.schemas.agent import Box, Layer, Mask, Metric
from app.schemas.imagery import Asset
from app.services import analysis
from app.services.raster import Scene

# Mask ids the workspace already maps onto its layer toggles
# (see MASK_LAYER in src/components/workspace/ImageryStage.tsx).
MASK_ID_WATER = "flood"
MASK_ID_PERMANENT = "channel"
MASK_ID_CHANGE = "gain"
MASK_ID_VEGETATION = "ag"
MASK_ID_BUILTUP = "urban"

COLOR_WATER = "#4ea3ff"
COLOR_PERMANENT = "#1d4ed8"
COLOR_CHANGE = "#ff5b7a"
COLOR_VEGETATION = "#7dce9a"
COLOR_URBAN = "#f0c674"
COLOR_GROUNDING = "#d4b56a"

TARGET_LABELS = {
    "water": "water",
    "vegetation": "vegetation",
    "builtup": "built-up",
    "ship": "ships",
    "tank": "storage tanks",
    "bare": "bare ground",
    "generic": "scene features",
}

BOX_KINDS = {
    "water": "urban",
    "vegetation": "urban",
    "builtup": "settlement",
    "ship": "ship",
    "tank": "tank",
    "bare": "urban",
    "generic": "urban",
}


@dataclass
class SceneStats:
    """Water / vegetation / built-up decomposition of a single scene.

    Not slotted: the cached properties below keep each mask to one computation
    per scene, and cached_property needs a __dict__.
    """

    asset: Asset
    scene: Scene

    @cached_property
    def water(self) -> analysis.MaskResult:
        return analysis.water_mask(self.scene, self.asset.modality)

    @cached_property
    def vegetation(self) -> analysis.MaskResult:
        return analysis.vegetation_mask(self.scene)

    @cached_property
    def builtup(self) -> analysis.MaskResult:
        return analysis.builtup_mask(self.scene, self.water.mask, self.vegetation.mask)

    def result(self, kind: str) -> analysis.MaskResult:
        return {"water": self.water, "vegetation": self.vegetation, "builtup": self.builtup}[kind]

    def mask(self, kind: str) -> np.ndarray:
        return self.result(kind).mask

    def coverage(self, kind: str) -> float:
        return analysis.coverage_percent(self.mask(kind))

    def area_km2(self, kind: str) -> float | None:
        meta = self.asset.meta
        if meta is None:
            return None
        return analysis.area_km2(self.mask(kind), meta.gsd_meters, meta.width, meta.height)

    @property
    def other_coverage(self) -> float:
        used = self.mask("water") | self.mask("vegetation") | self.mask("builtup")
        return float((~used).mean() * 100.0)

    def breakdown(self) -> list[tuple[str, float]]:
        items = [
            ("water", self.coverage("water")),
            ("vegetation", self.coverage("vegetation")),
            ("built-up", self.coverage("builtup")),
            ("other / bare", self.other_coverage),
        ]
        return sorted(items, key=lambda pair: pair[1], reverse=True)

    def dominant(self) -> tuple[str, float]:
        return self.breakdown()[0]


def stats_for(asset: Asset, scene: Scene) -> SceneStats:
    return SceneStats(asset=asset, scene=scene)


def nice(label: str) -> str:
    """Sentence case that does not mangle hyphenated class names ('Built-up')."""
    return label[:1].upper() + label[1:] if label else label


def fmt_percent(value: float) -> str:
    return f"{value:.1f}%"


def fmt_area(km2: float | None) -> str | None:
    if km2 is None:
        return None
    if km2 < 1:
        return f"{km2 * 100:.1f} ha"
    return f"{km2:.2f} km²"


def area_metric(label: str, coverage: float, km2: float | None, hint: str) -> Metric:
    """Report ground area when the geotransform allows it, coverage otherwise."""
    area = fmt_area(km2)
    if area:
        return Metric(label=label, value=area, hint=f"{fmt_percent(coverage)} of frame · {hint}")
    return Metric(label=label, value=fmt_percent(coverage), hint=f"{hint} · no geotransform for km²")


def build_mask(
    mask_id: str,
    label: str,
    color: str,
    array: np.ndarray,
    opacity: float = 0.38,
) -> Mask | None:
    path = analysis.mask_to_svg_path(array)
    if not path:
        return None
    return Mask(
        id=mask_id,
        label=label,
        color=color,
        opacity=opacity,
        d=path,
        coverage_percent=round(analysis.coverage_percent(array), 2),
    )


def water_masks(stats: SceneStats, flood_context: bool) -> list[Mask]:
    """Split water into 'core' (high confidence) and full extent so flood queries
    get an inundation-vs-permanent-water read."""
    masks: list[Mask] = []
    water = stats.mask("water")
    full = build_mask(
        MASK_ID_WATER,
        "Inundation" if flood_context else "Water extent",
        COLOR_WATER,
        water,
    )
    if full:
        masks.append(full)

    if flood_context:
        gray = stats.scene.gray
        core = water & (gray <= np.percentile(gray[water], 35)) if np.any(water) else water
        permanent = build_mask(MASK_ID_PERMANENT, "Persistent water core", COLOR_PERMANENT, core, opacity=0.45)
        if permanent:
            masks.append(permanent)
    return masks


def boxes_from_mask(
    array: np.ndarray,
    target: str,
    limit: int = 12,
    label_prefix: str | None = None,
) -> list[Box]:
    components = analysis.label_components(array)
    kind = BOX_KINDS.get(target, "urban")
    prefix = label_prefix or nice(TARGET_LABELS.get(target, "Region"))
    boxes: list[Box] = []
    for index, component in enumerate(components[:limit], start=1):
        fill = min(1.0, component.mean_value)
        boxes.append(
            Box(
                id=f"{target}-{index}",
                label=f"{prefix} {index:02d}",
                x=round(component.x, 2),
                y=round(component.y, 2),
                w=round(component.w, 2),
                h=round(component.h, 2),
                # Compactness is a usable proxy score: a blob that fills its own
                # bounding box is a cleaner detection than a straggly one.
                score=round(float(np.clip(0.55 + 0.4 * fill, 0.3, 0.98)), 3),
                kind=kind,
            )
        )
    return boxes


def layers_for(kinds: list[str], active: str | None = None) -> list[Layer]:
    catalogue = {
        "water": Layer(id="water", label="Water", color=COLOR_WATER),
        "flood": Layer(id="flood", label="Inundation", color=COLOR_WATER),
        "vegetation": Layer(id="vegetation", label="Vegetation", color=COLOR_VEGETATION),
        "urban": Layer(id="urban", label="Built-up", color=COLOR_URBAN),
        "change": Layer(id="change", label="Change", color=COLOR_CHANGE),
        "grounding": Layer(id="grounding", label="Grounded objects", color=COLOR_GROUNDING),
        "settlements": Layer(id="settlements", label="Settlements", color=COLOR_GROUNDING),
    }
    layers: list[Layer] = []
    for kind in kinds:
        layer = catalogue.get(kind)
        if layer is None:
            continue
        layers.append(layer.model_copy(update={"active": active is None or kind == active or True}))
    return layers


def describe_scene(stats: SceneStats) -> str:
    parts = [f"{label} {fmt_percent(value)}" for label, value in stats.breakdown() if value >= 1.0]
    return ", ".join(parts) if parts else "no class exceeds 1% of the frame"
