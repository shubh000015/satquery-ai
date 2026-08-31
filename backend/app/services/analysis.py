"""Heuristic remote-sensing baseline.

None of this pretends to be a fine-tuned model. It exists so the agentic
controller has real specialist outputs to orchestrate, integrate and cite:
thresholded indices, connected components, bi-temporal difference maps. Each
tool reports which method produced its numbers, so swapping in RS-adapted
weights later changes the evidence, not the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.services.raster import Scene

# Grids used for vector evidence. Coarser than the analysis raster on purpose:
# blocky-but-honest polygons beat pretending to sub-pixel accuracy.
LABEL_GRID = 256
PATH_GRID = 96
MIN_COMPONENT_FRACTION = 0.0015


@dataclass(slots=True)
class MaskResult:
    mask: np.ndarray
    method: str
    threshold: float = 0.0
    separability: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.mask.size == 0:
            return 0.0
        return float(self.mask.mean())


@dataclass(slots=True)
class Component:
    label: int
    area_fraction: float
    x: float
    y: float
    w: float
    h: float
    mean_value: float


def otsu_threshold(values: np.ndarray, bins: int = 128) -> tuple[float, float]:
    """Return (threshold, separability).

    The threshold is the upper edge of the low class, so `values >= threshold`
    selects the high class. Separability is the normalised between-class
    variance, which doubles as an honest confidence signal: a bimodal scene
    separates cleanly, a flat one does not.
    """
    flat = values[np.isfinite(values)].ravel()
    if flat.size == 0:
        return 0.5, 0.0
    lo, hi = float(flat.min()), float(flat.max())
    if hi - lo < 1e-6:
        return float(lo), 0.0

    hist, edges = np.histogram(flat, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float((lo + hi) / 2), 0.0

    prob = hist / total
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(denom > 1e-12, (mu_total * omega - mu) ** 2 / denom, 0.0)

    idx = int(np.nanargmax(between))
    variance = float(flat.var())
    separability = float(between[idx] / variance) if variance > 1e-12 else 0.0
    return float(edges[idx + 1]), float(np.clip(separability, 0.0, 1.0))


def _integral(img: np.ndarray) -> np.ndarray:
    padded = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    return padded


def box_mean(img: np.ndarray, radius: int = 2) -> np.ndarray:
    """Uniform filter via integral image (keeps scipy out of the dependency set)."""
    h, w = img.shape
    integ = _integral(img)
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - radius, 0, h)[:, None]
    y1 = np.clip(ys + radius + 1, 0, h)[:, None]
    x0 = np.clip(xs - radius, 0, w)[None, :]
    x1 = np.clip(xs + radius + 1, 0, w)[None, :]
    total = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
    count = (y1 - y0) * (x1 - x0)
    return (total / np.maximum(count, 1)).astype(np.float32)


def speckle_index(img: np.ndarray, radius: int = 2) -> float:
    """Coefficient of variation. SAR amplitude speckles far more than optical."""
    mean = box_mean(img, radius)
    sq_mean = box_mean(img * img, radius)
    var = np.clip(sq_mean - mean * mean, 0.0, None)
    valid = mean > 1e-3
    if not np.any(valid):
        return 0.0
    return float(np.median(np.sqrt(var[valid]) / mean[valid]))


def _normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.zeros_like(a, dtype=np.float32)
    np.divide(a - b, denom, out=out, where=np.abs(denom) > 1e-6)
    return np.clip(out, -1.0, 1.0)


def _resize_bool(mask: np.ndarray, size: int) -> np.ndarray:
    """Area-average a boolean mask onto a square grid, then re-threshold."""
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=bool)
    ys = (np.linspace(0, h, size + 1)).astype(int)
    xs = (np.linspace(0, w, size + 1)).astype(int)
    integ = _integral(mask.astype(np.float32))
    y0, y1 = ys[:-1][:, None], np.maximum(ys[1:], ys[:-1] + 1)[:, None]
    x0, x1 = xs[:-1][None, :], np.maximum(xs[1:], xs[:-1] + 1)[None, :]
    total = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
    count = (y1 - y0) * (x1 - x0)
    return (total / np.maximum(count, 1)) >= 0.5


def water_mask(scene: Scene, modality: str) -> MaskResult:
    """NDWI where a NIR band exists, SAR low-backscatter otherwise."""
    if modality == "sar" or scene.bands == 1:
        gray = box_mean(scene.refl_gray, radius=2)  # despeckle before thresholding
        threshold, separability = otsu_threshold(gray)
        mask = gray <= threshold
        if mask.mean() > 0.7:  # a dark scene is not an all-water scene
            mask = gray <= np.percentile(gray, 25)
        return MaskResult(
            mask=mask,
            method="sar-low-backscatter-otsu",
            threshold=threshold,
            separability=separability,
            notes=["SAR returns collapse over specular water, so dark pixels carry the signal."],
        )

    nir_index = scene.nir_index
    if nir_index is not None:
        green = scene.refl(1)
        nir = scene.refl(nir_index)
        index = _normalised_difference(green, nir)
        threshold, separability = otsu_threshold(index)
        threshold = max(threshold, 0.02)
        return MaskResult(
            mask=index >= threshold,
            method="ndwi-otsu",
            threshold=threshold,
            separability=separability,
            notes=[f"NDWI computed from band 2 (green) and band {nir_index + 1} (NIR)."],
        )

    # RGB only: water is dark and blue-dominant relative to red.
    blue = scene.refl(2) if scene.bands >= 3 else scene.refl_gray
    red = scene.refl(0)
    index = _normalised_difference(blue, red) - scene.refl_gray * 0.35
    threshold, separability = otsu_threshold(index)
    threshold = max(threshold, -0.05)
    return MaskResult(
        mask=index >= threshold,
        method="rgb-water-proxy-otsu",
        threshold=threshold,
        separability=separability * 0.75,
        notes=["No NIR band present; water proxied from blue-vs-red contrast and low brightness."],
    )


def vegetation_mask(scene: Scene) -> MaskResult:
    nir_index = scene.nir_index
    if nir_index is not None:
        index = _normalised_difference(scene.refl(nir_index), scene.refl(0))
        threshold, separability = otsu_threshold(index)
        return MaskResult(
            mask=index >= max(threshold, 0.1),
            method="ndvi-otsu",
            threshold=threshold,
            separability=separability,
            notes=["NDVI from NIR and red."],
        )
    if scene.bands < 3:
        return MaskResult(mask=np.zeros(scene.gray.shape, dtype=bool), method="unavailable-single-band")
    r, g, b = scene.refl(0), scene.refl(1), scene.refl(2)
    excess_green = np.clip(2.0 * g - r - b, -1.0, 1.0)
    threshold, separability = otsu_threshold(excess_green)
    return MaskResult(
        mask=excess_green >= max(threshold, 0.02),
        method="excess-green-otsu",
        threshold=threshold,
        separability=separability * 0.8,
        notes=["No NIR band; vegetation proxied from excess-green index."],
    )


def builtup_mask(scene: Scene, water: np.ndarray, vegetation: np.ndarray) -> MaskResult:
    """Bright surfaces *within the remaining land*, not within the whole frame.

    Thresholding global brightness would label ordinary bare land as built-up
    once water and vegetation are removed, so the split is computed on the
    residual pixels only. What stays below the threshold is reported as
    other/bare rather than being forced into a class.
    """
    gray = scene.refl_gray  # reflectance keeps the split comparable across dates
    residual = ~water & ~vegetation
    if residual.sum() < 32:
        return MaskResult(
            mask=np.zeros_like(residual),
            method="brightness-residual",
            notes=["Water and vegetation account for the whole frame; no residual to classify."],
        )

    threshold, separability = otsu_threshold(gray[residual])
    mask = residual & (gray >= threshold)
    return MaskResult(
        mask=mask,
        method="brightness-residual-otsu",
        threshold=threshold,
        separability=separability * 0.7,
        notes=[
            "Built-up proxied as the bright class among non-water, non-vegetated pixels; "
            "a fine-tuned classifier is needed to separate built-up from bare rock or sand."
        ],
    )


def _match_radiometry(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Relative radiometric normalisation (median/MAD).

    Two dates are rarely acquired under the same sun angle or atmosphere, so a
    raw difference flags the whole scene as changed. Matching the target's
    robust centre and spread to the reference removes that global offset while
    leaving genuine local change intact.
    """
    ref_median = float(np.median(reference))
    tgt_median = float(np.median(target))
    ref_scale = float(np.median(np.abs(reference - ref_median))) or 1e-3
    tgt_scale = float(np.median(np.abs(target - tgt_median))) or 1e-3
    return (target - tgt_median) * (ref_scale / tgt_scale) + ref_median


def _common_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def change_mask(before: Scene, after: Scene) -> MaskResult:
    """Bi-temporal change from the radiometrically normalised intensity difference."""
    a = box_mean(before.refl_gray, radius=1)
    b = box_mean(after.refl_gray, radius=1)
    a, b = _common_shape(a, b)
    b = _match_radiometry(a, b)

    magnitude = np.abs(b - a)
    threshold, separability = otsu_threshold(magnitude)
    threshold = max(threshold, 0.08)
    mask = magnitude >= threshold
    return MaskResult(
        mask=mask,
        method="bitemporal-difference-otsu",
        threshold=threshold,
        separability=separability,
        notes=[
            "t1 is radiometrically normalised to t0 (median/MAD) before differencing, so "
            "illumination differences are not reported as change.",
            "Change magnitude is |t1 - t0| on despeckled reflectance, thresholded by Otsu.",
        ],
    )


def directional_change(before: Scene, after: Scene, mask: np.ndarray) -> tuple[float, float]:
    """Fraction of changed pixels that brightened vs darkened.

    Brightening on optical pairs is the usual signature of new built-up surface;
    darkening tends to be vegetation loss, shadow or new water.
    """
    a = box_mean(before.refl_gray, radius=1)
    b = box_mean(after.refl_gray, radius=1)
    a, b = _common_shape(a, b)
    b = _match_radiometry(a, b)
    size = (min(a.shape[0], mask.shape[0]), min(a.shape[1], mask.shape[1]))
    a, b, m = a[: size[0], : size[1]], b[: size[0], : size[1]], mask[: size[0], : size[1]]
    if not np.any(m):
        return 0.0, 0.0
    delta = (b - a)[m]
    gained = float(np.mean(delta > 0))
    return gained, 1.0 - gained


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) spans of True in a boolean row."""
    padded = np.concatenate(([False], row, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist(), strict=True))


def label_components(mask: np.ndarray, min_fraction: float = MIN_COMPONENT_FRACTION) -> list[Component]:
    """Connected components (4-connectivity) via run-length union-find.

    Runs rather than pixels: a scene has thousands of runs but hundreds of
    thousands of pixels, which keeps the Python-level loop off the hot path.
    """
    grid = _resize_bool(mask, LABEL_GRID) if max(mask.shape) > LABEL_GRID else mask
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]  # index 0 is the background sentinel

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    previous: list[tuple[int, int, int]] = []  # (start, end, label)
    for y in range(h):
        current: list[tuple[int, int, int]] = []
        for start, end in _row_runs(grid[y]):
            label = 0
            for prev_start, prev_end, prev_label in previous:
                if prev_start >= end:
                    break
                if prev_end <= start:
                    continue
                if label == 0:
                    label = prev_label
                else:
                    union(label, prev_label)
            if label == 0:
                label = len(parent)
                parent.append(label)
            labels[y, start:end] = label
            current.append((start, end, label))
        previous = current

    if len(parent) == 1:
        return []

    lookup = np.array([find(label) for label in range(len(parent))], dtype=np.int32)
    resolved = lookup[labels]

    flat = resolved.ravel()
    areas = np.bincount(flat, minlength=len(parent)).astype(np.float64)
    total = float(h * w)
    min_area = max(4.0, min_fraction * total)

    ys, xs = np.nonzero(resolved)
    if ys.size == 0:
        return []
    owners = resolved[ys, xs]
    min_x = np.full(len(parent), w, dtype=np.int64)
    max_x = np.full(len(parent), -1, dtype=np.int64)
    min_y = np.full(len(parent), h, dtype=np.int64)
    max_y = np.full(len(parent), -1, dtype=np.int64)
    np.minimum.at(min_x, owners, xs)
    np.maximum.at(max_x, owners, xs)
    np.minimum.at(min_y, owners, ys)
    np.maximum.at(max_y, owners, ys)

    components: list[Component] = []
    for label in np.unique(owners):
        area = areas[label]
        if area < min_area:
            continue
        box_w = int(max_x[label] - min_x[label] + 1)
        box_h = int(max_y[label] - min_y[label] + 1)
        components.append(
            Component(
                label=int(label),
                area_fraction=area / total,
                x=float(min_x[label]) / w * 100.0,
                y=float(min_y[label]) / h * 100.0,
                w=float(box_w) / w * 100.0,
                h=float(box_h) / h * 100.0,
                # Compactness: how much of its own bounding box the blob fills.
                mean_value=float(area) / max(1.0, box_w * box_h),
            )
        )

    components.sort(key=lambda c: c.area_fraction, reverse=True)
    return components


def mask_to_svg_path(mask: np.ndarray, grid: int = PATH_GRID) -> str:
    """Encode a boolean mask as an SVG path in the UI's 0..100 viewBox."""
    if mask.size == 0 or not np.any(mask):
        return ""
    cells = _resize_bool(mask, grid)
    step = 100.0 / grid
    parts: list[str] = []
    for row in range(grid):
        line = cells[row]
        x = 0
        while x < grid:
            if not line[x]:
                x += 1
                continue
            start = x
            while x < grid and line[x]:
                x += 1
            x0 = start * step
            x1 = x * step
            y0 = row * step
            y1 = (row + 1) * step
            parts.append(
                f"M{x0:.2f} {y0:.2f}H{x1:.2f}V{y1:.2f}H{x0:.2f}Z"
            )
    return "".join(parts)


def coverage_percent(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean() * 100.0)


def area_km2(mask: np.ndarray, gsd_meters: float | None, native_width: int, native_height: int) -> float | None:
    """Scale a coverage fraction back to native resolution to get real ground area."""
    if not gsd_meters or gsd_meters <= 0 or mask.size == 0:
        return None
    pixels = mask.mean() * native_width * native_height
    return float(pixels * gsd_meters * gsd_meters / 1_000_000.0)


def confidence_from(separability: float, coverage: float, evidence_count: int = 0) -> float:
    """Explainable confidence: threshold separability, penalised when a mask
    degenerates to almost-nothing or almost-everything, nudged by evidence."""
    base = 0.45 + 0.4 * float(np.clip(separability, 0.0, 1.0))
    fraction = coverage / 100.0
    if fraction < 0.005 or fraction > 0.9:
        base -= 0.12
    elif 0.02 <= fraction <= 0.6:
        base += 0.05
    if evidence_count:
        base += min(0.05, 0.01 * evidence_count)
    return round(float(np.clip(base, 0.2, 0.97)), 3)
