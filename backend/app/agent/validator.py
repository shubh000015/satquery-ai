"""Input validator — the second box of the architecture diagram.

Checks count, format, modality and pair compatibility before any specialist runs,
so the controller can refuse impossible work instead of inventing an answer.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.schemas.imagery import (
    Asset,
    InputMode,
    ValidationIssue,
    ValidationReport,
)

_DIMENSION_TOLERANCE = 0.02  # 2% edge mismatch is tolerable for co-registration
_ASPECT_TOLERANCE = 0.05
_GSD_RATIO_LIMIT = 2.0
_OVERLAP_ERROR = 40.0
_OVERLAP_WARNING = 90.0


def infer_mode(assets: list[Asset], hint: InputMode | None = None) -> InputMode:
    """Cross-modal when the two inputs are different sensor families,
    bi-temporal when they are the same. An explicit user hint always wins."""
    if len(assets) < 2:
        return "single"
    if hint in ("cross-modal", "bi-temporal"):
        return hint
    first, second = assets[0].modality, assets[1].modality
    optical_family = {"optical", "multispectral"}
    if (first == "sar") != (second == "sar"):
        return "cross-modal"
    if first in optical_family and second in optical_family:
        return "bi-temporal"
    return "bi-temporal"


def _bounds_overlap_percent(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = ix * iy
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    if smaller <= 0:
        return 0.0
    return min(100.0, intersection / smaller * 100.0)


def _check_formats(assets: list[Asset], settings: Settings) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for asset in assets:
        if not asset.benchmark_only_format:
            continue
        message = (
            f"{asset.name} is {asset.format}. PNG/JPEG are accepted only for the "
            "prescribed public benchmark datasets; operational imagery must be GeoTIFF/TIFF."
        )
        if asset.benchmark_dataset:
            issues.append(
                ValidationIssue(
                    code="benchmark-format",
                    severity="info",
                    message=f"{asset.name} accepted as a {asset.benchmark_dataset} benchmark sample.",
                    asset_id=asset.id,
                )
            )
        elif settings.strict_format_policy:
            issues.append(
                ValidationIssue(
                    code="format-not-allowed",
                    severity="error",
                    message=message,
                    hint="Re-upload as GeoTIFF, or declare the benchmark dataset this chip comes from.",
                    asset_id=asset.id,
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    code="format-benchmark-only",
                    severity="warning",
                    message=message,
                    hint="Declare the benchmark dataset, or supply GeoTIFF for operational runs.",
                    asset_id=asset.id,
                )
            )

        if not asset.meta or not asset.meta.georeferenced:
            issues.append(
                ValidationIssue(
                    code="no-geotransform",
                    severity="warning",
                    message=f"{asset.name} carries no geotransform, so areas cannot be reported in km².",
                    hint="GeoTIFF with CRS and pixel scale enables ground-area metrics.",
                    asset_id=asset.id,
                )
            )
    return issues


def _check_pair(assets: list[Asset], mode: InputMode) -> tuple[list[ValidationIssue], bool | None, float | None]:
    issues: list[ValidationIssue] = []
    first, second = assets[0], assets[1]
    meta_a, meta_b = first.meta, second.meta
    co_registered: bool | None = None
    overlap: float | None = None

    if meta_a and meta_b:
        width_delta = abs(meta_a.width - meta_b.width) / max(meta_a.width, meta_b.width)
        height_delta = abs(meta_a.height - meta_b.height) / max(meta_a.height, meta_b.height)
        aspect_a = meta_a.width / max(1, meta_a.height)
        aspect_b = meta_b.width / max(1, meta_b.height)
        aspect_delta = abs(aspect_a - aspect_b) / max(aspect_a, aspect_b)

        if aspect_delta > _ASPECT_TOLERANCE:
            issues.append(
                ValidationIssue(
                    code="aspect-mismatch",
                    severity="error",
                    message=(
                        f"Aspect ratios differ ({aspect_a:.2f} vs {aspect_b:.2f}); the pair does not "
                        "cover the same footprint."
                    ),
                    hint="Supply a co-registered pair clipped to a common footprint.",
                )
            )
            co_registered = False
        elif max(width_delta, height_delta) > _DIMENSION_TOLERANCE:
            issues.append(
                ValidationIssue(
                    code="dimension-mismatch",
                    severity="warning",
                    message=(
                        f"Pixel dimensions differ by {max(width_delta, height_delta) * 100:.1f}%; "
                        "the pair will be resampled to the common overlap."
                    ),
                )
            )

        if meta_a.crs and meta_b.crs and meta_a.crs != meta_b.crs:
            issues.append(
                ValidationIssue(
                    code="crs-mismatch",
                    severity="error",
                    message=f"CRS mismatch: {meta_a.crs} vs {meta_b.crs}.",
                    hint="Reproject both scenes to a common CRS before analysis.",
                )
            )
            co_registered = False

        if meta_a.bounds and meta_b.bounds:
            overlap = _bounds_overlap_percent(meta_a.bounds, meta_b.bounds)
            if overlap < _OVERLAP_ERROR:
                issues.append(
                    ValidationIssue(
                        code="insufficient-overlap",
                        severity="error",
                        message=f"Footprints overlap only {overlap:.1f}%.",
                        hint="Joint reasoning needs the same geographic area in both scenes.",
                    )
                )
                co_registered = False
            elif overlap < _OVERLAP_WARNING:
                issues.append(
                    ValidationIssue(
                        code="partial-overlap",
                        severity="warning",
                        message=f"Footprints overlap {overlap:.1f}%; analysis is limited to the shared area.",
                    )
                )
                co_registered = co_registered if co_registered is False else True
            elif co_registered is not False:
                co_registered = True
        elif co_registered is None:
            issues.append(
                ValidationIssue(
                    code="coregistration-unverified",
                    severity="warning",
                    message="No geotransform on the pair, so co-registration is assumed, not verified.",
                    hint="Georeferenced GeoTIFF inputs let the validator prove footprint overlap.",
                )
            )

        gsd_a = meta_a.gsd_meters or 0.0
        gsd_b = meta_b.gsd_meters or 0.0
        if gsd_a > 0 and gsd_b > 0:
            ratio = max(gsd_a, gsd_b) / min(gsd_a, gsd_b)
            if ratio > _GSD_RATIO_LIMIT:
                issues.append(
                    ValidationIssue(
                        code="gsd-mismatch",
                        severity="warning",
                        message=f"Resolutions differ {ratio:.1f}× ({gsd_a:.1f} m vs {gsd_b:.1f} m).",
                        hint="The coarser scene limits the effective detail of joint outputs.",
                    )
                )

    same_family = (first.modality == "sar") == (second.modality == "sar")
    if mode == "cross-modal" and same_family:
        issues.append(
            ValidationIssue(
                code="not-cross-modal",
                severity="warning",
                message=(
                    f"Both inputs look {first.modality}; cross-modal fusion expects one optical/MSI "
                    "and one SAR scene."
                ),
                hint="Check the pair, or run this as a bi-temporal comparison instead.",
            )
        )
    if mode == "bi-temporal" and not same_family:
        issues.append(
            ValidationIssue(
                code="mixed-modality-pair",
                severity="warning",
                message=(
                    f"Pair mixes {first.modality} and {second.modality}; change detection across "
                    "sensors is unreliable."
                ),
                hint="Use same-sensor dates for change, or route this as optical–SAR fusion.",
            )
        )
    if mode == "bi-temporal" and first.date == second.date:
        issues.append(
            ValidationIssue(
                code="same-date",
                severity="warning",
                message=f"Both scenes report {first.date}; acquisition dates could not be separated.",
                hint="Filenames with dates (…_20240415…) let the trace label t0 and t1 properly.",
            )
        )

    return issues, co_registered, overlap


def validate(
    assets: list[Asset],
    mode_hint: InputMode | None = None,
    settings: Settings | None = None,
) -> ValidationReport:
    settings = settings or get_settings()
    mode = infer_mode(assets, mode_hint)
    issues: list[ValidationIssue] = []
    co_registered: bool | None = None
    overlap: float | None = None

    if not assets:
        issues.append(
            ValidationIssue(
                code="no-input",
                severity="error",
                message="No imagery supplied.",
                hint="Upload one scene, or a co-registered/bi-temporal pair.",
            )
        )
    elif len(assets) > settings.max_assets_per_query:
        issues.append(
            ValidationIssue(
                code="too-many-inputs",
                severity="error",
                message=(
                    f"{len(assets)} scenes supplied; the defined input scope is a single image "
                    f"or a pair (max {settings.max_assets_per_query})."
                ),
            )
        )

    issues.extend(_check_formats(assets, settings))

    if len(assets) == 2:
        pair_issues, co_registered, overlap = _check_pair(assets, mode)
        issues.extend(pair_issues)

    ok = not any(issue.severity == "error" for issue in issues)
    if not assets:
        summary = "No input to validate."
    elif len(assets) == 1:
        asset = assets[0]
        summary = f"Single {asset.modality} frame · {asset.format} · {asset.gsd}"
        if asset.crs:
            summary += f" · {asset.crs}"
    else:
        summary = (
            f"{mode} pair · {assets[0].modality}/{assets[1].modality} · "
            f"{assets[0].format}/{assets[1].format}"
        )
        if overlap is not None:
            summary += f" · overlap {overlap:.1f}%"

    return ValidationReport(
        ok=ok,
        mode=mode,
        asset_count=len(assets),
        issues=issues,
        co_registered=co_registered,
        overlap_percent=overlap,
        summary=summary,
    )
