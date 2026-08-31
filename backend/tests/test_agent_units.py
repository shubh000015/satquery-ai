"""Unit tests for the router, validator and analysis primitives."""

from __future__ import annotations

import numpy as np
import pytest

from app.agent import router, validator
from app.core.config import get_settings
from app.schemas.imagery import Asset, RasterMeta
from app.services import analysis


def _asset(
    asset_id: str,
    modality: str = "optical",
    *,
    role: str = "primary",
    width: int = 200,
    height: int = 200,
    crs: str | None = "EPSG:4326",
    bounds: list[float] | None = None,
    gsd: float | None = 10.0,
    fmt: str = "GeoTIFF",
    date: str = "15 Apr 2024",
    benchmark_only: bool = False,
) -> Asset:
    return Asset(
        id=asset_id,
        role=role,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        name=f"{asset_id}.tif",
        src=f"/api/assets/{asset_id}/preview",
        sensor="test",
        date=date,
        gsd="10 m",
        location="Georeferenced scene",
        coords="28.5° N, 77.0° E",
        format=fmt,
        crs=crs,
        benchmark_only_format=benchmark_only,
        meta=RasterMeta(
            width=width,
            height=height,
            bands=4,
            dtype="uint8",
            georeferenced=crs is not None,
            crs=crs,
            gsd_meters=gsd,
            bounds=bounds if bounds is not None else [77.0, 28.0, 77.1, 28.1],
        ),
    )


@pytest.mark.parametrize(
    ("query", "mode", "expected"),
    [
        ("Describe the land-cover and major objects visible in this image.", "single", "caption"),
        ("Highlight the water body referred to in the query.", "single", "grounding"),
        ("What changed between these two dates, and where did the change occur?", "bi-temporal", "change"),
        ("Has the built-up area increased, decreased, or remained unchanged?", "bi-temporal", "change-vqa"),
        ("Use the optical and SAR images together to identify built-up and water-covered regions.", "cross-modal", "cross-modal"),
        ("Measure the area of the largest water body.", "single", "measure"),
        ("Is agriculture the dominant class in this scene?", "single", "vqa"),
    ],
)
def test_router_picks_expected_task(query: str, mode: str, expected: str) -> None:
    decision = router.classify(query, mode)  # type: ignore[arg-type]
    assert decision.intent.task == expected


def test_router_warns_when_change_query_lacks_a_pair() -> None:
    decision = router.classify("What changed between these two dates?", "single")
    assert decision.intent.warning is not None
    # ...and the controller degrades it to something the input can answer.
    assert router.degrade(decision.intent.task, "single") == "vqa"


def test_router_resolves_target_class() -> None:
    assert router.resolve_target("Highlight the largest water body") == "water"
    assert router.resolve_target("Has the built-up area increased?") == "builtup"
    assert router.resolve_target("How many ships are berthed?") == "ship"
    assert router.resolve_target("Tell me about this scene") == "generic"


def test_validator_infers_cross_modal_from_mixed_modalities() -> None:
    report = validator.validate([_asset("a", "optical"), _asset("b", "sar", role="secondary")])
    assert report.mode == "cross-modal"
    assert report.ok


def test_validator_infers_bitemporal_from_same_modality() -> None:
    report = validator.validate(
        [_asset("a", "optical", date="15 Jan 2020"), _asset("b", "optical", role="secondary")]
    )
    assert report.mode == "bi-temporal"


def test_validator_rejects_non_overlapping_footprints() -> None:
    far_away = _asset("b", "optical", role="secondary", bounds=[90.0, 10.0, 90.1, 10.1])
    report = validator.validate([_asset("a", "optical"), far_away])
    assert not report.ok
    assert any(issue.code == "insufficient-overlap" for issue in report.issues)


def test_validator_rejects_crs_mismatch() -> None:
    report = validator.validate(
        [_asset("a", crs="EPSG:4326"), _asset("b", role="secondary", crs="EPSG:32644")]
    )
    assert not report.ok
    assert any(issue.code == "crs-mismatch" for issue in report.issues)


def test_validator_rejects_more_than_two_inputs() -> None:
    report = validator.validate([_asset("a"), _asset("b"), _asset("c")])
    assert not report.ok
    assert any(issue.code == "too-many-inputs" for issue in report.issues)


def test_benchmark_format_policy_is_a_warning_by_default() -> None:
    report = validator.validate([_asset("a", fmt="PNG", benchmark_only=True)])
    assert report.ok
    assert any(issue.code == "format-benchmark-only" for issue in report.issues)


def test_benchmark_format_policy_can_be_strict() -> None:
    settings = get_settings().model_copy(update={"strict_format_policy": True})
    report = validator.validate([_asset("a", fmt="PNG", benchmark_only=True)], None, settings)
    assert not report.ok
    assert any(issue.code == "format-not-allowed" for issue in report.issues)


def test_otsu_separates_a_bimodal_signal() -> None:
    values = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    threshold, separability = analysis.otsu_threshold(values)
    # `values >= threshold` must select exactly the high class.
    assert (values >= threshold).sum() == 500
    assert separability > 0.9

    flat = np.full(1000, 0.5)
    _threshold, flat_separability = analysis.otsu_threshold(flat)
    assert flat_separability == pytest.approx(0.0, abs=1e-6)


def test_label_components_finds_disjoint_blobs() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[60:90, 60:95] = True
    components = analysis.label_components(mask)
    assert len(components) == 2
    # Sorted largest first, and boxes are percentages of the frame.
    assert components[0].area_fraction > components[1].area_fraction
    assert 55 < components[0].x < 65


def test_mask_to_svg_path_is_empty_for_empty_mask() -> None:
    assert analysis.mask_to_svg_path(np.zeros((50, 50), dtype=bool)) == ""
    path = analysis.mask_to_svg_path(np.ones((50, 50), dtype=bool))
    assert path.startswith("M") and path.endswith("Z")


def test_area_km2_scales_with_gsd() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[:50, :] = True  # half the frame
    area = analysis.area_km2(mask, gsd_meters=10.0, native_width=1000, native_height=1000)
    assert area == pytest.approx(50.0, rel=1e-6)  # 1000*1000*100 m² * 0.5


def test_confidence_penalises_degenerate_masks() -> None:
    solid = analysis.confidence_from(0.9, coverage=30.0)
    degenerate = analysis.confidence_from(0.9, coverage=0.1)
    assert solid > degenerate
