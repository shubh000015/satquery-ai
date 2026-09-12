"""Tests for the band contract and the model registry. No weights, no GPU.

Run from ml/:   python -m pytest tests -q
or standalone:  python tests/test_bands_registry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery_ml import registry  # noqa: E402
from satquery_ml.bands import (  # noqa: E402
    BEN_ALL_BANDS,
    CANONICAL_BANDS,
    MissingBands,
    RGB_BANDS,
    BandStack,
    resize_chw,
)


def _stack(names: tuple[str, ...], size: int = 8) -> BandStack:
    rng = np.random.default_rng(0)
    array = rng.random((size, size, len(names)), dtype=np.float32)
    return BandStack(array=array, names=names)


# --------------------------------------------------------------------------
# BandStack validation
# --------------------------------------------------------------------------

def test_rejects_channel_name_mismatch():
    try:
        BandStack(array=np.zeros((4, 4, 2), dtype=np.float32), names=("VV",))
    except ValueError as exc:
        assert "2 channels but 1 band names" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rejects_duplicate_band_names():
    try:
        BandStack(array=np.zeros((4, 4, 2), dtype=np.float32), names=("VV", "VV"))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rejects_non_3d_array():
    try:
        BandStack(array=np.zeros((4, 4), dtype=np.float32), names=("VV",))
    except ValueError as exc:
        assert "(H, W, C)" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------
# modality inference
# --------------------------------------------------------------------------

def test_modality_is_read_from_bands_not_guessed():
    assert _stack(("VV", "VH")).modality == "sar"
    assert _stack(RGB_BANDS).modality == "optical"
    assert _stack(("VV", "VH") + RGB_BANDS).modality == "optical-sar"
    assert _stack(BEN_ALL_BANDS).modality == "optical-sar"


def test_has_and_missing():
    stack = _stack(("VV", "VH", "B04"))
    assert stack.has("VV", "VH")
    assert not stack.has("B08")
    assert stack.missing(("B04", "B08", "B11")) == ("B08", "B11")
    assert stack.missing(("VV",)) == ()


# --------------------------------------------------------------------------
# select: ordering is the thing that silently breaks models
# --------------------------------------------------------------------------

def test_select_returns_requested_order_not_storage_order():
    # Stored canonically (VV, VH, B02, B03, B04); ask for a different order.
    stack = _stack(("VV", "VH", "B02", "B03", "B04"))
    wanted = ("B04", "B02", "VH")
    selected = stack.select(wanted)

    assert selected.shape == (3, 8, 8)
    for i, band in enumerate(wanted):
        expected = stack.array[:, :, stack.names.index(band)]
        assert np.array_equal(selected[i], expected), band


def test_select_raises_with_a_useful_message_when_bands_are_absent():
    stack = _stack(RGB_BANDS)
    try:
        stack.select(BEN_ALL_BANDS, model="BigEarthNet ResNet-50")
    except MissingBands as exc:
        assert exc.model == "BigEarthNet ResNet-50"
        assert "VV" in exc.missing and "B08" in exc.missing
        # The point: it refuses rather than zero-filling channels.
        assert "does not carry" in str(exc)
    else:
        raise AssertionError("expected MissingBands")


def test_select_is_float32():
    stack = BandStack(array=np.ones((4, 4, 2), dtype=np.float64), names=("VV", "VH"))
    assert stack.select(("VV",)).dtype == np.float32


# --------------------------------------------------------------------------
# rgb rendering
# --------------------------------------------------------------------------

def test_rgb_from_optical_uses_b04_b03_b02():
    stack = _stack(("B02", "B03", "B04"))
    rgb = stack.rgb()
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8


def test_rgb_from_sar_only_builds_the_false_colour_composite():
    stack = _stack(("VV", "VH"))
    rgb = stack.rgb()
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8


def test_rgb_survives_a_flat_scene():
    # A constant scene makes the percentile stretch degenerate; must not divide by zero.
    stack = BandStack(array=np.full((4, 4, 3), 0.5, dtype=np.float32), names=RGB_BANDS)
    rgb = stack.rgb()
    assert rgb.shape == (4, 4, 3)
    assert not np.isnan(rgb).any()


# --------------------------------------------------------------------------
# wire format
# --------------------------------------------------------------------------

def test_encode_decode_round_trip_is_exact():
    stack = _stack(BEN_ALL_BANDS, size=16)
    payload = stack.encode()

    assert payload["bands"] == list(BEN_ALL_BANDS)
    assert payload["height"] == 16 and payload["width"] == 16
    assert isinstance(payload["npz"], str)

    restored = BandStack.decode(payload)
    assert restored.names == stack.names
    assert np.array_equal(restored.array, stack.array)


def test_encoded_payload_is_smaller_than_naive_json():
    import json

    stack = _stack(BEN_ALL_BANDS, size=120)
    encoded = len(stack.encode()["npz"])
    naive = len(json.dumps(stack.array.tolist()))
    assert encoded < naive


# --------------------------------------------------------------------------
# construction helpers
# --------------------------------------------------------------------------

def test_from_rgb_labels_channels_honestly():
    rgb = (np.random.default_rng(1).random((10, 10, 3)) * 255).astype(np.uint8)
    stack = BandStack.from_rgb(rgb)

    assert stack.names == RGB_BANDS
    assert stack.array.max() <= 1.0
    assert stack.modality == "optical"
    # The honesty test: a 12-band model must refuse this, not consume red as NIR.
    try:
        stack.select(BEN_ALL_BANDS, model="BEN")
    except MissingBands:
        pass
    else:
        raise AssertionError("an RGB stack must not satisfy a 12-band request")


def test_from_rgb_accepts_float_and_grayscale():
    floats = np.random.default_rng(2).random((6, 6, 3)).astype(np.float32)
    assert BandStack.from_rgb(floats).array.max() <= 1.0

    gray = np.full((6, 6), 128, dtype=np.uint8)
    stack = BandStack.from_rgb(gray)
    assert stack.array.shape == (6, 6, 3)


def test_from_planes_orders_canonically_regardless_of_insertion_order():
    planes = {
        "B04": np.ones((4, 4), dtype=np.float32),
        "VV": np.zeros((4, 4), dtype=np.float32),
        "B02": np.full((4, 4), 0.5, dtype=np.float32),
    }
    stack = BandStack.from_planes(planes)
    assert stack.names == ("VV", "B02", "B04")
    assert np.array_equal(stack.array[:, :, 0], planes["VV"])


def test_from_planes_rejects_unknown_bands_and_shape_mismatch():
    try:
        BandStack.from_planes({"B99": np.zeros((4, 4), dtype=np.float32)})
    except ValueError as exc:
        assert "unknown band names" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown band")

    try:
        BandStack.from_planes({
            "VV": np.zeros((4, 4), dtype=np.float32),
            "B04": np.zeros((8, 8), dtype=np.float32),
        })
    except ValueError as exc:
        assert "share one shape" in str(exc)
    else:
        raise AssertionError("expected ValueError for shape mismatch")


def test_subset_keeps_names_aligned():
    stack = _stack(BEN_ALL_BANDS)
    subset = stack.subset(RGB_BANDS)
    assert subset.names == RGB_BANDS
    assert subset.array.shape == (8, 8, 3)
    assert np.array_equal(subset.array[:, :, 0], stack.array[:, :, stack.names.index("B04")])


def test_for_croma_fills_b01_and_b09_from_a_bigearthnet_stack():
    stack = _stack(BEN_ALL_BANDS)
    ready, synthesized = stack.for_croma()
    assert synthesized == ("B01", "B09")
    assert ready.has("B01", "B09", "VV", "B08")
    assert np.array_equal(
        ready.array[:, :, ready.names.index("B01")],
        stack.array[:, :, stack.names.index("B02")],
    )
    assert np.array_equal(
        ready.array[:, :, ready.names.index("B09")],
        stack.array[:, :, stack.names.index("B8A")],
    )


def test_for_croma_refuses_an_rgb_stack():
    try:
        _stack(RGB_BANDS).for_croma()
    except MissingBands as exc:
        assert "VV" in exc.missing
    else:
        raise AssertionError("RGB cannot satisfy CROMA even with atmospheric fill")


def test_resize_preserves_channel_count():
    array = np.random.default_rng(3).random((12, 30, 30)).astype(np.float32)
    resized = resize_chw(array, 120)
    assert resized.shape == (12, 120, 120)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_every_spec_requests_real_bands():
    for key, model in registry.REGISTRY.items():
        unknown = set(model.bands) - set(CANONICAL_BANDS)
        assert not unknown, f"{key} wants unknown bands {unknown}"
        for name, extra in model.extra_bands.items():
            assert not set(extra) - set(CANONICAL_BANDS), f"{key}.{name}"


def test_task_plans_reference_real_models():
    for task in registry.TASK_PLANS:
        models = registry.plan_for(task)
        assert models, task
        for model in models:
            assert task in model.tasks or model.key == "vlm", (task, model.key)


def test_mandatory_tasks_are_all_planned():
    # The problem statement makes these mandatory.
    for task in ("vqa", "caption", "grounding", "change", "fusion"):
        assert task in registry.TASK_PLANS


def test_unknown_keys_raise_clearly():
    try:
        registry.spec("nope")
    except KeyError as exc:
        assert "unknown model" in str(exc)
    else:
        raise AssertionError("expected KeyError")

    try:
        registry.plan_for("segmentation")
    except KeyError as exc:
        assert "unknown task" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_provenance_is_filled_in_for_every_model():
    for key, model in registry.REGISTRY.items():
        block = model.provenance()
        assert block["model"] and block["source"], key
        assert block["trainedBy"], key
        # Nothing here is ours; if that changes the field must change too.
        assert block["trainedBy"] != "us", key


def test_vram_fits_two_t4s():
    totals = registry.vram_by_slot()
    assert set(totals) == {registry.SLOT_VLM, registry.SLOT_SPECIALIST}
    for slot, gigabytes in totals.items():
        assert gigabytes < 15.0, f"{slot} needs {gigabytes} GB, a T4 has 16"


def test_provenance_table_renders_every_model():
    table = registry.provenance_table()
    assert table.startswith("| Component |")
    for model in registry.REGISTRY.values():
        assert model.name in table
        assert model.trained_by in table


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            passed += 1
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
