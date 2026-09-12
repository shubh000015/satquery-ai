"""Tests for the inference stack that need no weights and no GPU.

What this can prove locally: the evidence contract, the grounding-prompt rules,
change measurement, fusion composition, device planning, and that the server's
endpoints wire together and degrade correctly when a model is unavailable.

What only Kaggle can prove: that each checkpoint downloads and produces sensible
predictions. Those paths are exercised here with stubs, so a signature change
in an adapter breaks a test rather than surfacing live on stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery_ml import registry  # noqa: E402
from satquery_ml.adapters import change as change_module  # noqa: E402
from satquery_ml.adapters import fusion as fusion_module  # noqa: E402
from satquery_ml.adapters.base import (  # noqa: E402
    Adapter,
    AdapterUnavailable,
    batch_tensor,
    module_dtype,
    move_batch,
    place_module,
)
from satquery_ml.adapters.grounding import _encode_rle, build_prompt  # noqa: E402
from satquery_ml.bands import BandStack  # noqa: E402
from satquery_ml.facts import Evidence, clamp_confidence, is_yes_no  # noqa: E402
from satquery_ml.loader import ModelLoader, plan_devices  # noqa: E402


# --------------------------------------------------------------------------
# Dtype alignment — the Kaggle crash was "expected BFloat16 but found Float"
# --------------------------------------------------------------------------

def test_batch_tensor_matches_a_bf16_module_so_conv_does_not_crash():
    import torch

    conv = torch.nn.Conv2d(3, 4, kernel_size=1).to(dtype=torch.bfloat16)
    array = np.ones((3, 8, 8), dtype=np.float32)
    tensor = batch_tensor(array, "cpu", conv)

    assert tensor.dtype == torch.bfloat16
    assert tensor.shape == (1, 3, 8, 8)
    # This is the exact failure mode from the notebook: float32 batch into bf16 weights.
    output = conv(tensor)
    assert output.dtype == torch.bfloat16


def test_move_batch_casts_pixels_but_leaves_token_ids_alone():
    import torch

    conv = torch.nn.Conv2d(3, 4, kernel_size=1).to(dtype=torch.bfloat16)
    batch = {
        "pixel_values": torch.ones(1, 3, 8, 8, dtype=torch.float32),
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    moved = move_batch(batch, "cpu", conv)

    assert moved["pixel_values"].dtype == torch.bfloat16
    assert moved["input_ids"].dtype == torch.long
    assert moved["attention_mask"].dtype == torch.long
    # Same crash the notebook hit: float32 pixels into a bf16 conv.
    conv(moved["pixel_values"])


def test_place_module_can_force_float32_even_when_weights_are_bf16():
    import torch

    conv = torch.nn.Conv2d(3, 4, kernel_size=1).to(dtype=torch.bfloat16)
    placed, dtype = place_module(conv, "cpu", dtype=torch.float32)
    assert dtype == torch.float32
    assert module_dtype(placed) == torch.float32
    tensor = batch_tensor(np.ones((3, 8, 8), dtype=np.float32), "cpu", placed)
    assert placed(tensor).dtype == torch.float32


def test_place_module_casts_bf16_weights_to_the_compute_dtype():
    import torch

    conv = torch.nn.Conv2d(3, 4, kernel_size=1).to(dtype=torch.bfloat16)
    placed, dtype = place_module(conv, "cpu")
    assert dtype == torch.float32
    assert module_dtype(placed) == torch.float32
    tensor = batch_tensor(np.ones((3, 8, 8), dtype=np.float32), "cpu", placed)
    assert placed(tensor).dtype == torch.float32


# --------------------------------------------------------------------------
# Evidence contract
# --------------------------------------------------------------------------

def test_one_word_prefers_the_verdict():
    evidence = Evidence(task="vqa", terse="wetland", verdict="no")
    assert evidence.one_word == "no"
    assert Evidence(task="vqa", terse="wetland").one_word == "wetland"


def test_fact_block_marks_the_verdict_as_binding():
    evidence = Evidence(
        task="vqa",
        question="Is there water?",
        findings=["inland water is present (score 0.91)"],
        verdict="yes",
        confidence=0.91,
    )
    block = evidence.fact_block()
    assert "must be preserved" in block
    assert "inland water is present" in block
    assert "Is there water?" in block


def test_fact_block_never_omits_measurements():
    evidence = Evidence(task="change", measurements={"changed area": "12.50% of the frame"})
    assert "12.50% of the frame" in evidence.fact_block()


def test_template_answer_works_without_a_language_model():
    evidence = Evidence(
        task="vqa",
        findings=["no built-up area detected (score 0.04)"],
        verdict="no",
        confidence=0.96,
    )
    answer = evidence.template_answer()
    assert answer.startswith("No.")
    assert "0.96" in answer


def test_as_dict_carries_the_keys_the_backend_reads():
    evidence = Evidence(task="vqa", verdict="yes", confidence=0.8, terse="sea")
    body = evidence.as_dict()
    for key in ("task", "oneWord", "verdict", "confidence", "findings", "models"):
        assert key in body


def test_add_model_does_not_duplicate():
    evidence = Evidence(task="vqa")
    evidence.add_model(registry.spec("landcover"))
    evidence.add_model(registry.spec("landcover"))
    assert len(evidence.models) == 1
    assert evidence.models[0]["trainedBy"] == "BIFOLD / TU Berlin"


def test_confidence_is_clamped_away_from_absolute_certainty():
    assert clamp_confidence(1.0) == 0.97
    assert clamp_confidence(0.0) == 0.15
    assert clamp_confidence(0.5) == 0.5


def test_yes_no_detection():
    assert is_yes_no("Is there water here?")
    assert is_yes_no("Any urban areas?")
    assert not is_yes_no("Describe the land cover.")
    assert not is_yes_no("What changed between these dates?")


# --------------------------------------------------------------------------
# Grounding prompt rules
# --------------------------------------------------------------------------

def test_prompt_is_lowercase_and_period_separated():
    # Grounding DINO silently loses recall if this format is wrong.
    for query in ("Highlight the water body", "Show me BUILDINGS", "find the road"):
        prompt = build_prompt(query)
        assert prompt == prompt.lower(), prompt
        assert prompt.endswith("."), prompt


def test_prompt_maps_domain_phrasing_to_detectable_nouns():
    assert "lake" in build_prompt("Highlight the water body referred to in the query")
    assert "building" in build_prompt("Where are the built-up areas?")
    assert "runway" in build_prompt("Find the airport")


def test_prompt_strips_instruction_words():
    prompt = build_prompt("please highlight the region in this image")
    # "highlight", "region", "image" are instructions, not objects.
    for word in ("highlight", "region", "image"):
        assert word not in prompt


def test_prompt_never_returns_empty():
    assert build_prompt("").endswith(".")
    assert build_prompt("show me the").endswith(".")


# --------------------------------------------------------------------------
# Mask RLE
# --------------------------------------------------------------------------

def test_rle_round_trips():
    rng = np.random.default_rng(0)
    mask = rng.random((16, 16)) > 0.5
    runs = _encode_rle(mask)

    rebuilt = np.zeros(mask.size, dtype=np.uint8)
    position = 0
    value = 0
    for run in runs:
        rebuilt[position : position + run] = value
        position += run
        value ^= 1

    assert position == mask.size
    assert np.array_equal(rebuilt.reshape(mask.shape).astype(bool), mask)


def test_rle_handles_all_true_and_all_false():
    assert sum(_encode_rle(np.ones((4, 4), dtype=bool))) == 16
    assert sum(_encode_rle(np.zeros((4, 4), dtype=bool))) == 16
    assert _encode_rle(np.zeros((0, 0), dtype=bool)) == []


def test_rle_is_much_smaller_than_a_raw_mask():
    mask = np.zeros((256, 256), dtype=bool)
    mask[64:128, 64:128] = True
    assert len(_encode_rle(mask)) < mask.size / 50


# --------------------------------------------------------------------------
# Change measurement
# --------------------------------------------------------------------------

def test_summarise_counts_regions_and_measures_area():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[10:30, 10:30] = 0.9   # 400 px
    probability[60:80, 60:80] = 0.9   # 400 px
    stats = change_module.summarise_mask(probability)

    assert stats["regionCount"] == 2
    assert abs(stats["changedPercent"] - 8.0) < 0.01
    assert stats["centroid"] is not None


def test_summarise_ignores_specks():
    probability = np.zeros((50, 50), dtype=np.float32)
    probability[0, 0] = 0.9  # 1 px, below MIN_REGION_PIXELS
    assert change_module.summarise_mask(probability)["regionCount"] == 0


def test_summarise_handles_no_change():
    stats = change_module.summarise_mask(np.zeros((32, 32), dtype=np.float32))
    assert stats["changedPercent"] == 0.0
    assert stats["regionCount"] == 0
    assert stats["centroid"] is None
    assert stats["where"] is None


def test_summarise_locates_change_in_the_frame():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[5:25, 5:25] = 1.0
    assert change_module.summarise_mask(probability)["where"] == "north-west"

    probability = np.zeros((100, 100), dtype=np.float32)
    probability[75:95, 75:95] = 1.0
    assert change_module.summarise_mask(probability)["where"] == "south-east"


def test_change_evidence_calls_small_change_unchanged():
    stats = change_module.summarise_mask(np.zeros((64, 64), dtype=np.float32))
    evidence = change_module.change_evidence(stats, "What changed?")
    assert evidence.verdict == "no"
    assert evidence.terse == "unchanged"


def test_change_evidence_reports_the_fallback_honestly():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[10:40, 10:40] = 1.0
    evidence = change_module.change_evidence(
        change_module.summarise_mask(probability),
        "What changed?",
        fallback_reason="checkpoint missing",
    )
    assert evidence.verdict == "yes"
    joined = " ".join(evidence.notes)
    assert "image differencing" in joined
    assert "indicative only" in joined


def test_change_evidence_admits_it_cannot_give_direction():
    probability = np.ones((32, 32), dtype=np.float32)
    evidence = change_module.change_evidence(
        change_module.summarise_mask(probability),
        "Has the built-up area increased or decreased?",
    )
    assert any("does not signal direction" in note for note in evidence.notes)


def test_difference_fallback_detects_an_obvious_change():
    before = np.zeros((64, 64, 3), dtype=np.uint8)
    after = before.copy()
    after[20:40, 20:40] = 255

    probability = change_module.difference_probability(before, after)
    stats = change_module.summarise_mask(probability)
    assert stats["changedPercent"] > 1.0


def test_difference_fallback_reports_nothing_for_identical_images():
    image = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    probability = change_module.difference_probability(image, image)
    assert change_module.summarise_mask(probability)["changedPercent"] == 0.0


# --------------------------------------------------------------------------
# Fusion composition
# --------------------------------------------------------------------------

def test_fusion_highlights_what_sar_adds():
    evidence = fusion_module.fusion_evidence(
        "Use both images to identify built-up and water regions.",
        optical_labels=["Urban fabric", "Arable land"],
        sar_labels=["Urban fabric", "Inland waters"],
        agreement={"cosineSimilarity": 0.62},
    )
    joined = " ".join(evidence.findings)
    assert "SAR adds what optical missed: inland water" in joined
    assert "visible only in optical: cropland" in joined
    assert "both sensors agree on: urban fabric" in joined
    assert evidence.measurements["optical-SAR embedding similarity"] == "0.62"


def test_fusion_says_so_when_the_sensors_agree_completely():
    evidence = fusion_module.fusion_evidence(
        "What do the two sensors show?",
        optical_labels=["Marine waters"],
        sar_labels=["Marine waters"],
    )
    assert any("mainly confirms" in item for item in evidence.findings)


def test_fusion_works_without_croma():
    evidence = fusion_module.fusion_evidence(
        "Compare the sensors.",
        optical_labels=["Pastures"],
        sar_labels=["Urban fabric"],
        agreement=None,
        notes=("CROMA joint encoder unavailable: needs 12 optical bands",),
    )
    assert "optical-SAR embedding similarity" not in evidence.measurements
    assert any("CROMA" in note for note in evidence.notes)
    assert evidence.terse == "disagreement"


# --------------------------------------------------------------------------
# Device planning
# --------------------------------------------------------------------------

def test_two_gpus_give_the_vlm_its_own_card():
    devices = plan_devices(2)
    assert devices[registry.SLOT_VLM] == "cuda:0"
    assert devices[registry.SLOT_SPECIALIST] == "cuda:1"


def test_one_gpu_shares_and_enables_eviction():
    assert plan_devices(1) == {registry.SLOT_VLM: "cuda:0", registry.SLOT_SPECIALIST: "cuda:0"}
    assert ModelLoader(gpu_count=1).evict is True
    assert ModelLoader(gpu_count=2).evict is False


def test_no_gpu_falls_back_to_cpu():
    assert plan_devices(0) == {registry.SLOT_VLM: "cpu", registry.SLOT_SPECIALIST: "cpu"}


def test_loader_builds_adapters_lazily_without_loading_weights():
    loader = ModelLoader(gpu_count=0)
    adapter = loader.get("landcover")
    assert isinstance(adapter, Adapter)
    assert not adapter.ready, "constructing an adapter must not download weights"


def test_both_grounding_rows_share_one_adapter():
    loader = ModelLoader(gpu_count=0)
    assert loader.get("grounding_detector") is loader.get("grounding_segmenter")


def test_status_reports_placement_and_provenance_for_every_model():
    status = ModelLoader(gpu_count=2).status()
    assert status["gpuCount"] == 2
    keys = {row["key"] for row in status["models"]}
    assert keys == set(registry.REGISTRY)
    for row in status["models"]:
        assert row["loaded"] is False
        assert row["trainedBy"]
        assert row["device"] in ("cuda:0", "cuda:1")


def test_a_failed_load_is_not_retried_every_request():
    class Broken(Adapter):
        attempts = 0

        def _load(self):
            type(self).attempts += 1
            raise RuntimeError("no network")

    adapter = Broken(registry.spec("landcover"), "cpu")
    for _ in range(3):
        try:
            adapter.ensure_loaded()
        except AdapterUnavailable as exc:
            assert "no network" in exc.reason
    assert Broken.attempts == 1, "a dead download must not be retried per request"


def test_phrase_falls_back_to_the_template_when_the_vlm_is_disabled():
    loader = ModelLoader(gpu_count=0, enable_vlm=False)
    evidence = Evidence(task="vqa", verdict="yes", findings=["water present"], confidence=0.9)
    answer, label = loader.phrase(evidence)
    assert label == "template"
    assert answer.startswith("Yes.")


def test_require_bands_refuses_instead_of_zero_filling():
    class Dummy(Adapter):
        def _load(self):
            pass

    adapter = Dummy(registry.spec("landcover"), "cpu")
    rgb = BandStack.from_rgb(np.zeros((8, 8, 3), dtype=np.uint8))
    try:
        adapter.require_bands(rgb)
    except AdapterUnavailable as exc:
        assert "VV" in exc.reason
    else:
        raise AssertionError("a 12-band model must refuse an RGB stack")


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
