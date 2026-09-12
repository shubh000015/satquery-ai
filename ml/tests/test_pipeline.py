"""Tests for the two-stage pipeline. CPU only — no GPU, no model downloads.

Run from ml/:   python -m pytest tests -q
or standalone:  python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery_ml import labels as label_vocab  # noqa: E402
from satquery_ml import metrics  # noqa: E402
from satquery_ml.dataset import (  # noqa: E402
    BigEarthNetPatches,
    class_frequencies,
    load_records,
    positive_weights,
    split_records,
)
from satquery_ml.inference import LandCoverPredictor, Prediction, detect_modality  # noqa: E402
from satquery_ml.model import (  # noqa: E402
    LandCoverClassifier,
    ModelConfig,
    load_checkpoint,
    save_checkpoint,
)
from satquery_ml.verbalizer import (  # noqa: E402
    Verbalizer,
    resolve_facts,
    template_answer,
)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def test_every_class_round_trips_alone():
    for class_name in label_vocab.CLASSES:
        joined = class_name.lower()
        assert label_vocab.parse_label_string(joined) == [class_name], class_name


def test_class_names_containing_commas_survive_the_join():
    """The join separator is ', ' and four class names contain ', ' themselves."""
    tricky = [
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Moors, heathland and sclerophyllous vegetation",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
    ]
    for class_name in tricky:
        assert "," in class_name
        combo = [class_name, "Inland waters"]
        joined = ", ".join(sorted(name.lower() for name in combo))
        parsed = label_vocab.parse_label_string(joined)
        assert set(parsed) == set(combo), (class_name, joined, parsed)

    # A naive split on ', ' would have produced fragments; confirm it really would.
    joined = ", ".join(sorted(n.lower() for n in ["Transitional woodland, shrub", "Inland waters"]))
    assert len(joined.split(", ")) == 3
    assert len(label_vocab.parse_label_string(joined)) == 2


def test_all_nineteen_classes_at_once():
    joined = ", ".join(sorted(name.lower() for name in label_vocab.CLASSES))
    parsed = label_vocab.parse_label_string(joined)
    assert set(parsed) == set(label_vocab.CLASSES)
    assert len(parsed) == label_vocab.NUM_CLASSES


def test_empty_and_garbage_label_strings():
    assert label_vocab.parse_label_string("") == []
    assert label_vocab.parse_label_string("   ") == []
    assert label_vocab.parse_label_string("nonsense, gibberish") == []


def test_labels_to_vector():
    vector = label_vocab.labels_to_vector(["Inland waters", "Urban fabric"])
    assert sum(vector) == 2
    assert vector[label_vocab.CLASS_TO_INDEX["inland waters"]] == 1.0


def test_groups_only_reference_real_classes():
    for group, members in label_vocab.GROUPS.items():
        for member in members:
            assert member in label_vocab.CLASSES, (group, member)


def test_question_matching_prefers_groups_for_vague_words():
    # Vague word -> coarse group, so an industrial patch still answers "yes".
    assert label_vocab.match_class_in_question("Is there any water here?") == "water"
    assert label_vocab.match_class_in_question("Any urban areas?") == "builtup"
    assert label_vocab.match_class_in_question("Any buildings?") == "builtup"
    assert label_vocab.match_class_in_question("Is there forest?") == "vegetation"
    assert label_vocab.match_class_in_question("Any farmland?") == "agriculture"


def test_question_matching_prefers_the_exact_class_when_named():
    assert label_vocab.match_class_in_question("Is there urban fabric?") == "Urban fabric"
    assert (
        label_vocab.match_class_in_question("Is there coniferous forest?") == "Coniferous forest"
    )
    # A specific short name still beats the bare group word it contains.
    assert label_vocab.match_class_in_question("Any conifer forest?") == "Coniferous forest"
    assert label_vocab.match_class_in_question("Any marine waters?") == "Marine waters"


def test_question_matching_returns_none_for_unrelated_questions():
    assert label_vocab.match_class_in_question("What is the weather") is None
    assert label_vocab.match_class_in_question("") is None


def test_short_names_cover_every_class():
    for class_name in label_vocab.CLASSES:
        assert class_name in label_vocab.SHORT_NAMES, class_name
    # Short names must be distinct, or two classes collapse into one answer.
    assert len(set(label_vocab.SHORT_NAMES.values())) == label_vocab.NUM_CLASSES


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_average_precision_perfect_and_inverted():
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    perfect = torch.tensor([0.9, 0.8, 0.2, 0.1])
    assert abs(metrics.average_precision(perfect, targets) - 1.0) < 1e-6

    inverted = torch.tensor([0.1, 0.2, 0.8, 0.9])
    assert metrics.average_precision(inverted, targets) < 0.6


def test_average_precision_is_nan_without_positives():
    value = metrics.average_precision(torch.tensor([0.5, 0.4]), torch.tensor([0.0, 0.0]))
    assert value != value  # NaN


def test_map_skips_classes_without_support():
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    map_score, per_class = metrics.mean_average_precision(scores, targets)
    assert abs(map_score - 1.0) < 1e-6      # driven by class 0 only
    assert per_class[1] != per_class[1]     # class 1 is NaN, not zero


def test_tune_thresholds_beats_a_flat_half():
    torch.manual_seed(0)
    # A rare class whose positives all score around 0.3 needs a low threshold.
    scores = torch.cat([torch.full((10, 1), 0.30), torch.full((90, 1), 0.05)])
    targets = torch.cat([torch.ones(10, 1), torch.zeros(90, 1)])

    tuned = metrics.tune_thresholds(scores, targets)
    assert tuned[0] < 0.5

    flat = metrics.f1_scores(scores, targets, torch.tensor([0.5]))
    fitted = metrics.f1_scores(scores, targets, tuned)
    assert fitted["microF1"] > flat["microF1"]
    assert abs(fitted["microF1"] - 1.0) < 1e-6


def test_report_shape():
    scores = torch.rand(40, label_vocab.NUM_CLASSES)
    targets = (torch.rand(40, label_vocab.NUM_CLASSES) > 0.7).float()
    summary = metrics.report(scores, targets, torch.full((label_vocab.NUM_CLASSES,), 0.5))
    for key in ("mAP", "microF1", "macroF1", "microPrecision", "microRecall", "samples"):
        assert key in summary
    assert summary["samples"] == 40
    assert len(summary["perClass"]) == label_vocab.NUM_CLASSES


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def test_forward_shape_and_modality_matters():
    torch.manual_seed(0)
    model = LandCoverClassifier(ModelConfig(backbone="resnet18"), pretrained_backbone=False)
    model.eval()

    pixels = torch.randn(2, 3, 224, 224)
    optical = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sar = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    with torch.inference_mode():
        out_optical = model(pixels, optical)
        out_sar = model(pixels, sar)

    assert out_optical.shape == (2, label_vocab.NUM_CLASSES)
    # The modality one-hot reaches the head, so the two must differ.
    assert not torch.allclose(out_optical, out_sar)


def test_checkpoint_round_trip(tmp_path: Path):
    model = LandCoverClassifier(ModelConfig(backbone="resnet18"), pretrained_backbone=False)
    thresholds = [0.3] * label_vocab.NUM_CLASSES
    path = tmp_path / "classifier.pt"
    save_checkpoint(path, model, thresholds=thresholds, metrics={"macroF1": 0.42})

    assert path.exists()
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["metrics"]["macroF1"] == 0.42

    restored, restored_thresholds, restored_metrics = load_checkpoint(path)
    assert restored_thresholds == thresholds
    assert restored_metrics["macroF1"] == 0.42

    pixels = torch.randn(1, 3, 224, 224)
    modality = torch.tensor([[1.0, 0.0]])
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(model(pixels, modality), restored(pixels, modality), atol=1e-6)


def test_unknown_backbone_rejected():
    try:
        LandCoverClassifier(ModelConfig(backbone="vgg16"), pretrained_backbone=False)
    except ValueError as exc:
        assert "backbone must be one of" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unsupported backbone")


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def _make_dataset(root: Path, patches: int = 6) -> Path:
    (root / "images_s2").mkdir(parents=True, exist_ok=True)
    (root / "images_s1").mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(patches):
        classes = ["Inland waters", "Transitional woodland, shrub"] if i % 2 else ["Arable land"]
        answer = ", ".join(sorted(name.lower() for name in classes))

        for folder, modality in (("images_s2", "optical"), ("images_s1", "sar")):
            rel = f"{folder}/patch_{i}.png"
            Image.new("RGB", (224, 224), (i * 20 % 256, 90, 140)).save(root / rel)
            rows.append({
                "image": rel, "question": "Which land-cover classes are present?",
                "answer": answer, "source": "bigearthnet",
                "modality": modality, "category": "multi-label",
            })
            # A presence row that must be ignored by the loader.
            rows.append({
                "image": rel, "question": "Is there any water?", "answer": "yes",
                "source": "bigearthnet", "modality": modality, "category": "presence",
            })

    with (root / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return root / "train.jsonl"


def test_load_records_ignores_presence_rows(tmp_path: Path):
    jsonl = _make_dataset(tmp_path, patches=6)
    records = load_records(jsonl, root=tmp_path)

    assert len(records) == 12  # 6 patches x 2 modalities, presence rows dropped
    assert {r.modality for r in records} == {"optical", "sar"}
    for record in records:
        assert record.label_indices
        assert all(0 <= i < label_vocab.NUM_CLASSES for i in record.label_indices)


def test_loader_skips_missing_images(tmp_path: Path):
    jsonl = _make_dataset(tmp_path, patches=4)
    (tmp_path / "images_s1" / "patch_0.png").unlink()
    records = load_records(jsonl, root=tmp_path)
    assert len(records) == 7
    assert all(r.rel_path != "images_s1/patch_0.png" for r in records)


def test_split_is_deterministic_and_disjoint(tmp_path: Path):
    jsonl = _make_dataset(tmp_path, patches=40)
    records = load_records(jsonl, root=tmp_path)

    train_a, val_a = split_records(records, 0.25, seed=13)
    train_b, val_b = split_records(records, 0.25, seed=13)
    assert [r.rel_path for r in val_a] == [r.rel_path for r in val_b]

    overlap = {r.rel_path for r in train_a} & {r.rel_path for r in val_a}
    assert not overlap
    assert len(train_a) + len(val_a) == len(records)

    _, val_other = split_records(records, 0.25, seed=99)
    assert [r.rel_path for r in val_a] != [r.rel_path for r in val_other]


def test_dataset_item_shapes(tmp_path: Path):
    jsonl = _make_dataset(tmp_path, patches=4)
    records = load_records(jsonl, root=tmp_path)
    dataset = BigEarthNetPatches(tmp_path, records, train=True)

    pixels, modality, target = dataset[0]
    assert pixels.shape == (3, 224, 224)
    assert modality.shape == (2,)
    assert modality.sum().item() == 1.0
    assert target.shape == (label_vocab.NUM_CLASSES,)
    assert target.sum().item() >= 1.0


def test_positive_weights_favour_rare_classes(tmp_path: Path):
    jsonl = _make_dataset(tmp_path, patches=20)
    records = load_records(jsonl, root=tmp_path)

    frequencies = class_frequencies(records)
    weights = positive_weights(records)
    assert weights.shape == (label_vocab.NUM_CLASSES,)
    assert float(weights.max()) <= 10.0
    assert float(weights.min()) >= 1.0

    # A class that never appears keeps weight 1 rather than exploding.
    never = label_vocab.CLASS_TO_INDEX["marine waters"]
    assert frequencies[never] == 0
    assert weights[never] == 1.0


# --------------------------------------------------------------------------
# inference + verbalizer
# --------------------------------------------------------------------------

def test_detect_modality_on_our_two_render_styles():
    import numpy as np

    # SAR render: B is exactly |R - G| by construction.
    rng = np.random.default_rng(0)
    red = rng.random((64, 64), dtype=np.float32)
    green = rng.random((64, 64), dtype=np.float32)
    sar = np.stack([red, green, np.abs(red - green)], axis=2)
    sar_image = Image.fromarray((sar * 255).astype("uint8"), mode="RGB")
    assert detect_modality(sar_image) == "sar"

    # Saturated optical patch.
    optical = np.zeros((64, 64, 3), dtype=np.float32)
    optical[:, :, 1] = 0.85
    optical[:, :, 0] = 0.15
    optical_image = Image.fromarray((optical * 255).astype("uint8"), mode="RGB")
    assert detect_modality(optical_image) == "optical"


def _prediction(present, scores=None, modality="optical") -> Prediction:
    full = {name: 0.02 for name in label_vocab.CLASSES}
    for name in present:
        full[name] = 0.88
    full.update(scores or {})
    return Prediction(
        modality=modality,
        scores=full,
        present=list(present),
        thresholds={name: 0.5 for name in label_vocab.CLASSES},
    )


def test_verdict_comes_from_the_classifier_not_the_llm():
    prediction = _prediction(["Inland waters", "Arable land"])

    yes = resolve_facts(prediction, "Is there any water in this image?")
    assert yes.verdict == "yes"
    assert yes.one_word == "yes"
    assert yes.confidence > 0.8

    no = resolve_facts(prediction, "Is there any urban fabric in this image?")
    assert no.verdict == "no"
    assert no.one_word == "no"
    assert no.confidence > 0.8  # confident in the negative


def test_borderline_score_reports_possibly():
    prediction = _prediction(["Arable land"], scores={"Urban fabric": 0.30})
    facts = resolve_facts(prediction, "Is there any urban fabric here?")
    assert facts.verdict == "possibly"


def test_listing_and_caption_have_no_verdict():
    prediction = _prediction(["Inland waters", "Arable land"])

    listing = resolve_facts(prediction, "Which land-cover classes are present?")
    assert listing.verdict is None
    assert "inland water" in listing.present
    assert listing.one_word == "inland water"

    caption = resolve_facts(prediction, "", task="caption")
    assert caption.verdict is None
    assert caption.present


def test_template_answer_matches_the_verdict():
    prediction = _prediction(["Inland waters"], modality="sar")

    yes = template_answer(resolve_facts(prediction, "Is there water here?"))
    assert yes.lower().startswith("yes")
    assert "SAR" in yes

    no = template_answer(resolve_facts(prediction, "Is there any urban fabric here?"))
    assert no.lower().startswith("no")

    caption = template_answer(resolve_facts(prediction, "", task="caption"))
    assert "inland water" in caption


def test_disabled_verbalizer_falls_back_to_template():
    verbalizer = Verbalizer(enabled=False)
    assert verbalizer.available is False
    assert verbalizer.label == "template"

    facts = resolve_facts(_prediction(["Inland waters"]), "Is there water here?")
    assert verbalizer.phrase(facts).lower().startswith("yes")


def test_predictor_end_to_end_on_a_saved_checkpoint(tmp_path: Path):
    """Train nothing, but prove weights -> image -> labels -> sentence works."""
    model = LandCoverClassifier(ModelConfig(backbone="resnet18"), pretrained_backbone=False)
    path = tmp_path / "classifier.pt"
    save_checkpoint(path, model, thresholds=[0.5] * label_vocab.NUM_CLASSES)

    predictor = LandCoverPredictor(path, device="cpu")
    image = Image.new("RGB", (224, 224), (40, 110, 60))
    prediction = predictor.predict(image, modality="optical")

    assert prediction.modality == "optical"
    assert len(prediction.scores) == label_vocab.NUM_CLASSES
    assert all(0.0 <= v <= 1.0 for v in prediction.scores.values())
    assert prediction.present  # never empty, even when nothing clears the bar
    assert prediction.dominant in label_vocab.CLASSES
    assert len(prediction.top(5)) == 5

    facts = resolve_facts(prediction, "Which land-cover classes are present?")
    sentence = Verbalizer(enabled=False).phrase(facts)
    assert isinstance(sentence, str) and sentence


if __name__ == "__main__":
    import tempfile
    import traceback

    passed = failed = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            if "tmp_path" in function.__code__.co_varnames[: function.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as directory:
                    function(Path(directory))
            else:
                function()
            passed += 1
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()

    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
