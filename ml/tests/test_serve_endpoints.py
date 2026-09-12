"""Endpoint wiring tests with stubbed models. No weights, no GPU.

These check the contract the backend depends on and, more importantly, the
degradation paths: what the server returns when a checkpoint is missing. Those
are the branches that only run on a bad day, so they are the ones worth pinning.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve  # noqa: E402
from satquery_ml import registry  # noqa: E402
from satquery_ml.adapters import AdapterUnavailable  # noqa: E402
from satquery_ml.bands import BEN_ALL_BANDS, BandStack  # noqa: E402
from satquery_ml.facts import Evidence  # noqa: E402


def png_b64(size: int = 32) -> str:
    rng = np.random.default_rng(0)
    array = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def band_payload(size: int = 32) -> dict:
    rng = np.random.default_rng(1)
    array = rng.random((size, size, len(BEN_ALL_BANDS))).astype(np.float32)
    return BandStack(array=array, names=BEN_ALL_BANDS).encode()


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------

class StubClassifier:
    spec = registry.spec("landcover")

    def __init__(self, available: bool = True):
        self.available = available
        self.ready = True

    def _check(self):
        if not self.available:
            raise AdapterUnavailable(self.spec.name, "needs the 12 published bands")

    def predict(self, stack, threshold=0.5):
        self._check()
        return {
            "scores": {name: 0.1 for name in ("Urban fabric", "Inland waters", "Arable land")},
            "present": ["Inland waters"],
            "threshold": threshold,
        }

    def evidence(self, stack, question="", task="vqa", threshold=0.5):
        self._check()
        evidence = Evidence(
            task=task,
            question=question,
            modality=stack.modality,
            findings=["land cover: inland water"],
            confidence=0.88,
            terse="inland water",
        )
        evidence.add_model(self.spec)
        if "water" in question.lower():
            evidence.verdict = "yes"
        return evidence


class StubGrounding:
    spec = registry.spec("grounding_detector")
    segmenter_spec = registry.spec("grounding_segmenter")

    def evidence(self, rgb, query):
        evidence = Evidence(
            task="grounding",
            question=query,
            verdict="yes",
            confidence=0.71,
            terse="a lake",
            boxes=[{"label": "a lake", "score": 0.71, "box": [0.1, 0.1, 0.5, 0.5]}],
            findings=["located 1 region matching 'a lake'"],
        )
        evidence.add_model(self.spec)
        return evidence


class StubChange:
    spec = registry.spec("change")

    def change_probability(self, before, after):
        probability = np.zeros((64, 64), dtype=np.float32)
        probability[10:40, 10:40] = 0.95
        return probability


class StubCroma:
    spec = registry.spec("fusion")

    def agreement(self, stack):
        return {"available": ["sar", "optical", "joint"], "cosineSimilarity": 0.66}


class StubLoader:
    """Stands in for ModelLoader; `missing` names models that fail to load."""

    def __init__(self, missing: tuple[str, ...] = (), classifier_ok: bool = True):
        self.missing = set(missing)
        self.classifier_ok = classifier_ok
        self.devices = {"vlm": "cpu", "specialist": "cpu"}

    def load(self, key: str):
        if key in self.missing:
            raise AdapterUnavailable(registry.spec(key).name, "checkpoint not found")
        if key == "landcover":
            return StubClassifier(self.classifier_ok)
        if key == "grounding_detector":
            return StubGrounding()
        if key == "change":
            return StubChange()
        if key == "fusion":
            return StubCroma()
        raise KeyError(key)

    def vlm(self):
        return None  # forces the deterministic template path

    def phrase(self, evidence):
        return evidence.template_answer(), "template"

    def status(self):
        return {"gpuCount": 0, "devices": self.devices, "models": []}


@pytest.fixture
def client(monkeypatch):
    def build(**kwargs):
        monkeypatch.setattr(serve, "LOADER", StubLoader(**kwargs))
        return TestClient(serve.app)

    return build


# --------------------------------------------------------------------------
# the contract the backend depends on
# --------------------------------------------------------------------------

def test_vqa_returns_the_four_keys_the_backend_client_reads(client):
    response = client().post(
        "/v1/vqa", json={"question": "Is there water here?", "imageB64": png_b64()}
    )
    assert response.status_code == 200
    body = response.json()
    for key in ("answer", "confidence", "model", "elapsedMs"):
        assert key in body, key
    assert body["answer"]
    assert 0.0 < body["confidence"] <= 1.0


def test_vqa_verdict_survives_into_the_answer(client):
    body = client().post(
        "/v1/vqa", json={"question": "Is there water here?", "imageB64": png_b64()}
    ).json()
    assert body["verdict"] == "yes"
    assert body["oneWord"] == "yes"
    assert body["answer"].startswith("Yes.")


def test_vqa_reports_model_provenance(client):
    body = client().post("/v1/vqa", json={"question": "Is there water?", "bands": band_payload()}).json()
    assert body["models"], "every answer must say which models produced it"
    assert body["models"][0]["trainedBy"] == "BIFOLD / TU Berlin"


def test_vqa_accepts_a_raw_band_stack(client):
    body = client().post(
        "/v1/vqa", json={"question": "Is there water?", "bands": band_payload()}
    ).json()
    assert body["modality"] == "optical-sar"


def test_vqa_rejects_a_body_with_no_image(client):
    assert client().post("/v1/vqa", json={"question": "hi"}).status_code == 400


def test_vqa_rejects_corrupt_image_data(client):
    response = client().post("/v1/vqa", json={"question": "hi", "imageB64": "not-base64!!"})
    assert response.status_code == 400


def test_vqa_rejects_a_corrupt_band_stack(client):
    response = client().post("/v1/vqa", json={"question": "hi", "bands": {"bands": ["VV"], "npz": "xx"}})
    assert response.status_code == 400


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------

def test_vqa_503s_when_nothing_can_answer(client):
    # Classifier cannot take RGB and the VLM is unavailable: say so, do not invent.
    response = client(classifier_ok=False).post(
        "/v1/vqa", json={"question": "Is there water?", "imageB64": png_b64()}
    )
    assert response.status_code == 503
    assert "needs the 12 published bands" in response.json()["detail"]


def test_grounding_503s_when_the_detector_is_missing(client):
    response = client(missing=("grounding_detector",)).post(
        "/v1/grounding", json={"query": "the water body", "imageB64": png_b64()}
    )
    assert response.status_code == 503


def test_change_falls_back_to_differencing_and_labels_it(client):
    body = client(missing=("change",)).post(
        "/v1/change",
        json={"question": "What changed?", "beforeB64": png_b64(), "afterB64": png_b64()},
    ).json()
    assert body["changeStats"]["changedPercent"] >= 0.0
    joined = " ".join(body["notes"])
    assert "image differencing" in joined, "an unlabelled fallback is worse than none"


def test_fusion_explains_why_croma_was_skipped(client):
    body = client().post(
        "/v1/fusion",
        json={"question": "What do both sensors show?", "opticalB64": png_b64(), "sarB64": png_b64()},
    ).json()
    assert any("CROMA needs one co-registered stack" in note for note in body["notes"])


# --------------------------------------------------------------------------
# the other mandatory tasks
# --------------------------------------------------------------------------

def test_grounding_returns_spatial_evidence(client):
    body = client().post(
        "/v1/grounding", json={"query": "Highlight the water body", "imageB64": png_b64()}
    ).json()
    assert body["boxes"], "grounding must return spatial evidence"
    assert len(body["boxes"][0]["box"]) == 4
    assert all(0.0 <= v <= 1.0 for v in body["boxes"][0]["box"]), "boxes must be normalised"


def test_change_returns_a_decodable_mask_and_measured_stats(client):
    body = client().post(
        "/v1/change",
        json={"question": "What changed?", "beforeB64": png_b64(), "afterB64": png_b64()},
    ).json()

    stats = body["changeStats"]
    assert abs(stats["changedPercent"] - 100.0 * 900 / 4096) < 0.5
    assert stats["regionCount"] == 1
    assert sum(body["changeMaskRle"]) == body["maskHeight"] * body["maskWidth"]
    # The measured number must appear in the prose, unaltered.
    assert f"{stats['changedPercent']:.2f}%" in body["answer"]


def test_fusion_uses_croma_when_a_joint_stack_is_supplied(client):
    body = client().post(
        "/v1/fusion", json={"question": "What does SAR add?", "bands": band_payload()}
    ).json()
    assert body["agreement"]["cosineSimilarity"] == 0.66
    assert "optical-SAR embedding similarity" in body["measurements"]


def test_classify_returns_ranked_scores_without_a_language_model(client):
    body = client().post("/v1/classify", json={"bands": band_payload()}).json()
    assert body["present"] == ["Inland waters"]
    scores = [row["score"] for row in body["scores"]]
    assert scores == sorted(scores, reverse=True)
    assert "answer" not in body, "/v1/classify is stage 1 only"


def test_caption_task_needs_no_question(client):
    # Captioning rides /v1/vqa with task=caption, which is what the backend sends.
    body = client().post("/v1/vqa", json={"task": "caption", "imageB64": png_b64()}).json()
    assert body["task"] == "caption"
    assert body["answer"]


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------

def test_health_reports_that_nothing_was_trained_by_us(client):
    body = client().get("/health").json()
    assert body["status"] == "ok"
    assert body["trainedByUs"] == []


def test_models_endpoint_lists_provenance_and_plans(client):
    body = client().get("/v1/models").json()
    assert len(body["models"]) == len(registry.REGISTRY)
    for task in ("vqa", "caption", "grounding", "change", "fusion"):
        assert task in body["taskPlans"]
    assert "BIFOLD / TU Berlin" in body["provenanceMarkdown"]
