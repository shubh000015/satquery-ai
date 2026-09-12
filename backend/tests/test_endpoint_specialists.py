"""Tests for the grounding, change and fusion endpoint seams.

A stub HTTP server plays ml/serve.py. What matters here is that model-produced
spatial evidence actually reaches the UI schema — boxes rescaled to the 0..100
viewBox, run-length masks decoded and re-rendered as SVG paths — because those
conversions are silent when wrong: the answer still looks fine and the overlay
is simply in the wrong place.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from app.tools.endpoint_client import band_names_for, decode_rle, scene_bands


# --------------------------------------------------------------------------
# band naming and transport
# --------------------------------------------------------------------------

def test_band_names_never_claim_more_bands_than_we_have():
    # A 1-band RISAT chip is VV and nothing else.
    assert band_names_for("sar", 1) == ("VV",)
    assert band_names_for("sar", 2) == ("VV", "VH")
    assert band_names_for("sar", 4) == ("VV", "VH"), "SAR has no B-bands to offer"

    assert band_names_for("optical", 3) == ("B04", "B03", "B02")
    assert band_names_for("multispectral", 4) == ("B04", "B03", "B02", "B08")
    # Cartosat gives at most 4 usable bands; do not invent B11/B12.
    assert band_names_for("optical", 8) == ("B04", "B03", "B02", "B08")


def test_band_names_handle_a_degenerate_band_count():
    assert band_names_for("optical", 0) == ("B04",)


def test_scene_bands_round_trips_into_the_ml_band_stack(optical_tif):
    import sys
    from pathlib import Path

    from app.services.raster import load_scene

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
    from satquery_ml.bands import BandStack

    scene = load_scene(optical_tif, max_edge=64)
    payload = scene_bands(scene, "optical")

    stack = BandStack.decode(payload)
    assert stack.names == band_names_for("optical", scene.bands)
    assert stack.array.dtype == np.float32
    assert stack.modality == "optical"
    # An RGB upload must not satisfy a 12-band model.
    assert not stack.has("VV", "B08", "B11")


def test_scene_bands_downsamples_large_scenes(optical_tif):
    from app.services.raster import load_scene

    scene = load_scene(optical_tif, max_edge=1024)
    payload = scene_bands(scene, "optical", max_edge=64)
    assert max(payload["height"], payload["width"]) <= 64


# --------------------------------------------------------------------------
# mask transport
# --------------------------------------------------------------------------

def test_decode_rle_inverts_the_server_encoding():
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:20, 6:28] = True

    # Encode the way ml/satquery_ml/adapters/grounding.py does.
    flat = mask.astype(np.uint8).ravel()
    changes = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(boundaries).tolist()
    if flat[0] == 1:
        runs = [0] + runs

    assert np.array_equal(decode_rle(runs, 32, 32), mask)


def test_decode_rle_is_robust_to_a_truncated_payload():
    # A run list that overruns the frame must clip, not raise.
    decoded = decode_rle([0, 10_000], 4, 4)
    assert decoded.shape == (4, 4)
    assert decoded.all()


def test_decode_rle_handles_an_empty_run_list():
    assert not decode_rle([], 4, 4).any()


# --------------------------------------------------------------------------
# stub endpoint
# --------------------------------------------------------------------------

GROUNDING_ANSWER = "The lake sits in the north-west quadrant."
CHANGE_ANSWER = "12.50% of the frame changed across 2 regions."
FUSION_ANSWER = "SAR adds inland water that optical missed."


def _rle_for(height: int, width: int, box: tuple[int, int, int, int]) -> list[int]:
    mask = np.zeros((height, width), dtype=np.uint8)
    y0, y1, x0, x1 = box
    mask[y0:y1, x0:x1] = 1
    flat = mask.ravel()
    changes = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(boundaries).tolist()
    return [0] + runs if flat[0] == 1 else runs


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        path = self.path

        if path.endswith("/v1/grounding"):
            payload = {
                "answer": GROUNDING_ANSWER,
                "confidence": 0.74,
                "model": "grounding-dino-base+sam2",
                "findings": ["located 1 region matching 'a lake'"],
                "notes": [],
                "boxes": [
                    {"label": "a lake", "score": 0.74, "box": [0.1, 0.2, 0.4, 0.6]}
                ],
                "masks": [
                    {
                        "label": "a lake",
                        "score": 0.7,
                        "rle": _rle_for(64, 64, (12, 38, 6, 26)),
                        "height": 64,
                        "width": 64,
                        "coveragePercent": 12.7,
                    }
                ],
            }
        elif path.endswith("/v1/change"):
            assert body.get("beforeB64") and body.get("afterB64")
            payload = {
                "answer": CHANGE_ANSWER,
                "confidence": 0.85,
                "model": "ChangeFormerV6",
                "findings": ["12.50% of the frame changed across 2 region(s)"],
                "notes": [],
                "boxes": [],
                "masks": [],
                "changeStats": {"changedPercent": 12.5, "regionCount": 2, "where": "north-west"},
                "changeMaskRle": _rle_for(64, 64, (4, 28, 4, 28)),
                "maskHeight": 64,
                "maskWidth": 64,
            }
        elif path.endswith("/v1/fusion"):
            assert body.get("optical") and body.get("sar"), "fusion needs both band stacks"
            payload = {
                "answer": FUSION_ANSWER,
                "confidence": 0.8,
                "model": "CROMA-base+BEN-ResNet50",
                "findings": ["SAR adds what optical missed: inland water"],
                "notes": ["CROMA joint encoder unavailable: separate images"],
                "boxes": [],
                "masks": [],
                "agreement": {"cosineSimilarity": 0.66},
            }
        else:
            payload = {"answer": "unused", "confidence": 0.5, "model": "stub"}

        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_all_endpoints(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"

    for name in ("GROUNDING", "CHANGE", "FUSION"):
        monkeypatch.setenv(f"SATQUERY_{name}_ENDPOINT", endpoint)

    from app.core.config import get_settings

    get_settings.cache_clear()
    import app.agent.controller as controller_module

    controller_module._controller = None

    yield endpoint

    server.shutdown()
    get_settings.cache_clear()
    controller_module._controller = None


def _upload(client, path):
    with path.open("rb") as handle:
        response = client.post(
            "/api/assets", files={"files": (path.name, handle, "image/tiff")}
        )
    assert response.status_code == 200
    return [asset["id"] for asset in response.json()["assets"]]


def test_grounding_uses_model_boxes_and_masks(client, optical_tif, stub_all_endpoints):
    asset_ids = _upload(client, optical_tif)
    result = client.post(
        "/api/query",
        json={"query": "Highlight the water body referred to in the query", "assetIds": asset_ids},
    ).json()

    assert result["answer"] == GROUNDING_ANSWER
    assert f"endpoint:{stub_all_endpoints}" in result["inferenceBackend"]

    boxes = result["boxes"]
    assert boxes, "model boxes must reach the response"
    box = boxes[0]
    # Normalised 0.1,0.2 -> 0.4,0.6 becomes x=10 y=20 w=30 h=40 in the 0..100 viewBox.
    assert box["x"] == pytest.approx(10.0, abs=0.1)
    assert box["y"] == pytest.approx(20.0, abs=0.1)
    assert box["w"] == pytest.approx(30.0, abs=0.1)
    assert box["h"] == pytest.approx(40.0, abs=0.1)
    assert result["masks"], "SAM 2 masks must be decoded into the overlay"
    assert result["masks"][0]["d"].startswith("M")


def test_change_uses_the_model_mask_and_measured_area(
    client, t0_tif, t1_tif, stub_all_endpoints
):
    asset_ids = _upload(client, t0_tif) + _upload(client, t1_tif)
    result = client.post(
        "/api/query",
        json={"query": "What changed between these two dates?", "assetIds": asset_ids},
    ).json()

    assert result["answer"] == CHANGE_ANSWER
    assert result["masks"], "the model change mask must reach the overlay"
    labels = [metric["label"] for metric in result["metrics"]]
    assert "Changed area (model)" in labels
    assert any("12.50%" in metric["value"] for metric in result["metrics"])


def test_fusion_reports_the_agreement_metric(
    client, optical_tif, sar_tif, stub_all_endpoints
):
    asset_ids = _upload(client, optical_tif) + _upload(client, sar_tif)
    result = client.post(
        "/api/query",
        json={
            "query": "Use the optical and SAR images together to identify built-up and water regions.",
            "assetIds": asset_ids,
        },
    ).json()

    assert result["answer"] == FUSION_ANSWER
    labels = [metric["label"] for metric in result["metrics"]]
    assert "Optical–SAR agreement" in labels
    assert any("SAR adds" in obs for obs in result["observations"])
