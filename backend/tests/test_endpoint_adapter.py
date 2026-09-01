"""End-to-end proof of the fine-tuned-model seam, no GPU required.

A stub HTTP server plays the role of ml/serve.py; the test wires it in via
SATQUERY_VLM_ENDPOINT and checks that the pipeline answer comes from the
endpoint while the deterministic evidence stack keeps working.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

STUB_ANSWER = "Yes — a river crosses the scene from north to south."


class _StubVlmHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        assert body.get("imageB64"), "backend must send the scene image"
        payload = json.dumps(
            {"answer": STUB_ANSWER, "confidence": 0.91, "model": "RS-LoRA-stub"}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture
def stub_vlm_endpoint(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _StubVlmHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"

    monkeypatch.setenv("SATQUERY_VLM_ENDPOINT", endpoint)
    from app.core.config import get_settings

    get_settings.cache_clear()
    import app.agent.controller as controller_module

    controller_module._controller = None

    yield endpoint

    server.shutdown()
    monkeypatch.delenv("SATQUERY_VLM_ENDPOINT", raising=False)
    get_settings.cache_clear()
    controller_module._controller = None


def test_vqa_routes_to_wired_endpoint(client, optical_tif: Path, stub_vlm_endpoint: str):
    health = client.get("/api/health").json()
    assert health["weightsWired"] >= 1

    with optical_tif.open("rb") as fh:
        upload = client.post("/api/assets", files={"files": (optical_tif.name, fh, "image/tiff")})
    assert upload.status_code == 200
    asset_ids = [a["id"] for a in upload.json()["assets"]]

    response = client.post(
        "/api/query",
        json={"query": "Is there water in this image?", "assetIds": asset_ids},
    )
    assert response.status_code == 200
    result = response.json()

    assert result["answer"] == STUB_ANSWER
    assert f"endpoint:{stub_vlm_endpoint}" in result["inferenceBackend"]
    # Deterministic evidence still rides along with the model answer.
    assert result["metrics"]
    assert any("deterministic analysis stack" in obs for obs in result["observations"])
