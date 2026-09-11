"""Interim LLM stand-in: registry wiring + VQA answer comes from the LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools.endpoint_client import VlmReply

STUB_ANSWER = "Yes — vegetation decreased along the southern parcels."


@pytest.fixture
def llm_key(monkeypatch):
    monkeypatch.setenv("SATQUERY_LLM_API_KEY", "test-key-not-real")
    monkeypatch.setenv("SATQUERY_LLM_PROVIDER", "gemini")
    from app.core.config import get_settings

    get_settings.cache_clear()
    import app.agent.controller as controller_module

    controller_module._controller = None
    yield
    monkeypatch.delenv("SATQUERY_LLM_API_KEY", raising=False)
    get_settings.cache_clear()
    controller_module._controller = None


def test_registry_marks_language_rows_wired_when_llm_key_set(client, llm_key):
    health = client.get("/api/health").json()
    assert health["llmWired"] is True
    assert health["weightsWired"] >= 1

    registry = client.get("/api/registry").json()
    vqa = next(m for m in registry["models"] if m["key"] == "vqa")
    assert vqa["status"] == "wired"
    assert registry["weightsWired"] >= 1


def test_vqa_uses_llm_answer_and_keeps_evidence(client, optical_tif: Path, llm_key, monkeypatch):
    def fake_ask_llm(*_args, **_kwargs):
        return VlmReply(answer=STUB_ANSWER, confidence=0.8, model="gemini:gemini-2.0-flash")

    monkeypatch.setattr("app.tools.llm_client.ask_llm", fake_ask_llm)

    with optical_tif.open("rb") as fh:
        upload = client.post("/api/assets", files={"files": (optical_tif.name, fh, "image/tiff")})
    assert upload.status_code == 200
    asset_ids = [a["id"] for a in upload.json()["assets"]]

    response = client.post(
        "/api/query",
        json={"query": "Did vegetation decrease in this scene?", "assetIds": asset_ids},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == STUB_ANSWER
    assert "llm" in result["inferenceBackend"]
    assert result["metrics"]
    assert any("deterministic analysis stack" in obs for obs in result["observations"])


def test_llm_ask_requires_key(client):
    response = client.post("/api/llm/ask", json={"query": "What is in this scene?"})
    assert response.status_code == 503


def test_llm_status_unconfigured(client):
    status = client.get("/api/llm/status").json()
    assert status["configured"] is False
