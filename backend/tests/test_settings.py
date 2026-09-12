"""Settings and registry wiring for the shared Kaggle ML endpoint."""

from app.agent import registry
from app.core.config import clean_endpoint, get_settings


def test_clean_endpoint_strips_noise_and_inline_comments():
    assert clean_endpoint("  https://x.trycloudflare.com/  ") == "https://x.trycloudflare.com"
    assert clean_endpoint("http://localhost:8100        # VQA, captioning") == "http://localhost:8100"
    assert clean_endpoint('  "https://host.example"  ') == "https://host.example"
    assert clean_endpoint("none") is None
    assert clean_endpoint("") is None
    assert clean_endpoint(None) is None


def test_ml_endpoint_fills_blank_specialists(monkeypatch):
    monkeypatch.setenv("SATQUERY_ML_ENDPOINT", "https://example.trycloudflare.com")
    for name in ("VLM", "GROUNDING", "CHANGE", "FUSION"):
        monkeypatch.setenv(f"SATQUERY_{name}_ENDPOINT", "")

    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.resolved_endpoint("vlm_endpoint") == "https://example.trycloudflare.com"
        assert registry.endpoint_for("vqa", settings) == "https://example.trycloudflare.com"
        assert registry.endpoint_for("grounding", settings) == "https://example.trycloudflare.com"
        assert registry.endpoint_for("change-mask", settings) == "https://example.trycloudflare.com"
        assert registry.endpoint_for("fusion", settings) == "https://example.trycloudflare.com"
        assert registry.endpoint_for("controller", settings) is None
        assert registry.weights_wired(settings) == 7
    finally:
        get_settings.cache_clear()


def test_specific_endpoint_wins_over_ml_endpoint(monkeypatch):
    monkeypatch.setenv("SATQUERY_ML_ENDPOINT", "https://shared.example")
    monkeypatch.setenv("SATQUERY_CHANGE_ENDPOINT", "https://change-only.example")
    monkeypatch.setenv("SATQUERY_VLM_ENDPOINT", "")
    monkeypatch.setenv("SATQUERY_GROUNDING_ENDPOINT", "")
    monkeypatch.setenv("SATQUERY_FUSION_ENDPOINT", "")

    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert registry.endpoint_for("vqa", settings) == "https://shared.example"
        assert registry.endpoint_for("change-vqa", settings) == "https://change-only.example"
    finally:
        get_settings.cache_clear()
