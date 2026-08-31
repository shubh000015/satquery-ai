"""End-to-end API tests: upload, query, stream, report."""

from __future__ import annotations

import json
from pathlib import Path


def _upload(client, paths: list[Path], **form) -> dict:
    files = [("files", (path.name, path.read_bytes(), "image/tiff")) for path in paths]
    response = client.post("/api/assets", files=files, data=form)
    assert response.status_code == 200, response.text
    return response.json()


def _ask(client, query: str, upload: dict, expect: int = 200) -> dict:
    response = client.post(
        "/api/query",
        json={
            "query": query,
            "assetIds": [asset["id"] for asset in upload["assets"]],
            "sessionId": upload["sessionId"],
        },
    )
    assert response.status_code == expect, response.text
    return response.json()


def test_health_and_registry(client) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["problemStatement"] == "SIH26167"

    registry = client.get("/api/registry").json()
    keys = {model["key"] for model in registry["models"]}
    assert {"vqa", "grounding", "segmentation", "caption", "change-vqa", "change-mask", "fusion"} <= keys
    # No weights are configured in tests, so every learned row is still planned.
    assert registry["weightsWired"] == 0
    assert all(tool["backend"] == "heuristic-baseline" for tool in registry["tools"])


def test_upload_single_scene_reports_metadata(client, optical_tif: Path) -> None:
    payload = _upload(client, [optical_tif])
    assert payload["mode"] == "single"
    assert payload["validation"]["ok"] is True

    asset = payload["assets"][0]
    assert asset["role"] == "primary"
    assert asset["modality"] in ("optical", "multispectral")
    assert asset["format"] == "GeoTIFF"
    assert asset["meta"]["georeferenced"] is True
    assert asset["meta"]["gsdMeters"] > 0
    assert payload["suggested"]

    preview = client.get(f"/api/assets/{asset['id']}/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"


def test_caption_query_returns_grounded_answer_and_trace(client, optical_tif: Path) -> None:
    upload = _upload(client, [optical_tif])
    result = _ask(client, "Describe the land-cover and major objects visible in this image.", upload)

    assert result["task"] == "caption"
    assert result["answer"]
    assert 0 < result["confidence"] <= 0.97
    assert result["inferenceBackend"] == "heuristic-baseline"

    trace = result["trace"]
    assert [step["id"] for step in trace] == ["v", "m", "q", "t", "e", "i"]
    assert all(step["status"] == "done" for step in trace)
    # The trace must name the tools and their parameters to be auditable.
    tools = next(step for step in trace if step["id"] == "e")["outputs"]["tools"]
    assert tools and all({"tool", "model", "backend", "params"} <= entry.keys() for entry in tools)

    assert any(model["name"] for model in result["models"])
    assert result["metrics"]


def test_water_question_reports_area_and_mask(client, optical_tif: Path) -> None:
    upload = _upload(client, [optical_tif])
    result = _ask(client, "How much of this scene is covered by water?", upload)

    assert result["task"] == "vqa"
    assert result["masks"], "a water query should return a mask layer"
    assert all(mask["d"].startswith("M") for mask in result["masks"])
    # Georeferenced input, so at least one metric should be in ground units.
    assert any("km²" in metric["value"] or "ha" in metric["value"] for metric in result["metrics"])


def test_grounding_query_returns_normalised_boxes(client, optical_tif: Path) -> None:
    upload = _upload(client, [optical_tif])
    result = _ask(client, "Highlight the water body referred to in the query.", upload)

    assert result["task"] == "grounding"
    assert result["boxes"], "grounding should produce at least one box"
    for box in result["boxes"]:
        assert 0 <= box["x"] <= 100 and 0 <= box["y"] <= 100
        assert 0 < box["w"] <= 100 and 0 < box["h"] <= 100
        assert 0 <= box["score"] <= 1


def test_cross_modal_pair_runs_fusion(client, optical_tif: Path, sar_tif: Path) -> None:
    upload = _upload(client, [optical_tif, sar_tif])
    assert upload["mode"] == "cross-modal"
    assert {a["modality"] for a in upload["assets"]} == {"optical", "sar"} or "sar" in {
        a["modality"] for a in upload["assets"]
    }

    result = _ask(
        client,
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        upload,
    )
    assert result["task"] == "cross-modal"
    assert result["mode"] == "cross-modal"
    labels = {metric["label"] for metric in result["metrics"]}
    assert "Sensor agreement" in labels
    assert result["compareDefault"] == "swipe"
    # Fusion should cite both sensors in its reasoning.
    assert "SAR" in result["answer"]


def test_bitemporal_pair_answers_change_direction(client, t0_tif: Path, t1_tif: Path) -> None:
    upload = _upload(client, [t0_tif, t1_tif])
    assert upload["mode"] == "bi-temporal"

    result = _ask(client, "Has the built-up area increased, decreased, or remained unchanged?", upload)
    assert result["task"] == "change-vqa"
    verdict = next(step for step in result["trace"] if step["id"] == "e")["outputs"]["tools"][0]["outputs"]["verdict"]
    assert verdict in ("increased", "decreased", "remained essentially unchanged")
    assert "increased" in result["answer"] or "decreased" in result["answer"] or "unchanged" in result["answer"]
    assert any(mask["id"] == "gain" for mask in result["masks"])


def test_change_query_on_single_scene_is_degraded_not_failed(client, optical_tif: Path) -> None:
    upload = _upload(client, [optical_tif])
    result = _ask(client, "What changed between these two dates?", upload)

    assert result["task"] == "vqa"  # degraded to what one scene can answer
    assert any("two dates" in warning for warning in result["warnings"])
    parse_step = next(step for step in result["trace"] if step["id"] == "q")
    assert parse_step["outputs"]["degraded"] is True


def test_measure_query_uses_the_geotransform(client, optical_tif: Path) -> None:
    upload = _upload(client, [optical_tif])
    result = _ask(client, "Measure the area of the largest water body.", upload)
    assert result["task"] == "measure"
    assert any(metric["label"] == "GSD" for metric in result["metrics"])


def test_query_stream_emits_steps_then_result(client, optical_tif: Path) -> None:
    upload = _upload(client, [optical_tif])
    with client.stream(
        "POST",
        "/api/query/stream",
        json={
            "query": "Describe this scene.",
            "assetIds": [a["id"] for a in upload["assets"]],
            "sessionId": upload["sessionId"],
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [chunk for chunk in body.split("\n\n") if chunk.strip()]
    kinds = [line.split("event: ", 1)[1] for chunk in events for line in chunk.splitlines() if line.startswith("event: ")]
    assert kinds[0] == "trace"
    assert kinds[-1] == "result"
    assert kinds.count("step") >= 6

    final = json.loads(events[-1].split("data: ", 1)[1])
    assert final["answer"]


def test_png_without_a_declared_benchmark_warns(client, benchmark_png: bytes) -> None:
    response = client.post(
        "/api/assets",
        files=[("files", ("rsvqa_sample.png", benchmark_png, "image/png"))],
    )
    assert response.status_code == 200
    validation = response.json()["validation"]
    assert validation["ok"] is True
    assert any(issue["code"] == "format-benchmark-only" for issue in validation["issues"])


def test_png_with_declared_benchmark_is_accepted_cleanly(client, benchmark_png: bytes) -> None:
    response = client.post(
        "/api/assets",
        files=[("files", ("rsvqa_sample.png", benchmark_png, "image/png"))],
        data={"benchmark_dataset": "RSVQA"},
    )
    assert response.status_code == 200
    validation = response.json()["validation"]
    codes = {issue["code"] for issue in validation["issues"]}
    assert "benchmark-format" in codes
    assert "format-benchmark-only" not in codes


def test_unsupported_extension_is_rejected(client) -> None:
    response = client.post("/api/assets", files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert response.status_code == 415
    assert "accepted format" in response.json()["detail"]


def test_more_than_two_files_is_rejected(client, optical_tif: Path, sar_tif: Path, t1_tif: Path) -> None:
    files = [
        ("files", (path.name, path.read_bytes(), "image/tiff"))
        for path in (optical_tif, sar_tif, t1_tif)
    ]
    response = client.post("/api/assets", files=files)
    assert response.status_code == 422


def test_query_without_assets_is_rejected(client) -> None:
    response = client.post("/api/query", json={"query": "Describe the scene.", "assetIds": []})
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "input-invalid"
    assert payload["issues"]


def test_unknown_asset_is_reported(client) -> None:
    response = client.post("/api/query", json={"query": "Describe.", "assetIds": ["deadbeef"]})
    assert response.status_code == 404
    assert response.json()["code"] == "asset-not-found"


def test_session_history_and_markdown_report(client, optical_tif: Path) -> None:
    upload = _upload(client, [optical_tif])
    result = _ask(client, "Describe the land-cover in this image.", upload)
    session_id = upload["sessionId"]

    listing = client.get("/api/sessions").json()
    assert any(item["id"] == session_id for item in listing)

    detail = client.get(f"/api/sessions/{session_id}").json()
    assert detail["queryCount"] >= 1
    assert detail["assets"]

    report = client.get(f"/api/sessions/{session_id}/report")
    assert report.status_code == 200
    assert "Execution trace" in report.text
    assert "attachment" in report.headers["content-disposition"]

    single = client.get(f"/api/sessions/{session_id}/report", params={"queryId": result["queryId"]})
    assert single.status_code == 200
    assert result["title"] in single.text

    assert client.delete(f"/api/sessions/{session_id}").json()["deleted"] is True
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_validate_endpoint_is_a_dry_run(client, optical_tif: Path, sar_tif: Path) -> None:
    upload = _upload(client, [optical_tif, sar_tif])
    response = client.post(
        "/api/assets/validate",
        json={"assetIds": [a["id"] for a in upload["assets"]], "mode": "bi-temporal"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "bi-temporal"
    # Declaring a mixed-modality pair as bi-temporal should be flagged.
    assert any(issue["code"] == "mixed-modality-pair" for issue in payload["issues"])
