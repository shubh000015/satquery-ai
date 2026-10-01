"""Band-plane grouping: 12 optical files or 2 SAR files become one scene each."""

from pathlib import Path

from app.services.bands_ingest import parse_band_token, plan_scenes, scene_stem
from tests.conftest import write_geotiff, _base_scene, _sar_scene


def test_parse_sentinel_and_sar_tokens() -> None:
    assert parse_band_token("T43SFR_20240715T053641_B04_10m.tif") == "B04"
    assert parse_band_token("scene_B8A.tif") == "B8A"
    assert parse_band_token("S1A_IW_GRDH_VV.tif") == "VV"
    assert parse_band_token("patch_VH.tif") == "VH"
    assert parse_band_token("kosi_optical_20240715.tif") is None


def test_twelve_optical_planes_are_one_scene() -> None:
    names = [f"kosi_B{b}.tif" for b in ("01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "11", "12")]
    plan = plan_scenes(names, band_counts=[1] * 12)
    assert len(plan.scenes) == 1
    assert plan.scenes[0].kind == "optical"
    assert plan.scenes[0].complete is True
    assert plan.ready is True


def test_partial_optical_stack_is_not_ready() -> None:
    names = ["flood_B02.tif", "flood_B03.tif", "flood_B04.tif"]
    plan = plan_scenes(names, band_counts=[1, 1, 1], expect="optical")
    assert len(plan.scenes) == 1
    assert plan.scenes[0].complete is False
    assert "B01" in plan.scenes[0].missing
    assert plan.ready is False


def test_vv_vh_are_one_sar_scene_not_a_pair() -> None:
    plan = plan_scenes(["kosi_VV.tif", "kosi_VH.tif"], band_counts=[1, 1])
    assert len(plan.scenes) == 1
    assert plan.scenes[0].kind == "sar"
    assert plan.scenes[0].complete is True
    assert plan.ready is True


def test_one_sar_polarisation_is_incomplete_when_declared() -> None:
    plan = plan_scenes(["kosi_VV.tif"], band_counts=[1], expect="sar")
    assert plan.scenes[0].complete is False
    assert plan.ready is False


def test_optical_plus_sar_stacks_are_two_scenes() -> None:
    optical = [f"fan_B{b}.tif" for b in ("01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "11", "12")]
    names = optical + ["fan_VV.tif", "fan_VH.tif"]
    plan = plan_scenes(names, band_counts=[1] * 14, expect="fusion")
    kinds = {scene.kind for scene in plan.scenes}
    assert kinds == {"optical", "sar"}
    assert all(scene.complete for scene in plan.scenes)
    assert plan.ready is True


def test_two_rgb_chips_stay_two_scenes() -> None:
    plan = plan_scenes(["left.jpg", "right.jpg"], band_counts=[3, 3])
    assert len(plan.scenes) == 2
    assert plan.ready is True


def test_stem_strips_band_token() -> None:
    assert scene_stem("kosi_B04.tif") == scene_stem("kosi_B8A.tif")


def test_upload_stacks_twelve_planes(client, tmp_path: Path) -> None:
    scene = _base_scene(64)[:, :, :1]
    files = []
    for band in ("01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "11", "12"):
        path = write_geotiff(tmp_path / f"patch_B{band}.tif", scene)
        files.append(("files", (path.name, path.read_bytes(), "image/tiff")))
    response = client.post("/api/assets", files=files, data={"expect": "optical"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["assets"]) == 1
    asset = payload["assets"][0]
    assert asset["meta"]["bands"] == 12
    assert "B04" in asset["bandNames"]
    assert payload["validation"]["ok"] is True


def test_upload_stacks_sar_pair(client, tmp_path: Path) -> None:
    plane = _sar_scene(64)
    files = []
    for pol in ("VV", "VH"):
        path = write_geotiff(tmp_path / f"patch_{pol}.tif", plane)
        files.append(("files", (path.name, path.read_bytes(), "image/tiff")))
    response = client.post("/api/assets", files=files, data={"expect": "sar"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["assets"]) == 1
    assert payload["assets"][0]["modality"] == "sar"
    assert set(payload["assets"][0]["bandNames"]) == {"VV", "VH"}


def test_incomplete_named_optical_fails_validation(client, tmp_path: Path) -> None:
    scene = _base_scene(64)[:, :, :1]
    files = []
    for band in ("02", "03", "04"):
        path = write_geotiff(tmp_path / f"patch_B{band}.tif", scene)
        files.append(("files", (path.name, path.read_bytes(), "image/tiff")))
    response = client.post("/api/assets", files=files, data={"expect": "optical"})
    assert response.status_code == 200, response.text
    validation = response.json()["validation"]
    assert validation["ok"] is True
    assert any(
        issue["code"] == "incomplete-optical-bands" and issue["severity"] == "warning"
        for issue in validation["issues"]
    )
