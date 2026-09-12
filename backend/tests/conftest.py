"""Test fixtures, including synthetic GeoTIFF scenes.

Scenes are generated rather than checked in: a dark river band on a bright land
background gives the water index something real to threshold, and the t1 variant
adds a bright block so change detection has an actual change to find.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

DEG_PER_METER = 1.0 / 111_320.0


@pytest.fixture(scope="session", autouse=True)
def _isolated_data_dir(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the app at a throwaway data directory before anything imports it."""
    import os

    data_dir = tmp_path_factory.mktemp("satquery-data")
    os.environ["SATQUERY_DATA_DIR"] = str(data_dir)
    os.environ["SATQUERY_STRICT_FORMAT_POLICY"] = "false"
    # Neutralize the committed Kaggle URL so API tests stay on the baseline
    # and never wait on a live tunnel.
    for key in (
        "SATQUERY_ML_ENDPOINT",
        "SATQUERY_VLM_ENDPOINT",
        "SATQUERY_GROUNDING_ENDPOINT",
        "SATQUERY_CHANGE_ENDPOINT",
        "SATQUERY_FUSION_ENDPOINT",
    ):
        os.environ[key] = ""

    from app.core.config import get_settings

    get_settings.cache_clear()

    import app.agent.controller as controller_module
    import app.services.sessions as sessions_module
    import app.services.storage as storage_module

    storage_module._store = None
    sessions_module._store = None
    controller_module._controller = None


def _base_scene(size: int = 192) -> np.ndarray:
    """RGB + NIR stack: bright land, dark water channel, a built-up block."""
    rng = np.random.default_rng(7)
    red = np.full((size, size), 0.55, dtype=np.float32)
    green = np.full((size, size), 0.5, dtype=np.float32)
    blue = np.full((size, size), 0.42, dtype=np.float32)
    nir = np.full((size, size), 0.7, dtype=np.float32)

    # Vegetated left third: high NIR, low red.
    red[:, : size // 3] = 0.2
    nir[:, : size // 3] = 0.85

    # Water channel down the middle: dark, blue-leaning, NIR collapses.
    channel = slice(size // 2 - 12, size // 2 + 12)
    red[:, channel] = 0.1
    green[:, channel] = 0.16
    blue[:, channel] = 0.3
    nir[:, channel] = 0.05

    # Built-up block: bright and flat across all bands.
    block = (slice(20, 60), slice(size - 70, size - 20))
    red[block] = 0.9
    green[block] = 0.88
    blue[block] = 0.86
    nir[block] = 0.8

    stack = np.stack([red, green, blue, nir], axis=2)
    stack += rng.normal(0, 0.01, stack.shape).astype(np.float32)
    return np.clip(stack, 0, 1)


def _sar_scene(size: int = 192) -> np.ndarray:
    """Single-band amplitude with speckle; water reads dark, plus a wet area the
    optical scene does not show (the cloud-penetration argument)."""
    rng = np.random.default_rng(11)
    amplitude = np.full((size, size), 0.6, dtype=np.float32)
    channel = slice(size // 2 - 12, size // 2 + 12)
    amplitude[:, channel] = 0.05
    amplitude[size - 60 : size - 20, 30:90] = 0.06  # SAR-only water
    speckle = rng.gamma(shape=4.0, scale=0.25, size=amplitude.shape).astype(np.float32)
    return np.clip(amplitude * speckle, 0, 1)[:, :, None]


def _changed_scene(size: int = 192) -> np.ndarray:
    scene = _base_scene(size)
    scene[110:160, 40:110, :3] = 0.92  # new built-up
    scene[110:160, 40:110, 3] = 0.75
    return np.clip(scene, 0, 1)


def _geotiff_tags(gsd_meters: float, origin: tuple[float, float]) -> ImageFileDirectory_v2:
    ifd = ImageFileDirectory_v2()
    scale = gsd_meters * DEG_PER_METER
    ifd[33550] = (scale, scale, 0.0)
    ifd.tagtype[33550] = 12
    ifd[33922] = (0.0, 0.0, 0.0, origin[0], origin[1], 0.0)
    ifd.tagtype[33922] = 12
    ifd[34735] = (1, 1, 0, 1, 2048, 0, 1, 4326)
    ifd.tagtype[34735] = 3
    return ifd


def write_geotiff(
    path: Path,
    array: np.ndarray,
    gsd_meters: float = 10.0,
    origin: tuple[float, float] = (77.0, 28.6),
) -> Path:
    """Write a georeferenced TIFF, via rasterio when available."""
    data = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    try:
        import rasterio
        from rasterio.transform import from_origin

        scale = gsd_meters * DEG_PER_METER
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=data.shape[2],
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(origin[0], origin[1], scale, scale),
        ) as dst:
            for band in range(data.shape[2]):
                dst.write(data[:, :, band], band + 1)
        return path
    except Exception:
        pass

    if data.shape[2] == 1:
        image = Image.fromarray(data[:, :, 0], mode="L")
    else:
        image = Image.fromarray(data[:, :, :3], mode="RGB")
    image.save(path, format="TIFF", tiffinfo=_geotiff_tags(gsd_meters, origin))
    return path


def png_bytes(array: np.ndarray) -> bytes:
    data = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    mode = "L" if data.shape[2] == 1 else "RGB"
    payload = data[:, :, 0] if mode == "L" else data[:, :, :3]
    buffer = io.BytesIO()
    Image.fromarray(payload, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def optical_tif(tmp_path: Path) -> Path:
    return write_geotiff(tmp_path / "kosi_optical_20240715.tif", _base_scene())


@pytest.fixture
def sar_tif(tmp_path: Path) -> Path:
    return write_geotiff(tmp_path / "kosi_risat_sar_20240715.tif", _sar_scene())


@pytest.fixture
def t0_tif(tmp_path: Path) -> Path:
    return write_geotiff(tmp_path / "gurugram_optical_20200115.tif", _base_scene(), gsd_meters=5.0)


@pytest.fixture
def t1_tif(tmp_path: Path) -> Path:
    return write_geotiff(tmp_path / "gurugram_optical_20240415.tif", _changed_scene(), gsd_meters=5.0)


@pytest.fixture
def benchmark_png() -> bytes:
    return png_bytes(_base_scene(128))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
