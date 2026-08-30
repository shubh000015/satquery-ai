"""Reading pixels and geospatial metadata off uploaded scenes.

rasterio is the preferred path because it understands GeoTIFF properly. It is an
optional dependency, so there is a Pillow fallback that parses the GeoTIFF tags
by hand (pixel scale, tiepoint, GeoKey directory) to recover CRS and GSD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.schemas.imagery import RasterMeta

try:  # pragma: no cover - depends on local install
    import rasterio
    from rasterio.enums import Resampling

    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover
    rasterio = None
    Resampling = None
    RASTERIO_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None

_TAG_PIXEL_SCALE = 33550
_TAG_TIEPOINT = 33922
_TAG_GEOKEYS = 34735
_GEOKEY_PROJECTED_CS = 3072
_GEOKEY_GEOGRAPHIC_CS = 2048
_METERS_PER_DEGREE = 111_320.0


@dataclass(slots=True)
class Scene:
    """Decimated pixels ready for analysis.

    Two views of the same data, because they are good at different things:
    `array` is per-band contrast-stretched (right for brightness, texture and
    previews) while `reflectance` is only scaled by the dtype range (right for
    band ratios like NDWI/NDVI, which a per-band stretch would destroy).
    """

    path: Path
    array: np.ndarray  # (H, W, C) float32 in 0..1, percentile-stretched
    reflectance: np.ndarray  # (H, W, C) float32 in 0..1, dtype-scaled only
    meta: RasterMeta

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def bands(self) -> int:
        return int(self.array.shape[2])

    @property
    def nir_index(self) -> int | None:
        """MSI stacks are conventionally R,G,B,NIR once stacked for display."""
        return 3 if self.bands >= 4 else None

    @property
    def gray(self) -> np.ndarray:
        if self.bands == 1:
            return self.array[:, :, 0]
        if self.bands >= 3:
            rgb = self.array[:, :, :3]
            return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return self.array.mean(axis=2)

    def band(self, index: int) -> np.ndarray:
        return self.array[:, :, min(index, self.bands - 1)]

    def refl(self, index: int) -> np.ndarray:
        """Band for index arithmetic (no per-band stretch applied)."""
        return self.reflectance[:, :, min(index, self.bands - 1)]

    @property
    def refl_gray(self) -> np.ndarray:
        return self.reflectance.mean(axis=2)

    @property
    def saturation(self) -> float:
        """0 for grayscale (SAR/PAN), higher for true colour."""
        if self.bands < 3:
            return 0.0
        rgb = self.array[:, :, :3]
        return float(np.mean(rgb.max(axis=2) - rgb.min(axis=2)))


def _epsg_from_geokeys(values: tuple[int, ...] | list[int]) -> str | None:
    if len(values) < 4:
        return None
    count = int(values[3])
    for i in range(count):
        base = 4 + i * 4
        if base + 3 >= len(values):
            break
        key_id, location, _, value = values[base : base + 4]
        if location != 0:
            continue
        if key_id in (_GEOKEY_PROJECTED_CS, _GEOKEY_GEOGRAPHIC_CS) and value not in (0, 32767):
            return f"EPSG:{int(value)}"
    return None


def _gsd_to_meters(pixel_size: float, crs: str | None) -> float:
    """Geographic CRS pixel sizes come out in degrees; normalise to metres."""
    if pixel_size <= 0:
        return 0.0
    looks_geographic = pixel_size < 0.01 and (crs is None or crs.endswith(("4326", "4979")))
    if looks_geographic:
        return pixel_size * _METERS_PER_DEGREE
    return pixel_size


def _probe_with_pillow(path: Path) -> RasterMeta:
    with Image.open(path) as im:
        width, height = im.size
        bands = len(im.getbands())
        dtype = {"I;16": "uint16", "I": "int32", "F": "float32"}.get(im.mode, "uint8")
        tags = getattr(im, "tag_v2", {}) or {}
        crs = None
        gsd = None
        bounds = None

        geokeys = tags.get(_TAG_GEOKEYS)
        if geokeys:
            crs = _epsg_from_geokeys(geokeys)

        scale = tags.get(_TAG_PIXEL_SCALE)
        if scale and len(scale) >= 2 and scale[0]:
            gsd = _gsd_to_meters(abs(float(scale[0])), crs)

        tiepoint = tags.get(_TAG_TIEPOINT)
        if tiepoint and len(tiepoint) >= 6 and scale and len(scale) >= 2:
            left = float(tiepoint[3])
            top = float(tiepoint[4])
            sx = abs(float(scale[0]))
            sy = abs(float(scale[1]))
            bounds = [left, top - sy * height, left + sx * width, top]

        georeferenced = bool(geokeys or tiepoint)
        driver = "TIFF" if (im.format or "").upper() == "TIFF" else (im.format or "UNKNOWN")

    return RasterMeta(
        width=width,
        height=height,
        bands=bands,
        dtype=dtype,
        georeferenced=georeferenced,
        crs=crs,
        gsd_meters=gsd,
        bounds=bounds,
        driver=driver,
    )


def probe(path: Path) -> RasterMeta:
    """Read dimensions, band count and georeferencing without loading all pixels."""
    if RASTERIO_AVAILABLE:
        try:  # pragma: no cover - exercised only with rasterio installed
            with rasterio.open(path) as ds:
                crs = str(ds.crs) if ds.crs else None
                gsd = _gsd_to_meters(abs(ds.transform.a), crs)
                return RasterMeta(
                    width=ds.width,
                    height=ds.height,
                    bands=ds.count,
                    dtype=str(ds.dtypes[0]),
                    georeferenced=ds.crs is not None,
                    crs=crs,
                    gsd_meters=gsd or None,
                    bounds=[float(v) for v in ds.bounds],
                    driver=ds.driver,
                    nodata=float(ds.nodata) if ds.nodata is not None else None,
                )
        except Exception:
            pass
    return _probe_with_pillow(path)


def _stretch(band: np.ndarray) -> np.ndarray:
    """Percentile stretch so 16-bit and float rasters land on a comparable 0..1."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(finite, (2.0, 98.0))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-9:
        lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        return np.zeros_like(band, dtype=np.float32)
    out = (band - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0).astype(np.float32)


_DTYPE_RANGES = {
    "uint8": 255.0,
    "int8": 127.0,
    "uint16": 65535.0,
    "int16": 32767.0,
    "uint32": 4294967295.0,
    "int32": 2147483647.0,
}


def _to_reflectance(array: np.ndarray, dtype: str) -> np.ndarray:
    """Scale to 0..1 by the dtype range so band ratios stay physically meaningful."""
    scale = _DTYPE_RANGES.get(dtype)
    data = np.nan_to_num(array.astype(np.float32), nan=0.0)
    if scale is None:  # float raster: already reflectance, or needs a soft rescale
        peak = float(np.nanpercentile(data, 99.9)) if data.size else 1.0
        scale = peak if peak > 1.5 else 1.0
    return np.clip(data / max(scale, 1e-6), 0.0, 1.0)


def _target_shape(width: int, height: int, max_edge: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_edge:
        return height, width
    factor = max_edge / longest
    return max(1, int(round(height * factor))), max(1, int(round(width * factor)))


def load_scene(path: Path, max_edge: int = 512) -> Scene:
    """Load a decimated, normalised copy of the raster for analysis."""
    meta = probe(path)
    out_h, out_w = _target_shape(meta.width, meta.height, max_edge)

    array: np.ndarray | None = None
    if RASTERIO_AVAILABLE:
        try:  # pragma: no cover - exercised only with rasterio installed
            with rasterio.open(path) as ds:
                count = min(ds.count, 8)
                data = ds.read(
                    indexes=list(range(1, count + 1)),
                    out_shape=(count, out_h, out_w),
                    resampling=Resampling.average,
                    masked=False,
                )
                array = np.transpose(data.astype(np.float32), (1, 2, 0))
                if ds.nodata is not None:
                    array = np.where(array == ds.nodata, np.nan, array)
        except Exception:
            array = None

    if array is None:
        with Image.open(path) as im:
            im.seek(0)
            if im.mode not in ("RGB", "RGBA", "L", "I;16", "I", "F"):
                im = im.convert("RGB")
            resized = im.resize((out_w, out_h), Image.BILINEAR)
            raw = np.asarray(resized).astype(np.float32)
        array = raw[:, :, None] if raw.ndim == 2 else raw

    if array.shape[2] == 4 and meta.bands == 4 and meta.driver in ("PNG", "JPEG"):
        # RGBA display alpha is not a spectral band.
        array = array[:, :, :3]

    stretched = np.stack([_stretch(array[:, :, i]) for i in range(array.shape[2])], axis=2)
    reflectance = _to_reflectance(array, meta.dtype)
    return Scene(path=path, array=stretched, reflectance=reflectance, meta=meta)


def write_preview(path: Path, dest: Path, max_edge: int = 1024) -> Path:
    """Browsers cannot display GeoTIFF, so keep an 8-bit PNG next to the upload."""
    scene = load_scene(path, max_edge=max_edge)
    data = scene.array
    if data.shape[2] == 1:
        rgb = np.repeat(data, 3, axis=2)
    elif data.shape[2] == 2:
        rgb = np.concatenate([data, data[:, :, :1]], axis=2)
    else:
        rgb = data[:, :, :3]
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB").save(dest, format="PNG")
    return dest
