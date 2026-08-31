"""Working out what kind of image was actually uploaded.

Users are not expected to label their own inputs, so modality comes from
filename hints first (they are usually right and cheap) and pixel statistics
second: SAR amplitude speckles heavily and carries no colour, multispectral
stacks carry more than three bands.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.imagery import Modality, RasterMeta
from app.services.analysis import speckle_index
from app.services.raster import Scene

_SAR_HINTS = ("sar", "risat", "sentinel-1", "sentinel1", "s1a", "s1b", "_s1_", "grd", "radar", "_vv", "_vh", "vv_", "vh_")
_OPTICAL_HINTS = ("optical", "msi", "rgb", "sentinel-2", "sentinel2", "s2a", "s2b", "cartosat", "c2s", "liss", "pan", "tci")
_MULTISPECTRAL_HINTS = ("multispectral", "msi", "bigearthnet", "l2a", "l1c", "mx")

_SENSOR_HINTS = {
    "risat": "RISAT SAR",
    "sentinel-1": "Sentinel-1 SAR",
    "sentinel1": "Sentinel-1 SAR",
    "s1a": "Sentinel-1 SAR",
    "s1b": "Sentinel-1 SAR",
    "sentinel-2": "Sentinel-2 MSI",
    "sentinel2": "Sentinel-2 MSI",
    "s2a": "Sentinel-2 MSI",
    "s2b": "Sentinel-2 MSI",
    "cartosat": "Cartosat-2S",
    "c2s": "Cartosat-2S",
    "liss": "Resourcesat LISS",
    "landsat": "Landsat",
    "bigearthnet": "BigEarthNet patch",
    "vrsbench": "VRSBench sample",
    "rsvqa": "RSVQA sample",
    "cdvqa": "CDVQA sample",
    "levir": "LEVIR-CD sample",
}

_DATE_PATTERNS = (
    re.compile(r"(?P<y>20\d{2})[-_]?(?P<m>0[1-9]|1[0-2])[-_]?(?P<d>0[1-9]|[12]\d|3[01])"),
    re.compile(r"(?P<d>0[1-9]|[12]\d|3[01])[-_](?P<m>0[1-9]|1[0-2])[-_](?P<y>20\d{2})"),
)


def infer_modality(filename: str, scene: Scene) -> tuple[Modality, str, list[str]]:
    """Return (modality, how we decided, notes)."""
    name = filename.lower()
    notes: list[str] = []

    if any(hint in name for hint in _SAR_HINTS):
        return "sar", "filename", [f"Filename hints SAR ({filename})."]

    if scene.bands >= 4:
        return "multispectral", "band-count", [f"{scene.bands} bands present — treated as multispectral."]

    if any(hint in name for hint in _MULTISPECTRAL_HINTS) and scene.bands >= 3:
        notes.append("Filename hints a multispectral product.")

    saturation = scene.saturation
    speckle = speckle_index(scene.gray)
    notes.append(f"Colour saturation {saturation:.3f}, speckle index {speckle:.3f}.")

    grayscale = scene.bands == 1 or saturation < 0.04
    if grayscale and speckle > 0.16:
        notes.append("Grayscale with high speckle — classified as SAR.")
        return "sar", "pixel-statistics", notes
    if grayscale and not any(hint in name for hint in _OPTICAL_HINTS):
        notes.append("Single-band/grayscale input; assumed SAR or panchromatic, treated as SAR.")
        return "sar", "pixel-statistics", notes

    return "optical", "pixel-statistics" if not any(h in name for h in _OPTICAL_HINTS) else "filename", notes


def guess_sensor(filename: str, modality: Modality) -> str:
    name = filename.lower()
    for hint, sensor in _SENSOR_HINTS.items():
        if hint in name:
            return sensor
    return {"sar": "Unknown SAR", "multispectral": "Unknown MSI", "optical": "Unknown optical"}[modality]


def guess_date(filename: str, path: Path) -> str:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(filename)
        if not match:
            continue
        try:
            parsed = datetime(int(match.group("y")), int(match.group("m")), int(match.group("d")))
        except ValueError:
            continue
        return parsed.strftime("%d %b %Y")
    try:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        stamp = datetime.now(tz=timezone.utc)
    return stamp.strftime("%d %b %Y")


def format_gsd(meta: RasterMeta) -> str:
    gsd = meta.gsd_meters
    if not gsd or gsd <= 0:
        return "native"
    if gsd < 1:
        return f"{gsd * 100:.0f} cm"
    return f"{gsd:.1f} m" if gsd < 10 else f"{gsd:.0f} m"


def describe_position(meta: RasterMeta) -> tuple[str, str]:
    """Return (location, coords) for the asset chip in the imagery stage."""
    if not meta.georeferenced or not meta.bounds:
        return "Local file", "no geotransform"

    left, bottom, right, top = meta.bounds
    cx = (left + right) / 2
    cy = (bottom + top) / 2
    crs = (meta.crs or "").upper()

    if crs.endswith("4326") or (abs(cx) <= 180 and abs(cy) <= 90):
        ns = "N" if cy >= 0 else "S"
        ew = "E" if cx >= 0 else "W"
        return "Georeferenced scene", f"{abs(cy):.2f}° {ns}, {abs(cx):.2f}° {ew}"

    return "Georeferenced scene", f"{cx:.0f}, {cy:.0f} ({meta.crs or 'projected'})"
