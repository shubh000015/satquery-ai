"""Canonical multispectral band stack, per-model band selection, and wire format.

Every model in the registry wants a different slice of the same scene, and they
disagree in ways that silently produce garbage if you feed the wrong thing:

  * BigEarthNet (BIFOLD reBEN) S1+S2 -> 12 channels, one tensor:
    VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
  * CROMA -> two tensors: SAR (VV, VH) and optical (12 S2 bands, cirrus removed)
  * ChangeFormer, Grounding DINO, SAM 2, the VLM -> plain 3-channel RGB

So the transport carries a named band stack rather than an image, and each
adapter asks for the slice it was trained on. A model whose bands are missing
says so instead of being handed zero-filled channels that look like valid data.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np

# Sentinel-1 backscatter, then Sentinel-2 in native band order. B10 (cirrus) is
# excluded throughout: it carries no surface signal and every model we use drops it.
SAR_BANDS: tuple[str, ...] = ("VV", "VH")
S2_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)
CANONICAL_BANDS: tuple[str, ...] = SAR_BANDS + S2_BANDS

RGB_BANDS: tuple[str, ...] = ("B04", "B03", "B02")

# BigEarthNet v2.0 (reBEN) published weights. Order is load-bearing.
BEN_S1_BANDS: tuple[str, ...] = ("VV", "VH")
BEN_S2_BANDS: tuple[str, ...] = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
)
BEN_ALL_BANDS: tuple[str, ...] = BEN_S1_BANDS + BEN_S2_BANDS

# CROMA: SAR fixed at 2 channels, optical fixed at 12.
CROMA_SAR_BANDS: tuple[str, ...] = SAR_BANDS
CROMA_OPTICAL_BANDS: tuple[str, ...] = S2_BANDS

# 60 m atmospheric bands CROMA wants but BigEarthNet (and our smoke stack)
# do not store. Fill from the nearest surface band rather than refusing the
# whole encoder — B01/B09 carry almost no spatial structure at 10 m.
ATMOSPHERIC_FILL: dict[str, str] = {
    "B01": "B02",  # coastal aerosol -> blue
    "B09": "B8A",  # water vapour -> narrow NIR
}


class MissingBands(ValueError):
    """Raised when a scene lacks bands a model requires."""

    def __init__(self, model: str, missing: tuple[str, ...], available: tuple[str, ...]):
        self.model = model
        self.missing = missing
        self.available = available
        super().__init__(
            f"{model} needs bands {list(missing)} which this scene does not carry. "
            f"Available: {list(available)}."
        )


@dataclass(slots=True)
class BandStack:
    """A scene as named channels.

    `array` is (H, W, C) float32, already scaled to roughly 0..1. `names[i]`
    labels channel i. Bands may be absent: an optical-only upload has no VV/VH,
    and a SAR-only upload has no B-bands.
    """

    array: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError(f"array must be (H, W, C), got shape {self.array.shape}")
        if self.array.shape[2] != len(self.names):
            raise ValueError(
                f"{self.array.shape[2]} channels but {len(self.names)} band names"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate band names: {self.names}")

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    def has(self, *names: str) -> bool:
        return set(names).issubset(self.names)

    def missing(self, names: tuple[str, ...]) -> tuple[str, ...]:
        present = set(self.names)
        return tuple(name for name in names if name not in present)

    @property
    def has_sar(self) -> bool:
        return self.has(*SAR_BANDS)

    @property
    def has_optical(self) -> bool:
        return self.has(*RGB_BANDS)

    @property
    def modality(self) -> str:
        """What this scene actually is, from the bands present rather than a guess."""
        if self.has_sar and self.has_optical:
            return "optical-sar"
        if self.has_sar:
            return "sar"
        return "optical"

    def select(self, names: tuple[str, ...], model: str = "model") -> np.ndarray:
        """(C, H, W) float32 in exactly the requested band order."""
        missing = self.missing(names)
        if missing:
            raise MissingBands(model, missing, self.names)

        index = {name: i for i, name in enumerate(self.names)}
        planes = [self.array[:, :, index[name]] for name in names]
        return np.stack(planes, axis=0).astype(np.float32, copy=False)

    def rgb(self) -> np.ndarray:
        """(H, W, 3) uint8 for the RGB-only models and for previews.

        Falls back to a SAR false-colour composite when there is no optical
        data, which is the same rendering the dataset builder used.
        """
        if self.has_optical:
            stack = np.stack(
                [self.array[:, :, self.names.index(b)] for b in RGB_BANDS], axis=2
            )
        elif self.has_sar:
            vv = self.array[:, :, self.names.index("VV")]
            vh = self.array[:, :, self.names.index("VH")]
            stack = np.stack([vv, vh, np.abs(vv - vh)], axis=2)
        else:
            gray = self.array[:, :, 0]
            stack = np.stack([gray, gray, gray], axis=2)

        return (np.clip(_stretch(stack), 0.0, 1.0) * 255.0).astype(np.uint8)

    def for_croma(self) -> tuple["BandStack", tuple[str, ...]]:
        """A stack CROMA can consume, plus the names that were synthesized.

        CROMA's 12 optical channels include B01 and B09. BigEarthNet's published
        10-band set does not. Those two are 60 m atmospheric bands, so copying
        B02 and B8A is an honest stand-in. Any other missing band still refuses.
        """
        needed = CROMA_SAR_BANDS + CROMA_OPTICAL_BANDS
        missing = self.missing(needed)
        unsynth = tuple(name for name in missing if name not in ATMOSPHERIC_FILL)
        if unsynth:
            raise MissingBands("CROMA-base", unsynth, self.names)

        planes = {name: self.array[:, :, i] for i, name in enumerate(self.names)}
        synthesized: list[str] = []
        for name in missing:
            source = ATMOSPHERIC_FILL[name]
            if source not in planes:
                raise MissingBands("CROMA-base", (name, source), self.names)
            planes[name] = planes[source]
            synthesized.append(name)
        return BandStack.from_planes(planes), tuple(synthesized)

    def subset(self, names: tuple[str, ...]) -> BandStack:
        selected = self.select(names)
        return BandStack(array=np.transpose(selected, (1, 2, 0)), names=names)

    # ---- wire format -----------------------------------------------------

    def encode(self) -> dict:
        """Compressed .npz in base64, plus the band names.

        JSON arrays of floats would be roughly 6x larger and lossy at the edges;
        npz keeps this exact and small enough to POST.
        """
        buffer = io.BytesIO()
        np.savez_compressed(buffer, array=self.array.astype(np.float32))
        return {
            "bands": list(self.names),
            "npz": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "height": self.height,
            "width": self.width,
        }

    @classmethod
    def decode(cls, payload: dict) -> BandStack:
        names = tuple(payload["bands"])
        raw = base64.b64decode(payload["npz"])
        with np.load(io.BytesIO(raw)) as loaded:
            array = loaded["array"]
        return cls(array=np.asarray(array, dtype=np.float32), names=names)

    # ---- construction helpers -------------------------------------------

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> BandStack:
        """Wrap an ordinary RGB image as B04/B03/B02.

        Lets the RGB-only models run on PNG/JPEG benchmark inputs. The band
        names are honest about what is there, so a model needing B08 will
        refuse rather than quietly consume red as near-infrared.
        """
        array = np.asarray(rgb, dtype=np.float32)
        if array.ndim == 2:
            array = array[:, :, None]
        if array.max() > 1.5:
            array = array / 255.0
        if array.shape[2] < 3:
            array = np.repeat(array[:, :, :1], 3, axis=2)
        return cls(array=array[:, :, :3].astype(np.float32), names=RGB_BANDS)

    @classmethod
    def from_planes(cls, planes: dict[str, np.ndarray]) -> BandStack:
        """Build from {band name: 2-D array}, ordered canonically.

        Unknown band names are rejected rather than silently appended, so a
        typo cannot produce a stack that no model can interpret.
        """
        unknown = sorted(set(planes) - set(CANONICAL_BANDS))
        if unknown:
            raise ValueError(
                f"unknown band names {unknown}; expected a subset of {list(CANONICAL_BANDS)}"
            )
        if not planes:
            raise ValueError("no bands supplied")

        names = tuple(band for band in CANONICAL_BANDS if band in planes)
        shapes = {planes[name].shape for name in names}
        if len(shapes) != 1:
            raise ValueError(f"all bands must share one shape, got {sorted(shapes)}")

        array = np.stack([planes[name] for name in names], axis=2)
        return cls(array=array.astype(np.float32), names=names)


def _stretch(array: np.ndarray) -> np.ndarray:
    """2-98 percentile contrast stretch, matching the dataset builder's rendering."""
    low, high = np.percentile(array, (2, 98))
    if high <= low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def resize_chw(array: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize of a (C, H, W) stack to (C, size, size).

    Uses torch so it runs on any channel count; PIL would force us down to
    3 channels, which is the whole thing we are avoiding here.
    """
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).numpy()
