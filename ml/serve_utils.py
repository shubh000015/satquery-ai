"""Image + prompt helpers shared by ml/serve.py and ml/chat.py.

Everything here is pure NumPy / Pillow. Torch stays out of this module so it
can be imported in lightweight tools (like chat.py's HTTP client mode) without
paying the model-load cost.

Two things matter:

1. **SAR rendering must match training.** The dataset builder rendered every
   S1 patch as `R=VV, G=VH, B=|VV-VH|` and stretched it with a 2/98 percentile
   clip (see ml/dataset_builder/build_bigearthnet_vqa.py). If we hand the model
   raw VV/VH bytes at inference we get garbage. Every helper below produces the
   same 8-bit RGB PIL image the model saw during QLoRA training.

2. **Training system prompt is verbatim** what kaggle_train_unsloth.ipynb used.
   Changing it silently kills answer quality — Qwen was fine-tuned to the
   exact wording.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


TRAIN_SYSTEM_PROMPT = (
    "You are SatQuery, an assistant for satellite and aerial imagery. "
    "Answer questions about optical and SAR remote sensing scenes "
    "concisely and factually."
)

# Qwen2.5-VL's official min/max pixel budget for a single image. 256 * 28*28
# = ~200k pixels is enough for 224x224 chips; 640 * 28*28 covers 720p rasters
# without blowing the vision encoder budget.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 640 * 28 * 28

# BigEarthNet was rendered at 224x224 (the size the model saw). We resize at
# ingest so a user-uploaded 1200x1200 tile behaves identically to a training
# patch. Not strictly required — Qwen tolerates other sizes — but it removes
# a variable when debugging low answer quality.
TRAIN_TILE = 224


# ---------------------------------------------------------------------------
# Byte-level detection
# ---------------------------------------------------------------------------

def is_tiff(head: bytes) -> bool:
    """Recognise TIFF/GeoTIFF/BigTIFF by its 4-byte magic."""
    return head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


# ---------------------------------------------------------------------------
# Pixel processing (identical to ml/dataset_builder/build_bigearthnet_vqa.py)
# ---------------------------------------------------------------------------

def percentile_stretch(arr: np.ndarray) -> np.ndarray | None:
    """Clip to the 2nd / 98th percentiles and rescale to [0, 1].

    Returns `None` if the input has no dynamic range (a fully-uniform tile).
    """
    lo, hi = np.percentile(arr, (2, 98))
    if hi <= lo:
        return None
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def render_s1_pseudo_rgb(vv: np.ndarray, vh: np.ndarray | None) -> Image.Image:
    """Build the SAR pseudo-RGB the training set was rendered with.

    R=VV, G=VH, B=|VV-VH|. If VH is missing (unusual), we fall back to VV in
    all three channels so at least VV information reaches the model.
    """
    vv = vv.astype(np.float32)
    if vh is None:
        vh = vv.copy()
    else:
        vh = vh.astype(np.float32)

    # Two BigEarthNet bands are guaranteed to be the same shape, but a stray
    # off-by-one geotransform can differ; crop to the common region.
    h = min(vv.shape[0], vh.shape[0])
    w = min(vv.shape[1], vh.shape[1])
    vv, vh = vv[:h, :w], vh[:h, :w]
    diff = np.abs(vv - vh)

    stack = np.stack([vv, vh, diff], axis=2)
    stretched = percentile_stretch(stack)
    if stretched is None:
        stretched = np.zeros_like(stack)
    return _finish(stretched)


def render_s2_pseudo_rgb(b04: np.ndarray, b03: np.ndarray, b02: np.ndarray) -> Image.Image:
    """Sentinel-2 true-colour composite from the red/green/blue bands."""
    stack = np.stack(
        [b04.astype(np.float32), b03.astype(np.float32), b02.astype(np.float32)],
        axis=2,
    )
    stretched = percentile_stretch(stack)
    if stretched is None:
        stretched = np.zeros_like(stack)
    return _finish(stretched)


def render_grayscale(arr: np.ndarray) -> Image.Image:
    """Replicate a single band into R=G=B so the vision encoder is happy."""
    stack = np.stack([arr, arr, arr], axis=2).astype(np.float32)
    stretched = percentile_stretch(stack)
    if stretched is None:
        stretched = np.zeros_like(stack)
    return _finish(stretched)


def _finish(stretched: np.ndarray) -> Image.Image:
    img = Image.fromarray((stretched * 255).astype(np.uint8), mode="RGB")
    return img.resize((TRAIN_TILE, TRAIN_TILE), Image.BICUBIC)


# ---------------------------------------------------------------------------
# High-level ingest — accept anything the user might drop on the CLI
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> tuple[Image.Image, str]:
    """Turn a file or folder into a training-shaped PIL image.

    Returns `(image, source_label)` where `source_label` explains what we
    rendered — handy for chat.py to print in the banner.

    Supported inputs:
      - a directory containing `<id>_VV.tif` + optional `<id>_VH.tif`
        (BigEarthNet-S1 patch folder)  -> pseudo-RGB SAR
      - a directory containing `<id>_B04.tif`, `<id>_B03.tif`, `<id>_B02.tif`
        (BigEarthNet-S2 patch folder)  -> true-colour S2
      - a single `.tif` file           -> read as grayscale (VV)
      - anything Pillow understands (`.png`, `.jpg`, …)
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"No such image: {p}")

    if p.is_dir():
        img, label = _load_directory(p)
        return img, f"{label} :: {p.name}"

    suffix = p.suffix.lower()
    if suffix in (".tif", ".tiff"):
        arr = np.asarray(Image.open(p), dtype=np.float32)
        return render_grayscale(arr), f"grayscale TIFF :: {p.name}"

    img = Image.open(p).convert("RGB")
    if img.size != (TRAIN_TILE, TRAIN_TILE):
        img = img.resize((TRAIN_TILE, TRAIN_TILE), Image.BICUBIC)
    return img, f"raster :: {p.name}"


def _load_directory(folder: Path) -> tuple[Image.Image, str]:
    tifs = {f.name.lower(): f for f in folder.iterdir() if f.suffix.lower() in (".tif", ".tiff")}

    # Sentinel-1 patch: something_VV.tif (+ optional something_VH.tif)
    vv = next((f for name, f in tifs.items() if name.endswith("_vv.tif")), None)
    if vv is not None:
        vh_name = vv.stem[:-3] + "_VH.tif"
        vh = folder / vh_name if (folder / vh_name).exists() else None
        vv_arr = np.asarray(Image.open(vv), dtype=np.float32)
        vh_arr = np.asarray(Image.open(vh), dtype=np.float32) if vh else None
        return render_s1_pseudo_rgb(vv_arr, vh_arr), "S1 SAR pseudo-RGB (VV/VH/|VV-VH|)"

    # Sentinel-2 patch: <id>_B04.tif etc.
    b04 = next((f for name, f in tifs.items() if name.endswith("_b04.tif")), None)
    b03 = next((f for name, f in tifs.items() if name.endswith("_b03.tif")), None)
    b02 = next((f for name, f in tifs.items() if name.endswith("_b02.tif")), None)
    if b04 and b03 and b02:
        return (
            render_s2_pseudo_rgb(
                np.asarray(Image.open(b04), dtype=np.float32),
                np.asarray(Image.open(b03), dtype=np.float32),
                np.asarray(Image.open(b02), dtype=np.float32),
            ),
            "S2 true-colour (B04/B03/B02)",
        )

    raise ValueError(
        f"{folder} is not a BigEarthNet patch folder (no *_VV.tif or "
        f"*_B04.tif/*_B03.tif/*_B02.tif). Pass a specific file instead."
    )


def image_to_b64(img: Image.Image, format: str = "PNG") -> str:
    """Serialize a PIL image to base64 for the HTTP API."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_image(data: str) -> Image.Image:
    """Inverse of image_to_b64 — with a TIFF fallback so serve.py accepts a
    raw single-band GeoTIFF if a caller ever sends one."""
    raw = base64.b64decode(data)
    if is_tiff(raw[:4]):
        arr = np.asarray(Image.open(io.BytesIO(raw)), dtype=np.float32)
        return render_grayscale(arr)
    return Image.open(io.BytesIO(raw)).convert("RGB")
