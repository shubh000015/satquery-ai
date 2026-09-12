"""Bi-temporal change: ChangeFormerV6 for the mask, measurements for the answer.

ChangeFormer emits a binary change map, not language. Everything the analyst
reads is derived from that mask arithmetically — changed fraction, number of
connected regions, where they sit in the frame — and only then handed to the LLM
to phrase. So "how much changed" is measured, never estimated by a language model.

Two caveats that stay visible in the trace rather than in a footnote:

  * The published weights are LEVIR-CD, i.e. RGB aerial *building* change. It is
    strong on construction and demolition, weaker on vegetation and water change.
  * The model definition is not on PyPI. `ml/vendor/ChangeFormer` must be cloned
    for `ChangeFormerV6`; without it this adapter reports unavailable and the
    difference-based fallback below runs instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..facts import Evidence, clamp_confidence
from .base import Adapter, AdapterUnavailable, torch_dtype

CHANGE_THRESHOLD = 0.5
MIN_REGION_PIXELS = 24

VENDOR_DIR = Path(
    os.environ.get("SATQUERY_CHANGEFORMER_DIR", "")
    or Path(__file__).resolve().parents[2] / "vendor" / "ChangeFormer"
)


class ChangeAdapter(Adapter):
    """ChangeFormerV6 pretrained on LEVIR-CD."""

    def __init__(self, spec, device: str = "cpu", checkpoint: Path | None = None):
        super().__init__(spec, device)
        self.checkpoint = Path(
            checkpoint
            or os.environ.get("SATQUERY_CHANGEFORMER_CKPT", "")
            or VENDOR_DIR / "checkpoints" / "ChangeFormer_LEVIR" / "best_ckpt.pt"
        )

    def _load(self) -> None:
        import torch

        if not VENDOR_DIR.exists():
            raise FileNotFoundError(
                f"ChangeFormer source not found at {VENDOR_DIR}. Clone it with "
                "`git clone https://github.com/wgcban/ChangeFormer ml/vendor/ChangeFormer` "
                "or set SATQUERY_CHANGEFORMER_DIR."
            )
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"ChangeFormer checkpoint not found at {self.checkpoint}. Download the "
                "v0.1.0 LEVIR release and unzip it so best_ckpt.pt is at that path."
            )

        if str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))

        from models.networks import ChangeFormerV6  # type: ignore

        self.model = ChangeFormerV6(embed_dim=256)

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        weights = state.get("model_G_state_dict", state) if isinstance(state, dict) else state
        weights = {key.replace("module.", "", 1): value for key, value in weights.items()}
        self.model.load_state_dict(weights, strict=False)

        self.model.eval().to(self.device)
        self.dtype = torch_dtype(self.device)

    # ---- inference -------------------------------------------------------

    def change_probability(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """(H, W) float32 change probability at the model's native 256x256."""
        self.ensure_loaded()

        import torch

        size = self.spec.input_size
        tensors = [
            torch.from_numpy(_to_chw(image, size)).unsqueeze(0).to(self.device, self.dtype)
            for image in (before, after)
        ]

        with torch.inference_mode():
            output = self.model(*tensors)
        if isinstance(output, (list, tuple)):
            output = output[-1]  # deep supervision: the last head is the finest

        if output.shape[1] == 2:
            probability = torch.softmax(output.float(), dim=1)[:, 1]
        else:
            probability = torch.sigmoid(output.float()).squeeze(1)
        return probability.squeeze(0).cpu().numpy()


def _to_chw(rgb: np.ndarray, size: int) -> np.ndarray:
    """(3, size, size) float32 in 0..1 from an HxWx3 uint8 image."""
    from ..bands import resize_chw

    array = np.asarray(rgb, dtype=np.float32)
    if array.max() > 1.5:
        array = array / 255.0
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    chw = np.transpose(array[:, :, :3], (2, 0, 1))
    if chw.shape[1] != size or chw.shape[2] != size:
        chw = resize_chw(chw, size)
    return np.ascontiguousarray(chw)


def difference_probability(before: np.ndarray, after: np.ndarray, size: int = 256) -> np.ndarray:
    """Fallback change signal when ChangeFormer is unavailable.

    Normalised absolute difference of the two images. Crude and sensitive to
    illumination, so the caller must label it as the fallback in the trace — an
    unlabelled fallback is worse than no answer.
    """
    a = _to_chw(before, size).mean(axis=0)
    b = _to_chw(after, size).mean(axis=0)
    delta = np.abs(b - a)
    peak = float(delta.max())
    return (delta / peak) if peak > 1e-6 else delta


def summarise_mask(probability: np.ndarray, threshold: float = CHANGE_THRESHOLD) -> dict:
    """Measured statistics from the change map. No model involved.

    Region counting is a small flood fill rather than scipy.label, because scipy
    is not guaranteed on a Kaggle image and this needs to work when it is absent.
    """
    binary = probability >= threshold
    total = float(binary.size)
    changed = float(binary.sum())
    fraction = changed / total if total else 0.0

    regions = _count_regions(binary)
    centroid = None
    quadrant = None
    if changed > 0:
        rows, cols = np.nonzero(binary)
        centre_y = float(rows.mean()) / binary.shape[0]
        centre_x = float(cols.mean()) / binary.shape[1]
        centroid = [round(centre_x, 4), round(centre_y, 4)]
        quadrant = (
            f"{'north' if centre_y < 0.45 else 'south' if centre_y > 0.55 else 'central'}"
            f"{'-west' if centre_x < 0.45 else '-east' if centre_x > 0.55 else ''}"
        )

    return {
        "changedFraction": round(fraction, 5),
        "changedPercent": round(100.0 * fraction, 2),
        "regionCount": regions,
        "centroid": centroid,
        "where": quadrant,
        "meanProbability": round(float(probability.mean()), 4),
        "threshold": threshold,
    }


def _count_regions(binary: np.ndarray, min_pixels: int = MIN_REGION_PIXELS) -> int:
    """4-connected component count, ignoring specks below `min_pixels`."""
    if not binary.any():
        return 0

    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    count = 0

    for start_y in range(height):
        for start_x in range(width):
            if not binary[start_y, start_x] or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if size >= min_pixels:
                count += 1
    return count


def change_evidence(
    stats: dict,
    question: str,
    model_spec=None,
    fallback_reason: str | None = None,
) -> Evidence:
    """Build the answer from measured statistics alone."""
    percent = stats["changedPercent"]
    regions = stats["regionCount"]

    evidence = Evidence(
        task="change",
        question=question,
        modality="bitemporal",
        confidence=clamp_confidence(0.85 if percent > 1.0 else 0.6),
    )
    if model_spec is not None:
        evidence.add_model(model_spec)
    if fallback_reason:
        evidence.notes.append(
            f"ChangeFormer unavailable ({fallback_reason}); these numbers come from "
            "normalised image differencing, which is sensitive to illumination and "
            "seasonal effects. Treat them as indicative only."
        )

    evidence.measurements["changed area"] = f"{percent:.2f}% of the frame"
    evidence.measurements["distinct changed regions"] = str(regions)
    if stats["where"]:
        evidence.measurements["change concentrated"] = stats["where"]

    if percent < 0.5:
        evidence.verdict = "no"
        evidence.terse = "unchanged"
        evidence.findings.append(
            f"only {percent:.2f}% of the frame changed, below the 0.5% "
            "significance floor, so the scene is effectively unchanged"
        )
    else:
        evidence.verdict = "yes"
        evidence.terse = "changed"
        where = f" concentrated in the {stats['where']} of the frame" if stats["where"] else ""
        evidence.findings.append(
            f"{percent:.2f}% of the frame changed across {regions} distinct "
            f"region(s){where}"
        )

    # "Increase or decrease?" cannot be answered by a binary change mask. Say so
    # instead of guessing a direction.
    lowered = question.lower()
    if any(word in lowered for word in ("increase", "decrease", "grown", "shrunk", "more", "less")):
        evidence.notes.append(
            "The change model produces a binary changed/unchanged mask, so it "
            "localises change but does not signal direction. Direction is taken "
            "from the per-date land-cover comparison where both dates allow it."
        )
    return evidence
