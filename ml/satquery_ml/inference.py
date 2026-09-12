"""Run the trained classifier on a single image and shape the result into facts.

This is stage 1 of the pipeline. Everything the language model is later allowed
to say has to come out of the `Prediction` produced here — that is what keeps
the wording fluent without letting it invent land cover that the classifier
never detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import labels as label_vocab
from .dataset import build_transform, modality_tensor
from .model import load_checkpoint


@dataclass
class Prediction:
    modality: str
    scores: dict[str, float]
    present: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def dominant(self) -> str | None:
        return self.present[0] if self.present else None

    def score(self, class_name: str) -> float:
        return self.scores.get(class_name, 0.0)

    def threshold(self, class_name: str) -> float:
        return self.thresholds.get(class_name, 0.5)

    def group_score(self, group: str) -> float:
        members = label_vocab.GROUPS.get(group, ())
        return max((self.scores.get(name, 0.0) for name in members), default=0.0)

    def group_present(self, group: str) -> list[str]:
        members = set(label_vocab.GROUPS.get(group, ()))
        return [name for name in self.present if name in members]

    def top(self, count: int = 5) -> list[tuple[str, float]]:
        ranked = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:count]

    def short_labels(self) -> list[str]:
        return [label_vocab.short_name(name) for name in self.present]

    def as_dict(self) -> dict:
        return {
            "modality": self.modality,
            "present": self.present,
            "presentShort": self.short_labels(),
            "scores": {name: round(value, 4) for name, value in self.scores.items()},
        }


def detect_modality(image: Image.Image) -> str:
    """Best-effort optical-vs-SAR guess from the pixels alone.

    Callers that know the modality should pass it explicitly — the backend has
    it on the asset record. Two signals are used here: our own SAR renders are
    built as R=VV, G=VH, B=|VV-VH|, so the blue channel is almost exactly the
    absolute channel difference; and SAR amplitude in general has far lower
    colour saturation than a true-colour optical patch.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]

    synthetic_blue = np.abs(red - green)
    if float(np.mean(np.abs(blue - synthetic_blue))) < 0.06:
        return "sar"

    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
    return "sar" if float(np.mean(saturation)) < 0.12 else "optical"


class LandCoverPredictor:
    def __init__(self, checkpoint: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, thresholds, self.metrics = load_checkpoint(checkpoint, self.device)
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)
        self.transform = build_transform(train=False)
        self.checkpoint = str(checkpoint)

    @torch.inference_mode()
    def predict(self, image: Image.Image, modality: str | None = None) -> Prediction:
        resolved = (modality or "").lower()
        if resolved not in {"optical", "sar"}:
            resolved = detect_modality(image)

        pixels = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        modality_vector = modality_tensor(resolved).unsqueeze(0).to(self.device)

        logits = self.model(pixels, modality_vector)
        probabilities = torch.sigmoid(logits)[0].float().cpu()

        scores = {
            label_vocab.CLASSES[i]: float(probabilities[i].item())
            for i in range(label_vocab.NUM_CLASSES)
        }
        above = [
            (label_vocab.CLASSES[i], float(probabilities[i].item()))
            for i in range(label_vocab.NUM_CLASSES)
            if probabilities[i] >= self.thresholds[i]
        ]
        above.sort(key=lambda pair: pair[1], reverse=True)

        # Never return an empty label set: the single best class is still the
        # most defensible thing to say about the scene.
        present = [name for name, _ in above]
        if not present:
            present = [max(scores, key=scores.get)]

        return Prediction(
            modality=resolved,
            scores=scores,
            present=present,
            thresholds={
                label_vocab.CLASSES[i]: float(self.thresholds[i].item())
                for i in range(label_vocab.NUM_CLASSES)
            },
        )
