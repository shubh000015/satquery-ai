"""Multi-label land-cover classifier for Sentinel-1 (SAR) and Sentinel-2 (optical).

One backbone handles both modalities. The pooled visual feature is concatenated
with a two-dimensional modality one-hot before the classification head, so the
model can apply a different decision boundary to SAR than to optical without
maintaining two separate sets of weights.

Output is 19 independent sigmoid scores (BigEarthNet-19), not a softmax: a patch
genuinely contains several land-cover classes at once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .labels import NUM_CLASSES

FEATURE_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}
MODALITY_DIMS = 2


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    num_classes: int = NUM_CLASSES
    dropout: float = 0.2


class LandCoverClassifier(nn.Module):
    def __init__(self, config: ModelConfig | None = None, pretrained_backbone: bool = True):
        super().__init__()
        self.config = config or ModelConfig()

        if self.config.backbone not in FEATURE_DIMS:
            raise ValueError(
                f"backbone must be one of {sorted(FEATURE_DIMS)}, got {self.config.backbone!r}"
            )

        builder = getattr(models, self.config.backbone)
        weights = "IMAGENET1K_V2" if self.config.backbone == "resnet50" else "IMAGENET1K_V1"
        backbone = builder(weights=weights if pretrained_backbone else None)

        feature_dim = FEATURE_DIMS[self.config.backbone]
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(feature_dim + MODALITY_DIMS, self.config.num_classes),
        )

    def forward(self, pixels: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixels)
        return self.head(torch.cat([features, modality], dim=1))


def save_checkpoint(
    path: Path,
    model: LandCoverClassifier,
    thresholds: list[float] | None = None,
    metrics: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "thresholds": thresholds or [0.5] * model.config.num_classes,
            "metrics": metrics or {},
        },
        path,
    )
    # Human-readable sidecar so the numbers are inspectable without torch.
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": asdict(model.config),
                "thresholds": thresholds or [0.5] * model.config.num_classes,
                "metrics": metrics or {},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[LandCoverClassifier, list[float], dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"])
    # The trained weights overwrite the backbone anyway, so skip the download.
    model = LandCoverClassifier(config, pretrained_backbone=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    thresholds = payload.get("thresholds") or [0.5] * config.num_classes
    return model, thresholds, payload.get("metrics") or {}
