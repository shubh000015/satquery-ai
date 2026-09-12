"""SatQuery two-stage remote-sensing pipeline.

Stage 1 (`model`, `inference`): a multi-label land-cover classifier we train on
BigEarthNet Sentinel-1 (SAR) and Sentinel-2 (optical) patches. It produces the
grounded facts — which of the 19 classes are present, with a score each.

Stage 2 (`verbalizer`): Qwen2.5-Instruct phrases those facts as an answer or a
caption. It is text-only and never sees the image, so every claim in the output
traces back to a classifier score.

`serve.py` at the repo's `ml/` root exposes both over the HTTP contract that
`backend/app/tools/endpoint_client.py` already speaks.
"""

from .labels import CLASSES, NUM_CLASSES, short_name
from .model import LandCoverClassifier, ModelConfig, load_checkpoint, save_checkpoint

__all__ = [
    "CLASSES",
    "NUM_CLASSES",
    "short_name",
    "LandCoverClassifier",
    "ModelConfig",
    "load_checkpoint",
    "save_checkpoint",
]
