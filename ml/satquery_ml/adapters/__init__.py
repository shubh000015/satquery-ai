"""Per-model adapters. Each owns one checkpoint and yields `facts.Evidence`."""

from .base import Adapter, AdapterUnavailable
from .change import ChangeAdapter
from .fusion import FusionAdapter
from .grounding import GroundingAdapter
from .landcover import LandCoverAdapter
from .vlm import VlmAdapter

__all__ = [
    "Adapter",
    "AdapterUnavailable",
    "ChangeAdapter",
    "FusionAdapter",
    "GroundingAdapter",
    "LandCoverAdapter",
    "VlmAdapter",
]
