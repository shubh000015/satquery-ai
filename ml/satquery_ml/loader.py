"""Lazy model loading and GPU placement.

Kaggle gives two T4s, 16 GB each. The registry's own VRAM figures say the VLM
wants ~6 GB and the five specialists together want ~3.3 GB, so the split is:

    cuda:0  the VLM, alone
    cuda:1  land cover, grounding, change, fusion

That fits without eviction, which matters for a live demo: nothing has to be
unloaded and reloaded mid-question. With one GPU everything shares cuda:0 and
eviction is enabled, and with no GPU at all it runs on CPU for tests.

Nothing loads at import. A question about land cover should not pay to download
Grounding DINO, and a missing checkpoint should only break the task that needs it.
"""

from __future__ import annotations

from . import registry
from .adapters import (
    Adapter,
    AdapterUnavailable,
    ChangeAdapter,
    FusionAdapter,
    GroundingAdapter,
    LandCoverAdapter,
    VlmAdapter,
)


def available_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def plan_devices(gpu_count: int | None = None) -> dict[str, str]:
    """Which device each logical slot lands on."""
    count = available_gpus() if gpu_count is None else gpu_count
    if count >= 2:
        return {registry.SLOT_VLM: "cuda:0", registry.SLOT_SPECIALIST: "cuda:1"}
    if count == 1:
        return {registry.SLOT_VLM: "cuda:0", registry.SLOT_SPECIALIST: "cuda:0"}
    return {registry.SLOT_VLM: "cpu", registry.SLOT_SPECIALIST: "cpu"}


class ModelLoader:
    """Owns one adapter per model and hands them out on demand."""

    def __init__(
        self,
        gpu_count: int | None = None,
        load_4bit: bool = True,
        enable_vlm: bool = True,
    ):
        self.devices = plan_devices(gpu_count)
        self.gpu_count = available_gpus() if gpu_count is None else gpu_count
        self.load_4bit = load_4bit
        self.enable_vlm = enable_vlm
        # With both cards nothing needs evicting; with one card the VLM shares
        # with the specialists and we free a specialist before loading another.
        self.evict = self.gpu_count == 1
        self._adapters: dict[str, Adapter] = {}

    def device_for(self, slot: str) -> str:
        return self.devices[slot]

    def _build(self, key: str) -> Adapter:
        spec = registry.spec(key)
        device = self.device_for(spec.slot)

        if key == "landcover":
            return LandCoverAdapter(spec, device)
        if key == "vlm":
            return VlmAdapter(spec, device, load_4bit=self.load_4bit)
        if key in ("grounding_detector", "grounding_segmenter"):
            return GroundingAdapter(
                registry.spec("grounding_detector"),
                registry.spec("grounding_segmenter"),
                device,
            )
        if key == "change":
            return ChangeAdapter(spec, device)
        if key == "fusion":
            return FusionAdapter(spec, device)
        raise KeyError(f"no adapter for {key!r}")

    def get(self, key: str) -> Adapter:
        """The adapter for `key`, built but not necessarily loaded."""
        # Both grounding rows are served by one adapter holding both checkpoints.
        if key == "grounding_segmenter":
            key = "grounding_detector"

        if key not in self._adapters:
            self._adapters[key] = self._build(key)
        return self._adapters[key]

    def load(self, key: str) -> Adapter:
        """The adapter for `key`, loaded. Raises `AdapterUnavailable` on failure."""
        adapter = self.get(key)
        if adapter.ready:
            return adapter

        if self.evict:
            self._make_room(adapter)
        adapter.ensure_loaded()
        return adapter

    def _make_room(self, incoming: Adapter) -> None:
        """Single-GPU path: free the other specialists before loading one more."""
        for key, adapter in self._adapters.items():
            if adapter is incoming or not adapter.ready:
                continue
            if adapter.spec.slot == incoming.spec.slot:
                adapter.unload()

    def vlm(self):
        """The VLM, or None when it is disabled or cannot load.

        Phrasing is optional by design: every task still answers from its
        specialist's evidence through the deterministic template.
        """
        if not self.enable_vlm:
            return None
        try:
            return self.load("vlm")
        except AdapterUnavailable:
            return None

    def phrase(self, evidence) -> tuple[str, str]:
        """(answer, model label). Falls back to the template when the VLM is out."""
        model = self.vlm()
        if model is None:
            return evidence.template_answer(), "template"
        return model.phrase(evidence), model.label

    def warmup(self, keys: tuple[str, ...]) -> dict[str, str]:
        """Preload models so the first real request is not the slow one.

        Returns a per-model status rather than raising, so a demo can start with
        four of five models working and know exactly which one is missing.
        """
        status: dict[str, str] = {}
        for key in keys:
            try:
                self.load(key)
                status[key] = "loaded"
            except AdapterUnavailable as exc:
                status[key] = f"unavailable: {exc.reason}"
            except KeyError as exc:
                status[key] = f"error: {exc}"
        return status

    def status(self) -> dict:
        """What /health reports: placement, load state, and full provenance."""
        models = []
        for key, spec in registry.REGISTRY.items():
            adapter = self._adapters.get(
                "grounding_detector" if key == "grounding_segmenter" else key
            )
            models.append({
                "key": key,
                "role": spec.role,
                "loaded": bool(adapter and adapter.ready),
                "device": self.device_for(spec.slot),
                "loadError": adapter.load_error if adapter else None,
                **spec.provenance(),
            })
        return {
            "gpuCount": self.gpu_count,
            "devices": dict(self.devices),
            "evictionEnabled": self.evict,
            "vramBySlot": registry.vram_by_slot(),
            "models": models,
        }
