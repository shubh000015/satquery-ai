"""Common shape for every model adapter.

An adapter owns one checkpoint: how to fetch it, which bands it eats, how to
turn its raw output into `Evidence`. It does not decide when to load — the
loader does that, so a task can pull in only the models it needs and the five
specialists can share a card.

Loading is allowed to fail. Weights come from HuggingFace, a GitHub release and
torchgeo, over a Kaggle network connection, so any of them can be unavailable at
any time. An adapter that cannot load raises `AdapterUnavailable`, the endpoint
reports which model was missing, and the backend falls back to its heuristic
baseline instead of the whole demo dying.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..bands import BandStack, MissingBands
from ..registry import ModelSpec


class AdapterUnavailable(RuntimeError):
    """Raised when a model cannot be loaded or cannot run on this input."""

    def __init__(self, model: str, reason: str):
        self.model = model
        self.reason = reason
        super().__init__(f"{model} unavailable: {reason}")


class Adapter(ABC):
    """One checkpoint, loaded on demand."""

    def __init__(self, spec: ModelSpec, device: str = "cpu"):
        self.spec = spec
        self.device = device
        self._loaded = False
        self.load_error: str | None = None

    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def ready(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.load_error is not None:
            # Do not retry a failed download on every request; one attempt per
            # process, then report it honestly.
            raise AdapterUnavailable(self.spec.name, self.load_error)
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - degrade, do not crash the server
            self.load_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: {self.spec.name} failed to load: {self.load_error}", flush=True)
            raise AdapterUnavailable(self.spec.name, self.load_error) from exc
        self._loaded = True
        print(f"loaded {self.spec.name} on {self.device}", flush=True)

    def unload(self) -> None:
        """Free VRAM. Called by the loader when another model needs the card."""
        if not self._loaded:
            return
        self._release()
        self._loaded = False

        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        print(f"unloaded {self.spec.name}", flush=True)

    def require_bands(self, stack: BandStack, names: tuple[str, ...] | None = None):
        """Band slice this model was trained on, or a clear refusal.

        Translating `MissingBands` into `AdapterUnavailable` is what stops an
        RGB upload from being fed to a 12-band model as zero-padded channels.
        """
        wanted = names if names is not None else self.spec.bands
        try:
            return stack.select(wanted, model=self.spec.name)
        except MissingBands as exc:
            raise AdapterUnavailable(self.spec.name, str(exc)) from exc

    @abstractmethod
    def _load(self) -> None:
        """Build the model and move it to `self.device`."""

    def _release(self) -> None:
        """Drop references so CUDA memory is actually freed."""
        for attribute in ("model", "processor", "predictor", "tokenizer"):
            if hasattr(self, attribute):
                setattr(self, attribute, None)


def torch_dtype(device: str):
    """fp16 on T4 (no bf16 support), bf16 where available, fp32 on CPU."""
    import torch

    if not device.startswith("cuda"):
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def force_float32(module):
    """Cast every floating parameter and buffer to fp32, including HF leftovers.

    `module.to(dtype=float32)` and `from_pretrained(dtype=float32)` both miss
    pieces of Grounding DINO: BERT LayerNorms, text-enhancer linears, and
    `config.torch_dtype` can stay bf16. `_apply` walks the whole tree.
    Returns the remaining floating dtypes — should be `(torch.float32,)`.
    """
    import torch

    def _cast(tensor):
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dtype != torch.float32:
            return tensor.float()
        return tensor

    module._apply(_cast)
    config = getattr(module, "config", None)
    if config is not None and hasattr(config, "torch_dtype"):
        config.torch_dtype = torch.float32
    return tuple(sorted({p.dtype for p in module.parameters() if p.is_floating_point()}, key=str))


def place_module(module, device: str, dtype=None):
    """Move a module onto `device` in a single compute dtype.

    HuggingFace / safetensors often load weights as bf16 regardless of the GPU.
    Leaving them that way and then sending a float32 batch is exactly
    `RuntimeError: expected scalar type BFloat16 but found Float`. Cast both
    sides here so every adapter agrees with itself.

    Pass `dtype` to override the card default. Grounding DINO needs this:
    its BERT text tower keeps activations in float32 even after a bf16 cast,
    then dies in the text-enhancer linear with `mat1 Float, mat2 BFloat16`.
    """
    if dtype is None:
        dtype = torch_dtype(device)
    return module.to(device=device, dtype=dtype), dtype


def module_dtype(module):
    """Compute dtype a module's conv/linear layers expect for activations.

    Skips 4-bit / int quantized weights so a BitsAndBytes VLM does not ask
    for int8 `pixel_values`. Prefers the HuggingFace `model.dtype` hint, then
    the first fp32/fp16/bf16 parameter.
    """
    import torch

    compute = (torch.float32, torch.float16, torch.bfloat16)
    hinted = getattr(module, "dtype", None)
    if hinted in compute:
        return hinted
    for param in module.parameters():
        if param.dtype in compute:
            return param.dtype
    return torch.float32


def batch_tensor(array, device: str, module):
    """`(1, C, H, W)` tensor on `device` in `module`'s parameter dtype."""
    import numpy as np
    import torch

    return (
        torch.from_numpy(np.ascontiguousarray(array))
        .unsqueeze(0)
        .to(device=device, dtype=module_dtype(module))
    )


def move_batch(batch, device: str, module, dtype=None) -> dict:
    """Move a processor batch onto `device` and cast floats to `dtype`.

    HuggingFace processors always emit float32 `pixel_values`. `.to(device)`
    does not change that, so a bf16 Grounding DINO / VLM then dies with
    `Input type (float) and bias type (c10::BFloat16) should be the same`.
    Integer tokens (`input_ids`, masks) stay integers.

    Pass `dtype` to override `module_dtype` — Grounding DINO must force
    float32, because HuggingFace `model.dtype` can still report bf16 after a
    partial cast.
    """
    if dtype is None:
        dtype = module_dtype(module)
    moved = {}
    for key, value in batch.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        value = value.to(device=device)
        if value.is_floating_point():
            value = value.to(dtype=dtype)
        moved[key] = value
    return moved
