"""Optical-SAR joint analysis: CROMA embeddings plus a per-modality label comparison.

CROMA is a self-supervised encoder trained on a million paired Sentinel-1/2
scenes. It has no classification head, so on its own it cannot answer anything —
it produces vectors. What it *can* tell us is how much the two sensors agree:
the cosine distance between its optical and SAR embeddings, and how far the joint
embedding sits from either one.

The user-facing answer therefore comes from two sources together:

  * CROMA — a measured agreement score between the modalities.
  * The BigEarthNet classifier run once per modality — actual labels from optical
    alone, from SAR alone, and their difference. That difference is the direct
    answer to "what does SAR add here", which is the mandatory cross-modal task.

CROMA's input geometry is fixed: exactly 2 SAR channels and 12 optical channels.
Cartosat/RISAT uploads will not have 12 Sentinel-2 bands, so this adapter will
often report unavailable on real uploads and the classifier comparison carries the
answer alone. That degradation is reported, not hidden.
"""

from __future__ import annotations

import numpy as np

from .. import labels as label_vocab
from ..bands import CROMA_OPTICAL_BANDS, CROMA_SAR_BANDS, BandStack, MissingBands, resize_chw
from ..facts import Evidence, clamp_confidence
from .base import Adapter, AdapterUnavailable, batch_tensor, place_module


class FusionAdapter(Adapter):
    """CROMA-base radar-optical encoder, loaded through torchgeo."""

    def _load(self) -> None:
        try:
            from torchgeo.models import CROMABase_Weights, croma_base
        except ImportError as exc:
            raise ImportError(
                "torchgeo>=0.7 is required for CROMA (pip install torchgeo)"
            ) from exc

        self.model, self.dtype = place_module(
            croma_base(weights=CROMABase_Weights.CROMA_VIT).eval(), self.device
        )

    def embeddings(self, stack: BandStack) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
        """Optical, SAR and joint embeddings, plus any synthesized band names."""
        self.ensure_loaded()

        import torch

        try:
            ready, synthesized = stack.for_croma()
        except MissingBands as exc:
            raise AdapterUnavailable(self.spec.name, str(exc)) from exc

        size = self.spec.input_size
        sar = resize_chw(ready.select(CROMA_SAR_BANDS, self.spec.name), size)
        optical = resize_chw(ready.select(CROMA_OPTICAL_BANDS, self.spec.name), size)

        sar_tensor = batch_tensor(sar, self.device, self.model)
        optical_tensor = batch_tensor(optical, self.device, self.model)

        with torch.inference_mode():
            output = self.model(x_sar=sar_tensor, x_optical=optical_tensor)

        vectors = {
            name: _vector(value)
            for name, value in _named_outputs(output).items()
        }
        return vectors, synthesized

    def agreement(self, stack: BandStack) -> dict:
        """How much the two sensors say the same thing, as a measured number."""
        vectors, synthesized = self.embeddings(stack)
        optical = vectors.get("optical")
        sar = vectors.get("sar")

        result: dict = {"available": list(vectors)}
        if synthesized:
            result["synthesizedBands"] = list(synthesized)
        if optical is not None and sar is not None:
            result["cosineSimilarity"] = round(_cosine(optical, sar), 4)
        joint = vectors.get("joint")
        if joint is not None and optical is not None:
            result["jointVsOptical"] = round(_cosine(joint, optical), 4)
        if joint is not None and sar is not None:
            result["jointVsSar"] = round(_cosine(joint, sar), 4)
        return result


def _named_outputs(output) -> dict:
    """Normalise CROMA's return value across torchgeo versions.

    It has returned a dict keyed on modality and a plain tuple depending on
    version, so both are handled rather than pinning a version we cannot test
    from here.
    """
    if isinstance(output, dict):
        mapping = {}
        for key, value in output.items():
            lowered = str(key).lower()
            if "joint" in lowered:
                mapping["joint"] = value
            elif "sar" in lowered or "radar" in lowered or "s1" in lowered:
                mapping["sar"] = value
            elif "optical" in lowered or "s2" in lowered:
                mapping["optical"] = value
        if mapping:
            return mapping
    if isinstance(output, (tuple, list)):
        names = ("sar", "optical", "joint")
        return {names[i]: value for i, value in enumerate(output) if i < len(names)}
    return {"joint": output}


def _vector(tensor) -> np.ndarray:
    """Mean-pool a (B, tokens, dim) or (B, dim) tensor down to (dim,)."""
    array = tensor.detach().float().cpu().numpy()
    while array.ndim > 2:
        array = array.mean(axis=1)
    return array.reshape(-1) if array.ndim == 1 else array[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denominator)


def fusion_evidence(
    question: str,
    optical_labels: list[str],
    sar_labels: list[str],
    agreement: dict | None = None,
    model_specs: tuple = (),
    notes: tuple[str, ...] = (),
) -> Evidence:
    """Compose the cross-modal answer from per-modality labels and CROMA agreement.

    The interesting content is the asymmetry: what SAR sees that optical does
    not, and the reverse. That is what "extract complementary information from a
    co-registered pair" actually asks for.
    """
    optical_short = [label_vocab.short_name(name) for name in optical_labels]
    sar_short = [label_vocab.short_name(name) for name in sar_labels]

    shared = [name for name in optical_short if name in sar_short]
    optical_only = [name for name in optical_short if name not in sar_short]
    sar_only = [name for name in sar_short if name not in optical_short]

    evidence = Evidence(
        task="fusion",
        question=question,
        modality="optical-sar",
        confidence=clamp_confidence(0.8 if shared else 0.55),
        terse=", ".join(shared[:2]) if shared else "disagreement",
    )
    for spec in model_specs:
        evidence.add_model(spec)
    evidence.notes.extend(notes)

    if optical_short:
        evidence.findings.append(f"optical alone indicates: {', '.join(optical_short)}")
    if sar_short:
        evidence.findings.append(f"SAR alone indicates: {', '.join(sar_short)}")
    if shared:
        evidence.findings.append(f"both sensors agree on: {', '.join(shared)}")
    if sar_only:
        evidence.findings.append(
            f"SAR adds what optical missed: {', '.join(sar_only)}"
        )
    if optical_only:
        evidence.findings.append(
            f"visible only in optical: {', '.join(optical_only)}"
        )
    if not sar_only and not optical_only and shared:
        evidence.findings.append(
            "the two sensors produced identical land cover, so SAR mainly confirms "
            "the optical reading here rather than adding new classes"
        )

    if agreement and "cosineSimilarity" in agreement:
        similarity = agreement["cosineSimilarity"]
        evidence.measurements["optical-SAR embedding similarity"] = f"{similarity:.2f}"
        evidence.findings.append(
            f"the joint encoder scores optical-SAR agreement at {similarity:.2f} "
            "cosine similarity"
        )

    evidence.measurements["classes from optical"] = str(len(optical_short))
    evidence.measurements["classes from SAR"] = str(len(sar_short))
    return evidence
