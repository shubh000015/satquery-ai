"""BigEarthNet-19 multi-label land cover, from the published BIFOLD reBEN weights.

This is the component that satisfies the problem statement's remote-sensing
adaptation requirement on the vision side: a ResNet-50 trained by BIFOLD/TU
Berlin on BigEarthNet v2.0 using Sentinel-1 and Sentinel-2 together. We did not
train it; see `registry.LANDCOVER` for attribution.

Two loading paths, because the official one has a heavy dependency:
  * `configilm` + `reben_publication` — the documented route.
  * a direct timm + safetensors route — used when configilm is unavailable,
    which is common on a fresh Kaggle image.

The model is multi-label, so outputs go through a sigmoid, never a softmax, and
each class is thresholded independently.
"""

from __future__ import annotations

import numpy as np

from .. import labels as label_vocab
from ..bands import BEN_ALL_BANDS, BEN_S1_BANDS, BEN_S2_BANDS, BandStack, resize_chw
from ..facts import Evidence, clamp_confidence, is_yes_no
from .base import Adapter, AdapterUnavailable, batch_tensor, place_module

# BigEarthNet v2.0 reference statistics are per-band, but the published models
# were trained on min-max scaled patches, so a plain 0..1 stack is the right
# input. Thresholds are the standard 0.5 for a sigmoid multi-label head.
DEFAULT_THRESHOLD = 0.5


class LandCoverAdapter(Adapter):
    """19-class multi-label classifier over Sentinel-1 + Sentinel-2 bands."""

    def _load(self) -> None:
        self.model = None
        self._loader = None

        try:
            from reben_publication.BigEarthNetv2_0_ImageClassifier import (  # type: ignore
                BigEarthNetv2_0_ImageClassifier,
            )

            repo = self.spec.source.rsplit("/", 2)[-2:]
            self.model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
                "/".join(repo)
            )
            self._loader = "configilm"
        except Exception as configilm_error:  # noqa: BLE001
            self.model = self._load_with_timm()
            self._loader = "timm+safetensors"
            print(
                f"note: configilm path unavailable ({type(configilm_error).__name__}); "
                "loaded the BigEarthNet weights through timm instead.",
                flush=True,
            )

        self.model, self.dtype = place_module(self.model.eval(), self.device)

    def _load_with_timm(self):
        """Rebuild the architecture with timm and load the published safetensors.

        The checkpoint is a plain state dict over a timm resnet50 with a 12-channel
        stem and a 19-way head, so it loads without the configilm wrapper.
        """
        import timm
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        repo_id = "/".join(self.spec.source.rsplit("/", 2)[-2:])
        path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state = load_file(path)

        model = timm.create_model(
            "resnet50",
            pretrained=False,
            in_chans=len(BEN_ALL_BANDS),
            num_classes=label_vocab.NUM_CLASSES,
        )

        # The published dict prefixes weights with the lightning module name.
        cleaned = {}
        for key, value in state.items():
            for prefix in ("model.", "net.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            cleaned[key] = value

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        recognised = len(cleaned) - len(unexpected)
        if recognised < 50:
            raise RuntimeError(
                f"only {recognised} tensors matched the resnet50 graph; the "
                f"checkpoint layout has changed (missing={len(missing)}, "
                f"unexpected={len(unexpected)}). Install configilm and use the "
                "official loader."
            )
        return model

    # ---- inference -------------------------------------------------------

    def band_plan(self, stack: BandStack) -> tuple[str, ...]:
        """Which published variant this scene can feed.

        The 'all' weights need all 12 bands. A SAR-only or optical-only upload
        cannot use them, so we say so rather than fabricating channels.
        """
        if stack.has(*BEN_ALL_BANDS):
            return BEN_ALL_BANDS
        missing = stack.missing(BEN_ALL_BANDS)
        raise AdapterUnavailable(
            self.spec.name,
            f"needs the 12 published bands; this scene is missing {list(missing)}. "
            f"Load the {'S1' if stack.has(*BEN_S1_BANDS) else 'S2'}-only variant "
            "for single-modality input.",
        )

    def scores(self, stack: BandStack) -> np.ndarray:
        """Per-class sigmoid probabilities, shape (19,)."""
        self.ensure_loaded()

        import torch

        array = self.require_bands(stack, self.band_plan(stack))
        if array.shape[1] != self.spec.input_size:
            array = resize_chw(array, self.spec.input_size)

        tensor = batch_tensor(array, self.device, self.model)
        with torch.inference_mode():
            logits = self.model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()

    def predict(self, stack: BandStack, threshold: float = DEFAULT_THRESHOLD) -> dict:
        scores = self.scores(stack)
        present = [
            label_vocab.CLASSES[i]
            for i in np.argsort(-scores)
            if scores[i] >= threshold
        ]
        if not present:
            # Never return an empty scene; the single best class is more useful
            # than "nothing", and the score travels with it.
            present = [label_vocab.CLASSES[int(np.argmax(scores))]]

        return {
            "scores": {label_vocab.CLASSES[i]: float(scores[i]) for i in range(len(scores))},
            "present": present,
            "threshold": threshold,
        }

    def group_score(self, scores: dict[str, float], group: str) -> float:
        members = label_vocab.GROUPS[group]
        return max((scores.get(name, 0.0) for name in members), default=0.0)

    def evidence(
        self,
        stack: BandStack,
        question: str = "",
        task: str = "vqa",
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Evidence:
        """Resolve the question from the classifier alone. No LLM involved."""
        prediction = self.predict(stack, threshold)
        scores = prediction["scores"]
        present = prediction["present"]

        short = [label_vocab.short_name(name) for name in present]
        mean = sum(scores[name] for name in present) / max(len(present), 1)

        evidence = Evidence(
            task=task,
            question=question,
            modality=stack.modality,
            findings=[f"land cover: {', '.join(short)}"],
            confidence=clamp_confidence(mean),
            terse=short[0] if short else "unknown",
        )
        evidence.add_model(self.spec)
        evidence.measurements["top class score"] = f"{scores[present[0]]:.2f}"

        target = label_vocab.match_class_in_question(question) if question else None
        if target is None:
            return evidence

        if target in label_vocab.GROUPS:
            score = self.group_score(scores, target)
            label = "built-up area" if target == "builtup" else target
            matched = score >= threshold
        else:
            score = scores.get(target, 0.0)
            label = label_vocab.short_name(target)
            matched = target in present

        evidence.measurements[f"score for {label}"] = f"{score:.2f}"

        if matched:
            evidence.verdict = "yes"
            evidence.confidence = clamp_confidence(score)
            evidence.findings.insert(0, f"{label} is present (score {score:.2f})")
        elif score >= 0.5 * threshold:
            evidence.verdict = "possibly"
            evidence.confidence = clamp_confidence(0.5)
            evidence.findings.insert(
                0, f"{label} scored {score:.2f}, below the {threshold:.2f} threshold"
            )
        else:
            evidence.verdict = "no"
            evidence.confidence = clamp_confidence(1.0 - score)
            evidence.findings.insert(0, f"no {label} detected (score {score:.2f})")

        if not is_yes_no(question) and not matched:
            # "What crops are here?" should not be answered with a bare "no".
            evidence.verdict = None

        return evidence
