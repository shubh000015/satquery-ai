"""Text-guided region grounding: Grounding DINO for boxes, SAM 2 for masks.

Grounding DINO reads the text and proposes boxes; SAM 2 does not read text at
all, it only refines a box into a pixel-accurate mask. So the two run in
sequence and the phrase the analyst typed is what drives both.

Honest caveat, recorded here and in the registry rather than buried: both models
were trained on natural images, not overhead imagery. Nadir views, tiny objects
and radar speckle are out of their domain, so scores come back as the model
reports them and low-confidence detections are returned as low-confidence rather
than being quietly promoted.
"""

from __future__ import annotations

import re

import numpy as np

from ..facts import Evidence, clamp_confidence
from .base import Adapter, AdapterUnavailable, torch_dtype

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
MAX_REGIONS = 12

# Analyst phrasing -> a noun phrase the detector has some chance of knowing.
_PROMPT_HINTS = {
    "water body": "a lake. a river. water.",
    "water": "water. a lake. a river.",
    "river": "a river. water.",
    "lake": "a lake. water.",
    "building": "a building. a house. a rooftop.",
    "built-up": "a building. a house. a rooftop.",
    "urban": "a building. a house. a rooftop.",
    "road": "a road. a highway.",
    "vegetation": "a tree. vegetation. a forest.",
    "forest": "a forest. trees.",
    "field": "a field. farmland.",
    "farmland": "farmland. a field.",
    "airport": "an airport. a runway. an aircraft.",
    "ship": "a ship. a boat.",
    "bridge": "a bridge.",
}

_STOPWORDS = {
    "highlight", "show", "find", "locate", "where", "is", "are", "the", "a", "an",
    "in", "this", "image", "please", "me", "referred", "to", "by", "query", "of",
    "identify", "segment", "mask", "outline", "region", "regions", "area", "areas",
}


def build_prompt(query: str) -> str:
    """Turn an analyst request into a Grounding DINO prompt.

    The model is strict about format: lowercase, phrases separated by periods.
    Getting this wrong quietly destroys recall rather than raising, which is why
    it lives in one tested function instead of inline at the call site.
    """
    lowered = query.lower().strip()

    for phrase, prompt in sorted(_PROMPT_HINTS.items(), key=lambda kv: -len(kv[0])):
        if phrase in lowered:
            return prompt

    words = [
        word
        for word in re.findall(r"[a-z]+", lowered)
        if word not in _STOPWORDS and len(word) > 2
    ]
    if not words:
        return "an object."
    return " ".join(f"a {word}." for word in words[:3])


class GroundingAdapter(Adapter):
    """Grounding DINO box proposals, optionally refined by SAM 2 masks.

    Holds both checkpoints because they are useless apart: boxes without masks
    are imprecise, masks without text prompts cannot be steered.
    """

    def __init__(self, detector_spec, segmenter_spec, device: str = "cpu"):
        super().__init__(detector_spec, device)
        self.segmenter_spec = segmenter_spec
        self.predictor = None
        self.segmenter_error: str | None = None

    def _load(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        detector_id = "/".join(self.spec.source.rstrip("/").rsplit("/", 2)[-2:])
        self.processor = AutoProcessor.from_pretrained(detector_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_id, dtype=torch_dtype(self.device)
        ).to(self.device)
        self.model.eval()

        # SAM 2 is optional: boxes alone still answer a grounding query, so a
        # missing sam2 package degrades the output instead of failing the task.
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            segmenter_id = "/".join(
                self.segmenter_spec.source.rstrip("/").rsplit("/", 2)[-2:]
            )
            self.predictor = SAM2ImagePredictor.from_pretrained(
                segmenter_id, device=self.device
            )
        except Exception as exc:  # noqa: BLE001
            self.segmenter_error = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING: SAM 2 unavailable ({self.segmenter_error}); "
                "grounding will return boxes without masks.",
                flush=True,
            )

    def _release(self) -> None:
        super()._release()
        self.predictor = None

    def detect(self, rgb: np.ndarray, query: str) -> list[dict]:
        """Boxes in normalised xyxy, highest scoring first."""
        self.ensure_loaded()

        import torch
        from PIL import Image

        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        prompt = build_prompt(query)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]

        height, width = image.height, image.width
        regions: list[dict] = []
        for box, score, text in zip(
            results["boxes"], results["scores"], results.get("text_labels", results.get("labels", []))
        ):
            x0, y0, x1, y1 = (float(v) for v in box.tolist())
            regions.append({
                "label": str(text),
                "score": round(float(score), 4),
                "box": [
                    round(x0 / width, 5), round(y0 / height, 5),
                    round(x1 / width, 5), round(y1 / height, 5),
                ],
                "boxPixels": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
            })

        regions.sort(key=lambda item: -item["score"])
        return regions[:MAX_REGIONS]

    def segment(self, rgb: np.ndarray, regions: list[dict]) -> list[dict]:
        """Refine boxes into masks. Returns [] when SAM 2 is unavailable."""
        if self.predictor is None or not regions:
            return []

        import torch

        array = np.asarray(rgb, dtype=np.uint8)
        boxes = np.array([region["boxPixels"] for region in regions], dtype=np.float32)

        try:
            with torch.inference_mode():
                self.predictor.set_image(array)
                masks, scores, _ = self.predictor.predict(
                    point_coords=None, point_labels=None, box=boxes, multimask_output=False
                )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: SAM 2 prediction failed ({exc}); returning boxes only.", flush=True)
            return []

        masks = np.asarray(masks)
        if masks.ndim == 4:  # (N, 1, H, W)
            masks = masks[:, 0]
        elif masks.ndim == 2:  # single box
            masks = masks[None]

        pixels = float(array.shape[0] * array.shape[1])
        out: list[dict] = []
        for index, mask in enumerate(masks):
            binary = mask > 0.0
            out.append({
                "label": regions[index]["label"],
                "score": round(float(np.atleast_1d(scores)[index]), 4),
                "coveragePercent": round(100.0 * float(binary.sum()) / pixels, 2),
                "rle": _encode_rle(binary),
                "height": int(binary.shape[0]),
                "width": int(binary.shape[1]),
            })
        return out

    def evidence(self, rgb: np.ndarray, query: str) -> Evidence:
        regions = self.detect(rgb, query)
        masks = self.segment(rgb, regions)

        evidence = Evidence(
            task="grounding",
            question=query,
            modality="optical",
            boxes=regions,
            masks=masks,
        )
        evidence.add_model(self.spec)
        if masks:
            evidence.add_model(self.segmenter_spec)
        elif self.segmenter_error:
            evidence.notes.append(f"SAM 2 unavailable: {self.segmenter_error}")

        if not regions:
            evidence.verdict = "no"
            evidence.terse = "none"
            evidence.confidence = clamp_confidence(0.35)
            evidence.findings.append(
                f"the detector found no region matching '{query}' above a "
                f"{BOX_THRESHOLD:.2f} confidence threshold"
            )
            evidence.notes.append(
                "Grounding DINO is trained on natural images; overhead views are "
                "out of its domain, so absence here is weaker evidence than presence."
            )
            return evidence

        best = regions[0]
        evidence.verdict = "yes"
        evidence.terse = best["label"] or "region"
        evidence.confidence = clamp_confidence(best["score"])
        evidence.measurements["regions found"] = str(len(regions))
        evidence.measurements["best match score"] = f"{best['score']:.2f}"
        evidence.findings.append(
            f"located {len(regions)} region(s) matching '{best['label']}', "
            f"best at normalised box {best['box']}"
        )
        if masks:
            evidence.measurements["mask coverage"] = f"{masks[0]['coveragePercent']:.1f}% of the image"
            evidence.findings.append(
                f"SAM 2 segmented the best region to "
                f"{masks[0]['coveragePercent']:.1f}% of the image area"
            )
        return evidence


def _encode_rle(mask: np.ndarray) -> list[int]:
    """Row-major run-length encoding, starting from a run of zeros.

    A full boolean mask as JSON would be megabytes; this keeps the response
    small while staying trivially decodable in the frontend.
    """
    flat = mask.astype(np.uint8).ravel()
    if flat.size == 0:
        return []
    changes = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(boundaries).tolist()
    # Convention: the first run is zeros, so prepend an empty run if it is not.
    return [0] + runs if flat[0] == 1 else runs
