"""Inference server for the SatQuery specialist models. Runs on Kaggle GPUs.

Nothing is trained here. Every checkpoint is downloaded from HuggingFace, a
GitHub release or torchgeo; `satquery_ml/registry.py` records what each one is and
who trained it, and that attribution is returned in every response.

The shape of every endpoint is the same and is the point of the architecture:

    specialist model  ->  Evidence (verdict + findings + measured numbers)
                      ->  VLM rephrases it, never overrides it
                      ->  JSON

Two input formats are accepted. `imageB64` is a plain RGB PNG, which is what the
benchmark datasets and the existing backend send. `bands` is a named multispectral
stack (see satquery_ml/bands.py), which is what the 12-band models actually need —
send that when you have it, because an RGB image cannot feed BigEarthNet or CROMA
and those models will say so rather than guess.

    python serve.py --port 8100
    python serve.py --no-vlm            # specialists only, template phrasing
    python serve.py --warmup            # preload before the demo
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from satquery_ml import registry  # noqa: E402
from satquery_ml.adapters import AdapterUnavailable  # noqa: E402
from satquery_ml.adapters import change as change_module  # noqa: E402
from satquery_ml.adapters import fusion as fusion_module  # noqa: E402
from satquery_ml.bands import BandStack  # noqa: E402
from satquery_ml.facts import Evidence  # noqa: E402
from satquery_ml.loader import ModelLoader  # noqa: E402

app = FastAPI(title="SatQuery specialist models", version="0.2.0")
LOADER: ModelLoader | None = None


def loader() -> ModelLoader:
    if LOADER is None:  # pragma: no cover - set in main()
        raise HTTPException(status_code=503, detail="server still starting")
    return LOADER


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------

class SceneInput(BaseModel):
    """One scene, as either an RGB PNG or a named band stack."""

    imageB64: str | None = None
    bands: dict | None = None

    def to_stack(self, field: str = "scene") -> BandStack:
        if self.bands is not None:
            try:
                return BandStack.decode(self.bands)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"{field}: could not decode band stack ({exc})")
        if self.imageB64:
            return BandStack.from_rgb(decode_png(self.imageB64, field))
        raise HTTPException(400, f"{field}: provide either imageB64 or bands")


class VqaRequest(SceneInput):
    question: str = ""
    task: str = Field(default="vqa")
    threshold: float = 0.5
    # Open questions the 19 BigEarthNet classes cannot express are better served
    # by asking the VLM to look. Set false to force the grounded-only path.
    allowVision: bool = True


class GroundingRequest(SceneInput):
    query: str


class ChangeRequest(BaseModel):
    question: str = ""
    beforeB64: str | None = None
    afterB64: str | None = None
    before: dict | None = None
    after: dict | None = None

    def pair(self) -> tuple[np.ndarray, np.ndarray]:
        before = SceneInput(imageB64=self.beforeB64, bands=self.before).to_stack("before")
        after = SceneInput(imageB64=self.afterB64, bands=self.after).to_stack("after")
        return before.rgb(), after.rgb()


class FusionRequest(BaseModel):
    question: str = ""
    # Preferred: one co-registered stack carrying both SAR and optical bands.
    bands: dict | None = None
    # Or the two scenes separately, which is what an upload of two files gives.
    opticalB64: str | None = None
    sarB64: str | None = None
    optical: dict | None = None
    sar: dict | None = None
    threshold: float = 0.5


def decode_png(encoded: str, field: str = "image") -> np.ndarray:
    try:
        raw = base64.b64decode(encoded)
        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"{field}: could not decode image ({exc})")


def respond(evidence: Evidence, answer: str, model_label: str, started: float) -> dict:
    """The response body. `answer` is prose; everything else is the evidence."""
    body = evidence.as_dict()
    body.update({
        "answer": answer,
        # The backend client reads these three keys, so they must not move.
        "confidence": round(evidence.confidence, 3),
        "model": model_label,
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    })
    return body


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    status = loader().status()
    return {
        "status": "ok",
        "trainedByUs": [],  # nothing here is ours; see each model's provenance
        **status,
    }


@app.get("/v1/models")
def models() -> dict:
    return {
        "models": [spec.provenance() | {"role": spec.role} for spec in registry.REGISTRY.values()],
        "taskPlans": {task: list(keys) for task, keys in registry.TASK_PLANS.items()},
        "provenanceMarkdown": registry.provenance_table(),
    }


@app.post("/v1/vqa")
def vqa(request: VqaRequest) -> dict:
    """Single-image VQA and captioning. Mandatory baseline task."""
    started = time.perf_counter()
    stack = request.to_stack()
    task = "caption" if request.task == "caption" else "vqa"
    models = loader()

    try:
        classifier = models.load("landcover")
        evidence = classifier.evidence(
            stack, question=request.question, task=task, threshold=request.threshold
        )
    except AdapterUnavailable as exc:
        # The classifier cannot run — usually an RGB input where 12 bands are
        # needed. Fall back to the VLM looking at the image directly, and label it.
        evidence = Evidence(
            task=task,
            question=request.question,
            modality=stack.modality,
            notes=[f"land-cover classifier unavailable: {exc.reason}"],
        )
        if not request.allowVision:
            raise HTTPException(503, f"no model could answer: {exc.reason}")
        vlm = models.vlm()
        if vlm is None:
            raise HTTPException(503, f"no model could answer: {exc.reason}")
        prompt = request.question or "Describe the land cover and major features visible."
        answer = vlm.look(stack.rgb(), prompt)
        evidence.add_model(registry.spec("vlm"))
        evidence.findings.append("answered by the vision-language model directly")
        evidence.confidence = 0.5
        evidence.terse = "see answer"
        return respond(evidence, answer, vlm.label, started)

    # The classifier answered. If the question is outside its 19 classes, let the
    # VLM look as well and note both, rather than pretending the label list is
    # a complete answer.
    grounded_target = evidence.verdict is not None
    if request.allowVision and not grounded_target and request.question:
        vlm = models.vlm()
        if vlm is not None:
            try:
                visual = vlm.look(stack.rgb(), request.question)
                evidence.add_model(registry.spec("vlm"))
                evidence.notes.append(
                    "The question falls outside the classifier's 19 land-cover "
                    "classes, so the vision-language model answered it directly; "
                    "the land-cover findings are shown as supporting context."
                )
                return respond(evidence, visual, vlm.label, started)
            except Exception as exc:  # noqa: BLE001
                evidence.notes.append(f"vision path failed ({exc}); used grounded phrasing")

    answer, label = models.phrase(evidence)
    return respond(evidence, answer, label, started)


@app.post("/v1/classify")
def classify(request: SceneInput) -> dict:
    """Stage 1 only: raw classifier output, no language model. Useful for debugging."""
    started = time.perf_counter()
    stack = request.to_stack()
    try:
        prediction = loader().load("landcover").predict(stack)
    except AdapterUnavailable as exc:
        raise HTTPException(503, exc.reason)

    ranked = sorted(prediction["scores"].items(), key=lambda kv: -kv[1])
    return {
        "modality": stack.modality,
        "present": prediction["present"],
        "threshold": prediction["threshold"],
        "scores": [{"label": name, "score": round(score, 4)} for name, score in ranked],
        "model": registry.spec("landcover").name,
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    }


@app.post("/v1/grounding")
def grounding(request: GroundingRequest) -> dict:
    """Text-guided region grounding. The second mandatory single-image task."""
    started = time.perf_counter()
    stack = request.to_stack()
    models = loader()

    try:
        adapter = models.load("grounding_detector")
    except AdapterUnavailable as exc:
        raise HTTPException(503, exc.reason)

    evidence = adapter.evidence(stack.rgb(), request.query)
    answer, label = models.phrase(evidence)
    return respond(evidence, answer, label, started)


@app.post("/v1/change")
def change(request: ChangeRequest) -> dict:
    """Bi-temporal change description. Mandatory multi-image task."""
    started = time.perf_counter()
    before, after = request.pair()
    models = loader()

    fallback_reason = None
    spec = registry.spec("change")
    try:
        adapter = models.load("change")
        probability = adapter.change_probability(before, after)
    except AdapterUnavailable as exc:
        fallback_reason = exc.reason
        probability = change_module.difference_probability(before, after)
        spec = None

    stats = change_module.summarise_mask(probability)
    evidence = change_module.change_evidence(
        stats, request.question, model_spec=spec, fallback_reason=fallback_reason
    )

    answer, label = models.phrase(evidence)
    body = respond(evidence, answer, label, started)
    body["changeStats"] = stats
    body["changeMaskRle"] = _mask_rle(probability >= stats["threshold"])
    body["maskHeight"], body["maskWidth"] = probability.shape
    return body


@app.post("/v1/fusion")
def fusion(request: FusionRequest) -> dict:
    """Co-registered optical + SAR joint analysis. Mandatory cross-modal task."""
    started = time.perf_counter()
    models = loader()

    combined: BandStack | None = None
    if request.bands is not None:
        combined = SceneInput(bands=request.bands).to_stack("bands")

    try:
        classifier = models.load("landcover")
    except AdapterUnavailable as exc:
        raise HTTPException(503, f"land-cover classifier required for fusion: {exc.reason}")

    notes: list[str] = []
    specs = [registry.spec("landcover")]

    # Per-modality labels. This is what actually answers "what does SAR add".
    optical_labels: list[str] = []
    sar_labels: list[str] = []
    if combined is not None:
        try:
            optical_labels = classifier.predict(combined, request.threshold)["present"]
            sar_labels = optical_labels
            notes.append(
                "Both readings come from the single co-registered 12-band stack, "
                "which the classifier consumes jointly rather than per sensor."
            )
        except AdapterUnavailable as exc:
            notes.append(f"joint classification unavailable: {exc.reason}")
    else:
        for field, target in (("optical", "optical"), ("sar", "sar")):
            payload = SceneInput(
                imageB64=getattr(request, f"{field}B64"), bands=getattr(request, field)
            )
            if payload.imageB64 is None and payload.bands is None:
                continue
            try:
                stack = payload.to_stack(field)
                labels = classifier.predict(stack, request.threshold)["present"]
            except AdapterUnavailable as exc:
                notes.append(f"{field} classification unavailable: {exc.reason}")
                continue
            if target == "optical":
                optical_labels = labels
            else:
                sar_labels = labels

    # CROMA agreement, when the bands allow it.
    agreement = None
    if combined is not None:
        try:
            croma = models.load("fusion")
            agreement = croma.agreement(combined)
            specs.append(registry.spec("fusion"))
        except AdapterUnavailable as exc:
            notes.append(f"CROMA joint encoder unavailable: {exc.reason}")
    else:
        notes.append(
            "CROMA needs one co-registered stack with 2 SAR and 12 optical bands; "
            "send `bands` rather than separate images to enable it."
        )

    if not optical_labels and not sar_labels:
        raise HTTPException(503, "; ".join(notes) or "no usable input for fusion")

    evidence = fusion_module.fusion_evidence(
        request.question,
        optical_labels=optical_labels,
        sar_labels=sar_labels,
        agreement=agreement,
        model_specs=tuple(specs),
        notes=tuple(notes),
    )
    answer, label = models.phrase(evidence)
    body = respond(evidence, answer, label, started)
    if agreement:
        body["agreement"] = agreement
    return body


def _mask_rle(mask: np.ndarray) -> list[int]:
    from satquery_ml.adapters.grounding import _encode_rle

    return _encode_rle(mask)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--no-vlm", action="store_true", help="skip the VLM; template phrasing only")
    parser.add_argument("--no-4bit", action="store_true", help="load the VLM in fp16 instead of 4-bit")
    parser.add_argument("--gpus", type=int, default=None, help="override detected GPU count")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="preload every model before serving so the first request is fast",
    )
    args = parser.parse_args()

    global LOADER
    LOADER = ModelLoader(
        gpu_count=args.gpus,
        load_4bit=not args.no_4bit,
        enable_vlm=not args.no_vlm,
    )

    print("device plan:", LOADER.devices, flush=True)
    print("estimated VRAM per slot (GB):", registry.vram_by_slot(), flush=True)

    if args.warmup:
        keys = tuple(k for k in registry.REGISTRY if k != "grounding_segmenter")
        if args.no_vlm:
            keys = tuple(k for k in keys if k != "vlm")
        print("warming up...", flush=True)
        for key, state in LOADER.warmup(keys).items():
            print(f"  {key}: {state}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
