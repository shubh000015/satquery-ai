"""Serve the two-stage pipeline to the SatQuery backend.

Implements the contract that backend/app/tools/endpoint_client.py already calls,
so nothing in backend/ has to change:

    GET  /health      -> {"status": "ok", "model": ..., "cuda": bool}
    POST /v1/vqa      -> {"answer", "confidence", "model", "elapsedMs"}
        body: {"question": str, "imageB64": str, "task": "vqa"|"caption",
               "modality": "optical"|"sar"|null}

    POST /v1/classify -> raw stage-1 output (labels + scores), for debugging
                         and for showing judges what the classifier alone said

Run it wherever the GPU is:

    python serve.py --checkpoint runs/landcover/classifier.pt --port 8100

then point the backend at it in backend/.env:

    SATQUERY_VLM_ENDPOINT=http://<host>:8100

On Kaggle use notebooks/kaggle_serve_tunnel.ipynb, which wraps this in a
cloudflared tunnel because Kaggle cannot expose a port directly.
"""

from __future__ import annotations

import argparse
import base64
import io
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from satquery_ml.inference import LandCoverPredictor
from satquery_ml.verbalizer import DEFAULT_LLM, Verbalizer, resolve_facts


class VqaRequest(BaseModel):
    question: str = ""
    imageB64: str
    task: str = "vqa"
    modality: str | None = None
    maxNewTokens: int = 120


class VqaResponse(BaseModel):
    answer: str
    confidence: float
    model: str
    elapsedMs: int
    oneWord: str
    labels: list[str]
    modality: str


class ClassifyResponse(BaseModel):
    modality: str
    present: list[str]
    presentShort: list[str]
    scores: dict[str, float]
    elapsedMs: int


def decode_image(image_b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"imageB64 is not a decodable image: {exc}"
        ) from exc


def build_app(
    checkpoint: Path,
    llm_name: str | None,
    load_4bit: bool = True,
) -> FastAPI:
    import torch

    print(f"loading classifier from {checkpoint} ...", flush=True)
    predictor = LandCoverPredictor(checkpoint)
    print(f"  classifier ready on {predictor.device}", flush=True)

    if llm_name:
        print(f"loading verbalizer {llm_name} (this can take a few minutes) ...", flush=True)
        verbalizer = Verbalizer(llm_name, load_4bit=load_4bit)
    else:
        verbalizer = Verbalizer(enabled=False)
        print("  language layer disabled - using template phrasing", flush=True)

    label = f"landcover-{predictor.model.config.backbone}+{verbalizer.label}"
    app = FastAPI(title="SatQuery land-cover pipeline", version="2.0.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": label,
            "classifier": predictor.checkpoint,
            "classifierMetrics": predictor.metrics,
            "verbalizer": verbalizer.label,
            "verbalizerError": verbalizer.load_error,
            "cuda": torch.cuda.is_available(),
        }

    @app.post("/v1/classify", response_model=ClassifyResponse)
    def classify(req: VqaRequest) -> ClassifyResponse:
        started = time.perf_counter()
        prediction = predictor.predict(decode_image(req.imageB64), req.modality)
        payload = prediction.as_dict()
        return ClassifyResponse(
            modality=payload["modality"],
            present=payload["present"],
            presentShort=payload["presentShort"],
            scores=payload["scores"],
            elapsedMs=int((time.perf_counter() - started) * 1000),
        )

    @app.post("/v1/vqa", response_model=VqaResponse)
    def vqa(req: VqaRequest) -> VqaResponse:
        started = time.perf_counter()
        image = decode_image(req.imageB64)

        prediction = predictor.predict(image, req.modality)
        facts = resolve_facts(prediction, req.question, task=req.task)
        answer = verbalizer.phrase(facts, max_new_tokens=min(req.maxNewTokens, 256))

        return VqaResponse(
            answer=answer,
            confidence=facts.confidence,
            model=label,
            elapsedMs=int((time.perf_counter() - started) * 1000),
            oneWord=facts.one_word,
            labels=facts.present,
            modality=prediction.modality,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="classifier.pt from train_classifier.py",
    )
    parser.add_argument("--llm", default=DEFAULT_LLM, help="Qwen2.5 instruct model to phrase answers")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the language model and return template phrasing (fast startup)",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Load the LLM in fp16 instead of 4-bit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"{args.checkpoint} not found. Train it first with train_classifier.py, "
            "or download the classifier.pt produced by the Kaggle notebook."
        )

    import uvicorn

    app = build_app(
        args.checkpoint,
        None if args.no_llm else args.llm,
        load_4bit=not args.no_4bit,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
