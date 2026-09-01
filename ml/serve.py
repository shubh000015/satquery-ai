"""Serve the fine-tuned Qwen2.5-VL adapter to the SatQuery agent backend.

Exposes the contract backend/app/tools/endpoint_client.py calls:

    GET  /health            -> {"status": "ok", "model": ..., "adapter": ...}
    POST /v1/vqa            -> {"answer": str, "confidence": float, "model": str}
        body: {"question": str, "imageB64": base64 PNG/JPEG, "task": "vqa"|"caption"}

Run it on the machine with the GPU:

    python serve.py --adapter runs/rsvqa-lora/adapter --port 8100

then point the SatQuery backend at it (backend/.env):

    SATQUERY_VLM_ENDPOINT=http://localhost:8100

If the GPU box is a different machine (e.g. Kaggle can't serve — use the
friend's laptop or a rented pod), any tunnel works; the backend only needs
plain HTTP reachability.
"""

from __future__ import annotations

import argparse
import base64
import io
import time

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 640 * 28 * 28

SYSTEM_PROMPT = (
    "You are SatQuery, an assistant for satellite and aerial imagery. "
    "Answer questions about remote sensing scenes concisely and factually."
)

CAPTION_FALLBACK_PROMPT = "Describe this remote sensing image in detail."


class VqaRequest(BaseModel):
    question: str = ""
    imageB64: str
    task: str = "vqa"
    maxNewTokens: int = 96


class VqaResponse(BaseModel):
    answer: str
    confidence: float
    model: str
    elapsedMs: int


def load_model(model_name: str, adapter: str | None):
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if torch.cuda.is_available()
        else None,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return processor, model


def build_app(model_name: str, adapter: str | None) -> FastAPI:
    processor, model = load_model(model_name, adapter)
    label = f"{model_name.split('/')[-1]}" + (" + RS-LoRA" if adapter else " (base)")

    app = FastAPI(title="SatQuery RS-VLM", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": label, "adapter": adapter, "cuda": torch.cuda.is_available()}

    @app.post("/v1/vqa", response_model=VqaResponse)
    def vqa(req: VqaRequest):
        started = time.perf_counter()
        try:
            image = Image.open(io.BytesIO(base64.b64decode(req.imageB64))).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"imageB64 is not a decodable image: {exc}")

        question = req.question.strip() or CAPTION_FALLBACK_PROMPT
        if req.task == "caption" and not req.question.strip():
            question = CAPTION_FALLBACK_PROMPT

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=min(req.maxNewTokens, 256),
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        new_tokens = generated.sequences[:, inputs["input_ids"].shape[1]:]
        answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        # Confidence = mean probability of the chosen tokens. Crude but honest,
        # and comparable across queries.
        probs = []
        for step, scores in enumerate(generated.scores):
            token_id = new_tokens[0, step]
            probs.append(torch.softmax(scores[0].float(), dim=-1)[token_id].item())
        confidence = round(sum(probs) / len(probs), 3) if probs else 0.5

        return VqaResponse(
            answer=answer,
            confidence=confidence,
            model=label,
            elapsedMs=int((time.perf_counter() - started) * 1000),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir from train.py (omit to serve the base model)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args.model, args.adapter), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
