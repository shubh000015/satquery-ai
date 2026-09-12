"""Serve the fine-tuned Qwen2.5-VL adapter to the SatQuery agent backend.

Exposes three endpoints:

    GET  /health           -> {"status": "ok", "model": ..., "adapter": ...}
    POST /v1/vqa           -> single-turn VQA/caption (backward compat with
                              backend/app/tools/endpoint_client.py)
    POST /v1/chat          -> multi-turn conversation; used by ml/chat.py so
                              the demo feels like a chatbot on one scene

Run it on the machine with the GPU:

    python ml/serve.py --adapter path/to/ben-lora/adapter --port 8100

then point the SatQuery backend at it (backend/.env):

    SATQUERY_VLM_ENDPOINT=http://localhost:8100

The adapter format is whatever `ml/train_unsloth.py` /
`ml/kaggle_train_unsloth.ipynb` produce: a directory with
`adapter_config.json` + `adapter_model.safetensors` + tokenizer files. We read
`adapter_config.json` to figure out which base model to load, so the same
serve.py works for both the Unsloth 4-bit base and the vanilla HF base.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from serve_utils import (
    MAX_PIXELS,
    MIN_PIXELS,
    TRAIN_SYSTEM_PROMPT,
    b64_to_image,
)


DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
CAPTION_FALLBACK_PROMPT = "Describe this remote sensing image in detail."


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class VqaRequest(BaseModel):
    """Single-turn contract the backend already speaks."""

    question: str = ""
    imageB64: str
    task: str = "vqa"  # "vqa" | "caption"
    maxNewTokens: int = 96


class VqaResponse(BaseModel):
    answer: str
    confidence: float
    model: str
    elapsedMs: int


class ChatContent(BaseModel):
    type: str  # "text" | "image"
    text: str | None = None
    imageB64: str | None = None


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: list[ChatContent] | str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    maxNewTokens: int = Field(default=192, ge=1, le=1024)
    temperature: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    model: str
    elapsedMs: int


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_base_model(adapter: str | None, override: str | None) -> str:
    """Prefer the base named in adapter_config.json — that's what training used."""
    if override:
        return override
    if adapter:
        cfg = Path(adapter) / "adapter_config.json"
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text())
                name = data.get("base_model_name_or_path")
                if name:
                    return name
            except Exception:
                pass
    return DEFAULT_BASE_MODEL


def load_model(base_model: str, adapter: str | None):
    on_gpu = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if on_gpu and torch.cuda.is_bf16_supported() else torch.float16

    kwargs: dict = {"torch_dtype": compute_dtype, "device_map": "auto" if on_gpu else None}

    # Only apply BitsAndBytes on the raw HF checkpoint. Unsloth's -bnb-4bit
    # models are already pre-quantised — passing another BitsAndBytesConfig on
    # top would trigger a re-quantise error.
    is_pre_quantised = "bnb-4bit" in base_model or "bnb_4bit" in base_model
    if on_gpu and not is_pre_quantised:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print(f"[serve] loading base: {base_model} (cuda={on_gpu}, bf16={compute_dtype == torch.bfloat16})", flush=True)
    processor = AutoProcessor.from_pretrained(base_model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model, **kwargs)

    if adapter:
        from peft import PeftModel

        print(f"[serve] attaching adapter: {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _messages_to_qwen(messages: list[ChatMessage]) -> tuple[list[dict], list]:
    """Convert our schema into Qwen's chat_template format + collect PIL images."""
    qwen_msgs: list[dict] = []
    images: list = []
    saw_system = False

    for msg in messages:
        content = msg.content
        parts: list[dict] = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        else:
            for item in content:
                if item.type == "text" and item.text is not None:
                    parts.append({"type": "text", "text": item.text})
                elif item.type == "image" and item.imageB64:
                    pil = b64_to_image(item.imageB64)
                    images.append(pil)
                    parts.append({"type": "image", "image": pil})
        qwen_msgs.append({"role": msg.role, "content": parts})
        if msg.role == "system":
            saw_system = True

    if not saw_system:
        qwen_msgs.insert(
            0,
            {"role": "system", "content": [{"type": "text", "text": TRAIN_SYSTEM_PROMPT}]},
        )
    return qwen_msgs, images


@torch.inference_mode()
def _generate(model, processor, messages: list[dict], images: list, max_new_tokens: int, temperature: float, label: str):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {"text": [text], "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs).to(model.device)

    gen_kwargs = {
        "max_new_tokens": min(max_new_tokens, 512),
        "do_sample": temperature > 0,
        "return_dict_in_generate": True,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    # output_scores is broken on some transformers+bitsandbytes builds; only
    # request it when we're going to use it, and don't let it crash the API.
    scores_ok = True
    try:
        gen_kwargs["output_scores"] = True
        out = model.generate(**inputs, **gen_kwargs)
    except (TypeError, RuntimeError):
        scores_ok = False
        gen_kwargs.pop("output_scores", None)
        out = model.generate(**inputs, **gen_kwargs)

    new_tokens = out.sequences[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    confidence = 0.5
    if scores_ok and getattr(out, "scores", None):
        probs = []
        for step, scores in enumerate(out.scores):
            token_id = new_tokens[0, step]
            probs.append(torch.softmax(scores[0].float(), dim=-1)[token_id].item())
        if probs:
            confidence = round(sum(probs) / len(probs), 3)

    return answer, confidence


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(base_model: str, adapter: str | None) -> FastAPI:
    processor, model = load_model(base_model, adapter)
    label = f"{base_model.split('/')[-1]}" + (" + RS-LoRA" if adapter else " (base)")

    app = FastAPI(title="SatQuery RS-VLM", version="0.2.0")

    @app.get("/health")
    def health():
        vram_gb = 0.0
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 1)
        return {
            "status": "ok",
            "model": label,
            "base": base_model,
            "adapter": adapter,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vramGb": vram_gb,
        }

    @app.post("/v1/vqa", response_model=VqaResponse)
    def vqa(req: VqaRequest):
        started = time.perf_counter()
        try:
            image = b64_to_image(req.imageB64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"imageB64 is not a decodable image: {exc}")

        question = req.question.strip() or CAPTION_FALLBACK_PROMPT
        if req.task == "caption" and not req.question.strip():
            question = CAPTION_FALLBACK_PROMPT

        messages = [
            {"role": "system", "content": [{"type": "text", "text": TRAIN_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
        ]
        try:
            answer, confidence = _generate(
                model, processor, messages, [image], req.maxNewTokens, 0.0, label
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

        return VqaResponse(
            answer=answer,
            confidence=confidence,
            model=label,
            elapsedMs=int((time.perf_counter() - started) * 1000),
        )

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        started = time.perf_counter()
        try:
            qwen_msgs, images = _messages_to_qwen(req.messages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad messages: {exc}") from exc

        try:
            answer, confidence = _generate(
                model, processor, qwen_msgs, images, req.maxNewTokens, req.temperature, label
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

        return ChatResponse(
            answer=answer,
            confidence=confidence,
            model=label,
            elapsedMs=int((time.perf_counter() - started) * 1000),
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (omit to serve the base model)")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Override the base model. Default: read from adapter_config.json, else {DEFAULT_BASE_MODEL}",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Use 0.0.0.0 to expose on LAN.")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    base_model = resolve_base_model(args.adapter, args.model)

    import uvicorn

    uvicorn.run(build_app(base_model, args.adapter), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
