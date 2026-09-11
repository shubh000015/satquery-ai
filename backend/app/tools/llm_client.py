"""External vision-language API used until the fine-tuned SatQuery adapter is ready.

Talks OpenAI-compatible Chat Completions (OpenAI, Groq, OpenRouter, Together)
or the Gemini generateContent API. stdlib only — same rule as endpoint_client.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

from app.core.config import Settings
from app.services.raster import Scene
from app.tools.endpoint_client import VlmReply, scene_png_b64

_TIMEOUT_S = 90

SATQUERY_SYSTEM = """You are SatQuery AI, an agentic vision-language assistant for remote sensing.

You answer plain-English questions about satellite and aerial scenes the way a
fine-tuned RS-VLM would: concise, factual, and tied to what is visible.

Rules:
- Prefer land-cover, hydrology, built-up, agriculture, ports, and change language
  (NDVI, inundation, SAR backscatter, GSD, AOI) over generic photo captions.
- If the user asks whether something increased, decreased, or stayed the same,
  give a clear verdict first, then the evidence.
- If a deterministic analysis stack is provided (cover %, areas, change %),
  treat those numbers as measurements. Do not invent km² or NDVI values that
  contradict them. You may interpret them.
- If you cannot see a class clearly, say so. Do not invent object counts.
- Keep the answer to 4–8 sentences unless the user asks for a short yes/no.
- Do not mention that you are a stand-in API, ChatGPT, Gemini, or a fallback
  unless asked how you were produced.
"""


def llm_configured(settings: Settings) -> bool:
    return bool((settings.llm_api_key or os.environ.get("SATQUERY_LLM_API_KEY") or "").strip())


def _api_key(settings: Settings) -> str:
    return (settings.llm_api_key or os.environ.get("SATQUERY_LLM_API_KEY") or "").strip()


def _provider(settings: Settings) -> str:
    return (settings.llm_provider or "gemini").strip().lower()


def _model(settings: Settings) -> str:
    if settings.llm_model:
        return settings.llm_model
    return {
        "gemini": "gemini-2.0-flash",
        "openai": "gpt-4o-mini",
        "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
        "openrouter": "google/gemini-2.0-flash-001",
        "compatible": "gpt-4o-mini",
    }.get(_provider(settings), "gemini-2.0-flash")


def _openai_base(settings: Settings) -> str:
    if settings.llm_base_url:
        return settings.llm_base_url.rstrip("/")
    return {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "compatible": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    }.get(_provider(settings), "https://api.openai.com/v1")


def _user_text(question: str, task: str, context: str | None) -> str:
    parts = [
        f"Task kind: {task}.",
        f"Question: {question.strip()}",
    ]
    if context:
        parts.append("Deterministic stack (use as measurements):\n" + context.strip())
    parts.append("Answer in English, as SatQuery.")
    return "\n\n".join(parts)


def ask_llm(
    settings: Settings,
    question: str,
    task: str = "vqa",
    scenes: Iterable[Scene] | None = None,
    context: str | None = None,
    image_pngs_b64: list[str] | None = None,
) -> VlmReply:
    """Vision+text call. Raises on transport or empty-answer failures."""
    key = _api_key(settings)
    if not key:
        raise RuntimeError("SATQUERY_LLM_API_KEY is not set")

    images: list[str] = list(image_pngs_b64 or [])
    if scenes is not None:
        images.extend(scene_png_b64(scene) for scene in scenes)
    # Cap at two frames (t0/t1 or optical/SAR).
    images = images[:2]

    provider = _provider(settings)
    model = _model(settings)
    if provider == "gemini" and not (settings.llm_base_url or "").strip():
        answer = _gemini_native(key, model, question, task, context, images)
    else:
        answer = _openai_compatible(settings, key, model, question, task, context, images)

    answer = answer.strip()
    if not answer:
        raise ValueError("LLM returned an empty answer")
    return VlmReply(answer=answer, confidence=0.78, model=f"{provider}:{model}")


def _openai_compatible(
    settings: Settings,
    key: str,
    model: str,
    question: str,
    task: str,
    context: str | None,
    images_b64: list[str],
) -> str:
    content: list[dict] = [{"type": "text", "text": _user_text(question, task, context)}]
    for raw in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{raw}"},
            }
        )
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.2,
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content": SATQUERY_SYSTEM},
                {"role": "user", "content": content},
            ],
        }
    ).encode("utf-8")
    url = f"{_openai_base(settings)}/chat/completions"
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    body = _read_json(request)
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected LLM payload: {body!r}") from exc


def _gemini_native(
    key: str,
    model: str,
    question: str,
    task: str,
    context: str | None,
    images_b64: list[str],
) -> str:
    parts: list[dict] = [{"text": _user_text(question, task, context)}]
    for raw in images_b64:
        parts.append({"inlineData": {"mimeType": "image/png", "data": raw}})
    payload = json.dumps(
        {
            "systemInstruction": {"parts": [{"text": SATQUERY_SYSTEM}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
        }
    ).encode("utf-8")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={urllib.parse.quote(key)}"
    )
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    body = _read_json(request)
    try:
        return str(body["candidates"][0]["content"]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected Gemini payload: {body!r}") from exc


def _read_json(request: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
