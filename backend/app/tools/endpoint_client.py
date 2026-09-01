"""HTTP client for fine-tuned model endpoints (ml/serve.py).

Kept to the standard library on purpose: the backend must not grow a hard
dependency for a feature that is optional until weights are wired.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.services.raster import Scene

_TIMEOUT_S = 90  # first request may compile CUDA kernels; be generous
_MAX_EDGE = 896  # plenty for a VLM, keeps the payload small


@dataclass(slots=True)
class VlmReply:
    answer: str
    confidence: float
    model: str


def scene_png_b64(scene: Scene, max_edge: int = _MAX_EDGE) -> str:
    """8-bit RGB PNG of the analysis scene, base64-encoded for the JSON body."""
    data = scene.array
    if data.shape[2] == 1:
        rgb = np.repeat(data, 3, axis=2)
    elif data.shape[2] == 2:
        rgb = np.concatenate([data, data[:, :, :1]], axis=2)
    else:
        rgb = data[:, :, :3]
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def ask_vlm(endpoint: str, scene: Scene, question: str, task: str = "vqa") -> VlmReply:
    """POST the scene + question to a served checkpoint. Raises on any failure;
    the Tool base class catches and falls back to the heuristic baseline."""
    payload = json.dumps(
        {"question": question, "imageB64": scene_png_b64(scene), "task": task}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/vqa",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        body = json.loads(response.read().decode("utf-8"))

    answer = str(body.get("answer", "")).strip()
    if not answer:
        raise ValueError("endpoint returned an empty answer")
    return VlmReply(
        answer=answer,
        confidence=float(body.get("confidence", 0.5)),
        model=str(body.get("model", "endpoint")),
    )
