"""HTTP client for fine-tuned model endpoints (ml/serve.py).

Kept to the standard library on purpose: the backend must not grow a hard
dependency for a feature that is optional until weights are wired.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.services.raster import Scene

_TIMEOUT_S = 90  # first request may compile CUDA kernels; be generous
_MAX_EDGE = 896  # plenty for a VLM, keeps the payload small
_BAND_EDGE = 384  # band stacks are float32 per channel, so keep them smaller

# How our band count maps onto sensor band names. The served models are trained
# on Sentinel bands, so naming them lets each model ask for the slice it knows.
# An RGB Cartosat chip honestly becomes B04/B03/B02 and nothing more, which is
# why the 12-band models decline it instead of reading red as near-infrared.
_SAR_NAMES = ("VV", "VH")
_OPTICAL_NAMES = ("B04", "B03", "B02", "B08")


@dataclass(slots=True)
class VlmReply:
    answer: str
    confidence: float
    model: str


@dataclass(slots=True)
class EndpointReply:
    """Everything the specialist endpoints return beyond a sentence."""

    answer: str
    confidence: float
    model: str
    findings: list[str]
    measurements: dict
    notes: list[str]
    boxes: list[dict]
    masks: list[dict]
    extra: dict


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


def band_names_for(modality: str, band_count: int) -> tuple[str, ...]:
    """Sensor band names for what we actually have. Never more than we have."""
    if modality == "sar":
        return _SAR_NAMES[: max(1, min(band_count, 2))]
    return _OPTICAL_NAMES[: max(1, min(band_count, 4))]


def scene_bands(scene: Scene, modality: str, max_edge: int = _BAND_EDGE) -> dict:
    """Named multispectral stack, base64 npz, matching ml/satquery_ml/bands.py.

    Uses `reflectance` rather than the display stretch: a per-band percentile
    stretch destroys the ratios between bands, which is exactly the signal a
    multispectral classifier reads.
    """
    names = band_names_for(modality, scene.bands)
    data = scene.reflectance[:, :, : len(names)].astype(np.float32)

    if max(scene.height, scene.width) > max_edge:
        scale = max_edge / max(scene.height, scene.width)
        target = (max(1, int(scene.width * scale)), max(1, int(scene.height * scale)))
        planes = [
            np.asarray(
                Image.fromarray(data[:, :, i]).resize(target, Image.BILINEAR),
                dtype=np.float32,
            )
            for i in range(data.shape[2])
        ]
        data = np.stack(planes, axis=2)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, array=data)
    return {
        "bands": list(names),
        "npz": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
    }


def decode_rle(runs: list[int], height: int, width: int) -> np.ndarray:
    """Inverse of the server's run-length encoding. First run is zeros."""
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    value = False
    for run in runs:
        end = min(position + int(run), flat.size)
        if value and end > position:
            flat[position:end] = True
        position = end
        value = not value
        if position >= flat.size:
            break
    return flat.reshape(height, width)


def _post(endpoint: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def _reply(body: dict, *keep: str) -> EndpointReply:
    answer = str(body.get("answer", "")).strip()
    if not answer:
        raise ValueError("endpoint returned an empty answer")
    return EndpointReply(
        answer=answer,
        confidence=float(body.get("confidence", 0.5)),
        model=str(body.get("model", "endpoint")),
        findings=list(body.get("findings", [])),
        measurements=dict(body.get("measurements", {})),
        notes=list(body.get("notes", [])),
        boxes=list(body.get("boxes", [])),
        masks=list(body.get("masks", [])),
        extra={key: body[key] for key in keep if key in body},
    )


def ask_vlm(endpoint: str, scene: Scene, question: str, task: str = "vqa") -> VlmReply:
    """POST the scene + question to a served checkpoint. Raises on any failure;
    the Tool base class catches and falls back to the heuristic baseline."""
    body = _post(
        endpoint,
        "/v1/vqa",
        {"question": question, "imageB64": scene_png_b64(scene), "task": task},
    )
    reply = _reply(body)
    return VlmReply(answer=reply.answer, confidence=reply.confidence, model=reply.model)


def ask_vqa_with_bands(
    endpoint: str, scene: Scene, modality: str, question: str, task: str = "vqa"
) -> EndpointReply:
    """VQA with the multispectral stack, so the 12-band classifier can run.

    Falls back to the RGB path when the server rejects the bands, which happens
    when the upload simply does not carry the bands the classifier needs.
    """
    try:
        body = _post(
            endpoint,
            "/v1/vqa",
            {
                "question": question,
                "task": task,
                "bands": scene_bands(scene, modality),
            },
        )
    except urllib.error.HTTPError as exc:
        if exc.code != 503:
            raise
        body = _post(
            endpoint,
            "/v1/vqa",
            {"question": question, "task": task, "imageB64": scene_png_b64(scene)},
        )
    return _reply(body)


def ask_grounding(endpoint: str, scene: Scene, query: str) -> EndpointReply:
    """Text-guided grounding. Sends the RGB view: the detector is RGB-only and
    spatial detail matters more here than spectral depth."""
    body = _post(
        endpoint, "/v1/grounding", {"query": query, "imageB64": scene_png_b64(scene)}
    )
    return _reply(body)


def ask_change(
    endpoint: str, before: Scene, after: Scene, question: str
) -> EndpointReply:
    """Bi-temporal change. ChangeFormer is RGB at 256x256, so PNGs suffice."""
    body = _post(
        endpoint,
        "/v1/change",
        {
            "question": question,
            "beforeB64": scene_png_b64(before),
            "afterB64": scene_png_b64(after),
        },
    )
    return _reply(body, "changeStats", "changeMaskRle", "maskHeight", "maskWidth")


def ask_fusion(
    endpoint: str,
    optical: Scene,
    sar: Scene,
    question: str,
    optical_modality: str = "optical",
) -> EndpointReply:
    """Optical-SAR joint analysis, one band stack per sensor.

    CROMA needs a single co-registered 2+12 band stack, which uploads of two
    separate files cannot provide, so the server reports that it skipped CROMA
    and answers from the per-modality land-cover comparison instead.
    """
    body = _post(
        endpoint,
        "/v1/fusion",
        {
            "question": question,
            "optical": scene_bands(optical, optical_modality),
            "sar": scene_bands(sar, "sar"),
        },
    )
    return _reply(body, "agreement")
