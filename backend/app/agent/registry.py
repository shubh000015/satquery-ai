"""Model and tool registry.

This is the team's model plan expressed as data. Nothing here loads weights: the
`status` field reports whether an inference endpoint has been configured for that
row, and the specialist tools consult it to decide between calling out to a
fine-tuned model and running the local heuristic baseline.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.schemas.registry import ModelSpec

MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="controller",
        requirement="Agent Controller",
        primary_model="Qwen2.5-7B (LangGraph)",
        fallback_model="Mistral-7B-Instruct",
        benchmarks=["Synthetic Routing Traces"],
        purpose="Task classification, tool execution, auditable trace logs.",
        tasks=[],
        endpoint_setting="vlm_endpoint",
    ),
    ModelSpec(
        key="vqa",
        requirement="Single-Image VQA",
        primary_model="Qwen2.5-VL-7B-Instruct",
        fallback_model="GeoChat / EarthGPT",
        benchmarks=["RSVQA", "RSVQAxBEN"],
        purpose="Mandatory baseline: visual QA on single optical/SAR tiles.",
        tasks=["vqa"],
        endpoint_setting="vlm_endpoint",
    ),
    ModelSpec(
        key="grounding",
        requirement="Text Grounding",
        primary_model="Qwen2.5-VL-7B",
        fallback_model="Grounding DINO",
        benchmarks=["VRSBench", "DIOR-RSVG"],
        purpose="Normalised bounding box coordinate output.",
        tasks=["grounding"],
        endpoint_setting="grounding_endpoint",
    ),
    ModelSpec(
        key="segmentation",
        requirement="Referring Segmentation",
        primary_model="Qwen2.5-VL + SAM 2",
        fallback_model="LISA / GLaMM",
        benchmarks=["RRSIS-D", "SAM-RS"],
        purpose="Pixel-accurate binary mask generation.",
        tasks=["grounding"],
        endpoint_setting="grounding_endpoint",
    ),
    ModelSpec(
        key="caption",
        requirement="Scene Captioning",
        primary_model="Qwen2.5-VL-7B",
        fallback_model="RemoteCLIP + Llama-3-8B",
        benchmarks=["VRSBench", "RSICD"],
        purpose="Detailed scene summaries and land-cover tagging.",
        tasks=["caption"],
        endpoint_setting="vlm_endpoint",
    ),
    ModelSpec(
        key="change-vqa",
        requirement="Bi-Temporal Change VQA",
        primary_model="ChangeChat / RS-TVLM",
        fallback_model="Multi-Image Qwen2.5-VL",
        benchmarks=["CDVQA", "LEVIR-CC"],
        purpose="Mandatory change analysis: reasoning over T1 vs T2.",
        tasks=["change-vqa"],
        endpoint_setting="change_endpoint",
    ),
    ModelSpec(
        key="change-mask",
        requirement="Spatial Change Masks",
        primary_model="ChangeFormer / BIT",
        fallback_model="STANet / SNUNet",
        benchmarks=["LEVIR-CD", "WHU-CD"],
        purpose="Generates binary/multi-class change heatmaps.",
        tasks=["change"],
        endpoint_setting="change_endpoint",
    ),
    ModelSpec(
        key="fusion",
        requirement="Optical–SAR Fusion",
        primary_model="CROMA + Qwen2.5",
        fallback_model="Dual-Branch (ViT + Swin)",
        benchmarks=["BigEarthNet-MM", "FUSAR"],
        purpose="Cross-modal fusion for cloud-penetrating joint analysis.",
        tasks=["cross-modal"],
        endpoint_setting="fusion_endpoint",
    ),
    ModelSpec(
        key="mensuration",
        requirement="Mensuration",
        primary_model="GSD-scaled geometry",
        fallback_model=None,
        benchmarks=[],
        purpose="Area and distance from the geotransform; no learned model needed.",
        tasks=["measure"],
        endpoint_setting=None,
    ),
)

_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def endpoint_for(key: str, settings: Settings | None = None) -> str | None:
    """Configured inference endpoint for a registry row, if any."""
    settings = settings or get_settings()
    spec = _BY_KEY.get(key)
    if spec is None or not spec.endpoint_setting:
        return None
    value = getattr(settings, spec.endpoint_setting, None)
    return value or None


def model_specs(settings: Settings | None = None) -> list[ModelSpec]:
    settings = settings or get_settings()
    resolved: list[ModelSpec] = []
    for spec in MODEL_SPECS:
        if spec.key == "mensuration":
            status = "available"
        else:
            status = "wired" if endpoint_for(spec.key, settings) else "planned"
        resolved.append(spec.model_copy(update={"status": status}))
    return resolved


def spec(key: str) -> ModelSpec:
    return _BY_KEY[key]


def display_name(key: str, settings: Settings | None = None) -> str:
    """What to show in the models panel: the real model once wired, the
    heuristic stand-in until then."""
    resolved = _BY_KEY[key]
    if endpoint_for(key, settings):
        return resolved.primary_model
    if key == "mensuration":
        return resolved.primary_model
    return f"{resolved.primary_model} (baseline)"


def weights_wired(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    return sum(1 for spec_ in MODEL_SPECS if endpoint_for(spec_.key, settings))
