"""What each model is, where it came from, and who adapted it.

Two jobs. First, it tells the loader what to build and which GPU to put it on.
Second, `trained_by` and `adapted_on` are reported in /health and in every
response's execution trace, which is both the honest answer to "whose weights
are these" and the "auditable execution summary containing the selected task,
model/tool names, key parameters, and outputs" the problem statement asks for.

Nothing here is trained by us. If that changes, update `trained_by` rather than
leaving it to a slide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bands import (
    BEN_ALL_BANDS,
    CROMA_OPTICAL_BANDS,
    CROMA_SAR_BANDS,
    RGB_BANDS,
)

# Logical GPU slots. The VLM is by far the largest, so it gets a card to itself
# and the five small specialists share the other one.
SLOT_VLM = "vlm"
SLOT_SPECIALIST = "specialist"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    role: str
    tasks: tuple[str, ...]
    source: str
    outputs: tuple[str, ...]
    bands: tuple[str, ...]
    input_size: int
    slot: str = SLOT_SPECIALIST
    approx_vram_gb: float = 1.0
    base_model: str | None = None
    adapted_on: str | None = None
    trained_by: str = "third party"
    license: str | None = None
    notes: str = ""
    extra_bands: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def provenance(self) -> dict:
        """The attribution block echoed into responses and /health."""
        return {
            "model": self.name,
            "source": self.source,
            "baseModel": self.base_model,
            "adaptedOn": self.adapted_on,
            "trainedBy": self.trained_by,
            "license": self.license,
        }


LANDCOVER = ModelSpec(
    key="landcover",
    name="BigEarthNet-19 ResNet-50 (S1+S2)",
    role="multi-label land-cover classification",
    tasks=("vqa", "caption", "fusion"),
    source="https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0",
    outputs=("labels",),
    bands=BEN_ALL_BANDS,
    input_size=120,
    approx_vram_gb=0.2,
    base_model="ResNet-50",
    adapted_on="BigEarthNet v2.0 (reBEN), Sentinel-1 + Sentinel-2",
    trained_by="BIFOLD / TU Berlin",
    license="CC-BY-4.0",
    notes=(
        "Needs all 12 bands in the published order. Loading requires the "
        "configilm package and reben_publication model code."
    ),
)

VLM = ModelSpec(
    key="vlm",
    name="RSCoVLM-7B (Qwen2.5-VL-7B, remote-sensing co-trained)",
    role="visual question answering, captioning, answer phrasing",
    tasks=("vqa", "caption"),
    source="https://huggingface.co/Qingyun/RSCoVLM-7B-2512",
    outputs=("text",),
    bands=RGB_BANDS,
    input_size=448,
    slot=SLOT_VLM,
    approx_vram_gb=6.0,
    base_model="Qwen/Qwen2.5-VL-7B-Instruct",
    adapted_on="remote-sensing multi-task recipe (VQA, captioning, detection)",
    trained_by="VisionXLab (Li et al., Remote Sensing 2026)",
    license="see model card",
    notes=(
        "Loaded in 4-bit to fit a 16 GB card. This is the component that "
        "satisfies 'a remote-sensing-adapted vision-language component' — plain "
        "Qwen2.5-VL would not. Set SATQUERY_VLM_MODEL to fall back to the "
        "unadapted base model for comparison."
    ),
)

GROUNDING_DETECTOR = ModelSpec(
    key="grounding_detector",
    name="Grounding DINO (base)",
    role="open-vocabulary text-to-box detection",
    tasks=("grounding",),
    source="https://huggingface.co/IDEA-Research/grounding-dino-base",
    outputs=("boxes",),
    bands=RGB_BANDS,
    input_size=800,
    approx_vram_gb=1.0,
    base_model="Swin-B + BERT",
    adapted_on="Objects365, OpenImages, GoldG (natural images)",
    trained_by="IDEA-Research",
    license="Apache-2.0",
    notes=(
        "Not remote-sensing adapted. Text prompts must be lowercase and end "
        "with a period or recall collapses. Overhead imagery is out of its "
        "training domain, so scores are reported as-is and not inflated."
    ),
)

GROUNDING_SEGMENTER = ModelSpec(
    key="grounding_segmenter",
    name="SAM 2.1 (Hiera-Large)",
    role="box-prompted referring segmentation",
    tasks=("grounding",),
    source="https://huggingface.co/facebook/sam2.1-hiera-large",
    outputs=("masks",),
    bands=RGB_BANDS,
    input_size=1024,
    approx_vram_gb=1.0,
    base_model="Hiera-L",
    adapted_on="SA-V / SA-1B (natural images and video)",
    trained_by="Meta AI",
    license="Apache-2.0",
    notes="Takes Grounding DINO's boxes as prompts; it does not read text itself.",
)

CHANGE = ModelSpec(
    key="change",
    name="ChangeFormerV6 (LEVIR-CD)",
    role="bi-temporal binary change segmentation",
    tasks=("change",),
    source=(
        "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/"
        "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_"
        "multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip"
    ),
    outputs=("mask",),
    bands=RGB_BANDS,
    input_size=256,
    approx_vram_gb=0.5,
    base_model="SegFormer-B2 (MiT)",
    adapted_on="LEVIR-CD (RGB aerial building change)",
    trained_by="Bandara & Patel (wgcban)",
    license="see repository",
    notes=(
        "Trained on RGB aerial building change at 256x256, so it is strongest "
        "on built-up change and weaker on vegetation or water change. The "
        "model definition is not on PyPI: the repo must be vendored and the "
        "checkpoint is best_ckpt.pt with net_G=ChangeFormerV6, embed_dim=256."
    ),
)

FUSION = ModelSpec(
    key="fusion",
    name="CROMA-base (radar-optical joint encoder)",
    role="co-registered optical + SAR joint representation",
    tasks=("fusion",),
    source="https://huggingface.co/torchgeo/croma (via torchgeo croma_base)",
    outputs=("embedding",),
    bands=CROMA_OPTICAL_BANDS,
    extra_bands={"sar": CROMA_SAR_BANDS, "optical": CROMA_OPTICAL_BANDS},
    input_size=120,
    approx_vram_gb=0.6,
    base_model="ViT-B",
    adapted_on="SSL4EO (1M paired Sentinel-1 + Sentinel-2, self-supervised)",
    trained_by="Fuller et al., NeurIPS 2023",
    license="see repository",
    notes=(
        "An encoder with no classification head, so it cannot answer on its "
        "own. We read its optical, SAR and joint embeddings and compare them "
        "against the land-cover classifier's per-modality predictions, which "
        "is what makes 'what does SAR add over optical here' answerable. "
        "Input is fixed at 2 SAR channels and 12 optical channels."
    ),
)

REGISTRY: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        LANDCOVER,
        VLM,
        GROUNDING_DETECTOR,
        GROUNDING_SEGMENTER,
        CHANGE,
        FUSION,
    )
}

# Which models a task needs, in execution order.
TASK_PLANS: dict[str, tuple[str, ...]] = {
    "vqa": ("landcover", "vlm"),
    "caption": ("landcover", "vlm"),
    "grounding": ("grounding_detector", "grounding_segmenter", "vlm"),
    "change": ("change", "vlm"),
    "fusion": ("landcover", "fusion", "vlm"),
}


def spec(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(REGISTRY)}") from None


def plan_for(task: str) -> tuple[ModelSpec, ...]:
    if task not in TASK_PLANS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASK_PLANS)}")
    return tuple(spec(key) for key in TASK_PLANS[task])


def vram_by_slot() -> dict[str, float]:
    totals: dict[str, float] = {}
    for model in REGISTRY.values():
        totals[model.slot] = round(totals.get(model.slot, 0.0) + model.approx_vram_gb, 2)
    return totals


def provenance_table() -> str:
    """Markdown attribution table for the README and the demo slide."""
    header = (
        "| Component | Model | Base | Adapted on | Trained by |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    rows = "".join(
        f"| {m.role} | [{m.name}]({m.source.split()[0]}) | {m.base_model or '-'} "
        f"| {m.adapted_on or '-'} | {m.trained_by} |\n"
        for m in REGISTRY.values()
    )
    return header + rows
