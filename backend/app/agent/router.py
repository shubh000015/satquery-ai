"""Router agent — reads the question and decides which task is being asked.

Keyword routing on purpose: the observable execution trace is what gets
evaluated, and a deterministic router is auditable and testable. Swapping in the
Qwen2.5-7B controller later means replacing `classify` while keeping the Intent
contract, which is also what src/lib/agent.ts uses on the client.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent import registry
from app.schemas.agent import Intent, TaskKind
from app.schemas.imagery import InputMode

TASK_LABELS: dict[TaskKind, str] = {
    "vqa": "Visual question answering",
    "caption": "Scene captioning",
    "grounding": "Text-guided grounding",
    "change": "Change description",
    "change-vqa": "Change VQA",
    "cross-modal": "Optical–SAR fusion",
    "measure": "Mensuration",
}

TASK_MODEL_KEYS: dict[TaskKind, list[str]] = {
    "vqa": ["vqa"],
    "caption": ["caption"],
    "grounding": ["grounding", "segmentation"],
    "change": ["change-mask", "change-vqa"],
    "change-vqa": ["change-vqa", "change-mask"],
    "cross-modal": ["fusion", "vqa"],
    "measure": ["mensuration"],
}

_MEASURE = ("measure", "how far", "area of", "how large", "distance", "km²", "km2", "hectare", "extent of")
_CHANGE = ("changed", "change", "increased", "decreased", "expansion", "grown", "shrunk", "between these", "compared to", "since", "before and after")
_CHANGE_VQA = ("increase", "decrease", "unchanged", "remained", "more or less", "any change", "how much has")
_CHANGE_VQA_OPENERS = ("has ", "have ", "did ", "is ", "are ", "was ", "were ")
_FUSION = ("optical and sar", "sar and optical", "both sensors", "both images together", "use both", "through cloud", "cloud cover", "fuse", "fusion", "built-up and water")
_GROUNDING = ("highlight", "locate", "point to", "mark", "where is", "where are", "which ships", "show me the", "outline", "segment", "delineate", "bounding box")
_CAPTION = ("describe", "caption", "summarise", "summarize", "land-cover", "land cover", "scene", "what do you see", "what is in this", "overview")
# A yes/no or counting question is VQA even if it mentions "scene" or "land cover".
_VQA_OPENERS = ("is ", "are ", "does ", "do ", "has ", "have ", "can ", "was ", "were ", "which ", "how many", "how much")
_FLOOD = ("flood", "inundat", "water-cover", "water covered", "submerged", "waterlogged")

TargetClass = str

_TARGET_TERMS: tuple[tuple[TargetClass, tuple[str, ...]], ...] = (
    ("water", ("water", "river", "lake", "pond", "reservoir", "flood", "inundat", "channel", "canal", "sea", "coast")),
    ("vegetation", ("vegetation", "forest", "tree", "crop", "agricultur", "farm", "field", "green", "plantation", "mangrove")),
    ("builtup", ("built-up", "built up", "building", "urban", "settlement", "village", "town", "city", "houses", "construction", "runway", "road")),
    ("ship", ("ship", "vessel", "boat", "barge", "tanker")),
    ("tank", ("tank", "storage tank", "silo", "tank farm")),
    ("bare", ("bare", "soil", "sand", "barren", "quarry", "mine")),
)


@dataclass(slots=True)
class RouteDecision:
    intent: Intent
    target: TargetClass
    matched: list[str]


def _matches(query: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in query]


def resolve_target(query: str) -> TargetClass:
    """Which land-cover class or object the question is about."""
    lowered = query.lower()
    for target, terms in _TARGET_TERMS:
        if any(term in lowered for term in terms):
            return target
    return "generic"


def _specialists(task: TaskKind) -> list[str]:
    return [registry.display_name(key) for key in TASK_MODEL_KEYS.get(task, [])]


def classify(query: str, mode: InputMode) -> RouteDecision:
    lowered = (query or "").strip().lower()
    matched: list[str] = []

    if not lowered:
        task: TaskKind = "caption"
        matched = ["<empty query — defaulted to scene description>"]
    elif hits := _matches(lowered, _MEASURE):
        task, matched = "measure", hits
    elif hits := _matches(lowered, _CHANGE):
        matched = hits
        # "Has built-up increased?" wants a verdict; "what changed and where?"
        # wants a description. Only yes/no framing routes to change VQA.
        vqa_hits = _matches(lowered, _CHANGE_VQA)
        if vqa_hits or lowered.startswith(_CHANGE_VQA_OPENERS):
            task = "change-vqa"
            matched += vqa_hits or ["<yes/no framing>"]
        else:
            task = "change"
    elif hits := _matches(lowered, _FUSION):
        task, matched = "cross-modal", hits
    elif hits := _matches(lowered, _FLOOD):
        # Flood questions go cross-modal when a SAR partner is available, because
        # SAR is what sees water through cloud.
        task = "cross-modal" if mode == "cross-modal" else "vqa"
        matched = hits
    elif hits := _matches(lowered, _GROUNDING):
        task, matched = "grounding", hits
    elif (hits := _matches(lowered, _CAPTION)) and not lowered.startswith(_VQA_OPENERS):
        task, matched = "caption", hits
    else:
        task, matched = "vqa", ["<no keyword match — defaulted to VQA>"]

    warning: str | None = None
    if task in ("change", "change-vqa") and mode != "bi-temporal":
        warning = (
            "Change analysis needs two dates of the same area. Load a bi-temporal pair, "
            "or ask about the single scene instead."
        )
    elif task == "cross-modal" and mode != "cross-modal":
        if mode == "single":
            warning = "Optical–SAR fusion expects an optical/MSI scene plus a co-registered SAR scene."
        else:
            warning = "The supplied pair is not optical + SAR; fusion will fall back to single-scene reasoning."
    elif task == "measure":
        warning = None

    intent = Intent(
        task=task,
        label=TASK_LABELS[task],
        specialists=_specialists(task),
        warning=warning,
        matched_terms=matched,
    )
    return RouteDecision(intent=intent, target=resolve_target(lowered), matched=matched)


def degrade(task: TaskKind, mode: InputMode) -> TaskKind:
    """If the query asks for something the inputs cannot support, fall back to
    the closest task the inputs *can* answer rather than failing outright."""
    if task in ("change", "change-vqa") and mode != "bi-temporal":
        return "cross-modal" if mode == "cross-modal" else "vqa"
    if task == "cross-modal" and mode == "single":
        return "vqa"
    return task
