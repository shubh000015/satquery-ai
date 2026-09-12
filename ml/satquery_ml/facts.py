"""The grounded-evidence contract shared by every specialist model.

The design rule for the whole server: specialists decide *what is true*, the
language model only decides *how it reads*. ChangeFormer emits a mask, Grounding
DINO emits boxes, CROMA emits embeddings, the classifier emits labels — none of
them produce prose. Each one reduces its output to `Evidence`: a verdict, some
findings, and numbers computed deterministically from the model output.

`Evidence` is then the only thing the LLM ever sees. It never receives the image
on these paths, so it cannot invent land cover, place names, dates or areas, and
the confidence reported to the user is the specialist's, not the LLM's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Question openers that make a bare "yes"/"no" a sensible answer on its own.
_YES_NO_OPENERS = (
    "is ", "are ", "does ", "do ", "did ", "has ", "have ", "was ", "were ",
    "can ", "could ", "any ", "there is ", "there are ", "have there ",
)

MODALITY_PHRASE = {
    "sar": "SAR (radar backscatter, no colour information)",
    "optical": "optical (true colour)",
    "optical-sar": "co-registered optical and SAR",
    "bitemporal": "a pair of images of the same area at two dates",
}

TASK_INSTRUCTION = {
    "vqa": "Answer the analyst's question from the findings above.",
    "caption": "Write a short description of this scene from the findings above.",
    "grounding": (
        "State what was located and where, using only the findings above. "
        "Do not describe anything that was not detected."
    ),
    "change": (
        "Describe what changed between the two dates using only the findings "
        "above. The change fraction and region count are measured, not estimated."
    ),
    "fusion": (
        "Explain what the optical and SAR images say, and specifically what SAR "
        "adds beyond optical, using only the findings above."
    ),
}


def is_yes_no(question: str) -> bool:
    return question.strip().lower().startswith(_YES_NO_OPENERS)


def clamp_confidence(value: float) -> float:
    return round(max(0.15, min(0.97, float(value))), 3)


@dataclass
class Evidence:
    """Everything a specialist established, in a form the LLM can only rephrase."""

    task: str
    question: str = ""
    modality: str = "optical"

    # Human-readable statements, each traceable to a model output.
    findings: list[str] = field(default_factory=list)
    # Measured quantities. Values are pre-formatted so the LLM cannot re-round them.
    measurements: dict[str, str] = field(default_factory=dict)

    verdict: str | None = None
    confidence: float = 0.5
    terse: str = "unknown"

    # Spatial evidence produced alongside the text.
    boxes: list[dict] = field(default_factory=list)
    masks: list[dict] = field(default_factory=list)

    # Attribution for the execution trace, one entry per model that ran.
    models: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def one_word(self) -> str:
        """The specialist's terse answer, before the LLM touches it."""
        return self.verdict or self.terse

    def add_model(self, spec) -> None:
        block = spec.provenance()
        if block not in self.models:
            self.models.append(block)

    def fact_block(self) -> str:
        """The only context the language model receives."""
        sensor = MODALITY_PHRASE.get(self.modality, self.modality)
        lines = [f"Input: {sensor}"]

        if self.findings:
            lines.append("Findings from the specialist models:")
            lines.extend(f"- {item}" for item in self.findings)
        else:
            lines.append("Findings: nothing above the decision threshold.")

        for name, value in self.measurements.items():
            lines.append(f"Measured {name}: {value}")

        if self.verdict:
            lines.append(f"Verdict (must be preserved): {self.verdict}")
        lines.append(f"Model confidence: {self.confidence:.2f}")

        if self.question.strip() and self.task != "caption":
            lines.append(f"Analyst question: {self.question.strip()}")
        lines.append(
            TASK_INSTRUCTION.get(self.task, TASK_INSTRUCTION["vqa"])
        )
        return "\n".join(lines)

    def template_answer(self) -> str:
        """Deterministic phrasing, used whenever the language model is unavailable."""
        sensor = MODALITY_PHRASE.get(self.modality, self.modality)
        listing = "; ".join(self.findings) if self.findings else "nothing above threshold"

        if self.verdict == "yes":
            head = f"Yes. {listing}."
        elif self.verdict == "no":
            head = f"No. {listing}."
        elif self.verdict == "possibly":
            head = f"Possibly, but below the decision threshold. {listing}."
        elif self.task == "caption":
            head = f"A {sensor} scene. {listing}."
        else:
            head = f"{listing.capitalize()}."

        measured = ", ".join(f"{k} {v}" for k, v in self.measurements.items())
        if measured:
            head = f"{head} Measured: {measured}."
        return f"{head} (confidence {self.confidence:.2f})"

    def as_dict(self) -> dict:
        """Response body shared by every endpoint."""
        return {
            "task": self.task,
            "modality": self.modality,
            "oneWord": self.one_word,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "findings": list(self.findings),
            "measurements": dict(self.measurements),
            "boxes": list(self.boxes),
            "masks": list(self.masks),
            "models": list(self.models),
            "notes": list(self.notes),
        }
