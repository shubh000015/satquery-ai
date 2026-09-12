"""Stage 2: turn the classifier's labels into a readable answer with Qwen2.5.

Division of labour, which is the whole point of the two-stage design:

  * The classifier decides *what is true* — which classes are present, and the
    yes/no verdict for a presence question. Those come from `Prediction`.
  * Qwen2.5 decides *how it reads* — it phrases the findings as a sentence and
    nothing more. It is never asked to look at the image, so it cannot
    hallucinate land cover, and the confidence we report is the classifier's.

If the language model cannot be loaded, `Verbalizer.phrase` degrades to a
template so the endpoint still answers instead of failing the demo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import labels as label_vocab
from .inference import Prediction

DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are SatQuery, an assistant that explains satellite image analysis to analysts. "
    "A land-cover classifier has already analysed the image and you are given its findings. "
    "Rules you must follow:\n"
    "1. Use ONLY the findings provided. Never introduce land-cover types, place names, "
    "dates, object counts, or area measurements that are not in the findings.\n"
    "2. If a verdict is provided, your answer must agree with it.\n"
    "3. You cannot see the image yourself, so never claim to observe anything directly.\n"
    "4. Reply with 1-3 plain sentences. No bullet points, headings, or preamble."
)

_YES_NO_OPENERS = (
    "is ", "are ", "does ", "do ", "did ", "has ", "have ", "was ", "were ",
    "can ", "could ", "any ", "there is ", "there are ",
)

MODALITY_PHRASE = {
    "sar": "Sentinel-1 SAR (radar amplitude, no colour information)",
    "optical": "Sentinel-2 optical (true colour)",
}


@dataclass
class Facts:
    task: str
    question: str
    modality: str
    present: list[str] = field(default_factory=list)
    target: str | None = None
    target_label: str | None = None
    target_score: float | None = None
    verdict: str | None = None
    confidence: float = 0.5

    @property
    def one_word(self) -> str:
        """The classifier's terse answer, before the language model touches it."""
        if self.verdict:
            return self.verdict
        return self.present[0] if self.present else "unknown"


def _is_yes_no(question: str) -> bool:
    lowered = question.strip().lower()
    return lowered.startswith(_YES_NO_OPENERS)


def _clamp(value: float) -> float:
    return round(max(0.15, min(0.97, value)), 3)


def resolve_facts(prediction: Prediction, question: str, task: str = "vqa") -> Facts:
    """Decide the answer from the classifier alone. No language model involved."""
    present_short = [label_vocab.short_name(name) for name in prediction.present]
    present_scores = [prediction.score(name) for name in prediction.present]
    mean_confidence = (
        sum(present_scores) / len(present_scores) if present_scores else 0.4
    )

    facts = Facts(
        task=task,
        question=question.strip(),
        modality=prediction.modality,
        present=present_short,
        confidence=_clamp(mean_confidence),
    )

    if task == "caption" or not question.strip():
        return facts

    target = label_vocab.match_class_in_question(question)
    if target is None:
        return facts

    is_group = target in label_vocab.GROUPS
    if is_group:
        score = prediction.group_score(target)
        hits = prediction.group_present(target)
        threshold = min(
            (prediction.threshold(name) for name in label_vocab.GROUPS[target]),
            default=0.5,
        )
        facts.target = target
        facts.target_label = target if target != "builtup" else "built-up area"
        matched = bool(hits)
    else:
        score = prediction.score(target)
        threshold = prediction.threshold(target)
        facts.target = target
        facts.target_label = label_vocab.short_name(target)
        matched = target in prediction.present

    facts.target_score = round(score, 4)

    if matched:
        facts.verdict = "yes"
        facts.confidence = _clamp(score)
    elif score >= 0.5 * threshold:
        facts.verdict = "possibly"
        facts.confidence = _clamp(0.5)
    else:
        facts.verdict = "no"
        facts.confidence = _clamp(1.0 - score)

    if not _is_yes_no(question):
        # An open question about a specific class still benefits from the
        # presence check, but "yes"/"no" is not a sensible standalone answer.
        facts.verdict = None if not matched else facts.verdict

    return facts


def template_answer(facts: Facts) -> str:
    """Deterministic phrasing, used when the language model is unavailable."""
    sensor = MODALITY_PHRASE.get(facts.modality, facts.modality)
    listing = ", ".join(facts.present) if facts.present else "no confident class"

    if facts.verdict == "yes" and facts.target_label:
        return (
            f"Yes — the classifier detected {facts.target_label} in this {sensor} scene "
            f"(confidence {facts.confidence:.2f}). Full land cover: {listing}."
        )
    if facts.verdict == "no" and facts.target_label:
        return (
            f"No — the classifier found no {facts.target_label} in this {sensor} scene. "
            f"What it did detect: {listing}."
        )
    if facts.verdict == "possibly" and facts.target_label:
        return (
            f"Possibly — {facts.target_label} scored below the decision threshold "
            f"({facts.target_score:.2f}), so this is uncertain. Detected land cover: {listing}."
        )
    if facts.task == "caption":
        return f"A {sensor} patch. The classifier identified {listing}."
    return f"This {sensor} scene contains {listing}."


def _fact_block(facts: Facts) -> str:
    sensor = MODALITY_PHRASE.get(facts.modality, facts.modality)
    lines = [
        f"Sensor: {sensor}",
        f"Land cover detected by the classifier: {', '.join(facts.present) or 'none above threshold'}",
    ]
    if facts.target_label is not None and facts.target_score is not None:
        lines.append(
            f"Presence check for '{facts.target_label}': score {facts.target_score:.2f}"
        )
    if facts.verdict:
        lines.append(f"Verdict (must be preserved): {facts.verdict}")
    lines.append(f"Classifier confidence: {facts.confidence:.2f}")

    if facts.task == "caption":
        lines.append(
            "Task: write a short description of this scene from the findings above."
        )
    else:
        lines.append(f"Analyst question: {facts.question}")
    return "\n".join(lines)


class Verbalizer:
    """Wraps a Qwen2.5 instruct model. Text-only — the vision work is stage 1."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM,
        load_4bit: bool = True,
        device_map: str = "auto",
        enabled: bool = True,
    ):
        self.model_name = model_name if enabled else "template"
        self.available = False
        self.tokenizer = None
        self.model = None
        self.load_error: str | None = None

        if not enabled:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            quantization_config = None
            if load_4bit and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig

                compute_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                dtype="auto",
                device_map=device_map if torch.cuda.is_available() else None,
            )
            self.model.eval()
            self.available = True
        except Exception as exc:  # noqa: BLE001 - degrade instead of crashing the server
            self.load_error = f"{type(exc).__name__}: {exc}"
            print(
                f"WARNING: could not load {model_name} ({self.load_error}).\n"
                "         Falling back to template phrasing.",
                flush=True,
            )

    @property
    def label(self) -> str:
        return self.model_name.split("/")[-1] if self.available else "template"

    def phrase(self, facts: Facts, max_new_tokens: int = 120) -> str:
        if not self.available:
            return template_answer(facts)

        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _fact_block(facts)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: generation failed ({exc}); using template.", flush=True)
            return template_answer(facts)

        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answer = re.sub(r"\s+", " ", answer)

        if not answer:
            return template_answer(facts)

        # Guard the one thing the model is not allowed to change.
        if facts.verdict in {"yes", "no"}:
            opening = answer[:24].lower()
            opposite = "no" if facts.verdict == "yes" else "yes"
            if opening.startswith(opposite):
                return template_answer(facts)

        return answer
