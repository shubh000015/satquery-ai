"""RSCoVLM-7B: the remote-sensing-adapted vision-language model.

RSCoVLM is Qwen2.5-VL-7B-Instruct co-trained on a remote-sensing recipe. That
adaptation is the point — the problem statement says outright that "a generic LLM
or VLM without remote-sensing adaptation will not satisfy the requirements", so
the unadapted base model is available only as a comparison via SATQUERY_VLM_MODEL.

One checkpoint serves two jobs, which is why it gets a GPU to itself:

  * `look()`  — genuine visual question answering and captioning. The model sees
    the image. Used for open-ended questions the classifier's 19 classes cannot
    express.
  * `phrase()` — text-only. Given `Evidence` from a specialist and no image, it
    rewrites the findings as a sentence. It cannot hallucinate here because it
    has nothing to hallucinate from.

Loading a second text-only Qwen just to phrase would waste ~6 GB for a job this
one already does.
"""

from __future__ import annotations

import os

import numpy as np

from ..facts import Evidence
from ..verbalizer import SYSTEM_PROMPT, enforce_verdict
from .base import Adapter, torch_dtype

# Vision prompt. Separate from SYSTEM_PROMPT because here the model *can* see the
# image, so the rule is calibration rather than abstention.
VISION_SYSTEM_PROMPT = (
    "You are SatQuery, a remote-sensing image analyst. You are looking at a "
    "satellite or aerial image. Answer in 1-3 plain sentences. Describe only what "
    "is visible; do not guess place names, dates, or coordinates. If the image is "
    "radar (grayscale, speckled, no natural colour), reason about structure and "
    "texture rather than colour. If you are unsure, say so."
)


class VlmAdapter(Adapter):
    """Qwen2.5-VL-family model used for both vision and grounded phrasing."""

    def __init__(self, spec, device: str = "cpu", load_4bit: bool = True):
        super().__init__(spec, device)
        # Env override exists so the unadapted base model can be swapped in for
        # a side-by-side, which is a fair comparison to show, not a default.
        self.model_id = os.environ.get("SATQUERY_VLM_MODEL", self._default_id())
        self.load_4bit = load_4bit

    def _default_id(self) -> str:
        # "https://huggingface.co/Qingyun/RSCoVLM-7B-2512" -> "Qingyun/RSCoVLM-7B-2512"
        return "/".join(self.spec.source.rstrip("/").rsplit("/", 2)[-2:])

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        quantization_config = None
        if self.load_4bit and self.device.startswith("cuda"):
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype(self.device),
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        # Qwen2.5-VL has a dedicated class; fall back for forks that register
        # themselves differently.
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VlmClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as VlmClass  # type: ignore

        self.model = VlmClass.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            dtype=torch_dtype(self.device),
            device_map={"": self.device} if self.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        self.model.eval()

    @property
    def label(self) -> str:
        return self.model_id.split("/")[-1]

    # ---- generation ------------------------------------------------------

    def _generate(self, messages: list[dict], images: list | None, max_new_tokens: int) -> str:
        import torch

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[1]
        trimmed = generated[0, prompt_length:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

    def phrase(self, evidence: Evidence, max_new_tokens: int = 140) -> str:
        """Rewrite `Evidence` as prose. Text only — the model gets no image.

        Falls back to the deterministic template on any failure, and the
        specialist's verdict wins if the model contradicts it.
        """
        self.ensure_loaded()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": evidence.fact_block()}]},
        ]
        try:
            answer = self._generate(messages, images=None, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: phrasing failed ({exc}); using template.", flush=True)
            return evidence.template_answer()
        return enforce_verdict(answer, evidence)

    def look(self, rgb: np.ndarray, question: str, max_new_tokens: int = 160) -> str:
        """Genuine visual QA. The model sees the image and answers directly."""
        self.ensure_loaded()

        from PIL import Image

        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VISION_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]
        return self._generate(messages, images=[image], max_new_tokens=max_new_tokens)

    def describe_pair(
        self, before: np.ndarray, after: np.ndarray, question: str, max_new_tokens: int = 180
    ) -> str:
        """Two-image prompt, used as the change-VQA path alongside ChangeFormer."""
        self.ensure_loaded()

        from PIL import Image

        images = [
            Image.fromarray(np.asarray(before, dtype=np.uint8), mode="RGB"),
            Image.fromarray(np.asarray(after, dtype=np.uint8), mode="RGB"),
        ]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": VISION_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1 (earlier date):"},
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": "Image 2 (later date):"},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": question},
                ],
            },
        ]
        return self._generate(messages, images=images, max_new_tokens=max_new_tokens)
