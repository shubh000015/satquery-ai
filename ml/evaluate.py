"""Score a (base or fine-tuned) Qwen2.5-VL on an RSVQA-style JSONL split.

Reports normalised exact-match accuracy overall and per question category —
run it once on the base model and once with --adapter to show the judges the
before/after delta, which is the core SIH26167 evidence that the model was
actually adapted to remote sensing.

  python evaluate.py --data data/val.jsonl --limit 500                    # base
  python evaluate.py --data data/val.jsonl --limit 500 --adapter runs/rsvqa-lora/adapter
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 640 * 28 * 28

SYSTEM_PROMPT = (
    "You are SatQuery, an assistant for satellite and aerial imagery. "
    "Answer questions about remote sensing scenes concisely and factually."
)


def normalise(text: str) -> str:
    text = text.strip().lower().rstrip(".")
    text = re.sub(r"\s+", " ", text)
    # RSVQA numeric answers: compare the first number if both sides have one
    return text


def match(prediction: str, truth: str) -> bool:
    p, t = normalise(prediction), normalise(truth)
    if p == t:
        return True
    # "yes, the ..." should count as "yes"; same for leading numbers
    if t in ("yes", "no") and p.startswith(t):
        return True
    p_num = re.match(r"-?\d+", p)
    t_num = re.match(r"-?\d+", t)
    if p_num and t_num:
        return p_num.group() == t_num.group()
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir from train.py")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected — run on Kaggle/Colab or an RTX laptop.")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
        print(f"loaded adapter: {args.adapter}")
    model.eval()

    rows = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    correct = 0
    per_category: dict[str, list[bool]] = defaultdict(list)
    for i, row in enumerate(rows):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row["image"]},
                    {"type": "text", "text": row["question"]},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[Image.open(row["image"]).convert("RGB")], return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        answer = processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]

        ok = match(answer, row["answer"])
        correct += ok
        per_category[row.get("category", "unknown")].append(ok)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} · running accuracy {correct / (i + 1):.3f}")

    report = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "samples": len(rows),
        "accuracy": round(correct / max(1, len(rows)), 4),
        "per_category": {
            cat: round(sum(v) / len(v), 4) for cat, v in sorted(per_category.items())
        },
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
