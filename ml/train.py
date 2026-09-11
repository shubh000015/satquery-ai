"""QLoRA fine-tuning of Qwen2.5-VL for remote-sensing VQA/captioning (SIH26167).

Method (what the current RS-VLM papers converged on):
  - Base model loaded in 4-bit NF4 (bitsandbytes) so a 3B/7B model fits a
    free 16 GB T4/P100.
  - LoRA adapters (r=16, alpha=32) on the language decoder's attention
    projections only (q/k/v/o). The vision encoder stays frozen so the
    pretrained visual representations are preserved.
  - Labels are masked over the prompt and padding: loss is computed only on
    the assistant's answer tokens.

Typical runs:

  # sanity check the loop first (minutes, tiny synthetic data)
  python train.py --train data/smoke.jsonl --out runs/smoke --epochs 2 --max-samples 96

  # real run on RSVQA-LR (subsampled) — roughly 3-6 h on a T4/P100
  python train.py --train data/train.jsonl --val data/val.jsonl --out runs/rsvqa-lora

The output directory contains the LoRA adapter (~100-200 MB), which is what
you download from Kaggle/Colab and load in serve.py. The base model is pulled
from Hugging Face at load time; only the adapter is yours.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class TimedCheckpointCallback(TrainerCallback):
    """Force a checkpoint every `every_minutes` wall-clock minutes (Kaggle-safe)."""

    def __init__(self, every_minutes: float = 10.0):
        self.every_seconds = max(60.0, every_minutes * 60.0)
        self._last = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        if now - self._last >= self.every_seconds:
            control.should_save = True
            self._last = now
            print(
                f"\n[checkpoint] timed save at step {state.global_step} "
                f"(every {self.every_seconds / 60:.0f} min)\n",
                flush=True,
            )
        return control


def find_latest_checkpoint(out: Path) -> Path | None:
    ckpts = [p for p in out.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None

    def step_num(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    return max(ckpts, key=step_num)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Cap vision tokens so batches fit a 16 GB card. 28*28 is the ViT patch grid
# unit Qwen's processor works in.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 640 * 28 * 28

SYSTEM_PROMPT = (
    "You are SatQuery, an assistant for satellite and aerial imagery. "
    "Answer questions about remote sensing scenes concisely and factually."
)


class JsonlVqaDataset(Dataset):
    def __init__(self, path: Path, max_samples: int | None = 12_000, seed: int = 13):
        # Do not path.read_text() a ~1 GB jsonl — that OOMs Kaggle CPU RAM.
        offsets: list[int] = []
        with path.open("rb") as fh:
            pos = 0
            for raw in fh:
                if raw.strip():
                    offsets.append(pos)
                pos += len(raw)
        if max_samples and len(offsets) > max_samples:
            offsets = random.Random(seed).sample(offsets, max_samples)
        rows = []
        with path.open("rb") as fh:
            for pos in offsets:
                fh.seek(pos)
                rows.append(json.loads(fh.readline()))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


@dataclass
class Collator:
    """Turns raw rows into padded batches with prompt/padding tokens masked out
    of the labels. Getting this masking right matters more than any
    hyperparameter — wrong masking trains on the question text."""

    processor: AutoProcessor

    def _messages(self, row: dict, with_answer: bool) -> list[dict]:
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
        if with_answer:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]})
        return messages

    def __call__(self, rows: list[dict]) -> dict:
        texts, prompt_texts, images = [], [], []
        for row in rows:
            full = self._messages(row, with_answer=True)
            prompt = self._messages(row, with_answer=False)
            texts.append(self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False))
            prompt_texts.append(self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
            images.append(Image.open(row["image"]).convert("RGB"))

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        # Mask everything up to (and including) the generation prompt so the
        # loss covers only the answer tokens.
        for i, prompt_text in enumerate(prompt_texts):
            prompt_ids = self.processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
            labels[i, : prompt_ids.shape[0]] = -100

        # Never train on image placeholder tokens.
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_token_id is not None:
            labels[batch["input_ids"] == image_token_id] = -100

        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="3B default; Qwen/Qwen2.5-VL-7B-Instruct also fits a T4")
    parser.add_argument("--train", type=Path, required=True, help="train JSONL from prepare_data.py")
    parser.add_argument("--val", type=Path, default=None, help="optional val JSONL")
    parser.add_argument("--out", type=Path, required=True, help="output dir for the LoRA adapter")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-every-minutes", type=float, default=10.0,
                        help="Wall-clock minutes between forced checkpoints (Kaggle)")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Also checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoint-* and start fresh")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. QLoRA needs an NVIDIA GPU — run this on "
            "Kaggle (Settings > Accelerator > GPU T4/P100) or Colab, or a "
            "laptop with an RTX card. See ml/README.md."
        )

    bf16_ok = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} · compute dtype: {compute_dtype}")

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

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Freeze the vision tower explicitly (LoRA targets below only exist in the
    # language decoder, but belt and braces).
    for name, param in model.named_parameters():
        if name.startswith("visual."):
            param.requires_grad = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = JsonlVqaDataset(args.train, args.max_samples, args.seed)
    val_ds = JsonlVqaDataset(args.val, max_samples=512, seed=args.seed) if args.val else None
    print(f"train samples: {len(train_ds)}" + (f" · val samples: {len(val_ds)}" if val_ds else ""))

    training_args = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if val_ds else "no",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
    )

    resume = None if args.no_resume else find_latest_checkpoint(args.out)
    if resume:
        print(f"RESUMING from {resume}")
    else:
        print("No checkpoint found — starting a new run")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(processor),
        callbacks=[TimedCheckpointCallback(args.save_every_minutes)],
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapter saved to {adapter_dir}")
    print("Next: evaluate.py to score it, serve.py to expose it to the SatQuery backend.")


if __name__ == "__main__":
    main()
