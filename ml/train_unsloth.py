"""Unsloth Qwen2.5-VL training with timed checkpoints + auto-resume (Kaggle-safe).

Why this exists
---------------
Kaggle GPU sessions die after a few hours (or on idle). This script:

  1. Saves a full Trainer checkpoint every N minutes (default 10)
  2. Also saves every N steps as a backup
  3. On the next run, automatically resumes from the latest checkpoint

Kaggle persistence (read this)
------------------------------
`/kaggle/working` is wiped when a *new* session starts unless you:

  A) Notebook Settings → Persistence → Files  (keeps working dir), OR
  B) After each session: zip checkpoints, download, re-upload as a Kaggle
     Dataset named e.g. `ben-lora-checkpoints`, then attach it next time, OR
  C) Save Version while training is mid-way, then "Copy and Edit" that version

Pass `--resume-dir` if checkpoints live under `/kaggle/input/...`.

Example (Kaggle cell)::

    !python train_unsloth.py \\
        --data /kaggle/input/bigearthnet-vqa \\
        --out /kaggle/working/ben-lora \\
        --save-every-minutes 10 \\
        --resume-dir /kaggle/input/ben-lora-checkpoints

Prefer a notebook? `ml/kaggle_train_unsloth.ipynb` is the same logic as
ready-to-upload Kaggle cells.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

# Full BigEarthNet VQA jsonl is ~1 GB / millions of rows. Loading it as Python
# dicts OOMs Kaggle CPU RAM (~13 GB). Cap keeps one session inside a T4 quota.
DEFAULT_MAX_SAMPLES = 12_000

from transformers import TrainerCallback


SYSTEM = (
    "You are SatQuery, an assistant for satellite and aerial imagery. "
    "Answer questions about optical and SAR remote sensing scenes "
    "concisely and factually."
)


class TimedCheckpointCallback(TrainerCallback):
    """Force a checkpoint every `every_minutes` wall-clock minutes."""

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


def find_latest_checkpoint(roots: list[Path]) -> Path | None:
    """Return newest checkpoint-* directory under any of the roots."""
    candidates: list[Path] = []
    for root in roots:
        if not root or not root.exists():
            continue
        candidates.extend(p for p in root.glob("checkpoint-*") if p.is_dir())
        # Also accept a nested layout: root/ben-lora/checkpoint-*
        candidates.extend(p for p in root.glob("**/checkpoint-*") if p.is_dir())
    if not candidates:
        return None

    def step_num(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    return max(candidates, key=step_num)


def resolve_data_root(data_root: Path) -> Path:
    """Accept the dataset root or a folder one level above train.jsonl.

    Kaggle datasets sometimes unpack with an extra top-level folder depending
    on how they were uploaded, so look one level down before giving up.
    """
    if (data_root / "train.jsonl").exists():
        return data_root
    if data_root.exists():
        for child in sorted(p for p in data_root.iterdir() if p.is_dir()):
            if (child / "train.jsonl").exists():
                return child
    raise SystemExit(
        f"train.jsonl not found under {data_root}. "
        "Attach the Kaggle dataset (knayamket/bigearthnet-vqa) or fix --data."
    )


def index_jsonl_offsets(path: Path) -> list[int]:
    """Byte offsets of non-empty lines. Does not parse JSON (cheap on a 1 GB file)."""
    size_gb = path.stat().st_size / 1e9
    print(f"indexing {path} ({size_gb:.2f} GB) — not loading it into RAM ...", flush=True)
    offsets: list[int] = []
    with path.open("rb") as fh:
        pos = 0
        for raw in fh:
            if raw.strip():
                offsets.append(pos)
            pos += len(raw)
            if len(offsets) and len(offsets) % 1_000_000 == 0:
                print(f"  indexed {len(offsets):,} rows", flush=True)
    print(f"indexed {len(offsets):,} jsonl rows", flush=True)
    return offsets


def _row_to_example(row: dict, data_root: Path) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM + "\n\n" + row["question"]},
                    {"type": "image", "image": str(data_root / row["image"])},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["answer"]}],
            },
        ]
    }


def load_dataset(data_root: Path, max_samples: int | None, seed: int = 13):
    """Parse only `max_samples` random rows (default 12k). Never json.loads the full file."""
    data_root = resolve_data_root(data_root)
    train_jsonl = data_root / "train.jsonl"
    offsets = index_jsonl_offsets(train_jsonl)
    total = len(offsets)
    if not total:
        raise SystemExit(f"{train_jsonl} is empty")

    if max_samples is None:
        if total > 50_000:
            raise SystemExit(
                f"{train_jsonl} has {total:,} rows (~1 GB). Loading them all as Python "
                f"objects will OOM Kaggle CPU RAM. Re-run without --full "
                f"(default cap is {DEFAULT_MAX_SAMPLES} random samples)."
            )
        print(f"using all {total:,} rows", flush=True)
    elif total > max_samples:
        rng = random.Random(seed)
        offsets = rng.sample(offsets, max_samples)
        print(f"sampling {len(offsets):,} / {total:,} rows (seed={seed})", flush=True)
    else:
        print(f"using all {total:,} rows", flush=True)

    dataset: list[dict] = []
    skipped = 0
    with train_jsonl.open("rb") as fh:
        for pos in offsets:
            fh.seek(pos)
            try:
                row = json.loads(fh.readline())
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "image" not in row or "question" not in row or "answer" not in row:
                skipped += 1
                continue
            dataset.append(_row_to_example(row, data_root))

    if skipped:
        print(f"warning: skipped {skipped} unreadable rows")
    if not dataset:
        raise SystemExit("No usable training samples")
    print(f"usable samples: {len(dataset)}", flush=True)
    return dataset


def seed_checkpoints_from_resume(resume_dir: Path | None, out_dir: Path) -> None:
    """Copy uploaded checkpoints into output_dir so Trainer can resume locally."""
    if resume_dir is None or not resume_dir.exists():
        return
    # If resume_dir already contains checkpoint-*, copy them into out_dir
    src_ckpts = list(resume_dir.glob("checkpoint-*"))
    if not src_ckpts:
        src_ckpts = list(resume_dir.glob("**/checkpoint-*"))
    if not src_ckpts:
        print(f"no checkpoint-* under {resume_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for src in src_ckpts:
        if not src.is_dir():
            continue
        dest = out_dir / src.name
        if dest.exists():
            continue
        print(f"seeding checkpoint -> {dest}")
        shutil.copytree(src, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/kaggle/input/bigearthnet-vqa"),
        help="Folder with train.jsonl + images_s2/ (+ images_s1/). "
        "Defaults to the attached Kaggle dataset knayamket/bigearthnet-vqa.",
    )
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/ben-lora"))
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Optional /kaggle/input/... folder that already has checkpoint-* from a previous session",
    )
    parser.add_argument("--model", default="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap (smoke tests)")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Random subset of train.jsonl (default {DEFAULT_MAX_SAMPLES}). "
        "Do NOT load the full ~1 GB file on Kaggle.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on every jsonl row. Will OOM on Kaggle CPU RAM — laptop/A100 only.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-every-minutes", type=float, default=10.0,
                        help="Wall-clock minutes between forced checkpoints (default 10)")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Also checkpoint every N optimizer steps (default 50)")
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Keep only the newest N checkpoints to save disk")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and start fresh")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    # Import heavy deps only when running (keeps --help light on Macs without CUDA).
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    cap = None if args.full else args.max_samples
    dataset = load_dataset(args.data, cap, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.no_resume:
        seed_checkpoints_from_resume(args.resume_dir, args.out)

    resume_path = None if args.no_resume else find_latest_checkpoint([args.out, args.resume_dir] if args.resume_dir else [args.out])
    if resume_path:
        print(f"RESUMING from {resume_path}")
    else:
        print("No checkpoint found — starting a new run")

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        random_state=args.seed,
    )
    FastVisionModel.for_training(model)

    sft_kwargs = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        learning_rate=args.lr,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        output_dir=str(args.out),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
    )
    if args.max_steps:
        sft_kwargs["max_steps"] = args.max_steps
    else:
        sft_kwargs["num_train_epochs"] = args.epochs

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(**sft_kwargs),
        callbacks=[TimedCheckpointCallback(args.save_every_minutes)],
    )

    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)

    adapter_dir = args.out / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nFinal adapter -> {adapter_dir}")
    print("Zip for download:")
    print(f"  !cd {args.out.parent} && zip -r ben-lora-adapter.zip {args.out.name}/adapter {args.out.name}/checkpoint-*")


if __name__ == "__main__":
    # Avoid HF token prompts on Kaggle
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
