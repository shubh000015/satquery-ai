"""Train the S1+S2 multi-label land-cover classifier (stage 1 of the pipeline).

This is the model we train ourselves. It starts from an ImageNet-pretrained
ResNet backbone — the standard practice for remote-sensing classification,
since ImageNet features transfer well to overhead imagery — and fine-tunes the
whole network on BigEarthNet patches with a fresh 19-class multi-label head.

Kaggle sessions get cut off, so the loop checkpoints on a wall-clock timer and
resumes automatically from the newest checkpoint in --out.

Typical run on a T4:

    python train_classifier.py \
        --data /kaggle/input/bigearthnet-vqa \
        --out /kaggle/working/landcover \
        --epochs 6 --batch-size 64

Quick sanity check that the plumbing works before spending GPU hours:

    python train_classifier.py --data <dataset> --out runs/smoke \
        --limit 400 --epochs 1 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from satquery_ml import metrics
from satquery_ml.dataset import (
    BigEarthNetPatches,
    class_frequencies,
    load_records,
    positive_weights,
    split_records,
)
from satquery_ml.labels import CLASSES
from satquery_ml.model import LandCoverClassifier, ModelConfig, save_checkpoint

RESUME_NAME = "last.ckpt"
BEST_NAME = "classifier.pt"
WARMUP_FRACTION = 0.1


def apply_lr(optimizer, base_lrs: list[float], step: int, total_steps: int) -> float:
    """Linear warmup then cosine decay, computed from the step count.

    Deliberately not a torch LRScheduler. A scheduler serialises its own
    `total_steps`, so resuming a run with a different --epochs restores the old
    schedule length and then throws once the step count passes it — which is
    exactly what happens on Kaggle, where you extend a run after a disconnect.
    Deriving the rate from global_step keeps resuming safe and stateless.
    """
    warmup = max(1, int(total_steps * WARMUP_FRACTION))
    if step < warmup:
        scale = (step + 1) / warmup
    else:
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))

    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale
    return base_lrs[-1] * scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Dataset root holding train.jsonl, images_s2/ and images_s1/",
    )
    parser.add_argument("--jsonl", type=Path, default=None, help="Override train.jsonl path")
    parser.add_argument("--out", type=Path, required=True, help="Checkpoint output dir")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="Head LR; backbone uses lr/10")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Cap patches (smoke tests)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--save-every-minutes",
        type=float,
        default=10.0,
        help="Wall-clock minutes between resume checkpoints",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return parser.parse_args()


def build_loaders(args: argparse.Namespace):
    jsonl = args.jsonl or (args.data / "train.jsonl")
    if not jsonl.exists():
        raise SystemExit(
            f"{jsonl} not found. Point --data at the folder produced by "
            "dataset_builder/build_bigearthnet_vqa.py (it contains train.jsonl)."
        )

    print(f"1/3 reading {jsonl} ...", flush=True)
    records = load_records(jsonl, root=args.data, limit=args.limit)
    train_records, val_records = split_records(records, args.val_fraction, args.seed)

    optical = sum(1 for r in records if r.modality == "optical")
    sar = len(records) - optical
    print(
        f"   {len(records)} patches ({optical} optical, {sar} SAR) -> "
        f"{len(train_records)} train / {len(val_records)} val"
    )

    train_loader = DataLoader(
        BigEarthNetPatches(args.data, train_records, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        BigEarthNetPatches(args.data, val_records, train=False),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, train_records


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for pixels, modality, targets in loader:
        pixels = pixels.to(device, non_blocking=True)
        modality = modality.to(device, non_blocking=True)
        logits = model(pixels, modality)
        all_scores.append(torch.sigmoid(logits).float().cpu())
        all_targets.append(targets.float())

    return torch.cat(all_scores), torch.cat(all_targets)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: no CUDA GPU found. This will be extremely slow — run it on "
            "Kaggle with Settings > Accelerator > GPU.",
            flush=True,
        )
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_loader, val_loader, train_records = build_loaders(args)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"2/3 building {args.backbone} + 19-class multi-label head ...", flush=True)
    model = LandCoverClassifier(ModelConfig(backbone=args.backbone)).to(device)

    pos_weight = positive_weights(train_records).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Pretrained features need gentler updates than the randomly initialised head.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr / 10},
            {"params": model.head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    base_lrs = [args.lr / 10, args.lr]
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs

    use_amp = not args.no_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    global_step = 0
    best_macro_f1 = -1.0

    resume_path = args.out / RESUME_NAME
    if resume_path.exists() and not args.no_resume:
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        start_epoch = payload["epoch"]
        global_step = payload["global_step"]
        best_macro_f1 = payload.get("best_macro_f1", -1.0)
        print(f"   RESUMING from {resume_path} at epoch {start_epoch}", flush=True)
    else:
        print("   starting a fresh run", flush=True)

    def save_resume(epoch: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_macro_f1": best_macro_f1,
            },
            resume_path,
        )

    print(f"3/3 training for {args.epochs} epochs ({total_steps} steps) ...", flush=True)
    last_save = time.monotonic()
    save_interval = max(60.0, args.save_every_minutes * 60.0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_started = time.monotonic()

        for batch_index, (pixels, modality, targets) in enumerate(train_loader, start=1):
            pixels = pixels.to(device, non_blocking=True)
            modality = modality.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            current_lr = apply_lr(optimizer, base_lrs, global_step, total_steps)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(pixels, modality)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * pixels.size(0)
            seen += pixels.size(0)
            global_step += 1

            if batch_index % 50 == 0:
                rate = seen / max(1e-6, time.monotonic() - epoch_started)
                print(
                    f"   epoch {epoch + 1}/{args.epochs} "
                    f"step {batch_index}/{steps_per_epoch} "
                    f"loss {running_loss / max(1, seen):.4f} "
                    f"lr {current_lr:.2e} ({rate:.0f} img/s)",
                    flush=True,
                )

            if time.monotonic() - last_save >= save_interval:
                save_resume(epoch)
                last_save = time.monotonic()
                print(f"   [checkpoint] saved at step {global_step}", flush=True)

        scores, targets_all = validate(model, val_loader, device)
        thresholds = metrics.tune_thresholds(scores, targets_all)
        summary = metrics.report(scores, targets_all, thresholds)

        print(
            f"   epoch {epoch + 1} done | train loss {running_loss / max(1, seen):.4f} "
            f"| val mAP {summary['mAP']:.4f} | macro F1 {summary['macroF1']:.4f} "
            f"| micro F1 {summary['microF1']:.4f}",
            flush=True,
        )

        if summary["macroF1"] > best_macro_f1:
            best_macro_f1 = summary["macroF1"]
            save_checkpoint(
                args.out / BEST_NAME,
                model,
                thresholds=[round(float(t), 3) for t in thresholds],
                metrics={"epoch": epoch + 1, **summary},
            )
            print(f"   new best macro F1 -> {args.out / BEST_NAME}", flush=True)

        save_resume(epoch + 1)
        last_save = time.monotonic()

    frequencies = class_frequencies(train_records)
    (args.out / "training_summary.json").write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "epochs": args.epochs,
                "batchSize": args.batch_size,
                "headLr": args.lr,
                "trainPatches": len(train_records),
                "bestMacroF1": best_macro_f1,
                "classSupport": {
                    CLASSES[i]: int(frequencies[i].item()) for i in range(len(CLASSES))
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"DONE - best val macro F1 {best_macro_f1:.4f}")
    print(f"  deployable weights : {args.out / BEST_NAME}")
    print(f"  metrics sidecar    : {(args.out / BEST_NAME).with_suffix('.json')}")
    print("  next: evaluate_classifier.py for the full report, then serve.py")


if __name__ == "__main__":
    main()
