"""Score a trained classifier on the held-out split and write a JSON report.

The split is reproduced by hashing image paths exactly as training did, so the
patches scored here are the ones the model never trained on. Results are broken
down by modality as well as overall, because optical and SAR are not equally
easy and a single headline number hides that.

    python evaluate_classifier.py \
        --data /kaggle/input/bigearthnet-vqa \
        --checkpoint /kaggle/working/landcover/classifier.pt \
        --out /kaggle/working/landcover/eval_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satquery_ml import metrics
from satquery_ml.dataset import BigEarthNetPatches, load_records, split_records
from satquery_ml.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset root")
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Where to write the JSON report")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--retune-thresholds",
        action="store_true",
        help="Re-fit thresholds on this split (reports an optimistic upper bound)",
    )
    return parser.parse_args()


@torch.inference_mode()
def collect(model, loader: DataLoader, device: str):
    scores: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    modalities: list[torch.Tensor] = []

    for pixels, modality, target in loader:
        logits = model(pixels.to(device), modality.to(device))
        scores.append(torch.sigmoid(logits).float().cpu())
        targets.append(target.float())
        modalities.append(modality.argmax(dim=1))

    return torch.cat(scores), torch.cat(targets), torch.cat(modalities)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    jsonl = args.jsonl or (args.data / "train.jsonl")
    records = load_records(jsonl, root=args.data, limit=args.limit)
    _, val_records = split_records(records, args.val_fraction, args.seed)
    print(f"scoring {len(val_records)} held-out patches on {device} ...", flush=True)

    loader = DataLoader(
        BigEarthNetPatches(args.data, val_records, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model, saved_thresholds, train_metrics = load_checkpoint(args.checkpoint, device)
    scores, targets, modalities = collect(model, loader, device)

    thresholds = (
        metrics.tune_thresholds(scores, targets)
        if args.retune_thresholds
        else torch.tensor(saved_thresholds, dtype=torch.float32)
    )

    overall = metrics.report(scores, targets, thresholds)

    by_modality = {}
    for name, index in (("optical", 0), ("sar", 1)):
        mask = modalities == index
        if int(mask.sum().item()) == 0:
            continue
        by_modality[name] = metrics.report(scores[mask], targets[mask], thresholds)
        by_modality[name].pop("perClass", None)

    payload = {
        "checkpoint": str(args.checkpoint),
        "thresholdSource": "retuned on this split" if args.retune_thresholds else "from checkpoint",
        "trainingMetrics": train_metrics,
        "overall": overall,
        "byModality": by_modality,
    }

    print()
    print(f"mAP        {overall['mAP']:.4f}")
    print(f"macro F1   {overall['macroF1']:.4f}")
    print(f"micro F1   {overall['microF1']:.4f}")
    print(f"precision  {overall['microPrecision']:.4f}   recall {overall['microRecall']:.4f}")
    for name, block in by_modality.items():
        print(f"  {name:8s} mAP {block['mAP']:.4f} | macro F1 {block['macroF1']:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
