"""Multi-label metrics: average precision, F1, and per-class threshold tuning.

Pure torch on purpose — no scikit-learn dependency to install on Kaggle, and
the numbers stay identical between the training loop and evaluate_classifier.py.
"""

from __future__ import annotations

import torch

from .labels import CLASSES


def average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Area under the precision-recall curve for one class.

    Returns NaN when the class has no positive example, so the caller can leave
    it out of the macro average instead of scoring it as zero.
    """
    positives = int(targets.sum().item())
    if positives == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order]

    true_positives = torch.cumsum(sorted_targets, dim=0)
    ranks = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
    precision = true_positives / ranks
    recall = true_positives / positives

    # Sum precision only where recall actually increases (i.e. at true positives).
    previous_recall = torch.cat([torch.zeros(1), recall[:-1]])
    return float(((recall - previous_recall) * precision).sum().item())


def mean_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> tuple[float, list[float]]:
    per_class = [
        average_precision(scores[:, i], targets[:, i]) for i in range(targets.shape[1])
    ]
    valid = [value for value in per_class if value == value]  # drop NaN
    return (sum(valid) / len(valid) if valid else 0.0), per_class


def _counts(predictions: torch.Tensor, targets: torch.Tensor):
    true_positive = (predictions * targets).sum(dim=0)
    false_positive = (predictions * (1 - targets)).sum(dim=0)
    false_negative = ((1 - predictions) * targets).sum(dim=0)
    return true_positive, false_positive, false_negative


def f1_scores(
    scores: torch.Tensor, targets: torch.Tensor, thresholds: torch.Tensor
) -> dict[str, float]:
    predictions = (scores >= thresholds.unsqueeze(0)).float()
    true_positive, false_positive, false_negative = _counts(predictions, targets)

    micro_tp = true_positive.sum()
    micro_precision = micro_tp / torch.clamp(micro_tp + false_positive.sum(), min=1.0)
    micro_recall = micro_tp / torch.clamp(micro_tp + false_negative.sum(), min=1.0)
    micro_f1 = (
        2 * micro_precision * micro_recall / torch.clamp(micro_precision + micro_recall, min=1e-8)
    )

    precision = true_positive / torch.clamp(true_positive + false_positive, min=1.0)
    recall = true_positive / torch.clamp(true_positive + false_negative, min=1.0)
    per_class_f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)

    present = (targets.sum(dim=0) > 0)
    macro_f1 = per_class_f1[present].mean() if present.any() else torch.zeros(1)

    exact_match = (predictions == targets).all(dim=1).float().mean()

    return {
        "microF1": float(micro_f1.item()),
        "macroF1": float(macro_f1.item()),
        "microPrecision": float(micro_precision.item()),
        "microRecall": float(micro_recall.item()),
        "exactMatch": float(exact_match.item()),
    }


def tune_thresholds(
    scores: torch.Tensor, targets: torch.Tensor, grid: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-class threshold that maximises that class's F1 on the given split.

    A single global 0.5 cutoff badly under-predicts the rare classes; tuning per
    class typically adds several points of macro F1. Tune on validation only.
    """
    grid = grid if grid is not None else torch.arange(0.05, 0.96, 0.05)
    num_classes = targets.shape[1]
    best = torch.full((num_classes,), 0.5)

    for class_index in range(num_classes):
        class_scores = scores[:, class_index]
        class_targets = targets[:, class_index]
        if class_targets.sum() == 0:
            continue

        best_f1 = -1.0
        for threshold in grid:
            predictions = (class_scores >= threshold).float()
            true_positive = (predictions * class_targets).sum()
            false_positive = (predictions * (1 - class_targets)).sum()
            false_negative = ((1 - predictions) * class_targets).sum()
            precision = true_positive / torch.clamp(true_positive + false_positive, min=1.0)
            recall = true_positive / torch.clamp(true_positive + false_negative, min=1.0)
            f1 = float(
                (2 * precision * recall / torch.clamp(precision + recall, min=1e-8)).item()
            )
            if f1 > best_f1:
                best_f1 = f1
                best[class_index] = float(threshold)

    return best


def report(
    scores: torch.Tensor, targets: torch.Tensor, thresholds: torch.Tensor
) -> dict:
    map_score, per_class_ap = mean_average_precision(scores, targets)
    summary = f1_scores(scores, targets, thresholds)
    summary["mAP"] = map_score
    summary["samples"] = int(targets.shape[0])
    summary["perClass"] = {
        CLASSES[i]: {
            "ap": None if per_class_ap[i] != per_class_ap[i] else round(per_class_ap[i], 4),
            "threshold": round(float(thresholds[i].item()), 2),
            "support": int(targets[:, i].sum().item()),
        }
        for i in range(targets.shape[1])
    }
    return summary
