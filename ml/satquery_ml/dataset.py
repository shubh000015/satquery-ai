"""Turn the BigEarthNet VQA folder into a multi-label classification dataset.

We reuse the Kaggle dataset produced by `dataset_builder/build_bigearthnet_vqa.py`
(`train.jsonl` + `images_s2/` + `images_s1/`) rather than re-deriving anything
from the 66 GB source archives. Only the `category == "multi-label"` rows carry
the full label set for a patch; the presence/modality rows are QA phrasings of
the same ground truth, so they are ignored here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from . import labels as label_vocab

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

MODALITY_INDEX = {"optical": 0, "sar": 1}


@dataclass(slots=True)
class PatchRecord:
    rel_path: str
    modality: str
    label_indices: tuple[int, ...]


def _modality_of(row: dict) -> str:
    modality = str(row.get("modality") or "").lower()
    if modality in MODALITY_INDEX:
        return modality
    # Fall back to the folder the builder wrote the image into.
    return "sar" if "images_s1/" in str(row.get("image", "")) else "optical"


def load_records(
    jsonl_path: Path,
    root: Path | None = None,
    limit: int | None = None,
    verify_files: bool = True,
) -> list[PatchRecord]:
    """Stream the JSONL and return one record per (patch, modality).

    The file is read line by line: on the full S1+S2 build it is ~1 GB, which
    does not fit comfortably in Kaggle's CPU RAM if slurped whole.
    """
    root = root or jsonl_path.parent
    seen: dict[str, PatchRecord] = {}
    missing = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("category") != "multi-label":
                continue

            rel_path = str(row.get("image") or "")
            if not rel_path or rel_path in seen:
                continue

            class_names = label_vocab.parse_label_string(str(row.get("answer") or ""))
            if not class_names:
                continue

            if verify_files and not (root / rel_path).exists():
                missing += 1
                continue

            seen[rel_path] = PatchRecord(
                rel_path=rel_path,
                modality=_modality_of(row),
                label_indices=tuple(
                    label_vocab.CLASS_TO_INDEX[name.lower()] for name in class_names
                ),
            )

            if limit and len(seen) >= limit:
                break

    records = list(seen.values())
    if not records:
        raise SystemExit(
            f"No usable multi-label rows in {jsonl_path}. Expected lines with "
            '"category": "multi-label" and an "image" path relative to '
            f"{root}."
        )
    if missing:
        print(f"  note: skipped {missing} rows whose image file was not extracted")
    return records


def split_records(
    records: list[PatchRecord],
    val_fraction: float = 0.1,
    seed: int = 13,
) -> tuple[list[PatchRecord], list[PatchRecord]]:
    """Deterministic split keyed on the image path.

    Hashing the path (instead of shuffling) keeps the split identical across
    runs and machines, so a resumed Kaggle session cannot leak validation
    patches into training.
    """
    train: list[PatchRecord] = []
    val: list[PatchRecord] = []
    threshold = int(val_fraction * 10_000)

    for record in records:
        digest = hashlib.sha1(f"{seed}:{record.rel_path}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10_000
        (val if bucket < threshold else train).append(record)

    if not val:
        raise SystemExit("Validation split is empty — raise --val-fraction.")
    return train, val


def build_transform(train: bool):
    if train:
        # Overhead imagery has no canonical up, so flips and 90-degree rotations
        # are label-preserving and effectively quadruple the patch count.
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation((0, 0)),
                        transforms.RandomRotation((90, 90)),
                        transforms.RandomRotation((180, 180)),
                        transforms.RandomRotation((270, 270)),
                    ]
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def modality_tensor(modality: str) -> torch.Tensor:
    onehot = torch.zeros(len(MODALITY_INDEX), dtype=torch.float32)
    onehot[MODALITY_INDEX.get(modality, 0)] = 1.0
    return onehot


class BigEarthNetPatches(Dataset):
    def __init__(self, root: Path, records: list[PatchRecord], train: bool):
        self.root = Path(root)
        self.records = records
        self.transform = build_transform(train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(self.root / record.rel_path).convert("RGB")
        pixels = self.transform(image)

        target = torch.zeros(label_vocab.NUM_CLASSES, dtype=torch.float32)
        for class_index in record.label_indices:
            target[class_index] = 1.0

        return pixels, modality_tensor(record.modality), target


def class_frequencies(records: list[PatchRecord]) -> torch.Tensor:
    counts = torch.zeros(label_vocab.NUM_CLASSES, dtype=torch.float32)
    for record in records:
        for class_index in record.label_indices:
            counts[class_index] += 1
    return counts


def positive_weights(records: list[PatchRecord], cap: float = 10.0) -> torch.Tensor:
    """BCE `pos_weight` from label frequency.

    BigEarthNet is heavily imbalanced (marine waters and coastal wetlands are
    rare), and without reweighting the model wins the loss by predicting all
    zeros for those classes. Capped so the rarest classes cannot dominate.
    """
    counts = class_frequencies(records)
    total = float(len(records))
    weights = torch.ones_like(counts)
    for i, count in enumerate(counts):
        if count > 0:
            weights[i] = min(cap, max(1.0, (total - float(count)) / float(count)))
    return weights
