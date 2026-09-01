"""Build train/val JSONL for fine-tuning from remote-sensing datasets.

Output format (one JSON object per line):

    {"image": "abs/path/to/tile.tif", "question": "...", "answer": "...", "source": "rsvqa-lr"}

Supported sources:

  BigEarthNet (the SIH26167 TRAINING dataset) — each Sentinel-2 patch carries
  multi-label land-cover classes; we turn them into VQA-style instruction
  pairs (this is exactly how the RSVQAxBEN benchmark was constructed).
  Works with v2.0 (metadata.parquet from bigearth.net) and v1.0
  (per-patch *_labels_metadata.json):

      python prepare_data.py bigearthnet \
          --images BigEarthNet-S2 \
          --metadata metadata.parquet \
          --out data/train.jsonl --max-patches 6000

  RSVQA-LR (EVALUATION ONLY per the problem statement) — download from
  https://rsvqa.sylvainlobry.com/ (Images_LR.zip + questions/answers JSON):

      python prepare_data.py rsvqa \
          --images Images_LR \
          --questions LR_split_test_questions.json \
          --answers  LR_split_test_answers.json \
          --out data/test.jsonl

  RSICD captions (optional extra data; Kaggle mirror
  kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset):

      python prepare_data.py rsicd \
          --images RSICD_images \
          --annotations dataset_rsicd.json \
          --out data/train_captions.jsonl

  Smoke test (no download; synthetic tiles, verifies the whole training loop):

      python prepare_data.py smoke --out data/smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _write_jsonl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples -> {out}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- RSVQA-LR

def _find_image(images_dir: Path, img_id: int) -> Path | None:
    for ext in (".tif", ".tiff", ".png", ".jpg"):
        candidate = images_dir / f"{img_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_rsvqa(images_dir: Path, questions_path: Path, answers_path: Path,
                max_samples: int | None, seed: int) -> list[dict]:
    """RSVQA official schema: questions.json -> {"questions": [{id, img_id,
    type, question, answers_ids, active}]}, answers.json -> {"answers":
    [{id, question_id, answer, active}]}."""
    questions = _load_json(questions_path).get("questions", [])
    answers = _load_json(answers_path).get("answers", [])
    answer_by_question = {
        a["question_id"]: str(a["answer"]).strip()
        for a in answers
        if a.get("active", True) and a.get("answer") is not None
    }

    rows: list[dict] = []
    missing_images = 0
    for q in questions:
        if not q.get("active", True):
            continue
        answer = answer_by_question.get(q["id"])
        if not answer:
            continue
        image = _find_image(images_dir, q["img_id"])
        if image is None:
            missing_images += 1
            continue
        rows.append({
            "image": str(image.resolve()),
            "question": str(q["question"]).strip(),
            "answer": answer,
            "source": "rsvqa-lr",
            "category": q.get("type", "unknown"),
        })

    if missing_images:
        print(f"warning: {missing_images} questions skipped (image file not found)", file=sys.stderr)
    if not rows:
        raise SystemExit(
            "No samples produced. Check --images points at the unzipped tile folder "
            "and the questions/answers JSONs are the matching split."
        )

    random.Random(seed).shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    return rows


# ------------------------------------------------------------------ RSICD

_CAPTION_PROMPTS = (
    "Describe this remote sensing image in detail.",
    "What does this satellite image show?",
    "Give a caption for this aerial scene.",
)


def build_rsicd(images_dir: Path, annotations_path: Path,
                max_samples: int | None, seed: int) -> list[dict]:
    """RSICD schema: dataset_rsicd.json -> {"images": [{filename, split,
    sentences: [{raw}]}]}. One (image, first sentence) pair per image keeps
    the caption data from swamping the VQA data."""
    rng = random.Random(seed)
    data = _load_json(annotations_path)
    rows: list[dict] = []
    for item in data.get("images", []):
        image = images_dir / item["filename"]
        if not image.exists():
            continue
        sentences = [s["raw"].strip() for s in item.get("sentences", []) if s.get("raw", "").strip()]
        if not sentences:
            continue
        rows.append({
            "image": str(image.resolve()),
            "question": rng.choice(_CAPTION_PROMPTS),
            "answer": sentences[0].rstrip(" ."). capitalize() + ".",
            "source": "rsicd",
            "category": "caption",
        })

    if not rows:
        raise SystemExit("No RSICD samples produced. Check --images and --annotations paths.")
    rng.shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    return rows


# ------------------------------------------------------------- BigEarthNet

# CLC2018 19-class nomenclature used by BigEarthNet (v2.0 and the 19-class
# variant of v1.0).
BEN_CLASSES = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)

_PRESENCE_QUESTIONS = (
    "Is there any {cls} in this image?",
    "Does this image contain {cls}?",
    "Can you see {cls} in this satellite image?",
)
_LIST_QUESTIONS = (
    "Which land-cover classes are present in this satellite image?",
    "What land cover types does this image contain?",
    "List the land-cover classes visible in this Sentinel-2 patch.",
)


def _index_patch_dirs(images_root: Path) -> dict[str, Path]:
    """One walk of the extracted archive: a patch dir is any folder that
    contains its own <dirname>_B04.tif. Handles both the v2.0 nested layout
    (BigEarthNet-S2/<tile>/<patch>/) and v1.0 flat patch folders."""
    import os

    index: dict[str, Path] = {}
    for root, _dirs, files in os.walk(images_root):
        name = Path(root).name
        if f"{name}_B04.tif" in files:
            index[name] = Path(root)
    return index


def _patch_rgb_png(patch_dir: Path, patch_id: str, out_dir: Path) -> Path | None:
    """Compose an 8-bit RGB PNG from the 10 m bands (B04/B03/B02, uint16
    reflectance) with a 2-98 percentile stretch. VLMs eat RGB, not raw DNs."""
    import numpy as np
    from PIL import Image

    dest = out_dir / f"{patch_id}.png"
    if dest.exists():
        return dest

    bands = []
    for band in ("B04", "B03", "B02"):
        path = patch_dir / f"{patch_id}_{band}.tif"
        if not path.exists():
            return None
        bands.append(np.asarray(Image.open(path), dtype=np.float32))
    stack = np.stack(bands, axis=2)

    lo, hi = np.percentile(stack, (2, 98))
    if hi <= lo:
        return None
    stretched = np.clip((stack - lo) / (hi - lo), 0, 1)
    image = Image.fromarray((stretched * 255).astype(np.uint8), mode="RGB")
    image = image.resize((224, 224), Image.BICUBIC)
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return dest


def _ben_labels(metadata_parquet: Path | None, patch_dirs: dict[str, Path],
                split: str | None) -> list[tuple[str, list[str]]]:
    """(patch_id, labels) pairs from metadata.parquet (v2.0) or the per-patch
    *_labels_metadata.json files (v1.0)."""
    if metadata_parquet is not None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("pip install pyarrow to read metadata.parquet") from exc
        table = pq.read_table(metadata_parquet)
        columns = set(table.column_names)
        rows = table.to_pylist()
        pairs = []
        for row in rows:
            if split and "split" in columns and row.get("split") != split:
                continue
            labels = list(row.get("labels") or [])
            if labels:
                pairs.append((row["patch_id"], labels))
        return pairs

    pairs = []
    for patch_id, patch_dir in patch_dirs.items():
        meta = patch_dir / f"{patch_id}_labels_metadata.json"
        if not meta.exists():
            continue
        labels = _load_json(meta).get("labels", [])
        if labels:
            pairs.append((patch_id, list(labels)))
    if not pairs:
        raise SystemExit(
            "No labels found. For BigEarthNet v2.0 pass --metadata metadata.parquet; "
            "for v1.0 the patch folders must contain *_labels_metadata.json."
        )
    return pairs


def build_bigearthnet(images_root: Path, metadata_parquet: Path | None, out: Path,
                      split: str | None, max_patches: int | None, seed: int) -> list[dict]:
    """Instruction pairs from BigEarthNet multi-labels: one 'list the classes'
    question plus one balanced yes/no presence question per patch."""
    rng = random.Random(seed)
    print("indexing patch folders (one-time walk)...")
    patch_dirs = _index_patch_dirs(images_root)
    if not patch_dirs:
        raise SystemExit(f"No patch folders with *_B04.tif found under {images_root}.")
    print(f"found {len(patch_dirs)} patch folders")

    pairs = _ben_labels(metadata_parquet, patch_dirs, split)
    rng.shuffle(pairs)

    png_dir = out.parent / "bigearthnet_rgb"
    rows: list[dict] = []
    used = skipped = 0
    for patch_id, labels in pairs:
        if max_patches and used >= max_patches:
            break
        patch_dir = patch_dirs.get(patch_id)
        if patch_dir is None:
            skipped += 1  # metadata covers the full archive; we may have a subset extracted
            continue
        png = _patch_rgb_png(patch_dir, patch_id, png_dir)
        if png is None:
            skipped += 1
            continue
        used += 1
        image = str(png.resolve())

        answer = ", ".join(sorted(label.lower() for label in labels))
        rows.append({
            "image": image,
            "question": rng.choice(_LIST_QUESTIONS),
            "answer": answer,
            "source": "bigearthnet",
            "category": "multi-label",
        })

        # Alternate yes/no so the model cannot cheat by always saying yes.
        if used % 2 == 0:
            cls, verdict = rng.choice(labels), "yes"
        else:
            absent = [c for c in BEN_CLASSES if c not in labels]
            cls, verdict = rng.choice(absent or list(labels)), "no" if absent else "yes"
        rows.append({
            "image": image,
            "question": rng.choice(_PRESENCE_QUESTIONS).format(cls=cls.lower()),
            "answer": verdict,
            "source": "bigearthnet",
            "category": "presence",
        })

    if not rows:
        raise SystemExit("No samples produced — no metadata rows matched the extracted patch folders.")
    if skipped:
        print(f"note: {skipped} metadata rows skipped (patch not extracted locally or unreadable)")
    print(f"{used} patches -> {len(rows)} instruction pairs")
    rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------ smoke

def build_smoke(out_dir: Path, n: int, seed: int) -> list[dict]:
    """Synthetic coloured tiles with trivially checkable answers. Exists so the
    whole train/eval/serve loop can be validated in minutes before spending
    GPU hours on the real datasets."""
    import numpy as np
    from PIL import Image

    rng = random.Random(seed)
    tiles_dir = out_dir / "smoke_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    classes = {
        "water": (30, 80, 200),
        "vegetation": (40, 160, 60),
        "built-up area": (170, 165, 155),
        "bare soil": (150, 110, 70),
    }
    for i in range(n):
        label, colour = rng.choice(list(classes.items()))
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :] = [max(0, c - 25) for c in colour]
        # dominant-class patch with a bit of noise so it is not a flat fill
        arr[30:200, 30:200] = colour
        noise = np.random.default_rng(seed + i).integers(-18, 18, arr.shape)
        arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
        path = tiles_dir / f"tile_{i:04d}.png"
        Image.fromarray(arr).save(path)
        rows.append({
            "image": str(path.resolve()),
            "question": "What is the dominant land cover in this image?",
            "answer": label,
            "source": "smoke",
            "category": "smoke",
        })
    return rows


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ben = sub.add_parser("bigearthnet", help="BigEarthNet multi-labels -> instruction pairs (training)")
    p_ben.add_argument("--images", type=Path, required=True, help="extracted BigEarthNet-S2 root (a subset is fine)")
    p_ben.add_argument("--metadata", type=Path, default=None, help="v2.0 metadata.parquet (omit for v1.0 JSON layout)")
    p_ben.add_argument("--split", default="train", help="v2.0 split filter: train/validation/test (default train)")
    p_ben.add_argument("--out", type=Path, required=True)
    p_ben.add_argument("--max-patches", type=int, default=6000,
                       help="patch cap; each patch yields 2 instruction pairs (default 6000)")
    p_ben.add_argument("--seed", type=int, default=13)

    p_rsvqa = sub.add_parser("rsvqa", help="RSVQA-LR/HR question-answer pairs (evaluation)")
    p_rsvqa.add_argument("--images", type=Path, required=True, help="unzipped tile folder (Images_LR)")
    p_rsvqa.add_argument("--questions", type=Path, required=True)
    p_rsvqa.add_argument("--answers", type=Path, required=True)
    p_rsvqa.add_argument("--out", type=Path, required=True)
    p_rsvqa.add_argument("--max-samples", type=int, default=None,
                         help="cap the split (e.g. 12000 for a first Kaggle run)")
    p_rsvqa.add_argument("--seed", type=int, default=13)

    p_rsicd = sub.add_parser("rsicd", help="RSICD image captions")
    p_rsicd.add_argument("--images", type=Path, required=True)
    p_rsicd.add_argument("--annotations", type=Path, required=True, help="dataset_rsicd.json")
    p_rsicd.add_argument("--out", type=Path, required=True)
    p_rsicd.add_argument("--max-samples", type=int, default=None)
    p_rsicd.add_argument("--seed", type=int, default=13)

    p_smoke = sub.add_parser("smoke", help="synthetic sanity-check dataset")
    p_smoke.add_argument("--out", type=Path, required=True)
    p_smoke.add_argument("--n", type=int, default=96)
    p_smoke.add_argument("--seed", type=int, default=13)

    args = parser.parse_args()

    if args.cmd == "bigearthnet":
        rows = build_bigearthnet(args.images, args.metadata, args.out, args.split, args.max_patches, args.seed)
    elif args.cmd == "rsvqa":
        rows = build_rsvqa(args.images, args.questions, args.answers, args.max_samples, args.seed)
    elif args.cmd == "rsicd":
        rows = build_rsicd(args.images, args.annotations, args.max_samples, args.seed)
    else:
        rows = build_smoke(args.out.parent, args.n, args.seed)

    _write_jsonl(rows, args.out)


if __name__ == "__main__":
    main()
