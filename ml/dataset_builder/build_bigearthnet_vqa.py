"""Build a BigEarthNet S1+S2 → RSVQA-style VQA dataset (standalone).

This script does ONLY dataset formation — no model training.
Run it on a laptop (CPU is fine) after you download/extract BigEarthNet.

You need BOTH archives (problem statement covers optical AND SAR):

  - BigEarthNet-S2  (Sentinel-2 optical / multispectral)
  - BigEarthNet-S1  (Sentinel-1 SAR: VV + VH)
  - metadata.parquet (v2.0; links each S2 patch to its S1 twin via s1_name)

What it produces (Kaggle-ready folder):

    out/
      train.jsonl
      images_s2/*.png     # optical RGB from B04/B03/B02
      images_s1/*.png     # SAR false-color from VV/VH
      manifest.json

Each JSONL line is RSVQA-style with a relative image path:

    {"image": "images_s2/....png", "question": "...", "answer": "...",
     "source": "bigearthnet", "modality": "optical", "category": "presence"}

    {"image": "images_s1/....png", "question": "...", "answer": "...",
     "source": "bigearthnet", "modality": "sar", "category": "multi-label"}

Typical one-third run on a Mac (extract ~1/3 of tiles from BOTH S1 and S2):

    python3 build_bigearthnet_vqa.py \\
        --images-s2 ~/ben/BigEarthNet-S2 \\
        --images-s1 ~/ben/BigEarthNet-S1 \\
        --metadata ~/ben/metadata.parquet \\
        --out ~/ben/ben_vqa_third \\
        --fraction 0.33 --require-s1

Install (once):

    python3 -m pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

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

_PRESENCE_OPTICAL = (
    "Is there any {cls} in this image?",
    "Does this optical satellite image contain {cls}?",
    "Can you see {cls} in this Sentinel-2 scene?",
)
_PRESENCE_SAR = (
    "Is there any {cls} in this SAR image?",
    "Does this Sentinel-1 radar image contain {cls}?",
    "Looking at this SAR scene, is {cls} present?",
)
_LIST_OPTICAL = (
    "Which land-cover classes are present in this satellite image?",
    "What land cover types does this Sentinel-2 image contain?",
    "List the land-cover classes visible in this optical patch.",
)
_LIST_SAR = (
    "Which land-cover classes are present in this SAR image?",
    "What land cover types does this Sentinel-1 radar image contain?",
    "List the land-cover classes visible in this SAR patch.",
)
_MODALITY_QUESTIONS = (
    ("What sensor modality is this image from?", "optical", "optical"),
    ("What sensor modality is this image from?", "sar", "sar"),
    ("Is this an optical or a SAR image?", "optical", "optical"),
    ("Is this an optical or a SAR image?", "sar", "sar"),
)


def index_s2_dirs(images_root: Path) -> dict[str, Path]:
    """Patch folders that contain <name>_B04.tif."""
    index: dict[str, Path] = {}
    for root, _dirs, files in os.walk(images_root):
        name = Path(root).name
        if f"{name}_B04.tif" in files:
            index[name] = Path(root)
    return index


def index_s1_dirs(images_root: Path) -> dict[str, Path]:
    """Patch folders that contain <name>_VV.tif (and usually _VH.tif)."""
    index: dict[str, Path] = {}
    for root, _dirs, files in os.walk(images_root):
        name = Path(root).name
        if f"{name}_VV.tif" in files:
            index[name] = Path(root)
    return index


def load_metadata_rows(
    metadata_parquet: Path | None,
    s2_dirs: dict[str, Path],
    split: str | None,
) -> list[dict]:
    """Return rows with patch_id (S2), optional s1_name, and labels."""
    if metadata_parquet is not None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit(
                "Missing pyarrow. Run: pip install pyarrow\n"
                "(needed to read BigEarthNet v2.0 metadata.parquet)"
            ) from exc

        table = pq.read_table(metadata_parquet)
        columns = set(table.column_names)
        rows: list[dict] = []
        for row in table.to_pylist():
            if split and "split" in columns and row.get("split") != split:
                continue
            labels = list(row.get("labels") or [])
            patch_id = row.get("patch_id")
            if not patch_id or not labels:
                continue
            rows.append({
                "patch_id": str(patch_id),
                "s1_name": str(row["s1_name"]) if row.get("s1_name") else None,
                "labels": labels,
            })
        if not rows:
            raise SystemExit(
                f"No labeled rows in {metadata_parquet} for split={split!r}. "
                "Try --split train or --no-split-filter."
            )
        return rows

    # v1.0 fallback: per-patch JSON on S2 folders only (no S1 link)
    rows = []
    for patch_id, patch_dir in s2_dirs.items():
        meta = patch_dir / f"{patch_id}_labels_metadata.json"
        if not meta.exists():
            continue
        with meta.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        labels = list(data.get("labels") or [])
        if not labels:
            continue
        rows.append({
            "patch_id": patch_id,
            "s1_name": data.get("corresponding_s1_patch") or data.get("s1_name"),
            "labels": labels,
        })
    if not rows:
        raise SystemExit(
            "No labels found. For v2.0 pass --metadata metadata.parquet; "
            "for v1.0 each S2 patch folder needs *_labels_metadata.json."
        )
    return rows


def _percentile_stretch(arr):
    import numpy as np

    lo, hi = np.percentile(arr, (2, 98))
    if hi <= lo:
        return None
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def s2_to_rgb_png(patch_dir: Path, patch_id: str, dest: Path) -> bool:
    """B04/B03/B02 -> 224x224 RGB PNG."""
    import numpy as np
    from PIL import Image

    if dest.exists():
        return True

    bands = []
    for band in ("B04", "B03", "B02"):
        path = patch_dir / f"{patch_id}_{band}.tif"
        if not path.exists():
            return False
        bands.append(np.asarray(Image.open(path), dtype=np.float32))

    stack = np.stack(bands, axis=2)
    stretched = _percentile_stretch(stack)
    if stretched is None:
        return False
    image = Image.fromarray((stretched * 255).astype(np.uint8), mode="RGB")
    image = image.resize((224, 224), Image.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return True


def s1_to_rgb_png(patch_dir: Path, s1_id: str, dest: Path) -> bool:
    """VV/VH -> false-color RGB (R=VV, G=VH, B=|VV-VH|) for the VLM."""
    import numpy as np
    from PIL import Image

    if dest.exists():
        return True

    vv_path = patch_dir / f"{s1_id}_VV.tif"
    vh_path = patch_dir / f"{s1_id}_VH.tif"
    if not vv_path.exists():
        return False

    vv = np.asarray(Image.open(vv_path), dtype=np.float32)
    if vh_path.exists():
        vh = np.asarray(Image.open(vh_path), dtype=np.float32)
    else:
        vh = vv.copy()

    # Align shapes if one band was stored at a slightly different size.
    h = min(vv.shape[0], vh.shape[0])
    w = min(vv.shape[1], vh.shape[1])
    vv, vh = vv[:h, :w], vh[:h, :w]
    diff = np.abs(vv - vh)

    stack = np.stack([vv, vh, diff], axis=2)
    stretched = _percentile_stretch(stack)
    if stretched is None:
        return False
    image = Image.fromarray((stretched * 255).astype(np.uint8), mode="RGB")
    image = image.resize((224, 224), Image.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return True


def select_rows(
    rows: list[dict],
    s2_dirs: dict[str, Path],
    s1_dirs: dict[str, Path],
    fraction: float | None,
    max_patches: int | None,
    require_s1: bool,
    seed: int,
) -> list[dict]:
    available = []
    for row in rows:
        if row["patch_id"] not in s2_dirs:
            continue
        if require_s1:
            s1 = row.get("s1_name")
            if not s1 or s1 not in s1_dirs:
                continue
        available.append(row)

    if not available:
        tip = (
            "Need matching S1+S2 patch folders locally. Extract the SAME tile "
            "subset from both BigEarthNet-S1 and BigEarthNet-S2."
            if require_s1
            else "Extract more BigEarthNet-S2 tile folders, or fix --images-s2."
        )
        raise SystemExit(
            "No usable patches found after matching metadata to local folders.\n" + tip
        )

    rng = random.Random(seed)
    rng.shuffle(available)

    if fraction is not None:
        if not 0 < fraction <= 1:
            raise SystemExit("--fraction must be between 0 and 1 (e.g. 0.33)")
        available = available[: max(1, int(len(available) * fraction))]

    if max_patches is not None:
        available = available[:max_patches]

    return available


def _qa_for_modality(
    rng: random.Random,
    rel_image: str,
    labels: list[str],
    modality: str,
    used_count: int,
) -> list[dict]:
    presence = _PRESENCE_SAR if modality == "sar" else _PRESENCE_OPTICAL
    listing = _LIST_SAR if modality == "sar" else _LIST_OPTICAL

    rows = [{
        "image": rel_image,
        "question": rng.choice(listing),
        "answer": ", ".join(sorted(label.lower() for label in labels)),
        "source": "bigearthnet",
        "modality": modality,
        "category": "multi-label",
    }]

    if used_count % 2 == 0:
        cls, verdict = rng.choice(labels), "yes"
    else:
        absent = [c for c in BEN_CLASSES if c not in labels]
        if absent:
            cls, verdict = rng.choice(absent), "no"
        else:
            cls, verdict = rng.choice(labels), "yes"

    rows.append({
        "image": rel_image,
        "question": rng.choice(presence).format(cls=cls.lower()),
        "answer": verdict,
        "source": "bigearthnet",
        "modality": modality,
        "category": "presence",
    })

    # Light modality-awareness so the model learns optical vs SAR wording.
    for q, mod, ans in _MODALITY_QUESTIONS:
        if mod == modality and used_count % 5 == 0:
            rows.append({
                "image": rel_image,
                "question": q,
                "answer": ans,
                "source": "bigearthnet",
                "modality": modality,
                "category": "modality",
            })
            break

    return rows


def build_rows(
    selected: list[dict],
    s2_dirs: dict[str, Path],
    s1_dirs: dict[str, Path],
    out_s2: Path,
    out_s1: Path,
    include_s1: bool,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    stats = {
        "s2_converted": 0,
        "s1_converted": 0,
        "s2_skipped": 0,
        "s1_skipped": 0,
        "pairs_with_both": 0,
    }

    for i, row in enumerate(selected, start=1):
        if i % 200 == 0 or i == len(selected):
            print(f"  converting patches {i}/{len(selected)} ...", flush=True)

        patch_id = row["patch_id"]
        labels = row["labels"]
        s1_id = row.get("s1_name")

        s2_ok = s2_to_rgb_png(s2_dirs[patch_id], patch_id, out_s2 / f"{patch_id}.png")
        if not s2_ok:
            stats["s2_skipped"] += 1
            continue
        stats["s2_converted"] += 1
        rows.extend(_qa_for_modality(rng, f"images_s2/{patch_id}.png", labels, "optical", stats["s2_converted"]))

        if include_s1 and s1_id and s1_id in s1_dirs:
            s1_ok = s1_to_rgb_png(s1_dirs[s1_id], s1_id, out_s1 / f"{s1_id}.png")
            if s1_ok:
                stats["s1_converted"] += 1
                stats["pairs_with_both"] += 1
                rows.extend(_qa_for_modality(rng, f"images_s1/{s1_id}.png", labels, "sar", stats["s1_converted"]))
            else:
                stats["s1_skipped"] += 1

    rng.shuffle(rows)
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images-s2",
        type=Path,
        required=True,
        help="Extracted BigEarthNet-S2 root (optical)",
    )
    parser.add_argument(
        "--images-s1",
        type=Path,
        default=None,
        help="Extracted BigEarthNet-S1 root (SAR). Strongly recommended.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="v2.0 metadata.parquet (links S2 patch_id <-> s1_name)",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fraction of AVAILABLE matched patches (e.g. 0.33 = one third)",
    )
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--no-split-filter",
        action="store_true",
        help="Ignore metadata split column",
    )
    parser.add_argument(
        "--require-s1",
        action="store_true",
        help="Only keep patches that have a local S1 twin (recommended when both archives are ready)",
    )
    parser.add_argument(
        "--s2-only",
        action="store_true",
        help="Skip SAR even if --images-s1 is set (not recommended for SIH26167)",
    )
    # Back-compat alias used in earlier docs
    parser.add_argument("--images", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.images and not args.images_s2:
        args.images_s2 = args.images

    if not args.images_s2.exists():
        raise SystemExit(f"--images-s2 not found: {args.images_s2}")

    include_s1 = bool(args.images_s1) and not args.s2_only
    if include_s1 and not args.images_s1.exists():
        raise SystemExit(f"--images-s1 not found: {args.images_s1}")
    if not include_s1:
        print(
            "WARNING: building OPTICAL-ONLY dataset. For SIH26167 you should "
            "pass --images-s1 BigEarthNet-S1 as well.",
            file=sys.stderr,
        )

    print("1/4 indexing local patch folders ...")
    s2_dirs = index_s2_dirs(args.images_s2)
    print(f"   S2 (optical) patches: {len(s2_dirs)}")
    if not s2_dirs:
        raise SystemExit(f"No S2 patch folders with *_B04.tif under {args.images_s2}")

    s1_dirs: dict[str, Path] = {}
    if include_s1:
        s1_dirs = index_s1_dirs(args.images_s1)
        print(f"   S1 (SAR) patches    : {len(s1_dirs)}")
        if not s1_dirs:
            raise SystemExit(f"No S1 patch folders with *_VV.tif under {args.images_s1}")

    print("2/4 loading labels from metadata ...")
    split = None if args.no_split_filter else args.split
    meta_rows = load_metadata_rows(args.metadata, s2_dirs, split)
    print(f"   {len(meta_rows)} labeled metadata rows (split={split!r})")

    require_s1 = args.require_s1 or include_s1
    # If user passed S1 but did not set --require-s1, still prefer pairs when possible
    # but allow S2-only leftovers unless --require-s1.
    if include_s1 and not args.require_s1:
        require_s1 = False

    print("3/4 selecting subset ...")
    selected = select_rows(
        meta_rows, s2_dirs, s1_dirs, args.fraction, args.max_patches, require_s1, args.seed
    )
    print(f"   keeping {len(selected)} S2 patches for conversion")

    out = args.out
    out_s2 = out / "images_s2"
    out_s1 = out / "images_s1"
    out_s2.mkdir(parents=True, exist_ok=True)
    if include_s1:
        out_s1.mkdir(parents=True, exist_ok=True)

    print("4/4 converting bands -> PNG + writing Q&A pairs ...")
    rows, stats = build_rows(selected, s2_dirs, s1_dirs, out_s2, out_s1, include_s1, args.seed)
    if not rows:
        raise SystemExit("No samples produced (all selected patches failed conversion).")

    jsonl_path = out / "train.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "s2_patches_found": len(s2_dirs),
        "s1_patches_found": len(s1_dirs),
        "metadata_labeled_rows": len(meta_rows),
        "patches_selected": len(selected),
        **stats,
        "instruction_pairs": len(rows),
        "fraction": args.fraction,
        "max_patches": args.max_patches,
        "split": split,
        "seed": args.seed,
        "include_s1": include_s1,
        "require_s1": require_s1,
        "train_jsonl": "train.jsonl",
        "images_s2": "images_s2",
        "images_s1": "images_s1" if include_s1 else None,
        "note": (
            "Image paths inside train.jsonl are relative to this folder "
            "(forward slashes). Upload the folder to Kaggle as-is with the "
            "Kaggle CLI, or zip it only if you use the website file picker."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Lets `kaggle datasets create -p <out>` work without a manual zip.
    metadata_path = out / "dataset-metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(
                {
                    "title": "BigEarthNet VQA",
                    "id": "YOUR_KAGGLE_USERNAME/bigearthnet-vqa",
                    "licenses": [{"name": "CC-BY-SA-4.0"}],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print()
    print("DONE")
    print(f"  S2 converted : {stats['s2_converted']}  (skipped {stats['s2_skipped']})")
    print(f"  S1 converted : {stats['s1_converted']}  (skipped {stats['s1_skipped']})")
    print(f"  S1+S2 pairs  : {stats['pairs_with_both']}")
    print(f"  Q&A pairs    : {len(rows)}")
    print(f"  output       : {out.resolve()}")
    print()
    print("Upload this folder as-is (Mac). No local zip required:")
    print(f"  cd {out.resolve()}")
    print("  # edit dataset-metadata.json: replace YOUR_KAGGLE_USERNAME")
    print("  kaggle datasets create -p . --dir-mode zip")
    print("  # --dir-mode zip only packs the upload; your disk stays a normal folder")


if __name__ == "__main__":
    main()
