"""Smoke-test the SAR pseudo-RGB pipeline without booting the LLM.

Generates a fake BigEarthNet-S1 patch (`_VV.tif` + `_VH.tif`) with plausible
SAR statistics — a bright roughness patch (rough water/urban), a dark smooth
patch (calm water), and a mid-brightness natural background — then runs
`ml/serve_utils.load_image()` on the folder and dumps the rendered chip.

If the output PNG looks like a proper RGB false-colour composite (red where VV
is loud, green where VH is loud, blue where they diverge), we know the
GeoTIFF path is byte-identical to what the training-time dataset builder did.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # so we can import serve_utils

from serve_utils import load_image  # noqa: E402

# Working area lives inside the (gitignored) ml/data/ tree so the generated
# int16 TIFFs and PNG never end up in commits.
WORK_ROOT = HERE.parent / "data" / "smoke"


def make_fake_s1_patch(folder: Path, patch_id: str = "S1B_FAKE_120x120") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)

    # 120x120 int16 background of speckle noise (~ -12 dB after stretch).
    vv = rng.normal(loc=1200, scale=180, size=(120, 120)).astype(np.int16)
    vh = rng.normal(loc=600, scale=120, size=(120, 120)).astype(np.int16)

    # Bright "urban" square (top-left) — high VV, moderate VH.
    vv[10:45, 10:45] = rng.normal(3200, 250, (35, 35)).astype(np.int16)
    vh[10:45, 10:45] = rng.normal(1400, 180, (35, 35)).astype(np.int16)

    # Dark "water" disk (bottom-right) — both bands very quiet.
    yy, xx = np.ogrid[:120, :120]
    disk = (yy - 85) ** 2 + (xx - 85) ** 2 < 25 ** 2
    vv[disk] = rng.normal(200, 40, size=disk.sum()).astype(np.int16)
    vh[disk] = rng.normal(80, 25, size=disk.sum()).astype(np.int16)

    Image.fromarray(vv, mode="I;16").save(folder / f"{patch_id}_VV.tif")
    Image.fromarray(vh, mode="I;16").save(folder / f"{patch_id}_VH.tif")
    print(f"[fake] wrote {folder}/{patch_id}_VV.tif + _VH.tif  (120x120 int16)")


def main() -> None:
    patch_dir = WORK_ROOT / "s1_patch"
    make_fake_s1_patch(patch_dir)

    img, label = load_image(patch_dir)
    out_png = WORK_ROOT / "s1_rendered.png"
    img.save(out_png)

    arr = np.asarray(img)
    print(f"[ok]   rendered chip label : {label}")
    print(f"[ok]   output image        : {out_png}  ({img.size[0]}x{img.size[1]}, mode={img.mode})")
    print(f"[ok]   channel means (R,G,B): {arr.reshape(-1, 3).mean(axis=0).round(1)}")
    print(f"[ok]   channel stds  (R,G,B): {arr.reshape(-1, 3).std(axis=0).round(1)}")


if __name__ == "__main__":
    main()
