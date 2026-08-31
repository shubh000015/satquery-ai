"""Run the whole pipeline on synthetic scenes and print the result.

Useful for demos and for sanity-checking answer text without a browser:

    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --report        # markdown report instead
    python scripts/demo_pipeline.py --write samples # just write sample GeoTIFFs
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Answers contain °, ², → and friends; Windows consoles default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agent.controller import AgentController  # noqa: E402
from app.services.report import result_to_markdown  # noqa: E402
from app.services.storage import get_asset_store  # noqa: E402

DEG_PER_METER = 1.0 / 111_320.0


def _write(path: Path, array: np.ndarray, gsd: float = 10.0) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    data = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    scale = gsd * DEG_PER_METER
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=data.shape[2],
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(86.9, 26.1, scale, scale),
    ) as dst:
        for band in range(data.shape[2]):
            dst.write(data[:, :, band], band + 1)
    return path


def _optical(size: int = 256, flood: bool = False, new_builtup: bool = False) -> np.ndarray:
    rng = np.random.default_rng(3)
    red = np.full((size, size), 0.55, np.float32)
    green = np.full((size, size), 0.5, np.float32)
    blue = np.full((size, size), 0.42, np.float32)
    nir = np.full((size, size), 0.7, np.float32)

    red[:, : size // 3] = 0.2
    nir[:, : size // 3] = 0.88  # vegetation

    channel = slice(size // 2 - 14, size // 2 + 14)
    red[:, channel], green[:, channel], blue[:, channel], nir[:, channel] = 0.1, 0.16, 0.3, 0.05

    if flood:
        red[140:210, 150:240], green[140:210, 150:240] = 0.12, 0.18
        blue[140:210, 150:240], nir[140:210, 150:240] = 0.32, 0.07

    for x0 in (200, 230):
        red[30:60, x0 : x0 + 20] = 0.92
        green[30:60, x0 : x0 + 20] = 0.9
        blue[30:60, x0 : x0 + 20] = 0.88
        nir[30:60, x0 : x0 + 20] = 0.8

    if new_builtup:  # a second date with a new colony on former farmland
        red[70:130, 20:80] = 0.93
        green[70:130, 20:80] = 0.91
        blue[70:130, 20:80] = 0.89
        nir[70:130, 20:80] = 0.82

    stack = np.stack([red, green, blue, nir], axis=2)
    return np.clip(stack + rng.normal(0, 0.01, stack.shape).astype(np.float32), 0, 1)


def _sar(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(5)
    amp = np.full((size, size), 0.6, np.float32)
    amp[:, size // 2 - 14 : size // 2 + 14] = 0.05
    amp[140:210, 150:240] = 0.06  # water the optical scene cannot see through cloud
    speckle = rng.gamma(4.0, 0.25, amp.shape).astype(np.float32)
    return np.clip(amp * speckle, 0, 1)[:, :, None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="print the markdown report")
    parser.add_argument(
        "--write",
        metavar="DIR",
        help="write sample GeoTIFFs there (for uploading through the UI) and exit",
    )
    args = parser.parse_args()

    if args.write:
        out = Path(args.write)
        out.mkdir(parents=True, exist_ok=True)
        written = [
            _write(out / "kosi_optical_20240715.tif", _optical(flood=True)),
            _write(out / "kosi_risat_sar_20240715.tif", _sar()),
            _write(out / "gurugram_optical_20200115.tif", _optical(), gsd=5.0),
            _write(out / "gurugram_optical_20240415.tif", _optical(new_builtup=True), gsd=5.0),
        ]
        for path in written:
            print(path)
        return 0

    store = get_asset_store()
    controller = AgentController()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        optical = _write(tmp_path / "kosi_optical_20240715.tif", _optical(flood=True))
        sar = _write(tmp_path / "kosi_risat_sar_20240715.tif", _sar())
        t0 = _write(tmp_path / "gurugram_optical_20200115.tif", _optical(), gsd=5.0)
        t1 = _write(tmp_path / "gurugram_optical_20240415.tif", _optical(new_builtup=True), gsd=5.0)

        optical_asset = store.save(optical.name, optical.read_bytes(), role="primary")
        sar_asset = store.save(sar.name, sar.read_bytes(), role="secondary")
        t0_asset = store.save(t0.name, t0.read_bytes(), role="primary")
        t1_asset = store.save(t1.name, t1.read_bytes(), role="secondary")

    runs = [
        ("Describe the land-cover and major objects visible in this image.", [optical_asset.id]),
        ("Highlight the water body referred to in the query.", [optical_asset.id]),
        ("Use the optical and SAR images together to identify built-up and water-covered regions.", [optical_asset.id, sar_asset.id]),
        ("Has the built-up area increased, decreased, or remained unchanged?", [t0_asset.id, t1_asset.id]),
        ("What changed between these two dates, and where did the change occur?", [t0_asset.id, t1_asset.id]),
    ]

    for query, asset_ids in runs:
        result = controller.run(query, asset_ids)
        if args.report:
            print(result_to_markdown(result, heading_level=1))
            print("\n---\n")
            continue

        print("=" * 100)
        print(f"Q: {query}")
        print(f"   task={result.task} mode={result.mode} confidence={result.confidence} in {result.elapsed_ms} ms")
        print(f"A: {result.answer}")
        for observation in result.observations:
            print(f"   · {observation}")
        print("   metrics: " + " | ".join(f"{m.label}={m.value}" for m in result.metrics))
        print(f"   evidence: {len(result.boxes)} boxes, {len(result.masks)} masks -> {[m.id for m in result.masks]}")
        for warning in result.warnings:
            print(f"   ! {warning}")
        print("   trace:")
        for step in result.trace:
            print(f"     [{step.status}] {step.label}: {step.detail} ({step.duration_ms} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
