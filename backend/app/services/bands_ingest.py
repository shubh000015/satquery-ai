"""Group uploaded files into scenes — not one scene per file.

Optical MSI is stored as 12 Sentinel-2 band planes (B01–B12, no cirrus B10).
SAR is stored as VV + VH. Those files are bands of one scene. Treating two
TIFFs as two scenes was the old pair dialog; this module stacks them instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

S2_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)
SAR_BANDS: tuple[str, ...] = ("VV", "VH")
RGB_BANDS: tuple[str, ...] = ("B04", "B03", "B02")
BEN_S2_BANDS: tuple[str, ...] = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
)

Kind = Literal["optical", "sar", "rgb"]
Expect = Literal["optical", "sar", "fusion"] | None  # noqa: UP007

_ALIASES = {
    "B1": "B01", "B01": "B01", "BAND1": "B01", "BAND01": "B01",
    "B2": "B02", "B02": "B02", "BAND2": "B02", "BAND02": "B02",
    "B3": "B03", "B03": "B03", "BAND3": "B03", "BAND03": "B03",
    "B4": "B04", "B04": "B04", "BAND4": "B04", "BAND04": "B04",
    "B5": "B05", "B05": "B05", "BAND5": "B05", "BAND05": "B05",
    "B6": "B06", "B06": "B06", "BAND6": "B06", "BAND06": "B06",
    "B7": "B07", "B07": "B07", "BAND7": "B07", "BAND07": "B07",
    "B8": "B08", "B08": "B08", "BAND8": "B08", "BAND08": "B08",
    "B8A": "B8A", "B08A": "B8A", "B8a": "B8A",
    "B9": "B09", "B09": "B09", "BAND9": "B09", "BAND09": "B09",
    "B11": "B11", "BAND11": "B11",
    "B12": "B12", "BAND12": "B12",
    "VV": "VV", "VH": "VH",
}

_TOKEN = re.compile(
    r"(?:^|[^A-Za-z0-9])(?P<tok>B8A|B08A|B0?[1-9]|B1[12]|BAND[_-]?0?[1-9]|BAND[_-]?1[12]|VV|VH)(?:[^A-Za-z0-9]|$)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"20\d{2}")


def parse_band_token(filename: str) -> str | None:
    """Return a canonical band name from a filename, or None."""
    text = Path(filename).name
    match = _TOKEN.search(text)
    if not match:
        return None
    raw = match.group("tok").upper().replace("-", "").replace("_", "")
    if raw.startswith("BAND"):
        raw = "B" + raw[4:]
    return _ALIASES.get(raw)


def scene_stem(filename: str) -> str:
    """Filename minus extension and band token, used to group planes."""
    name = Path(filename).stem
    token = parse_band_token(filename)
    if token:
        extras = {token}
        if token.startswith("B"):
            rest = token[1:]
            extras.update({f"B{rest}", f"BAND{rest}"})
            if rest.isdigit():
                extras.update(
                    {f"B{int(rest)}", f"B{int(rest):02d}", f"BAND{int(rest)}", f"BAND{int(rest):02d}"}
                )
        for variant in sorted(extras, key=len, reverse=True):
            name = re.sub(
                rf"(^|[_\-.]){re.escape(variant)}(?=[_\-.]|$)",
                r"\1",
                name,
                flags=re.IGNORECASE,
            )
    cleaned = re.sub(r"[_\-.]+", "_", name).strip("_").lower()
    return cleaned or "scene"


def looks_sar(filename: str) -> bool:
    name = filename.lower()
    return any(hint in name for hint in ("sar", "risat", "sentinel-1", "sentinel1", "_s1_", "radar", "vv", "vh"))


@dataclass
class ScenePlan:
    key: str
    kind: Kind
    indices: list[int]
    names: list[str]
    required: tuple[str, ...]
    missing: tuple[str, ...]
    complete: bool
    stacked: bool
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.kind == "sar":
            have = "+".join(self.names) if self.names else "SAR"
            return f"SAR {have}" if self.complete else f"SAR ({len(self.names)}/2 bands)"
        if self.kind == "optical":
            return (
                f"Optical MSI ({len(self.names)}/12 bands)"
                if not self.complete
                else f"Optical MSI ({len(self.names)} bands)"
            )
        return "RGB preview"

    @property
    def modality(self) -> str:
        return "sar" if self.kind == "sar" else "optical"


@dataclass
class IngestPlan:
    scenes: list[ScenePlan]
    needs_kind: bool
    ready: bool
    notes: list[str]

    @property
    def message(self) -> str:
        parts: list[str] = []
        for scene in self.scenes:
            if scene.complete:
                continue
            missing = ", ".join(scene.missing) if scene.missing else "more band planes"
            if scene.kind == "sar":
                parts.append(f"SAR needs VV and VH. Missing: {missing}.")
            elif scene.kind == "optical":
                parts.append(f"Optical needs all 12 Sentinel-2 bands (B01–B12, no B10). Missing: {missing}.")
            else:
                parts.append("This RGB preview is not a 12-band optical stack. Add the 12 GeoTIFF band planes.")
        return " ".join(parts) or "Scene bands are complete."


def infer_cube_names(filename: str, band_count: int) -> tuple[str, ...]:
    """Names for an already-stacked raster."""
    if band_count >= 12:
        return S2_BANDS
    if band_count == 10:
        return BEN_S2_BANDS
    if band_count == 2 and looks_sar(filename):
        return SAR_BANDS
    if band_count == 4:
        return ("B04", "B03", "B02", "B08")
    if band_count == 3:
        return RGB_BANDS
    token = parse_band_token(filename)
    if band_count == 1 and token:
        return (token,)
    return tuple(f"B{i + 1:02d}" for i in range(band_count)) if band_count > 1 else ()


def _finish_group(
    key: str,
    indices: list[int],
    names: list[str],
    kind: Kind,
    *,
    expect: Expect = None,
) -> ScenePlan:
    required = SAR_BANDS if kind == "sar" else S2_BANDS if kind == "optical" else RGB_BANDS
    present = [name for name in names if name]
    unique: list[str] = []
    for name in present:
        if name not in unique:
            unique.append(name)
    ordered = [name for name in required if name in unique] + [n for n in unique if n not in required]
    missing = tuple(name for name in required if name not in unique)
    if kind == "rgb":
        complete = True
        missing = ()
    elif kind == "sar":
        complete = not missing
        # A single amplitude GeoTIFF (no VV/VH in the name) is the old demo
        # product. Still warn, but do not block unless the user asked for SAR.
        if not complete and len(indices) == 1 and not unique and expect != "sar":
            complete = True
            missing = ()
    else:
        complete = not missing
    stacked = len(indices) > 1 or (kind != "rgb" and len(unique) >= 2)
    notes: list[str] = []
    if kind == "optical" and not complete:
        notes.append(
            f"Optical scene is {len(unique)}/12 bands. Missing {', '.join(missing)}. "
            "Add the remaining Sentinel-2 GeoTIFFs (B01–B12, no B10)."
        )
    if kind == "sar" and (not complete or (len(indices) == 1 and not unique)):
        notes.append("SAR scene needs both VV and VH polarisations as two GeoTIFFs.")
    if kind == "optical" and unique and set(BEN_S2_BANDS).issubset(unique) and missing:
        notes.append("10-band BigEarthNet set is present; B01 and B09 can be filled from neighbours.")
    return ScenePlan(
        key=key,
        kind=kind,
        indices=indices,
        names=ordered,
        required=required,
        missing=missing,
        complete=complete,
        stacked=stacked,
        notes=notes,
    )


def plan_scenes(
    filenames: list[str],
    band_counts: list[int] | None = None,
    expect: Expect = None,
) -> IngestPlan:
    """Turn a file list into 1–2 scenes.

    Named band planes (B04.tif, scene_VV.tif) stack. A 12-band cube is already
    one optical scene. Two undated RGB chips stay two scenes (the old pair).
    """
    if not filenames:
        return IngestPlan(scenes=[], needs_kind=False, ready=False, notes=["No files."])

    counts = list(band_counts) if band_counts is not None else [0] * len(filenames)
    if len(counts) != len(filenames):
        raise ValueError("band_counts must match filenames")

    tokens = [parse_band_token(name) for name in filenames]
    stems = [scene_stem(name) for name in filenames]

    # One already-stacked cube (or RGB chip) per file when it is not a band plane.
    cubes: list[tuple[int, Kind, list[str]]] = []
    planes: list[int] = []
    for index, (name, token, count) in enumerate(zip(filenames, tokens, counts)):
        is_plane = token is not None or count == 1
        if is_plane and count <= 2:
            planes.append(index)
            continue
        if count >= 12 or (count == 10 and token is None):
            cubes.append((index, "optical", list(infer_cube_names(name, count or 12))))
        elif count == 2 and looks_sar(name):
            cubes.append((index, "sar", list(SAR_BANDS)))
        elif count >= 3:
            cubes.append((index, "rgb", list(infer_cube_names(name, count))))
        elif token:
            planes.append(index)
        else:
            cubes.append((index, "rgb" if count != 1 else "optical", list(infer_cube_names(name, max(count, 1)))))

    groups: dict[str, list[int]] = {}
    for index in planes:
        token = tokens[index]
        kind_hint = "sar" if token in SAR_BANDS or looks_sar(filenames[index]) else "optical"
        year = _YEAR.search(stems[index] + filenames[index])
        year_key = year.group(0) if year else ""
        key = f"{kind_hint}:{stems[index]}:{year_key}"
        groups.setdefault(key, []).append(index)

    scenes: list[ScenePlan] = []
    for key, indices in groups.items():
        kind: Kind = "sar" if key.startswith("sar:") else "optical"
        names = [tokens[i] or "" for i in indices]
        if kind == "optical" and expect == "optical" and not any(names) and len(indices) == 12:
            names = list(S2_BANDS)
        if kind == "sar" and expect == "sar" and not any(names) and len(indices) == 2:
            names = list(SAR_BANDS)
        scenes.append(_finish_group(key, indices, names, kind, expect=expect))

    for index, kind, names in cubes:
        required = SAR_BANDS if kind == "sar" else S2_BANDS if kind == "optical" else RGB_BANDS
        missing = tuple(n for n in required if n not in names) if kind != "rgb" else ()
        complete = kind == "rgb" or not missing
        notes: list[str] = []
        if kind == "rgb":
            notes.append(
                "RGB/JPEG preview only — specialists that need the 12-band MSI stack "
                "will be less accurate. Upload B01–B12 GeoTIFFs for a full optical scene."
            )
            complete = True  # allowed, with a warning
        scenes.append(
            ScenePlan(
                key=f"cube:{index}",
                kind=kind,
                indices=[index],
                names=names,
                required=required,
                missing=missing,
                complete=complete,
                stacked=False,
                notes=notes,
            )
        )

    # 12 unlabeled 1-band files with no tokens and a declared optical intent —
    # already handled per-group. If they landed as one optical group of 12
    # unnamed planes, assign S2 names.
    for scene in scenes:
        if (
            scene.kind == "optical"
            and len(scene.indices) == 12
            and not any(scene.names)
        ):
            scene.names = list(S2_BANDS)
            scene.missing = ()
            scene.complete = True
            scene.stacked = True
            scene.notes = ["Assigned B01–B12 in filename order to 12 unlabeled band planes."]
        if (
            scene.kind == "sar"
            and len(scene.indices) == 2
            and not any(scene.names)
        ):
            scene.names = list(SAR_BANDS)
            scene.missing = ()
            scene.complete = True
            scene.stacked = True

    # Ambiguous: two unlabeled single-band files and no expect.
    unlabeled_planes = [
        i for i in range(len(filenames)) if tokens[i] is None and counts[i] <= 1
    ]
    needs_kind = (
        expect is None
        and len(filenames) == 2
        and len(unlabeled_planes) == 2
        and all(not looks_sar(filenames[i]) for i in unlabeled_planes)
        and len(scenes) == 1
        and scenes[0].kind == "optical"
        and not scenes[0].complete
    )

    # Fusion expect: keep optical and SAR as two scenes (already split by kind).
    if expect == "fusion":
        optical = [s for s in scenes if s.kind != "sar"]
        sar = [s for s in scenes if s.kind == "sar"]
        if not optical:
            scenes.append(_finish_group("optical:pending", [], [], "optical", expect=expect))
        if not sar:
            scenes.append(_finish_group("sar:pending", [], [], "sar", expect=expect))

    scenes.sort(key=lambda s: (0 if s.kind == "optical" else 1, s.key))

    # Named-band optical/SAR must be complete to be ready. RGB cubes are ready
    # only when the user did not declare an optical/fusion stack, so JPEG demos
    # still run and a declared optical upload still asks for 12 bands.
    present = [scene for scene in scenes if scene.indices]
    ready = bool(present) and all(
        scene.complete
        and not (scene.kind == "rgb" and expect in ("optical", "fusion"))
        for scene in present
    )
    notes = [note for scene in scenes for note in scene.notes]
    return IngestPlan(scenes=scenes, needs_kind=needs_kind, ready=ready, notes=notes)
