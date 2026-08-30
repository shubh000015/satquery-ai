"""Suggested queries offered after upload, drawn from the problem statement's
representative queries and narrowed to what the supplied inputs can answer."""

from __future__ import annotations

from app.schemas.imagery import Asset, InputMode

_SINGLE = [
    "Describe the land-cover and major objects visible in this image.",
    "Highlight the water body referred to in the query.",
    "What is the dominant land-cover class in this scene?",
]

_BI_TEMPORAL = [
    "What changed between these two dates, and where did the change occur?",
    "Has the built-up area increased, decreased, or remained unchanged?",
    "Describe the urban expansion between the two dates.",
]

_CROSS_MODAL = [
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Which settlements are affected by water-covered areas?",
    "Does SAR reveal water that the optical scene misses?",
]


def suggest(mode: InputMode, assets: list[Asset]) -> list[str]:
    if mode == "bi-temporal":
        base = list(_BI_TEMPORAL)
    elif mode == "cross-modal":
        base = list(_CROSS_MODAL)
    else:
        base = list(_SINGLE)

    georeferenced = any(a.meta and a.meta.georeferenced and a.meta.gsd_meters for a in assets)
    if georeferenced:
        base.append("Measure the area of the largest water body.")
    if any(a.modality == "sar" for a in assets) and mode == "single":
        base[0] = "Describe the structures and surfaces visible in this SAR scene."
    return base[:4]
