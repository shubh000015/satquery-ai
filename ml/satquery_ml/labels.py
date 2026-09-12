"""BigEarthNet-19 label vocabulary and the parser for our dataset's answer strings.

The 19-class nomenclature is the standard BigEarthNet v2.0 label set, which is
what `dataset_builder/build_bigearthnet_vqa.py` wrote into `train.jsonl`.
"""

from __future__ import annotations

CLASSES: tuple[str, ...] = (
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

NUM_CLASSES = len(CLASSES)

CLASS_TO_INDEX: dict[str, int] = {name.lower(): i for i, name in enumerate(CLASSES)}

# Short, speakable names. The classifier's job is to produce these; the language
# model's job is to turn them into a sentence. They have to read as nouns, since
# they get dropped straight into "the classifier found no ___".
SHORT_NAMES: dict[str, str] = {
    "Urban fabric": "urban fabric",
    "Industrial or commercial units": "industrial units",
    "Arable land": "cropland",
    "Permanent crops": "orchards",
    "Pastures": "pasture",
    "Complex cultivation patterns": "mixed farmland",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "farmland with natural vegetation",
    "Agro-forestry areas": "agroforestry",
    "Broad-leaved forest": "broadleaf forest",
    "Coniferous forest": "conifer forest",
    "Mixed forest": "mixed forest",
    "Natural grassland and sparsely vegetated areas": "grassland",
    "Moors, heathland and sclerophyllous vegetation": "heathland",
    "Transitional woodland, shrub": "shrubland",
    "Beaches, dunes, sands": "beach and dunes",
    "Inland wetlands": "wetland",
    "Coastal wetlands": "coastal wetland",
    "Inland waters": "inland water",
    "Marine waters": "sea",
}

# Coarse groups matching the vocabulary the SatQuery backend and UI already use
# (water / vegetation / builtup), so a fine-grained prediction can still answer
# "is there water in this scene?".
GROUPS: dict[str, tuple[str, ...]] = {
    "water": ("Inland waters", "Marine waters", "Inland wetlands", "Coastal wetlands"),
    "vegetation": (
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Agro-forestry areas",
    ),
    "builtup": ("Urban fabric", "Industrial or commercial units"),
    "agriculture": (
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
    ),
    "bare": ("Beaches, dunes, sands",),
}

GROUP_OF: dict[str, str] = {
    cls: group for group, members in GROUPS.items() for cls in members
}

# Longest first so the greedy parser below prefers the most specific match.
_SORTED_LOWER: tuple[str, ...] = tuple(
    sorted((name.lower() for name in CLASSES), key=len, reverse=True)
)


def short_name(class_name: str) -> str:
    return SHORT_NAMES.get(class_name, class_name.lower())


def parse_label_string(answer: str) -> list[str]:
    """Recover the class list from a ', '-joined answer string.

    Four class names contain commas ("Transitional woodland, shrub",
    "Beaches, dunes, sands", "Moors, heathland ...", "Land principally
    occupied by agriculture, ..."), so splitting on ', ' silently shreds them
    into fragments that match nothing. Instead, walk the string and consume the
    longest known class name at each position.
    """
    text = answer.strip().lower()
    if not text:
        return []

    found: list[str] = []
    pos = 0
    length = len(text)
    while pos < length:
        for candidate in _SORTED_LOWER:
            if text.startswith(candidate, pos):
                index = CLASS_TO_INDEX[candidate]
                if index not in [CLASS_TO_INDEX[f.lower()] for f in found]:
                    found.append(CLASSES[index])
                pos += len(candidate)
                break
        else:
            # Not the start of a class name: skip the separator/stray character.
            pos += 1
            continue
        # Consume the ', ' separator if it follows.
        if text.startswith(", ", pos):
            pos += 2

    return found


def labels_to_vector(class_names: list[str]) -> list[float]:
    vector = [0.0] * NUM_CLASSES
    for name in class_names:
        index = CLASS_TO_INDEX.get(name.lower())
        if index is not None:
            vector[index] = 1.0
    return vector


def match_class_in_question(question: str) -> str | None:
    """Find which land-cover class (or coarse group) a question is asking about.

    Returns a class name, a group key ('water'/'vegetation'/'builtup'/...), or
    None when the question is not about a specific target.
    """
    lowered = question.lower()

    for candidate in _SORTED_LOWER:
        if candidate in lowered:
            return CLASSES[CLASS_TO_INDEX[candidate]]

    # Coarse group words come first: a bare "urban" or "water" should resolve to
    # the group, so an industrial-only patch still answers "yes" to "any urban
    # areas?". The exact class name is caught by the loop above.
    group_aliases: list[tuple[str, str]] = [
        ("water", "water"),
        ("vegetation", "vegetation"),
        ("forest", "vegetation"),
        ("built-up", "builtup"),
        ("built up", "builtup"),
        ("builtup", "builtup"),
        ("urban", "builtup"),
        ("building", "builtup"),
        ("city", "builtup"),
        ("agriculture", "agriculture"),
        ("agricultural", "agriculture"),
        ("farm", "agriculture"),
        ("crop", "agriculture"),
    ]
    short_aliases = [(short, class_name) for class_name, short in SHORT_NAMES.items()]

    # First occurrence wins, so a group alias beats an identical short name.
    deduped: dict[str, str] = {}
    for alias, target in group_aliases + short_aliases:
        deduped.setdefault(alias, target)

    # Longest alias first, so "conifer forest" beats the bare "forest".
    for alias, target in sorted(deduped.items(), key=lambda pair: len(pair[0]), reverse=True):
        if alias in lowered:
            return target

    return None
