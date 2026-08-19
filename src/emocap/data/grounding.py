"""Detecting captions that contradict their own photograph.

Every defect rate quoted in `docs/lab-notebook.md` and in the prompt-v5 commit comes
from this module. It existed only as ad-hoc regexes while the prompt was being iterated,
which meant the headline numbers were not reproducible from the repository -- so it lives
here now.

**What it is for.** These are the five register shortcuts the prompt forbids: reaching a
target mood by altering the scene rather than the prose. Three are checked against an
independent description of the image, two against the source caption alone.

**The independent description.** `archive/v1-pilot/data/v1_moondream_factual_captions.csv.gz`
covers all 8,091 images (zero missing, zero empty, mean 42 words) and was verified by eye
to be reliable on *subject count, identity, primary action and setting* and unreliable on
*accessories, small-object colour and inferred activity* -- it put a "red collar" and a
"black harness" on a collarless dog. Only the reliable band is used, which is exactly the
band the failures fall in. It is deliberately **not** used as prompt input: that would put
a VLM's output into the dataset, which `neutral_source: human_annotations  # NOT a VLM`
exists to prevent.

**Precision, honestly.** These are lexical heuristics, so they flag for review rather
than judge. An earlier version of the solitude rule matched "several bowls" as evidence
of several *people*; the person/animal scoping below fixes that specific case, but the
rates should be read as a consistent comparative measure across prompt versions, not as
a verified hallucination count.
"""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path
from typing import Iterable, Mapping

__all__ = [
    "DEFECT_CLASSES",
    "load_factual_captions",
    "caption_defects",
    "defect_counts",
]

#: The five classes, matching THE FIVE FORBIDDEN SHORTCUTS in prompt.py.
DEFECT_CLASSES = ("solitude", "pace", "contact", "light", "posture")

# ── shortcut 1: company or solitude ─────────────────────────────────────────
_SOLO = re.compile(r"\b(alone|single|lone|solitary|by (?:him|her)self)\b", re.I)
#: Scoped to people and animals. A bare plural-noun match also counts "several bowls" as
#: multiple subjects, which produced false positives on a girl painting alone.
_MULTI = re.compile(
    r"\b(two|three|four|five|several|many|group|crowd|couple|pair|both)\s+(?:\w+\s+){0,2}"
    r"(people|persons|men|women|girls|boys|children|kids|players|dogs|riders|climbers|"
    r"workers|friends)\b"
    r"|\b(people|men|women|girls|boys|children|kids|players|dogs|others|workers|climbers)\b",
    re.I,
)

# ── shortcut 2: pace and manner of motion ───────────────────────────────────
_SLOW = re.compile(
    r"\b(slow|slowly|still|motionless|calm(?:ly)?|gracefully|gently|softly|paused?|"
    r"resting|lingers?|walks?)\b",
    re.I,
)
_FAST = re.compile(
    r"\b(run(?:s|ning)?|sprint\w*|jump(?:s|ing)?|leap\w*|chas\w*|dash\w*|gallop\w*|"
    r"racing|mid-?air)\b",
    re.I,
)

# ── shortcut 3: contact and relationship ────────────────────────────────────
_CONTACT = re.compile(
    r"\b(a couple|lovers?|lean(?:s|ing)? into|embrac\w+|holding each other|arm in arm)\b",
    re.I,
)

# ── shortcut 4: light, weather, time of day ─────────────────────────────────
_LIGHT = re.compile(
    r"\b(light fades|sunlight|sunlit|sunset|sunrise|dusk|dawn|twilight|golden hour|"
    r"glow(?:s|ing)?|shadows?|breeze|windy|gust|overcast|gloom|dim(?:ly)?|fading light|"
    r"warm light|pale light)\b",
    re.I,
)

# ── shortcut 5: posture, gaze and grip ──────────────────────────────────────
_POSTURE = re.compile(
    r"\b(heads? bowed|shoulders? (?:slumped|hunched|drawn)|rests? (?:her|his|its) head|"
    r"jaws? set|hands? gripped|eyes locked|neck extended|gaze (?:drops|falls)|slumped|"
    r"hunched)\b",
    re.I,
)

DEFAULT_FACTUAL_PATH = "archive/v1-pilot/data/v1_moondream_factual_captions.csv.gz"


def load_factual_captions(path: str | Path = DEFAULT_FACTUAL_PATH) -> dict[str, str]:
    """image_id -> the archived independent scene description."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. It is the v1 pilot artifact used as an independent "
            f"description of each image; without it only the caption-only defect "
            f"classes (contact, light, posture) can be checked."
        )
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return {row["image_id"]: row["factual_caption"] for row in csv.DictReader(f)}


def caption_defects(
    caption: str, *, source_caption: str, factual_caption: str | None = None
) -> list[str]:
    """Which forbidden shortcuts this one caption appears to take.

    ``solitude`` and ``pace`` need ``factual_caption`` and are skipped without it.
    ``contact``, ``light`` and ``posture`` are judged against the source caption: adding
    any of them is forbidden whether or not the photograph would support it.
    """
    found: list[str] = []
    if factual_caption:
        if _MULTI.search(factual_caption) and _SOLO.search(caption):
            found.append("solitude")
        if (
            _FAST.search(factual_caption)
            and _SLOW.search(caption)
            and not _FAST.search(caption)
        ):
            found.append("pace")
    if _CONTACT.search(caption) and not _CONTACT.search(source_caption):
        found.append("contact")
    if _LIGHT.search(caption) and not _LIGHT.search(source_caption):
        found.append("light")
    if _POSTURE.search(caption) and not _POSTURE.search(source_caption):
        found.append("posture")
    return found


def defect_counts(
    records: Iterable[Mapping], factual: Mapping[str, str] | None = None
) -> dict:
    """Defect counts over a caption store, overall and per class and per register."""
    by_class: dict[str, int] = {c: 0 for c in DEFECT_CLASSES}
    by_register: dict[str, int] = {}
    cells = 0
    flagged_images: set[str] = set()

    for rec in records:
        image_id = str(rec.get("image_id", ""))
        source = str(rec.get("source_caption", ""))
        fact = (factual or {}).get(image_id)
        for register, text in rec.get("captions", {}).items():
            cells += 1
            hits = caption_defects(str(text), source_caption=source, factual_caption=fact)
            for h in hits:
                by_class[h] += 1
            if hits:
                by_register[register] = by_register.get(register, 0) + len(hits)
                flagged_images.add(image_id)

    total = sum(by_class.values())
    return {
        "cells": cells,
        "defects": total,
        "rate": round(total / cells, 4) if cells else 0.0,
        "by_class": by_class,
        "by_register": by_register,
        "images_with_a_defect": len(flagged_images),
        "used_factual_captions": bool(factual),
    }
