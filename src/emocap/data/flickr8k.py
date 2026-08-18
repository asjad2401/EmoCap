"""Flickr8k ingestion and the canonical split.

The v2 study uses Flickr8k's **five human captions per image** as the neutral
source, not a VLM's dense description. That single choice is what gives five
references per (image, emotion) cell, which is what makes BLEU and CIDEr
measurable at all -- v1 had one reference per cell and could not have separated
its variants no matter how good they were.

The split is written to disk once and read by every later stage. v1 re-derived
splits in more than one notebook, which is how a split gets silently reordered.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Iterable, NamedTuple

__all__ = [
    "CaptionRow",
    "read_captions",
    "assign_splits",
    "write_splits",
    "read_splits",
    "caption_stats",
]

# "1000268201_693b08cb0e.jpg#0\tA child in a pink dress ..." (original release)
_TOKEN_LINE = re.compile(r"^(?P<image>\S+?)#(?P<idx>\d+)\s+(?P<caption>.*)$")


class CaptionRow(NamedTuple):
    image_id: str
    caption_idx: int
    caption: str


def read_captions(path: str | Path) -> list[CaptionRow]:
    """Parse a Flickr8k caption file.

    Handles both distributions without being told which:

    * the Kaggle ``captions.txt`` CSV with an ``image,caption`` header
    * the original ``Flickr8k.token.txt`` with ``image#idx<TAB>caption``

    ``caption_idx`` is the index of the caption within its image, and is stable:
    it becomes part of the generation key, so it must not shift between runs.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")

    lines = text.splitlines()
    rows: list[CaptionRow] = []

    # Token format is unambiguous -- try it first.
    if _TOKEN_LINE.match(lines[0]) and "\t" in lines[0]:
        for line in lines:
            m = _TOKEN_LINE.match(line)
            if m:
                rows.append(
                    CaptionRow(m["image"], int(m["idx"]), m["caption"].strip())
                )
    else:
        start = 1 if lines[0].lower().replace(" ", "").startswith("image,") else 0
        seen: dict[str, int] = {}
        for parts in csv.reader(lines[start:]):
            if len(parts) < 2:
                continue
            image_id = parts[0].strip()
            caption = ",".join(parts[1:]).strip()
            if not image_id or not caption:
                continue
            idx = seen.get(image_id, 0)
            seen[image_id] = idx + 1
            rows.append(CaptionRow(image_id, idx, caption))

    if not rows:
        raise ValueError(f"parsed no captions from {path}")
    return rows


def caption_stats(rows: Iterable[CaptionRow]) -> dict:
    """Per-image caption counts and word-length distribution.

    Printed by stage 01 so the corpus is characterised *before* anything is
    tokenized. v1 set its sequence cap without measuring, and truncated roughly
    half its targets mid-sentence.
    """
    rows = list(rows)
    per_image: dict[str, int] = {}
    lengths: list[int] = []
    for r in rows:
        per_image[r.image_id] = per_image.get(r.image_id, 0) + 1
        lengths.append(len(r.caption.split()))
    lengths.sort()

    def pct(p: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

    counts = sorted(per_image.values())
    return {
        "n_rows": len(rows),
        "n_images": len(per_image),
        "captions_per_image": {
            "min": counts[0] if counts else 0,
            "median": counts[len(counts) // 2] if counts else 0,
            "max": counts[-1] if counts else 0,
        },
        "images_without_exactly_5": sum(1 for c in per_image.values() if c != 5),
        "caption_words": {
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": lengths[-1] if lengths else 0,
        },
    }


def assign_splits(
    image_ids: Iterable[str],
    *,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Split by ``image_id``, never by row -- otherwise captions of one image
    land in different splits and the test set leaks.

    Deterministic given ``seed``: image ids are sorted before shuffling, so the
    result does not depend on the order they were read in.
    """
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    for name in ("train", "val", "test"):
        if name not in ratios:
            raise ValueError(f"missing split ratio: {name}")

    ids = sorted(set(image_ids))
    if not ids:
        raise ValueError("no image ids to split")
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    out: dict[str, str] = {}
    for i, img in enumerate(ids):
        if i < n_train:
            out[img] = "train"
        elif i < n_train + n_val:
            out[img] = "val"
        else:
            out[img] = "test"
    return out


def write_splits(splits: dict[str, str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "split"])
        for image_id in sorted(splits):
            w.writerow([image_id, splits[image_id]])
    return path


def read_splits(path: str | Path) -> dict[str, str]:
    """Read the frozen split. Every stage after 01 uses this, never re-derives."""
    with Path(path).open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["image_id", "split"]:
            raise ValueError(f"unexpected split file header: {r.fieldnames}")
        return {row["image_id"]: row["split"] for row in r}
