#!/usr/bin/env python
"""Materialise the six data arms and report what the lock file must record.

    uv run python scripts/build_arms.py

Writes data/arms/<arm>.jsonl plus data/arms/manifest.json (counts, sha256, and the
selection rules). Run this BEFORE applying the prereg tag: the lock names arm sizes, and
naming a size the corpus cannot supply is exactly the failure mode this study is trying
to avoid.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.arms import ARMS, build_all_arms, write_arm  # noqa: E402
from emocap.data.prompt import EMOTIONS  # noqa: E402

STRICT = {"joyful": "Happy", "sad": "Gloomy", "tense": "Anxious",
          "romantic": "Romantic", "humorous": "Humorous"}


def human_cells(root: Path, images: Path) -> list[dict]:
    """Personality-Captions under the strict 1:1 trait mapping, images on disk only."""
    trait_to_reg = {v: k for k, v in STRICT.items()}
    have = {p.stem for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")}
    out = []
    for name in ("train.json", "val.json"):
        f = root / name
        if not f.exists():
            continue
        for r in json.loads(f.read_text()):
            reg = trait_to_reg.get(r.get("personality", ""))
            txt = str(r.get("comment", "")).strip()
            h = str(r.get("image_hash") or "")
            if reg and txt and h in have:
                out.append({"image_id": f"{h}.jpg", "emotion": reg, "text": txt})
    return out


def main() -> None:
    arms = build_all_arms(
        corpus_path=ROOT / "data/generated/captions_corpus.jsonl",
        v1_path=ROOT / "data/generated/captions_raw.jsonl",
        human_cells=human_cells(ROOT / "data/external/personality_captions",
                                ROOT / "data/external/yfcc_images"),
    )
    man: dict = {"arms": {}, "folds": 5, "fold_split_by": "image_id",
                 "selection": {
                     "fold": "sha1('fold-v1' + image_id) % 5",
                     "paired5_caption_idx": "sha1('cap-v1' + image_id) % 5",
                     "unpaired_images": "878 x 5 by sha1('unpaired-v1' + image_id), "
                                        "registers round-robin within fold",
                     "v1_matching": "same images, same caption_idx, same register as S"}}
    out = ROOT / "data/arms"
    for a in ARMS:
        cells = arms[a]
        digest = write_arm(cells, out / f"{a}.jsonl")
        reg = Counter(c["emotion"] for c in cells)
        fold = Counter(c["fold"] for c in cells)
        man["arms"][a] = {
            "cells": len(cells), "images": len({c["image_id"] for c in cells}),
            "sha256": digest, "per_register": {e: reg[e] for e in EMOTIONS},
            "per_fold": {str(k): fold[k] for k in sorted(fold)}}
        print(f"{a:12s} cells {len(cells):7,}  images {man['arms'][a]['images']:6,}  "
              f"per-register {sorted(set(reg.values()))}  "
              f"folds {[fold[k] for k in sorted(fold)]}")
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"\nwrote {out}/manifest.json")


if __name__ == "__main__":
    main()
