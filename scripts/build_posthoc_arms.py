#!/usr/bin/env python
"""Materialise the post-hoc arms and prove they are nested inside the registered ones.

    uv run python scripts/build_posthoc_arms.py

Writes ``data/arms/S_paired_matched.jsonl`` and ``data/arms/posthoc_manifest.json``.
Kept apart from ``scripts/build_arms.py`` and from ``data/arms/manifest.json`` so that
the registered manifest -- whose hashes are quoted in ``configs/prereg.lock.yaml`` --
never changes because of work decided after the tag.

The containment checks are the point of this script, not decoration. ``S_paired_matched``
is only worth reporting if it is literally a view of cells the registered arms already
trained on; the moment it stops being one, "the post-hoc arm drew easier data" becomes an
untestable alternative explanation for whatever it shows. So the script asserts, and
fails loudly rather than writing an arm it cannot vouch for.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.arms import load_corpus, write_arm  # noqa: E402
from emocap.data.exclusions import load_exclusions  # noqa: E402
from emocap.data.posthoc import build_s_paired_matched, load_v1_pilot  # noqa: E402
from emocap.data.prompt import EMOTIONS  # noqa: E402

PILOT = "archive/v1-pilot/data/v1_emotion_captions.csv.gz"

DECIDED = "2026-08-23"
WHY = ("Isolates paired structure from data volume, which P3 as registered confounds. "
       "Decided after the prereg-v2 tag; must be reported as post-hoc.")


def read_arm(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def key(c: dict) -> tuple:
    return (c["image_id"], c["caption_idx"], c["emotion"])


def main() -> None:
    out = ROOT / "data/arms"
    corpus = load_corpus(ROOT / "data/generated/captions_corpus.jsonl")
    cells = build_s_paired_matched(
        corpus=corpus,
        v1_captions=load_v1_pilot(ROOT / PILOT),
        excluded=frozenset(load_exclusions()))

    # ── containment, against the registered arms as they sit on disk ─────────
    unpaired = read_arm(out / "S_unpaired.jsonl")
    paired5 = read_arm(out / "S_paired5.jsonl")
    imgs = {c["image_id"] for c in cells}
    unpaired_imgs = {c["image_id"] for c in unpaired}
    mine, theirs = {key(c) for c in cells}, {key(c) for c in paired5}

    stray_imgs = sorted(imgs - unpaired_imgs)
    stray_cells = sorted(mine - theirs)
    # Every S_unpaired cell standing on a shared image must reappear here: same image,
    # same source caption, that image's one register among the five.
    missing = sorted(key(c) for c in unpaired if c["image_id"] in imgs
                     and key(c) not in mine)
    if stray_imgs or stray_cells or missing:
        raise SystemExit(
            f"NOT NESTED -- refusing to write.\n"
            f"  images not in S_unpaired : {len(stray_imgs)}  {stray_imgs[:3]}\n"
            f"  cells not in S_paired5   : {len(stray_cells)}  {stray_cells[:3]}\n"
            f"  S_unpaired cells dropped : {len(missing)}  {missing[:3]}")

    reg = Counter(c["emotion"] for c in cells)
    fold = Counter(c["fold"] for c in cells)
    if len(set(reg.values())) != 1:
        raise SystemExit(f"registers are not balanced: {dict(reg)}")

    digest = write_arm(cells, out / "S_paired_matched.jsonl")
    man = {
        "post_hoc": True, "decided": DECIDED, "why": WHY,
        "registered_manifest": "data/arms/manifest.json (unchanged by this script)",
        "selection": {
            "images": "first 878 of sorted(shared_images, key=sha1('unpaired-v1' + id)) "
                      "-- the same ranking S_unpaired draws its 4,390 from",
            "caption_idx": "sha1('cap-v1' + image_id) % 5, as in S_paired5",
            "registers": "all five per image",
            "fold": "sha1('fold-v1' + image_id) % 5, unchanged"},
        "nested_in": {
            "images_subset_of_S_unpaired": True,
            "cells_subset_of_S_paired5": True,
            "covers_every_S_unpaired_cell_on_shared_images": True},
        "arms": {"S_paired_matched": {
            "cells": len(cells), "images": len(imgs), "sha256": digest,
            "per_register": {e: reg[e] for e in EMOTIONS},
            "per_fold": {str(k): fold[k] for k in sorted(fold)}}}}
    (out / "posthoc_manifest.json").write_text(json.dumps(man, indent=2))

    print(f"S_paired_matched  cells {len(cells):,}  images {len(imgs):,}  "
          f"per-register {sorted(set(reg.values()))}  "
          f"folds {[fold[k] for k in sorted(fold)]}")
    print(f"  nested: {len(imgs):,} images all in S_unpaired's {len(unpaired_imgs):,}; "
          f"all {len(cells):,} cells in S_paired5; "
          f"all {sum(1 for c in unpaired if c['image_id'] in imgs):,} shared "
          f"S_unpaired cells present")
    print(f"  size match: S_unpaired has {len(unpaired):,} cells, this arm has "
          f"{len(cells):,}")
    print(f"\nwrote {out}/S_paired_matched.jsonl  ({digest[:16]}...)")
    print(f"wrote {out}/posthoc_manifest.json")


if __name__ == "__main__":
    main()
