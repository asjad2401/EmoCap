#!/usr/bin/env python
"""Materialise the post-hoc arms and prove they are nested inside the registered ones.

    uv run python scripts/build_posthoc_arms.py

Writes ``data/arms/S_paired_matched.jsonl``, ``data/arms/S_unpaired_scaled.jsonl`` and
``data/arms/posthoc_manifest.json``. Kept apart from ``scripts/build_arms.py`` and from
``data/arms/manifest.json`` so that the registered manifest -- whose hashes are quoted in
``configs/prereg.lock.yaml`` -- never changes because of work decided after the tag.

The containment checks are the point of this script, not decoration. A post-hoc arm is only
worth reporting if it is literally a view of cells the registered arms already trained on;
the moment it stops being one, "the post-hoc arm drew easier data" becomes an untestable
alternative explanation for whatever it shows. So the script asserts, and fails loudly
rather than writing an arm it cannot vouch for.

The two arms answer opposite halves of the same question. ``S_paired_matched`` removes the
volume and keeps the pairing; ``S_unpaired_scaled`` keeps the volume and removes the
pairing. Only together do they separate the two.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.runtime import rel  # noqa: E402

from emocap.data.arms import load_corpus, write_arm  # noqa: E402
from emocap.data.exclusions import load_exclusions  # noqa: E402
from emocap.data.posthoc import (build_s_paired_matched,  # noqa: E402
                                 build_s_unpaired_scaled, load_v1_pilot)
from emocap.data.prompt import EMOTIONS  # noqa: E402

PILOT = "archive/v1-pilot/data/v1_emotion_captions.csv.gz"

DECIDED = {"S_paired_matched": "2026-08-23", "S_unpaired_scaled": "2026-08-24"}
WHY = {
    "S_paired_matched":
        "Volume held at S_unpaired's 4,390 cells, paired structure kept. Answers whether "
        "pairing works at a small budget. Decided after the prereg-v2 tag; post-hoc.",
    "S_unpaired_scaled":
        "Volume held at S_paired5's 40,235 cells and images, register contrast removed. "
        "Answers whether the volume works WITHOUT pairing -- the half S_paired_matched "
        "could not reach. Decided after the prereg-v2 tag; post-hoc.",
}


def read_arm(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def key(c: dict) -> tuple:
    return (c["image_id"], c["caption_idx"], c["emotion"])


def check_paired_matched(cells: list[dict], unpaired: list[dict],
                         paired5: list[dict]) -> dict:
    """Its images sit inside S_unpaired's; its cells sit inside S_paired5's."""
    imgs = {c["image_id"] for c in cells}
    mine, theirs = {key(c) for c in cells}, {key(c) for c in paired5}
    stray_imgs = sorted(imgs - {c["image_id"] for c in unpaired})
    stray_cells = sorted(mine - theirs)
    # Every S_unpaired cell standing on a shared image must reappear here: same image,
    # same source caption, that image's one register among the five.
    missing = sorted(key(c) for c in unpaired if c["image_id"] in imgs
                     and key(c) not in mine)
    if stray_imgs or stray_cells or missing:
        raise SystemExit(
            f"S_paired_matched NOT NESTED -- refusing to write.\n"
            f"  images not in S_unpaired : {len(stray_imgs)}  {stray_imgs[:3]}\n"
            f"  cells not in S_paired5   : {len(stray_cells)}  {stray_cells[:3]}\n"
            f"  S_unpaired cells dropped : {len(missing)}  {missing[:3]}")
    return {"images_subset_of_S_unpaired": True,
            "cells_subset_of_S_paired5": True,
            "covers_every_S_unpaired_cell_on_shared_images": True}


def check_unpaired_scaled(cells: list[dict], unpaired: list[dict], paired5: list[dict],
                          paired25: list[dict]) -> dict:
    """Matched to S_paired5 on images and count, and holding ONE register per image.

    The last check is the arm's whole reason for existing: if any image carries two
    registers, the register contrast this arm is built to remove has leaked back in and
    the comparison against S_paired5 measures nothing.
    """
    imgs = {c["image_id"] for c in cells}
    p5_imgs = {c["image_id"] for c in paired5}
    per_img_regs = {}
    per_img_idx = {}
    for c in cells:
        per_img_regs.setdefault(c["image_id"], set()).add(c["emotion"])
        per_img_idx.setdefault(c["image_id"], set()).add(c["caption_idx"])

    problems = []
    if imgs != p5_imgs:
        problems.append(f"image set differs from S_paired5 by "
                        f"{len(imgs ^ p5_imgs)} images")
    if len(cells) != len(paired5):
        problems.append(f"{len(cells)} cells against S_paired5's {len(paired5)}")
    multi = [i for i, r in per_img_regs.items() if len(r) != 1]
    if multi:
        problems.append(f"{len(multi)} images carry more than one register: {multi[:3]}")
    short = [i for i, k in per_img_idx.items() if len(k) != 5]
    if short:
        problems.append(f"{len(short)} images do not carry all 5 source captions: "
                        f"{short[:3]}")
    stray = sorted({key(c) for c in cells} - {key(c) for c in paired25})
    if stray:
        problems.append(f"{len(stray)} cells are not in S_paired25: {stray[:3]}")
    # The volume ladder: S_unpaired must sit inside this arm, cell for cell.
    dropped = sorted(key(c) for c in unpaired if key(c) not in {key(x) for x in cells})
    if dropped:
        problems.append(f"{len(dropped)} S_unpaired cells are absent: {dropped[:3]}")
    if problems:
        raise SystemExit("S_unpaired_scaled FAILED ITS CHECKS -- refusing to write.\n"
                         + "".join(f"  {p}\n" for p in problems))
    return {"images_identical_to_S_paired5": True,
            "cell_count_identical_to_S_paired5": True,
            "exactly_one_register_per_image": True,
            "all_five_source_captions_per_image": True,
            "cells_subset_of_S_paired25": True,
            "contains_every_S_unpaired_cell": True}


def summarise(name: str, cells: list[dict], digest: str) -> dict:
    reg = Counter(c["emotion"] for c in cells)
    fold = Counter(c["fold"] for c in cells)
    imgs = {c["image_id"] for c in cells}
    print(f"{name:<18} cells {len(cells):>7,}  images {len(imgs):>6,}  "
          f"per-register {sorted(set(reg.values()))}  "
          f"folds {[fold[k] for k in sorted(fold)]}")
    return {"cells": len(cells), "images": len(imgs), "sha256": digest,
            "per_register": {e: reg[e] for e in EMOTIONS},
            "per_fold": {str(k): fold[k] for k in sorted(fold)}}


def main() -> None:
    out = ROOT / "data/arms"
    corpus = load_corpus(ROOT / "data/generated/captions_corpus.jsonl")
    pilot = load_v1_pilot(ROOT / PILOT)
    excluded = frozenset(load_exclusions())

    unpaired = read_arm(out / "S_unpaired.jsonl")
    paired5 = read_arm(out / "S_paired5.jsonl")
    paired25 = read_arm(out / "S_paired25.jsonl")

    matched = build_s_paired_matched(corpus=corpus, v1_captions=pilot, excluded=excluded)
    scaled = build_s_unpaired_scaled(corpus=corpus, v1_captions=pilot, excluded=excluded)

    nested = {
        "S_paired_matched": check_paired_matched(matched, unpaired, paired5),
        "S_unpaired_scaled": check_unpaired_scaled(scaled, unpaired, paired5, paired25),
    }

    reg_matched = Counter(c["emotion"] for c in matched)
    if len(set(reg_matched.values())) != 1:
        raise SystemExit(f"S_paired_matched registers not balanced: {dict(reg_matched)}")
    # S_unpaired_scaled cannot be exactly balanced -- 8,047 images do not divide by five --
    # so the tolerance is stated rather than assumed. 8,050 vs 8,045 is 0.06%.
    reg_scaled = Counter(c["emotion"] for c in scaled)
    spread = max(reg_scaled.values()) - min(reg_scaled.values())
    if spread > 5 * len(EMOTIONS):
        raise SystemExit(f"S_unpaired_scaled registers too uneven: {dict(reg_scaled)}")

    arms: dict[str, dict] = {}
    for name, cells in (("S_paired_matched", matched), ("S_unpaired_scaled", scaled)):
        digest = write_arm(cells, out / f"{name}.jsonl")
        arms[name] = summarise(name, cells, digest)
        arms[name]["decided"] = DECIDED[name]
        arms[name]["why"] = WHY[name]
        arms[name]["nested_in"] = nested[name]

    man = {
        "post_hoc": True,
        "registered_manifest": "data/arms/manifest.json (unchanged by this script)",
        "the_pair": "S_paired_matched removes the volume and keeps the pairing; "
                    "S_unpaired_scaled keeps the volume and removes the pairing. Only "
                    "together do they separate the two, which the registered P3 cannot.",
        "selection": {
            "S_paired_matched": {
                "images": "first 878 of sorted(shared_images, key=sha1('unpaired-v1' + id))"
                          " -- the same ranking S_unpaired draws its 4,390 from",
                "caption_idx": "sha1('cap-v1' + image_id) % 5, as in S_paired5",
                "registers": "all five per image"},
            "S_unpaired_scaled": {
                "images": "all 8,047 -- identical to S_paired5",
                "caption_idx": "all five per image",
                "registers": "ONE per image; the 4,390 images S_unpaired used keep the "
                             "register it assigned, the rest are dealt round-robin within "
                             "fold over the same sha1('unpaired-v1' + id) ordering"}},
        "fold": "sha1('fold-v1' + image_id) % 5, unchanged for every arm",
        "arms": arms,
    }
    (out / "posthoc_manifest.json").write_text(json.dumps(man, indent=2))

    print(f"\n  S_paired_matched: {len(matched):,} cells vs S_unpaired's {len(unpaired):,}"
          f"  (volume matched, pairing kept)")
    print(f"  S_unpaired_scaled: {len(scaled):,} cells vs S_paired5's {len(paired5):,}"
          f"  (volume matched, pairing removed)")
    print(f"  registers in S_unpaired_scaled: {dict(reg_scaled)}  spread {spread}")
    print(f"\nwrote both arms and {rel(out / 'posthoc_manifest.json')}")


if __name__ == "__main__":
    main()
