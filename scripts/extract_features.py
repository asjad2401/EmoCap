#!/usr/bin/env python
"""Encode every image any arm references, once, with the frozen CLIP tower.

    uv run python scripts/extract_features.py

Reads the arm files rather than a directory listing, so the feature store covers exactly
the images the study uses -- no more (2.4 GB of unused YFCC) and, critically, no less.
Refuses to finish quietly if an arm has an image the store lacks.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.arms import ARMS  # noqa: E402
from emocap.features.clip import extract_features  # noqa: E402
from emocap.runtime import load_config  # noqa: E402

FLICKR = ROOT / "data/flickr8k/Images"
YFCC = ROOT / "data/external/yfcc_images"


def main() -> None:
    cfg = load_config("model")["vision"]
    wanted: dict[str, Path] = {}
    per_arm: dict[str, set[str]] = {}
    for arm in ARMS:
        f = ROOT / "data/arms" / f"{arm}.jsonl"
        ids = {json.loads(l)["image_id"] for l in f.read_text().splitlines() if l.strip()}
        per_arm[arm] = ids
        for i in ids:
            wanted[i] = (YFCC if arm.startswith("H") else FLICKR) / i
    print(f"images referenced by the six arms: {len(wanted):,}")
    missing = [i for i, p in wanted.items() if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} image files are absent, e.g. {missing[:5]}")

    t0 = time.time()
    stats = extract_features(
        wanted, weights=ROOT / cfg["weights"], out_dir=ROOT / "data/features",
        progress=lambda n, tot: print(f"  {n:,}/{tot:,}  [{(time.time()-t0)/60:.1f} min]",
                                      flush=True) if n % 1024 == 0 or n == tot else None)
    print(f"\nencoded {stats['images']:,} images  dim {stats['dim']}  "
          f"device {stats['device']}  failed {len(stats['failed'])}  "
          f"[{(time.time()-t0)/60:.1f} min]")

    idx = set(json.loads((ROOT / "data/features/clip_vit_b32_index.json").read_text()))
    for arm, ids in per_arm.items():
        gap = ids - idx
        print(f"  {arm:12s} {len(ids):6,} images  uncovered {len(gap)}")
        if gap:
            raise SystemExit(f"{arm} references {len(gap)} images with no features")


if __name__ == "__main__":
    main()
