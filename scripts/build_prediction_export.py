#!/usr/bin/env python
"""Stage the arms' generated captions for upload, with the reference text removed.

    uv run python scripts/build_prediction_export.py
    uv run python scripts/push_kaggle.py --part predictions

Writes `runs/kaggle-preds/<tag>/predictions.jsonl`, one stripped line per decoded cell.

**Why this exists.** Analyses that only read generated captions -- P4's judges, any future
judge or metric -- are pure inference and belong wherever there is a spare GPU. They cannot
run on Kaggle today because `predictions.jsonl` is gitignored and lives only on the laptop,
and a laptop running an hour of sustained cross-validation is exactly what wedged the first
P4 attempt overnight.

**What is dropped, and why that is the point.** The full prediction records carry
`reference`: for the Flickr8k arms that is Gemini's caption, and for `H_unpaired` it is
verbatim Personality-Captions human text. No analysis of this kind needs it. Removing it
halves the payload AND means the upload contains **only captions this project's own models
wrote** -- no third-party corpus text leaves the machine. That is a better answer to the
redistribution question than "the dataset is private", which is the answer the arms upload
has to rely on.

`image_id` is kept because clustering is by image, and `fold` because a per-fold breakdown
is otherwise unrecoverable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEEP = ("image_id", "emotion", "generated", "fold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/arms")
    ap.add_argument("--out", default="runs/kaggle-preds")
    args = ap.parse_args()

    src_root, out = ROOT / args.runs, ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = kept_bytes = raw_bytes = 0
    runs = 0
    for d in sorted(src_root.iterdir()):
        p = d / "predictions.jsonl"
        if not d.is_dir() or not p.exists():
            continue
        raw_bytes += p.stat().st_size
        lines = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            missing = [k for k in KEEP if k not in r]
            if missing:
                raise SystemExit(f"{p} line is missing {missing}")
            lines.append(json.dumps({k: r[k] for k in KEEP}))
        body = "\n".join(lines) + "\n"
        dst = out / d.name
        dst.mkdir()
        (dst / "predictions.jsonl").write_text(body)
        kept_bytes += len(body.encode())
        total += len(lines)
        runs += 1

    # A cheap guarantee that no reference text slipped through, checked rather than assumed.
    for f in out.rglob("predictions.jsonl"):
        first = json.loads(f.open().readline())
        if set(first) != set(KEEP):
            raise SystemExit(f"{f} has unexpected fields: {sorted(first)}")

    print(f"{runs} runs, {total:,} predictions")
    print(f"  {raw_bytes/1e6:.0f} MB with references -> {kept_bytes/1e6:.0f} MB without")
    print(f"  fields kept: {', '.join(KEEP)}   (reference dropped)")
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("  now: uv run python scripts/push_kaggle.py --part predictions")


if __name__ == "__main__":
    main()
