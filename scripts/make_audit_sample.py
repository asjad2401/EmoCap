#!/usr/bin/env python
"""Build the pre-registered hand-audit sample.

`configs/data.yaml` sets `generation.audit_sample: 200`. This draws that sample and writes
it as a document a person can actually read, grouped **by image** rather than by caption —
because the judgement that matters is whether the five registers of one image are
distinguishable *and* all still true of the same photograph, and that is impossible to see
from a flat list.

    uv run python scripts/make_audit_sample.py

Two sections, deliberately separated:

* **Part A — 40 random images (200 cells).** An *unbiased* sample. Whatever rate of problems
  is in here is the rate in the corpus. This is the audit.
* **Part B — flagged examples, NOT part of the 200.** Cells the validator or the grounding
  detectors flagged, so the reader knows what the 1.93% actually looks like. Excluded from
  the count on purpose: mixing them in would inflate the apparent defect rate.

Every image records its `strain` values and any flags, so a reader can check whether the
automated signals agree with their own eyes — which is the point of auditing an instrument
rather than trusting it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.grounding import caption_defects, load_factual_captions  # noqa: E402
from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.runtime import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=40,
                    help="images in Part A; one source caption each, so cells = images x 5")
    ap.add_argument("--flagged", type=int, default=8, help="flagged images in Part B")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", default="AUDIT.md")
    args = ap.parse_args()

    cfg = load_config("data")
    store = ROOT / cfg["paths"]["raw_generations"]
    recs = [json.loads(l) for l in store.read_text().splitlines() if l.strip()]
    by_image: dict[str, list[dict]] = {}
    for r in recs:
        by_image.setdefault(r["image_id"], []).append(r)
    for v in by_image.values():
        v.sort(key=lambda r: r["caption_idx"])

    try:
        factual = load_factual_captions(ROOT / "archive/v1-pilot/data/"
                                        "v1_moondream_factual_captions.csv.gz")
    except FileNotFoundError:
        factual = {}

    def flags_for(rec: dict) -> dict[str, list[str]]:
        out = {}
        for reg, text in rec["captions"].items():
            f = []
            if reg in (rec.get("rejected") or {}):
                f.append("validator: " + rec["rejected"][reg])
            d = caption_defects(str(text), source_caption=rec["source_caption"],
                                factual_caption=factual.get(rec["image_id"]))
            f += [f"grounding: {x}" for x in d]
            if f:
                out[reg] = f
        return out

    rng = random.Random(args.seed)
    all_images = sorted(by_image)
    sample = rng.sample(all_images, args.images)

    # Part B: images with at least one flagged cell, excluding anything already in Part A
    flagged_pool = [i for i in all_images
                    if i not in set(sample) and any(flags_for(r) for r in by_image[i])]
    rng.shuffle(flagged_pool)
    flagged = flagged_pool[: args.flagged]

    imgdir = ROOT / cfg["paths"]["images_dir"]
    L: list[str] = []
    L.append("# EmoCap — hand audit\n")
    L.append(f"Sample seed `{args.seed}` — rerun `scripts/make_audit_sample.py` with the "
             f"same seed to regenerate this exact document.\n")
    L.append(f"Corpus: **{len(recs):,} records / {sum(len(r['captions']) for r in recs):,} "
             f"cells / {len(all_images):,} images**, prompt v5.\n")
    L.append("""
## What to judge

For each image, open it and ask four things:

1. **Is every caption true of the photograph?** Anything invented, wrong, or not visible.
2. **Would you guess the intended register from the caption alone?** Which land, which don't.
3. **Are the five meaningfully different**, or the same sentence with a swapped adjective?
4. **Does it read like a person wrote it**, or like a template?

`strain` is the *generator's own* claim about how well the register fits: 0 natural,
1 strained, 2 no honest reading exists. Checking whether you agree with it is part of the
audit — it is an instrument being validated, not a fact.

**Verdict line per image:** replace `_____` with anything. A word is enough.
""")
    L.append("\n---\n")
    L.append(f"# PART A — the audit ({args.images} images, {args.images * 5} cells)\n")
    L.append("**Unbiased random sample.** Whatever rate of problems you find here is the "
             "rate in the corpus.\n")

    def block(img: str, n: int, total: int) -> None:
        L.append(f"\n## A{n}/{total} · `{img}`\n")
        L.append(f"`{imgdir / img}`\n")
        for rec in by_image[img]:
            fl = flags_for(rec)
            st = rec.get("strain") or {}
            L.append(f"\n**source {rec['caption_idx']}:** {rec['source_caption'].strip()}\n")
            L.append("| register | caption | strain | flags |")
            L.append("|---|---|---|---|")
            for reg in EMOTIONS:
                text = rec["captions"].get(reg, "—")
                s = st.get(reg, "–")
                f = "; ".join(fl.get(reg, [])) or ""
                L.append(f"| {reg} | {text} | {s} | {f} |")
        L.append(f"\n**Verdict:** _____  **Worst caption here:** _____\n")

    for n, img in enumerate(sample, 1):
        block(img, n, len(sample))

    L.append("\n---\n")
    L.append(f"# PART B — flagged examples ({len(flagged)} images) — NOT part of the 200\n")
    L.append("These were selected *because* an automated check flagged them, so they are "
             "**not** representative. They are here so you can see what the corpus-wide "
             "1.93% defect rate and 1.5% validator-flag rate actually look like — and "
             "judge whether the detectors are right.\n")
    for n, img in enumerate(flagged, 1):
        L.append(f"\n## B{n}/{len(flagged)} · `{img}`\n")
        L.append(f"`{imgdir / img}`\n")
        for rec in by_image[img]:
            fl = flags_for(rec)
            if not fl:
                continue
            st = rec.get("strain") or {}
            L.append(f"\n**source {rec['caption_idx']}:** {rec['source_caption'].strip()}\n")
            L.append("| register | caption | strain | flags |")
            L.append("|---|---|---|---|")
            for reg in EMOTIONS:
                if reg not in fl:
                    continue
                L.append(f"| {reg} | {rec['captions'].get(reg, '—')} | "
                         f"{st.get(reg, '–')} | {'; '.join(fl[reg])} |")
        L.append(f"\n**Detector right?** _____\n")

    L.append("\n---\n\n# Summary — fill in when done\n")
    L.append("""
| | |
|---|---|
| Captions that contradict their image | ___ / 200 |
| Registers you could NOT guess from the caption | ___ / 200 |
| Images where the five were interchangeable | ___ / 40 |
| Cells where you disagreed with `strain` | ___ |
| Part B: detectors you judged **wrong** | ___ / |

**Is this corpus good enough to train on?** _____

**The single biggest weakness:** _____

**Anything that should change before the preregistration is tagged:** _____
""")
    out = ROOT / args.out
    out.write_text("\n".join(L), encoding="utf-8")
    n_flagged_cells = sum(len(flags_for(r)) for i in sample for r in by_image[i])
    print(f"wrote {out}")
    print(f"  Part A: {len(sample)} images, {len(sample) * 5} cells, "
          f"{n_flagged_cells} already flagged by a detector")
    print(f"  Part B: {len(flagged)} flagged images (not counted)")


if __name__ == "__main__":
    main()
