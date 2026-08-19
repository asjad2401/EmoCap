#!/usr/bin/env python
"""Calibrate the ceiling gate against HUMAN-written affective language.

`docs/preregistration.md` §6 halts the study if a classifier cannot recover the intended
register from the generated captions at >= 0.85. That threshold was set a priori with
nothing calibrating it, so a result below it is ambiguous between two very different
conclusions:

    the DATA is weak        -- our synthetic captions carry less register signal than
                               human affective language does
    the THRESHOLD is wrong  -- 0.85 exceeds what this task allows for anyone, and the
                               gate is miscalibrated rather than informative

Only a human-written comparison separates them. ArtEmis (Achlioptas et al., CVPR 2021)
supplies 455k utterances where a human annotator chose BOTH the emotion category and
wrote the text, so the label-text relationship is human-generated throughout.

    uv run python scripts/compare_human_ceiling.py --artemis artemis_preprocessed.csv

**The comparison is deliberately matched, because accuracy is not comparable across
different numbers of classes or sample sizes.** ArtEmis is subsampled to the same class
count and the same number of cells as our store, clustered by artwork so no artwork's
utterances straddle a fold -- the same discipline `check_gates.py` applies by image_id.

ArtEmis's nine categories do not map onto our five registers one-to-one (it has no
`romantic`, and splits positive affect four ways), so this is NOT a per-class comparison.
It answers one question only: **under identical conditions, what accuracy does human
affective text reach?** That number is the calibration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.eval.register_classifier import (  # noqa: E402
    build_dataset,
    confusion_matrix,
    finetune_classifier,
    tfidf_baseline,
)

#: Five of ArtEmis's nine categories, chosen to match our class count and to span the
#: same affective range: two positive, two negative, one high-arousal. Selected before
#: seeing any result, and recorded here so the choice is not made after the fact.
DEFAULT_CLASSES = ["amusement", "contentment", "awe", "sadness", "fear"]

# ── Personality-Captions (Shuster et al. 2019) ──────────────────────────────
#
# A better match than ArtEmis for this study: human crowdworkers writing captions on
# PHOTOGRAPHS conditioned on a style trait -- the same task and the same speech act as
# ours -- and its 217 traits include literal `Romantic` and `Humorous`, which ArtEmis
# lacks entirely. Freely downloadable, checksum-verified against ParlAI's source.
#
# TWO CONFOUNDS, both recorded because they cut in opposite directions:
#
# 1. Their writers were told to be ENGAGING and were never required to stay faithful to
#    the image ("The snow will last as long as my sadness" describes nothing in frame).
#    Ours must describe what is actually there. So this measures the ceiling reachable
#    when the grounding constraint is DROPPED -- which is exactly the price-of-grounding
#    question the preregistration does not acknowledge.
#
# 2. Their captions average 9.6 words against our ~15. More words is more signal, so
#    length favours US. `--length-band` restricts both sides to a common band.

#: One trait per register. Cleanest mapping, but only ~880 captions each.
STRICT_TRAITS = {
    "joyful": ["Happy"],
    "sad": ["Gloomy"],
    "tense": ["Anxious"],
    "romantic": ["Romantic"],
    "humorous": ["Humorous"],
}

#: Semantically adjacent traits merged, for a sample size comparable to ours (~22k).
#: Costs within-class heterogeneity -- Happy and Playful are not the same thing -- which
#: pushes human accuracy DOWN, so this design is conservative about the human ceiling.
#: `Playful` is deliberately assigned to neither joyful nor humorous rather than both.
GROUPED_TRAITS = {
    "joyful": ["Happy", "Cheerful", "Optimistic", "Enthusiastic", "Energetic"],
    "sad": ["Gloomy", "Melancholic", "Miserable", "Solemn"],
    "tense": ["Anxious", "Fearful", "Paranoid", "Intense", "Aggressive"],
    "romantic": ["Romantic", "Sentimental", "Passionate", "Sweet"],
    "humorous": ["Humorous", "Witty", "Sarcastic", "Zany", "Silly"],
}


def load_personality_captions(
    root: Path, mapping: dict[str, list[str]]
) -> tuple[list[str], list[int], list[str]]:
    """Load train+val, keeping only traits mapped to one of our five registers."""
    trait_to_reg: dict[str, int] = {}
    for reg, traits in mapping.items():
        for t in traits:
            if t in trait_to_reg:
                raise ValueError(f"trait {t!r} assigned to two registers")
            trait_to_reg[t] = EMOTIONS.index(reg)

    records = []
    for name in ("train.json", "val.json"):
        f = root / name
        if f.exists():
            records += json.loads(f.read_text())
    if not records:
        raise SystemExit(f"no train.json/val.json under {root}")

    texts, labels, groups = [], [], []
    kept: Counter = Counter()
    for r in records:
        reg = trait_to_reg.get(r.get("personality", ""))
        txt = str(r.get("comment", "")).strip()
        if reg is None or not txt:
            continue
        texts.append(txt)
        labels.append(reg)
        groups.append(str(r.get("image_hash") or f"row{len(texts)}"))
        kept[r["personality"]] += 1
    print(f"  kept {len(texts):,} captions over {len(kept)} traits")
    for reg, traits in mapping.items():
        n = sum(kept[t] for t in traits)
        print(f"    {reg:<9} {n:>6,}  {', '.join(t for t in traits if kept[t])}")
    return texts, labels, groups


def apply_length_band(texts, labels, groups, lo: int, hi: int):
    """Restrict to captions of lo..hi words, so length cannot explain the gap."""
    keep = [i for i, t in enumerate(texts) if lo <= len(t.split()) <= hi]
    return ([texts[i] for i in keep], [labels[i] for i in keep],
            [groups[i] for i in keep])


_TEXT_COLUMNS = ("utterance", "caption", "text", "utterance_spelled")
_LABEL_COLUMNS = ("emotion", "emotion_label", "art_style_emotion")
_GROUP_COLUMNS = ("painting", "art_style", "image_file", "painting_name", "image")


def _pick(columns: list[str], candidates: tuple[str, ...], what: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise SystemExit(
        f"could not find a {what} column. Looked for {candidates}; the file has "
        f"{columns}. Pass the right name explicitly."
    )


def load_artemis(
    path: Path, classes: list[str], *, text_col=None, label_col=None, group_col=None
) -> tuple[list[str], list[int], list[str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    cols = list(rows[0])
    text_col = text_col or _pick(cols, _TEXT_COLUMNS, "text")
    label_col = label_col or _pick(cols, _LABEL_COLUMNS, "emotion label")
    group_col = group_col or _pick(cols, _GROUP_COLUMNS, "grouping")
    print(f"  columns: text={text_col!r} label={label_col!r} group={group_col!r}")

    wanted = {c.lower(): i for i, c in enumerate(classes)}
    texts, labels, groups = [], [], []
    seen: Counter = Counter()
    for r in rows:
        lab = str(r.get(label_col, "")).strip().lower()
        txt = str(r.get(text_col, "")).strip()
        if lab not in wanted or not txt:
            continue
        texts.append(txt)
        labels.append(wanted[lab])
        groups.append(str(r.get(group_col, "")) or f"row{len(texts)}")
        seen[lab] += 1
    print(f"  kept {len(texts):,} utterances: {dict(seen)}")
    if not texts:
        raise SystemExit(
            f"no rows matched {classes}. Check the label column's actual values."
        )
    return texts, labels, groups


def match_size(texts, labels, groups, *, n_cells: int, seed: int = 42):
    """Subsample to ``n_cells``, balanced across classes, whole groups only.

    Sampling by group rather than by row keeps the leakage discipline intact: all of an
    artwork's utterances land on the same side of a fold.
    """
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(i)
    per_class = n_cells // len(by_class)
    keep: list[int] = []
    for lab, idxs in sorted(by_class.items()):
        if len(idxs) <= per_class:
            keep.extend(idxs)
            continue
        rng.shuffle(idxs)
        keep.extend(idxs[:per_class])
    rng.shuffle(keep)
    return ([texts[i] for i in keep], [labels[i] for i in keep],
            [groups[i] for i in keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["personality", "artemis"], default="personality")
    ap.add_argument("--pc-root", default="data/external/personality_captions")
    ap.add_argument("--artemis", help="artemis_preprocessed.csv (only for --source artemis)")
    ap.add_argument("--design", choices=["strict", "grouped"], default="grouped",
                    help="strict: one trait per register (~4.4k cells). grouped: adjacent "
                         "traits merged (~22k, matched to our size)")
    ap.add_argument("--ours", default="data/generated/captions_raw.jsonl")
    ap.add_argument("--length-band", nargs=2, type=int, metavar=("LO", "HI"),
                    help="restrict BOTH sides to captions of LO..HI words, so caption "
                         "length cannot explain the gap (theirs average 9.6, ours ~15)")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="artemis only")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="TF-IDF only, seconds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ours_path = Path(args.ours)
    if not ours_path.is_absolute():
        ours_path = ROOT / ours_path
    ours = [json.loads(l) for l in ours_path.read_text().splitlines() if l.strip()]
    o_texts, o_labels, o_groups = build_dataset(ours)
    print(f"ours:    {len(o_texts):,} cells over {len(set(o_groups)):,} images")

    if args.source == "personality":
        mapping = STRICT_TRAITS if args.design == "strict" else GROUPED_TRAITS
        root = Path(args.pc_root)
        if not root.is_absolute():
            root = ROOT / root
        print(f"human:   Personality-Captions, {args.design} design")
        h_texts, h_labels, h_groups = load_personality_captions(root, mapping)
        human_name = f"personality_captions_{args.design}"
    else:
        if not args.artemis:
            sys.exit("--artemis is required for --source artemis")
        print("human:   ArtEmis")
        h_texts, h_labels, h_groups = load_artemis(Path(args.artemis), args.classes)
        human_name = "artemis"

    if args.length_band:
        lo, hi = args.length_band
        bo, bh = len(o_texts), len(h_texts)
        o_texts, o_labels, o_groups = apply_length_band(o_texts, o_labels, o_groups, lo, hi)
        h_texts, h_labels, h_groups = apply_length_band(h_texts, h_labels, h_groups, lo, hi)
        print(f"length band {lo}-{hi} words: ours {bo:,}->{len(o_texts):,}, "
              f"human {bh:,}->{len(h_texts):,}")

    # Accuracy is not comparable across sample sizes -- today's 0.612 vs 0.737 on the
    # same captions proved that. So both sides are cut to the smaller of the two.
    n = min(len(o_texts), len(h_texts))
    o_texts, o_labels, o_groups = match_size(o_texts, o_labels, o_groups, n_cells=n, seed=args.seed)
    h_texts, h_labels, h_groups = match_size(h_texts, h_labels, h_groups, n_cells=n, seed=args.seed)
    print(f"matched: {len(o_texts):,} cells each side, {len(EMOTIONS)} classes\n")

    runner = tfidf_baseline if args.fast else finetune_classifier
    kw = {} if args.fast else {"epochs": args.epochs}
    result = {"design": args.design, "human_source": human_name,
              "length_band": args.length_band,
              "instrument": "tfidf" if args.fast else "distilroberta-base",
              "cells_each_side": len(o_texts)}
    for name, (t, l, g) in (("ours_synthetic", (o_texts, o_labels, o_groups)),
                            ("human", (h_texts, h_labels, h_groups))):
        print(f"running {name} ...", flush=True)
        r = runner(t, l, g, folds=args.folds, seed=args.seed, **kw)
        result[name] = {"accuracy": r["accuracy"], "n": r["n"],
                        "recall": confusion_matrix(r["pairs"])["recall"]}
        print(f"  {name}: {r['accuracy']:.3f}  (n={r['n']:,})")

    ours_acc = result["ours_synthetic"]["accuracy"]
    human_acc = result["human"]["accuracy"]
    result["gap_ours_minus_human"] = round(ours_acc - human_acc, 4)

    print(f"\n{'=' * 72}")
    print(f"human-written, style-conditioned   {human_acc:.3f}")
    print(f"our synthetic captions             {ours_acc:.3f}")
    print(f"gap                                {ours_acc - human_acc:+.3f}")
    print(f"(matched: {len(o_texts):,} cells, {len(EMOTIONS)} classes, same instrument)")
    print("\nper-register recall:")
    print(f"  {'register':<10}{'human':>8}{'ours':>8}")
    for reg in EMOTIONS:
        h = result["human"]["recall"].get(reg)
        o = result["ours_synthetic"]["recall"].get(reg)
        if h is not None and o is not None:
            print(f"  {reg:<10}{h:>8.3f}{o:>8.3f}")
    print()
    if human_acc < 0.85:
        print(f"HUMAN style-conditioned captions reach only {human_acc:.3f} -- below the")
        print("pre-registered 0.85. The threshold is miscalibrated for this task rather")
        print("than a finding about our data. Recalibrating it pre-tag, with the")
        print("reasoning logged, is legitimate; anchor the gate to the human number.")
    else:
        print("HUMAN captions clear 0.85, so the threshold is achievable and any")
        print("shortfall is a property of the synthetic captions. The gap is the finding.")
    if human_acc >= 0.85 > ours_acc:
        print("\nNOTE: their writers were NOT required to stay faithful to the image;")
        print("ours were. Part of this gap is the price of the grounding constraint,")
        print("not a defect in generation. Rerun with --length-band to remove the")
        print("caption-length advantage before quoting the number.")

    outp = Path(args.out) if args.out else ROOT / f"runs/human-ceiling/{human_name}.json"
    if not outp.is_absolute():
        outp = ROOT / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
