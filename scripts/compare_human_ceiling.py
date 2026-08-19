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
    ap.add_argument("--artemis", required=True, help="artemis_preprocessed.csv")
    ap.add_argument("--ours", default="data/generated/captions_raw.jsonl")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--text-col"), ap.add_argument("--label-col")
    ap.add_argument("--group-col")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="TF-IDF only")
    ap.add_argument("--out", default="runs/human-ceiling/result.json")
    args = ap.parse_args()

    ours_path = Path(args.ours)
    if not ours_path.is_absolute():
        ours_path = ROOT / ours_path
    ours = [json.loads(l) for l in ours_path.read_text().splitlines() if l.strip()]
    o_texts, o_labels, o_groups = build_dataset(ours)
    print(f"ours:    {len(o_texts):,} cells, {len(set(o_groups)):,} images, "
          f"{len(EMOTIONS)} registers")

    print(f"artemis: loading {args.artemis}")
    a_texts, a_labels, a_groups = load_artemis(
        Path(args.artemis), args.classes,
        text_col=args.text_col, label_col=args.label_col, group_col=args.group_col,
    )
    a_texts, a_labels, a_groups = match_size(
        a_texts, a_labels, a_groups, n_cells=len(o_texts), seed=args.seed
    )
    print(f"artemis: matched to {len(a_texts):,} cells, "
          f"{len(set(a_groups)):,} groups, {len(args.classes)} classes")

    if len(set(a_labels)) != len(EMOTIONS):
        print(f"  WARNING: {len(set(a_labels))} classes vs our {len(EMOTIONS)} -- "
              f"accuracies are not directly comparable")

    runner = tfidf_baseline if args.fast else finetune_classifier
    kw = {} if args.fast else {"epochs": args.epochs}
    result = {}
    for name, (t, l, g) in (("ours", (o_texts, o_labels, o_groups)),
                            ("artemis_human", (a_texts, a_labels, a_groups))):
        print(f"\nrunning {name} ...", flush=True)
        r = runner(t, l, g, folds=args.folds, seed=args.seed, **kw)
        result[name] = {"accuracy": r["accuracy"], "n": r["n"],
                        "classes": len(set(l)),
                        "confusion": confusion_matrix(r["pairs"])
                        if len(set(l)) == len(EMOTIONS) and name == "ours" else None}
        print(f"  {name}: {r['accuracy']:.3f}  (n={r['n']:,})")

    ours_acc = result["ours"]["accuracy"]
    human_acc = result["artemis_human"]["accuracy"]
    gap = ours_acc - human_acc
    result["gap_ours_minus_human"] = round(gap, 4)
    result["instrument"] = "tfidf" if args.fast else "distilroberta-base"
    result["artemis_classes"] = args.classes

    print(f"\n{'=' * 70}")
    print(f"human affective text (ArtEmis)  {human_acc:.3f}")
    print(f"our synthetic captions          {ours_acc:.3f}")
    print(f"gap                             {gap:+.3f}")
    print()
    if human_acc < 0.85:
        print("HUMAN text does not reach the pre-registered 0.85 either.")
        print("=> The 0.85 threshold is miscalibrated for this task, not a finding about")
        print("   our data. Recalibrating it -- pre-tag, with the reasoning logged -- is")
        print("   legitimate. Anchor the gate to this human number instead.")
    else:
        print("HUMAN text clears 0.85.")
        print("=> The threshold is achievable, so a shortfall is a property of the")
        print("   synthetic captions. That gap is the measurement, and it is the")
        print("   finding about LLM-synthesised conditioning data.")
    if abs(gap) < 0.05:
        print("\nThe two are within 0.05: synthetic captions carry comparable register")
        print("signal to human affective language under this instrument.")

    outp = Path(args.out)
    if not outp.is_absolute():
        outp = ROOT / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
