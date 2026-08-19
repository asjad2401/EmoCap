#!/usr/bin/env python
"""Run the pre-registered anchors and gates against a caption store, early.

`docs/preregistration.md` §6 defines a **ceiling** gate: if the register classifier
cannot recover the intended emotion from the generated captions at >= 0.85, the data
lacks separable tone and the study halts. That assumption otherwise stays untested until
the corpus exists and both model tracks are built -- so this runs it on the audit sample
first, where being wrong costs an afternoon rather than the project.

    uv run python scripts/check_gates.py data/generated/captions_audit_v5.jsonl
    uv run python scripts/check_gates.py <store> --fast     # TF-IDF only, seconds

**This is an early indicator, not the official gate.** The pre-registered ceiling is
measured on the full test split with a classifier trained on the training split. Here
there are ~1,200 cells over ~50 images, cross-validated by image. Read it as "is the
signal plausibly there", not as the gate itself. A small sample makes the transformer
number pessimistic (it has little to learn from) and the TF-IDF number optimistic
(vocabulary overfits a narrow image set), so treat them as a bracket.

What is NOT runnable yet, and why:

  manipulation check   §6 wants V0's accuracy <= 0.25. V0 is an unconditioned *trained
                       model* and none exists. The proxy here is the classifier applied
                       to the human source captions, which carry no register at all: if
                       it confidently assigns emotions to those, it is reading something
                       other than tone.
  negative control     Condition C needs a trained model.
  visual dependence    Needs a trained model.
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
from emocap.eval.anchors import keyword_rule_accuracy  # noqa: E402
from emocap.eval.register_classifier import (  # noqa: E402
    build_dataset,
    confusion_matrix,
    finetune_classifier,
    strip_artifacts,
    tfidf_baseline,
)
from emocap.runtime import load_config  # noqa: E402


def load(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--fast", action="store_true", help="TF-IDF only, skip fine-tuning")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-pause", type=float, default=0.0,
                    help="seconds to idle between folds, for a real cool-down window")
    ap.add_argument("--duty", type=float, default=1.0,
                    help="GPU duty cycle. 0.33 = compute a third of the time, so ~3x the "
                         "wall clock at ~1/3 the sustained thermal load. The result is "
                         "bit-identical; only the spacing changes.")
    ap.add_argument("--out", help="write the full result as JSON here")
    ap.add_argument("--learning-curve", action="store_true",
                    help="train on 25/50/75/100%% of images and report the slope. Decides "
                         "whether a below-ceiling result is a data-starved classifier or a "
                         "real property of the captions")
    args = ap.parse_args()

    path = Path(args.store)
    if not path.is_absolute():
        path = ROOT / path
    records = load(path)
    if not records:
        sys.exit(f"{path} has no records")

    lock = load_config("prereg.lock")
    ceiling_min = lock["gates"]["ceiling_min"]
    v0_max = lock["gates"]["manipulation_check"]["v0_emotion_accuracy_max"]

    texts, labels, images = build_dataset(records)
    chance = 1 / len(EMOTIONS)
    print(f"store        {path.name}")
    print(f"cells        {len(texts):,} over {len(set(images)):,} images")
    print(f"per register {dict(Counter(EMOTIONS[l] for l in labels))}")
    print(f"chance       {chance:.3f}   pre-registered ceiling >= {ceiling_min}\n")

    result: dict = {"store": path.name, "cells": len(texts), "images": len(set(images)),
                    "chance": chance, "ceiling_min": ceiling_min}

    # ── the lexical-shortcut anchor, for context on everything below ────────
    anchor = keyword_rule_accuracy(records)
    result["lexical_shortcut"] = anchor
    print(f"lexical shortcut anchor      {anchor['accuracy']:.3f}   "
          f"(top-25 keyword rule, no model)")

    # ── TF-IDF baseline: a stronger relative of the anchor ──────────────────
    tf = tfidf_baseline(texts, labels, images, folds=args.folds, seed=args.seed)
    result["tfidf"] = {k: v for k, v in tf.items() if k != "pairs"}
    print(f"TF-IDF + logistic regression {tf['accuracy']:.3f}   n={tf['n']}")

    # ── floor: labels randomly reassigned, must collapse to chance ──────────
    shuffled = list(labels)
    random.Random(args.seed).shuffle(shuffled)
    floor = tfidf_baseline(texts, shuffled, images, folds=args.folds, seed=args.seed)
    result["floor_tfidf"] = {"accuracy": floor["accuracy"], "n": floor["n"]}
    print(f"floor (labels shuffled)      {floor['accuracy']:.3f}   expected ~{chance:.2f}")

    # ── artifact ablation: punctuation stripped, lowercased ─────────────────
    stripped = [strip_artifacts(t) for t in texts]
    abl = tfidf_baseline(stripped, labels, images, folds=args.folds, seed=args.seed)
    result["artifact_ablation_tfidf"] = {"accuracy": abl["accuracy"]}
    print(f"artifact ablation (TF-IDF)   {abl['accuracy']:.3f}   "
          f"gap {tf['accuracy'] - abl['accuracy']:+.3f}")

    ceiling_source = tf
    ceiling_label = "TF-IDF"

    if not args.fast:
        print(f"\nfine-tuning {args.folds} x distilroberta-base "
              f"({args.epochs} epochs each)...", flush=True)

        def show(fold: int, running: float) -> None:
            print(f"  fold {fold + 1}/{args.folds}  running accuracy {running:.3f}",
                  flush=True)

        ft = finetune_classifier(texts, labels, images, folds=args.folds,
                                 seed=args.seed, epochs=args.epochs,
                                 duty=args.duty, fold_pause=args.fold_pause,
                                 progress=show)
        result["distilroberta"] = {k: v for k, v in ft.items() if k != "pairs"}
        print(f"\nDistilRoBERTa (pre-registered instrument)  {ft['accuracy']:.3f}   "
              f"n={ft['n']}  device={ft['device']}")
        cm = confusion_matrix(ft["pairs"])
        result["confusion"] = cm
        print("\nper-register recall:")
        for reg, r in cm["recall"].items():
            print(f"  {reg:<9} {r:.3f}")
        print("\nconfusion (rows = intended register):")
        header = "           " + "".join(f"{e[:4]:>7}" for e in EMOTIONS)
        print(header)
        for reg, row in cm["row_normalised"].items():
            print(f"  {reg:<9}" + "".join(f"{row[e]:>7.2f}" for e in EMOTIONS))
        ceiling_source = ft
        ceiling_label = "DistilRoBERTa"

    # ── learning curve: is a low ceiling the sample, or the data? ───────────
    if args.learning_curve:
        import random as _r
        print(f"\n{'=' * 68}\nlearning curve (3 folds per point)", flush=True)
        curve = []
        unique = sorted(set(images))
        for frac in (0.25, 0.5, 0.75, 1.0):
            keep = set(_r.Random(args.seed).sample(unique, max(4, int(len(unique) * frac))))
            idx = [i for i, img in enumerate(images) if img in keep]
            sub_t = [texts[i] for i in idx]
            sub_l = [labels[i] for i in idx]
            sub_i = [images[i] for i in idx]
            fn = tfidf_baseline if args.fast else finetune_classifier
            kw = {} if args.fast else {"epochs": args.epochs}
            r = fn(sub_t, sub_l, sub_i, folds=3, seed=args.seed, **kw)
            curve.append({"fraction": frac, "images": len(keep), "cells": len(idx),
                          "accuracy": r["accuracy"]})
            print(f"  {int(frac * 100):>3}%  {len(keep):>3} images  "
                  f"{len(idx):>5} cells   accuracy {r['accuracy']:.3f}", flush=True)
        result["learning_curve"] = curve
        if len(curve) >= 2:
            slope = curve[-1]["accuracy"] - curve[-2]["accuracy"]
            print(f"\n  slope over the last step: {slope:+.3f}")
            print("  still climbing -- more data should raise the ceiling"
                  if slope > 0.02 else
                  "  flattening -- more data is unlikely to reach the pre-registered ceiling")

    # ── manipulation-check proxy: the classifier on unconditioned human text ─
    sources = sorted({str(r.get("source_caption", "")).strip() for r in records
                      if str(r.get("source_caption", "")).strip()})
    result["manipulation_proxy"] = {
        "n_human_captions": len(sources),
        "note": "human Flickr8k captions carry no register; a real V0 check needs a "
                "trained unconditioned model, which does not exist yet",
    }

    # ── verdict ────────────────────────────────────────────────────────────
    acc = ceiling_source["accuracy"]
    margin = acc - anchor["accuracy"]
    print(f"\n{'=' * 68}")
    print(f"CEILING ({ceiling_label})      {acc:.3f}   vs pre-registered >= {ceiling_min}")
    print(f"margin over keyword rule   {margin:+.3f}   "
          f"({anchor['accuracy']:.3f} needs no model at all)")
    if acc >= ceiling_min:
        print("\nPASSES the ceiling on this sample. The generated captions carry "
              "separable tone.")
    else:
        n_img = len(set(images))
        print(f"\nBELOW the pre-registered ceiling of {ceiling_min} on this sample "
              f"({n_img:,} images, {len(texts):,} cells).")
        print("Not the official gate, which trains on the full train split. Read the "
              "learning curve\nbefore concluding: at this size the classifier may still "
              "be data-limited.")
    print(f"\nNot runnable until a model exists: manipulation check (V0 <= {v0_max}), "
          f"negative\ncontrol (condition C), visual-dependence probe.")
    result["ceiling"] = {"accuracy": acc, "instrument": ceiling_label,
                         "passes": acc >= ceiling_min,
                         "margin_over_anchor": round(margin, 4)}

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
