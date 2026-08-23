#!/usr/bin/env python
"""Score one run's predictions with the frozen instrument. The unit of the analysis.

    uv run python scripts/score_arm.py --run runs/arms/S_unpaired-f0
    uv run python scripts/score_arm.py --run runs/arms/S_unpaired-f0 --classifier models/register_classifier

Produces, for one (arm, fold):

    accuracy        the PRIMARY metric -- frozen classifier, top-1, on generated captions
    anchor          the top-25 keyword rule recomputed ON THESE SAME CAPTIONS at this n
    margin          accuracy - anchor. This, not accuracy, is what the study claims.
    ci              cluster bootstrap by image_id, 10,000 resamples (from the lock)

**Why the margin and not the accuracy.** §4 establishes that a large share of the primary
metric is reachable from register vocabulary with no model at all, and that the share is
sample-size dependent -- so an accuracy quoted without its own anchor at its own n is
uninterpretable, and three findings were already retracted for exactly that. This script
therefore refuses to report an accuracy without recomputing the anchor beside it.

**The classifier is loaded, never trained.** Its sha256 is recorded in the output so a
number can be traced to the instrument that produced it. A run scored by a differently
trained classifier is not comparable to one scored by this one, and the hash is what makes
that checkable rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.eval.anchors import keyword_rule_accuracy  # noqa: E402
from emocap.eval.bootstrap import cluster_bootstrap_mean  # noqa: E402
from emocap.eval.register_classifier import confusion_matrix  # noqa: E402
from emocap.runtime import load_config  # noqa: E402


def predict(texts: list[str], clf_dir: Path, batch: int = 128) -> list[int]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(clf_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(clf_dir)).to(dev).eval()
    out: list[int] = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], truncation=True, max_length=48,
                  padding="max_length", return_tensors="pt").to(dev)
        with torch.no_grad():
            out += mdl(**enc).logits.argmax(-1).cpu().tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a runs/arms/<tag> directory")
    ap.add_argument("--classifier", default="models/register_classifier")
    ap.add_argument("--out", default=None, help="defaults to <run>/score.json")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    clf = Path(args.classifier)
    if not clf.is_absolute():
        clf = ROOT / clf
    preds = [json.loads(l) for l in (run / "predictions.jsonl").read_text().splitlines()
             if l.strip()]
    if not preds:
        raise SystemExit(f"{run} has no predictions")

    lock = load_config("prereg.lock")
    boot = lock["metrics"]["bootstrap"]

    # ── primary metric ───────────────────────────────────────────────────────
    texts = [p["generated"] for p in preds]
    truth = [EMOTIONS.index(p["emotion"]) for p in preds]
    got = predict(texts, clf)
    correct = [float(a == b) for a, b in zip(got, truth)]
    acc = sum(correct) / len(correct)

    # Clustered by image, because one image's register cells are not independent.
    ci = cluster_bootstrap_mean(correct, [p["image_id"] for p in preds],
                                n_resamples=boot["resamples"], ci=boot["ci"], seed=42)

    # ── the anchor, on THESE captions, at THIS n ─────────────────────────────
    by_img: dict[str, dict] = {}
    for p in preds:
        by_img.setdefault(p["image_id"], {"image_id": p["image_id"], "captions": {}})
        by_img[p["image_id"]]["captions"][p["emotion"]] = p["generated"]
    anchor = keyword_rule_accuracy(list(by_img.values()))

    # ── visual-dependence probe (§7 exclusion criterion) ─────────────────────
    # A run that writes the same caption with the image blanked was never using the image.
    # The prereg requires the captions to "change substantially" but sets no threshold, so
    # this reports the evidence and writes the captions out for human review. It does NOT
    # decide pass/fail -- inventing that number after seeing results is the failure this
    # project has already had four times.
    novis_path = run / "predictions_novis.jsonl"
    visual = None
    if novis_path.exists():
        nov = [json.loads(l) for l in novis_path.read_text().splitlines() if l.strip()]
        nov_texts = [p["generated"] for p in nov]
        nov_truth = [EMOTIONS.index(p["emotion"]) for p in nov]
        nov_got = predict(nov_texts, clf)
        nov_acc = sum(float(a == b) for a, b in zip(nov_got, nov_truth)) / len(nov)

        # Paired by (image, register) rather than by position, so a reordered or partial
        # probe file compares like with like instead of silently misaligning.
        blanked = {(p["image_id"], p["emotion"]): p["generated"] for p in nov}
        pairs = [(p, blanked[(p["image_id"], p["emotion"])]) for p in preds
                 if (p["image_id"], p["emotion"]) in blanked]
        identical = sum(1 for a, b in pairs if a["generated"].strip() == b.strip())

        # The probe runs on a subsample, so the drop is measured against the with-image
        # accuracy ON THE SAME CELLS. Comparing a 500-cell blanked accuracy against the
        # full-set accuracy would fold sampling noise into a number read as an effect.
        by_key = {(p["image_id"], p["emotion"]): c for p, c in zip(preds, correct)}
        same_cells = [by_key[(a["image_id"], a["emotion"])] for a, _ in pairs]
        acc_same = sum(same_cells) / max(1, len(same_cells))

        sample = [{"image_id": a["image_id"], "emotion": a["emotion"],
                   "with_image": a["generated"], "blanked": b} for a, b in pairs[:40]]
        visual = {"n_paired": len(pairs), "identical": identical,
                  "identical_rate": round(identical / max(1, len(pairs)), 4),
                  "accuracy_with_image_same_cells": round(acc_same, 4),
                  "accuracy_blanked": round(nov_acc, 4),
                  "accuracy_drop": round(acc_same - nov_acc, 4),
                  "unique_captions_blanked": len(set(nov_texts)),
                  "sample": sample}

        # A plain-text side-by-side for the reviewers who make the call.
        lines = [f"visual-dependence probe -- {run.name}",
                 f"{identical:,}/{len(pairs):,} captions ({visual['identical_rate']:.1%}) "
                 f"unchanged when the image is blanked",
                 f"accuracy on these cells: {acc_same:.4f} with image, "
                 f"{nov_acc:.4f} blanked ({visual['accuracy_drop']:+.4f})", ""]
        for a, b in pairs[:40]:
            same = "IDENTICAL" if a["generated"].strip() == b.strip() else ""
            lines += [f"{a['image_id']}  [{a['emotion']}]  {same}",
                      f"   with image: {a['generated']}",
                      f"   blanked   : {b}", ""]
        (run / "probe_review.txt").write_text("\n".join(lines))

    manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    result = {
        "run": run.name,
        "arm": manifest.get("config", {}).get("arm"),
        "fold": manifest.get("config", {}).get("fold"),
        "negative_control": manifest.get("config", {}).get("negative_control"),
        "classifier_sha256": (clf / "sha256.txt").read_text().strip()
        if (clf / "sha256.txt").exists() else None,
        "n_cells": len(preds),
        "n_images": len(by_img),
        "accuracy": round(acc, 4),
        "ci": ci,
        "anchor": anchor["accuracy"],
        "anchor_n_cells": anchor["n_cells"],
        "margin_over_anchor": round(acc - anchor["accuracy"], 4),
        "chance": round(1 / len(EMOTIONS), 4),
        "confusion": confusion_matrix(list(zip(truth, got))),
        "empty_captions": sum(1 for t in texts if not t.strip()),
        "mean_words": round(sum(len(t.split()) for t in texts) / len(texts), 1),
        "unique_captions": len(set(texts)),
        "visual_dependence": visual,
    }
    out = Path(args.out) if args.out else run / "score.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"{result['run']}   n={result['n_cells']:,} cells over {result['n_images']:,} images")
    print(f"  accuracy   {acc:.4f}   95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]")
    print(f"  anchor     {anchor['accuracy']:.4f}   (recomputed on these captions, n={anchor['n_cells']:,})")
    print(f"  MARGIN     {acc - anchor['accuracy']:+.4f}   <- what the study claims")
    print(f"  chance     {result['chance']:.4f}")
    print(f"  recall     " + "  ".join(
        f"{k[:3]} {v}" for k, v in result["confusion"]["recall"].items()))
    print(f"  captions   {result['unique_captions']:,} unique, "
          f"{result['mean_words']} mean words, {result['empty_captions']} empty")
    if visual:
        print(f"  PROBE      {visual['identical_rate']:.1%} of captions unchanged with the "
              f"image blanked ({visual['identical']:,}/{visual['n_paired']:,})")
        print(f"             accuracy blanked {visual['accuracy_blanked']:.4f} "
              f"({visual['accuracy_drop']:+.4f})  -> reviewers decide, see probe_review.txt")
    else:
        print("  PROBE      MISSING -- no predictions_novis.jsonl. This run cannot be "
              "checked against the \u00a77 exclusion criterion.")
    if result["negative_control"]:
        cap = lock["gates"]["manipulation_check"]["negative_control_accuracy_max"]
        verdict = "PASSES" if acc <= cap else "FAILS -- results not interpretable"
        print(f"  GATE       negative control {acc:.4f} vs max {cap}  {verdict}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
