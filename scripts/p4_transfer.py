#!/usr/bin/env python
"""P4, asymmetric transfer: two more judges, and what each one makes of the other's data.

    uv run python scripts/p4_transfer.py
    uv run python scripts/p4_transfer.py --skip-cv          # judges only, no own-accuracy
    uv run python scripts/p4_transfer.py --include-paired   # also score the big S arms

P4 as registered: *"Models trained on S score lower under an H-trained judge than H-trained
models do under an S-trained judge (asymmetric transfer)."* Falsified if symmetric, or
reversed. It is about **judges**, not generators, so no arm is retrained -- two extra
classifiers are trained and every arm is scored again under each.

The frozen instrument in `models/register_classifier` is untouched. It stays the primary
metric; these two are a separate, single-provenance pair and live in `models/judge_S` and
`models/judge_H`.

Why a raw cross-provenance accuracy would not answer P4
-------------------------------------------------------
Suppose the H-judge scores S captions at 0.30. That is uninterpretable on its own: it could
mean transfer fails, or it could mean the H-judge is simply a weak classifier. The two are
distinguished only by knowing what each judge scores on its OWN corpus, so this measures
that first, by 5-fold CV within each corpus, and reports transfer as a **drop from
own-corpus accuracy**. An asymmetry claim needs the drops to differ, not the raw numbers.

Two levels, because P4's wording and its mechanism are not the same thing
------------------------------------------------------------------------
``corpus``     each judge applied to the other's *reference* captions -- the human-written
               and Gemini-written text itself. This is pure judge asymmetry, and it is the
               control: if the corpora already transfer asymmetrically, anything measured on
               generated text inherits it and cannot be attributed to the captioners.
``generated``  each judge applied to the arms' *generated* captions. This is what P4
               literally says, and it is the reported test.

The P4 pair is ``S_unpaired`` against ``H_unpaired`` -- matched on cell count and structure,
which is the only pair where "trained on S" and "trained on H" differ by provenance alone.
Using ``S_paired25`` would confound provenance with a 46x difference in gradient steps.
Other arms are reported for context and carry no P4 verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.eval.register_classifier import (confusion_matrix,  # noqa: E402
                                             finetune_classifier,
                                             train_final_classifier)

#: Judge name -> the arm whose reference captions train it. Single-provenance on purpose:
#: that is what makes them judges "trained on S" and "trained on H".
JUDGES = {"S": "S_unpaired", "H": "H_unpaired"}

#: Scored under both judges by default. Matched at 4,390 cells and 1 caption/image, so a
#: difference between them is provenance rather than data volume.
ARMS = ("S_unpaired", "H_unpaired", "V1_unpaired")
PAIRED = ("S_paired5", "V1_paired5", "S_paired25")


def arm_cells(arm: str) -> tuple[list[str], list[int], list[str]]:
    texts, labels, images = [], [], []
    for line in (ROOT / "data/arms" / f"{arm}.jsonl").read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            texts.append(c["text"])
            labels.append(EMOTIONS.index(c["emotion"]))
            images.append(c["image_id"])
    return texts, labels, images


def generated(arm: str) -> tuple[list[str], list[int]]:
    """Every fold's held-out generated captions for one arm. Controls excluded."""
    texts, labels = [], []
    for d in sorted((ROOT / "runs/arms").iterdir()):
        if not d.is_dir() or not d.name.startswith(f"{arm}-f") or d.name.endswith("-nc"):
            continue
        p = d / "predictions.jsonl"
        if not p.exists():
            raise SystemExit(f"{p} is missing (gitignored) -- pull the run first")
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                texts.append(r["generated"])
                labels.append(EMOTIONS.index(r["emotion"]))
    return texts, labels


def predict(texts: list[str], clf_dir: Path, batch: int = 128) -> list[int]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(clf_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(clf_dir)).to(device)
    model.eval()
    out: list[int] = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], truncation=True, max_length=48,
                      padding=True, return_tensors="pt").to(device)
            out.extend(int(i) for i in model(**enc).logits.argmax(-1).cpu())
    return out


def accuracy(pred: list[int], truth: list[int]) -> float:
    return sum(1 for a, b in zip(pred, truth) if a == b) / max(1, len(truth))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/p4_transfer.json")
    ap.add_argument("--models", default="models")
    ap.add_argument("--skip-cv", action="store_true",
                    help="skip each judge's own-corpus CV -- transfer then has no baseline "
                         "to be a drop FROM, and P4 cannot be evaluated")
    ap.add_argument("--include-paired", action="store_true")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain judges even if they already exist on disk")
    args = ap.parse_args()

    report: dict = {"prediction": "P4 -- models trained on S score lower under an H-trained "
                                 "judge than H-trained models do under an S-trained judge",
                    "falsified_if": "symmetric, or reversed",
                    "p4_pair": ["S_unpaired", "H_unpaired"],
                    "p4_pair_why": "matched on cell count and structure, so the difference "
                                   "is provenance and not volume or gradient steps",
                    "frozen_instrument_untouched": True,
                    "judges": {}}

    corpora = {k: arm_cells(a) for k, a in JUDGES.items()}
    for k, a in JUDGES.items():
        print(f"judge {k}: {len(corpora[k][0]):,} cells from {a}")

    # ── each judge's own accuracy, then the judge itself ─────────────────────
    for k, arm in JUDGES.items():
        texts, labels, images = corpora[k]
        entry: dict = {"trained_on": arm, "cells": len(texts)}
        # The CV is the expensive half and it is deterministic, so its result is cached the
        # moment it exists. A first attempt at this script lost 24 minutes of completed
        # judge-S cross-validation when the process wedged on MPS partway through judge H --
        # the trained judge had been saved, the measurement had not.
        cache = ROOT / f"results/p4_cv_{k}.json"
        if not args.skip_cv and cache.exists() and not args.retrain:
            entry.update(json.loads(cache.read_text()))
            print(f"\njudge {k}: reusing cached own-corpus CV "
                  f"{entry['own_cv_accuracy']:.4f} from {cache.relative_to(ROOT)}")
        elif not args.skip_cv:
            t0 = time.time()
            print(f"\njudge {k}: 5-fold CV by image on its own corpus ...", flush=True)
            cv = finetune_classifier(
                texts, labels, images, folds=5, seed=42,
                progress=lambda f, acc: print(f"  fold {f+1}/5  running {acc:.4f}  "
                                              f"[{(time.time()-t0)/60:.0f} min]", flush=True))
            entry["own_cv_accuracy"] = round(cv["accuracy"], 4)
            entry["own_cv_n"] = cv["n"]
            entry["own_confusion"] = confusion_matrix(cv["pairs"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(
                {k2: entry[k2] for k2 in ("own_cv_accuracy", "own_cv_n", "own_confusion")},
                indent=2))
            print(f"  own-corpus accuracy {cv['accuracy']:.4f}  (n={cv['n']:,})"
                  f"  -> cached")

        out_dir = Path(args.models) / f"judge_{k}"
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        if out_dir.exists() and not args.retrain:
            print(f"judge {k}: reusing {out_dir}")
            entry["sha256"] = (out_dir / "sha256.txt").read_text().strip() \
                if (out_dir / "sha256.txt").exists() else None
        else:
            print(f"judge {k}: training the deployed judge on all {len(texts):,} cells ...",
                  flush=True)
            fin = train_final_classifier(texts, labels, out_dir=out_dir)
            entry.update({"sha256": fin["sha256"], "epochs": fin["epochs"]})
            print(f"  sha256 {fin['sha256'][:16]}")
        entry["path"] = str(out_dir.relative_to(ROOT))
        report["judges"][k] = entry

    paths = {k: ROOT / report["judges"][k]["path"] for k in JUDGES}

    # ── level 1: the corpora themselves ──────────────────────────────────────
    # The control. If the reference captions already transfer asymmetrically, anything
    # measured on generated text inherits that and says nothing about the captioners.
    print("\n── corpus-level transfer (reference captions) ──")
    report["corpus_transfer"] = {}
    for judge in JUDGES:
        for target in JUDGES:
            if judge == target:
                continue
            texts, labels, _ = corpora[target]
            acc = accuracy(predict(texts, paths[judge]), labels)
            report["corpus_transfer"][f"{judge}_judge_on_{target}_corpus"] = round(acc, 4)
            print(f"  {judge}-judge on {target} corpus   {acc:.4f}")

    # ── level 2: generated captions, which is what P4 is about ───────────────
    print("\n── generated-caption transfer ──")
    arms = list(ARMS) + (list(PAIRED) if args.include_paired else [])
    report["generated"] = {}
    for arm in arms:
        texts, labels = generated(arm)
        row = {"n": len(texts)}
        for judge in JUDGES:
            row[f"{judge}_judge"] = round(accuracy(predict(texts, paths[judge]), labels), 4)
        report["generated"][arm] = row
        cells = "  ".join(f"{j}-judge {row[f'{j}_judge']:.4f}" for j in JUDGES)
        print(f"  {arm:<14} n={len(texts):>7,}   {cells}")

    # ── the P4 verdict ───────────────────────────────────────────────────────
    # Both sides expressed as a DROP from the judge that shares the captions' provenance,
    # so a weak judge cannot masquerade as failed transfer.
    g = report["generated"]
    s_native = g["S_unpaired"]["S_judge"]
    s_foreign = g["S_unpaired"]["H_judge"]
    h_native = g["H_unpaired"]["H_judge"]
    h_foreign = g["H_unpaired"]["S_judge"]
    drop_s = s_native - s_foreign     # S captions losing under the H judge
    drop_h = h_native - h_foreign     # H captions losing under the S judge

    verdict = {
        "S_generated_under_own_judge": s_native,
        "S_generated_under_H_judge": s_foreign,
        "S_drop": round(drop_s, 4),
        "H_generated_under_own_judge": h_native,
        "H_generated_under_S_judge": h_foreign,
        "H_drop": round(drop_h, 4),
        "asymmetry_S_minus_H": round(drop_s - drop_h, 4),
        "p4_direction_holds": bool(drop_s > drop_h),
        "reading": "P4 predicts S captions lose MORE under a foreign judge than H captions "
                   "do. asymmetry > 0 is the predicted direction; <= 0 falsifies it as "
                   "symmetric or reversed. No magnitude was registered for P4, so the "
                   "direction and the size are both reported and neither is thresholded.",
    }
    report["p4_verdict"] = verdict
    print("\n── P4 ──")
    print(f"  S captions: own judge {s_native:.4f} -> H judge {s_foreign:.4f}   "
          f"drop {drop_s:+.4f}")
    print(f"  H captions: own judge {h_native:.4f} -> S judge {h_foreign:.4f}   "
          f"drop {drop_h:+.4f}")
    print(f"  asymmetry (S drop - H drop) {drop_s - drop_h:+.4f}   -> "
          f"{'direction holds' if drop_s > drop_h else 'FALSIFIED: symmetric or reversed'}")
    print("  No magnitude was registered for P4; the direction and size are both reported.")

    if not args.skip_cv:
        own = {k: report["judges"][k].get("own_cv_accuracy") for k in JUDGES}
        if all(v is not None for v in own.values()):
            print(f"\n  judge quality, own-corpus CV: " +
                  "  ".join(f"{k} {v:.4f}" for k, v in own.items()))
            if abs(own["S"] - own["H"]) > 0.10:
                print("  NOTE the two judges differ by more than 10 points on their own "
                      "corpora, so they are not equally good instruments and the transfer "
                      "numbers above must be read as drops, never as raw accuracies.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
