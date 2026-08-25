#!/usr/bin/env python
"""The registered artifact ablation, on the instrument that actually scores the study.

    uv run python scripts/artifact_ablation.py                 # both levels
    uv run python scripts/artifact_ablation.py --skip-cv       # arms only, minutes
    uv run python scripts/artifact_ablation.py --skip-arms     # instrument CV only

`metrics.classifier.robustness_reported` registers `artifact_ablation`, and §4.1 of the
preregistration says what it is for: *"Accuracy on punctuation-stripped, lowercased captions
reported alongside raw. A large gap means the classifier reads punctuation, and the stripped
number becomes primary."* The worry is concrete and was measured on the v1 corpus -- captions
differed by exclamation marks and clause structure as much as by tone.

**Why this is being redone.** The ablation in `runs/classifier/report.json` reports a gap of
exactly **0.0000** -- 0.7619 raw against 0.7619 stripped. That is not a clean result, it is a
test that could not fail: it was run on the TF-IDF baseline, whose tokenizer discards
punctuation before it ever sees the text, so stripping punctuation changes the input by
nothing. The registered check has therefore never actually been performed on the frozen
DistilRoBERTa, which does see punctuation and is the instrument every number in the study
comes from.

Two levels, answering different questions
-----------------------------------------
``arms``       the frozen classifier applied to each arm's generated captions, raw and
               stripped. This is the one that matters for the results table: it says
               whether the reported accuracies depend on formatting. Minutes, no training.
``instrument`` the instrument's own 5-fold CV rerun on stripped text. This is what §4.1
               literally describes -- the classifier's accuracy, not the arms' -- and it
               requires retraining, so it is slower.

Both are reported. If the arms' accuracies survive stripping but the instrument's CV does
not, the classifier leans on punctuation in general while the arms happen not to exploit it,
and that distinction belongs in the paper rather than averaged away.

**Nothing is retrained into the frozen instrument.** The CV models are transient; the
classifier in `models/register_classifier` is not touched and its sha256 does not change.
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
from emocap.eval.register_classifier import (finetune_classifier,  # noqa: E402
                                             strip_artifacts)

POOL = ("S_unpaired", "V1_unpaired", "H_unpaired")


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


def arm_generated(arm: str, runs_root: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or not d.name.startswith(f"{arm}-f") or d.name.endswith("-nc"):
            continue
        p = d / "predictions.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                texts.append(r["generated"])
                labels.append(EMOTIONS.index(r["emotion"]))
    return texts, labels


def pool_cells(data_root: Path) -> tuple[list[str], list[int], list[str]]:
    texts, labels, images = [], [], []
    for arm in POOL:
        for line in (data_root / "arms" / f"{arm}.jsonl").read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                texts.append(c["text"])
                labels.append(EMOTIONS.index(c["emotion"]))
                images.append(c["image_id"])
    return texts, labels, images


def acc(pred: list[int], truth: list[int]) -> float:
    return sum(1 for a, b in zip(pred, truth) if a == b) / max(1, len(truth))


def punctuation_rate(texts: list[str]) -> float:
    """Share of captions that change at all under stripping.

    Reported because a zero gap is only meaningful if the transformation did something.
    The TF-IDF version of this ablation had a gap of 0.0000 precisely because nothing it
    could see changed, and that must never be mistaken for robustness again.
    """
    changed = sum(1 for t in texts if strip_artifacts(t) != t)
    return changed / max(1, len(texts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="models/register_classifier")
    ap.add_argument("--data-root", default=None,
                    help="where arms/ lives; defaults to the repo's data/")
    ap.add_argument("--runs-root", default="runs/arms",
                    help="decoded predictions for the arms level; the stripped Kaggle "
                         "export works too")
    ap.add_argument("--baseline-report", default="runs/classifier/report.json",
                    help="where the instrument's RAW cv accuracy was recorded when it was "
                         "frozen; the stripped run is compared against it")
    ap.add_argument("--out", default="results/artifact_ablation.json")
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-arms", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root) if args.data_root else ROOT / "data"
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    clf = Path(args.classifier)
    if not clf.is_absolute():
        clf = ROOT / clf
    report: dict = {
        "why": "the registered ablation in runs/classifier/report.json ran on TF-IDF, whose "
               "tokenizer discards punctuation, so its 0.0000 gap was a test that could not "
               "fail; this runs it on the frozen transformer",
        "classifier_sha256": (clf / "sha256.txt").read_text().strip()
        if (clf / "sha256.txt").exists() else None,
        "strip": "lowercase, replace every non-word non-space character with a space, "
                 "collapse whitespace (emocap.eval.register_classifier.strip_artifacts)",
    }

    # ── level 1: the arms' reported accuracies ───────────────────────────────
    if not args.skip_arms:
        print("── the frozen classifier on each arm's generated captions ──")
        print(f"{'arm':<19}{'raw':>9}{'stripped':>10}{'gap':>9}{'changed':>9}{'n':>9}")
        rows: dict[str, dict] = {}
        arms = sorted({d.name.rsplit("-f", 1)[0] for d in runs_root.iterdir()
                       if d.is_dir() and (d / "predictions.jsonl").exists()})
        for arm in arms:
            texts, labels = arm_generated(arm, runs_root)
            if not texts:
                continue
            raw = acc(predict(texts, clf), labels)
            stripped = acc(predict([strip_artifacts(t) for t in texts], clf), labels)
            rate = punctuation_rate(texts)
            rows[arm] = {"raw": round(raw, 4), "stripped": round(stripped, 4),
                         "gap": round(raw - stripped, 4), "changed_by_strip": round(rate, 4),
                         "n": len(texts)}
            print(f"{arm:<19}{raw:>9.4f}{stripped:>10.4f}{raw - stripped:>+9.4f}"
                  f"{rate:>9.1%}{len(texts):>9,}", flush=True)
        report["arms"] = rows
        gaps = [v["gap"] for v in rows.values()]
        report["arms_summary"] = {
            "max_abs_gap": round(max(abs(g) for g in gaps), 4),
            "mean_gap": round(st.mean(gaps), 4),
            "reading": "a large positive gap would mean the reported accuracies lean on "
                       "punctuation and the stripped column becomes primary"}
        print(f"\n  largest |gap| across arms {report['arms_summary']['max_abs_gap']:.4f}"
              f"   mean {report['arms_summary']['mean_gap']:+.4f}")

    # ── level 2: the instrument's own accuracy ───────────────────────────────
    if not args.skip_cv:
        texts, labels, images = pool_cells(data_root)
        rate = punctuation_rate(texts)
        print(f"\n── the instrument's own 5-fold CV, stripped text ──")
        print(f"  pool {len(texts):,} cells, {rate:.1%} changed by stripping")
        if rate < 0.05:
            print("  WARNING stripping barely changes this pool; a small gap below would "
                  "say little.")
        t0 = time.time()
        cv = finetune_classifier(
            [strip_artifacts(t) for t in texts], labels, images, folds=5, seed=42,
            progress=lambda f, a: print(f"  fold {f+1}/5  running {a:.4f}  "
                                        f"[{(time.time()-t0)/60:.0f} min]", flush=True))
        bp = Path(args.baseline_report)
        if not bp.is_absolute():
            bp = ROOT / bp
        prior = json.loads(bp.read_text())
        raw_cv = prior["cv"]["accuracy"]
        report["instrument"] = {
            "raw_cv_accuracy": raw_cv,
            "stripped_cv_accuracy": round(cv["accuracy"], 4),
            "gap": round(raw_cv - cv["accuracy"], 4),
            "n": cv["n"], "changed_by_strip": round(rate, 4),
            "note": "raw value is the CV accuracy recorded when the instrument was frozen; "
                    "the stripped run retrains transient models and does NOT touch "
                    "models/register_classifier"}
        print(f"\n  instrument CV  raw {raw_cv:.4f}  stripped {cv['accuracy']:.4f}  "
              f"gap {raw_cv - cv['accuracy']:+.4f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
