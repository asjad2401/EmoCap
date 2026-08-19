#!/usr/bin/env python
"""Corpus effect on register recoverability, holding groundedness fixed.

The bin-matched comparison and the regression adjustment disagreed sharply -- matched said
human is 9.6 points better, regression said ours is 1.3 points better. The cause is that
the regression used TF-IDF correctness, and TF-IDF is blind to exactly the non-lexical
signal the two corpora differ on: the transformer gains +0.210 over TF-IDF on human
captions but only +0.136 on ours. A bag-of-ngrams adjustment therefore cannot arbitrate.

This redoes it with the pre-registered instrument, and adds two things the earlier version
lacked:

* **A curvature check.** The linear-in-logit assumption was asserted, not tested. A
  quadratic term in CLIPScore is fitted and reported; if it matters, the linear adjustment
  is not trustworthy.
* **Fitted accuracy across the whole CLIPScore range, per corpus**, rather than a single
  number at the pooled mean. If the corpus gap varies with groundedness, one number hides
  it -- and the bin-matched sample sits in the tail of both distributions, so whether the
  gap is constant is the question that decides if that result generalises.

    uv run python scripts/clipscore_regression.py

Per-caption correctness is cached in runs/clipscore/correctness.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.eval.register_classifier import finetune_classifier, folds_by_image  # noqa: E402

SCORES = ROOT / "runs/clipscore/scores.json"
CORRECT = ROOT / "runs/clipscore/correctness.json"


def per_caption_correctness(rows, *, folds: int, seed: int, epochs: int) -> list[int]:
    """0/1 per caption from the pre-registered instrument, folds split by image.

    Every caption is held out exactly once regardless of ``folds``, so coverage is
    complete; ``folds`` only sets how much training data each fine-tune sees.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from emocap.data.prompt import EMOTIONS

    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    images = [r["image"] for r in rows]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("distilroberta-base")
    enc = tok(texts, truncation=True, max_length=48, padding="max_length",
              return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    ys = torch.tensor(labels)

    out = [-1] * len(rows)
    for fi, (tr, te) in enumerate(folds_by_image(images, folds=folds, seed=seed)):
        torch.manual_seed(seed + fi)
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilroberta-base", num_labels=len(EMOTIONS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
        loader = DataLoader(TensorDataset(ids[tr], mask[tr], ys[tr]),
                            batch_size=16, shuffle=True)
        model.train()
        for _ in range(epochs):
            for a, b, y in loader:
                opt.zero_grad()
                model(input_ids=a.to(device), attention_mask=b.to(device),
                      labels=y.to(device)).loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            for s in range(0, len(te), 64):
                ch = te[s:s + 64]
                pred = model(input_ids=ids[ch].to(device),
                             attention_mask=mask[ch].to(device)).logits.argmax(-1).cpu()
                for i, p in zip(ch, pred.tolist()):
                    out[i] = 1 if labels[i] == p else 0
        print(f"    fold {fi + 1}/{folds} done", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not SCORES.exists():
        sys.exit(f"{SCORES} missing -- run scripts/clipscore_matched.py first")
    data = json.loads(SCORES.read_text())

    if CORRECT.exists():
        cache = json.loads(CORRECT.read_text())
        print(f"using cached correctness from {CORRECT}")
    else:
        cache = {}
        for name in ("human", "ours"):
            rows = data["corpora"][name]
            print(f"  {name}: {len(rows):,} captions, {args.folds} fine-tunes", flush=True)
            cache[name] = per_caption_correctness(
                rows, folds=args.folds, seed=args.seed, epochs=args.epochs)
        CORRECT.write_text(json.dumps(cache))
        print(f"cached to {CORRECT}")

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X, y, corpus = [], [], []
    for flag, name in ((0.0, "human"), (1.0, "ours")):
        rows = data["corpora"][name]
        for r, c in zip(rows, cache[name]):
            if c < 0:
                continue
            X.append([r["clipscore"], flag])
            y.append(c)
            corpus.append(name)

    X = np.array(X)
    y = np.array(y)
    for name in ("human", "ours"):
        sel = [i for i, c in enumerate(corpus) if c == name]
        print(f"{name:<7} n={len(sel):,}  raw accuracy {y[sel].mean():.4f}  "
              f"CLIPScore mean {X[sel, 0].mean():.4f}")

    lin = LogisticRegression(max_iter=3000).fit(X, y)
    b_clip, b_corp = lin.coef_[0]
    print(f"\nlinear model:  CLIPScore {b_clip:+.3f}   corpus=ours {b_corp:+.3f}")

    # Curvature check: does a quadratic term in CLIPScore change the corpus coefficient?
    Xq = np.column_stack([X[:, 0], X[:, 0] ** 2, X[:, 1]])
    quad = LogisticRegression(max_iter=3000).fit(Xq, y)
    qc, qc2, q_corp = quad.coef_[0]
    print(f"quadratic:     CLIPScore {qc:+.3f}  CLIPScore^2 {qc2:+.3f}  "
          f"corpus=ours {q_corp:+.3f}")
    shift = abs(q_corp - b_corp)
    print(f"  corpus coefficient moves {shift:.3f} when curvature is allowed -> "
          f"{'linear form is NOT safe' if shift > 0.15 else 'linear form is adequate'}")

    def p(model, clip, flag, quadratic=False):
        v = ([clip, clip ** 2, flag] if quadratic else [clip, flag])
        z = model.intercept_[0] + float(np.dot(model.coef_[0], v))
        return 1 / (1 + math.exp(-z))

    print(f"\nfitted accuracy across the CLIPScore range (linear model):")
    print(f"  {'CLIPScore':>10}{'human':>9}{'ours':>9}{'gap':>9}   support")
    rows_out = []
    for clip in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        ph, po = p(lin, clip, 0.0), p(lin, clip, 1.0)
        nh = int(((X[:, 1] == 0) & (abs(X[:, 0] - clip) < 0.025)).sum())
        no = int(((X[:, 1] == 1) & (abs(X[:, 0] - clip) < 0.025)).sum())
        tag = "both" if nh > 30 and no > 30 else ("human only" if nh > 30 else
                                                 "ours only" if no > 30 else "neither")
        print(f"  {clip:>10.2f}{ph:>9.3f}{po:>9.3f}{po - ph:>+9.3f}   {tag} "
              f"(h={nh}, o={no})")
        rows_out.append({"clipscore": clip, "human": round(ph, 4), "ours": round(po, 4),
                         "gap": round(po - ph, 4), "n_human": nh, "n_ours": no})

    out = {
        "instrument": "distilroberta-base",
        "folds": args.folds, "epochs": args.epochs,
        "raw": {n: round(float(y[[i for i, c in enumerate(corpus) if c == n]].mean()), 4)
                for n in ("human", "ours")},
        "linear": {"coef_clipscore": round(float(b_clip), 4),
                   "coef_corpus_ours": round(float(b_corp), 4)},
        "quadratic": {"coef_clipscore": round(float(qc), 4),
                      "coef_clipscore_sq": round(float(qc2), 4),
                      "coef_corpus_ours": round(float(q_corp), 4),
                      "corpus_coef_shift": round(float(shift), 4)},
        "fitted_curve": rows_out,
    }
    (ROOT / "runs/clipscore/regression.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'=' * 70}")
    gap_mid = [r["gap"] for r in rows_out if r["n_human"] > 30 and r["n_ours"] > 30]
    if gap_mid:
        print(f"where BOTH corpora have support, the fitted gap runs "
              f"{min(gap_mid):+.3f} to {max(gap_mid):+.3f}")
    print("The corpus effect survives adjustment for groundedness"
          if abs(float(b_corp)) > 0.15 else
          "The corpus effect is small once groundedness is held fixed")
    print(f"wrote {ROOT / 'runs/clipscore/regression.json'}")


if __name__ == "__main__":
    main()
