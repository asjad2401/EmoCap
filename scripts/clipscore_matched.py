#!/usr/bin/env python
"""Compare human and synthetic register signal at MATCHED groundedness.

`clipscore_asymmetry.py` established two things: our captions are much more grounded
(CLIPScore 0.792 vs 0.578), and within each corpus less-grounded captions are *easier* to
classify by register (human -0.058 across quartiles, ours -0.028). Together those suggest
the human corpus's 4.3-point accuracy advantage is bought by being allowed to ignore the
image, not by being human-written.

Suggestive is not established. One overlapping quartile pair is not a controlled
comparison, so this does it two ways:

1. **Bin matching.** Fine CLIPScore bins; take min(n_human, n_ours) from each bin. The two
   samples then have near-identical groundedness distributions by construction, and the
   accuracy difference is the residual. Transparent, but throws away everything outside the
   overlap.
2. **Regression adjustment.** Logistic fit of per-caption correctness on CLIPScore with a
   corpus indicator, using every caption. The indicator's coefficient is the corpus effect
   holding groundedness fixed. Uses all the data; assumes the functional form.

Agreeing answers make the finding robust to either method's weakness.

    uv run python scripts/clipscore_matched.py

Per-caption CLIPScores are cached in runs/clipscore/scores.json, so re-running the
matching does not re-encode 9,000 images.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.eval.register_classifier import (  # noqa: E402
    finetune_classifier,
    folds_by_image,
    tfidf_baseline,
)

CACHE = ROOT / "runs/clipscore/scores.json"


def build_scored(n_per_corpus: int, seed: int) -> dict:
    """Per-caption (text, label, image, clipscore) for both corpora, cached."""
    if CACHE.exists():
        d = json.loads(CACHE.read_text())
        if d.get("n_per_corpus") == n_per_corpus and d.get("seed") == seed:
            print(f"using cached scores from {CACHE}")
            return d

    import torch
    spec = importlib.util.spec_from_file_location(
        "asym", ROOT / "scripts/clipscore_asymmetry.py")
    asym = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asym)
    spec2 = importlib.util.spec_from_file_location(
        "cmp", ROOT / "scripts/compare_human_ceiling.py")
    cmp_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(cmp_mod)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"scoring on {device} (one-off; results cached)")
    model, proc = asym.load_clip(device)

    t, l, g = cmp_mod.load_personality_captions(
        ROOT / "data/external/personality_captions", cmp_mod.STRICT_TRAITS)
    yfcc = ROOT / "data/external/yfcc_images"
    human = [(t[i], l[i], g[i], yfcc / f"{g[i]}.jpg") for i in range(len(t))]
    human = [x for x in human if x[3].exists()]
    random.Random(seed).shuffle(human)
    human = human[:n_per_corpus]

    recs = [json.loads(x) for x in
            (ROOT / "data/generated/captions_raw.jsonl").read_text().splitlines() if x.strip()]
    imgdir = ROOT / "data/flickr8k/Images"
    ours = [(str(txt).strip(), EMOTIONS.index(reg), r["image_id"], imgdir / r["image_id"])
            for r in recs for reg, txt in (r.get("captions") or {}).items()
            if reg in EMOTIONS and str(txt).strip()]
    random.Random(seed).shuffle(ours)
    ours = ours[:n_per_corpus]

    out = {"n_per_corpus": n_per_corpus, "seed": seed, "corpora": {}}
    for name, rows in (("human", human), ("ours", ours)):
        print(f"  {name}: {len(rows):,} captions", flush=True)
        sc = asym.clipscores([(r[3], r[0]) for r in rows], model, proc, device)
        out["corpora"][name] = [
            {"text": r[0], "label": r[1], "image": r[2], "clipscore": s}
            for r, s in zip(rows, sc) if s is not None
        ]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    print(f"cached to {CACHE}")
    return out


def bin_match(a: list[dict], b: list[dict], *, width: float, seed: int):
    """Take min(count) per CLIPScore bin from each side, so distributions align."""
    def binned(rows):
        d = defaultdict(list)
        for r in rows:
            d[round(r["clipscore"] / width)].append(r)
        return d

    ba, bb = binned(a), binned(b)
    rng = random.Random(seed)
    ka, kb = [], []
    for k in sorted(set(ba) & set(bb)):
        n = min(len(ba[k]), len(bb[k]))
        ra, rb = list(ba[k]), list(bb[k])
        rng.shuffle(ra)
        rng.shuffle(rb)
        ka += ra[:n]
        kb += rb[:n]
    return ka, kb


def run(rows: list[dict], *, fast: bool, folds: int, epochs: int, seed: int) -> dict:
    fn = tfidf_baseline if fast else finetune_classifier
    kw = {} if fast else {"epochs": epochs}
    r = fn([x["text"] for x in rows], [x["label"] for x in rows],
           [x["image"] for x in rows], folds=folds, seed=seed, **kw)
    return {"accuracy": r["accuracy"], "n": r["n"], "pairs": r["pairs"]}


def logistic_adjust(human: list[dict], ours: list[dict], *, seed: int) -> dict:
    """Correctness ~ CLIPScore + corpus, so the corpus effect holds groundedness fixed."""
    from sklearn.linear_model import LogisticRegression

    # Per-caption correctness from a TF-IDF classifier, folds by image, within each corpus
    # (a single pooled classifier would learn the corpus, not the register).
    feats, ys = [], []
    for corpus_flag, rows in ((0.0, human), (1.0, ours)):
        texts = [r["text"] for r in rows]
        labels = [r["label"] for r in rows]
        images = [r["image"] for r in rows]
        correct = {}
        for tr, te in folds_by_image(images, folds=5, seed=seed):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline
            m = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
            m.fit([texts[i] for i in tr], [labels[i] for i in tr])
            for i, pr in zip(te, m.predict([texts[i] for i in te])):
                correct[i] = 1 if labels[i] == pr else 0
        for i, r in enumerate(rows):
            if i in correct:
                feats.append([r["clipscore"], corpus_flag])
                ys.append(correct[i])

    lr = LogisticRegression(max_iter=2000).fit(feats, ys)
    b_clip, b_corpus = lr.coef_[0]
    common = statistics.mean(f[0] for f in feats)
    import math
    p_h = 1 / (1 + math.exp(-(lr.intercept_[0] + b_clip * common)))
    p_o = 1 / (1 + math.exp(-(lr.intercept_[0] + b_clip * common + b_corpus)))
    return {
        "coef_clipscore": round(float(b_clip), 4),
        "coef_corpus_ours": round(float(b_corpus), 4),
        "at_common_clipscore": round(common, 4),
        "predicted_human": round(p_h, 4),
        "predicted_ours": round(p_o, 4),
        "adjusted_gap_ours_minus_human": round(p_o - p_h, 4),
        "n": len(ys),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4486)
    ap.add_argument("--bin-width", type=float, default=0.02)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    d = build_scored(args.n, args.seed)
    human, ours = d["corpora"]["human"], d["corpora"]["ours"]
    for name, rows in (("human", human), ("ours", ours)):
        s = [r["clipscore"] for r in rows]
        print(f"{name:<7} n={len(rows):,}  CLIPScore mean {statistics.mean(s):.4f}  "
              f"sd {statistics.stdev(s):.4f}  range {min(s):.3f}-{max(s):.3f}")

    mh, mo = bin_match(human, ours, width=args.bin_width, seed=args.seed)
    sh = [r["clipscore"] for r in mh]
    so = [r["clipscore"] for r in mo]
    print(f"\nbin-matched at width {args.bin_width}: {len(mh):,} per side")
    print(f"  human CLIPScore {statistics.mean(sh):.4f}   "
          f"ours {statistics.mean(so):.4f}   diff {statistics.mean(so)-statistics.mean(sh):+.4f}")
    if len(mh) < 400:
        print("  WARNING: the overlap is small; treat the matched result as weak evidence")

    inst = "TF-IDF" if args.fast else "DistilRoBERTa"
    print(f"\nrunning {inst} on the matched sets ...", flush=True)
    rh = run(mh, fast=args.fast, folds=args.folds, epochs=args.epochs, seed=args.seed)
    ro = run(mo, fast=args.fast, folds=args.folds, epochs=args.epochs, seed=args.seed)
    print(f"  human {rh['accuracy']:.3f}   ours {ro['accuracy']:.3f}   "
          f"diff {ro['accuracy']-rh['accuracy']:+.3f}")

    print("\nregression adjustment over all captions ...", flush=True)
    adj = logistic_adjust(human, ours, seed=args.seed)
    print(f"  coef CLIPScore {adj['coef_clipscore']:+.3f}  "
          f"(negative = less grounded is easier to classify)")
    print(f"  coef corpus=ours {adj['coef_corpus_ours']:+.3f}")
    print(f"  at common CLIPScore {adj['at_common_clipscore']:.3f}: "
          f"human {adj['predicted_human']:.3f}  ours {adj['predicted_ours']:.3f}  "
          f"gap {adj['adjusted_gap_ours_minus_human']:+.3f}")

    out = {"instrument": inst, "bin_width": args.bin_width,
           "matched_n_per_side": len(mh),
           "matched": {"human": rh["accuracy"], "ours": ro["accuracy"],
                       "diff": round(ro["accuracy"] - rh["accuracy"], 4)},
           "unmatched_clipscore": {"human": round(statistics.mean(
               [r["clipscore"] for r in human]), 4),
               "ours": round(statistics.mean([r["clipscore"] for r in ours]), 4)},
           "regression": adj}
    p = ROOT / "runs/clipscore/matched.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n{'=' * 66}")
    print("Both methods agree the corpus gap is small at matched groundedness"
          if abs(out["matched"]["diff"]) < 0.03 and abs(adj["adjusted_gap_ours_minus_human"]) < 0.03
          else "The two methods DISAGREE, or a real corpus effect survives adjustment")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
