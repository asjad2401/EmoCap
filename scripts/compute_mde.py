#!/usr/bin/env python
"""Compute the minimum detectable effect for the arm comparisons, for the registration.

The ICC is estimated from a real classifier's per-cell correctness on the generated test
split, clustered by image_id -- not assumed. Everything else follows from it.

    uv run python scripts/compute_mde.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.eval.power import design_effect, icc_from_clusters, mde_table, mde_two_arms  # noqa: E402
from emocap.eval.register_classifier import build_dataset, folds_by_image, tfidf_baseline  # noqa: E402

recs = [json.loads(l) for l in (ROOT / "data/generated/captions_raw.jsonl").read_text().splitlines() if l.strip()]
texts, labels, images = build_dataset(recs)
print(f"test-split data: {len(texts):,} cells over {len(set(images)):,} images")

# Per-cell correctness with its image id. TF-IDF is used only to estimate the ERROR
# STRUCTURE (how correctness clusters within an image), which is a property of the data
# rather than of the model; the transformer's absolute accuracy is higher but there is no
# reason its clustering differs materially, and this costs seconds instead of an hour.
print("estimating per-cell correctness (TF-IDF, folds by image)...")
correct: list[float] = []
clusters: list[str] = []
for train, test in folds_by_image(images, folds=5, seed=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
    m.fit([texts[i] for i in train], [labels[i] for i in train])
    for i, pred in zip(test, m.predict([texts[i] for i in test])):
        correct.append(1.0 if labels[i] == pred else 0.0)
        clusters.append(images[i])

stats = icc_from_clusters(correct, clusters)
print(f"\nintra-cluster correlation of correctness: {stats['icc']}")
for k in ("clusters", "cells", "mean_cluster_size", "mean_correct"):
    print(f"  {k:<20} {stats[k]}")
deff = design_effect(stats["icc"], stats["mean_cluster_size"])
print(f"  design effect        {deff:.3f}   -> effective n is "
      f"{100 / deff:.0f}% of nominal")

# CLUSTER SIZE DIFFERS BY ARM and this dominates the result.
#
# S-paired has 25 cells per image, so its effective n is only ~37% of nominal. The
# UNPAIRED arms have exactly one caption per image, so there is no intra-image clustering
# at all and their effective n equals nominal. Applying the paired arm's design effect to
# them would overstate their MDE by ~1.6x.
#
# Evaluation-set size is the real lever, so three splitting regimes are shown.
P = 0.72   # accuracy region both arms occupy
CASES = [
    # (name, cells per arm, cluster size)
    ("unpaired arms, 1k-image test split", 1000, 1.0),
    ("unpaired arms, 30% test split", 2426, 1.0),
    ("unpaired arms, 5-fold CV over the arm", 8088, 1.0),
    ("S-paired, 1k-image test split", 25000, 25.0),
]
print(f"\nMDE at 80% power, alpha 0.05 split over 3 comparisons (Bonferroni,")
print(f"conservative vs the registered Holm), accuracy region p={P}\n")
print(f"{'comparison':<40}{'cells':>7}{'DEFF':>6}{'seed SD':>9}{'MDE':>8}")
print("-" * 70)
# The REGISTERED curve: 4 confirmatory comparisons, limiting arm 4,390 cells (the
# 1-caption-per-image arms under the strict trait mapping), 3 runs averaged. The 3-run
# model governs because folds partition one dataset rather than drawing independently,
# so averaging 5 cannot be assumed to cut noise by sqrt(5).
registered = {"n_comparisons": 4, "n_cells": 4390, "cluster_size": 1.0,
              "criterion_points": 0.07, "curve": {}, "curve_5runs": {}}
for _sd in (0.0, 0.005, 0.01, 0.02, 0.03):
    for _ns, _k in ((3, "curve"), (5, "curve_5runs")):
        registered[_k][str(_sd)] = round(mde_two_arms(
            P, 4390, icc=stats["icc"], cluster_size=1.0, n_comparisons=4,
            seed_sd=_sd, n_seeds=_ns)["mde"], 4)
print("\nREGISTERED curve (4 comparisons, n=4,390, 3 runs):")
for _sd, _m in registered["curve"].items():
    print(f"  seed SD {_sd:>5}  MDE {_m*100:5.1f} pts"
          + ("   <- criterion 7.0 clears" if 0.07 > _m else "   <- UNDERPOWERED by rule"))

out = {"icc": stats, "design_effect": round(deff, 3), "p_assumed": P,
       "registered": registered, "cases": []}
for name, n, csize in CASES:
    rows = mde_table(P, n, icc=stats["icc"], cluster_size=csize,
                     n_comparisons=3, power=0.80, n_seeds=3)
    for r in rows:
        print(f"{name if r['seed_sd'] == 0.0 else '':<40}{n:>7}"
              f"{r['design_effect']:>6.2f}{r['seed_sd']:>9.3f}{r['mde']:>8.3f}")
    out["cases"].append({"comparison": name, "cells_per_arm": n,
                         "cluster_size": csize, "rows": rows})
    print()

p = ROOT / "runs/mde/result.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2))
print(f"wrote {p}")
