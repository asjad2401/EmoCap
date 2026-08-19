#!/usr/bin/env python
"""Sensitivity of the keyword anchor to its hyperparameters.

The second review asked how the 25 keywords were chosen. `anchors.py` fits them on
training folds only, split by `image_id`, so the baseline is not fitted to its own test
set. But `top_k=25`, `min_doc_freq=4` and the `+3` damping were chosen while writing the
code after seeing v1 data, NOT pre-registered. If the headline "72% of the human-data
advantage is keyword-recoverable" moves a lot with top_k, that figure is a function of a
choice rather than of the data, and the paper has to say so.

    uv run python scripts/sweep_anchor.py
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.eval.anchors import keyword_rule_accuracy  # noqa: E402

spec = importlib.util.spec_from_file_location("cmp", ROOT / "scripts/compare_human_ceiling.py")
cmp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp_mod)
EMOTIONS = ["joyful", "sad", "tense", "romantic", "humorous"]

# ── ours, sampled to the human sample size so the two are comparable ────────
recs = [json.loads(l) for l in open(ROOT / "data/generated/captions_raw.jsonl")]
rng = random.Random(42)
imgs = sorted({r["image_id"] for r in recs})
rng.shuffle(imgs)
ours, cells = [], 0
for i in imgs:
    rs = [r for r in recs if r["image_id"] == i]
    ours += rs
    cells += sum(len(r["captions"]) for r in rs)
    if cells >= 4486:
        break

# ── human, strict 1:1 trait mapping ────────────────────────────────────────
t, l, g = cmp_mod.load_personality_captions(
    ROOT / "data/external/personality_captions", cmp_mod.STRICT_TRAITS)
human = [{"image_id": gi, "captions": {EMOTIONS[li]: ti}} for ti, li, gi in zip(t, l, g)]

# Measured accuracies of the pre-registered instrument at these sample sizes; the
# margin is what the paper quotes, so the sweep must move the margin, not the anchor.
ACC = {"ours": 0.683, "human": 0.726}

print(f"ours {sum(len(r['captions']) for r in ours):,} cells | "
      f"human {len(human):,} cells\n")
print(f"{'top_k':>6}{'min_df':>8}{'anchor ours':>13}{'anchor human':>14}"
      f"{'margin ours':>13}{'margin human':>14}{'margin diff':>13}{'% of gap':>10}")
print("-" * 91)
out = []
for top_k in (10, 25, 50, 100):
    for min_df in (2, 4, 8):
        a_o = keyword_rule_accuracy(ours, top_k=top_k, min_doc_freq=min_df)["accuracy"]
        a_h = keyword_rule_accuracy(human, top_k=top_k, min_doc_freq=min_df)["accuracy"]
        m_o, m_h = ACC["ours"] - a_o, ACC["human"] - a_h
        acc_gap = ACC["human"] - ACC["ours"]
        pct = 100 * (a_h - a_o) / acc_gap if acc_gap else float("nan")
        out.append({"top_k": top_k, "min_doc_freq": min_df, "anchor_ours": a_o,
                    "anchor_human": a_h, "margin_ours": round(m_o, 4),
                    "margin_human": round(m_h, 4),
                    "margin_diff": round(m_h - m_o, 4),
                    "pct_of_gap_explained_by_anchor": round(pct, 1)})
        print(f"{top_k:>6}{min_df:>8}{a_o:>13.3f}{a_h:>14.3f}"
              f"{m_o:>13.3f}{m_h:>14.3f}{m_h - m_o:>13.3f}{pct:>9.0f}%")

p = ROOT / "runs/anchor-sweep/result.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"note": "ACC values are the measured DistilRoBERTa accuracies "
                                 "at these sample sizes; only the anchor is swept.",
                         "accuracies": ACC, "sweep": out}, indent=2))
diffs = [r["margin_diff"] for r in out]
pcts = [r["pct_of_gap_explained_by_anchor"] for r in out]
print(f"\nmargin difference ranges {min(diffs):+.3f} to {max(diffs):+.3f} "
      f"across all 12 settings")
print(f"share of the accuracy gap explained by the anchor: "
      f"{min(pcts):.0f}% to {max(pcts):.0f}%")
print(f"\nwrote {p}")
