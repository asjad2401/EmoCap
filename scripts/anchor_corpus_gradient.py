"""Recompute the three-corpus lexical-stereotypy gradient on the FINAL corpora.

WHY THIS EXISTS. `docs/preregistration.md` quotes the gradient as v1 0.718 / ours 0.507 /
human 0.450 at 4,486 cells, while `configs/prereg.lock.yaml` quotes ours as 0.440 at the
same n. Both files are frozen at `prereg-v2` and neither names the corpus it measured.

Resolved by measurement (see docs/deviations.md, 2026-08-25):

  * 0.440 is the **v5** corpus, `data/generated/captions_raw.jsonl` (gemini-3.1-flash-lite).
    It reproduces here at 0.4443 +/- 0.0111, and its 24,925-cell value at 0.4075 against the
    registered 0.403. That corpus trains NO arm in this study.
  * 0.507 reproduces from nothing. It was a FORECAST written while the v10 corpus did not
    yet exist -- the registration says so itself: "Every anchor in this document is
    provisional until the corpus exists."
  * The v10 corpus that actually trains every S arm, `captions_corpus.jsonl`, had never been
    measured at 4,486 cells. It is **0.589**.

v1 (0.7171 against a registered 0.718) and the human corpus (0.4501) both reproduce, because
neither changed when the prompt did. Only "ours" moved.

STRUCTURE MATTERS AS WELL AS n. The three corpora carry different numbers of cells per image
-- human 1, v1 5, ours 25 -- so "matched at 4,486 cells" does not match the diversity of the
fit pool. Views of our own corpus at 1 and 5 cells per image are therefore reported beside
the raw figure, and the gradient's ORDERING survives all of them.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from emocap.eval.anchors import keyword_rule_accuracy, keyword_rule_matched
from emocap.runtime import REPO_ROOT as ROOT, rel

EMOTIONS = ["joyful", "sad", "tense", "romantic", "humorous"]
MATCHED_N = 4486


def _sha(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _load_v1(path: Path) -> list[dict]:
    """The pilot ships one row per (image, emotion); regroup to one record per image."""
    by: dict[str, dict] = defaultdict(dict)
    with gzip.open(path, "rt") as fh:
        for row in csv.DictReader(fh):
            emotion = row["emotion"].strip().lower()
            if emotion in EMOTIONS and row["emotion_caption"].strip():
                by[row["image_id"]][emotion] = row["emotion_caption"]
    return [{"image_id": k, "captions": v} for k, v in by.items() if len(v) == len(EMOTIONS)]


def _views(records: list[dict]) -> dict[str, list[dict]]:
    """Our corpus at 1 and 5 cells per image, using the arms' own hash rules.

    Same selection as `data/arms/manifest.json` -- so these views are the corpus text behind
    S_unpaired and S_paired5, not a fresh sample.
    """
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by[r["image_id"]].append(r)

    one, five = [], []
    for img, rows in by.items():
        rows = sorted(rows, key=lambda r: r["caption_idx"])
        emotion = EMOTIONS[_sha("unpaired-v1" + img) % 5]
        if emotion in rows[0]["captions"]:
            one.append({"image_id": img, "captions": {emotion: rows[0]["captions"][emotion]}})
        five.append({"image_id": img,
                     "captions": rows[_sha("cap-v1" + img) % len(rows)]["captions"]})
    return {"1_cell_per_image": one, "5_cells_per_image": five}


def main() -> None:
    gen = ROOT / "data" / "generated"
    v10 = _load_jsonl(gen / "captions_corpus.jsonl")
    v5 = _load_jsonl(gen / "captions_raw.jsonl")
    v1 = _load_v1(ROOT / "archive" / "v1-pilot" / "data" / "v1_emotion_captions.csv.gz")

    def matched(recs: list[dict]) -> dict:
        out = keyword_rule_matched(recs, n_cells=MATCHED_N, n_subsamples=10)
        return {k: out[k] for k in ("mean", "sd", "min", "max", "n_cells")}

    result = {
        "why": "resolve 0.507 vs 0.440; neither describes the corpus that trained the arms",
        "estimator": "cv5_by_image_id_3seeds_fractional_ties",
        "matched_n_cells": MATCHED_N,
        "chance": 0.2,
        "corpora": {
            "v1_pilot": {
                "store": "archive/v1-pilot/data/v1_emotion_captions.csv.gz",
                "cells_per_image": 5, "images": len(v1),
                "matched": matched(v1),
                "full_n": keyword_rule_accuracy(v1)["accuracy"],
                "registered_value": 0.718, "reproduces": True,
            },
            "ours_v10_FINAL": {
                "store": "data/generated/captions_corpus.jsonl",
                "model": "gemini-3.7-flash", "prompt": "v10",
                "cells_per_image": 25, "images": len({r["image_id"] for r in v10}),
                "matched": matched(v10),
                "full_n": keyword_rule_accuracy(v10)["accuracy"],
                "registered_value": None,
                "note": "NEVER MEASURED BEFORE. This is the corpus every S arm trains on.",
            },
            "ours_v5_SUPERSEDED": {
                "store": "data/generated/captions_raw.jsonl",
                "model": "gemini-3.1-flash-lite", "prompt": "v5",
                "cells_per_image": 25, "images": len({r["image_id"] for r in v5}),
                "matched": matched(v5),
                "full_n": keyword_rule_accuracy(v5)["accuracy"],
                "registered_value": 0.440, "reproduces": True,
                "note": "this is what the lock's 0.440 measured; it trains no arm",
            },
            "human_personality_captions": {
                "source": "runs/anchor-matched/result.json (unchanged by the prompt rewrite)",
                "cells_per_image": 1,
                "matched": {"mean": 0.4501, "sd": 0.0, "n_cells": MATCHED_N},
                "registered_value": 0.450, "reproduces": True,
            },
        },
        "structure_matched_views_of_ours_v10": {
            name: matched(view) for name, view in _views(v10).items()
        },
        "forecast_0_507": {
            "reproduces_from": None,
            "verdict": "a pre-corpus forecast, not a measurement; the registration labels "
                       "every anchor in it provisional until the corpus exists",
            "measured_instead": "0.589 at 4,486 cells",
            "error": "+0.082 -- the forecast UNDERSTATED our corpus's stereotypy",
        },
    }
    out = ROOT / "results" / "anchor_corpus_gradient.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {rel(out)}")
    for name, c in result["corpora"].items():
        m = c["matched"]
        print(f"  {name:32} @4,486 {m['mean']:.4f}  full {c.get('full_n', '--')}")
    for name, m in result["structure_matched_views_of_ours_v10"].items():
        print(f"  ours v10 [{name:18}]  @4,486 {m['mean']:.4f}")


if __name__ == "__main__":
    main()
