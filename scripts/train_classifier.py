#!/usr/bin/env python
"""Build and freeze the study's primary instrument, before any arm is trained.

    uv run python scripts/train_classifier.py

The registered instrument (prereg-v2, `metrics.classifier`) is ONE DistilRoBERTa trained on
a provenance-balanced pool -- the 4,390 cells of each of S_unpaired, V1_unpaired and
H_unpaired, 13,170 in total -- frozen, hashed, and applied unchanged to every arm.

Balance is the whole point. A classifier trained on our corpus alone would recognise our
generator's fingerprint and flatter S over V1 and H by construction; scoring each arm with
its own classifier would put the arms on different scales and make their differences
unsubtractable. Equal cells from all three provenances is the only version where the
number means the same thing for every arm.

Order matters: this runs BEFORE the first arm run. An instrument trained after seeing an
arm's generated captions could have been selected on the outcome, which is exactly what
the registration forbids.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.eval.register_classifier import (  # noqa: E402
    confusion_matrix,
    finetune_classifier,
    strip_artifacts,
    tfidf_baseline,
    train_final_classifier,
)

POOL = ("S_unpaired", "V1_unpaired", "H_unpaired")
OUT = ROOT / "models/register_classifier"


def load_pool() -> tuple[list[str], list[int], list[str], list[str]]:
    texts, labels, images, prov = [], [], [], []
    for arm in POOL:
        for line in (ROOT / "data/arms" / f"{arm}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            texts.append(c["text"])
            labels.append(EMOTIONS.index(c["emotion"]))
            images.append(c["image_id"])
            prov.append(c["source"])
    return texts, labels, images, prov


def main() -> None:
    texts, labels, images, prov = load_pool()
    print(f"pool  {len(texts):,} cells   provenance {dict(Counter(prov))}")
    print(f"      registers {dict(Counter(EMOTIONS[l] for l in labels))}")
    report: dict = {"pool_cells": len(texts), "provenance": dict(Counter(prov)),
                    "arms": list(POOL)}

    # The lexical baseline is reported first on purpose: if TF-IDF matches the transformer,
    # the instrument is reading vocabulary and the "classifier" adds nothing over the
    # keyword anchor it is supposed to be measured against.
    print("\nTF-IDF baseline (5-fold by image)...")
    tf = tfidf_baseline(texts, labels, images, folds=5, seed=42)
    report["tfidf"] = {"accuracy": tf["accuracy"], "n": tf["n"]}
    print(f"  accuracy {tf['accuracy']:.4f}")

    t0 = time.time()
    print("\nDistilRoBERTa, 5-fold CV by image (this is the instrument's ACCURACY)...")
    cv = finetune_classifier(
        texts, labels, images, folds=5, seed=42,
        progress=lambda f, a: print(f"  fold {f+1}/5  running acc {a:.4f}  "
                                    f"[{(time.time()-t0)/60:.0f} min]", flush=True))
    report["cv"] = {"accuracy": cv["accuracy"], "n": cv["n"], "epochs": cv["epochs"]}
    report["confusion"] = confusion_matrix(cv["pairs"])
    print(f"  pooled held-out accuracy {cv['accuracy']:.4f}  (n={cv['n']:,})")

    # Per-provenance accuracy needs per-cell indices, which finetune_classifier does not
    # return; measured with the lexical baseline instead, where it is cheap. A large gap
    # is a finding rather than a defect -- it says one provenance's register is easier to
    # read than another's, and every cross-provenance comparison must be read in that light.
    print("\nTF-IDF accuracy by provenance...")
    report["tfidf_by_provenance"] = {}
    for name in sorted(set(prov)):
        keep = [i for i, q in enumerate(prov) if q == name]
        r = tfidf_baseline([texts[i] for i in keep], [labels[i] for i in keep],
                           [images[i] for i in keep], folds=5, seed=42)
        report["tfidf_by_provenance"][name] = r["accuracy"]
        print(f"  {name:3s} {r['accuracy']:.4f}  (n={r['n']:,})")

    print("\nartifact ablation (punctuation stripped, lowercased)...")
    ab = tfidf_baseline([strip_artifacts(t) for t in texts], labels, images,
                        folds=5, seed=42)
    report["tfidf_stripped"] = {"accuracy": ab["accuracy"]}
    print(f"  TF-IDF on stripped text {ab['accuracy']:.4f} "
          f"(raw {tf['accuracy']:.4f}, gap {tf['accuracy']-ab['accuracy']:+.4f})")

    print("\ntraining the FINAL frozen instrument on all cells...")
    fin = train_final_classifier(texts, labels, out_dir=OUT)
    report["frozen"] = fin
    print(f"  saved {fin['path']}\n  sha256 {fin['sha256']}")

    (ROOT / "runs/classifier").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/classifier/report.json").write_text(json.dumps(report, indent=2))
    print("\nwrote runs/classifier/report.json")


if __name__ == "__main__":
    main()
