#!/usr/bin/env python
"""Two instrument checks that must happen before any arm is trained.

    uv run python scripts/check_instrument.py

**1. Does the stereotypy ordering survive a change of instrument?**
§6 registers the ORDERING of the three-corpus anchor gradient as a prediction (v1 pilot >
ours > human). The top-25 keyword rule gave 0.718 / 0.507 / 0.450. Bigram TF-IDF on the
same three arms gave S 0.807 > H 0.612 > V1 0.584 -- a different ordering. Both are
"lexical shortcut" measures, so a reviewer will ask which one the prediction was about.
This runs both estimators on the identical 4,390-cell arms and writes the answer down
before any result could motivate a preference.

A plausible mechanism, which this also tests: the v10 vocabulary cap suppressed
high-document-frequency register words *by construction*, and the top-25 keyword rule is
exactly what such words feed. If the cap moved the shortcut into the tail rather than
removing it, the keyword anchor falls while TF-IDF stays high -- and only the pair of
measurements can show that.

**2. The artifact ablation, on the real classifier.**
§4.1 requires accuracy on punctuation-stripped, lowercased text beside raw. The version run
during training used TF-IDF, whose tokenizer discards punctuation anyway -- so it moved by
exactly 0.0000 and could not have failed. This runs the frozen transformer.

The frozen instrument was trained on all 13,170 cells, so its accuracy HERE is in-sample and
its level means nothing. The meaningful quantity is level-free: the share of cells whose
PREDICTION CHANGES when punctuation and case are removed. A model reading register would
barely move; a model reading formatting would flip often.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.eval.anchors import keyword_rule_accuracy  # noqa: E402
from emocap.eval.register_classifier import strip_artifacts, tfidf_baseline  # noqa: E402

ARMS = ("S_unpaired", "V1_unpaired", "H_unpaired")
CLF = ROOT / "models/register_classifier"


def load(arm: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / "data/arms" / f"{arm}.jsonl").read_text().splitlines() if l.strip()]


def as_records(cells: list[dict]) -> list[dict]:
    """The keyword estimator wants {image_id, captions:{emotion: text}} per image."""
    by: dict[str, dict] = {}
    for c in cells:
        by.setdefault(c["image_id"], {"image_id": c["image_id"], "captions": {}})
        by[c["image_id"]]["captions"][c["emotion"]] = c["text"]
    return list(by.values())


def main() -> None:
    report: dict = {"arms": {}}
    print("1. STEREOTYPY, TWO INSTRUMENTS, SAME 4,390 CELLS")
    print(f"{'arm':<14}{'keyword@25':>12}{'tfidf':>10}{'gap':>9}")
    for arm in ARMS:
        cells = load(arm)
        kw = keyword_rule_accuracy(as_records(cells))
        tf = tfidf_baseline([c["text"] for c in cells],
                            [EMOTIONS.index(c["emotion"]) for c in cells],
                            [c["image_id"] for c in cells], folds=5, seed=42)
        report["arms"][arm] = {"keyword_at_25": kw["accuracy"], "per_seed": kw["per_seed"],
                              "tfidf": tf["accuracy"], "n_cells": len(cells)}
        print(f"{arm:<14}{kw['accuracy']:>12.4f}{tf['accuracy']:>10.4f}"
              f"{tf['accuracy']-kw['accuracy']:>+9.4f}")
    order_kw = sorted(ARMS, key=lambda a: -report["arms"][a]["keyword_at_25"])
    order_tf = sorted(ARMS, key=lambda a: -report["arms"][a]["tfidf"])
    report["ordering_keyword"] = order_kw
    report["ordering_tfidf"] = order_tf
    report["ordering_agrees"] = order_kw == order_tf
    print(f"\n  keyword ordering  {' > '.join(order_kw)}")
    print(f"  tfidf   ordering  {' > '.join(order_tf)}")
    print(f"  ORDERINGS AGREE:  {order_kw == order_tf}")

    print("\n2. ARTIFACT ABLATION ON THE FROZEN CLASSIFIER")
    if not (CLF / "config.json").exists():
        print("  frozen classifier not found -- skipped")
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cells = [c for a in ARMS for c in load(a)]
        raw = [c["text"] for c in cells]
        stripped = [strip_artifacts(t) for t in raw]
        y = [EMOTIONS.index(c["emotion"]) for c in cells]
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(str(CLF))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(CLF)).to(dev).eval()

        def predict(texts):
            out = []
            for s in range(0, len(texts), 128):
                e = tok(texts[s:s + 128], truncation=True, max_length=48,
                        padding="max_length", return_tensors="pt").to(dev)
                with torch.no_grad():
                    out += mdl(**e).logits.argmax(-1).cpu().tolist()
            return out

        pr, ps = predict(raw), predict(stripped)
        acc_r = sum(int(a == b) for a, b in zip(pr, y)) / len(y)
        acc_s = sum(int(a == b) for a, b in zip(ps, y)) / len(y)
        flips = [i for i, (a, b) in enumerate(zip(pr, ps)) if a != b]
        report["ablation"] = {
            "in_sample_accuracy_raw": round(acc_r, 4),
            "in_sample_accuracy_stripped": round(acc_s, 4),
            "accuracy_gap": round(acc_r - acc_s, 4),
            "prediction_flip_rate": round(len(flips) / len(y), 4),
            "n": len(y), "note": "levels are IN-SAMPLE; the flip rate is the real measure"}
        print(f"  in-sample accuracy   raw {acc_r:.4f}   stripped {acc_s:.4f}   "
              f"gap {acc_r-acc_s:+.4f}")
        print(f"  PREDICTION FLIP RATE {len(flips)/len(y):.4f}  "
              f"({len(flips):,} of {len(y):,} cells)")
        print(f"  flips by true register: "
              f"{dict(Counter(EMOTIONS[y[i]] for i in flips))}")

    (ROOT / "runs/instrument").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/instrument/report.json").write_text(json.dumps(report, indent=2))
    print("\nwrote runs/instrument/report.json")


if __name__ == "__main__":
    main()
