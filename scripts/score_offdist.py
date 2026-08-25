#!/usr/bin/env python
"""Score the hand-written captions with the frozen instrument. The registered off-distribution check.

    uv run python scripts/score_offdist.py
    uv run python scripts/score_offdist.py --results runs/offdist

§4.1 calls this **"the instrument's real measurement error"**, and it is the only number in
the study measured on text from outside the classifier's training distribution.

**What it settles.** The frozen classifier learned register from 13,170 captions, 8,780 of
them written by Gemini. Every arm was then trained on Gemini text and produces
Gemini-flavoured text, so the classifier could be recognising a house style rather than
emotional register -- and every result in the study would look identical either way. The
keyword anchor cannot separate them either, being fitted on the same text. Only captions
nobody in that loop wrote can.

**How to read the number.** Compare it to the instrument's cross-validated accuracy on its
own pool, **0.8279**, recorded when it was frozen:

* **holds near 0.83** -- it learned register, and the criticism is answered outright.
* **drops toward chance (0.20)** -- a real share of every reported accuracy is generator
  recognition. That does not invalidate the arm COMPARISONS, which all use the same
  instrument, but it does mean the absolute accuracies overstate register conditioning and
  the paper has to say so.
* **in between** -- report the gap as the instrument's measurement error and carry it into
  how the headline numbers are described.

Every outcome is publishable and none of them is a reason to change the instrument. The
classifier is frozen and hashed; this measures it, it does not tune it.

**The labels need no second annotator.** In a normal annotation task two people label the
same text and their agreement bounds the label quality. Here the register was *shown to the
writer before they wrote*, so the label precedes the text rather than being inferred from
it. What a second annotator would measure -- whether a reader agrees the caption sounds
joyful -- is a different question, and the human evaluation already answers it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.eval.anchors import keyword_rule_accuracy  # noqa: E402
from emocap.eval.bootstrap import cluster_bootstrap_mean  # noqa: E402
from emocap.eval.register_classifier import confusion_matrix  # noqa: E402
from emocap.runtime import load_config, rel  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/offdist")
    ap.add_argument("--key", default="runs/offdist/key.json")
    ap.add_argument("--classifier", default="models/register_classifier")
    ap.add_argument("--out", default="results/offdist.json")
    args = ap.parse_args()

    keyf = json.loads((ROOT / args.key).read_text())
    key, build = keyf["key"], keyf["build"]
    clf = ROOT / args.classifier

    written: dict[str, str] = {}
    chunks: list[int] = []
    for p in sorted((ROOT / args.results).glob("*.json")):
        if p.name == "key.json":
            continue
        d = json.loads(p.read_text())
        if "captions" not in d:
            continue
        if d.get("build") != build:
            raise SystemExit(f"{p.name} was built from {d.get('build')!r}, key is {build!r}"
                             f" -- different item sets must not be pooled")
        chunks.append(d.get("chunk"))
        for c in d["captions"]:
            t = str(c.get("text", "")).strip()
            if t and c["id"] in key:
                written[c["id"]] = t
    if not written:
        raise SystemExit(
            f"no writer exports in {args.results}/ (key.json aside).\n"
            f"Collect the downloaded JSON files there and re-run.")

    ids = sorted(written)
    texts = [written[i] for i in ids]
    truth = [EMOTIONS.index(key[i]["register"]) for i in ids]
    per_reg = Counter(key[i]["register"] for i in ids)

    print(f"build {build}   chunks returned {sorted(set(chunks))} of {keyf['chunks']}")
    print(f"{len(texts)} captions of {keyf['n']}   per register {dict(per_reg)}")
    spread = max(per_reg.values()) - min(per_reg.values()) if per_reg else 0
    if spread > 2:
        print(f"  NOTE registers are uneven by {spread}; the accuracy below is a weighted "
              f"average across unequal samples, so read the per-register recall too.")
    words = [len(t.split()) for t in texts]
    print(f"  {sum(words)/len(words):.1f} mean words, "
          f"{len(set(texts))} distinct of {len(texts)}")

    got = predict(texts, clf)
    correct = [float(a == b) for a, b in zip(got, truth)]
    acc = sum(correct) / len(correct)
    # Clustered by image, as everywhere else -- one image contributes one caption here, so
    # this is an ordinary bootstrap, but the estimator stays the registered one.
    boot = load_config("prereg.lock")["metrics"]["bootstrap"]
    ci = cluster_bootstrap_mean(correct, [key[i]["image_id"] for i in ids],
                                n_resamples=boot["resamples"], ci=boot["ci"], seed=42)

    prior = json.loads((ROOT / "runs/classifier/report.json").read_text())
    in_dist = prior["cv"]["accuracy"]

    # The keyword anchor on this text too. If the hand-written captions are ALSO
    # keyword-separable, a high classifier score here does not prove the classifier reads
    # register -- it may just be reading the same vocabulary a rule can.
    by_img: dict[str, dict] = {}
    for i in ids:
        img = key[i]["image_id"]
        by_img.setdefault(img, {"image_id": img, "captions": {}})
        by_img[img]["captions"][key[i]["register"]] = written[i]
    anchor = None
    try:
        anchor = keyword_rule_accuracy(list(by_img.values()))
    except ValueError as e:
        print(f"  anchor not computable on this sample: {e}")

    print(f"\nOFF-DISTRIBUTION ACCURACY   {acc:.4f}   95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]")
    print(f"  in-distribution CV          {in_dist:.4f}   (the instrument's own pool)")
    print(f"  drop                        {in_dist - acc:+.4f}")
    print(f"  chance                      {1/len(EMOTIONS):.4f}")
    if anchor:
        print(f"  keyword anchor here         {anchor['accuracy']:.4f} "
              f"(n={anchor['n_cells']:,}) -- margin {acc - anchor['accuracy']:+.4f}")
    conf = confusion_matrix(list(zip(truth, got)))
    print("  recall  " + "  ".join(f"{k[:3]} {v}" for k, v in conf["recall"].items()))

    drop = in_dist - acc
    if drop < 0.05:
        verdict = ("holds. The instrument reads register rather than one generator's style, "
                   "and the criticism is answered.")
    elif acc < 0.35:
        verdict = ("collapses toward chance. A real share of every reported accuracy is "
                   "generator recognition. Arm COMPARISONS survive -- they all use this one "
                   "instrument -- but the absolute accuracies overstate conditioning and the "
                   "paper must say so.")
    else:
        verdict = ("drops materially. Report this gap as the instrument's measurement error "
                   "and carry it into how the headline accuracies are described.")
    print(f"\n  VERDICT: it {verdict}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "build": build, "n_captions": len(texts), "n_registered": keyf["n"],
        "chunks_returned": sorted(set(chunks)), "per_register": dict(per_reg),
        "classifier_sha256": (clf / "sha256.txt").read_text().strip()
        if (clf / "sha256.txt").exists() else None,
        "accuracy": round(acc, 4), "ci": ci,
        "in_distribution_cv": in_dist, "drop": round(drop, 4),
        "chance": round(1 / len(EMOTIONS), 4),
        "keyword_anchor": anchor["accuracy"] if anchor else None,
        "anchor_n_cells": anchor["n_cells"] if anchor else None,
        "margin_over_anchor": round(acc - anchor["accuracy"], 4) if anchor else None,
        "confusion": conf,
        "mean_words": round(sum(words) / len(words), 1),
        "distinct_captions": len(set(texts)),
        "verdict": verdict,
        "label_provenance": keyf["label_provenance"],
    }, indent=2))
    print(f"\nwrote {rel(out)}")


if __name__ == "__main__":
    main()
