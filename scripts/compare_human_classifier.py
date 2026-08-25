#!/usr/bin/env python
"""Human and classifier on the SAME captions. Post-hoc, and the like-for-like P5 comparison.

    uv run python scripts/compare_human_classifier.py \
        --export runs/guess/emocap-guess-text-300.json \
        --key runs/guess/legibility300_key.json

**Post-hoc and unregistered.** Decided 2026-08-25, after the legibility task returned. The
paper must label it so.

**What it fixes.** P5 asks whether any model exceeds the human blind-guess score on its own
corpus. As measured, the two sides were not comparable: the human read 300 *corpus* captions
while the classifier's accuracies are on *generated* captions, so the text and the judge both
differed. Running the frozen classifier over the identical 300 captions the human just
guessed removes both differences at once, and costs nothing -- the key already stores the
text.

**Three numbers, and only one of them is a fair comparison.**

``FCE``            the human's forced-choice-equivalent, 0.864 here. The classifier must
                   always pick one of five, so this is the number to compare it against:
                   it reconstructs what the reader would have scored with no neutral option.
``decided-only``   both judges restricted to the captions the human was willing to place.
                   Cleaner still, at the cost of dropping the hardest items -- which flatters
                   both sides, so it is reported beside FCE rather than instead of it.
``raw``            the human's accuracy counting neutrals as wrong. Not comparable to
                   anything the classifier does; printed only for completeness.

**Six of the 300 captions are in the classifier's training pool.** The legibility task drew
from the same corpus that supplies ``S_unpaired``, so a small overlap is unavoidable. The
headline classifier figure EXCLUDES them; the with-overlap figure is printed beside it so the
size of the effect is visible rather than asserted to be negligible.

**Agreement is reported separately from accuracy**, because two judges can score identically
while disagreeing constantly. Cohen's kappa on the decided items answers whether the
classifier is reading what the person read, or arriving at the same score by another route.
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
from emocap.runtime import load_config, rel  # noqa: E402

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


def kappa(a: list[str], b: list[str], labels: tuple[str, ...]) -> float | None:
    """Cohen's kappa between two label sequences."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="the reader's guess export")
    ap.add_argument("--key", required=True)
    ap.add_argument("--classifier", default="models/register_classifier")
    ap.add_argument("--out", default="results/human_vs_classifier.json")
    args = ap.parse_args()

    key = json.loads((ROOT / args.key).read_text())["key"]
    answers = json.loads((ROOT / args.export).read_text())["answers"]
    clf = ROOT / args.classifier

    trained = set()
    for arm in POOL:
        for line in (ROOT / "data/arms" / f"{arm}.jsonl").read_text().splitlines():
            if line.strip():
                trained.add(json.loads(line)["text"].strip())

    ids = sorted(i for i in answers if i in key)
    texts = [key[i]["text"] for i in ids]
    truth = [key[i]["truth"] for i in ids]
    seen = [key[i]["text"].strip() in trained for i in ids]
    human = [answers[i] for i in ids]

    got = [EMOTIONS[j] for j in predict(texts, clf)]
    boot = load_config("prereg.lock")["metrics"]["bootstrap"]

    def acc(mask: list[bool], pred: list[str]) -> tuple[float, dict, int]:
        keep = [k for k, m in enumerate(mask) if m]
        vals = [1.0 if pred[k] == truth[k] else 0.0 for k in keep]
        ci = cluster_bootstrap_mean(vals, [key[ids[k]]["image_id"] for k in keep],
                                    n_resamples=boot["resamples"], ci=boot["ci"], seed=42)
        return sum(vals) / len(vals), ci, len(vals)

    all_true = [True] * len(ids)
    unseen = [not s for s in seen]
    decided = [h != "neutral" for h in human]
    dec_unseen = [d and u for d, u in zip(decided, unseen)]

    c_all, c_all_ci, n_all = acc(all_true, got)
    c_un, c_un_ci, n_un = acc(unseen, got)
    c_dec, c_dec_ci, n_dec = acc(dec_unseen, got)
    h_dec, h_dec_ci, _ = acc(dec_unseen, human)

    n = len(ids)
    n_neutral = sum(1 for h in human if h == "neutral")
    nr = n_neutral / n
    h_raw = sum(1 for h, t in zip(human, truth) if h == t) / n
    h_decided_all = (sum(1 for h, t, d in zip(human, truth, decided) if d and h == t)
                     / max(1, sum(decided)))
    fce = h_decided_all * (1 - nr) + (1 / len(EMOTIONS)) * nr

    print(f"{n} captions, the same text both judges saw")
    print(f"  {sum(seen)} are in the classifier's training pool and are EXCLUDED below")
    print(f"  reader declined on {n_neutral} ({nr:.1%}); the classifier cannot decline\n")

    print("FAIR COMPARISON -- the classifier must always choose, so compare it to FCE")
    print(f"  reader  FCE          {fce:.4f}")
    print(f"  classifier           {c_un:.4f}  [{c_un_ci['lo']:.4f}, {c_un_ci['hi']:.4f}]"
          f"   n={n_un}")
    print(f"  difference           {c_un - fce:+.4f}   "
          f"({'classifier ahead' if c_un > fce else 'reader ahead'})")
    print(f"  with the 6 overlaps  {c_all:.4f}  n={n_all}   "
          f"(shift {c_all - c_un:+.4f})\n")

    print("ON THE CAPTIONS THE READER WAS WILLING TO PLACE -- flatters both, so reported beside")
    print(f"  reader               {h_dec:.4f}")
    print(f"  classifier           {c_dec:.4f}   n={n_dec}")
    print(f"  difference           {c_dec - h_dec:+.4f}\n")

    print(f"NOT COMPARABLE, for completeness: reader raw {h_raw:.4f} "
          f"(neutrals counted wrong)\n")

    # Agreement, which accuracy cannot show: two judges can score alike and disagree often.
    dk = [k for k, m in enumerate(dec_unseen) if m]
    ha = [human[k] for k in dk]
    ca = [got[k] for k in dk]
    agree = sum(1 for x, y in zip(ha, ca) if x == y) / max(1, len(dk))
    kap = kappa(ha, ca, EMOTIONS)
    print("AGREEMENT, not accuracy -- do they read the same thing?")
    print(f"  same label on         {agree:.1%} of {len(dk)} decided items")
    print(f"  Cohen's kappa         {kap:+.4f}" if kap is not None else "  kappa undefined")
    both_wrong = sum(1 for k in dk if human[k] != truth[k] and got[k] != truth[k])
    only_h = sum(1 for k in dk if human[k] == truth[k] and got[k] != truth[k])
    only_c = sum(1 for k in dk if human[k] != truth[k] and got[k] == truth[k])
    print(f"  reader right, classifier wrong  {only_h}")
    print(f"  classifier right, reader wrong  {only_c}")
    print(f"  both wrong                      {both_wrong}")

    by_img: dict[str, dict] = {}
    for k in range(n):
        if not unseen[k]:
            continue
        img = key[ids[k]]["image_id"]
        by_img.setdefault(img, {"image_id": img, "captions": {}})
        by_img[img]["captions"][truth[k]] = texts[k]
    anchor = None
    try:
        anchor = keyword_rule_accuracy(list(by_img.values()))
        print(f"\n  keyword anchor on this same text  {anchor['accuracy']:.4f} "
              f"(n={anchor['n_cells']:,})")
    except ValueError as e:
        print(f"\n  anchor not computable here: {e}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "post_hoc": True, "decided": "2026-08-25",
        "why": "P5 compared a human on corpus captions against the classifier on generated "
               "captions -- different text and a different judge. This runs the classifier "
               "over the identical captions the reader saw.",
        "n_items": n, "n_in_training_pool_excluded": sum(seen),
        "reader": {"fce": round(fce, 4), "decided_only": round(h_decided_all, 4),
                   "raw": round(h_raw, 4), "neutral_rate": round(nr, 4)},
        "classifier": {"accuracy_excluding_overlap": round(c_un, 4), "ci": c_un_ci,
                       "n": n_un, "accuracy_all_items": round(c_all, 4)},
        "fair_difference_classifier_minus_reader_fce": round(c_un - fce, 4),
        "decided_subset": {"reader": round(h_dec, 4), "classifier": round(c_dec, 4),
                           "n": n_dec},
        "agreement": {"same_label_rate": round(agree, 4),
                      "cohens_kappa": round(kap, 4) if kap is not None else None,
                      "reader_only_right": only_h, "classifier_only_right": only_c,
                      "both_wrong": both_wrong},
        "keyword_anchor_same_text": anchor["accuracy"] if anchor else None,
        "caveat": "Both judges read CORPUS captions. This makes human and classifier "
                  "comparable to each other; it does not make either comparable to an "
                  "arm's accuracy on GENERATED captions.",
    }, indent=2))
    print(f"\nwrote {rel(out)}")


if __name__ == "__main__":
    main()
