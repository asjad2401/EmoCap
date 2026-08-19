"""The lexical-shortcut anchor.

`configs/prereg.lock.yaml` pins `lexical_shortcut_value` and names its estimator. An
anchor recorded as a number with no implementation cannot be checked or recomputed, so
the estimator lives here and the lock's `lexical_shortcut_estimator` string must match
:data:`ESTIMATOR_ID`.

**What it measures.** How much of the primary metric is obtainable by keyword spotting
alone. Fit the top-k most register-discriminative words per register on training folds,
then classify held-out captions by counting keyword hits. If that scores well above
chance, a classifier credited with "recognising the emotional register" is substantially
just detecting vocabulary, and the captioning model's real contribution is the margin
above this anchor rather than the raw accuracy.

Measured on the Part 0 audit sets (chance is 0.200):

    v1 (original prompt)   0.639
    v3                     0.431
    v4                     0.468
    v5 (final)             0.395
    Sonnet reference       0.347

An earlier 0.539 in the lock came from a different, undocumented estimator and is not
comparable to any of these -- which is the reason this module exists.

**THE ANCHOR IS STRONGLY SAMPLE-SIZE DEPENDENT. A POINT ESTIMATE IS MEANINGLESS WITHOUT
ITS n.** Measured on the same generated captions:

    4,500 cells      0.443 +/- 0.010   (SD over 10 image-disjoint subsamples)
    124,750 cells    0.344

A keyword rule fitted on few images transfers well to held-out images drawn from that same
narrow pool; widen the pool and it degrades. The anchor therefore *falls* by ~10 points as
n grows, and comparing two corpora at different n is invalid. Three claims on this project
were retracted for exactly that error (see docs/lab-notebook.md, 2026-08-19).

Two consequences, enforced below rather than left to discipline:

* :func:`keyword_rule_accuracy` requires an explicit ``n_cells`` whenever the result will
  be compared against another corpus, and returns the SD across image-disjoint subsamples
  rather than a single number.
* :func:`keyword_rule_curve` reports the anchor as a function of n, so its dependence is
  visible by construction instead of discovered later.

This also matters beyond this project: FlickrStyle10K is 7K images and SentiCap 2,360 --
precisely the scale at which a keyword baseline is most inflated, and precisely where such
baselines are computed.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

from emocap.data.prompt import EMOTIONS

__all__ = [
    "ESTIMATOR_ID",
    "STOPWORDS",
    "keyword_rule_accuracy",
    "keyword_rule_matched",
    "keyword_rule_curve",
    "top_keywords",
]

#: Must equal `anchors.lexical_shortcut_estimator` in configs/prereg.lock.yaml. Any
#: change to the procedure below requires a new id and a deviations.md entry, because
#: figures computed under different procedures must never be compared.
ESTIMATOR_ID = "cv5_by_image_id_3seeds_fractional_ties"

#: Function words plus the counting words Flickr8k captions are full of. Excluded so the
#: anchor measures *register* vocabulary rather than scene vocabulary.
STOPWORDS = frozenset(
    "a an the of in on at to and with his her their its is are as for into while by "
    "two one over near out up down from that this it he she they them there".split()
)

_WORD_RE = re.compile(r"[a-z']+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}


def top_keywords(
    records: Iterable[Mapping], *, top_k: int = 25, min_doc_freq: int = 4
) -> dict[str, set[str]]:
    """The ``top_k`` most register-discriminative words per register.

    Scored by ``count_in_register / (count_overall + 3)``: a word earns its place by
    appearing disproportionately in one register, not merely often. The +3 damps words
    seen only a handful of times, and ``min_doc_freq`` drops the long tail entirely --
    without both, hapax legomena dominate and the anchor measures noise.
    """
    per_register: dict[str, Counter] = {e: Counter() for e in EMOTIONS}
    for rec in records:
        for emotion, text in rec.get("captions", {}).items():
            if emotion in per_register:
                # set(), not list: a word repeated inside one caption must not count twice
                per_register[emotion].update(_content_words(str(text)))

    overall: Counter = Counter()
    for counts in per_register.values():
        overall.update(counts)

    out: dict[str, set[str]] = {}
    for emotion in EMOTIONS:
        scored = (
            (word, per_register[emotion][word] / (overall[word] + 3))
            for word in per_register[emotion]
            if overall[word] >= min_doc_freq
        )
        out[emotion] = {w for w, _ in sorted(scored, key=lambda kv: (-kv[1], kv[0]))[:top_k]}
    return out


def keyword_rule_accuracy(
    records: Sequence[Mapping],
    *,
    top_k: int = 25,
    folds: int = 5,
    seeds: Sequence[int] = (0, 1, 2),
    min_doc_freq: int = 4,
) -> dict:
    """Accuracy of a top-``top_k`` keyword rule at recovering the intended register.

    Cross-validated **by ``image_id``**, never by caption: the 25 cells of one image
    share their source captions, so splitting by row would put near-duplicate text on
    both sides of the fold and inflate the anchor. Averaged over ``seeds`` because a
    50-image sample is small enough for the fold assignment to matter.

    Ties are credited fractionally (``1/len(tied)`` when the true register is among the
    tied winners). Awarding a tie in full would flatter a rule that discriminates
    nothing; awarding zero would understate it. Chance is ``1 / len(EMOTIONS)``.

    Returns the mean, the per-seed values, and the chance level.
    """
    records = [r for r in records if r.get("captions")]
    if not records:
        raise ValueError("no records with captions")

    image_ids = sorted({str(r["image_id"]) for r in records})
    if len(image_ids) < folds:
        raise ValueError(f"{len(image_ids)} images cannot support {folds} folds")

    per_seed: list[float] = []
    for seed in seeds:
        shuffled = list(image_ids)
        random.Random(seed).shuffle(shuffled)
        fold_scores: list[float] = []
        for fold in range(folds):
            held_out = set(shuffled[fold::folds])
            train = [r for r in records if str(r["image_id"]) not in held_out]
            test = [r for r in records if str(r["image_id"]) in held_out]
            if not train or not test:
                continue
            keywords = top_keywords(train, top_k=top_k, min_doc_freq=min_doc_freq)

            correct = 0.0
            total = 0
            for rec in test:
                for emotion, text in rec.get("captions", {}).items():
                    if emotion not in keywords:
                        continue
                    words = _content_words(str(text))
                    hits = {e: len(words & keywords[e]) for e in EMOTIONS}
                    best = max(hits.values())
                    tied = [e for e in EMOTIONS if hits[e] == best]
                    if emotion in tied:
                        correct += 1.0 / len(tied)
                    total += 1
            if total:
                fold_scores.append(correct / total)
        if fold_scores:
            per_seed.append(sum(fold_scores) / len(fold_scores))

    if not per_seed:
        raise ValueError("no fold produced a score")
    return {
        "estimator": ESTIMATOR_ID,
        "accuracy": round(sum(per_seed) / len(per_seed), 4),
        "per_seed": [round(x, 4) for x in per_seed],
        "chance": round(1 / len(EMOTIONS), 4),
        "n_images": len(image_ids),
        "n_cells": sum(len(r.get("captions", {})) for r in records),
        "top_k": top_k,
        "folds": folds,
        "seeds": list(seeds),
    }


def _subsample_by_image(
    records: Sequence[Mapping], n_cells: int, seed: int
) -> list[Mapping]:
    """Take whole images until ``n_cells`` cells are collected.

    Whole images, never individual cells: a partial image would put its remaining cells
    nowhere, and the fold discipline in :func:`keyword_rule_accuracy` assumes an image is
    wholly present or wholly absent.
    """
    by_image: dict[str, list[Mapping]] = {}
    for r in records:
        by_image.setdefault(str(r["image_id"]), []).append(r)
    order = sorted(by_image)
    random.Random(seed).shuffle(order)
    out: list[Mapping] = []
    cells = 0
    for img in order:
        out.extend(by_image[img])
        cells += sum(len(r.get("captions") or {}) for r in by_image[img])
        if cells >= n_cells:
            break
    return out


def keyword_rule_matched(
    records: Sequence[Mapping],
    *,
    n_cells: int,
    n_subsamples: int = 10,
    top_k: int = 25,
    min_doc_freq: int = 4,
    folds: int = 5,
) -> dict:
    """The anchor at a FIXED cell count, with its spread across subsamples.

    **Use this, not :func:`keyword_rule_accuracy`, whenever two corpora are compared.**
    The anchor moves ~10 points between 4.5k and 125k cells, so an unmatched comparison
    measures the size difference rather than the corpora. Reporting the SD alongside is
    equally load-bearing: a single subsample of our captions gave 0.419 where the mean is
    0.443, and a claim was built on that draw before the spread was checked.

    Returns mean, SD, min, max and every subsample value.
    """
    if n_subsamples < 2:
        raise ValueError(
            "n_subsamples must be >= 2 -- the point of this function is the spread, and "
            "one draw is what produced the retracted 0.419"
        )
    values: list[float] = []
    for seed in range(n_subsamples):
        sub = _subsample_by_image(records, n_cells, seed)
        got = sum(len(r.get("captions") or {}) for r in sub)
        if got < n_cells * 0.9:
            raise ValueError(
                f"only {got} cells available, asked for {n_cells}; this corpus cannot "
                f"support the requested matched size"
            )
        values.append(
            keyword_rule_accuracy(sub, top_k=top_k, min_doc_freq=min_doc_freq,
                                  folds=folds)["accuracy"]
        )
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return {
        "estimator": ESTIMATOR_ID,
        "n_cells": n_cells,
        "n_subsamples": n_subsamples,
        "mean": round(mean, 4),
        "sd": round(var ** 0.5, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "values": [round(v, 4) for v in values],
        "chance": round(1 / len(EMOTIONS), 4),
        "top_k": top_k,
        "min_doc_freq": min_doc_freq,
    }


def keyword_rule_curve(
    records: Sequence[Mapping],
    *,
    cell_counts: Sequence[int] = (2000, 5000, 10000, 25000, 50000, 100000),
    n_subsamples: int = 3,
    top_k: int = 25,
    min_doc_freq: int = 4,
) -> list[dict]:
    """The anchor as a function of n, so its dependence is visible rather than latent.

    Any ``cell_counts`` entry the corpus cannot supply is skipped rather than silently
    truncated to the corpus size, which would report a smaller n's value under a larger
    n's label.
    """
    total = sum(len(r.get("captions") or {}) for r in records)
    out: list[dict] = []
    for n in sorted(cell_counts):
        if n > total:
            continue
        r = keyword_rule_matched(records, n_cells=n, n_subsamples=n_subsamples,
                                 top_k=top_k, min_doc_freq=min_doc_freq)
        out.append({"n_cells": n, "mean": r["mean"], "sd": r["sd"]})
    if not out:
        raise ValueError(f"corpus has {total} cells, fewer than the smallest requested")
    return out
