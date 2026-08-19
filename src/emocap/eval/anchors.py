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
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

from emocap.data.prompt import EMOTIONS

__all__ = ["ESTIMATOR_ID", "STOPWORDS", "keyword_rule_accuracy", "top_keywords"]

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
