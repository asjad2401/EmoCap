"""Caption quality metrics, shared by the stage-02 probe and stage-03 QA gates.

These are the deterministic measurements. They exist here rather than in a script
because the probe uses them to *choose* a generation configuration and stage 03
uses the same ones to *gate* the full run -- and those two must not drift apart.

The metrics answer four questions:

* **Grounding** -- did the rewrite keep the caption's content, and what did it add?
  Added content is not automatically bad (mood words are the point) but it is the
  surface where hallucination lives, so it is reported rather than scored away.
* **Register divergence** -- are the five registers actually different from each
  other? v1's LSTM baseline emitted one identical caption for all five emotions;
  if the *training targets* are near-identical the study has nothing to detect.
* **Length** -- against the 8-24 word target, so truncation and rambling are visible.
* **Configuration divergence** -- how much two generation configs disagree on the
  same input. This is what makes "does 512px beat 384px?" and "does the image
  change anything at all?" answerable from a paired sample.
"""

from __future__ import annotations

import re
import statistics as stats
from typing import Iterable, Mapping, Sequence

__all__ = [
    "STOPWORDS",
    "content_words",
    "content_recall",
    "novel_content",
    "grounding_report",
    "pairwise_similarity",
    "register_divergence",
    "length_stats",
    "config_divergence",
]

#: Function words carry no content, so they would inflate every overlap measure.
STOPWORDS = frozenset(
    """a an the is are was were be been being am i you he she it we they this that
    these those of in on at to for with by from as and or but if while has have had
    do does did will would can could should may might must shall there his her its
    their s t not no very just also then than so up out down off over under again
    into onto near about across along around
    """.split()
)

_WORD_RE = re.compile(r"[a-z][a-z'-]*")


def content_words(text: str) -> set[str]:
    """Lowercase content words: alphabetic, 3+ characters, not a stopword."""
    return {
        w
        for w in _WORD_RE.findall(str(text).lower())
        if len(w) >= 3 and w not in STOPWORDS
    }


# ── grounding ───────────────────────────────────────────────────────────────


def content_recall(source: str, generated: str) -> float:
    """Fraction of the source's content words that survive into the rewrite.

    Low recall means the rewrite drifted off the caption. It is allowed to drop
    detail to fit the word budget, so this is a distribution to inspect rather
    than a hard threshold.
    """
    src = content_words(source)
    if not src:
        return 0.0
    return len(src & content_words(generated)) / len(src)


def novel_content(source: str, generated: str) -> set[str]:
    """Content words in the rewrite that appear nowhere in the source.

    Some of this is the mood vocabulary we asked for ("bright", "braced"); some is
    invention ("sunlight" when no light is mentioned). The probe prints these so a
    human can tell which, because no regex can.
    """
    return content_words(generated) - content_words(source)


def grounding_report(source: str, generated: str) -> dict:
    novel = novel_content(source, generated)
    gen = content_words(generated)
    return {
        "content_recall": round(content_recall(source, generated), 4),
        "novel_words": sorted(novel),
        "novel_rate": round(len(novel) / max(1, len(gen)), 4),
        "n_content_words": len(gen),
    }


# ── register divergence ─────────────────────────────────────────────────────


def pairwise_similarity(a: str, b: str) -> float:
    """Jaccard overlap of content words. 1.0 = same content, 0.0 = disjoint."""
    A, B = content_words(a), content_words(b)
    if not A and not B:
        return 1.0
    union = A | B
    return len(A & B) / len(union) if union else 1.0


def register_divergence(captions: Mapping[str, str]) -> dict:
    """How distinguishable the registers are for one source caption.

    Returns mean and max pairwise similarity plus the most-similar pair. High
    similarity means the generator produced five paraphrases rather than five
    registers, and the ablation would have nothing to detect.
    """
    items = [(k, v) for k, v in captions.items() if v and str(v).strip()]
    if len(items) < 2:
        return {"n": len(items), "mean_similarity": None, "max_similarity": None,
                "most_similar_pair": None}
    sims = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sims.append((pairwise_similarity(items[i][1], items[j][1]),
                         (items[i][0], items[j][0])))
    top = max(sims, key=lambda s: s[0])
    return {
        "n": len(items),
        "mean_similarity": round(stats.mean(s for s, _ in sims), 4),
        "max_similarity": round(top[0], 4),
        "most_similar_pair": list(top[1]),
    }


# ── length ──────────────────────────────────────────────────────────────────


def length_stats(texts: Iterable[str], *, lo: int = 8, hi: int = 24) -> dict:
    lens = [len(str(t).split()) for t in texts if t and str(t).strip()]
    if not lens:
        return {"n": 0}
    lens_sorted = sorted(lens)
    return {
        "n": len(lens),
        "mean": round(stats.mean(lens), 2),
        "p05": lens_sorted[int(0.05 * (len(lens) - 1))],
        "p50": lens_sorted[len(lens) // 2],
        "p95": lens_sorted[int(0.95 * (len(lens) - 1))],
        "min": lens_sorted[0],
        "max": lens_sorted[-1],
        "in_target": round(sum(1 for x in lens if lo <= x <= hi) / len(lens), 4),
    }


# ── configuration divergence ────────────────────────────────────────────────


def config_divergence(
    left: Sequence[str], right: Sequence[str]
) -> dict:
    """How much two generation configs disagree on the same inputs, pairwise.

    ``left[i]`` and ``right[i]`` must be captions for the same (image, source
    caption, register). This is what answers the two questions the probe exists
    for:

    * **Does resolution matter?** If 224px and 512px outputs are near-identical,
      the extra image tokens buy nothing and we take the cheap one.
    * **Does the image matter at all?** If text-only and multimodal outputs are
      near-identical, the multimodal decision is unjustified and we save 5x the
      input tokens by dropping it.

    ``identical_rate`` is exact string match; ``mean_similarity`` is content-word
    Jaccard, which ignores pure rewording.
    """
    pairs = [
        (str(a).strip(), str(b).strip())
        for a, b in zip(left, right)
        if a and b and str(a).strip() and str(b).strip()
    ]
    if not pairs:
        return {"n": 0}
    sims = [pairwise_similarity(a, b) for a, b in pairs]
    return {
        "n": len(pairs),
        "identical_rate": round(sum(1 for a, b in pairs if a == b) / len(pairs), 4),
        "mean_similarity": round(stats.mean(sims), 4),
        "median_similarity": round(stats.median(sims), 4),
        "frac_similarity_above_0.8": round(
            sum(1 for s in sims if s > 0.8) / len(sims), 4
        ),
    }
