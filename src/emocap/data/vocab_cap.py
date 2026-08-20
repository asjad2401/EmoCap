"""Corpus-level vocabulary cap: stop one word from becoming a register's label.

Why this cannot live in the prompt
----------------------------------
Prompt v6 replaced v5's prohibitions with technique -- for ``sad``, *"find what the scene
has LESS of: space, company, colour, motion, warmth"*. The model read that as a word list.
Measured on 695 records:

===========  ====================================  ==========================
register     top word, share of its captions       same word, other registers
===========  ====================================  ==========================
joyful       ``bright``   34%                      3%
romantic     ``soft``     23%                      0%
sad          ``empty``    20%                      1%
===========  ====================================  ==========================

**28 words** clear 6% in-register with >=2x lift, against **5** for v5. Masking them drops
the lexical anchor 0.638 -> 0.504, so roughly 70% of v6's anchor rise over v5 (0.449) is
carried by template vocabulary rather than by how the scene is described.

A prompt cannot fix that. Each call sees ONE image and cannot observe what the other eight
thousand calls emitted, so "vary your vocabulary" asks the model to control a statistic it
has no access to -- the same reason technique-in-words collapsed into a template. Corpus
frequency is only visible to the pipeline, so the cap lives here: measure what the corpus
is over-using, then forbid it *by name* in later prompts.

Why the cap is not uniform across registers
-------------------------------------------
Some of these words earn their keep. Per-register, classifier accuracy minus the human
blind-guess score:

===========  ==========  =====  ======  ===============================================
register     classifier  human  gap     reading
===========  ==========  =====  ======  ===============================================
joyful       0.902       0.444  +0.458  ``bright`` works for the model, not the reader
tense        0.835       0.500  +0.335
romantic     0.804       0.500  +0.304
humorous     0.840       0.667  +0.173
sad          0.886       0.722  +0.164  ``empty``/``cold``/``grey`` read as sad to a person
===========  ==========  =====  ======  ===============================================

A large gap means the word is a shortcut the model exploits and the reader does not; a
small gap means it is doing legitimate work. So ``joyful`` is capped hard and ``sad``
gently -- banning ``empty`` and ``cold`` risks pushing ``sad`` back toward the 0.20
chance-level score it had under v5.

**This is a corpus-construction rule and must be pre-registered as one**, with its caps
fixed in `configs/data.yaml` before the corpus is built. Tuning caps after seeing the
anchor would be fitting the instrument to the result.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

from emocap.data.prompt import EMOTIONS

__all__ = [
    "DEFAULT_CAPS",
    "STOPWORDS",
    "banned_hit",
    "content_words",
    "register_frequencies",
    "over_cap_words",
    "cap_report",
    "stem",
]

#: Per-register document-frequency cap, keyed by register. Set from the classifier-human
#: gap above: the wider the gap, the more the register's vocabulary is a model-only
#: shortcut and the tighter the cap.
#:
#: These are DELIBERATELY not a single global number. A uniform cap would treat
#: ``bright`` (gap +0.458, pure shortcut) and ``empty`` (gap +0.164, genuine signal) as
#: the same problem.
DEFAULT_CAPS: dict[str, float] = {
    "joyful": 0.08,
    "tense": 0.10,
    "romantic": 0.10,
    "humorous": 0.14,
    "sad": 0.15,
}

#: Minimum in-register / elsewhere ratio before a word counts as register-specific.
#: Without it the cap would ban scene vocabulary: ``white`` and ``water`` sit near 9% in
#: *all five* registers, carry no label information, and are exactly what the captions
#: are supposed to be about.
DEFAULT_MIN_LIFT = 2.0

#: Function words and ubiquitous Flickr8k subjects. Excluded because capping them would
#: fight the dataset rather than the template -- these photographs really are mostly of
#: people, dogs and children.
STOPWORDS = frozenset("""
a an the and or but of in on at to for with within without his her its their they them he
she it is are was were be been as by from into onto while during over under up down out
off through across against near behind beside between around above below has have had had
that this these those there here what which who whom whose not no nor so then than too very
one two three four five man woman men women boy girl boys girls people person child children
kid kids dog dogs puppy guy lady group crowd someone something wearing wears wear
""".split())

_WORD_RE = re.compile(r"[a-z]+")

#: Suffixes stripped so a cap counts a word and its inflections as one thing. Banning
#: `soft` while `softly` stayed legal is not a loophole the model had to look for -- it
#: took it immediately: `romantic` was rebuilt on `gentle`/`softly`/`gently` after `soft`
#: and `gently` were capped, and 8 of 28 `soft*` uses evaded an exact-match ban.
#:
#: Deliberately crude. A real stemmer would collapse words this corpus needs to keep
#: apart, and the cap only has to catch a model reaching for the nearest form of a word
#: it was just denied.
#: ``le`` is here for the English `-le`/`-ly` pair: `gently` stems to `gent`, so without
#: it `gentle` survives a ban on `gently` -- which is exactly what `romantic` did.
_SUFFIXES = ("ingly", "edly", "ness", "less", "ing", "ely", "est", "ly", "le", "ed",
             "er", "es", "s", "y")


def stem(word: str) -> str:
    """A crude stem: strip one known suffix, keeping at least four characters.

    The four-character floor stops `close` -> `clo` or `grey` -> `gre`, which would
    match far too much.
    """
    w = word.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            base = w[: -len(suf)]
            # English y -> i before a suffix: empty/emptied, happy/happier. Without this
            # `emptied` stems to `empti` and escapes a ban derived from `empty` -> `empt`.
            if base.endswith("i") and len(base) > 4:
                base = base[:-1]
            return base
    return w


def banned_hit(text: str, words: Sequence[str]) -> str | None:
    """The first token of ``text`` whose stem matches a banned word, or ``None``.

    Tokenises and compares **stems** rather than matching a regex against raw text, so
    enforcement and measurement use the identical rule -- :func:`content_words` stems the
    corpus the caps are computed from, and a mismatch between the two is what let the
    model keep a capped habit by changing an ending.

    A regex of ``stem + optional-suffix`` was tried first and is not equivalent: it misses
    forms whose ending is not in the suffix list (`emptied` = `empt` + `ied`) while
    risking hits on unrelated words that merely begin with a stem (`bright` inside
    `Brighton`). Comparing stems has neither failure.
    """
    if not words:
        return None
    banned = {stem(w) for w in words}
    for token in _WORD_RE.findall(text.lower()):
        if stem(token) in banned:
            return token
    return None


def content_words(text: str, *, min_len: int = 3) -> set[str]:
    """The distinct content words of one caption.

    A **set**, not a list: the cap is a *document* frequency ("what share of joyful
    captions contain this word"), so a caption saying "bright" twice must count once.
    Using raw counts would let one verbose caption look like a corpus-wide pattern.

    Words are reduced to :func:`stem` first, so `soft`, `softly` and `softer` are one
    entry. Measuring exact forms while banning exact forms let the model keep the habit
    and change only the ending.
    """
    return {stem(w) for w in _WORD_RE.findall(text.lower())
            if len(w) >= min_len and w not in STOPWORDS}


def register_frequencies(
    records: Iterable[Mapping],
    *,
    registers: Sequence[str] = EMOTIONS,
) -> tuple[dict[str, Counter], dict[str, int]]:
    """Document frequency per register, and the caption count per register.

    Returns ``(freq, totals)`` where ``freq[register][word]`` is how many of that
    register's captions contain the word.
    """
    freq: dict[str, Counter] = {r: Counter() for r in registers}
    totals: dict[str, int] = {r: 0 for r in registers}
    for rec in records:
        caps = rec.get("captions") or {}
        for r in registers:
            text = caps.get(r)
            if not text:
                continue
            totals[r] += 1
            freq[r].update(content_words(str(text)))
    return freq, totals


def over_cap_words(
    records: Iterable[Mapping],
    *,
    caps: Mapping[str, float] | None = None,
    min_lift: float = DEFAULT_MIN_LIFT,
    min_support: int = 40,
    max_per_register: int = 12,
    carry: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Which words each register is over-using, worst first.

    A word is over-cap when it clears that register's document-frequency cap **and** is
    at least ``min_lift`` times more common there than in any other register.

    ``min_support`` is the floor on captions-per-register before any word is banned:
    at 20 captions a single unlucky draw reads as a 15% pattern. Below it the function
    returns nothing rather than banning on noise -- which matters because generation
    calls this from the very first chunk, when the corpus is tiny.

    ``max_per_register`` bounds the list. An unbounded ban list is how v5 happened: v5
    forbade the obvious words and got five interchangeable captions, because a model
    denied every way of signalling a register stops signalling it. Capping the list keeps
    this a correction to the worst offenders, not a return to prohibition.

    ``carry`` is a previously-banned list that stays banned regardless of its current
    frequency. **Without it the cap is self-defeating**: banning `soft` drops its share to
    1.6%, which takes it back off the list, so the next run permits it and the model
    returns to it immediately -- `soft` went 22.7% (v6) -> 1.6% (v7, banned) -> 19.6% (v8,
    no longer over cap, so unbanned). The word never stopped being the model's first
    choice; only the measurement moved. Carried words are exempt from
    ``max_per_register`` so a long history cannot crowd out a live offender.
    """
    caps = dict(caps or DEFAULT_CAPS)
    freq, totals = register_frequencies(records)
    out: dict[str, list[str]] = {}
    for r in EMOTIONS:
        n = totals.get(r, 0)
        if n < min_support:
            # too little data to ADD a word, but a carried ban still stands
            out[r] = [stem(w) for w in (carry or {}).get(r, ())]
            continue
        cap = caps.get(r, 0.10)
        scored: list[tuple[float, str]] = []
        for word, count in freq[r].items():
            share = count / n
            if share < cap:
                continue
            elsewhere = max(
                (freq[o][word] / totals[o]) if totals.get(o) else 0.0
                for o in EMOTIONS if o != r
            )
            # A word absent elsewhere has infinite lift; treat it as maximally specific
            # rather than dividing by zero.
            lift = share / elsewhere if elsewhere > 0 else float("inf")
            if lift >= min_lift:
                scored.append((share, word))
        scored.sort(reverse=True)
        fresh = [w for _, w in scored[:max_per_register]]
        held = [stem(w) for w in (carry or {}).get(r, ())]
        # carried first, then this run's offenders, de-duplicated, order preserved
        out[r] = list(dict.fromkeys(held + fresh))
    return out


def cap_report(
    records: Iterable[Mapping],
    *,
    caps: Mapping[str, float] | None = None,
    min_lift: float = DEFAULT_MIN_LIFT,
    min_support: int = 40,
) -> list[dict]:
    """Per-word detail behind :func:`over_cap_words`, for logging and the manifest.

    Records the numbers the ban was made on, so a corpus can be audited after the fact
    without re-deriving them from the captions.
    """
    caps = dict(caps or DEFAULT_CAPS)
    freq, totals = register_frequencies(records)
    rows: list[dict] = []
    for r in EMOTIONS:
        n = totals.get(r, 0)
        if n < min_support:
            continue
        cap = caps.get(r, 0.10)
        for word, count in freq[r].items():
            share = count / n
            if share < cap:
                continue
            elsewhere = max(
                (freq[o][word] / totals[o]) if totals.get(o) else 0.0
                for o in EMOTIONS if o != r
            )
            lift = share / elsewhere if elsewhere > 0 else float("inf")
            if lift >= min_lift:
                rows.append({"register": r, "word": word, "share": round(share, 4),
                             "elsewhere": round(elsewhere, 4),
                             "lift": (None if lift == float("inf") else round(lift, 2)),
                             "cap": cap, "n_captions": n})
    rows.sort(key=lambda d: (-d["share"], d["register"]))
    return rows
