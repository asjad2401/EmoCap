"""Arms decided AFTER the ``prereg-v2`` tag. Deliberately not in ``arms.py``.

``emocap.data.arms.ARMS`` holds the registered six, and ``configs/prereg.lock.yaml``
names the same six in the same order. A seventh entry there would make the lock disagree
with the code, and -- far worse -- would make a post-hoc arm indistinguishable from a
registered one to anyone reading this repository later. Everything in this module is
post-hoc *by construction*, and the paper must label it so wherever it is reported.

What ``S_paired_matched`` is for
-------------------------------
As registered, P3 compares ``S_paired5`` against ``S_unpaired`` and moves three things at
once: 9x the cells, roughly twice the images, **and** the paired structure. The prereg is
candid about this -- it calls P3a "a sanity check on the extra data" -- so as registered
the claim is about volume, not about pairing.

``S_paired_matched`` holds volume fixed. 878 images x 5 registers = **4,390 cells**,
exactly ``S_unpaired``'s size. If the paired arm clears its keyword anchor and the
unpaired arm does not at identical cell counts, pairing is isolated from volume, and the
finding stops being "more data helps" and becomes a claim about *parallel* data.

Image count is the one thing that cannot also be held fixed, and that is arithmetic
rather than an oversight: 4,390 cells is either 4,390 images with one register each or
878 images with five. Trading images for registers **is** what pairing means here, so the
comparison is stated that way rather than dressed up as a clean single-variable contrast.

The arm is nested inside the registered ones, on purpose
-------------------------------------------------------
Its 878 images are the **first 878** of the same ``sha1("unpaired-v1" + image_id)``
ranking that ``S_unpaired`` draws its 4,390 from, and each one contributes the same source
caption ``pick_caption_idx`` already chose. Three containments follow, and
``scripts/build_posthoc_arms.py`` asserts all three:

* its images are a strict subset of ``S_unpaired``'s images;
* every ``S_unpaired`` cell on a shared image reappears here, same source caption;
* its cells are a strict subset of ``S_paired5``'s cells.

So nothing new is sampled. The arm is a *view* of data the registered arms already used,
which is why it costs no generation and why "the post-hoc arm got luckier cells" is not
available as an explanation of whatever it shows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

# Private names are imported on purpose. Re-deriving the hash function or the cell
# constructor here would let two definitions of "the same image" drift apart, and the
# entire value of this arm rests on it being nested inside the registered ones.
from emocap.data.arms import _cells, _h, load_corpus, pick_caption_idx, shared_images

__all__ = ["POSTHOC_ARMS", "build_s_paired_matched"]

#: Post-hoc arm names. Never merged into ``ARMS``; see the module docstring.
POSTHOC_ARMS = ("S_paired_matched",)


def build_s_paired_matched(
    *,
    corpus: Mapping[tuple[str, int], Mapping],
    v1_captions: Mapping[str, Mapping[str, str]],
    excluded: frozenset[str] | set[str] = frozenset(),
    per_register: int = 878,
) -> list[dict]:
    """``per_register`` images x 5 registers, drawn from ``S_unpaired``'s own ranking.

    ``per_register`` is the *image* count here and the *cells-per-register* count in the
    1/image arms. Both give 4,390 cells, which is the point: the default matches
    ``S_unpaired`` exactly, and passing anything else breaks the match this arm exists for.
    """
    full = shared_images(corpus, v1_captions, excluded)
    images = sorted(full, key=lambda i: _h("unpaired-v1", i))[:per_register]

    cells: list[dict] = []
    for img in images:
        k = pick_caption_idx(img)
        cells += _cells(corpus[(img, k)], img, k, "S")
    return cells


def load_v1_pilot(path: Path) -> dict[str, dict[str, str]]:
    """The archived v1 pilot as ``image_id -> {register: caption}``.

    Duplicated from ``scripts/build_arms.py`` rather than imported, because scripts are
    not importable modules. The path it is called with is the thing that matters, and
    ``scripts/build_posthoc_arms.py`` passes the same archived CSV the registered build
    uses -- not ``data/generated/captions_raw.jsonl``, which is a different corpus.
    """
    import csv
    import gzip

    from emocap.data.prompt import EMOTIONS  # noqa: F401  (documents the expected keys)

    out: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            text = (r.get("emotion_caption") or "").strip()
            if text:
                out.setdefault(r["image_id"], {})[r["emotion"]] = text
    return out
