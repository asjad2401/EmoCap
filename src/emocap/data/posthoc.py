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

What ``S_unpaired_scaled`` is for, and why it came second
--------------------------------------------------------
``S_paired_matched`` answered half the question and went to the floor: paired structure at
4,390 cells does nothing. That establishes volume is *necessary*. It says nothing about
whether pairing adds anything on top of volume -- and without that, "more data helps" and
"parallel data helps" remain the same observation.

``S_unpaired_scaled`` is the other half. It matches ``S_paired5`` on images, on cell count,
and on cells per image, and differs in one thing only:

    S_paired5           the 5 cells of an image are ONE source caption in FIVE registers
    S_unpaired_scaled   the 5 cells of an image are FIVE source captions in ONE register

That is the single-variable contrast the registered P3 could not be. It also extends
``S_unpaired`` into a volume ladder at constant unpaired structure -- 4,390 to 40,235 cells
with the register contrast absent at both ends -- because the 4,390 images ``S_unpaired``
already used keep the register it gave them.

A correction belongs here. This arm was earlier described, in a commit message, as
impossible to build from Flickr8k: an unpaired arm needs one image per cell, so 8,047
images cap it at 8,047 cells. That was wrong. "Unpaired" in this design means one
*register* per image, not one *cell* per image, and every image carries five source
captions -- so 8,047 x 5 = 40,235 is available, exactly ``S_paired5``'s size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

# Private names are imported on purpose. Re-deriving the hash function or the cell
# constructor here would let two definitions of "the same image" drift apart, and the
# entire value of this arm rests on it being nested inside the registered ones.
from emocap.data.arms import (_cells, _h, _unpaired_assignment, fold_of, load_corpus,
                             pick_caption_idx, shared_images)
from emocap.data.prompt import EMOTIONS

__all__ = ["POSTHOC_ARMS", "build_s_paired_matched", "build_s_unpaired_scaled"]

#: Post-hoc arm names. Never merged into ``ARMS``; see the module docstring.
POSTHOC_ARMS = ("S_paired_matched", "S_unpaired_scaled")


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


def build_s_unpaired_scaled(
    *,
    corpus: Mapping[tuple[str, int], Mapping],
    v1_captions: Mapping[str, Mapping[str, str]],
    excluded: frozenset[str] | set[str] = frozenset(),
    per_register: int = 878,
) -> list[dict]:
    """``S_paired5``'s size and images, with the register contrast removed.

    **This is the arm that actually tests P3.** ``S_paired_matched`` asked whether pairing
    works at a small budget -- it does not. This asks the other half: whether the volume
    works *without* pairing. Until one of the two exists, "more data helps" and "parallel
    data helps" are the same observation.

    The contrast is as clean as this corpus allows, and cleaner than the registered P3:

        S_paired5           8,047 images x 5 cells = 40,235.  The 5 cells of one image
                            hold ONE source caption in FIVE registers.
        S_unpaired_scaled   8,047 images x 5 cells = 40,235.  The 5 cells of one image
                            hold FIVE source captions in ONE register.

    Same images, same cell count, same cells-per-image, same source corpus, same training
    budget. The single difference is whether an image's five cells vary by *register* or by
    *source text*. Nothing else in the study isolates the paired structure at matched volume.

    **It nests on top of ``S_unpaired``.** The first 4,390 images keep the register
    ``_unpaired_assignment`` already gave them, so every ``S_unpaired`` cell reappears here
    and the pair forms a volume ladder at constant (unpaired) structure: 4,390 -> 40,235
    cells, register contrast absent at both ends. If the margin appears at the top of that
    ladder, volume alone is sufficient and pairing is not the story.

    Registers stay balanced within each fold, for the reason ``_unpaired_assignment`` gives:
    at ~1,609 images per register spread over five folds, assignment by hash alone leaves a
    fold short in one register by enough to matter.
    """
    full = shared_images(corpus, v1_captions, excluded)

    # Inherit S_unpaired's assignment, then extend it over the rest of the universe.
    assign = dict(_unpaired_assignment(full, per_register))
    rest = [i for i in sorted(full, key=lambda i: _h("unpaired-v1", i)) if i not in assign]
    by_fold: dict[int, list[str]] = {}
    for img in rest:
        by_fold.setdefault(fold_of(img), []).append(img)
    turn = 0
    for fold in sorted(by_fold):
        for img in by_fold[fold]:
            assign[img] = EMOTIONS[turn % len(EMOTIONS)]
            turn += 1

    cells: list[dict] = []
    for img in full:
        emo = assign[img]
        for k in range(5):
            cell = next((c for c in _cells(corpus[(img, k)], img, k, "S")
                         if c["emotion"] == emo), None)
            if cell:
                cells.append(cell)
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

    out: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            text = (r.get("emotion_caption") or "").strip()
            if text:
                out.setdefault(r["image_id"], {})[r["emotion"]] = text
    return out
