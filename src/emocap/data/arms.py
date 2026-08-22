"""Assemble the six data arms, deterministically, from three corpora.

The study compares *provenance* and *structure*, so the arms have to differ in exactly
one of those at a time and in nothing else. That is a construction problem, not a
training one, and it is solved here rather than in the training script so that the arms
are a hashable artifact on disk that every run reads.

Three structure classes, and the reason each exists
---------------------------------------------------
``25/image``  every register of every source caption. Only our corpus can supply it.
``5/image``   one source caption, all five of its registers. Holds the source text fixed
              and varies only the register -- so ``S_paired25`` vs ``S_paired5`` is a
              pure *count* comparison, and the 5/image arms are a strict subset of the
              25/image one rather than a differently-sampled corpus.
``1/image``   one register of one source caption, 878 per register. Matched at 4,390
              because that is what the human corpus supplies under the strict trait
              mapping, and matching on COUNT as well as structure is required or P1
              confounds "human data" with "half as much data".

Matching is by construction, not by sampling twice
--------------------------------------------------
``V1_paired5`` selects **the same images and the same ``caption_idx``** as ``S_paired5``;
``V1_unpaired`` selects the same images *and the same register per image* as
``S_unpaired``. The two corpora are rewrites of the identical Flickr8k source sentences,
so this makes the S-vs-V1 comparisons differ in generator and nothing else. Sampling each
arm independently from a shared pool would have left a sampling difference on top of the
provenance difference, which is the confound the whole design exists to avoid.

Everything is chosen by hashing the ``image_id``
------------------------------------------------
Fold, source-caption index, and register are all derived from ``sha1(image_id + salt)``.
A hash does not depend on iteration order, on how many images are in the pool, or on any
RNG whose state a later edit could disturb -- so an arm rebuilt after the corpus gains or
loses an image keeps every other image exactly where it was. That property is what makes
a half-finished study safe to resume, and it is why this is not ``random.shuffle``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from emocap.data.prompt import EMOTIONS

__all__ = ["ARMS", "fold_of", "pick_caption_idx", "build_all_arms", "write_arm"]

#: Arm names in the order the preregistration lists them. `arms` in prereg.lock.yaml
#: must match this exactly.
ARMS = ("S_paired25", "S_paired5", "S_unpaired", "V1_paired5", "V1_unpaired",
        "H_unpaired")

_FOLDS = 5


def _h(*parts: str) -> int:
    return int(hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:12], 16)


def fold_of(image_id: str, *, folds: int = _FOLDS, salt: str = "fold-v1") -> int:
    """Which CV fold an image belongs to. Split by image, never by caption."""
    return _h(salt, image_id) % folds


def pick_caption_idx(image_id: str, n: int = 5, *, salt: str = "cap-v1") -> int:
    """Which of the five source captions the 5/image arms use for this image."""
    return _h(salt, image_id) % n


def _load_jsonl(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[(r["image_id"], r["caption_idx"])] = r
    return out


def _cells(rec: Mapping, image_id: str, caption_idx: int, source: str) -> list[dict]:
    caps = rec.get("captions") or {}
    return [{"source": source, "image_id": image_id, "caption_idx": caption_idx,
             "emotion": e, "text": caps[e].strip(), "fold": fold_of(image_id)}
            for e in EMOTIONS if caps.get(e) and caps[e].strip()]


def _unpaired_assignment(images: Sequence[str], per_register: int) -> dict[str, str]:
    """Choose which images carry which register in the 1/image arms.

    Balanced *within each fold* rather than globally. At 878 cells per register spread
    over five folds a purely random assignment leaves a fold short by a dozen cells in
    one register, which at this n is a real imbalance rather than a rounding detail.
    """
    need = per_register * len(EMOTIONS)
    ranked = sorted(images, key=lambda i: _h("unpaired-v1", i))[:need]
    by_fold: dict[int, list[str]] = {}
    for img in ranked:
        by_fold.setdefault(fold_of(img), []).append(img)
    out: dict[str, str] = {}
    turn = 0
    for fold in sorted(by_fold):
        for img in by_fold[fold]:
            out[img] = EMOTIONS[turn % len(EMOTIONS)]
            turn += 1
    return out


def build_all_arms(
    *,
    corpus_path: Path,
    v1_path: Path,
    human_cells: Iterable[Mapping],
    per_register: int = 878,
) -> dict[str, list[dict]]:
    """Return every arm as a list of cells. Pure function of its inputs."""
    S = _load_jsonl(corpus_path)
    V = _load_jsonl(v1_path)

    # The shared image universe. Every Flickr8k arm is drawn from this one set, so a
    # missing image is missing from all of them and no comparison silently becomes
    # unmatched.
    s_imgs = {i for i, _ in S}
    v_imgs = {i for i, _ in V}
    images = sorted(s_imgs & v_imgs)
    full = [i for i in images
            if all((i, k) in S and (i, k) in V for k in range(5))]

    arms: dict[str, list[dict]] = {a: [] for a in ARMS}
    picks = {i: pick_caption_idx(i) for i in full}
    for img in full:
        for k in range(5):
            arms["S_paired25"] += _cells(S[(img, k)], img, k, "S")
        arms["S_paired5"] += _cells(S[(img, picks[img])], img, picks[img], "S")
        arms["V1_paired5"] += _cells(V[(img, picks[img])], img, picks[img], "V1")

    assign = _unpaired_assignment(full, per_register)
    for img, emo in assign.items():
        k = picks[img]
        for src, rec, arm in (("S", S[(img, k)], "S_unpaired"),
                              ("V1", V[(img, k)], "V1_unpaired")):
            cell = next((c for c in _cells(rec, img, k, src) if c["emotion"] == emo),
                        None)
            if cell:
                arms[arm].append(cell)

    # The human arm has its own image universe (YFCC, not Flickr8k) and so its own fold
    # assignment. It is never matched on images to anything -- which is exactly why the
    # comparisons that involve it are REFERENCE rather than confirmatory.
    by_reg: dict[str, list[Mapping]] = {e: [] for e in EMOTIONS}
    for c in human_cells:
        by_reg[c["emotion"]].append(c)
    for emo, cs in by_reg.items():
        for c in sorted(cs, key=lambda c: _h("human-v1", c["image_id"]))[:per_register]:
            arms["H_unpaired"].append(
                {"source": "H", "image_id": c["image_id"], "caption_idx": 0,
                 "emotion": emo, "text": str(c["text"]).strip(),
                 "fold": fold_of(c["image_id"])})
    return arms


def write_arm(cells: Sequence[Mapping], path: Path) -> str:
    """Write one arm as JSONL and return its sha256, for the lock file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(c, sort_keys=True) + "\n" for c in cells)
    path.write_text(body)
    return hashlib.sha256(body.encode()).hexdigest()
