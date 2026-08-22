"""Turn an arm file plus the feature store into batched training tensors.

One arm cell is ``(image_id, emotion, text)``. Training needs ``(clip_feature,
emotion_id, token_ids)``, and the only interesting part is what must NOT happen:

* **Folds split by image, never by caption.** ``fold`` is precomputed in the arm file, so
  the split cannot be re-derived differently here and drift from the one in the manifest.
* **The negative control shuffles emotion labels WITHIN the arm**, destroying the
  image-emotion pairing while leaving the marginal label distribution untouched. Shuffling
  the texts instead, or resampling labels, would change the class balance and make the
  control's accuracy incomparable to the arm's.
* **Captions get an EOS.** Without it the model never learns to stop and every caption runs
  to ``max_new_tokens``, which looks like a decode bug and is a data bug.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from emocap.data.prompt import EMOTIONS

__all__ = ["ArmCells", "load_arm", "make_split", "collate"]


class ArmCells:
    """The cells of one arm, with their folds."""

    def __init__(self, cells: list[dict]):
        self.cells = cells

    def __len__(self) -> int:
        return len(self.cells)

    def fold(self, n: int, *, train: bool) -> "ArmCells":
        return ArmCells([c for c in self.cells if (c["fold"] != n) == train])

    def shuffled_labels(self, seed: int) -> "ArmCells":
        """The registered negative control: same texts, same images, permuted emotions."""
        emos = [c["emotion"] for c in self.cells]
        random.Random(seed).shuffle(emos)
        return ArmCells([{**c, "emotion": e} for c, e in zip(self.cells, emos)])


def load_arm(path: Path) -> ArmCells:
    return ArmCells([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])


def make_split(arm: ArmCells, fold: int) -> tuple[ArmCells, ArmCells]:
    return arm.fold(fold, train=True), arm.fold(fold, train=False)


def collate(cells: Sequence[dict], store, tokenizer, *, max_len: int):
    """``(features, emotion_ids, input_ids, attention_mask)`` for one batch."""
    import torch

    feats = torch.from_numpy(store.rows([c["image_id"] for c in cells])).float()
    emos = torch.tensor([EMOTIONS.index(c["emotion"]) for c in cells])
    texts = [c["text"].strip() + tokenizer.eos_token for c in cells]
    enc = tokenizer(texts, truncation=True, max_length=max_len,
                    padding="max_length", return_tensors="pt")
    return feats, emos, enc["input_ids"], enc["attention_mask"]
