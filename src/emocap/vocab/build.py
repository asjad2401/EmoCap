"""Track A's vocabulary: a byte-level BPE trained on our own caption corpus.

Why not reuse CLIP's tokenizer, as v1 did
-----------------------------------------
v1 tokenized with CLIP's 49,408-token BPE, kept the tokens appearing >= 3 times,
and maintained a `local2clip` / `clip2local` mapping between the two id spaces.
That machinery bought nothing. Track A's decoder is trained from scratch, learns
its own embeddings, and never feeds text to CLIP -- only CLIP's *vision* tower is
used. There was no reason for its output layer to speak CLIP's text vocabulary.

The mapping did, however, cost something: `VOCAB_SIZE = len(local2clip)` while
`local2clip` had no entry for `UNK`, so ids ran 0,1,2,4..9199 and the maximum id
*equalled* the vocab size. The training notebook then papered over it by remapping
id 9199 to UNK.

Two consequences of training our own instead:

* **UNK cannot exist.** A byte-level BPE has every byte in its alphabet, so any
  string decomposes. v1's frequency pruning sent 2-16% of dev tokens to UNK
  depending on vocab size; here it is 0% at every size.
* **Merges are chosen for this corpus**, so a small vocab costs far less sequence
  length. Measured on Flickr8k dev: at V=4000 a corpus BPE needs 14.7 tokens per
  caption against CLIP's 14.2 -- but CLIP sends 2.1% of tokens to UNK to get there,
  and needs 9,000 types (3.46M embedding params) to reach 0.94%.

Vocabulary size is the dominant parameter cost in Track A. At v1's V=9,048 with
D=384 and weight tying, the embedding matrix was 3.48M of the model's 5.02M
parameters -- 69.4%, leaving 1.5M for the LSTM and the visual projection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

__all__ = ["Vocab", "train_vocab", "PAD_ID", "SOS_ID", "EOS_ID", "SPECIALS"]

PAD_ID, SOS_ID, EOS_ID = 0, 1, 2
SPECIALS = ("[PAD]", "[SOS]", "[EOS]")


def _byte_level_bpe(captions: Iterable[str], vocab_size: int, min_frequency: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(
        captions,
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=False,
            special_tokens=list(SPECIALS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return tk


@dataclass
class Vocab:
    """Encodes captions to ``[SOS] ... [EOS]`` id sequences and back.

    ``size`` is authoritative: ids are exactly ``0 .. size - 1``, asserted at
    construction so v1's off-by-one cannot recur.
    """

    tokenizer: object
    max_seq_len: int

    def __post_init__(self) -> None:
        vocab = self.tokenizer.get_vocab()
        self.size = self.tokenizer.get_vocab_size()
        max_id = max(vocab.values())
        if max_id != self.size - 1:
            raise ValueError(
                f"vocab ids are not contiguous: max_id={max_id} but size={self.size}. "
                f"Ids must be exactly 0..size-1 (this is v1's off-by-one bug)."
            )
        for name, expected in zip(SPECIALS, (PAD_ID, SOS_ID, EOS_ID)):
            got = vocab.get(name)
            if got != expected:
                raise ValueError(f"special token {name} must have id {expected}, got {got}")

    # ── encoding ────────────────────────────────────────────────────────────
    def encode(self, text: str, *, max_len: int | None = None, add_special: bool = True) -> list[int]:
        """Encode one caption. Truncates the *body* so EOS is always present."""
        ids = self.tokenizer.encode(text).ids
        if not add_special:
            return ids
        cap = self.max_seq_len if max_len is None else max_len
        return [SOS_ID] + ids[: max(0, cap - 2)] + [EOS_ID]

    def encode_batch(self, texts: Sequence[str], *, max_len: int | None = None) -> list[list[int]]:
        return [self.encode(t, max_len=max_len) for t in texts]

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids to text, stopping at EOS and dropping specials."""
        out: list[int] = []
        for i in ids:
            i = int(i)
            if i in (EOS_ID, PAD_ID):
                break
            if i == SOS_ID:
                continue
            out.append(i)
        return self.tokenizer.decode(out).strip()

    # ── diagnostics that gate the pipeline ──────────────────────────────────
    def token_length(self, text: str) -> int:
        """Length with SOS and EOS, before any truncation."""
        return len(self.tokenizer.encode(text).ids) + 2

    def truncation_rate(self, texts: Iterable[str], *, max_len: int | None = None) -> float:
        texts = list(texts)
        if not texts:
            return 0.0
        cap = self.max_seq_len if max_len is None else max_len
        return sum(1 for t in texts if self.token_length(t) > cap) / len(texts)

    def suggest_max_seq_len(self, texts: Iterable[str], *, quantile: float = 0.995) -> int:
        """Length cap covering ``quantile`` of the corpus.

        v1 set MAX_SEQ_LEN=30 without measuring and truncated roughly half its
        targets mid-sentence, teaching the decoder to stop at arbitrary points.
        """
        lengths = sorted(self.token_length(t) for t in texts)
        if not lengths:
            return self.max_seq_len
        return lengths[min(len(lengths) - 1, int(quantile * len(lengths)))]

    def unk_rate(self, texts: Iterable[str]) -> float:
        """Always 0.0 for a byte-level BPE. Kept as an explicit assertion."""
        total = lossy = 0
        for t in texts:
            ids = self.tokenizer.encode(t).ids
            total += len(ids)
            if self.tokenizer.decode(ids).strip() != t.strip():
                lossy += len(ids)
        return lossy / max(1, total)

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))
        (path / "vocab_meta.json").write_text(
            json.dumps(
                {"size": self.size, "max_seq_len": self.max_seq_len,
                 "pad_id": PAD_ID, "sos_id": SOS_ID, "eos_id": EOS_ID},
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        from tokenizers import Tokenizer

        path = Path(path)
        meta = json.loads((path / "vocab_meta.json").read_text())
        tk = Tokenizer.from_file(str(path / "tokenizer.json"))
        v = cls(tokenizer=tk, max_seq_len=meta["max_seq_len"])
        if v.size != meta["size"]:
            raise ValueError(f"tokenizer size {v.size} != recorded size {meta['size']}")
        return v


def train_vocab(
    captions: Iterable[str],
    *,
    vocab_size: int,
    min_frequency: int = 2,
    max_seq_len: int | None = None,
    length_quantile: float = 0.995,
) -> Vocab:
    """Train a vocabulary on ``captions``.

    Pass **training-split captions only** -- fitting on val or test leaks their
    vocabulary into the model's output space.

    With ``max_seq_len=None`` the cap is derived from the corpus at
    ``length_quantile``, rather than guessed.
    """
    captions = list(captions)
    if not captions:
        raise ValueError("no captions to train on")
    tk = _byte_level_bpe(captions, vocab_size, min_frequency)
    v = Vocab(tokenizer=tk, max_seq_len=max_seq_len or 32)
    if max_seq_len is None:
        v.max_seq_len = v.suggest_max_seq_len(captions, quantile=length_quantile)
    return v
