"""Tests for Track A's vocabulary.

Two v1 defects are pinned here:

* ``VOCAB_SIZE = len(local2clip)`` while ``local2clip`` had no ``UNK`` entry, so ids
  ran 0,1,2,4..9199 and the maximum id *equalled* the vocab size. The training
  notebook detected it at runtime and patched id 9199 to UNK.
* ``MAX_SEQ_LEN = 30`` was guessed rather than measured, truncating roughly half the
  targets mid-sentence.
"""

from __future__ import annotations

import pytest

from emocap.vocab import EOS_ID, PAD_ID, SOS_ID, Vocab, train_vocab

CORPUS = [
    "a child in a pink dress is climbing up a set of stairs in an entry way.",
    "a black dog and a spotted dog are fighting in the street.",
    "two dogs of different breeds looking at each other on the road.",
    "a girl going into a wooden building with a red door.",
    "a man in a black jacket is taking a photo of a woman in a red coat.",
    "three children on skates in an arena, two girls and one boy.",
    "the woman in the white shirt holds a clear mug of brown liquid.",
    "a boy in a camouflage coat is jumping onto a blue sled in the snow.",
] * 40


@pytest.fixture(scope="module")
def vocab():
    return train_vocab(CORPUS, vocab_size=600, max_seq_len=32)


# ── the v1 off-by-one ───────────────────────────────────────────────────────


def test_ids_are_contiguous_from_zero(vocab):
    ids = set(vocab.tokenizer.get_vocab().values())
    assert min(ids) == 0
    assert max(ids) == vocab.size - 1
    assert len(ids) == vocab.size


def test_specials_occupy_the_first_three_ids(vocab):
    v = vocab.tokenizer.get_vocab()
    assert (v["[PAD]"], v["[SOS]"], v["[EOS]"]) == (PAD_ID, SOS_ID, EOS_ID) == (0, 1, 2)


def test_a_vocab_with_an_id_gap_is_rejected():
    """Exactly v1's failure: size reported as N while an id equals N."""

    class Gappy:
        def get_vocab(self):
            return {"[PAD]": 0, "[SOS]": 1, "[EOS]": 2, "a": 4}  # 3 missing

        def get_vocab_size(self):
            return 4

    with pytest.raises(ValueError, match="off-by-one"):
        Vocab(tokenizer=Gappy(), max_seq_len=32)


def test_misplaced_special_token_is_rejected():
    class Shuffled:
        def get_vocab(self):
            return {"[SOS]": 0, "[PAD]": 1, "[EOS]": 2}

        def get_vocab_size(self):
            return 3

    with pytest.raises(ValueError, match=r"\[PAD\] must have id 0"):
        Vocab(tokenizer=Shuffled(), max_seq_len=32)


# ── no UNK is possible ─────────────────────────────────────────────────────


def test_unseen_words_roundtrip_losslessly(vocab):
    """A byte-level BPE has every byte in its alphabet, so nothing falls to UNK.

    v1's frequency-pruned CLIP vocab sent 2.1% of dev tokens to UNK at V=4000 and
    still 0.94% at V=9000.
    """
    for text in [
        "a zebra gallops onomatopoeically",
        "quixotic bioluminescence",
        "Kaggle notebook 12345",
    ]:
        ids = vocab.encode(text, max_len=200)
        assert vocab.decode(ids) == text, f"lossy roundtrip: {text!r}"


def test_unk_rate_is_zero(vocab):
    assert vocab.unk_rate(["utterly unseen tokens 98765 zzzz"]) == 0.0


def test_unicode_and_emoji_survive(vocab):
    for text in ["a café in montréal", "a dog 🐕 runs", "naïve façade"]:
        assert vocab.decode(vocab.encode(text, max_len=200)) == text


def test_a_tiny_vocab_still_covers_everything():
    """Coverage must not depend on vocab size -- only sequence length should."""
    small = train_vocab(CORPUS, vocab_size=300, max_seq_len=400)
    text = "an unprecedented juxtaposition"
    assert small.decode(small.encode(text, max_len=400)) == text
    assert small.unk_rate([text]) == 0.0


# ── encode / decode contract ───────────────────────────────────────────────


def test_encode_wraps_with_sos_and_eos(vocab):
    ids = vocab.encode("a black dog and a spotted dog are fighting in the street.")
    assert ids[0] == SOS_ID and ids[-1] == EOS_ID


def test_truncation_preserves_eos(vocab):
    """A truncated sequence must still end in EOS, or the model never learns to stop."""
    ids = vocab.encode(" ".join(["word"] * 200), max_len=12)
    assert len(ids) == 12
    assert ids[0] == SOS_ID and ids[-1] == EOS_ID


def test_decode_stops_at_eos_and_ignores_padding(vocab):
    ids = vocab.encode("two dogs on the road.")
    padded = ids + [PAD_ID] * 8
    assert vocab.decode(padded) == vocab.decode(ids)


def test_decode_ignores_trailing_garbage_after_eos(vocab):
    ids = vocab.encode("two dogs on the road.")
    assert vocab.decode(ids + [50, 60, 70]) == vocab.decode(ids)


def test_encode_batch_matches_encode(vocab):
    texts = CORPUS[:4]
    assert vocab.encode_batch(texts) == [vocab.encode(t) for t in texts]


# ── length measurement, not guessing ───────────────────────────────────────


def test_suggested_length_covers_the_requested_quantile(vocab):
    cap = vocab.suggest_max_seq_len(CORPUS, quantile=0.99)
    over = sum(1 for t in CORPUS if vocab.token_length(t) > cap)
    assert over / len(CORPUS) <= 0.01


def test_truncation_rate_is_measured(vocab):
    assert vocab.truncation_rate(CORPUS, max_len=1000) == 0.0
    assert vocab.truncation_rate(CORPUS, max_len=4) == 1.0


def test_derived_length_meets_the_pipeline_gate():
    """configs/prereg.lock.yaml gates truncation_rate_max at 0.02."""
    v = train_vocab(CORPUS, vocab_size=600)  # max_seq_len derived from the corpus
    assert v.truncation_rate(CORPUS) <= 0.02


# ── persistence ────────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path, vocab):
    vocab.save(tmp_path / "vocab")
    loaded = Vocab.load(tmp_path / "vocab")
    assert loaded.size == vocab.size
    assert loaded.max_seq_len == vocab.max_seq_len
    text = "a boy jumping onto a blue sled in the snow."
    assert loaded.encode(text) == vocab.encode(text)
    assert loaded.decode(loaded.encode(text)) == vocab.decode(vocab.encode(text))


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="no captions"):
        train_vocab([], vocab_size=100)
