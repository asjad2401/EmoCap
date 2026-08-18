"""Regression tests for the decoder.

The v1 pilot shipped a beam search that handed every beam child the same mutable
`past_key_values` object, so each beam decoded against a mixture of its siblings'
tokens. Training loss looked healthy while every caption was word salad, and the
defect survived for months because nothing here existed.

Four tests here fail against a structural reproduction of that implementation:

  * ``test_reordering_hands_the_model_fresh_storage`` -- the direct catch
  * ``test_mutating_state_still_correct``
  * ``test_no_repeat_ngram_blocks_repeats`` -- v1's n-gram blocking was also wrong
  * ``test_opaque_state_leaf_is_rejected``

``test_batch_independence`` does *not* catch it, because v1 decoded one example at a
time; its contamination was between beams, not between examples. It guards the
batching this implementation adds, which is what makes evaluation affordable.
"""

from __future__ import annotations

import pytest
import torch

from emocap.decode import DecodeConfig, beam_search, greedy_search
from fake_decoders import (
    A,
    B,
    EOS,
    PAD,
    SOS,
    HistoryModel,
    MutatingHistoryModel,
    brute_force_best,
)

FORBIDDEN = (PAD, SOS)


def _decode(model, batch_size, **kw):
    cfg = DecodeConfig(**{
        "beam_size": 4, "max_new_tokens": 6, "min_new_tokens": 0,
        "length_penalty": 1.0, "no_repeat_ngram_size": 0, **kw,
    })
    return beam_search(
        model.step,
        model.init_state(batch_size),
        batch_size=batch_size,
        start_token=SOS,
        eos_token=EOS,
        pad_token=PAD,
        config=cfg,
        forbidden_tokens=FORBIDDEN,
    )


# ── the two tests that catch the v1 bug ─────────────────────────────────────


def test_batch_independence():
    """Examples decoded together must match those examples decoded alone.

    v1 decoded one row at a time, which is why BLEU eval over 3,722 test rows was
    abandoned on a KeyboardInterrupt. Batching is what makes evaluation
    affordable, and this is the invariant that makes batching safe. Per-example
    biases make each row's search genuinely different, so any leakage shows up.
    """
    n = 6
    torch.manual_seed(0)
    bias = torch.randn(n, 5) * 1.5

    batched = _decode(HistoryModel(bias=bias), n, beam_size=3)

    alone = []
    for i in range(n):
        alone.extend(_decode(HistoryModel(bias=bias[i : i + 1]), 1, beam_size=3))

    assert batched == alone, f"batched {batched} != individually {alone}"


def test_mutating_state_still_correct():
    """A model that mutates the state it is handed must still decode correctly.

    `DynamicCache` mutates in place. The decoder's defence is that reordering via
    `index_select` always allocates, so no beam can observe a sibling's write.
    """
    plain = _decode(HistoryModel(), 3, beam_size=3)
    mutating = _decode(MutatingHistoryModel(), 3, beam_size=3)
    assert plain == mutating


def test_reordering_hands_the_model_fresh_storage():
    """Each step must receive storage the previous step did not return."""
    model = HistoryModel()
    _decode(model, 2, beam_size=3, max_new_tokens=5)
    assert len(model.received_ptrs) >= 3
    for t in range(len(model.returned_ptrs) - 1):
        assert model.received_ptrs[t + 1] != model.returned_ptrs[t], (
            f"step {t + 1} was handed the same storage step {t} returned -- "
            "beam reordering did not copy"
        )


def test_opaque_state_leaf_is_rejected():
    """An unwrapped cache-like object must fail loudly, not silently pass through."""

    class OpaqueCache:
        pass

    def step(tokens, state):
        return torch.zeros((tokens.size(0), 5)), state

    with pytest.raises(TypeError, match="v1 cache-aliasing bug class"):
        beam_search(
            step,
            {"cache": OpaqueCache()},
            batch_size=1,
            start_token=SOS,
            eos_token=EOS,
            pad_token=PAD,
            config=DecodeConfig(beam_size=2, max_new_tokens=3),
        )


# ── search correctness ──────────────────────────────────────────────────────


def test_beam_size_one_matches_greedy():
    model_b, model_g = HistoryModel(), HistoryModel()
    beam = _decode(model_b, 3, beam_size=1, max_new_tokens=6)
    greedy = greedy_search(
        model_g.step,
        model_g.init_state(3),
        batch_size=3,
        start_token=SOS,
        eos_token=EOS,
        pad_token=PAD,
        max_new_tokens=6,
        forbidden_tokens=FORBIDDEN,
    )
    assert beam == greedy


def test_beam_beats_greedy_on_a_trap():
    """A locally-best first token that leads into a high-entropy dead end.

    Greedy takes A because its logit is higher; the good path is B, whose
    continuation is nearly deterministic and so costs almost nothing.
    """
    table = {
        (): [-9.0, -9.0, -9.0, 0.0, -0.1],   # A marginally beats B
        (A,): [0.0, 0.0, 0.0, 0.0, 0.0],     # 5-way tie: every continuation is costly
        (B,): [-9.0, -9.0, 0.0, -9.0, -9.0],  # EOS almost free
    }
    model_g, model_b = HistoryModel(table=table), HistoryModel(table=table)

    greedy = greedy_search(
        model_g.step, model_g.init_state(1), batch_size=1,
        start_token=SOS, eos_token=EOS, pad_token=PAD,
        max_new_tokens=3, forbidden_tokens=FORBIDDEN,
    )
    beam = _decode(model_b, 1, beam_size=2, max_new_tokens=3)

    assert greedy == [[A]], f"expected greedy to fall into the trap, got {greedy}"
    assert beam == [[B]], f"expected beam to escape it, got {beam}"


def test_beam_matches_exhaustive_oracle():
    """On a problem small enough to enumerate, beam search finds the optimum."""
    table = {
        (): [-9.0, -9.0, -9.0, 0.0, -0.1],
        (A,): [0.0, 0.0, 0.0, 0.0, 0.0],
        (B,): [-9.0, -9.0, 0.0, -9.0, -9.0],
    }
    model = HistoryModel(table=table)
    oracle_seq, _ = brute_force_best(
        model, start_token=SOS, eos_token=EOS, max_new_tokens=3, forbidden=FORBIDDEN
    )
    got = _decode(HistoryModel(table=table), 1, beam_size=3, max_new_tokens=3)
    assert got == [oracle_seq]


def test_wider_beam_never_scores_worse():
    """Best normalised score must be non-decreasing in beam width."""
    prev = float("-inf")
    for k in (1, 2, 3, 4):
        model = HistoryModel()
        _, scores = beam_search(
            model.step, model.init_state(1), batch_size=1,
            start_token=SOS, eos_token=EOS, pad_token=PAD,
            config=DecodeConfig(beam_size=k, max_new_tokens=6, min_new_tokens=0,
                                no_repeat_ngram_size=0),
            forbidden_tokens=FORBIDDEN, return_scores=True,
        )
        assert scores[0] >= prev - 1e-6, f"beam {k} scored {scores[0]} < beam {k-1} {prev}"
        prev = scores[0]


def test_decode_is_deterministic():
    a = _decode(HistoryModel(), 4, beam_size=4)
    b = _decode(HistoryModel(), 4, beam_size=4)
    assert a == b


# ── constraints ─────────────────────────────────────────────────────────────


def test_eos_and_pad_never_appear_in_output():
    out = _decode(HistoryModel(), 5, beam_size=4, max_new_tokens=8)
    for seq in out:
        assert EOS not in seq and PAD not in seq and SOS not in seq


def test_min_new_tokens_is_respected():
    table = {(): [-9.0, -9.0, 5.0, -9.0, -9.0]}  # model desperately wants to stop
    out = _decode(HistoryModel(table=table), 2, beam_size=3,
                  min_new_tokens=4, max_new_tokens=8)
    for seq in out:
        assert len(seq) >= 4, f"min_new_tokens=4 violated: {seq}"


def test_max_new_tokens_is_respected():
    out = _decode(HistoryModel(), 3, beam_size=3, min_new_tokens=6, max_new_tokens=6)
    for seq in out:
        assert len(seq) <= 6


def test_no_repeat_ngram_blocks_repeats():
    """A model that only ever wants A must be forced off it."""
    table = {}
    for prefix in [(), (A,), (A, A), (A, A, A), (A, B), (A, A, B)]:
        table[prefix] = [-9.0, -9.0, -9.0, 5.0, 0.0]  # A strongly preferred, B available
    out = _decode(HistoryModel(table=table), 1, beam_size=2,
                  min_new_tokens=4, max_new_tokens=5, no_repeat_ngram_size=2)
    seq = out[0]
    bigrams = list(zip(seq, seq[1:]))
    assert len(bigrams) == len(set(bigrams)), f"repeated bigram in {seq}"


def test_beam_size_larger_than_vocab_is_rejected():
    model = HistoryModel(vocab=5)
    with pytest.raises(ValueError, match="exceeds vocab size"):
        _decode(model, 1, beam_size=6)


def test_bad_logits_shape_is_rejected():
    def step(tokens, state):
        return torch.zeros((tokens.size(0), 3, 5)), state

    with pytest.raises(ValueError, match=r"\(rows, vocab\)"):
        beam_search(
            step, {"x": torch.zeros(1, 1)}, batch_size=1,
            start_token=SOS, eos_token=EOS, pad_token=PAD,
            config=DecodeConfig(beam_size=2, max_new_tokens=3),
        )
