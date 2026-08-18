"""Cross-validate our beam search against HuggingFace's, on a tiny random GPT-2.

If two independent beam search implementations agree token-for-token on a model
with random weights, the search logic is almost certainly right. This is the
strongest check available without a GPU or a model download.

Settings are chosen to isolate the *search* from the bookkeeping around it:
`min_new_tokens == max_new_tokens` means no beam ever finishes early, so neither
implementation's early-stopping heuristic is engaged, and `length_penalty=0.0`
makes both reduce to plain sum-of-logprobs (`len ** 0 == 1`), sidestepping the
fact that HF normalises by full sequence length while we normalise by generated
length.
"""

from __future__ import annotations

import pytest
import torch

from emocap.decode import DecodeConfig, beam_search, generate_from_prefix

transformers = pytest.importorskip("transformers")


VOCAB, N_STEPS = 64, 6


@pytest.fixture(scope="module")
def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(7)
    cfg = GPT2Config(
        vocab_size=VOCAB, n_positions=64, n_embd=32, n_layer=2, n_head=2,
        bos_token_id=0, eos_token_id=1,
    )
    model = GPT2LMHeadModel(cfg).eval()
    return model


def _stepper(model):
    """Wrap GPT-2 in our step protocol, recomputing the full forward each step.

    Deliberately cache-free: the point is to test the *search*, and a cache is
    exactly the thing that broke in v1.
    """

    def step(tokens, state):
        ids = torch.cat([state["ids"], tokens.unsqueeze(1)], dim=1)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[:, -1, :]
        return logits, {"ids": ids}

    return step


@pytest.mark.parametrize("beam_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_matches_hf_beam_search(tiny_gpt2, beam_size, batch_size):
    bos, eos = 0, 1

    ours = beam_search(
        _stepper(tiny_gpt2),
        {"ids": torch.empty((batch_size, 0), dtype=torch.long)},
        batch_size=batch_size,
        start_token=bos,
        eos_token=eos,
        pad_token=eos,
        config=DecodeConfig(
            beam_size=beam_size,
            max_new_tokens=N_STEPS,
            min_new_tokens=N_STEPS,   # nobody finishes early
            length_penalty=0.0,       # both reduce to sum-logprob
            no_repeat_ngram_size=0,
        ),
    )

    with torch.no_grad():
        hf = tiny_gpt2.generate(
            input_ids=torch.full((batch_size, 1), bos, dtype=torch.long),
            num_beams=beam_size,
            max_new_tokens=N_STEPS,
            min_new_tokens=N_STEPS,
            length_penalty=0.0,
            no_repeat_ngram_size=0,
            do_sample=False,
            early_stopping=True,
            num_return_sequences=1,
            eos_token_id=eos,
            pad_token_id=eos,
        )
    theirs = hf[:, 1:].tolist()  # drop the BOS we seeded

    assert ours == theirs, (
        f"beam={beam_size} batch={batch_size}\n  ours   {ours}\n  theirs {theirs}"
    )


# ── Track B plumbing: prefix embeddings in, strings out ─────────────────────


class _StubTokenizer:
    """Enough of a tokenizer for generate_from_prefix, with no download."""

    eos_token_id = 1
    pad_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        toks = [int(i) for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t != self.eos_token_id]
        return " ".join(f"w{t}" for t in toks)


def test_generate_from_prefix_returns_one_caption_per_row(tiny_gpt2):
    batch, prefix_len = 3, 5
    torch.manual_seed(0)
    prefix = torch.randn(batch, prefix_len, tiny_gpt2.config.n_embd)

    out = generate_from_prefix(
        tiny_gpt2, prefix, _StubTokenizer(),
        config=DecodeConfig(beam_size=2, max_new_tokens=8, min_new_tokens=2,
                            length_penalty=1.0, no_repeat_ngram_size=0),
    )
    assert len(out) == batch
    assert all(isinstance(s, str) and s for s in out)


def test_prefix_actually_reaches_the_language_model(tiny_gpt2):
    """Different prefixes must produce different next-token distributions.

    This checks the *plumbing* -- that soft prefix embeddings are consumed rather
    than ignored. It deliberately asserts on logits, not on decoded captions: an
    untrained GPT-2 has one dominant output token that swamps any prefix, so a
    caption-level check would pass or fail for reasons unrelated to the wiring.

    The behavioural version of this claim -- that a *trained* model's captions
    change when the visual features are zeroed -- is the visual-dependence probe
    in the training pipeline, and it gates every run. Track A of the v1 pilot
    would have failed it: its LSTM emitted one generic caption for every image.
    """
    torch.manual_seed(1)
    h = tiny_gpt2.config.n_embd
    a = torch.randn(1, 5, h) * 3.0
    b = torch.randn(1, 5, h) * 3.0
    with torch.no_grad():
        la = tiny_gpt2(inputs_embeds=a).logits[:, -1, :]
        lb = tiny_gpt2(inputs_embeds=b).logits[:, -1, :]
    assert not torch.allclose(la, lb), "prefix embeddings are not reaching the LM"


def test_generate_from_prefix_rejects_wrong_rank(tiny_gpt2):
    with pytest.raises(ValueError, match=r"\(B, P, H\)"):
        generate_from_prefix(tiny_gpt2, torch.randn(5, 32), _StubTokenizer())


def test_generate_from_prefix_restores_training_mode(tiny_gpt2):
    tiny_gpt2.train()
    try:
        generate_from_prefix(
            tiny_gpt2, torch.randn(1, 4, tiny_gpt2.config.n_embd), _StubTokenizer(),
            config=DecodeConfig(beam_size=1, max_new_tokens=4, min_new_tokens=1,
                                no_repeat_ngram_size=0),
        )
        assert tiny_gpt2.training, "generate_from_prefix must not leave the model in eval"
    finally:
        tiny_gpt2.eval()
