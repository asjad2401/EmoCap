"""Deterministic toy autoregressive models, for testing decoders without a GPU.

The state carries the *full* generated history, so a decoder that reorders beams
incorrectly produces visibly different output. That is the property the v1 pilot's
beam search violated, and what these fixtures exist to catch.
"""

from __future__ import annotations

import torch

PAD, SOS, EOS = 0, 1, 2
A, B = 3, 4  # two ordinary "words"


def _stable_seed(key: tuple[int, ...]) -> int:
    h = 17
    for x in key:
        h = (h * 31 + int(x) + 7) % (2**31 - 1)
    return h


class HistoryModel:
    """Logits are a deterministic function of the whole generated prefix.

    ``table`` maps a generated-token tuple (start token excluded) to explicit
    logits. Unlisted prefixes get stable pseudo-random logits, so the search
    problem is non-trivial but reproducible.

    Records what it was handed and what it returned, so tests can assert the
    decoder never hands a beam storage another beam wrote to.
    """

    def __init__(self, vocab: int = 5, table: dict | None = None, bias: torch.Tensor | None = None):
        self.vocab = vocab
        self.table = table or {}
        self.bias = bias  # (B, vocab) per-example offset, exercises state reordering
        self.received_ptrs: list[int] = []
        self.returned_ptrs: list[int] = []

    def init_state(self, batch_size: int) -> dict:
        state = {"hist": torch.zeros((batch_size, 0), dtype=torch.long)}
        if self.bias is not None:
            state["bias"] = self.bias.clone()
        return state

    def _logits_for(self, key: tuple[int, ...]) -> torch.Tensor:
        if key in self.table:
            return torch.tensor(self.table[key], dtype=torch.float)
        g = torch.Generator().manual_seed(_stable_seed(key))
        return torch.rand(self.vocab, generator=g) * 4.0 - 2.0

    def step(self, tokens: torch.Tensor, state: dict):
        self.received_ptrs.append(state["hist"].data_ptr())
        hist = torch.cat([state["hist"], tokens.unsqueeze(1)], dim=1)
        rows = hist.size(0)
        logits = torch.empty((rows, self.vocab))
        for r in range(rows):
            key = tuple(int(x) for x in hist[r, 1:].tolist())  # drop the start token
            logits[r] = self._logits_for(key)
        if "bias" in state:
            logits = logits + state["bias"]
        new_state = {"hist": hist}
        if "bias" in state:
            new_state["bias"] = state["bias"]
        self.returned_ptrs.append(hist.data_ptr())
        return logits, new_state


class MutatingHistoryModel(HistoryModel):
    """Same model, but it mutates the state it was handed, in place.

    This is what `transformers`' `DynamicCache` does, and what broke v1: the pilot
    handed one cache object to every beam child, so each beam's forward extended
    what its siblings had written. Here the history is a *list* the model appends
    to and returns by reference. Correct output proves the decoder's
    `index_select` reordering neutralises the mutation.
    """

    def init_state(self, batch_size: int) -> dict:
        return {"chunks": [torch.zeros((batch_size, 0), dtype=torch.long)]}

    def step(self, tokens: torch.Tensor, state: dict):
        state["chunks"].append(tokens.unsqueeze(1))  # in-place mutation, by design
        hist = torch.cat(state["chunks"], dim=1)
        rows = hist.size(0)
        logits = torch.empty((rows, self.vocab))
        for r in range(rows):
            key = tuple(int(x) for x in hist[r, 1:].tolist())
            logits[r] = self._logits_for(key)
        return logits, state  # returned by reference, exactly like a Cache


def brute_force_best(
    model: HistoryModel,
    *,
    start_token: int,
    eos_token: int,
    max_new_tokens: int,
    forbidden: tuple[int, ...] = (),
    length_penalty: float = 1.0,
) -> tuple[list[int], float]:
    """Exhaustive search, as an independent oracle for tiny problems."""
    import torch.nn.functional as F

    best: tuple[list[int], float] = ([], float("-inf"))

    def walk(gen: list[int], logprob: float, last: int) -> None:
        nonlocal best
        hist_key = tuple(gen)
        logits = model._logits_for(hist_key).clone()
        for f in forbidden:
            logits[f] = -1e9
        logp = F.log_softmax(logits, dim=-1)
        for tok in range(model.vocab):
            if tok in forbidden:
                continue
            score = logprob + float(logp[tok])
            seq = gen + [tok]
            if tok == eos_token:
                norm = score / (len(seq) ** length_penalty)
                if norm > best[1]:
                    best = (seq[:-1], norm)
            elif len(seq) < max_new_tokens:
                walk(seq, score, tok)
            else:
                norm = score / (len(seq) ** length_penalty)
                if norm > best[1]:
                    best = (seq, norm)

    walk([], 0.0, start_token)
    return best
