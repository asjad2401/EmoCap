"""Batched beam search, and a thin correct wrapper over HF generation.

Why this module exists as its own file with its own tests
--------------------------------------------------------
The v1 pilot's beam search pushed every beam child holding the *same*
``past_key_values`` object::

    for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
        new_beams.append((score + lp, toks + [idx], out.past_key_values))

Under the legacy tuple-of-tensors cache API that was correct. Modern
``transformers`` returns a ``DynamicCache`` and mutates it in place, so beam 1's
forward extended the cache beam 0 had just written to, and every beam ended up
decoding against a mixture of its siblings' tokens. Training loss looked healthy
(teacher forcing never runs the decoder) while every sampled caption was word
salad.

Two structural defences here, both covered by tests:

1. Beam state is reordered with ``index_select`` every step. That always
   allocates fresh storage, so no two beams can share a tensor even if the model
   mutates what it is handed.
2. Track B does not get a hand-rolled cache at all -- it calls
   ``model.generate(inputs_embeds=...)``, which is batched and maintained
   upstream.

Both tracks read their settings from the same :class:`DecodeConfig` so the
ablation cannot be confounded by per-variant decode tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

__all__ = ["DecodeConfig", "beam_search", "greedy_search", "generate_from_prefix"]

NEG_INF = -1e9


@dataclass(frozen=True)
class DecodeConfig:
    """Pre-registered decode settings. Identical for every variant and track.

    Locked in ``configs/prereg.lock.yaml``; the results notebook refuses to run
    if a live config has drifted from it.
    """

    beam_size: int = 4
    max_new_tokens: int = 40
    min_new_tokens: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── state plumbing ──────────────────────────────────────────────────────────
# State is any nesting of dicts / lists / tuples whose tensor leaves are
# BATCH-FIRST. Non-tensor leaves are passed through untouched and so must be
# beam-invariant (vocab sizes, flags -- not per-beam data).


#: Leaf types allowed through a state tree untouched. Anything else is rejected:
#: an opaque mutable container (a ``transformers`` ``Cache``, say) would silently
#: escape beam reordering and reintroduce exactly the v1 aliasing bug.
_PASSTHROUGH = (type(None), bool, int, float, str, bytes)


def _tree_map_tensors(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _tree_map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tree_map_tensors(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_tree_map_tensors(v, fn) for v in obj)
    if isinstance(obj, _PASSTHROUGH):
        return obj
    raise TypeError(
        f"decode state may only contain tensors and dict/list/tuple nestings of them, "
        f"plus scalars; got {type(obj).__name__}. An opaque container would escape beam "
        f"reordering -- unwrap it into tensors (this is the v1 cache-aliasing bug class)."
    )


def _first_tensor_device(obj: Any) -> torch.device | None:
    """Device of the first tensor leaf, so callers need not pass one."""
    found: list[torch.device] = []

    def probe(t: torch.Tensor) -> torch.Tensor:
        if not found:
            found.append(t.device)
        return t

    _tree_map_tensors(obj, probe)
    return found[0] if found else None


def _expand_state(state: Any, beam_size: int) -> Any:
    """(B, ...) -> (B * K, ...) with each example's copies adjacent."""
    return _tree_map_tensors(state, lambda t: t.repeat_interleave(beam_size, dim=0))


def _reorder_state(state: Any, index: torch.Tensor) -> Any:
    """Select rows of every tensor leaf. Always allocates -- never aliases."""
    return _tree_map_tensors(state, lambda t: t.index_select(0, index))


# ── constraint masks ────────────────────────────────────────────────────────


def _ban_tokens(logp: torch.Tensor, tokens: Iterable[int]) -> None:
    for t in tokens:
        if t is not None:
            logp[:, int(t)] = NEG_INF


def _ban_repeat_ngrams(logp: torch.Tensor, seqs: torch.Tensor, n: int) -> None:
    """Ban any token that already followed the current (n-1)-gram in that row.

    ``seqs`` is (rows, T) of generated tokens only -- the start token must not be
    included, or the first n-gram straddles a boundary that never recurs.
    """
    if n <= 0:
        return
    rows, length = seqs.shape
    if length < n - 1:
        return
    seq_list = seqs.tolist()
    for r in range(rows):
        row = seq_list[r]
        suffix = tuple(row[length - (n - 1) :])
        for i in range(length - (n - 1)):
            if tuple(row[i : i + n - 1]) == suffix:
                logp[r, row[i + n - 1]] = NEG_INF


# ── beam search ─────────────────────────────────────────────────────────────


def beam_search(
    step_fn: Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]],
    init_state: Any,
    *,
    batch_size: int,
    start_token: int,
    eos_token: int,
    pad_token: int,
    config: DecodeConfig | None = None,
    forbidden_tokens: Sequence[int] = (),
    device: torch.device | str | None = None,
    return_scores: bool = False,
):
    """Batched beam search over an arbitrary autoregressive step function.

    ``step_fn(tokens, state) -> (logits, new_state)`` where ``tokens`` is
    ``(rows,)`` of the last emitted token and ``logits`` is ``(rows, vocab)``.
    Every tensor leaf of ``state`` must be batch-first with leading dim ``rows``.

    Returns one token list per example -- generated tokens only, with the start
    token and any trailing EOS/PAD stripped. Ties in score resolve to the
    lower beam index, so results are deterministic.

    Length normalisation is ``sum_logprob / len ** length_penalty``, matching
    HuggingFace's ``BeamSearchScorer`` so that Track A and Track B are scored on
    the same scale.
    """
    cfg = config or DecodeConfig()
    K, B = cfg.beam_size, batch_size
    if K < 1:
        raise ValueError(f"beam_size must be >= 1, got {K}")
    # Bookkeeping tensors must live wherever the model's tensors do, or
    # index_select fails on a device mismatch the moment anyone uses a GPU.
    dev = torch.device(device) if device is not None else _first_tensor_device(init_state)

    state = _expand_state(init_state, K)
    rows = B * K

    tokens = torch.full((rows, 1), start_token, dtype=torch.long, device=dev)
    generated = torch.empty((rows, 0), dtype=torch.long, device=dev)

    # Only beam 0 of each example starts live, so step 0 takes the top-K of one
    # distribution instead of K identical copies of the top-1.
    scores = torch.full((B, K), NEG_INF, dtype=torch.float, device=dev)
    scores[:, 0] = 0.0
    scores = scores.view(rows)

    done = torch.zeros(rows, dtype=torch.bool, device=dev)
    # Length used for normalisation: frozen at the step a beam emits EOS.
    lengths = torch.zeros(rows, dtype=torch.long, device=dev)

    banned = tuple(forbidden_tokens)

    for t in range(cfg.max_new_tokens):
        logits, state = step_fn(tokens[:, -1], state)
        if logits.dim() != 2 or logits.size(0) != rows:
            raise ValueError(
                f"step_fn must return (rows, vocab) logits; got {tuple(logits.shape)} "
                f"for rows={rows}"
            )
        vocab = logits.size(-1)
        if vocab < K:
            raise ValueError(f"beam_size {K} exceeds vocab size {vocab}")
        logp = F.log_softmax(logits.float(), dim=-1)
        if t == 0 and logp.device != tokens.device:
            # Caller gave no device and the state held no tensors to infer from.
            tokens = tokens.to(logp.device)
            generated = generated.to(logp.device)
            scores = scores.to(logp.device)
            done = done.to(logp.device)
            lengths = lengths.to(logp.device)

        _ban_tokens(logp, banned)
        if t < cfg.min_new_tokens:
            logp[:, eos_token] = NEG_INF
        if cfg.no_repeat_ngram_size > 0 and generated.size(1) >= cfg.no_repeat_ngram_size - 1:
            _ban_repeat_ngrams(logp, generated, cfg.no_repeat_ngram_size)

        # A finished beam keeps its exact score and stays in the running by
        # continuing with PAD at zero cost.
        if done.any():
            frozen = torch.full_like(logp[0], NEG_INF)
            frozen[pad_token] = 0.0
            logp[done] = frozen

        cand = (scores.unsqueeze(1) + logp).view(B, K * vocab)
        top_scores, top_flat = cand.topk(K, dim=-1)          # (B, K)

        beam_src = top_flat // vocab                          # (B, K) in [0, K)
        next_tok = top_flat % vocab                           # (B, K)

        offsets = torch.arange(B, device=logp.device).unsqueeze(1) * K
        gather_idx = (beam_src + offsets).view(rows)

        state = _reorder_state(state, gather_idx)
        generated = generated.index_select(0, gather_idx)
        tokens = tokens.index_select(0, gather_idx)
        done = done.index_select(0, gather_idx)
        lengths = lengths.index_select(0, gather_idx)

        flat_tok = next_tok.reshape(rows)
        scores = top_scores.reshape(rows)
        generated = torch.cat([generated, flat_tok.unsqueeze(1)], dim=1)
        tokens = torch.cat([tokens, flat_tok.unsqueeze(1)], dim=1)

        newly_done = (~done) & (flat_tok == eos_token)
        lengths = torch.where(newly_done, torch.full_like(lengths, t + 1), lengths)
        done = done | newly_done

        if bool(done.all()):
            break

    # Beams that never emitted EOS are scored at full generated length.
    lengths = torch.where(lengths == 0, torch.full_like(lengths, generated.size(1)), lengths)
    norm = scores / lengths.clamp(min=1).to(scores.dtype) ** cfg.length_penalty

    best_seqs: list[list[int]] = []
    best_scores: list[float] = []
    norm_b = norm.view(B, K)
    gen_b = generated.view(B, K, -1)
    len_b = lengths.view(B, K)
    for b in range(B):
        # argmax returns the first maximal index, so equal-scoring beams resolve
        # to the lowest beam index and repeated runs agree.
        k = int(torch.argmax(norm_b[b]).item())
        seq = gen_b[b, k, : int(len_b[b, k].item())].tolist()
        if seq and seq[-1] == eos_token:
            seq = seq[:-1]
        best_seqs.append([int(x) for x in seq if x != pad_token])
        best_scores.append(float(norm_b[b, k].item()))

    return (best_seqs, best_scores) if return_scores else best_seqs


def greedy_search(
    step_fn: Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]],
    init_state: Any,
    *,
    batch_size: int,
    start_token: int,
    eos_token: int,
    pad_token: int,
    max_new_tokens: int = 40,
    min_new_tokens: int = 0,
    forbidden_tokens: Sequence[int] = (),
    device: torch.device | str | None = None,
) -> list[list[int]]:
    """Plain argmax decode. Exists as an independent reference for the tests."""
    dev = torch.device(device) if device is not None else _first_tensor_device(init_state)
    state = init_state
    tok = torch.full((batch_size,), start_token, dtype=torch.long, device=dev)
    out: list[list[int]] = [[] for _ in range(batch_size)]
    live = [True] * batch_size

    for t in range(max_new_tokens):
        logits, state = step_fn(tok, state)
        logp = F.log_softmax(logits.float(), dim=-1)
        _ban_tokens(logp, forbidden_tokens)
        if t < min_new_tokens:
            logp[:, eos_token] = NEG_INF
        tok = logp.argmax(dim=-1)
        for b in range(batch_size):
            if not live[b]:
                continue
            v = int(tok[b].item())
            if v == eos_token:
                live[b] = False
            else:
                out[b].append(v)
        if not any(live):
            break
    return out


# ── Track B: frozen GPT-2 behind a soft prefix ──────────────────────────────


@torch.no_grad()
def generate_from_prefix(
    model,
    prefix_embeds: torch.Tensor,
    tokenizer,
    *,
    config: DecodeConfig | None = None,
) -> list[str]:
    """Decode a batch of ClipCap captions from soft prefix embeddings.

    ``prefix_embeds`` is ``(B, P, H)`` already in the LM's embedding space. All
    prefix positions are attended; generation starts straight after them.

    This deliberately delegates to ``model.generate``. The v1 pilot hand-rolled
    this and shipped the cache-aliasing bug; there is no upside to owning the
    cache bookkeeping ourselves.
    """
    cfg = config or DecodeConfig()
    if prefix_embeds.dim() != 3:
        raise ValueError(f"prefix_embeds must be (B, P, H); got {tuple(prefix_embeds.shape)}")

    was_training = model.training
    model.eval()
    try:
        attn = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        out = model.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=attn,
            num_beams=cfg.beam_size,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            length_penalty=cfg.length_penalty,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            early_stopping=True,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    finally:
        model.train(was_training)

    # With inputs_embeds and no input_ids, generate() returns only new tokens.
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in out]
