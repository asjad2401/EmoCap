"""ClipCap: frozen CLIP prefix -> mapping network -> GPT-2 with LoRA.

The architecture is a **control, not a contribution** (preregistration §2). It exists to
turn a data arm into captions under settings identical across all six arms, so that any
difference in the primary metric is a property of the data. Nothing here is tuned per arm,
and every hyperparameter comes from ``configs/model.yaml``.

Shape of one training example
-----------------------------
``[ 10 visual slots | 1 emotion slot | caption tokens ]`` in GPT-2's embedding space. The
emotion is **one prefix slot** -- a learned 768-d vector per register, five in total. That
is the registered conditioning, and the variants that used to compare against it (emotion
broadcast to every slot, emotion-queried cross-attention) were dropped with the
five-condition design. Prefix positions are masked out of the loss with ``-100``: the model
is never asked to predict its own prefix.

Why LoRA is hand-rolled here
----------------------------
``peft`` is not a dependency of this project, and adding one whose version compatibility
cannot be tested from the machine that writes the notebooks is a worse risk than owning
forty lines. GPT-2's ``c_attn`` is a ``Conv1D``, not an ``nn.Linear`` -- its weight is
``(in, out)`` and it is applied as ``x @ W + b``, transposed relative to ``nn.Linear``.
Wrapping it with an adapter that assumed ``nn.Linear`` is a silent shape bug that trains
happily and learns nothing, so the wrapper below matches ``Conv1D`` explicitly.

The base model is frozen; only the mapper, the emotion table and the LoRA factors train.
``trainable_parameters`` is reported into every run manifest because §2 requires the
trainable count matched within ±5% across arms -- which is trivially true here, since the
architecture is identical, but it is asserted rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["ClipCap", "ClipCapConfig", "LoRAConv1D", "apply_lora"]


@dataclass
class ClipCapConfig:
    """Mirrors configs/model.yaml. Built with :meth:`from_config`, never by hand."""

    decoder: str = "gpt2"
    clip_dim: int = 512
    prefix_length: int = 10
    emotion_slots: int = 1
    hidden_mult: int = 2
    n_emotions: int = 5
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("c_attn",)

    @classmethod
    def from_config(cls, cfg: dict) -> "ClipCapConfig":
        m, v, d = cfg["mapping"], cfg["vision"], cfg["decoder"]
        return cls(decoder=d["model"], clip_dim=v["embed_dim"],
                   prefix_length=m["prefix_length"], emotion_slots=m["emotion_slots"],
                   hidden_mult=m["hidden_mult"], lora_r=d["lora"]["r"],
                   lora_alpha=d["lora"]["alpha"], lora_dropout=d["lora"]["dropout"],
                   lora_targets=tuple(d["lora"]["targets"]))


class LoRAConv1D(nn.Module):
    """Low-rank adapter around a GPT-2 ``Conv1D``.

    ``Conv1D`` holds ``weight`` as ``(in_features, out_features)`` and computes
    ``x @ W + b`` -- the transpose of ``nn.Linear``. The factors below follow that
    convention, so ``A`` is ``(in, r)`` and ``B`` is ``(r, out)``. Getting this backwards
    produces a module that runs, trains, and cannot learn.
    """

    def __init__(self, base: nn.Module, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        nx, nf = base.weight.shape
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(nx, r))
        self.B = nn.Parameter(torch.zeros(r, nf))
        self.dropout = nn.Dropout(dropout)
        # A ~ Kaiming, B = 0, so the adapter is an exact no-op at initialisation and the
        # first forward pass reproduces the pretrained model exactly.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.A @ self.B) * self.scale


def apply_lora(model: nn.Module, *, r: int, alpha: int, dropout: float,
               targets: Sequence[str]) -> int:
    """Replace every module whose attribute name is in ``targets``. Returns the count."""
    n = 0
    for module in model.modules():
        for name in list(vars(module).get("_modules", {})):
            if name in targets:
                child = module._modules[name]
                if isinstance(child, LoRAConv1D):
                    continue
                module._modules[name] = LoRAConv1D(child, r, alpha, dropout)
                n += 1
    return n


class MappingMLP(nn.Module):
    """CLIP embedding -> ``prefix_length`` soft tokens in the LM's embedding space."""

    def __init__(self, clip_dim: int, hidden: int, prefix_length: int, embed_dim: int):
        super().__init__()
        self.prefix_length = prefix_length
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, prefix_length * embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(-1, self.prefix_length, self.embed_dim)


class ClipCap(nn.Module):
    """The whole captioner. ``forward`` returns a loss; ``prefix`` feeds the decoder."""

    def __init__(self, cfg: ClipCapConfig):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.cfg = cfg
        self.lm = AutoModelForCausalLM.from_pretrained(cfg.decoder)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        embed_dim = self.lm.get_input_embeddings().embedding_dim
        self.embed_dim = embed_dim
        self.mapper = MappingMLP(cfg.clip_dim, cfg.clip_dim * cfg.hidden_mult,
                                 cfg.prefix_length, embed_dim)
        # One learned vector per register. This IS the conditioning.
        self.emotion = nn.Embedding(cfg.n_emotions, cfg.emotion_slots * embed_dim)
        self.n_lora = apply_lora(self.lm, r=cfg.lora_r, alpha=cfg.lora_alpha,
                                 dropout=cfg.lora_dropout, targets=cfg.lora_targets)

    @property
    def total_prefix(self) -> int:
        return self.cfg.prefix_length + self.cfg.emotion_slots

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def prefix(self, features: torch.Tensor, emotion_ids: torch.Tensor) -> torch.Tensor:
        """``(B, prefix_length + emotion_slots, H)`` -- visual slots then the emotion."""
        vis = self.mapper(features)
        emo = self.emotion(emotion_ids).view(-1, self.cfg.emotion_slots, self.embed_dim)
        return torch.cat([vis, emo], dim=1)

    def forward(self, features: torch.Tensor, emotion_ids: torch.Tensor,
                input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pre = self.prefix(features, emotion_ids)
        tok = self.lm.get_input_embeddings()(input_ids)
        embeds = torch.cat([pre, tok], dim=1)
        attn = torch.cat([
            torch.ones(pre.shape[:2], dtype=attention_mask.dtype, device=pre.device),
            attention_mask], dim=1)
        # The prefix is never a prediction target, and padding is not either -- otherwise
        # the model is scored on reproducing soft vectors and <pad>, and the loss stops
        # meaning "how well does it caption".
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        labels = torch.cat([
            torch.full(pre.shape[:2], -100, dtype=labels.dtype, device=pre.device),
            labels], dim=1)
        return self.lm(inputs_embeds=embeds, attention_mask=attn, labels=labels).loss

    @torch.no_grad()
    def generate(self, features: torch.Tensor, emotion_ids: torch.Tensor, tokenizer,
                 config=None) -> list[str]:
        from emocap.decode.beam import generate_from_prefix

        return generate_from_prefix(self.lm, self.prefix(features, emotion_ids),
                                    tokenizer, config=config)

    def zeroed_visual(self, features: torch.Tensor) -> torch.Tensor:
        """Features with the image zeroed, for the registered visual-dependence probe."""
        return torch.zeros_like(features)
