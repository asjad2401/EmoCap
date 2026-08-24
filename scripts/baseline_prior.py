#!/usr/bin/env python
"""What the frozen language model supplies on its own. A post-hoc baseline.

    uv run python scripts/baseline_prior.py --arm S_paired5 --fold 0
    uv run python scripts/baseline_prior.py --arm S_paired5 --fold 0 --mode text

    uv run python scripts/score_arm.py --run runs/baselines/BASE_prior-S_paired5-f0

Decided on 2026-08-23, after the ``prereg-v2`` tag, and post-hoc wherever it is reported.

**Why the study needs it.** Every arm is measured against its own keyword anchor, which
says how much accuracy a lexical rule reaches with no model. Nothing yet says how much
GPT-2 reaches with no *image*. Without that, "this arm is a good emotion-conditioned
captioner" has nothing to be good relative to. This trains nothing: the same frozen GPT-2
the arms use, prompted with the register word, decoded with the frozen ``DecodeConfig``,
scored by the same frozen classifier.

Two modes, because one of them is degenerate and that is itself the finding
--------------------------------------------------------------------------
``--mode prior`` is the baseline named in ``docs/remaining-work.md``: the register word
and nothing else. Beam search is deterministic and the prompt depends only on the
register, so this produces **exactly five captions** for the whole held-out set. Its
accuracy is a draw over five outcomes, not a measurement -- the same trap the blanked
visual-dependence probe already walked into, where a negative control appeared to score
0.58. It is generated once per register and copied, and the collapse is printed rather
than buried. Report it as a floor, with the five captions beside it.

``--mode text`` conditions on the neutral Flickr8k source caption as well, and is the
baseline that actually carries information: same frozen weights, same absence of an image,
but per-cell variation. It answers the sharper question -- how much of an arm's accuracy
is reachable from the source text alone, with the photograph removed. An arm that barely
beats it is doing much less than its raw accuracy suggests.

**Quote margins, not raw accuracy, against this.** The study's central finding is that
most of the primary metric is keyword-reachable, so leaning on raw accuracy to praise a
model contradicts the paper's own argument. ``score_arm.py`` recomputes the anchor on
these captions at this n, exactly as it does for a trained arm.

**Post-processing is stated, and it favours the baseline.** A free-running LM writes past
the end of a caption, so output is cut at the first newline. That makes the baseline more
caption-like and therefore *stronger*, which is the safe direction for a floor: any
comparison it loses, it loses on merit rather than on formatting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.arms import load_corpus  # noqa: E402
from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.decode.beam import DecodeConfig  # noqa: E402
from emocap.models.dataset import load_arm, make_split  # noqa: E402
from emocap.runtime import Manifest, load_config, seed_everything  # noqa: E402

#: Fixed before any baseline score existed. Prompt wording is a real degree of freedom --
#: a better prompt would raise the floor -- so these are recorded in the run manifest and
#: must not be tuned against results. If they are ever changed, the old runs stop being
#: comparable and have to be regenerated.
TEMPLATES = {
    "prior": "A {register} photo caption:",
    "text": "Photo caption: {source}\nThe same photo, described in a {register} way:",
}


def build_prompt(mode: str, register: str, source: str) -> str:
    return TEMPLATES[mode].format(register=register, source=source)


def first_line(text: str) -> str:
    """Cut a free-running continuation down to one caption. See the module docstring."""
    return text.strip().split("\n")[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="S_paired5",
                    help="whose held-out cells to caption; the baseline is scored on the "
                         "same cells as the arm it is compared against")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--mode", choices=sorted(TEMPLATES), default="prior")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default="runs/baselines")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: cap held-out cells -- NOT for a reported run")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config("model")
    lock = load_config("prereg.lock")
    dc = DecodeConfig(**dict(lock["decode"]))
    root = Path(args.data_root) if args.data_root else ROOT / "data"
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    seed_everything(args.seed)

    _, held = make_split(load_arm(root / "arms" / f"{args.arm}.jsonl"), args.fold)
    cells = held.cells[:args.limit] if args.limit else held.cells

    sources: dict[tuple[str, int], str] = {}
    if args.mode == "text":
        corpus = load_corpus(root / "generated/captions_corpus.jsonl")
        sources = {k: str(r.get("source_caption") or "").strip()
                   for k, r in corpus.items()}
        blank = [c for c in cells if not sources.get((c["image_id"], c["caption_idx"]))]
        if blank:
            raise SystemExit(f"{len(blank)} cells have no source_caption; "
                             f"first: {blank[0]['image_id']}")

    name = cfg["decoder"]["model"]
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    # Left padding, or a right-padded batch resumes generation after the pad tokens and
    # every short prompt decodes gibberish.
    tok.padding_side = "left"
    lm = AutoModelForCausalLM.from_pretrained(name).to(device).eval()

    tag = f"BASE_{args.mode}-{args.arm}-f{args.fold}"
    run_dir = ROOT / args.out_root / tag
    man = Manifest(run_id=tag, stage="09_baseline_prior", config={
        "arm": f"BASE_{args.mode}", "fold": args.fold, "negative_control": False,
        "post_hoc": True, "decided": "2026-08-23",
        "cells_from": args.arm, "held_out_cells": len(cells),
        "limit": args.limit,
        "trainable_parameters": 0, "model": name, "device": device,
        "seed": args.seed, "template": TEMPLATES[args.mode],
        "post_process": "first line only",
        "decode": dc.as_dict()})
    man.save(run_dir)
    print(f"{tag}  cells {len(cells):,}  frozen {name}  device {device}", flush=True)

    @torch.no_grad()
    def decode(prompts: list[str]) -> list[str]:
        out: list[str] = []
        t0 = time.time()
        for s in range(0, len(prompts), args.batch_size):
            enc = tok(prompts[s:s + args.batch_size], return_tensors="pt",
                      padding=True, truncation=True, max_length=64).to(device)
            gen = lm.generate(
                **enc, num_beams=dc.beam_size, max_new_tokens=dc.max_new_tokens,
                min_new_tokens=dc.min_new_tokens, length_penalty=dc.length_penalty,
                no_repeat_ngram_size=dc.no_repeat_ngram_size, early_stopping=True,
                do_sample=False, num_return_sequences=1,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            # Padding is on the left and identical in width across the batch, so the
            # prompt occupies exactly the first `enc.input_ids.shape[1]` columns.
            for row in gen[:, enc["input_ids"].shape[1]:]:
                out.append(first_line(tok.decode(row, skip_special_tokens=True)))
            if s % (args.batch_size * 20) == 0:
                print(f"  decoded {s + len(enc['input_ids']):,}/{len(prompts):,}  "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
        return out

    t0 = time.time()
    if args.mode == "prior":
        # The prompt depends only on the register, and beam search is deterministic. One
        # generation per register is the whole output; decoding it thousands of times
        # would burn GPU hours to recompute five strings.
        caps = dict(zip(EMOTIONS, decode([build_prompt("prior", e, "") for e in EMOTIONS])))
        generated = [caps[c["emotion"]] for c in cells]
    else:
        generated = decode([build_prompt("text", c["emotion"],
                                         sources[(c["image_id"], c["caption_idx"])])
                            for c in cells])

    preds = [{"image_id": c["image_id"], "emotion": c["emotion"],
              "reference": c["text"], "generated": g, "fold": args.fold}
             for c, g in zip(cells, generated)]
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in preds))

    # No predictions_novis.jsonl on purpose. The visual-dependence probe asks whether
    # captions change when the image is removed; this baseline never had an image, so the
    # probe has nothing to compare and score_arm.py correctly reports none.
    uniq = len(set(generated))
    stats = {"mode": args.mode, "wall_minutes": round((time.time() - t0) / 60, 1),
             "decoded": len(preds), "trainable_parameters": 0,
             "empty_captions": sum(1 for g in generated if not g),
             "unique_captions": uniq,
             "template": TEMPLATES[args.mode],
             "distinct_prompts": len(EMOTIONS) if args.mode == "prior" else len(cells)}
    if args.mode == "prior":
        stats["captions_by_register"] = caps
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    man.finalise(run_dir)

    print(f"\n{tag}  {len(preds):,} cells, {uniq} unique captions  -> {run_dir}")
    if args.mode == "prior":
        print("  the prior has one caption per register; its accuracy is a draw over "
              f"{uniq} outcomes, not an effect:")
        for e in EMOTIONS:
            print(f"    {e:<9} {caps[e]!r}")
    print(f"\n  score it:  uv run python scripts/score_arm.py --run "
          f"{run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
