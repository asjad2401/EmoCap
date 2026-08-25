#!/usr/bin/env python
"""The registered secondary metrics: CIDEr-D, BLEU-4, CLIPScore, distinctiveness.

    uv run --extra eval python scripts/secondary_metrics.py --run runs/arms/S_paired5-f0
    uv run --extra eval python scripts/secondary_metrics.py --all

Writes ``<run>/secondary.json``. Needs Java on PATH for ``PTBTokenizer``.

**Why these exist.** The primary metric asks one question: can a frozen classifier recover
the register a caption was asked for. A model can score 0.91 on that while writing captions
that are repetitive, ungrounded, or barely describe the photograph. Nothing in the primary
analysis would notice. `metrics.secondary` registers four measures that would.

The four, and what each one can catch
-------------------------------------
``CIDEr-D`` / ``BLEU-4``  against the five original Flickr8k captions per image. Content
                          overlap with what humans said about that photo. A register-
                          conditioned caption is *supposed* to differ from a neutral one, so
                          these are not quality scores in the usual sense -- they are a floor
                          check that the caption is still about the picture.
``CLIPScore``             reference-free grounding, ``2.5 * max(cos(image, text), 0)`` after
                          Hessel et al. (2021). Needs no references, so it is the only one of
                          the four that works on the human arm.
``distinctiveness``       mean pairwise self-BLEU across the five register captions written
                          for one image. **This is the one that matters here.** If a model
                          writes one caption and swaps two tone words, self-BLEU is near 1
                          and the register conditioning is cosmetic -- a failure mode the
                          classifier would happily score as success.

**CIDEr-D is corpus-dependent and MUST NOT be compared across arms at different n.**
Its IDF term is computed from the reference set it is given, so the same captions score
differently depending on how many images are in the evaluation. This is the same trap the
lexical anchor has, and three findings on this project were already retracted for it (see
`docs/lab-notebook.md`, 2026-08-19). Every CIDEr value here is written with its ``n_images``,
and the script prints which cross-arm comparisons are safe: arms sharing a fold share their
held-out images and their n, so ``S_paired5`` vs ``V1_paired5`` is comparable while
``S_paired25`` vs ``S_unpaired`` is not. BLEU-4 has no corpus term and CLIPScore is computed
per pair, so neither is affected.

**Two arms cannot take every metric, and that is the data, not a gap.**
``H_unpaired`` is Personality-Captions over YFCC: there is no five-reference set for those
photographs, so CIDEr-D and BLEU-4 are undefined and reported as null with the reason.
The three 1-caption-per-image arms carry one register per image, so distinctiveness -- which
is defined across five registers of one image -- is undefined for them too.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.runtime import load_config, rel  # noqa: E402

CLIP_W = 2.5  # Hessel et al.'s rescaling; affects the scale, never the ordering


def flickr_references(path: Path) -> dict[str, list[str]]:
    """``image_id -> [5 human captions]`` from Flickr8k.token.txt."""
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        tag, text = line.split("\t", 1)
        img = tag.split("#")[0]
        out.setdefault(img, []).append(text.strip())
    return out


def caption_index(arm: str, data_root: Path) -> dict[tuple[str, str, str], int]:
    """``(image_id, register, text) -> caption_idx`` from the arm file.

    ``predictions.jsonl`` carries the reference text but not which source caption it came
    from, and S_paired25 holds five per (image, register). Distinctiveness is defined over
    the five registers of ONE caption, so the groups have to be reconstructed rather than
    guessed -- pooling all 25 would compare captions written from different source text and
    report them as insufficiently distinct.
    """
    path = data_root / "arms" / f"{arm}.jsonl"
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str], int] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["image_id"], r["emotion"], r["text"])] = r["caption_idx"]
    return out


def tokenize(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    return PTBTokenizer().tokenize(
        {k: [{"caption": c} for c in v] for k, v in groups.items()})


def cider_bleu(hyps: dict[str, list[str]], refs: dict[str, list[str]]) -> dict:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    keys = sorted(set(hyps) & set(refs))
    if not keys:
        return {}
    h = {k: hyps[k] for k in keys}
    r = {k: refs[k] for k in keys}
    cider, _ = Cider().compute_score(r, h)
    bleu, _ = Bleu(4).compute_score(r, h)
    return {"cider_d": round(float(cider), 4), "bleu_4": round(float(bleu[3]), 4),
            "bleu_1": round(float(bleu[0]), 4), "n_scored": len(keys)}


def self_bleu(groups: list[list[str]]) -> float | None:
    """Mean pairwise self-BLEU. 1.0 = five identical captions, 0 = nothing shared.

    Each caption is scored against the other four as references, which is the standard
    leave-one-out formulation. LOWER IS BETTER: it means the five registers actually
    produced different sentences rather than one sentence with the tone words swapped.
    """
    from pycocoevalcap.bleu.bleu import Bleu

    hyps: dict[str, list[str]] = {}
    refs: dict[str, list[str]] = {}
    for gi, g in enumerate(groups):
        if len(g) < 2:
            continue
        for ci, cap in enumerate(g):
            key = f"{gi}:{ci}"
            hyps[key] = [cap]
            refs[key] = [c for j, c in enumerate(g) if j != ci]
    if not hyps:
        return None
    bleu, _ = Bleu(4).compute_score(refs, hyps)
    return round(float(bleu[3]), 4)


def clipscores(texts: list[str], images: list[str], store, device: str,
               batch: int = 256) -> list[float]:
    import numpy as np
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    weights = ROOT / "data/external/clip-vit-base-patch32"
    tok = CLIPTokenizerFast.from_pretrained(str(weights))
    model = CLIPTextModelWithProjection.from_pretrained(str(weights)).to(device).eval()

    out: list[float] = []
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=77,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            te = model(**enc).text_embeds
        te = torch.nn.functional.normalize(te.float(), dim=-1)
        ie = torch.from_numpy(
            np.stack([np.asarray(store[i]) for i in images[s:s + batch]])).to(device)
        ie = torch.nn.functional.normalize(ie.float(), dim=-1)
        cos = (ie * te).sum(-1).clamp(min=0.0) * CLIP_W
        out.extend(float(x) for x in cos.cpu())
        if s % (batch * 40) == 0:
            print(f"    clipscore {s + len(chunk):,}/{len(texts):,}", flush=True)
    return out


def one_run(run: Path, refs: dict[str, list[str]], data_root: Path, device: str) -> dict:
    from emocap.features.clip import load_features

    man = json.loads((run / "manifest.json").read_text())
    arm = man["config"]["arm"]
    preds = [json.loads(l) for l in (run / "predictions.jsonl").read_text().splitlines()
             if l.strip()]
    print(f"{run.name}   {len(preds):,} captions   arm {arm}")

    result: dict = {"run": run.name, "arm": arm, "fold": man["config"]["fold"],
                    "negative_control": man["config"]["negative_control"],
                    "n_captions": len(preds)}

    # ── CIDEr-D / BLEU-4 ─────────────────────────────────────────────────────
    if arm.startswith("H_"):
        result["cider_bleu"] = None
        result["cider_bleu_reason"] = (
            "Personality-Captions has no five-reference set for its YFCC photographs; "
            "these metrics are undefined for this arm, not missing")
    else:
        # One entry per (image, register): a hypothesis needs its own key, and the same
        # image's five registers are five different hypotheses against the same references.
        hyps = {f"{p['image_id']}|{p['emotion']}|{i}": [p["generated"]]
                for i, p in enumerate(preds)}
        rmap = {k: refs[k.split("|")[0]] for k in hyps if k.split("|")[0] in refs}
        tok_h = tokenize({k: v for k, v in hyps.items() if k in rmap})
        tok_r = tokenize(rmap)
        cb = cider_bleu(tok_h, tok_r)
        cb["n_images"] = len({k.split("|")[0] for k in rmap})
        cb["cider_warning"] = ("CIDEr-D's IDF comes from this reference set, so it is "
                               "comparable only against a run with the same n_images")
        result["cider_bleu"] = cb

    # ── distinctiveness ──────────────────────────────────────────────────────
    idx = caption_index(arm, data_root)
    buckets: dict[tuple[str, int], dict[str, str]] = {}
    for p in preds:
        ci = idx.get((p["image_id"], p["emotion"], p["reference"]))
        if ci is None:
            continue
        buckets.setdefault((p["image_id"], ci), {})[p["emotion"]] = p["generated"]
    groups = [[b[e] for e in EMOTIONS if e in b]
              for b in buckets.values() if len(b) == len(EMOTIONS)]
    if groups:
        result["distinctiveness"] = {
            "self_bleu_4": self_bleu(groups), "n_groups": len(groups),
            "reading": "self-BLEU over the five registers of one source caption; "
                       "LOWER is more distinct, 1.0 would mean five identical captions"}
    else:
        result["distinctiveness"] = None
        result["distinctiveness_reason"] = (
            "this arm carries one register per image, so there is no five-register group "
            "to compare; undefined rather than missing")

    # ── CLIPScore ────────────────────────────────────────────────────────────
    store = load_features(data_root / "features")
    keep = [p for p in preds if p["image_id"] in store]
    scores = clipscores([p["generated"] for p in keep],
                        [p["image_id"] for p in keep], store, device)
    result["clipscore"] = {
        "mean": round(statistics.mean(scores), 4),
        "sd": round(statistics.stdev(scores), 4) if len(scores) > 1 else None,
        "n": len(scores), "weight": CLIP_W,
        "formula": "2.5 * max(cos(image_embed, text_embed), 0), Hessel et al. 2021"}
    if len(keep) < len(preds):
        result["clipscore"]["missing_features"] = len(preds) - len(keep)
    return result


def show(r: dict) -> None:
    cb = r.get("cider_bleu")
    d = r.get("distinctiveness")
    cs = r["clipscore"]
    print(f"  CLIPScore      {cs['mean']:.4f}" + (f" (sd {cs['sd']:.4f})" if cs["sd"] else "")
          + f"   n={cs['n']:,}")
    if cb:
        print(f"  CIDEr-D        {cb['cider_d']:.4f}   over {cb['n_images']:,} images "
              f"-- comparable only at equal n")
        print(f"  BLEU-4         {cb['bleu_4']:.4f}   (BLEU-1 {cb['bleu_1']:.4f})")
    else:
        print(f"  CIDEr-D/BLEU   n/a -- {r['cider_bleu_reason']}")
    if d:
        print(f"  self-BLEU-4    {d['self_bleu_4']:.4f}   over {d['n_groups']:,} "
              f"five-register groups -- LOWER is more distinct")
    else:
        print(f"  self-BLEU      n/a -- {r['distinctiveness_reason']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="a runs/arms/<tag> directory")
    ap.add_argument("--all", action="store_true", help="every run under --runs-dir")
    ap.add_argument("--runs-dir", default="runs/arms")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    import torch
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root) if args.data_root else ROOT / "data"
    cfg = load_config("data")
    refs = flickr_references(ROOT / cfg["paths"]["captions_file"])
    print(f"{len(refs):,} Flickr8k images with references   device {device}\n")

    runs: list[Path]
    if args.all:
        runs = sorted(d for d in (ROOT / args.runs_dir).iterdir()
                      if (d / "predictions.jsonl").exists())
    elif args.run:
        p = Path(args.run)
        runs = [p if p.is_absolute() else ROOT / p]
    else:
        raise SystemExit("pass --run <dir> or --all")

    for run in runs:
        out = run / "secondary.json"
        if args.skip_existing and out.exists():
            print(f"{run.name}: already present, skipping\n")
            continue
        r = one_run(run, refs, data_root, device)
        out.write_text(json.dumps(r, indent=2))
        show(r)
        print(f"  -> {rel(out)}\n", flush=True)


if __name__ == "__main__":
    main()
