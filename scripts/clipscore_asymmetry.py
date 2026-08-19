#!/usr/bin/env python
"""Quantify the faithfulness asymmetry between the human and synthetic corpora.

Personality-Captions writers were told to be engaging and were **never required to stay
faithful to the image** ("The snow will last as long as my sadness" describes nothing in
frame). Our captions must describe what is actually there. Unconstrained expression should
be more register-distinctive, so this asymmetry mechanically favours the human arm on
emotion accuracy — and it points straight at the study's central comparison.

Disclosing that is not enough; it has to be bounded. `grounding.py` cannot do it (it needs
a source caption and an independent description, and Personality-Captions has neither), so
this uses **CLIPScore** — reference-free image-text alignment, already a secondary metric.

    uv run python scripts/clipscore_asymmetry.py

Two outputs, and the second is the one that matters:

1. Mean CLIPScore per corpus. Confirms (or refutes) that the human captions are less
   grounded.
2. **Emotion accuracy stratified by CLIPScore quartile, within each corpus.** If poorly
   grounded captions are *easier* to classify, then the human arm's accuracy advantage is
   partly purchased by ignoring the image — and the size of that effect is the bound.

CLIPScore follows Hessel et al. (2021): 2.5 * max(cos(image, text), 0), computed with the
same frozen CLIP ViT-B/32 the captioning models use, so no second encoder is introduced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.prompt import EMOTIONS  # noqa: E402
from emocap.eval.register_classifier import folds_by_image, tfidf_baseline  # noqa: E402

CLIP_W = 2.5   # Hessel et al.'s rescaling; affects the scale, never the ordering


#: Local, self-contained CLIP copy. The HF client stalled twice mid-download with no
#: progress output and no resume, so the weights and the processor configs live on disk and
#: are loaded with `local_files_only`. This also pins the encoder: CLIPScore computed
#: against a silently-updated remote checkpoint would not be comparable across runs.
CLIP_DIR = ROOT / "data/external/clip-vit-base-patch32"


def load_clip(device: str):
    """Load frozen CLIP ViT-B/32 from the local copy, never over the network."""
    from transformers import CLIPModel, CLIPProcessor

    if not (CLIP_DIR / "pytorch_model.bin").exists():
        raise SystemExit(
            f"{CLIP_DIR} is missing pytorch_model.bin. Fetch the repo's config,\n"
            f"preprocessor_config, tokenizer and vocab files plus the weights from\n"
            f"https://huggingface.co/openai/clip-vit-base-patch32 into that directory."
        )
    model = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
    return model, proc


def clipscores(pairs, model, proc, device: str, batch: int = 64, progress=None):
    """pairs: [(image_path, text)] -> [score or None if the image is unreadable]."""
    import torch
    from PIL import Image

    out: list[float | None] = [None] * len(pairs)
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        imgs, texts, idx = [], [], []
        for j, (path, text) in enumerate(chunk):
            try:
                imgs.append(Image.open(path).convert("RGB"))
                texts.append(text)
                idx.append(start + j)
            except Exception:
                continue          # a missing or corrupt image scores None, never 0.0
        if not imgs:
            continue
        with torch.no_grad():
            enc = proc(text=texts, images=imgs, return_tensors="pt",
                       padding=True, truncation=True, max_length=77).to(device)
            o = model(**enc)
            ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
            te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
            cos = (ie * te).sum(-1).clamp(min=0.0) * CLIP_W
        for k, i in enumerate(idx):
            out[i] = float(cos[k])
        if progress:
            progress(min(start + batch, len(pairs)), len(pairs))
    return out


def stratified_accuracy(texts, labels, images, scores, *, folds=5, seed=42) -> list[dict]:
    """Classifier accuracy by CLIPScore quartile, folds split by image as always."""
    keep = [i for i, s in enumerate(scores) if s is not None]
    if len(keep) < 200:
        raise ValueError(f"only {len(keep)} scored captions; too few to stratify")
    ordered = sorted(keep, key=lambda i: scores[i])
    quartiles = [ordered[i::4] for i in range(4)]      # interleaved, then re-sorted below
    quartiles = [sorted(ordered[a:b]) for a, b in
                 [(0, len(ordered) // 4), (len(ordered) // 4, len(ordered) // 2),
                  (len(ordered) // 2, 3 * len(ordered) // 4),
                  (3 * len(ordered) // 4, len(ordered))]]
    rows = []
    for q, members in enumerate(quartiles):
        t = [texts[i] for i in members]
        l = [labels[i] for i in members]
        g = [images[i] for i in members]
        s = [scores[i] for i in members]
        try:
            acc = tfidf_baseline(t, l, g, folds=folds, seed=seed)["accuracy"]
        except Exception as exc:  # noqa: BLE001
            acc = None
        rows.append({
            "quartile": q + 1,
            "n": len(members),
            "clipscore_mean": round(statistics.mean(s), 4),
            "clipscore_range": [round(min(s), 4), round(max(s), 4)],
            "accuracy": acc,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4486, help="captions per corpus (matched)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/clipscore/result.json")
    args = ap.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device {device}\nloading CLIP ViT-B/32 ...", flush=True)
    model, proc = load_clip(device)

    spec = importlib.util.spec_from_file_location(
        "cmp", ROOT / "scripts/compare_human_ceiling.py")
    cmp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp_mod)

    corpora = {}

    # ── human: Personality-Captions, strict 1:1 trait mapping ───────────────
    t, l, g = cmp_mod.load_personality_captions(
        ROOT / "data/external/personality_captions", cmp_mod.STRICT_TRAITS)
    yfcc = ROOT / "data/external/yfcc_images"
    h = [(t[i], l[i], g[i], yfcc / f"{g[i]}.jpg") for i in range(len(t))]
    h = [x for x in h if x[3].exists()]
    random.Random(args.seed).shuffle(h)
    h = h[: args.n]
    corpora["human"] = h

    # ── ours: generated captions on Flickr8k ────────────────────────────────
    recs = [json.loads(x) for x in
            (ROOT / "data/generated/captions_raw.jsonl").read_text().splitlines() if x.strip()]
    imgdir = ROOT / "data/flickr8k/Images"
    ours = []
    for r in recs:
        p = imgdir / r["image_id"]
        for reg, text in (r.get("captions") or {}).items():
            if reg in EMOTIONS and str(text).strip():
                ours.append((str(text).strip(), EMOTIONS.index(reg), r["image_id"], p))
    random.Random(args.seed).shuffle(ours)
    corpora["ours"] = ours[: args.n]

    result = {"clip_w": CLIP_W, "n_requested": args.n, "device": device, "corpora": {}}
    for name, rows in corpora.items():
        print(f"\n{name}: {len(rows):,} captions, scoring ...", flush=True)
        texts = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        images = [r[2] for r in rows]
        scores = clipscores([(r[3], r[0]) for r in rows], model, proc, device,
                            progress=lambda d, tot: print(f"  {d:,}/{tot:,}", flush=True)
                            if d % 1024 == 0 else None)
        got = [s for s in scores if s is not None]
        print(f"  scored {len(got):,}/{len(rows):,}   "
              f"mean CLIPScore {statistics.mean(got):.4f}")
        strat = stratified_accuracy(texts, labels, images, scores)
        result["corpora"][name] = {
            "n_scored": len(got),
            "clipscore_mean": round(statistics.mean(got), 4),
            "clipscore_sd": round(statistics.stdev(got), 4),
            "stratified": strat,
        }
        for row in strat:
            print(f"    Q{row['quartile']}  n={row['n']:>5}  "
                  f"CLIPScore {row['clipscore_mean']:.3f}  "
                  f"accuracy {row['accuracy']:.3f}")

    hm = result["corpora"]["human"]["clipscore_mean"]
    om = result["corpora"]["ours"]["clipscore_mean"]
    print(f"\n{'=' * 68}")
    print(f"mean CLIPScore   human {hm:.4f}   ours {om:.4f}   diff {om - hm:+.4f}")
    if om > hm:
        print("Our captions are MORE grounded, as designed. The human arm's emotion")
        print("accuracy is therefore partly purchased by ignoring the image.")
    else:
        print("Human captions are no less grounded than ours -- the asymmetry we assumed")
        print("does not appear in CLIPScore, and the disclosure should say so.")

    for name in ("human", "ours"):
        st = result["corpora"][name]["stratified"]
        lo, hi = st[0]["accuracy"], st[-1]["accuracy"]
        print(f"\n{name}: accuracy Q1 (least grounded) {lo:.3f} -> "
              f"Q4 (most grounded) {hi:.3f}   delta {hi - lo:+.3f}")
        print("  => less grounded is EASIER to classify; register signal partly "
              "substitutes for description" if lo > hi else
              "  => groundedness does not trade off against register recoverability")

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
