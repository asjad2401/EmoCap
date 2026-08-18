# Deviations Log

Every departure from [`docs/preregistration.md`](preregistration.md) and the published plan,
dated, with a reason. Append only.

Deviations recorded *before* the `prereg-v1` tag are design refinements. Anything after the
tag is a real deviation and must be reported in the write-up.

---

## Pre-tag (design refinements)

### 2026-08-18 — `src/emocap/vocab/` instead of `src/emocap/tokenize/`
The published plan named the subpackage `tokenize/`. Renamed to `vocab/` to avoid a name
that collides with Python's standard-library `tokenize` module — harmless for absolute
imports, but a confusing thing to leave in a research codebase.

### 2026-08-18 — Python 3.12.14 rather than 3.11
`pyproject.toml` requires `>=3.11`; `uv` resolved 3.12.14. No reason to pin lower. Locked
versions: `torch 2.13.0`, `transformers 5.15.0`.

### 2026-08-18 — `DecodeConfig.length_penalty` semantics stated explicitly
Normalisation is `sum_logprob / len ** length_penalty` over the **generated** length,
matching HuggingFace's `BeamSearchScorer` so Track A and Track B are scored on one scale.
Note that HF normalises by *full sequence* length; for Track B the prefix is not part of the
generated sequence, so the two agree. Verified by
`tests/test_decode_vs_hf.py::test_matches_hf_beam_search`.

### 2026-08-18 — generation store keyed by `(image_id, caption_idx)`, not `(…, emotion)`
The plan named the key as `(image_id, caption_idx, emotion)`. One API call produces all five
registers, so the store holds one JSONL record per `(image_id, caption_idx)` with a
`captions: {emotion: text}` map. Resume is a set difference on that pair; a record missing
any of the five is not counted as complete, so it is retried. Per-emotion regeneration is
still possible by rewriting a record. Fewer lines, simpler resume, same granularity.

### 2026-08-18 — v1 caption CSV stored gzipped in the archive
`archive/v1-pilot/data/v1_emotion_captions.csv.gz`, 17 MB → 2.6 MB, verified byte-identical
by sha256 before the original was removed. Also renamed from
`emotion_captions (3) (1).csv`. Provenance is preserved; the repo stays clonable.

### 2026-08-18 — generation is multimodal: image + 5 human captions
The plan described text-only generation from a neutral caption. Gemini 2.0 Flash is natively
multimodal, so each call now receives the **image** plus that image's five human captions.
This removes the `image -> Moondream -> text -> Gemini` two-hop and its hallucination layer.

Measured justification on the real files: Moondream omits 68.4% of the content words five
humans mention, uses 2.7x fewer spatial-relation words, drops actions ("jumping onto a sled"
becomes "is sledding"), contradicts human attribute descriptions, and 3.0% of its captions
are degenerate repetition loops (worst case: "The flag is flying." repeated 80 times). It
does carry more colour vocabulary (2.46 vs 1.37 words per image) and more small-object
detail, which is the real gap in terse human captions -- and which Gemini can supply from the
image directly, from a much stronger model.

Constraint carried into the prompt: targets must stay within what CLIP ViT-B/32 at 224px can
recover, so the prompt steers toward global properties (lighting, weather, density, setting,
posture, expression) and away from small-object inventory. A target carrying unrecoverable
detail trains confident invention.

`neutral_source: human_annotations` is unchanged -- the human captions remain the content
anchor and the reference set.

### 2026-08-18 — use the official Flickr8k splits, not a random split
The plan specified an 80/10/10 random split by `image_id`, seed 42. The downloaded
distribution ships the original Hodosh et al. 2013 lists (`Flickr_8k.trainImages.txt`,
`devImages`, `testImages`) at 6000/1000/1000. Every published Flickr8k number uses these, so
adopting them makes our BLEU/CIDEr directly comparable -- which was a main argument for using
human captions in the first place.

The official lists cover 8,000 of the 8,091 valid images. The remaining 91 are assigned to
train with `split_seed: 42`, so nothing leaks into val or test and no image is discarded.

---

## Post-tag (real deviations)

*None yet.*
