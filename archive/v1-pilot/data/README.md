# v1 Pilot Data — Provenance Only

**Do not train on this.** It is retired as a training asset and kept as the record of
what the pilot actually ran on. See [`../../../docs/v1-pilot-postmortem.md`](../../../docs/v1-pilot-postmortem.md).

## `v1_emotion_captions.csv.gz`

The caption set `tokenizationfinal.ipynb` consumed. 43,891 rows (including header) over
~7,945 unique images — roughly 8,000 Gemini calls, which is why it is preserved rather than
deleted.

Stored gzipped (17 MB → 2.6 MB). Byte-identical on decompression; verified by sha256 at
the time of compression. `pandas.read_csv` reads it directly:

```python
df = pandas.read_csv("archive/v1-pilot/data/v1_emotion_captions.csv.gz")
```

| Column | Notes |
|---|---|
| `image_id` | Flickr8k filename |
| `img_path` | dead `/kaggle/input/...` path from the pilot session |
| `neutral_caption` | **Moondream VLM output**, not a human annotation. Mean 37.9 words, multi-sentence. |
| `emotion` / `emotion_id` | one of joyful, sad, tense, romantic, humorous |
| `emotion_caption` | Gemini rewrite. **One per (image, emotion)** — the single-reference problem. |
| `quality_score` | the pilot's own heuristic. Mean 0.56, and unreliable — the pilot disabled filtering on it. |

## `v1_moondream_factual_captions.csv.gz`

The Moondream VLM descriptions that fed the pilot's generation prompt — the pilot's
"neutral" captions. 8,091 rows, one per Flickr8k image. Gzipped 1.9 MB → 592 KB,
sha256-verified byte-identical. Columns: `image_id`, `factual_caption`.

Measured against Flickr8k's five human captions per image:

| | Human (5 captions) | Moondream (1) |
|---|---|---|
| Unique content words per image | **20.1** | 18.1 |
| Colour words per image | 1.37 | **2.46** |
| Spatial-relation words per image | **0.81** | 0.30 |

Moondream omits 68.4% of the content words the five humans mention, and 63.9% of its own
content words appear in no human caption — unverifiable without the image. It loses actions
("jumping onto a sled" becomes "is sledding"), contradicts human attribute descriptions, and
**3.0% of its captions (241 images) are degenerate repetition loops** — the worst repeats
"The flag is flying." 80 times, another "The dog is wearing a collar." 73 times, at up to 481
words. It does carry more colour and small-object detail, which is the real gap in terse
human captions.

v2 supersedes this by giving Gemini the **image itself** plus the five human captions, which
gets that atmospheric detail from a far stronger model with no intermediate hallucination
layer to inherit. See `docs/deviations.md`.

## Why v2 does not use it

1. **One reference per `(image, emotion)` cell.** BLEU-4 against a single 25-word ornate
   reference floors near zero for any model, so the pilot's metric could not have separated
   its variants.
2. **Figurative style leaked throughout.** The prompt forbade metaphor; the output is full
   of it. v2's deterministic filter catches six of six sampled examples from this file — see
   `tests/test_data_generate.py::test_rejects_the_figurative_style_v1_leaked`, where they are
   pinned verbatim as regression fixtures.
3. **1,827 images were flagged for regeneration and never regenerated.** The repair loop was
   quadratic and abandoned after image 1. Those rows are still in here.
4. **The neutral source is a VLM.** v2 uses Flickr8k's five human captions, which is what
   yields five references per cell and makes the numbers comparable to published work.

This file remains useful for exactly one thing: quoting the pilot's failure modes in the
write-up with real examples.
