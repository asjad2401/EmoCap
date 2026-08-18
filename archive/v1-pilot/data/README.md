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
