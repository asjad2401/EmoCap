# v1 Pilot — Postmortem

The v1 pilot is preserved in [`archive/v1-pilot/`](../archive/v1-pilot/). It produced no
usable results. This document records why, because the reasons are what shaped the v2
design — and because "we tried this first and here is what it taught us" is legitimate
content for the write-up.

## What the pilot was

Seven notebooks in three generations:

| Files | What they were |
|---|---|
| `01_data_preparation`, `02_model_training`, `03_evaluation_analysis` | earliest Colab drafts — GPT-2 caption rewrites, frozen ResNet-50, LSTM decoder |
| `trainold` | LSTM-only training, the version immediately before the final one |
| `caption-generation`, `tokenizationfinal`, `notebooka1ac23db48` | the final chain: data → vocab → two-track training |

The final training notebook was written to train **eight** models: Track A (four LSTM
conditioning variants) and Track B (four ClipCap/GPT-2+LoRA variants).

## How far it actually got

Not far, and this was invisible from the notebook's own narrative.

- **One model trained.** `CC_ARCHS` had been cut down to a single variant
  (`B3_CCEmotionAtEvery`) during debugging and never restored.
- **Track A Stage B stopped at epoch 8 of 20**, interrupted.
- **BLEU evaluation never completed.** It died on a `KeyboardInterrupt` inside GPT-2's MLP,
  after per-sample beam search over 3,722 test rows proved far too slow.
- **The results tables never ran at all** — those cells have no execution count.

So there was never an ablation table. The "implausible results" were the qualitative
samples from a single model, decoded by a broken decoder.

## Root causes, ranked

### 1. The beam search corrupted every caption

`notebooka1ac23db48.ipynb`, cell 34:

```python
for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
    new_beams.append((score + lp, toks + [idx], out.past_key_values))
```

Every beam child was pushed holding the **same** `past_key_values` object. Modern
`transformers` returns a `DynamicCache` and mutates it in place, so beam 1's forward
extended the cache beam 0 had just written to, beam 2 extended that, and so on. Each beam
decoded against a mixture of its siblings' tokens; by step 30 the cache held roughly 130
positions instead of 40.

**This code would have been correct under the legacy tuple-of-tensors cache API.** It broke
silently on a library upgrade, with no error and no warning.

The evidence fits exactly. Track B's validation loss was *healthy* — 3.10, perplexity ≈ 22,
because teacher forcing never runs the decoder — while its samples were word salad:

```
GEN: striped shirt, the hat, aubergebrite.
GEN: the boy, black, skis, pure joy!
GEN: two black and white and brown and brown.
```

And the few short outputs that escaped contamination were fine: `two dogs run through the
green lawn` is grammatical and correctly grounded. **The model had learned; the decoder
destroyed the output.**

> **The lesson, generalised:** healthy loss with broken samples is a decoder bug, essentially
> always, because training never exercises the decoder. This belongs in the lab notebook of
> every future run.

### 2. One reference per (image, emotion) cell

Each cell had exactly one 25-word ornate reference. BLEU-4 against a single reference of
that length floors near zero for any model, so the metric could not have separated the
variants even with a working decoder.

### 3. Targets were truncated mid-sentence

`MAX_SEQ_LEN = 30` capped captions at 28 CLIP BPE ids ≈ 22 words. Emotion captions averaged
22.1 words, so roughly half were cut. Neutral captions averaged 37.9 words, so nearly every
Stage-A target was chopped mid-sentence with EOS appended — training the decoder to stop at
arbitrary points.

### 4. Track A collapsed to a language prior

From `trainold.ipynb` cell 25, the LSTM baseline emitted *the same* caption for every image:

```
Image: two people on a rocky outcropping overlooking a river
GEN:   the man in the red shirt and blue jeans stands alone on the dirt path, ...
```

The visual CLS entered only through `h0` and was ignored. Separately, every reported loss
sat ~1.24 nats above true cross-entropy, because `label_smoothing=0.1` over a 9,199-token
vocab has that floor — so Stage B's val 5.2 was really ≈3.96 CE. Any comparison against
those logged numbers is off by a constant nobody had subtracted.

### 5. Style leakage in the generated data

The generation prompt forbade metaphor and poetry. Gemini ignored it on most rows: *"a
tender ballet performed beneath the gaze of the red wooden house"*. The filter caught only
13 hardcoded regexes.

### 6. 1,827 flagged images were never regenerated

The repair loop re-read and rewrote the entire 35,000-row CSV once per image — quadratic —
and was abandoned after image 1 of 1,827. The flagged rows stayed in the training data.

### 7. A vocabulary off-by-one

`VOCAB_SIZE = len(local2clip)` gave 9,199, but `local2clip` never got an entry for
`UNK = 3`, so ids ran 0, 1, 2, 4…9199 and the maximum id *equalled* the vocab size. The
training notebook detected this and papered over it by remapping 9199 → UNK.

### 8. Documentation described a pipeline that was not running

`tokenizationfinal.ipynb`'s header advertised hard filters — quality ≥ 0.5, length-delta
≤ 12, overlap ≥ 0.25. Cell 10 then overrode them to quality-disabled, delta ≤ 35, overlap
≥ 0.03, keeping 91.4% of rows. The prose and the code had silently diverged.

## What v2 changes as a result

| Cause | v2 countermeasure |
|---|---|
| 1 | Decoder moved to `src/emocap/decode/beam.py`, cross-validated against HuggingFace's beam search, and covered by tests that fail against the v1 implementation. Track B calls `model.generate()` rather than owning a cache. |
| 2 | Five references per cell by construction, from Flickr8k's five human captions. |
| 3 | Sequence length set from the measured corpus percentile; truncation rate asserted < 2%. |
| 4 | Visual-dependence probe gates every run. Raw CE logged separately from smoothed loss. |
| 5 | LLM-judge rubric replaces regexes; 200-row manual audit before the full generation run. |
| 6 | Append-only JSONL keyed by `(image_id, caption_idx, emotion)`; regeneration is a filter over keys. |
| 7 | Vocabulary built as an explicit ordered list; `max(id) == size − 1` asserted at save time. |
| 8 | Every threshold lives in `configs/`, is read once, and is echoed into each run's manifest. Prose never restates a number the config owns. |

## The thing worth keeping

The pilot's model architectures were not the problem. All eight survive into v2 largely
intact, with the visual-feature handling corrected. What failed was everything around them:
the decoder, the metric, the data, and the absence of any way to notice.
