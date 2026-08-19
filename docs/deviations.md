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

### 2026-08-18 — reference model is `gemini-3.6-flash`, chosen on price and task fit
Costed the full run from measured per-call token counts against user-supplied prices
(now in `configs/data.yaml` under `pricing`, dated and sourced):

| plan | model | total |
|---|---|---|
| B (per image) | 3.1-flash-lite | **$12.32** |
| A (per caption) | 3.1-flash-lite | $27.39 |
| B | **3.6-flash** | **$35.77** |
| B | 3.5-flash | $79.51 |
| A | 3.6-flash | $92.63 |
| A | 3.5-flash | **$192.36** |

Halve all of it with the Batch API, which fits this workload exactly.

`gemini-3.5-flash` is dropped as the reference: it costs 2x the input and 2.4x the
output of the **newer** 3.6-flash, and is tuned for coding and reasoning, which this
task does not use, while 3.6-flash is specifically stronger at visual reasoning, which
it does. Kept in `probe_models` only as a quality ceiling.

Three findings that shape the plan:

* **Image tokens are a flat 1,064 per call at every resolution** — 224px, 384px, 512px
  and native all cost identically (measured with the API's own counter; 0 with no image
  attached, which also confirms the image really is being sent). So `--max-image-dim` is
  **not a cost lever at all**, and resolution is now a pure quality decision.
* **Batching is the only input-side lever**, worth a consistent 55-61%.
* **Thinking is a 10-12% output surcharge that cannot be switched off** on the full
  flash models. Only the lite models honour `thinking_budget=0`, which is part of why
  3.1-flash-lite is 6x cheaper.

3.6-flash's introductory rate **ends 2026-12-31** and then doubles to $1.50/$7.50, so a
regeneration in 2027 costs twice as much — an argument for getting the prompt right
before the full run rather than after. `scripts/cost_report.py --standard-rates` models
that case.

An earlier estimate in this conversation put the A-vs-B gap at "under four dollars"
using guessed prices of $0.10/$0.40 per 1M. The real rates are 6-15x higher and the
gap is 16x, not marginal. Recorded because the lesson is the general one: do not reason
about cost from remembered prices.

### 2026-08-18 — listing is not usability; the whole 2.5 family is retired
`models.list()` returns models this key **cannot call**. `gemini-2.5-flash` appeared in
the listing and then returned `404 "no longer available to new users. Please update
your code to use models/gemini-3.6-flash"` on every request. Same for
`gemini-2.5-flash-lite` and `gemini-2.5-pro`.

So `--validate-models` now makes one real 256-token call per candidate. Availability is
only knowable by calling.

Verified callable, with measured single-call latency:

| model | s/call | thinking tokens | note |
|---|---|---|---|
| `gemini-flash-lite-latest` | 0.9 | 0 | **excluded: moving alias** |
| `gemini-3.1-flash-lite` | 1.4 | 0 | honours `thinking_budget=0` |
| `gemini-3-flash-preview` | 1.5 | 25 | preview |
| `gemini-3.5-flash` | 1.6 | 77 | **reference** |
| `gemini-3.6-flash` | 2.4 | 96 | newest GA flash |
| `gemini-pro-latest` | 38.1 | 116 | ~50 h for the full run |
| `gemini-3.5-flash-lite` | 37.8 | 0 | 1.0 s on an earlier call -- huge variance |

Two cautions that follow:

* **Single-call latency is unreliable.** `gemini-3.5-flash-lite` measured 1.0 s and
  37.8 s minutes apart. Model selection needs several samples, not one.
* **Availability is transient.** `gemini-3.7-flash` returned OK, then `503 UNAVAILABLE`
  ten minutes later. 503 is retryable and now handled as such; a 404 is not.

`thinking_budget=0` is honoured only by the *lite* models. The full flash models still
spend 25-96 tokens thinking, billed as output and counted against
`max_output_tokens` -- so a tight budget silently returns empty text.

### 2026-08-18 — retry only transient errors
`gemini_llm` retried every exception with exponential backoff, so a retired model's
clean 0.7 s 404 became a silent ~14 s hang per call and the actual message never
surfaced. It now retries only 408/409/429/5xx and transport faults, and raises
permanent errors immediately with the API's own text.

### 2026-08-18 — `gemini-2.0-flash` is retired; model choice moved to the probe
Both configs named `gemini-2.0-flash`, the pilot's model. Querying the API's own model
list shows it is **no longer served** — the configs pointed at a model that does not
exist, and stage 02 would have failed on its first call.

Set provisionally to `gemini-2.5-flash` (GA). The stage-02 probe compares it against
`gemini-3.7-flash` and `gemini-3.1-flash-lite` on identical inputs, and the winner is
pinned before the `prereg-v1` tag.

Two rules adopted from this:

* **Never a moving alias.** `gemini-flash-latest` and `gemini-flash-lite-latest` exist
  and are tempting, but an alias means the dataset was generated by an unknown model
  version and the study is not reproducible. Always an explicit version.
* **Never a hardcoded model list.** `--list-models` queries the API. Availability and
  naming change faster than any list in a repo, and this is exactly how a run fails
  30,000 calls in.

Also raised `generation.max_output_tokens` from 512 to 4096: Option B returns 25
captions in one response, and 512 would have truncated it — which the probe would have
reported as a batching quality problem rather than a config error.

---

## Post-tag (real deviations)

*None yet.*

---

## 2026-08-19 — generation prompt v5, and the frozen generation config

Still pre-tag: `prereg-v1` has not been applied, so `configs/prereg.lock.yaml` remains
editable. Everything below is a **pre-tag decision**, recorded here so the reasoning
survives. Rationale and measurements are in `docs/lab-notebook.md`.

### What changed in the prompt, and why

| # | change | reason |
|---|---|---|
| 1 | `REGISTERS` rewritten from *content to lead with* → *manner of writing* | the old table instructed the crutch words it produced: "alone" 36% of sad, "soft" 41% of romantic |
| 2 | five named forbidden shortcuts with shown BAD examples | solitude · pace · contact · light/weather/time · posture/gaze/grip. A flat "do not invent" did not hold |
| 3 | five legitimate devices, incl. stance-toward-the-scene | prohibition alone flattened sad/romantic/tense into neutral restatement |
| 4 | `USE THE IMAGE FOR MOOD` rewritten | it granted "light, weather, posture, expression" while another section forbade adding them; the model resolved the contradiction by adding light |
| 5 | BAD-example parentheticals critique the sentence, never the photo | they asserted facts ("two people are visibly setting it up") about images not attached to 8,090 of 8,091 calls |
| 6 | every GOOD example adds no noun, adjective or intent absent from its caption | six of them violated the prompt's own faithfulness rule |
| 7 | `NEVER PAD` with eight banned filler phrases | the 8-word floor was met with "they are present", "in this portrait" |
| 8 | dignity rule on `humorous` | ~40,000 deadpan captions about photographed people, many children, with nothing forbidding mockery of appearance |
| 9 | self-revision block **deleted** | inert: `thinking_budget: 0` plus a constrained JSON schema leaves no scratchpad and no way to revise an emitted cell. It occupied the highest-attention position |
| 10 | `strain` field added to the response schema | see below |

### The `strain` field — RECORDED, NOT ACTED ON

Every register now returns `{"text": ..., "strain": 0|1|2}`: 0 the register fits
naturally, 1 strained, 2 no honest reading of this register exists for this caption.

**It is recorded only.** It was proposed as a way to identify invented cells and it
fails at that (defect rate 1.6% / 2.6% / 2.4% by strain level — flat). It is retained
because it measures register *difficulty*, which is a reportable result. **Any decision
to exclude strain-2 cells from training is a pre-registration decision and has not been
made.** `GenerationRecord.strain` is absent on records written before 2026-08-19, so a
missing key means *unknown* and must never be read as 0.

### Anchor correction

`anchors.lexical_shortcut` recorded **0.539**. That figure and the ones now quoted come
from different estimators and must not be compared. Under one estimator — top-25
keywords per register, 5-fold cross-validated by `image_id`, averaged over 3 seeds,
ties credited fractionally — the measurements are:

| prompt | keyword-rule accuracy | chance |
|---|---|---|
| v1 (original) | 0.639 | 0.200 |
| v3 | 0.431 | 0.200 |
| v4 | 0.468 | 0.200 |
| **v5 (final)** | **0.395** | 0.200 |
| Sonnet reference | 0.347 | 0.200 |

The lock file now carries 0.395 with the estimator named. Under the original prompt,
two-thirds of the primary metric was obtainable by keyword spotting; under v5 it is
under half.

### Pre-registered prediction that the data contradicts

`anchors.expected_hardest_registers: [tense, humorous]`. Measured difficulty ordering is
**romantic (1.05) > humorous (0.90) > sad (0.89) > tense (0.85) > joyful (0.15)**. Tense
is not among the hardest. **The prediction is left as registered and the disagreement is
reported.** Do not retrofit the prediction to the measurement.

### Operational config changes (not pre-registered)

* `scripts/run_stage02.py` gains `--audit-offset` and `--out`, so a second audit lands on
  images the first never saw and does not pool with the main corpus. The manifest now
  records `image_ids`, `audit_offset` and `out_path`.
* `scripts/recover_batch.py` added. A submitted batch job bills whether or not the
  process is watching; two runs lost their connection and one succeeded unclaimed. This
  attaches to a job by name, matches responses positionally against the manifest's
  `image_ids`, refuses to write on a count mismatch, and retries transport faults.
* `data/generated/captions_raw.jsonl` renamed to `captions_audit_v1.jsonl`. It held 250
  records generated under the v1 prompt; leaving it at the production store path would
  have made `completed_keys` skip those 50 images, silently baking the broken prompt into
  the corpus. **The production store must be empty before the full run.**
* `.gitignore` now tracks `captions_audit_*.jsonl` and `captions_reference_*.jsonl`.
  Generation runs at temperature 0.9 against a hosted model, so a caption set cannot be
  regenerated byte-identically: these are primary evidence, not build artifacts.

### Cost

The `strain` field raises output tokens from ~32k to ~50k per 50 images. Full-corpus
projection moves from ~$7.00 to **~$10.50** at the batch rate. Measured per-run costs are
in each `runs/*/stats.json`.
