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
| v1 (original) | 0.641 | 0.200 |
| v3 | 0.431 | 0.200 |
| v4 | 0.466 | 0.200 |
| **v5 (final)** | **0.394** | 0.200 |
| Sonnet reference | 0.342 | 0.200 |

The lock file carries **0.394** with the estimator named, and the estimator is now
implemented in `src/emocap/eval/anchors.py` with `ESTIMATOR_ID` required to match the
lock's `lexical_shortcut_estimator`. These figures are ~0.005 off the ones first
recorded, because the ad-hoc script had no deterministic tie-break when selecting
keywords; `anchors.py` sorts by `(-score, word)`. The committed code is authoritative
and `scripts/audit_captions.py` recomputes every figure in this table. Under the original prompt,
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

---

## 2026-08-19 — three images excluded by generator safety refusal

Three test-split images returned **zero output tokens** on every attempt, in the batch run
and again at preflight. All three show young children in or near water, minimally clothed:

| image | content |
|---|---|
| `2762301555_48a0d0aa24.jpg` | young child wrapped in a towel at a pool |
| `3143982558_9e2d44c155.jpg` | baby on a sofa with an adult |
| `3564312955_716e86c48b.jpg` | young child in a swimsuit in shallow water |

The generator's child-safety filtering declines them. That is the filter working
correctly, and **no attempt was made to work around it** — not by re-encoding the image,
not by rewording the prompt. They are recorded in
`data/generated/excluded_images.json` and skipped on subsequent runs.

### Why this is a disclosed limitation, not attrition

The exclusion is **not random**. It correlates with subject matter — children, water,
swimwear — so the corpus is missing a systematically-defined slice rather than a random
sample. At 3/1000 on the test split, expect roughly 20–25 images across the full 8,091.
Small in magnitude, but a content-correlated gap belongs in the write-up's limitations
rather than being averaged away silently.

This is a **data-collection** exclusion and is deliberately kept separate from §7's run
exclusion criteria, which govern training runs and remain unchanged.

### Operational consequence, fixed

Without a register such an image is never "done", so it is retried on every run — and
because `run_stage02.py` preflights on the first pending image, one refused image aborted
an otherwise healthy run before it submitted anything. The runner now skips registered
exclusions, and a preflight refusal (zero output tokens) records the image and exits
asking for a re-run rather than reporting a config error.

### A duplicate-write bug this surfaced

Investigating the gap found a latent bug in `run_batch_generation`. An image is selected
as pending when **any** of its five `caption_idx` keys are missing, and the API cannot be
asked for a single index — so a retry regenerates all 25 captions. The write loop appended
all five records unconditionally, which would have added four duplicate
`(image_id, caption_idx)` keys for every partially-returned image.

It did not fire here (the three images had no records at all), but the train split is 6x
larger and partial returns are likely. The loop now skips records already on disk and
counts them as `records_already_present`. Regression test in `tests/test_batch.py`.

### Cost estimate was understating by ~40%

`run_stage02.py` estimated cost from hardcoded 2,152 input / 672 output tokens per image,
constants predating both the prompt rewrite and the `strain` field. Measured on the v5
audit: 4,472 / 1,019. The old constants would have quoted $0.77 for the test split while
billing $1.32, and $6.25 for the corpus against ~$10.70. Now uses measured rates.

### Manifests record the prompt

The prompt has been through five measured versions; a manifest recording the model but
not the prompt cannot answer "which prompt produced this caption". `run_stage02.py` now
hashes the rendered batch prompt into `config.prompt_sha256` (v5 = `72133efaffddefa3`).

---

## 2026-08-19 — final exclusion count: 15 images, all photographs of children

The corpus is complete: **8,076 of 8,091 images, 201,900 of 201,900 cells (100%)**.
Fifteen images are excluded, every one confirmed by a live call returning **zero output
tokens** — the generator declining, not a transport failure or a parse error.

The content pattern is unambiguous. All fifteen are photographs of young children, and
most involve minimal clothing or bathing contexts. From their human captions:
*"a little girl in only socks and a necklace"*, *"a group of mostly nude children"*,
*"a girl in a red polka dot bikini"*, *"a little girl in a red swimsuit"*, *"a man is
lifting a little girl above his head"*, *"a child sleeping with a pacifier"*.

**No attempt was made to circumvent this**, at any point — not by re-encoding images, not
by rewording the prompt, not by trying a different model. Each refusal is registered in
`data/generated/excluded_images.json` and skipped on subsequent runs.

### For the write-up

The gap is **0.19% (15/8,091) and content-correlated, not random**. The corpus therefore
under-represents photographs of children in bathing or swimwear contexts relative to
Flickr8k. That belongs in the limitations: a reader should know the exclusion has a
subject-matter signature rather than assuming attrition was arbitrary.

This is a **data-collection** exclusion and remains deliberately separate from §7's run
exclusion criteria, which govern training runs and are unchanged.

### Tooling change this forced

`run_stage02.py`'s preflight tested only `pending[0]`, so each refused image consumed one
whole invocation to register — twelve images would have needed twelve runs. The preflight
now walks up to 25 candidates, registering refusals as it goes and proceeding on the first
success. If every candidate is refused it reports that and exits cleanly rather than
claiming a config error.

---

## 2026-08-20 — three decisions taken before the tag

### `strain` is recorded, never filtered on

The generator self-reports 0/1/2 per caption cell for register fit. **Decision: record, do
not exclude.** Two reasons, the second decisive.

The flag fails at what filtering assumes — defect rate by strain level is **1.6% / 2.6% /
2.4%**, flat, so a cell marked "no honest reading exists" is no likelier to contain a
grounding defect than one marked "natural".

And excluding would wreck class balance:

| register | cells | strain-2 | % lost |
|---|---|---|---|
| joyful | 40,380 | 132 | 0.3% |
| tense | 40,380 | 1,073 | 2.7% |
| sad | 40,380 | 2,006 | 5.0% |
| humorous | 40,380 | 3,223 | 8.0% |
| **romantic** | 40,380 | **6,942** | **17.2%** |

The corpus is perfectly even at 40,380 per register. Excluding leaves 33,438–40,248 — a
**16.9% imbalance**, and `romantic` loses **52× as much data as `joyful`**. Any subsequent
finding that `romantic` is harder would be inseparable from it having had less data. A
self-inflicted confound, so declined.

Recorded in the lock as `strain_used_as_exclusion: false` and named in §7.

### `romantic`'s definition is unchanged

Considered redefining it — it is the hardest register by the generator's strain report (1.11)
and the independent human judge singled it out. **Declined.** The corpus was generated under
the current definition, so changing it would leave the data not matching the registration
unless ~$10.70 were spent regenerating. Renaming the label was also declined: it would
propagate through `emotion_id` everywhere for no gain in the data.

Note as a factual consequence, not a re-litigation: §4 requires the confusion matrix be
reported always, so `romantic`'s per-register recall will appear in it regardless of whether
the difficulty is discussed in prose.

### Training runs as Kaggle notebooks

Confirmed: one notebook per pipeline stage, executed as **Save & Run All**, with everything
produced during a run preserved for export. No stage may depend on a session surviving, so no
checkpointing is required — which is why the corpus generation was built append-only in the
first place. Outputs must be written where Kaggle actually exports them.

This makes `notebooks/` a real directory rather than the aspirational one the README has
described since the start.

---

## 2026-08-21 — the pre-tag rewrite: what changed between the draft and `prereg-v1`

Every item here was decided **before** the tag and is logged so the change is visible rather
than silent. After the tag these would require `prereg-v2`.

### The question changed from architecture to data

**Was:** *"Where in a decoder should the emotion signal be injected, and does the answer
depend on the decoder family?"*
**Is:** what emotion-conditioning accuracy measures when the training data is LLM-synthesised.

Two external reviews independently judged the conditioning-placement question well covered
and not a 2026 contribution on its own. The decoder is now a fixed control. The superseded
text is preserved in collapsed blocks in §1–§3 rather than deleted.

**Consequence:** five conditions × two tracks × three seeds = 30 runs becomes three data arms
× five folds + three negative controls = **18 runs**. `A_lstm` is dropped outright — 8,076
captions cannot train a decoder from scratch.

### Evaluation regime: seed spread → 5-fold CV over each arm

**Was:** variance reported as the spread across three seeds on a single 1,000-cell test split.
**Is:** 5-fold cross-validation over each arm, folds split by `image_id`, with fold-to-fold
spread as the reported variance.

**Why:** the MDE on a single 1,000-cell split is **6.5 points** against **2.3** under CV. A
5-point criterion had been registered against the former, meaning **it could not have been
falsified**. Computing the MDE before committing the criterion is what caught it.

Seeds remain live in the lock — they still fix initialisation and data order. Only what is
*reported* as variance changed. The rule that a gap smaller than the observed spread is
reported as null **regardless of its p-value** is unchanged.

### The ceiling gate: a level → a detectability criterion

**Was:** `ceiling_min: 0.85`.
**Is:** the study proceeds if the floor-to-ceiling range admits the smallest effect of
interest (+7 points) at the design's MDE.

Three reasons the registered value fails as written: our corpus reaches ~0.77 so it says
*halt*; **human-written text reaches only 0.726** on the identical instrument, so it would
halt a study on data matching human performance; and the ceiling is sample-size dependent
(0.612 at 1,225 cells → 0.773 at 201,900), so there is no single number to threshold.

Gating on the **margin over the anchor** was considered and **rejected by measurement** — the
margin moves +0.198 across the n-range against accuracy's +0.090, amplifying the confound it
was meant to control.

### A human-legibility gate is ADDED

New, with no predecessor: a blind caption-only register guess by a human, reported with its
neutral rate beside the same corpus's keyword anchor. **The corpus must be more legible to a
reader than to a bag of 25 keywords per register.**

Added because the first corpus passed floor, artifact-ablation, anchor and fold-variance
checks while a human read its register at **0.310** against 0.200 chance — and a keyword bag
scored 0.441, *above* the human. Four automated gates agreed with each other and disagreed
with a reader. The current corpus scores **0.592 human vs 0.510 anchor**.

### Generator: prompt v10 on gemini-3.7-flash

The corpus is regenerated from scratch. The v5 flash-lite corpus (201,900 captions) is
**discarded, not repaired** — selective re-runs cannot fix a register that is absent.
Decision evidence: `RESEARCH-LOG.md` Part 18.

### A corpus-construction rule is ADDED: the vocabulary cap

No content word may exceed its register's document-frequency cap when it is also ≥2× more
common there than in any other register. Caps are per-register, fixed in `configs/data.yaml`
**before** the corpus is built.

Added because prompt v6 turned technique into a template — `bright` reached 34% of joyful
captions against 3% elsewhere, and masking such words dropped the lexical anchor
0.638 → 0.504. A decoder trained on that learns `joyful → bright`. Tuning the caps after
seeing the anchor would be fitting the instrument to the result, which is why they are frozen
pre-tag.

### The retired v1 pilot corpus is ADDED as two arms

**Was:** the v1 pilot corpus was archived as provenance only, marked *"do not train on this"*.
**Is:** two arms — **V1-paired5** (5/image) and **V1-unpaired** (1/image) — bringing the
design to **six arms and 36 runs**.

Two reasons, decided 2026-08-21 before the tag:

1. **It separates the pilot's code faults from its data.** The post-mortem lists several
   causes at once, including a beam-search KV-cache aliasing bug in the decoder. That bug is
   fixed in this version, so training the pilot's *data* through the corrected pipeline
   answers a question the post-mortem could not: did the pilot fail because of its code or
   because of its data? If the v1 arms match the v10 arms, the corpus was discarded on an
   incomplete diagnosis — and that is worth reporting.

2. **It is the extreme point of the study's central axis.** Keyword anchors at matched n:
   **v1 0.718**, ours 0.507, human 0.450. A three-point gradient in lexical stereotypy makes
   **P6** falsifiable in a way no two-corpus comparison could be.

**It is a contrast arm, not a rehabilitation.** The v1 captions are known to be worse — 24.2%
name their own emotion, 42.4% fail the deterministic validator, and the neutral source was a
VLM that hallucinated, so grounding is broken upstream of the rewrite. Presenting it as a fair
competitor would be dishonest; presenting it as the high-fingerprint end of a gradient is what
it is.

**Availability verified before registering:** 8,048 of 8,091 pilot images carry all five
registers, are present in Flickr8k, survive the safety exclusions, and have their image file
on disk. No duplicate `(image, emotion)` cells.

### The Personality-Captions trait mapping is FROZEN

23 traits are available; the strict 1:1 mapping uses 5 of them; `Breezy (Relaxed, Informal)` is dropped because only one
usable row survives the image download. The mapping is in `configs/prereg.lock.yaml`
(`study.personality_map`) and fixed **before** training, because an earlier mapping choice
flipped the sign of a human-vs-synthetic comparison. §2 requires its sensitivity be reported.

**Availability verified:** 19,990 Personality-Captions rows have their YFCC image on disk
across 19,987 distinct images — ample for an 8,076-caption arm balanced at 878 per register (4,390 = 878 x 5, exact),
with `romantic` the scarcest at 2,608 available.

### The MDE was recomputed for six arms — and three criteria were unreachable

Caught during the pre-tag feasibility check, **before** the tag.

The MDE of **2.3 points** quoted in the first draft of this rewrite was computed for the
*three*-arm design and assumed **zero seed noise**. Neither holds:

- Six arms give 15 pairwise combinations against the previous design's 3, and the stored
  MDE used `n_comparisons=3`.
- Seed noise is a real variance component that the 2.3 figure omitted entirely.

Recomputed for the limiting comparison (1-caption-per-image arms, ~8,048 cells):

| seed SD | 0.000 | 0.005 | 0.010 | 0.020 | 0.030 |
|---|---|---|---|---|---|
| MDE, 6 confirmatory comparisons | 2.7 | 2.9 | 3.5 | **5.2** | 7.1 |
| MDE, all 15 corrected | 2.9 | 3.1 | 3.8 | **5.6** | 7.7 |

**P2, P3b and P6 were registered at ≥3 points against an MDE of 5.2 — unreachable.** Raised
to **≥6 points**, which clears the MDE at seed SD 0.02.

**Three structural changes followed:**

1. **Confirmatory/exploratory split.** Six comparisons carry hypotheses and Holm correction;
   the other nine are exploratory, uncorrected and labelled. Correcting over all 15 costs 0.4
   points of MDE for comparisons that test no prediction.
2. **An underpowered-reporting rule is ADDED.** A comparison whose observed fold spread
   implies an MDE above its criterion is reported as **underpowered, not as a null**. This is
   the specific failure that damaged an earlier pre-registered study in this line of work:
   numbers were registered that the design could not reach, and the resulting nulls were
   reported as findings.
3. **P1 carries a stated caveat.** Its expected effect is 4.3 points against an MDE of 5.2 at
   seed SD 0.02, so a null on P1 is underpowered unless the observed spread brings the MDE
   below 4.3. Registered as such rather than discovered afterwards.

The MDE is registered **as a curve over seed SD** (`study.mde_curve_by_seed_sd`) rather than
as a single number, because seed noise cannot be known before models exist. The only
comparable spread measured so far — the register classifier across three matched subsamples,
SD 0.0076 — suggests seed SD ≈ 0.01, making 0.02 the conservative case.

---

## 2026-08-21 (later) — an external audit found seven blockers; all fixed pre-tag

An independent audit recomputed every registered figure against this repository's own
estimators rather than reading them off the documents. It found seven blockers, and **all
seven were real**. Recorded here because the errors are instructive, not because they are
flattering.

### The audit's own lesson

The internal feasibility check that ran an hour earlier passed 23 assertions — because it was
written against the numbers already registered rather than against
`emocap.eval.power` and `emocap.eval.anchors`. **It verified internal consistency and called
that validation.** Same failure class as the retracted anchors: checking that nothing
complained instead of checking that something is true.

### B1 — a corpus described in the present tense that does not exist

The arm table listed S-paired25 at 201,900 captions as though on disk; the real figure was 240
of 8,076 images. Registering before collecting is normal; describing uncollected data as
existing is not. §2 now says so explicitly, and the gates note which corpus produced them.

### B2 — the MDE was not reproducible, and the criteria did not clear it

The commit that "recomputed the MDE" changed three documentation files and no code.
`scripts/compute_mde.py` still used `n_comparisons=3`, and the curve had nothing behind it. The
figures were also computed with invented inputs (p 0.60, ICC 0.070) rather than the repo's
measured ones (**p 0.72, ICC 0.0722**).

Recomputed on the registered design — 4 confirmatory comparisons, 4,390-cell limiting arm, 3
runs averaged — the MDE is **3.2 / 3.5 / 4.2 / 6.3 / 8.8** points at seed SD 0 / .005 / .01 /
.02 / .03. **The ≥6-point criteria did not clear 6.3.** Criteria raised to **≥7**, and
`compute_mde.py` now emits the registered curve so it is reproducible by command.

### B3 — the human anchor belonged to a different corpus

**0.342 is the Sonnet reference set's anchor**, recorded as such in `lab-notebook.md:274` and
in this file's own table. It was attributed to Personality-Captions in the arm table, P6, §4,
the README and the lock. The strict-mapping human anchor is **0.450**. The gradient still runs
0.718 > 0.507 > 0.450, so P6 survives, but every anchor now carries its n and is marked
provisional (see S4).

### B4/B5 — the frozen trait mapping was unmeasured and flipped the sign

The lock froze the **grouped** mapping. `lab-notebook.md` records that the sign of the
stereotypy comparison flips between mappings and concludes *"the strict 1:1 mapping is the more
defensible of the two"*. Worse, grouped had never been run through any instrument — every human
number in the repo came from `compare_human_ceiling.py`, which uses strict, and the two
disagreed even internally (`Sentimental` in `sad` versus `romantic`).

**Switched to strict.** Cost: strict yields 4,486 cells, so all three 1-caption-per-image arms
drop from 8,076 to **4,390** (878 × 5, exact). That raises the MDE by 0.4 points at seed SD
0.02 — paid to avoid registering an instrument no measurement had used, chosen because it
supplied the arm size we wanted.

### B6 — the lock pinned the retired generator, and v10 existed nowhere

`generator_model: gemini-3.1-flash-lite` sat in both configs while a newly added
`generation_model: gemini-3.7-flash` sat 98 lines below it — two keys one letter apart naming
different models, the frozen one naming the v5 generator. And `prompt_version: v10` was a
config string with no definition: `prompt.py` documented v6, and the producing run's manifest
recorded v6.

Fixed: one `generator_model`, `PROMPT_VERSION = "v10"` in `emocap.data.prompt` with its version
history, and two tests that fail if either drifts again.

### B7 — the arms cannot share images, and the document said three times that they did

H-unpaired is Personality-Captions over **YFCC100M**; every other arm is **Flickr8k**. The
image sets do not intersect, so P1 and P2 varied provenance, image distribution and
groundedness at once — CLIPScore **0.578 human against 0.792 ours**, 2.64 SD apart — while §2
claimed they isolated provenance.

**The confirmatory set is now defined by matched images:** S vs V1 at 1/image and 5/image, and
the two structure comparisons within our own corpus. **P1 and P2 are reference comparisons**,
reported with the confound stated and with the rival explanation registered from
`clipscore_matched.py`: *"the human corpus's advantage is bought by being allowed to ignore the
image, not by being human-written."*

Two alternatives were considered and rejected. Generating our captions on the YFCC images is
impossible without inventing neutral captions with a VLM — the v1 pilot's fatal design.
A groundedness-matched subsample survives only 18.6% distribution overlap, ~816 captions, so it
is registered as **directional secondary evidence** with its MDE stated, not as an arm.

**A side effect worth naming: the v1 arm is now the study's cleanest confirmatory test** —
same images, same structure, same pipeline, a 21-point anchor gap.

### Softer findings, all applied

**S3** the legibility gate now carries its CI (0.592 ±0.049, ~1.7 SE) and the non-equivalence
between a trained anchor and a zero-shot reader, with the human sample to widen to ≥300 before
the gate is applied. **S4** every anchor carries its n and the gradient is marked provisional,
since this estimator falls ~11 points from 4.5k to 200k cells. **S5** 57 duplicated README
lines removed. **M1–M12** stale V0/track keys retired, ICC corrected to 0.0722 (design effects
1.00 / 1.29 / 2.73), the 1,225-cell ceiling corrected to 0.620, the smallest effect of interest
brought to +7 to match the criteria, 23-not-24 traits, 19,990 rows, and the superseded
thresholds in tracked run records annotated rather than rewritten.

### A second audit found ten blockers; all fixed pre-tag

Recomputed against the repo's own estimators again. The most serious were not the stale
numbers but two faults the first pass created or missed:

**The generation pipeline was pinned to the retired generator.** `configs/data.yaml`
`generation.model` said `gemini-3.1-flash-lite` — the key `run_stage02.py` and
`run_vertex.py` actually read. The earlier "one generator key" fix corrected
`data.generator_model`, which nothing in the generation path reads. **The corpus would have
been generated with the wrong model.** A test now asserts every key whose name contains
"model" with a Gemini-shaped value agrees across every config.

**P1 and P6 registered opposite signs for the same comparison.** P1 predicts H beats S; P6's
mechanism predicts accuracy ranks with stereotypy, and ours is more stereotyped than the human
corpus, so it predicts the reverse. Separated by scope: P6 covers only the matched-image pair,
P1 stays a cross-dataset reference. The conflict is named in §3 rather than quietly resolved.

**The gate runner was dead code.** `check_gates.py` crashed with `KeyError: 'ceiling_min'`
because the gates were rewritten around it, and the two replacement gates had no
implementation at all — so the registration committed to halt conditions that could not be
evaluated. 256 tests passed because none touched the script. It now evaluates the
detectability range, `scripts/score_guess.py` implements the human-legibility gate, and a
smoke test asserts the lock keys both scripts read still exist.

**"Correcting over all 15 buys nothing" was wrong in direction and hid a criterion break.**
More comparisons cannot lower an MDE. Recomputed: 4 → 6.3, 6 → 6.6, **15 → 7.1** — above the
7.0 criterion. The 5.6 figure was stale from the superseded 8,048-cell curve. The paragraph
now states the real cost.

Also fixed: the confirmatory family read four/six/four in three places (now four everywhere);
the README still listed 8,076-caption arms the strict mapping makes impossible; the
detectability gate quoted the optimistic 5-run MDE row the document forbids; smallest effect
of interest was +7 in three places and +10 in two; `data-attribution.md` said three exclusions
where disk holds fifteen, and still named the v5 generator and the dropped two-track design.

## 2026-08-22 — prereg-v2: arm sizes measured, and three unregistered choices closed

**These are POST-TAG deviations against `prereg-v1`** (commit `3a61cde`, tagged 2026-08-21),
not pre-tag edits, and they are logged as such because the tag is already public. The lock's
own procedure is followed: logged here, bumped to `prereg-v2`, and carried into the write-up.

**No outcome existed when any of these were decided.** Not one model has been trained, no arm
has been evaluated, and the primary metric has never been computed on any arm. Every choice
below was therefore made blind to the results it could affect — which is the property that
makes a post-tag specification defensible, and it is stated here so a reader can check it
against the commit history rather than take it on trust. Two of the four are gaps the
registration left open rather than commitments it made; one is forced by the data; one adds a
config the registration claimed already existed.

**Arm sizes are now realised, not arithmetic.** `scripts/build_arms.py` materialises all six
arms to `data/arms/*.jsonl` with per-arm sha256 in `data/arms/manifest.json`. Two drafted
figures were wrong because they were computed on 8,076 images before the corpus existed: one
image never returned a usable batch row, so the shared Flickr8k universe is **8,075**.
`paired25_arm_cells` 201,900 → **201,875**; `paired5_arm_images` 8,048 → **8,075**. The
1/image arms are unchanged at 4,390 — they are capped by the human corpus, not by Flickr8k.
The registered MDE curve is therefore unchanged: it is computed at n=4,390, which is exactly
what the arms deliver, and `scripts/compute_mde.py` reproduces {0.032, 0.035, 0.042, 0.063,
0.088} with the criterion at 0.07 still clearing the 0.063 governing row.

**Arm selection was a shape, not a rule.** "5 cells per image" is satisfied by at least three
different experiments. Registered: the 5/image arms take one source caption and all five of
its registers — making them a strict subset of the 25/image arm, so P3 is a pure count
comparison — and the V1 arms take the same images, same source caption and same register as
their S counterparts, so the provenance comparisons differ in generator and nothing else.
Everything is chosen by hashing the `image_id`, so no result depends on RNG state or
iteration order.

**The classifier's provenance was open, and it changes every comparison.** "Trained on the
training split only" did not say *whose* captions. One classifier trained on our corpus would
know our generator's fingerprint and flatter S over V1 and H by construction; one classifier
per arm puts the arms on different scales and makes the differences unsubtractable. Registered:
a single instrument trained on a provenance-balanced 13,170-cell pool (4,390 each from
S/V1/H-unpaired), frozen and hashed before any arm trains.

**Evaluation subsampling considered and rejected.** A fixed per-fold eval sample would have
been a new unregistered parameter, and it was only tempting because decoding was costed on a
laptop. Training moved to Kaggle GPU, so every held-out cell of every fold is decoded.

**Training hyperparameters are now frozen in `configs/model.yaml`** and hashed with the lock —
§2 claimed they were "held constant by config, not by discipline" while no such config
existed. Epochs are fixed rather than steps, so S-paired25 gets ~46× the optimizer steps of
S-unpaired; that is what "more data" means, but it makes P3's effect data volume *and*
compute, which is now stated rather than implied. Truncation at `max_seq_len: 48` was measured
on all six arms before registering it: worst case 0.0046 (H-unpaired), against the 0.02 gate.

## 2026-08-22 — the V1 arms were built from the wrong corpus, and caught before training

`scripts/build_arms.py` originally sourced the V1 arms from
`data/generated/captions_raw.jsonl`. That file is **not** the v1 pilot. It is the v2
rebuild's first generation attempt — prompt v1 on `gemini-3.1-flash-lite`, 8,076 images —
and it sat in the same directory under a plausible name. The pilot is
`archive/v1-pilot/data/v1_emotion_captions.csv.gz`, 8,091 images, `gemini-2.0-flash`.

Measured with the registered estimator at 4,390 cells:

| corpus | keyword@25 |
|---|---|
| archived v1 pilot (correct) | **0.7064** |
| `captions_raw.jsonl` (what was used) | **0.4357** |

**What this would have done.** The v1 arm exists to be the high-stereotypy end of the
study's central axis; §2 registers the gradient v1 0.718 > ours 0.507 > human 0.450, and P6
is falsifiable only because that spread exists. Under the wrong corpus the ordering read
S > H > V1 — the contrast arm was *less* stereotyped than human text, inverting the axis.
Every automated gate would have passed. It was found only because the three-corpus ordering
was recomputed from the arm files before training, and disagreed with the registration.

**How it was found matters more than the fix.** The check that caught it was run for an
unrelated reason: to test whether the stereotypy ordering was instrument-dependent
(keyword rule vs TF-IDF). The orderings agreed with each other and disagreed with the
registered values, which pointed at the data rather than the estimator. Recomputing a
registered number from its own artifacts, rather than trusting the document, is what
surfaced it.

**Fixed.** V1 arms now come from the archived pilot, and the safety exclusions are applied
during arm construction (they were not before). Realised counts, recomputed at 4,390 cells:

    keyword@25   V1 0.7329  >  S 0.6068  >  H 0.4512      registered: 0.718 / 0.507 / 0.450
    tfidf        V1 0.8943  >  S 0.8118  >  H 0.6123      same ordering, both instruments

V1 and H reproduce their registered values; **ours measures 0.607 against a registered
0.507**. §6 already governs this — the gradient's *ordering* is the registered prediction and
its *values* are not — but the direction is worth stating: the v10 corpus is more lexically
stereotyped than the figure drafted for it, and the margin the study cares about is
correspondingly smaller.

**A consequence for the S-vs-V1 comparisons, now stated rather than implied.** The pilot
rewrote ONE Moondream neutral caption per image; our corpus rewrites FIVE human Flickr8k
captions. There is no shared `caption_idx`, so matching stops at the image and those
comparisons vary generator *and* source text on the same photographs.

**Arm sizes after the rebuild.** Images 8,047 — the drafted 8,048 was correct and came from
the pilot; one image never returned a usable batch row. `paired25_arm_cells` 201,175;
`paired5` 40,235; all 1/image arms 4,390. The MDE is unaffected: it is computed at n=4,390.

**The frozen classifier was retrained**, because its provenance-balanced pool had contained
the wrong V1 cells. New instrument `sha256 572eaa80…`, 5-fold CV **0.8279** (was 0.7595 on
the bad pool), TF-IDF baseline 0.7619.

**The artifact ablation now runs on the transformer.** The version run during training used
TF-IDF, whose tokenizer discards punctuation anyway, so it moved by exactly 0.0000 and could
not have failed. On the frozen classifier: prediction flip rate **0.0308** (405/13,170) when
punctuation and case are stripped, in-sample accuracy gap +0.0222. The instrument is reading
register rather than formatting. Flips concentrate in `humorous` (114) and `tense` (98).

## 2026-08-23 — the visual-dependence probe had no implementation

§7 lists a failed visual-dependence probe as one of only three grounds for excluding a run,
and the gates table specifies it as "zero the features → captions change substantially".
Neither the training script nor the scoring script implemented it. `ClipCap.zeroed_visual()`
existed and was never called. **The first sixteen runs therefore cannot be checked against a
registered exclusion criterion**, and are being re-run.

**Fixed.** `train_arm.py` now decodes every held-out cell twice — once normally, once with
the image blanked — in the same process on the same model object, writing
`predictions_novis.jsonl` beside `predictions.jsonl`. Running both passes on one loaded model
is deliberate: a probe compared against a separately reloaded copy of the weights would be
testing the loading path as much as the model. The trainable parameters (mapper + LoRA, 29
tensors, 34.8 MB) are saved as `adapter.pt` so no future probe or re-decode requires a
retrain. `score_arm.py` reports the identical-caption rate, the accuracy on blanked captions,
and the drop, and writes `probe_review.txt` with forty side-by-side pairs.

**No threshold is registered, and none is being invented.** The prereg says "substantially"
without a number. Rather than pick one after seeing which runs would pass — the failure mode
this document exists to prevent — the scripts report the evidence and **the pass/fail call is
made by human reviewers on the caption pairs**. Decided 2026-08-23, before any probe output
existed. `score_arm.py` prints `PROBE MISSING` for runs that predate the change rather than
scoring them as if they had passed.

**Why this surfaced now.** `H_unpaired` decodes 0.58-0.64 unique captions per fold against
0.98-1.00 for the other arms, at 7.2-7.6 mean words, and produces text like "I love this
place!" and "This is a great place to visit" — close to image-independent. It is also the arm
with the highest accuracy of the three at 1 caption/image, so the one encouraging result
among them may come from a run the registered criterion would exclude. That is not decidable
without the probe, which is what made the gap worth an hour of GPU to close.

**Probe size: 500 held-out cells per run, not all of them.** Decided 2026-08-23, before any
probe output existed. Blanking every held-out cell doubles the sweep to ~30 GPU hours
against a 30 h weekly quota, leaving no margin for a re-run; on 500 cells the whole sweep
costs ~17 h. The probe exists so human reviewers can see whether captions change when the
image is removed, and 500 cells pins the identical-caption rate to about ±2 points — far
finer than that judgement requires. The subset is chosen by `sha1('probe-v1' + image_id +
emotion)`, so the same cells are probed on every re-run and in every arm rather than sampled
at runtime. `score_arm.py` measures the accuracy drop against the with-image accuracy **on
those same cells**, never against the full-set accuracy, so sampling noise is not reported
as an effect.

---

## 2026-08-24 — two additions decided after the tag: `S_paired_matched` and the frozen-LM baselines

**Neither is registered, and both must be labelled post-hoc wherever they are reported.**
They are recorded here rather than folded into the design because the whole value of a
preregistration is that a reader can tell which decisions preceded the data. Both were
decided on 2026-08-23, after the first arm results existed, and are implemented in code kept
deliberately apart from the registered path: `src/emocap/data/posthoc.py`,
`scripts/build_posthoc_arms.py`, `scripts/baseline_prior.py`, `notebooks/02f_posthoc.ipynb`,
and their own output directories.

`emocap.data.arms.ARMS` still names exactly the registered six, and
`configs/prereg.lock.yaml`'s `arms:` list is untouched. Adding a seventh entry there would
have made the lock disagree with the code and, worse, made a post-hoc arm indistinguishable
from a registered one to anyone reading this repository later.
`data/arms/manifest.json` — whose per-arm sha256 the lock quotes — is not written by the
post-hoc build; it writes `data/arms/posthoc_manifest.json` instead.

### `S_paired_matched`: pairing, separated from volume

As registered, P3 compares `S_paired5` against `S_unpaired` and moves three things at once:
9× the cells, roughly twice the images, **and** the paired structure. The prereg is candid
about this — it calls P3a "a sanity check on the extra data" — so as registered the claim is
about volume, not about pairing. That was not noticed before the tag, and it should have
been.

The new arm is 878 images × 5 registers = **4,390 cells**, exactly `S_unpaired`'s size.
Image count is the one thing that cannot also be held fixed, and that is arithmetic rather
than an oversight: 4,390 cells is either 4,390 images with one register each or 878 images
with five. Trading images for registers *is* what pairing means here, and the comparison is
stated that way rather than dressed up as a clean single-variable contrast.

**It is nested inside the registered arms, and the build refuses to write it otherwise.**
Its 878 images are the first 878 of the same `sha1('unpaired-v1' + image_id)` ranking
`S_unpaired` draws its 4,390 from; each contributes the source caption `sha1('cap-v1' +
image_id) % 5` already chose. So its images are a strict subset of `S_unpaired`'s, its cells
a strict subset of `S_paired5`'s, and every `S_unpaired` cell on a shared image reappears in
it. `scripts/build_posthoc_arms.py` asserts all three and exits rather than writing an arm it
cannot vouch for. Nothing new is sampled, which is what makes "the post-hoc arm drew easier
data" unavailable as an explanation of whatever it shows. Built: 4,390 cells, 878 images, 878
per register, folds [820, 950, 905, 790, 925], sha256 `1b012fdd77862bf0…`.

Extracting `shared_images()` out of `build_all_arms()` was required so the post-hoc arm
selects from the identical image universe instead of recomputing it. That extraction is a
no-op: `scripts/build_arms.py` was re-run and all six registered arms' sha256 are unchanged.

### The baselines: what the frozen LM supplies with no image

The study has no prior-art or ablation comparison, so "this arm is a good emotion-conditioned
captioner" has nothing to be good *relative to*. Every arm is measured against its keyword
anchor, which says what a lexical rule reaches with no model; nothing said what GPT-2 reaches
with no **image**. `scripts/baseline_prior.py` trains nothing — same frozen GPT-2, same frozen
`DecodeConfig`, same frozen classifier — in two modes:

- **`prior`**, the register word alone. This is the baseline named in
  `docs/remaining-work.md`, and it is **degenerate by construction**: beam search is
  deterministic and the prompt depends only on the register, so the entire held-out set
  decodes to **five captions**. Its accuracy is a draw over five outcomes, exactly the trap
  the blanked visual-dependence probe walked into when a negative control appeared to score
  0.58. The script generates once per register, reports `unique_captions`, and prints all
  five strings, which must be published beside its accuracy. On `S_unpaired`-f0 the five are
  first-person quotes — "I love you so much.", "We are in the middle of a war." — not photo
  captions at all. That is the honest floor, and it is a floor rather than a measurement.
- **`text`**, the register word plus the neutral Flickr8k source caption. Per-cell variation,
  still no image. This is the informative one: it says how much of an arm's accuracy is
  reachable from the source text with the photograph removed. A 24-cell smoke test produced
  14 unique captions and output that barely moves with the register.

**Prompt wording is a real degree of freedom** — a better prompt raises the floor — so both
templates were fixed before any baseline was scored, are recorded in every run manifest, and
must not be tuned against results. Output is cut at the first newline, which makes the
baseline more caption-like and therefore *stronger*; that is the safe direction for a floor,
so any comparison it loses, it loses on merit rather than on formatting. `S_paired25` gets
`prior` but not `text`: 40,235 cells × 5 folds is ~4 GPU hours to measure a frozen model that
never sees the image.

**Systems claims against these must be stated in margins, not raw accuracy.** The paper
argues accuracy overstates conditioning because most of it is keyword-reachable; leaning on
that same raw number to praise the model would contradict its own central finding.

---

## 2026-08-24 — the human evaluation: what the registration left open, and how it was fixed

`configs/prereg.lock.yaml` registers `human_eval` as **confirmatory**: 150 items, 3 raters,
tone-match and grounding on 1-5 scales, Krippendorff's alpha, blinded, with attention
checks, "run once, on final models only". It does not say which arms the items come from,
how the checks are built, or what a failed check means. Those three were decided on
2026-08-24, **before any rating existed**, and are recorded here because deciding them
afterwards is exactly the freedom a registration exists to remove.

**Items come from three arms, not six.** S_paired25, S_paired5 and V1_paired5, 50 each, 10
per register. The three 1-caption-per-image arms sit at their own keyword anchor — they
write essentially register-free text — so half the rating budget would have gone to captions
nobody disputes. The exclusion and its reason are recorded in `runs/human-eval/key.json` and
must be reported.

**No photograph appears twice in the whole task.** All three arms caption the same 8,047
Flickr8k images, so sampling them independently would show one scene under two systems and
hand the rater a side-by-side comparison — destroying the blind the design depends on.
Images are partitioned round-robin from a `sha1("humaneval-v1" + image_id)` ordering, and
the checks draw from a further reserved slice. Verified at build time: 162 items, 162
distinct photographs, 0 reused.

**Raters see the requested register, and that is not a leak.** Tone-match cannot be judged
without knowing which tone was asked for. What they never see is the arm, or whether an item
is a check — the page data carries only `id`, `text`, `asked` and the embedded image, and
that was verified by grepping the built file for every arm name and every key field.

**Attention checks, 12, additional to the registered 150** — so exactly 150 scored items
reach the analysis. Six *wrong-register*: a real caption shown under a register it was not
written for, and only ever a distant one (joyful shown as sad, tense as romantic), never a
defensible neighbour like joyful-vs-humorous. Six *wrong-image*: a real caption against an
unrelated photograph. **A check passes when the targeted scale is <= 2**, and a rater failing
more than half of theirs is **FLAGGED, not dropped** — `score_human_eval.py` reports their
numbers alongside everyone else's. Excluding a rater is a judgement to be made and written
down here, the same discipline the visual-dependence probe uses.

**Krippendorff's alpha is ordinal, not nominal.** On a 1-5 scale a 3-vs-4 disagreement is
smaller than a 1-vs-5 one; nominal alpha calls them identical and understates agreement on
rating data badly. The implementation lives in `src/emocap/eval/human_eval.py` rather than in
a script, and was checked against the reference `krippendorff` PyPI package to five decimal
places on four cases — perfect agreement, Krippendorff's own 12-unit example with missing
values (0.8049), independent raters (-0.0593), and raters agreeing within one point (0.6489).
Those values are pinned in `tests/test_human_eval.py` so the check survives without adding
the dependency. Alpha returns `None`, never a number, when it is undefined: a degenerate task
reported as 1.0 would read as maximal reliability.

**Alpha is reported before the arm means, and gates nothing.** The lock asks for it to be
reported. If it comes out near zero the per-arm means below it are three private impressions
averaged together, and that has to be visible above the numbers rather than in a footnote.

**Pairwise arm differences from this evaluation are uncorrected and labelled so.** The lock
names human_eval as a confirmatory *metric* and names no arm pairs for it, so Holm correction
— which is registered over exactly the four confirmatory comparisons — does not extend here.
The arms also share no photographs in this task by construction, so these are unpaired
differences between independent item sets.
