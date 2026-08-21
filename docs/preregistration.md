# Pre-registration — EmoCap v2

> ## Scope of this registration
>
> This registers a study about **training data**, not about decoder architecture. An earlier
> draft asked where in a decoder an emotion signal should be injected; that question is well
> covered and two external reviews were right that it is not a 2026 contribution on its own.
> The superseded text is kept in collapsed blocks below rather than deleted, because it is
> what was actually drafted and the change of direction is part of the record.
>
> Everything measured before this tag characterises the **corpus** — its lexical shortcut,
> its human legibility, its grounding. **No model comparison has been run.** The hypotheses
> below are therefore untested and the tag retains its force.

**Status:** complete. Tagged `prereg-v1`; the tag's commit timestamp is the record.
**Frozen settings:** [`configs/prereg.lock.yaml`](../configs/prereg.lock.yaml)
**Registration mechanism:** git tag. The tag's commit timestamp is the record. OSF
registration is deliberately deferred and can be added later without invalidating this.

---

## 1. Question

Emotion-conditioned captioning is normally evaluated by asking whether a classifier can
recover the requested emotion from the generated caption. **This study asks what that number
measures when the training data is LLM-synthesised.**

A synthetic corpus is produced by prompting one model, and a prompt is a shared instruction
applied 8,000 times. That gives every caption of a register a common origin, which a
classifier can exploit without the register being present to a reader. We measured exactly
that before registering: on our first corpus a DistilRoBERTa classifier reached **0.775**
while a human reading the same captions scored **0.310**, barely above the 0.200 chance
level, and a bag of 25 keywords per register scored **0.441** — above the human.

So the question is:

> **When emotion-conditioning accuracy is measured on a model trained from synthetic
> captions, how much of it reflects an emotional register a reader can perceive, and how much
> reflects the generator's lexical fingerprint? And does training on human-written emotive
> captions change the answer?**

The decoder is a **control**, not a contribution: one architecture, held fixed, so that
differences between arms are attributable to their data.

### Why this is worth registering

The failure mode is invisible to the checks that normally guard against it. On the corpus
above, the floor (0.201), the artifact ablation (gap 0.000), the lexical anchor and the
fold-to-fold spread all passed while a human read the register at close to chance. **Four
automated gates agreed with each other and disagreed with a reader.** A study that reports
emotion accuracy alone can be entirely valid internally and still measure the wrong thing.

## 2. Design

**Six data arms, one decoder.** Every arm trains the same architecture with the same
schedule; only the captions differ.

| arm | captions from | images | structure | captions |
|---|---|---|---|---|
| **S-paired25** | ours — prompt v10, gemini-3.7-flash | Flickr8k | 25/image | 201,900 |
| **S-paired5** | ours, 1 source caption | Flickr8k | 5/image | 40,240 |
| **S-unpaired** | ours, subsampled | Flickr8k | 1/image | 4,390 |
| **V1-paired5** | the retired v1 pilot corpus | Flickr8k | 5/image | 40,240 |
| **V1-unpaired** | the v1 pilot, subsampled | Flickr8k | 1/image | 4,390 |
| **H-unpaired** | Personality-Captions (human) | **YFCC100M** | 1/image | 4,390 |

> **The S and V1 corpora do not exist yet.** They are generated from the registered
> configuration *after* this tag. Only the v1 pilot corpus and Personality-Captions are on
> disk today. Figures above are the planned sizes; the realised counts are reported with the
> results.

### The image sets are not shared, and that governs which comparisons are confirmatory

The human arm is Personality-Captions, whose captions describe **YFCC100M** photographs. Every
other arm is **Flickr8k**. **The image sets do not intersect.** Any human-vs-synthetic
comparison therefore varies provenance, image distribution and groundedness at once —
CLIPScore is **0.578 for the human captions against 0.792 for ours**, 2.64 SD apart.

This is why the confirmatory set is defined by matched images, not by interest:

| | comparisons | images |
|---|---|---|
| **Confirmatory** | S-unpaired vs V1-unpaired · S-paired5 vs V1-paired5 · S-paired25 vs S-unpaired · S-paired25 vs S-paired5 | **identical Flickr8k images** |
| **Reference** | H-unpaired vs S-unpaired (P1, P2) · H-unpaired vs V1-unpaired | **cross-dataset — confounded** |

**P1 and P2 are registered as reference comparisons and are not claimed to isolate
provenance.** `scripts/clipscore_matched.py` already states the live alternative explanation:
*"the human corpus's 4.3-point accuracy advantage is bought by being allowed to ignore the
image, not by being human-written."* That reading is registered here as the rival hypothesis,
not discovered afterwards.

**A groundedness-matched secondary analysis is registered with its weakness stated.** The two
CLIPScore distributions overlap **18.6%**, so a matched subsample of a 4,390-caption arm
retains roughly **816 captions** — far too few to confirm anything. It is registered as
directional evidence only, and its MDE is reported alongside it.

**Generating our captions on the YFCC images was considered and rejected.**
Personality-Captions supplies no neutral caption to rewrite, so it would require inventing one
with a VLM — which is exactly the v1 pilot's fatal design, where Moondream hallucinated and
every rewrite inherited the error.

### Why the 1-caption-per-image arms are 4,390 and not 8,076

The trait mapping is **strict 1:1** (one Personality-Captions trait per register), which yields
4,486 usable cells, so the balanced arm is 878 × 5 = **4,390**. All three 1/image arms are cut
to that size: matching provenance while leaving the human arm half as large would confound P1
with data volume.

The grouped mapping would have supplied 8,076, and was rejected. `lab-notebook.md` records
that **the sign of the stereotypy comparison flips between the two mappings** — strict: human
0.450 vs ours 0.440; grouped: human 0.352 vs ours 0.441 — and concludes *"the strict 1:1
mapping is the more defensible of the two and is what the paper will privilege."* Grouped had
also never been run through any instrument; every human number in this repo was computed under
strict. Choosing the unmeasured mapping because it supplied the arm size we wanted is exactly
the decision this registration exists to prevent. **Cost: 0.4 points of MDE at seed SD 0.02.**

### Why the v1 pilot corpus is an arm

The v1 pilot produced unusable results, and its post-mortem lists several causes at once:
a beam-search KV-cache aliasing bug in the decoder, a VLM (Moondream) rather than humans as
the neutral source, one reference per cell, and figurative style leaking past the prompt.
**The code faults are fixed in this version.** Training the pilot's *data* through the
corrected pipeline therefore separates two explanations that were previously entangled:
**did the pilot fail because of its code, or because of its data?**

It is also the most useful point on this study's central axis. Measured with the same
instrument at matched n, the keyword anchor is:

| corpus | anchor **@ 4,486 cells** | names its own emotion | fails the deterministic validator |
|---|---|---|---|
| **v1 pilot** | **0.718** | **24.2%** | **42.4%** |
| ours (v10) | 0.507 | 3.5% | 2.4% |
| Personality-Captions (human), strict mapping | 0.450 | — | — |

That is a **three-point gradient in lexical stereotypy**, from a corpus that names the emotion
outright to human text that does not. P2 asks whether emotion-conditioning accuracy tracks the
register or the fingerprint; this gradient is the sharpest available test of it, and the v1
corpus is the extreme case.

**It is included as a contrast, not as a rehabilitation.** Its captions are known to be worse:
the neutral source hallucinated, so grounding is broken upstream of the rewrite. Reporting it
as a fair competitor would be dishonest; reporting it as the high-fingerprint end of a
gradient is what it is.

**S-unpaired exists to separate two things S-paired confounds**: having synthetic captions,
and having *every register for the same photograph*. Without it, any S-paired advantage could
be either.

**H-unpaired is matched to S-unpaired on caption count, structure and class count** — 4,390
captions, 1 per image, 5 balanced classes. It is **not** matched on images, and cannot be; see
above.

### Architecture (control, not contribution)

ClipCap-style: frozen CLIP ViT-B/32 → mapping network → GPT-2 with LoRA. The emotion is one
prefix slot. Chosen because it is standard, trains on 8,076 images, and is not the object of
study. **No conditioning-placement variants**: that comparison is dropped, and with it the
five-condition, two-track, 30-run design.

**Held constant by config, not by discipline:** splits, data order per seed,
pre-extracted CLIP features, optimizer, schedule, epochs, batch size, early stopping, decode
settings, and trainable parameter count matched within ±5%.

### Runs

6 arms × 5 folds = **30 training runs**, plus one negative control per arm (shuffled emotion
labels, single fold) for **36 total**. Kaggle notebooks, *Save & Run All*, all outputs preserved for
export; no stage depends on a session surviving.

<details><summary>SUPERSEDED §2 as originally drafted (click to expand)</summary>


Five conditions per track, three seeds each. 5 × 2 × 3 = **30 runs**.

| Cond. | Track A — LSTM from scratch | Track B — ClipCap (GPT-2 + LoRA) |
|-------|------------------------------|-----------------------------------|
| `V0`  | no emotion input | visual prefix only |
| `V1`  | emotion concatenated into `h0` | emotion as first prefix slot |
| `V2`  | emotion appended to every word embedding | emotion broadcast across all prefix slots |
| `V3`  | emotion-queried cross-attention over patches | emotion-queried mapping network |
| `C`   | best variant, retrained on **shuffled** emotion labels | *(negative control)* |

Two tracks rather than one because a single track cannot distinguish "this conditioning
strategy is better" from "GPT-2 happens to like this input shape". Track A is expected to
produce worse captions in absolute terms; that is not its job.

**Held constant** (by config, not by discipline): data and splits; data order per seed;
pre-extracted CLIP features; optimizer, schedule, epochs, batch size, early stopping;
decode settings; trainable parameter count matched within ±5% per track.

</details>

## 3. Hypotheses

Every criterion below is stated **in points of accuracy** and every one has been checked
against the minimum detectable effect **before** registration.

### The MDE, honestly

The MDE has two components and **only one is knowable before training**. Evaluation noise is
measurable now; **seed noise is not knowable until models exist**. So the MDE is registered as
a curve over seed SD, computed with this repository's own estimator
(`emocap.eval.power.mde_two_arms`) on its own measured inputs — **ICC 0.0722, p = 0.72**, from
`runs/mde/result.json`. Both figures come from the v5 corpus, which the study will not use;
ICC is a property of caption clustering and is not expected to move much, but that is an
assumption and it is stated rather than buried.

For the limiting comparison — the 1-caption-per-image arms at **4,390 cells** each:

| seed SD → | 0.000 | 0.005 | 0.010 | **0.020** | 0.030 |
|---|---|---|---|---|---|
| **MDE, 4 confirmatory comparisons, 3 runs averaged** *(governing)* | 3.2 | 3.5 | 4.2 | **6.3** | 8.8 |
| MDE, same but 5 runs averaged | 3.2 | 3.4 | 3.8 | 5.3 | 7.1 |

**The 3-run row governs.** Folds partition one dataset rather than drawing independently, so
averaging five of them cannot be assumed to cut noise by √5. Registering against the
optimistic row is the mistake this section exists to avoid.

The only comparable spread measured so far — the register classifier across three matched
subsamples of one corpus, **SD 0.0076** — puts the expectation near seed SD 0.01, making 0.02
the conservative case.

Reproduce every cell with **`uv run python scripts/compute_mde.py`**, which writes
`runs/mde/result.json`.

**Criteria are therefore set at ≥7 points**, which clears the governing MDE at seed SD 0.02
(6.3) with margin, and clears it at every smaller value.

### Two rules that make a null mean something

1. **A comparison whose observed fold spread implies an MDE above its criterion is reported
   as UNDERPOWERED, not as a null.** This is the rule this study exists to obey: a previous
   pre-registration in this line of work committed to numbers that were not statistically
   reachable, and reporting those as nulls is what cost it credibility.
2. **A gap smaller than the observed fold spread is reported as null regardless of its
   p-value.** Unchanged from the original registration.

### Confirmatory vs exploratory

**Six comparisons are confirmatory** and carry Holm correction: H vs S (unpaired),
S-paired25 vs S-unpaired, S-unpaired vs V1-unpaired, H-unpaired vs V1-unpaired, S-paired5 vs
V1-paired5, and S-paired25 vs H-unpaired. The remaining nine pairwise comparisons are
**exploratory**: reported uncorrected, explicitly labelled, and **no hypothesis is registered
on them**. Correcting over all 15 would push the MDE to 5.6 points at seed SD 0.02 and buy
nothing, since nine of them test no prediction.

**P1 carries a caveat that must be reported with it.** Its expected effect is 4.3 points,
measured on training data. At seed SD 0.02 the MDE is 5.2 — *above* that. So a null on P1 is
**underpowered unless the observed spread is small enough to bring the MDE below 4.3**, and it
will be reported that way rather than as evidence of no difference.

| | Prediction | Falsified if | margin over MDE |
|---|---|---|---|
| **P1** | H-unpaired beats S-unpaired on raw emotion accuracy | S-unpaired ≥ H-unpaired, or the difference straddles zero | 4.3 pt effect — **near the MDE, see caveat above** |
| **P2** | The gap is **not** fully explained by lexical stereotypy: H's margin over its own keyword anchor exceeds S's margin over its own, by **≥7 points** | margin difference ≤ 0 | 7.0 vs MDE 6.3 at seed SD .02 |
| **P3a** | S-paired beats S-unpaired on raw accuracy (sanity check on the extra data) | S-paired ≤ S-unpaired | — |
| **P3b** | That improvement **also** appears in the margin over the anchor, by **≥7 points** | margin flat (≤0) — meaning per-image register contrast teaches vocabulary and nothing else | 7.0 vs MDE 6.3 |
| **P4** | Models trained on S score lower under an H-trained judge than H-trained models do under an S-trained judge (asymmetric transfer) | symmetric, or reversed | — |
| **P5** | No model exceeds the **human** blind-guess score on its own corpus | — (a bound, not a hypothesis; reported either way) | — |
| **P6** | **Emotion accuracy tracks the corpus's keyword anchor, not its human legibility.** Across the three provenances at matched structure, accuracy ranks with the anchor (v1 0.718 > ours 0.507 > human 0.450), so V1-unpaired scores **highest** on raw accuracy and lowest on the margin over its own anchor, **each step ≥7 points** | accuracy does not rank with the anchor, or V1-unpaired's margin is not the smallest | **≥7 pts** vs MDE 6.3 |

**P6 is what the v1 arm buys.** With three corpora whose anchors span 0.450 to 0.718, the
prediction is falsifiable in a way no two-corpus comparison could be: if raw accuracy ranks
with stereotypy while the margin over the anchor ranks the other way, then "emotion accuracy"
is substantially a measure of lexical provenance. **P6 failing is as informative as P6
holding** — it would mean the metric is more robust to provenance than we think.

**P2 is the study's actual claim.** P1 alone would be a result about accuracy; P2 is the
result about what accuracy *means*, because it asks whether the human-written corpus wins for
a reason a keyword bag cannot reproduce.

**P5 has no falsification condition on purpose.** It is a bound we report whether or not it
holds. A model exceeding the human score would not be a triumph — on this corpus a reader
scores 0.592 and a keyword bag 0.510, so a model at 0.85 is prima facie evidence of a
shortcut, not of comprehension.

### What was dropped, and why

The original P2 ("human and synthetic margins are identical") is **dead**: the accuracy gap
largely survived the anchor correction, so the claim now runs the other way. A sixth
prediction about the anchor declining monotonically with evaluation-set size was dropped with
the v1-prompt corpus it depended on.

<details><summary>SUPERSEDED §3 as originally drafted (click to expand)</summary>


| | Claim | Falsified if |
|---|---|---|
| **H1** | V1–V3 produce captions whose intended emotion is recoverable above chance; V0 does not. | No variant exceeds V0 emotion accuracy by ≥10 points. |
| **H2** | Persistent conditioning (V2, V3) beats one-shot conditioning (V1). | V1 ≥ V2 and V1 ≥ V3, or the differences straddle zero. |
| **H3** | The V0–V3 ranking is the same in both tracks (Spearman ρ = 1.0). | The rankings disagree on any adjacent pair outside its CI. |
| **H4** | Conditioning trades descriptive accuracy for tone: CIDEr-D declines as conditioning strengthens. | CIDEr-D is flat or rises with conditioning strength. |

</details>

## 4. Metrics

**Primary — emotion accuracy.** Top-1 accuracy of a held-out DistilRoBERTa classifier at
recovering the requested emotion from the generated caption alone. Trained on the
**training split only**. The one metric that directly tests the thesis.

**Secondary.** CIDEr-D and BLEU-4 against the five references per cell; CLIPScore for
reference-free image grounding; distinctiveness as mean pairwise self-BLEU across the five
captions generated for one image.

**Confirmatory.** Human evaluation: 150 items, 3 raters, blinded and randomised with
attention checks, 5-point tone-match and grounding scales, Krippendorff's α reported. Run
once, on final models only.

### The classifier is an instrument and gets validated as one

Trained on Gemini output, it may learn surface artifacts — an exclamation mark for
*joyful*, a dependent clause for *tense* — rather than emotional register. Three
mitigations, all reported:

1. **Artifact ablation.** Accuracy on punctuation-stripped, lowercased captions reported
   alongside raw. A large gap means the classifier reads punctuation, and the stripped
   number becomes primary.
2. **Off-distribution validation.** 250 hand-labelled captions not written by Gemini. This
   is the instrument's real measurement error.
3. **Confusion matrix**, always. If *romantic* and *sad* collapse into each other, that is
   a finding about the taxonomy, not noise to average away.
4. **Report against the lexical-shortcut anchor.** Auditing Part 0 found stereotyped
   register vocabulary that a 25-keyword-per-register rule can exploit with no model at
   all. Under the original prompt that rule reached **0.641** against 0.200 chance —
   nearly two-thirds of the primary metric available from vocabulary alone — driven by
   "alone"/"empty" in 36% of sad captions, "soft"/"gentle" in 41% of romantic and
   "bright" in 31% of joyful. The prompt was rewritten (see `docs/deviations.md`,
   2026-08-19); under v5 those rates fall to 1%, 6% and 14%.

   **The anchor is sample-size dependent and is never quoted as a bare number.** On the
   generated corpus it falls from 0.440 at 4,486 cells to 0.403 at 24,925 and **0.332 at
   the full 201,900** — a keyword rule fitted on few images transfers well within that
   narrow pool and degrades as the pool widens. The decline is a property of the
   estimator rather than of one corpus: the human comparison corpus falls 0.034 over the
   same span where ours falls 0.031. The anchor is therefore **recomputed on whatever
   evaluation set the primary metric is measured on, at that set's own n**, and reported
   beside it. Three findings were retracted on 2026-08-19 for violating exactly that.
   This threatens *construct* validity, not internal validity. Stereotypy cannot explain an
   arm's advantage over its own negative control, since both share a corpus — but it very
   much can explain one arm beating another, because **the arms differ in exactly how
   stereotyped they are** (anchor 0.718 for v1, 0.507 for ours, 0.450 for human text). That
   is why every arm comparison is reported as the margin over *that arm's own* anchor, and
   why P2 is the study's real claim rather than P1.

## 5. Analysis plan

- Each arm compared to every other on the primary metric, and each to its own negative control.
- Paired bootstrap over test **images** (clustered — the five emotion cells of one image
  are not independent), 10,000 resamples, 95% CIs.
- Holm–Bonferroni across the **6 confirmatory** arm comparisons. The other 9 pairwise
  comparisons are exploratory, reported uncorrected and labelled as such.
- **Evaluation regime: 5-fold CV over each arm, folds split by `image_id`.** Never by
  caption — the five register cells of one photograph are not independent, and splitting by
  caption leaks the image across the split. Fold-to-fold spread absorbs both data and
  initialisation variance, replacing the earlier "spread across three seeds".

  This changed because the MDE on a single 1,000-cell test split is **6.5 points** against
  **2.7–5.2** under CV depending on seed noise, and a 5-point criterion was registered against
  the former — i.e. it could not have been falsified. Logged as a pre-tag deviation in `docs/deviations.md`.

- **A gap smaller than the observed fold spread is reported as null regardless of its
  p-value.** This rule is unchanged and is not weakened by the regime change. It exists
  because a 9.6-point gap once survived a 4-SE check and was still a small-sample artifact:
  **fold variance rules out noise, not bias.**
- **Smallest effect of interest: +7 percentage points**, matching the falsification criteria
  in §3. The earlier value of +10 was inherited from the dropped conditioning design and would
  have made a 7-point effect simultaneously confirmatory and below the threshold of interest.
  The detectability gate uses the same +7.

## 6. Anchors and gates

| Check | Expectation | If it fails |
|---|---|---|
| **Detectability** — floor-to-ceiling range vs the design's MDE | the range must admit the smallest effect of interest (+7 pts) at the MDE. Currently floor 0.201, ceiling 0.773, range 57 pts, MDE ≤7.1 at any plausible seed SD → **passes** | The design cannot resolve the effect it claims to test. **Halt the study.** |
| **Human legibility** — blind register guess on the corpus, caption only | reported with its neutral rate and CI, beside the anchor for the same corpus at the same n. Pilot: **0.592 ±0.049 (n=100) vs anchor 0.507 (n=5,625)** — a margin of ~1.7 SE, and see the caveat below | A corpus a reader cannot decode above its own keyword anchor is not measuring register. **Halt.** |
| Floor — accuracy with randomly reassigned labels | ≈ 0.20 | The metric is broken. |
| **Lexical shortcut** — top-25-keyword-per-register rule | recomputed per evaluation set at its own n (0.332 on the full corpus) | Not a failure, a correction: this much of the primary metric needs no emotional register at all. The model's contribution is the margin above it. |
| Manipulation check — negative control accuracy | ≤ 0.25 | The classifier reads something other than the requested register. **Results not interpretable.** |
| Negative control — shuffled emotion labels, per arm | ≈ floor | The conditioning is not doing the work. |
| Visual-dependence probe — zero the features | captions change substantially | The run collapsed to a language prior. **Exclude the run.** |

### Two things the gates rest on that are weaker than they look

**The legibility gate's pilot value is one unblinded self-measurement.** 100 items, scored by
the study's author, binomial SE ≈0.049, so the +0.082 margin is about **1.7 SE**. It is also
not like-for-like: the anchor is a *trained* rule doing 5-fold CV over thousands of cells,
while the reader was zero-shot on 100 items. **"Keywords beat a human" is therefore not by
itself proof of leakage**, and the gate is registered with that stated. Before the gate is
applied to the finished corpus the human sample is widened to ≥300 items and reported with
its CI.

**Every anchor in this document is provisional until the corpus exists.** The three-corpus
gradient (0.718 / 0.507 / 0.450) is measured at **4,486 cells**. This project's own central
finding is that this estimator falls ~11 points from 4.5k to 200k cells — the v5 corpus went
0.440 → 0.332 over exactly that span. The finished 201,900-cell corpus will therefore sit well
below 0.507, and so will the v1 arm at 40,240. §4's rule governs: **the anchor is recomputed on
whatever evaluation set the metric is measured on, at that set's own n.** The gradient's
*ordering* is the registered prediction; its *values* are not.

### Why the ceiling gate is a detectability criterion, not a level

The registered gate was `ceiling_min: 0.85`, set a priori with nothing calibrating it. Three
things falsify it as written:

1. **Our corpus does not reach it** — the ceiling is ~0.77, so the gate says *halt*.
2. **Neither does human-written text.** Measured with the identical instrument at matched n,
   Personality-Captions reaches **0.726**. A gate that halts a study on data matching human
   performance is a miscalibrated instrument, not a finding.
3. **The ceiling is sample-size dependent**: 0.620 at 1,225 cells → 0.737 at 24,925 → 0.773
   at 201,900. There is no single number to threshold.

**The replacement:** the study proceeds if the floor-to-ceiling range leaves room for the
smallest effect of interest (+10 points) to be detected at the design's MDE. Floor and ceiling
are reported **with their n**, beside that evaluation set's recomputed anchor.

This cannot be gamed by choosing a level, it does not make the human number load-bearing for a
gate, and it asks the right question — not *"is 0.773 high enough?"*, which has no principled
answer, but *"can this design resolve the effect it claims to test?"*, which does.

**Rejected by measurement:** gating on the *margin over the anchor*. Accuracy rises with n
while the anchor falls, so the margin moves +0.198 across the range against accuracy's +0.090
— it amplifies the very confound it was meant to control. Rejected alternatives in full:
`RESEARCH-LOG.md` Part 11.

### Why 0.85 is not calibrated, and human text does not reach it

0.85 was set a priori with nothing calibrating it. Measured with the identical instrument
at matched sample size, **human-written style-conditioned captions reach 0.726**
(Personality-Captions, 4,486 cells, DistilRoBERTa, folds by image) — so the gate as
written would halt a study on data matching human performance. That is a miscalibrated
instrument, not a finding about the corpus.

Recalibrating the threshold against the measured human ceiling is legitimate **pre-tag,
with the reasoning logged**, and indefensible afterwards. The final value is set before
`prereg-v1` is applied; until then 0.85 stands as registered and the measured human number
is recorded here so the change is visible rather than silent.

Two limits on the human figure, both stated because they bound the claim:
it is measured at 4,486 cells (the largest the human corpus supports under a 1:1 trait
mapping) and the anchor-style sample-size dependence means the level may rise with n; and
the comparison is 5-way, so it is **not** comparable to binary style-accuracy figures
reported on FlickrStyle10K or SentiCap.

### A pre-registered prediction the data already contradicts

`anchors.expected_hardest_registers` was registered as `[tense, humorous]`, on the
reasoning that those two have the weakest lexical markers. Measured register difficulty
on the v5 Part 0 sample — the generator's own strain report over 1,225 cells — orders
them **romantic 1.05 > humorous 0.90 > sad 0.89 > tense 0.85 > joyful 0.15**. `tense` is
not among the hardest; `romantic` is, which the independent human judge found as well
(`docs/part0-human-judge-notes.md`).

The prediction is **left as registered and the disagreement reported**. It is recorded
here before any model is trained, so it also functions as a prior on where the confusion
matrix in §4.3 should concentrate.

## 7. Exclusion criteria

A run is excluded **only** for: crashed or incomplete training; NaN loss; or a failed
visual-dependence probe.

**Poor metric performance is never grounds for exclusion.** Stating this before seeing any
results is most of what this document buys.

**Nor is a high self-reported `strain` value.** The generator marks each caption cell 0/1/2
for how well the register fits the scene, and 13,376 cells (6.6%) are marked 2, "no honest
reading exists". Those cells are **recorded and retained, never filtered**. Decided
2026-08-20, before the tag, for two reasons: the flag does not predict grounding defects
(rate by level is 1.6% / 2.6% / 2.4%, flat), and excluding it would take the corpus from
perfectly balanced at 40,380 cells per register to a 16.9% imbalance concentrated in
`romantic`, which would lose 52× as much data as `joyful` — making any later finding about
`romantic` inseparable from it having had less training data.

## 8. What a null result looks like

**If P1 is falsified** — the human-written arm does not beat the synthetic arms — that is
reported as the headline. It would mean synthetic emotive captions are as good a training
signal as human ones for this task, which is a useful negative for anyone building such a
corpus.

**If P2 is falsified** — the human arm's advantage *is* fully explained by lexical stereotypy
— that is a stronger result than the positive would be. It would say emotion-conditioning
accuracy is measuring vocabulary provenance rather than register, and that the metric as
commonly reported does not mean what it appears to mean. **This is the outcome the study is
built to be able to detect.**

**If the arms are indistinguishable at the design's MDE**, that is reported as such and not
resolved by adding comparisons after the fact. A difference smaller than the observed fold spread is reported as null regardless of its
p-value, and a comparison whose spread implies an MDE above its criterion is reported as
underpowered rather than null.

**If the v1 arms match the v10 arms**, the pilot's failure was its code and not its data —
which reverses the post-mortem's emphasis and is worth reporting plainly, including that the
retired corpus was discarded on an incomplete diagnosis.

No outcome here requires a positive finding to be publishable, and the exclusion rules in §7
are what make that credible: nothing can be dropped for performing badly.
