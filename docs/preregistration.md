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

**Three data arms, one architecture.** Every arm trains the same decoder on the same images
with the same schedule; only the captions differ.

| arm | captions from | images | captions | structure |
|---|---|---|---|---|
| **S-paired** | ours (prompt v10, gemini-3.7-flash) | 8,076 | 201,900 | 25 per image — all 5 registers × 5 source captions |
| **S-unpaired** | ours, subsampled from the same corpus | 8,076 | 8,076 | 1 per image, 1 register |
| **H-unpaired** | Personality-Captions (human-written) | ~8,076 | ~8,076 | 1 per image, 1 style |

**S-unpaired exists to separate two things S-paired confounds**: having synthetic captions,
and having *every register for the same photograph*. Without it, any S-paired advantage could
be either.

**H-unpaired is matched to S-unpaired** on images, caption count, structure and class count,
so the arms differ in provenance and not in shape. Personality-Captions styles are mapped to
our five registers; the mapping is fixed in `configs/prereg.lock.yaml` before training and
**the mapping's sensitivity is reported**, because an earlier mapping choice flipped the sign
of a human-vs-synthetic comparison.

### Architecture (control, not contribution)

ClipCap-style: frozen CLIP ViT-B/32 → mapping network → GPT-2 with LoRA. The emotion is one
prefix slot. Chosen because it is standard, trains on 8,076 images, and is not the object of
study. **No conditioning-placement variants**: that comparison is dropped, and with it the
five-condition, two-track, 30-run design.

**Held constant by config, not by discipline:** images and splits, data order per seed,
pre-extracted CLIP features, optimizer, schedule, epochs, batch size, early stopping, decode
settings, and trainable parameter count matched within ±5%.

### Runs

3 arms × 5 folds = **15 training runs**, plus a negative control per arm (shuffled emotion
labels) for **18 total**. Kaggle notebooks, *Save & Run All*, all outputs preserved for
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
against the minimum detectable effect **before** registration. Under 5-fold CV over a whole
arm the MDE is **2.3 points** (80% power, α 0.05 Holm-adjusted over 6 comparisons, ICC 0.070,
design effect 2.69 on the paired arm). **No criterion below sits under 2.3.** This check is
not decorative: the first draft of P2 used a 5-point criterion against an MDE of 6.5, which
made it unfalsifiable, and that was caught by computing the MDE rather than by review.

| | Prediction | Falsified if | margin over MDE |
|---|---|---|---|
| **P1** | H-unpaired beats S-unpaired on raw emotion accuracy | S-unpaired ≥ H-unpaired, or the difference straddles zero | 4.3 pt effect vs 2.3 |
| **P2** | The gap is **not** fully explained by lexical stereotypy: H's margin over its own keyword anchor exceeds S's margin over its own, by ≥3 points | margin difference ≤ 0 | 3.0 vs 2.3 |
| **P3a** | S-paired beats S-unpaired on raw accuracy (sanity check on the extra data) | S-paired ≤ S-unpaired | — |
| **P3b** | That improvement **also** appears in the margin over the anchor, by ≥3 points | margin flat (≤0) — meaning per-image register contrast teaches vocabulary and nothing else | 3.0 vs 2.3 |
| **P4** | Models trained on S score lower under an H-trained judge than H-trained models do under an S-trained judge (asymmetric transfer) | symmetric, or reversed | — |
| **P5** | No model exceeds the **human** blind-guess score on its own corpus | — (a bound, not a hypothesis; reported either way) | — |

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
   This threatens *construct* validity, not internal validity: stereotypy affects every
   condition equally, so it cannot explain V0 vs V2, but it does mean "emotion accuracy"
   partly measures keyword emission rather than register.

## 5. Analysis plan

- Each arm compared to every other on the primary metric, and each to its own negative control.
- Paired bootstrap over test **images** (clustered — the five emotion cells of one image
  are not independent), 10,000 resamples, 95% CIs.
- Holm–Bonferroni across the six arm comparisons.
- **Evaluation regime: 5-fold CV over each arm, folds split by `image_id`.** Never by
  caption — the five register cells of one photograph are not independent, and splitting by
  caption leaks the image across the split. Fold-to-fold spread absorbs both data and
  initialisation variance, replacing the earlier "spread across three seeds".

  This changed because the MDE on a single 1,000-cell test split is **6.5 points** against
  **2.3** under CV, and a 5-point criterion was registered against the former — i.e. it could
  not have been falsified. Logged as a pre-tag deviation in `docs/deviations.md`.

- **A gap smaller than the observed fold spread is reported as null regardless of its
  p-value.** This rule is unchanged and is not weakened by the regime change. It exists
  because a 9.6-point gap once survived a 4-SE check and was still a small-sample artifact:
  **fold variance rules out noise, not bias.**
- Smallest effect of interest fixed at **+10 percentage points** of emotion accuracy.

## 6. Anchors and gates

| Check | Expectation | If it fails |
|---|---|---|
| **Detectability** — floor-to-ceiling range vs the design's MDE | the range must admit the smallest effect of interest (+10 pts) at the MDE. Currently floor 0.201, ceiling 0.773, range 57 pts, MDE 2.3 → **passes** | The design cannot resolve the effect it claims to test. **Halt the study.** |
| **Human legibility** — blind register guess on the corpus, caption only | reported with its neutral rate, beside the anchor for the same corpus. Currently **0.592 human vs 0.510 anchor** | A corpus a reader cannot decode above its own keyword anchor is not measuring register. **Halt.** |
| Floor — accuracy with randomly reassigned labels | ≈ 0.20 | The metric is broken. |
| **Lexical shortcut** — top-25-keyword-per-register rule | recomputed per evaluation set at its own n (0.332 on the full corpus) | Not a failure, a correction: this much of the primary metric needs no emotional register at all. The model's contribution is the margin above it. |
| Manipulation check — negative control accuracy | ≤ 0.25 | The classifier reads something other than the requested register. **Results not interpretable.** |
| Negative control — shuffled emotion labels, per arm | ≈ floor | The conditioning is not doing the work. |
| Visual-dependence probe — zero the features | captions change substantially | The run collapsed to a language prior. **Exclude the run.** |

### Why the ceiling gate is a detectability criterion, not a level

The registered gate was `ceiling_min: 0.85`, set a priori with nothing calibrating it. Three
things falsify it as written:

1. **Our corpus does not reach it** — the ceiling is ~0.77, so the gate says *halt*.
2. **Neither does human-written text.** Measured with the identical instrument at matched n,
   Personality-Captions reaches **0.726**. A gate that halts a study on data matching human
   performance is a miscalibrated instrument, not a finding.
3. **The ceiling is sample-size dependent**: 0.612 at 1,225 cells → 0.737 at 24,925 → 0.773
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

If H1 is falsified — no conditioning strategy beats the unconditioned baseline by the
smallest effect of interest — that is the finding, and it is reported as the headline. The
plausible mechanism is already identified: with five references per cell and a shared
visual encoder, the emotion embedding may be a low-capacity channel relative to the
language prior. Reporting that is more useful than a tuned positive.
