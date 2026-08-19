# Pre-registration — EmoCap v2

> ## ⚠ THIS DOCUMENT IS MID-REWRITE — DO NOT CITE §1–§3
>
> The study design changed on 2026-08-19/20 after the corpus was generated and two external
> reviews landed. **§1, §2, §3, part of §5 and §6's ceiling gate still describe the
> SUPERSEDED design** (five conditions, two tracks, an LSTM trained from scratch) and are
> fenced as such below. They are preserved because they are what was actually drafted, not
> because they are current.
>
> The decided replacement content is **not written here yet, deliberately** — it lands in one
> pass together with the tag, so the pre-tag state stays legible. Placeholders below carry IDs
> (`PH-nn`); the tracker is `PLACEHOLDERS.md` (local only).
>
> **Authoritative right now:** `RESEARCH-LOG.md` for findings and the gate reformulation
> (Part 11), `TASKS.md` Phase 1 for what is pending, `METHODOLOGY.md` for instruments.
> §4, §7 and §8 of this document ARE current.

**Status:** draft, mid-rewrite. To be tagged `prereg-v1` before any training.
**Frozen settings:** [`configs/prereg.lock.yaml`](../configs/prereg.lock.yaml)
**Registration mechanism:** git tag. The tag's commit timestamp is the record. OSF
registration is deliberately deferred and can be added later without invalidating this.

---

## 1. Question

<!-- PH-01 -->
> **PLACEHOLDER PH-01 — the research question is being rewritten.**
>
> The question below is **SUPERSEDED**. It asks where in a decoder emotion should be
> injected; that design space is well covered and the reviews were right that it is not a
> 2026 contribution on its own. The study's claim is now about **data**, with the decoder
> demoted to a control.
>
> Decided direction (`RESEARCH-LOG.md` Part 0): what emotion-conditioning accuracy measures
> when the training data is LLM-synthesised. Final wording pending.

~~Given an image and a target emotion, generate a caption that describes the image *and*
reads in that register. **Where in a decoder should the emotion signal be injected, and
does the answer depend on the decoder family?**~~

## 2. Design

<!-- PH-02 -->
> **PLACEHOLDER PH-02 — the design is being rewritten from conditioning variants to data arms.**
>
> **SUPERSEDED below.** Track A (LSTM from scratch) is **dropped** — 8,076 captions cannot
> train a decoder from scratch, and architecture-dependence is no longer the claim.
> Conditioning variants collapse from five to two. 30 runs becomes 6–9.
>
> Decided direction: **three data arms** — S-paired (ours, 25 registers per image), S-unpaired
> (ours, 1 per image), H-unpaired (Personality-Captions, human) — matched on images, caption
> count, structure and class count. A possible fourth arm (generator capability, gemini-3.5-flash
> via Vertex) is **pending the pilot**; see `TASKS.md` 1.2.
>
> Also pending: whether a uniform visual pathway (patch features for every condition) is still
> needed now that variants collapse to two. See `TASKS.md` Phase 4.
>
> Arm table, sizes and the judge design are in `RESEARCH-LOG.md` Part 7. Not transcribed here
> until the one-pass rewrite.

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

<!-- PH-03 -->
> **PLACEHOLDER PH-03 — H1–H4 are superseded by P1–P6.**
>
> **SUPERSEDED below.** H1–H4 concern conditioning placement, which is no longer the claim.
> H3 in particular (Spearman ρ = 1.0 across tracks) is void — there is only one track now.
>
> Replacements P1–P6 are drafted with falsification criteria in `TASKS.md` 1.5. **P2 as first
> drafted is dead**: it predicted the human/synthetic margins would be identical, and the
> accuracy gap largely survives the anchor correction, so the claim now runs the other way.
> Each replacement must be checked against the MDE before it is registered — the original P2's
> 5-point criterion was **unfalsifiable** at the planned sample size.

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

- Each variant compared to its track's V0 on the primary metric.
- Paired bootstrap over test **images** (clustered — the five emotion cells of one image
  are not independent), 10,000 resamples, 95% CIs.
- Holm–Bonferroni across the three comparisons within each track.
<!-- PH-04 -->
- **PH-04 — PENDING.** ~~Seed variance reported as the spread across three runs.~~ The
  evaluation regime changes to **5-fold CV over each arm**, because the MDE on a single
  1,000-cell test split is 6.5 points against 2.3 under CV, and the original P2 criterion was
  unfalsifiable at the former. Fold-to-fold spread absorbs both data and initialisation
  variance. **The rule itself stands and is not weakened: a gap smaller than the observed
  spread is reported as null regardless of its p-value.**
- Smallest effect of interest fixed at **+10 percentage points** of emotion accuracy.

## 6. Anchors and gates

| Check | Expectation | If it fails |
|---|---|---|
| Ceiling — accuracy on the reference captions | **PH-05 — PENDING.** ~~≥ 0.85~~ Replaced by a *detectability* criterion; see below | The generated data lacks separable tone. **Halt the study.** |
| Floor — accuracy with randomly reassigned labels | ≈ 0.20 | The metric is broken. |
| **Lexical shortcut** — top-25-keyword-per-register rule | recomputed per evaluation set at its own n (0.332 on the full corpus) | Not a failure, a correction: this much of the primary metric needs no emotional register at all. The model's contribution is the margin above it. |
| Manipulation check — V0 accuracy | ≤ 0.25 | The classifier reads something other than tone. **Results not interpretable.** |
| Negative control — condition C | indistinguishable from V0 | Conditioning is not doing the work. |
| Visual-dependence probe — zero the features | captions change substantially | The run collapsed to a language prior. **Exclude the run.** |

### PH-05 — the ceiling gate, decided but not yet applied

<!-- PH-05 -->
> **The replacement is decided and deliberately not written into the gate table yet.**
>
> **Revised gate:** the study proceeds if the range between the measured floor and ceiling
> leaves room for the smallest effect of interest (+10 points) to be detected at the design's
> MDE. Ceiling and floor are reported **with their n** and beside that evaluation set's
> recomputed anchor.
>
> Against current numbers: floor 0.201, ceiling 0.773 → a 57-point range at an MDE of 2.3
> points under 5-fold CV. Passes.
>
> Gating on the **margin over the anchor** was considered and **rejected by measurement**:
> accuracy rises with n while the anchor falls, so the margin moves +0.198 across the range
> against accuracy's +0.090 — it amplifies the confound. Full reasoning and the rejected
> alternatives are in `RESEARCH-LOG.md` Part 11.

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
