# Pre-registration — EmoCap v2

**Status:** draft, to be tagged `prereg-v1` before any data generation or training.
**Frozen settings:** [`configs/prereg.lock.yaml`](../configs/prereg.lock.yaml)
**Registration mechanism:** git tag. The tag's commit timestamp is the record. OSF
registration is deliberately deferred and can be added later without invalidating this.

---

## 1. Question

Given an image and a target emotion, generate a caption that describes the image *and*
reads in that register. **Where in a decoder should the emotion signal be injected, and
does the answer depend on the decoder family?**

## 2. Design

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

## 3. Hypotheses

| | Claim | Falsified if |
|---|---|---|
| **H1** | V1–V3 produce captions whose intended emotion is recoverable above chance; V0 does not. | No variant exceeds V0 emotion accuracy by ≥10 points. |
| **H2** | Persistent conditioning (V2, V3) beats one-shot conditioning (V1). | V1 ≥ V2 and V1 ≥ V3, or the differences straddle zero. |
| **H3** | The V0–V3 ranking is the same in both tracks (Spearman ρ = 1.0). | The rankings disagree on any adjacent pair outside its CI. |
| **H4** | Conditioning trades descriptive accuracy for tone: CIDEr-D declines as conditioning strengthens. | CIDEr-D is flat or rises with conditioning strength. |

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
   2026-08-19); under the final v5 prompt the rule reaches **0.394**, with those rates
   at 1%, 6% and 14%. Classifier accuracy is always reported alongside this baseline,
   recomputed by `scripts/audit_captions.py` under the estimator named in the lock.
   This threatens *construct* validity, not internal validity: stereotypy affects every
   condition equally, so it cannot explain V0 vs V2, but it does mean "emotion accuracy"
   partly measures keyword emission rather than register.

## 5. Analysis plan

- Each variant compared to its track's V0 on the primary metric.
- Paired bootstrap over test **images** (clustered — the five emotion cells of one image
  are not independent), 10,000 resamples, 95% CIs.
- Holm–Bonferroni across the three comparisons within each track.
- Seed variance reported as the spread across three runs. **A gap smaller than that spread
  is reported as null regardless of its p-value.**
- Smallest effect of interest fixed at **+10 percentage points** of emotion accuracy.

## 6. Anchors and gates

| Check | Expectation | If it fails |
|---|---|---|
| Ceiling — accuracy on the reference captions | ≥ 0.85 | The generated data lacks separable tone. **Halt the study.** |
| Floor — accuracy with randomly reassigned labels | ≈ 0.20 | The metric is broken. |
| **Lexical shortcut** — top-25-keyword-per-register rule | measured **0.394** | Not a failure, a correction: this much of the primary metric needs no emotional register at all. The model's contribution is the margin above it. |
| Manipulation check — V0 accuracy | ≤ 0.25 | The classifier reads something other than tone. **Results not interpretable.** |
| Negative control — condition C | indistinguishable from V0 | Conditioning is not doing the work. |
| Visual-dependence probe — zero the features | captions change substantially | The run collapsed to a language prior. **Exclude the run.** |

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

## 8. What a null result looks like

If H1 is falsified — no conditioning strategy beats the unconditioned baseline by the
smallest effect of interest — that is the finding, and it is reported as the headline. The
plausible mechanism is already identified: with five references per cell and a shared
visual encoder, the emotion embedding may be a low-capacity channel relative to the
language prior. Reporting that is more useful than a tuned positive.
