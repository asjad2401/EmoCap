# Lab Notebook

Append-only, dated. What was run, what was observed, what was decided, what surprised us.
Written as it happens, never reconstructed.

Negative entries are the valuable ones.

---

## 2026-08-18 — v1 postmortem and v2 scaffold

**Read all seven pilot notebooks end to end, including outputs.** Findings in
[`docs/v1-pilot-postmortem.md`](v1-pilot-postmortem.md).

The headline: the pilot's beam search handed every beam child the same mutable
`DynamicCache`. Track B's val loss was healthy at 3.10 (ppl ≈ 22) while every sampled
caption was word salad. **Healthy loss with broken samples is a decoder bug, essentially
always — training never exercises the decoder.** That single heuristic would have found this
in an afternoon instead of ending the pilot.

Also learned the pilot got much less far than its narrative suggested: one model of eight
ever trained, Track A stopped at epoch 8/20, BLEU eval died on a `KeyboardInterrupt`, and
the results cells never executed. Worth remembering that a notebook full of confident
markdown says nothing about what actually ran.

**Decisions locked** (see the artifact plan and `configs/prereg.lock.yaml`): Flickr8k human
captions as the neutral source; git tag for pre-registration; three seeds; human eval in
scope at 150 items / 3 raters. Track A kept, on my recommendation, because H3 needs it.

**Built:** repo scaffold, `src/emocap/decode/beam.py`, and 25 tests.

- All 25 pass in 2.68 s on CPU.
- Our beam search **matches HuggingFace's token-for-token** on a tiny random GPT-2 at beam
  1, 2, and 4 across batch sizes 1 and 3. Isolated the search by setting
  `min_new_tokens == max_new_tokens` (so no early stopping) and `length_penalty=0.0` (so
  both reduce to sum-logprob).
- Verified the tests actually catch the v1 defect by reproducing v1's implementation and
  running the same suite against it: **4 genuine catches** —
  `test_reordering_hands_the_model_fresh_storage` (the direct one),
  `test_mutating_state_still_correct`, `test_no_repeat_ngram_blocks_repeats`, and
  `test_opaque_state_leaf_is_rejected`. Three further failures were only new API surface
  v1 lacked, not evidence.

**Surprise:** v1's no-repeat-ngram blocking was *also* broken — it produced `[3,3,3,3,4]`
with `no_repeat_ngram_size=2`. I had read that code and judged it merely redundant. Reading
is not testing.

**Wrong turn worth recording:** I first wrote a test asserting that different prefixes give
different *captions*. It failed — a randomly initialised GPT-2 has one dominant output token
that swamps any prefix. The unit test now asserts on logits (plumbing); the behavioural
claim moved to the visual-dependence probe, where it belongs and where it will gate real
runs.

**Next:** stage 01 (`data_flickr8k_splits`) and stage 02 (`captions_generate`) — the latter
runs locally against the Gemini key and is not GPU-bound, so it can proceed in parallel with
everything else.

## 2026-08-18 (later) — runtime, stage 01, stage 02

**Built:** `emocap.runtime` (config + prereg lock + manifests + seeds), `emocap.data.flickr8k`
(stage 01), `emocap.data.prompt` and `emocap.data.generate` (stage 02). Suite now **83 tests,
3.1 s on CPU**.

The lock check earns its keep already: `test_shipped_data_config_agrees_with_the_lock` and
`test_decode_defaults_match_the_lock` mean `configs/data.yaml` and `DecodeConfig`'s defaults
cannot drift from the pre-registration without a test going red. That is the mechanism
against v1's actual failure, where a notebook header advertised filters the code had
overridden.

**Prompt rewrite.** v1's prompt forbade metaphor in one flat line and Gemini ignored it on
most rows. Two changes: the input is now one short human caption rather than a 38-word dense
VLM paragraph (less room to elaborate), and the prohibitions carry worked BAD/GOOD pairs for
the exact failure modes v1 exhibited rather than being merely named.

**Useful calibration.** I tuned the deterministic style filter against six references taken
verbatim from v1's own output. First version caught **0 of 6** — my regex expected
`"a reminder of"` while v1's actual leakage is the appositive form with modifiers wedged in:
*"the sled a cold reminder"*, *"a tender ballet"*, *"the house a silent, distant witness"*.
After widening it to allow up to two modifiers between article and abstraction noun: **6 of
6 caught, 0 false positives on 5 clean captions.** Those six are now pinned as regression
fixtures. Worth remembering that a filter written from imagination catches imaginary bugs.

**Two of my own bugs, both caught by writing the test:**
- `git_sha(short=True)` built `["git", "rev-parse", "--short"]` and dropped `HEAD`, so every
  manifest would have recorded `"unknown"` — silently destroying provenance on every run.
- `load_config("prereg.lock")` failed because `.lock` counts as a suffix, so it never got
  `.yaml` appended.

**Also:** the v1 caption CSV turned up in the tree (17 MB, 43,891 rows, ~8,000 Gemini calls).
Compressed to 2.6 MB, sha256-verified byte-identical, documented in
`archive/v1-pilot/data/README.md`. Retired as a training asset, kept as provenance.

**Next:** stage 03 (`captions_qa` — the LLM-judge rubric and the four quality gates), then
the notebook runners. Stage 02 is ready to run against the real key whenever the Flickr8k
captions file is in place.

## 2026-08-18 (later) — vocab sweep: what it settled, and where I overspent

**Settled, and worth the time:**

- **UNK is not a tradeoff, it is an artifact of pruning someone else's vocabulary.** A
  byte-level BPE trained on our corpus has every byte in its alphabet: 0% UNK at every
  vocab size, with only sequence length varying. v1's frequency-pruned CLIP vocab sent
  8.0% / 4.4% / 2.1% / 0.9% of dev tokens to UNK at V = 1k / 2k / 4k / 9k.
- **Track A never needed CLIP's text vocabulary at all.** Its decoder trains from scratch
  and only the CLIP *vision* tower is used. The whole `local2clip` mapping was dead weight,
  and it is where the off-by-one came from.
- **V=9000 does not exist for this corpus.** The BPE trainer saturates at 7,667 types at
  `min_frequency=2`, so v1's 9,048-token vocabulary was larger than its corpus could support.
- **Per-token perplexity rises with V (14.6 -> 28.6) while bits/word falls.** Anyone tuning
  vocab on per-token loss -- which is what v1 logged -- picks the smallest vocab for entirely
  spurious reasons. Per-token loss is not comparable across vocabularies; bits per word is.

**Where I overspent.** I then relaunched the whole sweep at 34 epochs to reach convergence,
and killed it 8 epochs in when the user asked why we were doing it. They were right. The
measured spread was 1.92% bits/word across a 7.7x vocab range, and the decision is not
actually driven by that number -- it is driven by parameter budget. Even if convergence
widened the gap threefold in favour of V=7667, spending 71% of a ~5M-param model on token
lookup is wrong for a study about visual and emotional conditioning. **The experiment could
not change the decision, so finishing it was waste.** Worth asking "what outcome would change
my mind?" before spending the compute, not after.

**Vocab size is not pre-registered, and should not be.** It is held constant across all
conditions, so it cannot confound the ablation. It belongs in a tunable config, and Track A
runs cost ~4 min each on Kaggle, so it can be swept for nearly free alongside the real runs.

**A real bug the audit surfaced.** Adopting the official splits landed in `configs/data.yaml`
but *not* in `configs/prereg.lock.yaml` -- my edit matched on text that differed by a trailing
comment in the lock, and the script printed success without verifying the replacement. So the
two files contradicted each other: `split_source: official_flickr8k` in one,
`split_ratios: 80/10/10` in the other. **The lock test passed anyway**, because
`lock_violations` only compared keys present in *both* files and neither key was.

Fixed the lock, and closed the hole: `lock_violations(..., require_sections=("data",))` now
treats a locked key *missing* from a config that owns that section as a violation. A dropped
pre-registered setting is exactly as dangerous as a contradicting one. Two lessons: assert
that an edit actually applied, and a guard that only checks agreement is not a guard.

## 2026-08-18 (evening) — stage-02 probe: what is now settled

Matrix 1: 8 arms, 30 images for the A arms and 60 for the B arms, ~1,020 calls, ~$4.40.
Raw calls in `results/probe/probe_raw.jsonl`. The two model-comparison arms were still
running when these were banked; everything below comes from completed arms.

### Certain

**Batching works, and the schema is what makes it work.** This was the largest open
question -- $17.88 against $46.32 -- and it is settled.

| position | with schema | without schema |
|---|---|---|
| 0 | 5.00 [5.00, 5.00] | 4.17 [3.67, 4.58] |
| 1 | 5.00 | 4.17 |
| 2 | 5.00 | 4.15 |
| 3 | 5.00 | 4.17 |
| 4 | 5.00 | 4.17 |
| **overall** | **1500/1500 = 100%** | 1249/1500 = 83.3% |

The feared failure mode -- a 25-field response degrading toward the end -- does not
happen. And the loss without a schema is *uniform across positions*, so it is random
key omission rather than the model tiring; a constrained schema removes it entirely.

**Resolution is a non-decision.** 224 / 384 / 512 are indistinguishable on every
per-arm metric (recall 0.820/0.830/0.829, novel 0.531/0.533/0.527, reg-sim
0.337/0.341/0.345, words 18.1/18.5/18.1) *and* cost identically -- the image is a flat
**1,064 tokens at any resolution**, measured with the API's own counter, and 0 with no
image attached, which also confirms the image really is being sent. Downscaling saves
nothing. Picked 384 arbitrarily.

**The image earns its 1,064 tokens.** Text-only sticks closer to the source caption
(recall 0.865 vs 0.83) but adds less (novel 0.416 vs 0.53), runs shorter (16.0 vs 18.3
words), and crucially its **five registers are less distinguishable** (reg-sim 0.437 vs
0.34). Register distinguishability is the property the whole study depends on: if the
five emotions read as paraphrases, the ablation has nothing to detect.

**The rewritten prompt works.** Rejection 1.1-1.3% across all four A arms, and 100% of
captions inside the 8-24 word target with zero truncation. v1's failure mode was
rampant figurative leakage; this is a different regime.

### A metric I should not have trusted

My pairwise config-divergence measure is **noise-dominated at temperature 0.9** and has
no resolving power for config comparisons. Measured the noise floor directly by running
one identical config twice: **self-similarity 0.457 [0.427, 0.487]**, with only 1.7% of
captions identical. Every between-config number (A-224 vs A-384 = 0.4515, A-384 vs
A-512 = 0.4706, A-text vs A-384 = 0.4304) sits *inside* that interval.

The same prompt sampled twice gives two good captions with different wording, which
scores low on content-word overlap. The metric measures caption *identity*, which is
unstable under sampling; the per-arm aggregates measure caption *properties*, which are
stable because the noise averages over 750 captions.

Had the aggregates been silent and I leaned on divergence, I would have reported
"resolution matters materially" -- for two near-identical configs. **Second time on this
project a number looked informative and was not**, the first being per-token perplexity
across vocab sizes. Same shape of error: a quantity that varies for reasons unrelated to
what I was attributing it to. The general fix is to measure the noise floor *before*
believing a difference.

### Still open

- Does 100% batching completion hold on 3.6-flash and 3.1-flash-lite? (matrix 2, n=15)
- Model quality comparison, and whether lite is good enough to save $12 (matrix 1's
  last two arms).

### Operational

Concurrency 24 sustains 3.37 calls/s with zero errors. The whole 2.5 model family is
retired for this key, and `models.list()` still lists it -- **listing is not usability**,
so `--validate-models` makes one real call per candidate.

---

## 2026-08-19 — Part 0 prompt iteration: five versions, 250 images, $0.29

Part 0 (50 train images, 1,250 captions) came back at 100% completion, so the pipeline
was sound. The **captions** were not. Five prompt versions later the corpus is
measurably better on every axis that matters, and the reason the first one failed is
the most useful thing recorded here.

### The register table was writing the failures

An independent Sonnet agent, asked to judge 10 images by eye with no metrics, found
captions that contradicted their own photographs: a tent being set up by two visible
people described as *"A single tent sits alone on the vast ice, waiting to be set up"*;
five people sitting apart on a wall as *"A couple of friends lean into each other."* I
verified both against the images.

Measuring the whole 1,250-caption sample found the pattern was systematic — "alone" or
"empty" in **36%** of sad captions, "soft/gentle/graceful" in **41%** of romantic,
"bright" in 31% of joyful, "grips"/"edge" in 12% of tense.

Then the cause, in `prompt.py`'s own register table:

| register | what the prompt instructed | what came out |
|---|---|---|
| sad | "Leads with … what is **alone** or worn" | alone/empty, 36% |
| romantic | "Leads with **touch** … warmth **between subjects**" | soft/gentle 41%, invented couples |
| tense | "Leads with … **grip, edges**" | grips 12%, edge 12% |
| joyful | "Leads with motion, **colour**" | bright 31% |

And the few-shot example for `sad` read *"climbs the entryway stairs slowly, one step at
a time, alone"* — demonstrating both crutches as correct, on an image where neither is
true. **The model was complying.** Not laziness, not capability: my instructions.

**The rule extracted:** a register defined by *what content to lead with* is an
invitation to invent that content when the scene does not supply it. A register defined
by *manner* — rhythm, verb choice, restraint — is not. The whole table was rewritten as
manner.

### Each fix opened the next failure class

This is the part worth remembering, because it happened four times in a row.

1. **v2** closed solitude/pace/contact with named shortcuts and shown examples.
   Grounding lies 5.0% → 0.2%. But the registers went flat: *"The yard holds a bulldog,
   a sheep dog, and a boxer standing there."* Overlap rose 0.339 → 0.426.
2. **v3** added legitimate devices (foregrounding, verb precision, visible affect,
   sentence shape). Recovered distinctness, paired −0.036 [−0.060, −0.013].
3. **v4** added a stance device and banned padding. Distinctness improved again — and
   invented **light** (0.8% → 2.9%) and **posture** (0.2% → 0.9%) appeared instead:
   *"as the light fades"*, *"their heads bowed low"*, *"hands gripped tight"*.
4. **v5** closed those two. Defects 4.3% → 2.4%, best content recall of any version.

Root cause of the v4 regression, found by a blind Opus review of the prompt text: two
sections contradicted each other. `USE THE IMAGE FOR MOOD` granted the model "light,
weather, colour, crowding, posture, expression"; `WHAT MUST STAY TRUE` forbade adding
"weather, light, time of day, emotion on a face". **The model resolved my contradiction
by adding light.** Rewritten so the image informs word choice and never named content.

### The overlap metric was misleading on its own

Overlap kept rising as the captions got more honest, which read as a loss until the
Sonnet reference landed: **overlap 0.411 with keyword-rule 0.342**. High overlap, low
keyword-detectability is the *good* regime — registers separated by structure rather
than vocabulary, which is exactly what the study needs. v5 at **0.418 / 0.394** is in
that same regime. Read either number alone and you draw the wrong conclusion. **Third
time on this project a number looked informative in isolation and was not** (after
per-token perplexity and register divergence).

### Sonnet is not a viable generator, and that is settled

Measured: **45 minutes and 234k tokens for 10 images.** Scaled to 8,091 that is ~607
hours and ~190M tokens against **~$10.50** on Gemini batch. Images are only 4% of those
tokens, so nothing about resizing helps; the driver is turn count × accumulated
transcript, which makes *small* batches worse. On quality it was near-parity with
Flash-Lite on averages and clearly better only on hard cells, and two of its three
self-nominated best captions were near-verbatim reuse of worked examples from the
prompt. Its real value was as a distillation source and a reference ceiling.

### The strain flag: failed its own test, then measured something else

v5 asks the model to report, per register, how well that register fits the scene
(0 natural / 1 strained / 2 no honest reading exists). It was proposed as a way to turn
silent invention into a filterable signal. **It does not do that.** Defect rate by
self-reported strain: **1.6% at 0, 2.6% at 1, 2.4% at 2** — flat, no monotonic
relationship. On the criterion set in advance, it does not earn a place as a filter.

It does measure register difficulty, and that ordering is structured and corroborated:

| register | mean strain | strain-2 cells |
|---|---|---|
| joyful | 0.15 | 0 |
| tense | 0.85 | 12 |
| sad | 0.89 | 12 |
| humorous | 0.90 | 28 |
| romantic | **1.05** | **30** |

Romantic hardest, joyful trivial. That independently agrees with the human judge, which
said romantic "never actually reads as romantic in any of the 10 images". It also
**disagrees with the pre-registration**, which predicted `[tense, humorous]` as hardest;
tense is measurably not among the hardest. The prediction stands as registered and the
disagreement is reported — that is the mechanism working.

Caveat: 64% of cells came back strain 1, which smells like middle-anchoring. Trust the
ordering between registers, not the absolute levels.

### Moondream's v1 captions are useful after all — as a checker, not an input

The archived `v1_moondream_factual_captions.csv.gz` covers all 8,091 images with zero
missing and zero empty rows, mean 42 words. Verified against the images, it is reliable
on **subject count, identity, primary action, setting** and unreliable on **accessories,
small-object colour, and inferred activity** (it put a "red collar" and "black harness"
on a collarless dog). That split is exactly right for the failures that mattered — all
of which were core-fact failures — and it caught 5 of 5 of the judge's grounding errors.

Used only as an offline contradiction detector for the tables above. **Not** as prompt
input: that would put a VLM's output into the dataset, which is what
`neutral_source: human_annotations  # NOT a VLM` exists to prevent, and would leak its
hallucinated accessories into the captions.

### Operational: a dropped connection was discarding paid-for work

Two runs died with `httpx.RemoteProtocolError: Server disconnected` — once during batch
submission, once while polling. In both cases **the job was already submitted and
billing on Google's side**; only the collection was lost. I initially reported one of
these as having cost nothing, which was wrong: the job succeeded and its output sat
unclaimed. `scripts/recover_batch.py` now attaches to a job by name, matches responses
positionally against `config.image_ids` in the manifest (refusing to write on a count
mismatch), and retries transport faults with backoff. The 33-job full run will hit this.

### Still open

- **Impossible cells.** Some image × register pairs may have no faithful answer. v5
  gives the model a legal way to say so, but the flag does not identify *invention*, so
  the underlying question is unresolved and is a study-design decision, not a prompt one.
- **`romantic`'s definition.** Hardest register by every measure taken. Whether it is
  redefined or renamed touches the `emotions` list and therefore `emotion_id`.
- Five cells in v5 were rejected as "identical to the joyful rewrite" — a duplication
  mode that did not appear in v1–v4.
- One v5 image's response failed entirely (49/50 images, 1,225/1,250 captions).

---

## 2026-08-19 — the ceiling gate, tested early on 1,225 captions

§6 halts the study if the register classifier cannot recover the intended emotion from
the generated captions at >= 0.85. That assumption otherwise stays untested until the
corpus exists and both model tracks are built, so it was run on the v5 audit sample
(`scripts/check_gates.py`, 49 images, 1,225 cells, cross-validated by `image_id`).

    lexical shortcut anchor        0.394   (no model at all)
    TF-IDF + logistic regression   0.514
    DistilRoBERTa                  0.620   pre-registered ceiling >= 0.85
    floor (labels shuffled)        0.192   expected ~0.20  PASS
    artifact ablation gap         +0.000   PASS -- reads words, not punctuation

**The gate is undetermined, not failed.** Each fold fine-tunes DistilRoBERTa on ~980
examples across five classes; the pre-registered ceiling trains on the full 6,000-image
split, roughly 150,000 captions. A learning curve confirms the sample is the binding
constraint rather than the data:

    300 cells   0.410
    600 cells   0.480   (+0.070)
    900 cells   0.591   (+0.111)
    1,225 cells 0.612   (+0.021)

+0.202 for 4.1x the data. The final step decelerates, but with three folds over 12-49
images that is inside the noise, and 122x more data is still to come. **Log-linear
extrapolation of a four-point curve is not evidence** -- the honest statement is that
0.612 is a data-starved floor and the gate cannot be decided from this sample.

### Three independent instruments agree on which registers are broken

| register | classifier recall | generator strain | human judge |
|---|---|---|---|
| joyful | 0.722 | 0.15 (easiest) | "nails it" |
| humorous | 0.678 | 0.90 | "lands well" |
| tense | 0.645 | 0.85 | "lands cleanly" |
| sad | 0.563 | 0.89 | crutch-driven |
| romantic | **0.494** | **1.05 (hardest)** | "never reads as romantic" |

A trained classifier, the generator's own self-report, and a human-style judge shown only
images and text produce the same ordering. That is convergent validity, and it makes
"`romantic` is broken" a finding rather than an impression. The confusion matrix adds the
mechanism: `romantic` distributes 35% of its mass onto joyful (0.17) and sad (0.18), and
`sad`/`tense` swap at 0.20/0.18 -- exactly the taxonomy collapse §4.3 said to look for.

### The anchor and the ceiling are in direct tension, and the prereg does not say so

Under the v1 prompt the keyword rule alone reached **0.641** -- higher than what a trained
transformer extracts from v5 captions (0.620). A naive pipeline would therefore have
reported *better* emotion accuracy from *worse* data: 8.3% grounding defects, captions
asserting solitude over two visible people.

So the two pre-registered checks pull against each other. Suppressing stereotyped register
vocabulary is required for construct validity and it removes classifier-recoverable signal
at the same time. A caption set scoring 0.95 on the ceiling *and* 0.20 on the shortcut
anchor may not exist. **Nothing in the preregistration acknowledges this**, and it should,
because as written a pipeline is rewarded for stereotypy.

Stated against my own interest: I rewrote the prompt to suppress stereotypy, so presenting
the resulting drop in classifiability as a finding risks motivated interpretation. The
defence is that the trade-off is measured across five committed caption sets under a pinned
estimator, and that the v1 -> v5 comparison runs in the direction that makes my own work
look worse on the primary metric.

### Is 0.85 the right threshold?

It was set a priori with nothing calibrating it. For five-way affective classification of
one-sentence captions, 0.85 may exceed human-human agreement -- ArtEmis reports modest
inter-annotator agreement on emotion categories over a comparable label set. A gate no
achievable dataset can pass is a miscalibrated gate, not a finding about the data.

Recalibrating it against a human-agreement benchmark is legitimate **now**, pre-tag, with
the rationale logged. Doing it after seeing model results would be indefensible. This is
precisely the kind of decision the tag exists to separate.

### Still open

- The decisive curve point is **25,000 cells**, obtainable by generating one 1,000-image
  split for ~$1.30. It is needed for the study under either framing, so it is the cheapest
  way to settle the gate. At ~0.75 the gate is plausibly reachable; at ~0.65 it is not.
- Whether the study's primary question changes (see the reviewer critiques): a conditioning
  ablation over an LSTM and ClipCap is well-covered ground, and the targets are synthetic
  where FlickrStyle10K supplies human romantic and humorous captions on Flickr images.
  §1-§3 would be rewritten under the alternative framing, so `prereg-v1` stays untagged
  until that is decided.

### Ceiling at proper sample size: 0.737, and a correction

Re-run on the generated test split (`runs/gate-check-test/result.json`), 24,925 cells over
997 images, cross-validated by `image_id`:

    lexical shortcut anchor        0.403   (no model at all)
    TF-IDF + logistic regression   0.682
    DistilRoBERTa                  0.737   folds 0.737 / 0.736 / 0.737
    floor (labels shuffled)        0.198   PASS
    artifact ablation gap         +0.000   PASS

Per-fold spread of 0.001 across independent image splits, so this is settled. Against the
audit's 0.612 on 1,225 cells, **the earlier figure was measuring sample size**, exactly as
the learning curve suggested. The anchor moved only 0.394 -> 0.403, confirming the
stereotypy measurement is a property of the captions rather than of 49 images.

**Correction to an earlier claim in this notebook.** At 1,225 cells `romantic` had the
worst classifier recall (0.494) and I reported three independent instruments agreeing it
was the broken register. At 24,925 cells it recovers to 0.725 — mid-pack — and `sad`
becomes weakest at 0.661:

| register | 1,225 cells | 24,925 cells |
|---|---|---|
| humorous | 0.678 | 0.775 |
| joyful | 0.722 | 0.769 |
| tense | 0.645 | 0.755 |
| romantic | 0.494 | 0.725 |
| sad | 0.563 | **0.661** |

So the convergence was **two** instruments, not three: the generator's strain report and
the human judge still both rank `romantic` hardest, but the classifier no longer does once
it has enough data. The claim was over-stated and is withdrawn to that extent. `sad` now
leaks into `tense` (0.12) and `romantic` (0.13), which is consistent with the v5 prompt
having stripped `sad` of "alone"/"empty" — its most distinctive markers — without giving it
an equally separable replacement.

Twice now a number on the 1,225-cell audit has pointed the wrong way. **The audit sample is
adequate for measuring caption properties (defect rates, crutch words, the anchor) and
inadequate for anything requiring a trained model.**

The learning curve was killed before completing. It extrapolated from 25k cells to predict
152k, which the train split measures directly — redundant once the corpus exists, and not
worth the heat.

---

## 2026-08-19 (later) — three claims retracted, and the estimator bug behind them

A second review asked how the 25 keywords were chosen. The audit that followed invalidated
three findings recorded earlier today. All three failed the same way.

### The bug: the anchor is strongly sample-size dependent

Measured on the same generated captions, `keyword_rule_curve`:

    2,000 cells   0.441      25,000 cells   0.405
    5,000 cells   0.445      50,000 cells   0.371
   10,000 cells   0.427     100,000 cells   0.351

A keyword rule fitted on few images transfers well to held-out images from that narrow
pool; widen the pool and it degrades. So the anchor **falls ~9 points** across this range.

I had compared our anchor measured on one ~4,500-cell subsample against the human anchor
on 4,486 cells, and separately quoted our anchor at 24,925 cells, without noticing the two
were not comparable. Worse, the single subsample I used returned 0.419 where the mean over
ten image-disjoint subsamples is **0.440 (sd 0.007, range 0.427-0.449)**. The claim rested
on one unlucky draw.

### What was retracted

| claim as recorded earlier | corrected |
|---|---|
| "72% of the human-data advantage is keyword-recoverable" | **25%** |
| "the margins are identical, differing by 1.2 points" | they differ by **3.2 points** |
| "careful prompting made ours less stereotyped than human writing" | **not supported** — 0.440 vs 0.450 is 1.5 pooled sd |

### Corrected comparison, matched at 4,486 cells

|  | accuracy | anchor | margin |
|---|---|---|---|
| human | 0.726 | 0.450 | 0.276 |
| ours | 0.683 | 0.440 (sd 0.007) | 0.244 |
| difference | −0.043 | −0.011 | **−0.032** |

**The accuracy gap largely survives the anchor correction.** So it is mostly a real
difference in register signal, not a metric artifact — which points the *opposite* way from
the thesis drafted this morning. P2 ("the margins are the same") is not supportable as
written.

### The decline is estimator-wide, not corpus-specific

Same mapping, two sizes, so the mapping cannot explain it:

    4,486 -> 20,000 cells:   human -0.034      ours -0.031

Both corpora decline at the same rate. Two consequences: matched-n comparison **is** valid
because the bias is shared, and the inflation is a property of the estimator that would
affect anyone using a keyword baseline. FlickrStyle10K is 7K images and SentiCap 2,360 --
exactly the scale where the inflation is largest and exactly where such baselines are
computed. That is a genuine methodological finding, measured across two orders of
magnitude, and it is more solid than the claim it replaces.

### A second, worse problem: the anchor comparison is not identifiable

Our anchor does not depend on the trait mapping; the human anchor does, heavily. At 4,486
cells:

    strict mapping (1 trait per register)    human 0.450   ours 0.440   ours LESS stereotyped
    grouped mapping (traits merged)          human 0.352   ours 0.441   ours MORE stereotyped

**The sign of the difference flips with an arbitrary choice.** Grouping merges
heterogeneous traits and destroys the human classes' keyword coherence. There is no
principled unique grouping, so "which corpus is more lexically stereotyped" is not
answerable from this data. The strict 1:1 mapping is the more defensible of the two and is
what the paper will privilege, with this sensitivity reported rather than buried.

### Fixes applied

* `keyword_rule_matched` requires `n_subsamples >= 2` and returns mean, sd, min, max and
  every value. A bare point estimate is no longer obtainable when comparing corpora.
* `keyword_rule_curve` reports the anchor against n, so the dependence is visible by
  construction rather than discovered after a claim is built on it.
* Both documented at the top of `anchors.py`, with the retraction referenced.

### The pattern, now four for four

Per-token perplexity across vocab sizes; register divergence inside its own noise floor;
the 0.612 ceiling; and now the anchor. **Every one was a point estimate quoted before its
sampling distribution was checked.** The procedural fix is the code change above: the
functions that feed claims now return distributions, because remembering to check has
failed four times.

## 2026-08-23 — first GPU runs: the instrument works, the unpaired arms are at floor

Sixteen of the 36 registered runs are done: `S_paired5` fold 0, and all five folds of the
three unpaired arms. Every number below is the primary metric — frozen classifier
`572eaa80`, top-1, on generated held-out captions — with the keyword anchor recomputed on
those same captions at that same n, as §4 requires.

### Operational: P100 does not work, and T4 does

The first attempt died on the first `F.linear` with `CUDA error: no kernel image is
available`. Kaggle's P100 is compute capability sm_60; the installed PyTorch ships sm_70
and up. Nothing to do with the code — switching the accelerator to T4 x2 fixed it
unchanged. T4 also has fp16 tensor cores, so `configs/model.yaml`'s `amp: fp16` buys real
speed there; on P100 it would have run at roughly fp32 rate.

Measured cost: `S_paired5` one fold (32,150 train / 8,085 decoded) in **22.2 min**; an
unpaired fold (3,517 / ~880) in **2.4 min**. Extrapolating over the 36 runs gives ≈17 GPU
hours, which fits inside a single 30 h week rather than the two that were budgeted.

`notebooks/02_train_arm.ipynb` now takes a list of `(arm, fold, negative_control)` and runs
them in one session, skipping any whose `predictions.jsonl` already exists. A non-zero exit
is collected and re-raised at the end instead of aborting the batch — the runs are
independent, and losing eleven good ones to a crash in the fourth is not a failure worth
having twice.

### The instrument works

`S_paired5` fold 0:

    accuracy  0.7553   95% CI [0.7458, 0.7649]
    anchor    0.6719   (same captions, n=8,085)
    MARGIN   +0.0834   chance 0.2000

The margin clears the registered `smallest_effect_of_interest` of 0.07 with the governing
MDE (0.063 at seed SD 0.02) below it. 7,997 unique captions of 8,085, no empties. So the
architecture learns register, the frozen classifier reads it, and the anchor recomputation
behaves. None of that was established before today, and all of it was assumed by the design.

### All three unpaired arms are at floor

Five folds each, 4,390 cells per arm:

    arm            acc mean  acc sd   anchor   margin   margin sd
    S_unpaired       0.2089  0.0127   0.2101  -0.0012      0.0291
    V1_unpaired      0.2167  0.0136   0.2415  -0.0247      0.0072
    H_unpaired       0.2457  0.0295   0.2491  -0.0034      0.0165

Chance is 0.2000. The arms clear it by 0.89, 1.67 and 4.57 points. Against the registered
predictions, in points:

    P1   H acc - S acc          +3.68   needs > 0, MDE 6.3      direction right, underpowered
    P2   H margin - S margin    -0.22   needs >= +7.0           condition met, but at floor
    P6a  V1 acc - S acc         +0.79   needs > 0               direction right, trivial
    P6b  V1 margin - S margin   -2.35   needs <= -7.0           direction right, far short

Observed fold SD (0.013-0.030) maps onto the 0.01-0.03 rows of the registered MDE curve,
i.e. MDE 4.2-8.8. Every effect above sits under that, so `underpowered_rule` governs all
four: **reported as underpowered, not as evidence of no difference.**

### Why the registered P6 test could not have worked

P6's mechanism rests on a 21-point anchor gap between the corpora — v1 0.718 against ours
0.507 at 4,486 cells. In the captions these models actually generated, that gap is:

    S_unpaired  anchor 0.2101
    V1_unpaired anchor 0.2415     gap: 3.1 points

At 4,390 cells neither model learned enough register vocabulary for its corpus's stereotypy
to survive into the output. P6 predicts that accuracy tracks the corpus anchor; if the
generated text carries almost none of the corpus-specific anchor signal, the mechanism has
no room to express itself regardless of whether P6 is true.

This is a measurement-validity problem with the registered test, not evidence about the
hypothesis, and it is the fifth instance of the same pattern in this notebook: a quantity
was assumed to transfer from the corpus to the model's output without that transfer being
measured. Contrast `S_paired5`, whose generated captions anchor at **0.6719** — at 40,235
cells the model does reproduce corpus register vocabulary, so the mechanism has somewhere
to live. The confirmatory pair `S_paired5` vs `V1_paired5` is therefore the load-bearing
version of this contrast. P6 is *not* registered over it, so that reading is exploratory
and must be labelled as such; the confirmatory claim on that pair stands on its own terms.

### The floor was predicted before it was observed

After `S_unpaired` fold 0 came back at chance and before any other unpaired run existed, the
prediction on record was that `V1_unpaired` and `H_unpaired` would also sit at floor, and
that the pair would only be uninformative if *both* did — since P6 predicts asymmetry, a
floor at S alone would have been P6's predicted pattern rather than a null. Both came back
at floor. Logging the ordering because a prediction written after the fact is not one.

### Still open

* **`H_unpaired` is decoding degenerately.** Unique-caption rate is 0.58-0.64 across all
  five folds, against 0.98-1.00 for S and V1, at 7.2-7.6 mean words against 16 and 22. It
  also carries the largest fold SD (0.0295, twice the others). A model collapsed onto a few
  high-frequency register-typical strings can beat chance without conditioning on anything,
  and the anchor cannot catch it because the anchor is computed on that same degenerate
  text. **P1's +3.68 should not be quoted until this is ruled out.** Personality-Captions
  text is genuinely shorter, so part of the gap is the corpus, but not 40% duplicates.
* **P6b is directionally consistent in a way the mean hides.** V1's margin is negative on
  5/5 folds (sd 0.0072) while S's straddles zero (3 positive, 2 negative, sd 0.0291). V1
  systematically scores below its own keyword anchor and S does not — P6's predicted
  pattern, reproducibly, at about a third the registered effect size.
* **Register collapse, all three arms.** Fold-0 recall is dominated by one register per arm
  (S: joyful 0.503; V1: sad 0.451; H: joyful 0.594) with humorous at 0.012-0.046
  everywhere. Whether this survives at paired5 scale is the thing to watch — `S_paired5`
  fold 0 already spreads much better (0.588-0.923), which suggests it is a data-volume
  effect and not a taxonomy failure.
* Negative controls not yet run: the manipulation-check gate is untested on every arm.

---

## 2026-08-24 — the sweep completes, the margin gets an interval, and a rater finds the anchor by eye

> **Partly retracted the same day.** `S_paired25`'s anchor below was measured at the wrong n, so
> its margin here (+0.0608) and the `S_paired25` vs `S_paired5` comparison are both wrong. See
> the entry of 2026-08-24 (later). The rest of this entry stands.

All **36 registered runs** are in. `emocap-s25a`, `emocap-s25b` and `emocap-v1` finished on
Kaggle; every arm was re-scored so its `cells.jsonl` carries a per-cell anchor. All six
negative controls pass (0.1821–0.2131 against a 0.25 cap), all 36 probe verdicts are
recorded, none excluded. `compare_arms.py` printed FINAL rather than PROVISIONAL for the
first time.

| arm | accuracy | anchor | margin |
|---|---|---|---|
| S_paired25 | 0.9119 | 0.8511 | +0.0608 |
| V1_paired5 | 0.8748 | 0.8143 | +0.0605 |
| S_paired5 | 0.7685 | 0.6826 | +0.0860 |
| H_unpaired | 0.2457 | 0.2491 | −0.0034 |
| V1_unpaired | 0.2167 | 0.2415 | −0.0247 |
| S_unpaired | 0.2089 | 0.2101 | −0.0012 |

### The margin had no interval, and that was hiding two things

The study claims the margin. Every registered magnitude criterion — P2, P3b, P6 — is written
in margin points. But the anchor was a whole-corpus number, so a margin *difference* could
only be reported as a mean over five folds with the fold spread beside it. A criterion of
"≥7 points" cannot be judged against a quantity with no interval on it.

`keyword_rule_cell_scores` now decomposes the registered estimator to the cell — same keyword
fitting, same CV by image, same fractional tie credit, walking the same generator as the point
estimate, which reproduces the stored anchors exactly. Two things changed once the margin
could be bootstrapped:

* **S_unpaired vs V1_unpaired, P6's own registered pair, is significant.** Margin gap +0.0232,
  CI [+0.0026, +0.0436], p 0.027. At fold level (mean 0.0235, sd 0.0286) it read as noise.
  P6 needs ≥7 points and gets 2.3, so it is **not confirmed** — but "a real gap, smaller than
  predicted" is a different report from "nothing detected", and the previous write-up said the
  latter.
* **P3b lands at +0.0586, CI [+0.0418, +0.0747].** Clearly above zero, best estimate *below*
  the registered 7 points, and the interval cannot rule out that the true value meets it.
  Reported as exactly that — neither a confirmation nor a null.

**All four confirmatory margin gaps clear zero and none reaches the 0.07 criterion.** The
criteria were set at 7 points because that is what the power analysis could detect; the
effects that exist are 2–6. That is a finding about the design and belongs in the paper
rather than in a footnote.

**P2 is falsified on its registered condition.** H_unpaired beats S_unpaired by 3.75 accuracy
points, but its margin is −0.0021, CI [−0.0232, +0.0185], p 0.84, and P2 registered "margin
difference ≤ 0" as falsification. The human corpus's advantage *is* explained by lexical
stereotypy — which is what P2 predicted it would not be.

### More data raised accuracy and lowered conditioning, inside one arm family

`S_paired25` vs `S_paired5` is a pure source-caption-count comparison on the same images.
Five times the captions moved **accuracy +0.1434** and **margin −0.0252**, CI [−0.0299,
−0.0207], negative on 5/5 folds. The model got better at the metric and worse at what the
metric is meant to measure, because the keyword anchor rose faster (0.6826 → 0.8511) than the
accuracy did (0.7685 → 0.9119). This is the study's own argument demonstrated without leaving
one corpus, and it is the cleanest version of it anywhere in the data.

### A rater found the lexical shortcut by reading, and it is one arm

The first human evaluation came back (150/150 complete, 8/12 attention checks). Unprompted,
the rater said some *joyful* captions seemed joyful mainly because they contained the word
"joy" — and that it was some, not the majority.

Measured across all 8,047 images:

| arm | joyful captions containing joy / joyful / joyous |
|---|---|
| S_paired25 | **0 / 8,047** |
| S_paired5 | **0 / 8,047** |
| V1_paired5 | **6,100 / 8,047 = 75.8%** |

Entirely the archived v1 pilot corpus. Our own captions never use the word once. `joyful`,
`joyfully` and `joyous` are all in V1's top-25 keyword list, so **the anchor had already
quantified this** — it is a large part of why V1's anchor is 0.8143 against ours at 0.6826.
The rater's "some, not the majority" was exact for what they saw: 10 of their 30 joyful items
carried the word and all 10 were V1.

A reader with no idea which system wrote what, and no knowledge of the anchor, located by eye
the precise property the anchor exists to measure. That is external validation of the
instrument, and it is worth reporting as such.

**And V1 is where the three measurements disagree, in the informative direction.**

| arm | classifier accuracy | human tone-match |
|---|---|---|
| S_paired25 | 0.9119 | 4.12 |
| V1_paired5 | 0.8748 | **3.68 — last** |
| S_paired5 | 0.7685 | 4.00 |

The classifier ranks V1 second, ahead of `S_paired5` by 10.6 accuracy points. The human ranks
it last on tone. The keyword stuffing that wins over a text classifier reads as *worse*
writing to a person — the thesis, measured a third independent way after the anchor and the
margin.

### Still open

* **One rater, so Krippendorff's alpha is undefined.** Every human number above is one
  person's impression with a bootstrap interval around it. Nothing here is confirmatory until
  raters two and three return, and the arm ordering could move.
* **The `wrong_register` attention check may be harder than intended.** All four of the
  rater's misses were that type, all scored exactly **3**, while all six `wrong_image` checks
  passed. That is a scale-usage pattern, not inattention: a mis-registered caption is not
  *unrelated*, so "neither" is defensible. The ≤2 pass rule was fixed before any data existed
  and is **not** being changed — but the next two raters should be watched for the same
  pattern, and if it repeats it is a property of the check rather than of the raters.
* **S_paired25's anchor and its accuracy are measured on different amounts of text.**
  `score_arm.py` keys captions by (image, register), so with five source captions per key only
  the last survives: the anchor comes from 8,050 captions while accuracy uses all 40,425. The
  anchor is strongly n-dependent, so this is not cosmetic — a larger n would lower the anchor
  and raise the margin. Left alone because changing it moves a registered figure; flagged in
  `compare_arms.py` and needs a decision before the paper.
* **Secondary metrics are still not computed on any arm.** CIDEr-D, BLEU-4, CLIPScore and
  distinctiveness are registered and absent. `S_paired25` is about to be described as a
  0.91-accuracy emotion captioner and there is no evidence yet that its captions are good
  *as captions* — the human grounding score of 3.74 from one rater is currently the only
  thing pointing at that question.

---

## 2026-08-24 (later) — RETRACTION: "more data lowers the margin" was an anchor bug

**The entry above is wrong on one point and it needs saying at the top rather than in a
footnote.** It reported `S_paired25` vs `S_paired5` as accuracy +0.1434 with margin
**−0.0252**, negative on 5/5 folds, and called it "the study's own argument inside one arm
family — the model got better at the metric while getting worse at what the metric is meant
to measure". That finding does not exist. It was an artifact of measuring `S_paired25`'s
anchor on the wrong number of captions.

`score_arm.py` keyed the anchor estimator's records by `image_id` with captions stored as
`{register: text}`. `S_paired25` holds five source captions per (image, register), so four of
every five were overwritten: its accuracy was measured on 40,425 captions per fold and its
anchor on **8,050**. The anchor falls as n grows — this notebook recorded 0.440 at 4,486
cells against 0.332 at 201,900 on 2026-08-19 — so the anchor was too high and the margin too
small. Correcting it is `docs/deviations.md`, 2026-08-24.

| | before | after |
|---|---|---|
| S_paired25 accuracy | 0.9119 | 0.9119 — unchanged |
| S_paired25 anchor | 0.8511 @ n=8,085 | **0.7722 @ n=40,235** |
| S_paired25 margin | +0.0608 | **+0.1397** |

Only comparisons touching `S_paired25` moved, which is the check that the diagnosis is right:
`S_unpaired` vs `V1_unpaired` stayed at +0.0232 and `S_paired5` vs `V1_paired5` at +0.0254,
to four decimals. Re-scoring `S_paired5`-f0 after the change reproduced its anchor (0.6719)
and margin (+0.0834) exactly, because that arm never had duplicate keys.

**What replaces the retracted finding.** More source captions raise accuracy **and** the
margin: +0.1434 and **+0.0535**, CI [+0.0487, +0.0585], positive on 5/5 folds. Scaling the
data improves conditioning rather than degrading it.

**What survives untouched is the level, not the trend.** 0.9119 accuracy against a 0.7722
anchor means **77 of those 91 points are reachable by keyword spotting with no model at all**.
That is the study's claim and it is unaffected. What is gone is the stronger version — that
scaling actively makes conditioning worse — which was never predicted, was believed for about
six hours, and came from a bug.

### P3b is confirmed, and that is the first criterion this study has met

`S_paired25` vs `S_unpaired`: margin **+0.1360**, CI [+0.1190, +0.1528], p 0.0001, significant
after Holm on both accuracy and margin. The registered criterion is 0.07 and this clears it
by a factor of two. With P3a already confirmed at +69.9 accuracy points, **P3 is confirmed on
its registered comparison.**

The correction moved a registered prediction from "misses its criterion" to "meets it", so it
is reported with both numbers, and the three facts that make that defensible are in the
deviations entry: the rule the fix restores was registered in advance, the direction of the
change was predictable from this notebook's own 2026-08-19 measurement, and the bug was
flagged in `compare_arms.py` and in the Still-open list before it was fixed.

### The baselines, and one of them is unusable by construction

| baseline | accuracy | margin | unique captions |
|---|---|---|---|
| `BASE_text` on `S_paired5` cells | **0.2188** | −0.0211 | 4,608 |
| `BASE_prior`, every arm | 0.8000 | −0.2000 | **5** |

Frozen GPT-2 given the source caption and the register word, with no image and no training,
scores **0.2188** where `S_paired5` scores 0.7685. So no part of the arms' performance is
reachable from the source text alone — the obvious reviewer question, answered.

`BASE_prior` is degenerate exactly as predicted, and its own numbers prove it: with five
captions in existence the keyword anchor scores **1.0000**. An anchor of 1.0 is not a
measurement. Publish the five captions and never the accuracy.

### Still open

* **The 6 `S_paired_matched` runs have no probe verdict**, so `compare_arms.py` prints
  PROVISIONAL. All six changed 100% of their captions with the image blanked.
* **`S_paired_matched`'s failure is not a visual-dependence failure and must not be read as
  one.** Its self-BLEU is 0.8631: with the image present it writes one caption per image and
  reuses it across all five registers. The probe asks a different question — whether captions
  change when the image is *removed* — and it can pass while the arm is degenerate this other
  way. Two independent failure modes, and only the second one is what sank that arm.
* **`S_unpaired_scaled` is running**, 40,235 cells at one register per image. It is the arm
  that separates pairing from volume, and its reading was fixed before the run.
* Human evaluation is at 2 of 3 raters. Krippendorff's alpha is computable for the first time
  — grounding **+0.617**, tone-match **+0.457** — both below the 0.667 conventional threshold
  for tentative conclusions, so the per-arm tone means carry little weight yet.

---

## 2026-08-25 — the human evaluation closes, the ablation clears, and a bug that ate three runs

### The registered human evaluation is complete

Three raters, 150 items, blinded and randomised, 12 attention checks, run once on final
models. All three were external and none had any connection to the study.

| scale | Krippendorff's alpha, ordinal |
|---|---|
| grounding | **+0.684** |
| tone-match | **+0.579** |

The conventional bars are 0.800 for firm conclusions and 0.667 for tentative ones. Grounding
clears the second; tone-match does not. **That split is itself worth reporting**: grounding is
close to a factual judgement and tone is closer to taste. It also lands on the study's own
assumption — the primary metric is a classifier reading tone from text, and three blind
humans managed only 0.58 agreement on the same judgement.

| arm | tone-match | grounding |
|---|---|---|
| S_paired25 | 4.19 [3.92, 4.43] | 3.93 [3.58, 4.25] |
| S_paired5 | 4.15 [3.91, 4.37] | 3.38 [3.02, 3.72] |
| V1_paired5 | **3.88** [3.61, 4.13] | 3.61 [3.26, 3.95] |

**V1 is last on tone at every rater count** — one, two and three — by about 0.3 against both
S arms. That is now the fifth independent measure ranking V1 last, after the keyword anchor,
CIDEr-D, CLIPScore and the punctuation ablation. The classifier still ranks it second. Five
measures agreeing, and the one that disagrees is the one the study is arguing about.

V1's grounding position moved between one rater and three, from last to middle. The
single-rater table was not settled and should not have been read as though it were.

### One rater carries almost all of the disagreement, and is retained

Attention checks: IM 12/12, MA 12/12, MM 8/12. Leave-one-out, **post-hoc**:

| dropped | tone-match | grounding |
|---|---|---|
| IM | +0.4566 | +0.6171 |
| MA | +0.4004 | +0.5555 |
| **MM** | **+0.8705** | **+0.8978** |
| (none) | +0.5789 | +0.6841 ← registered |

Without MM both scales clear the 0.800 bar. MA and IM are close to interchangeable.

**MM stays in, and the reason is not politeness.** They passed the attention-check rule fixed
before any data existed, which flags only below 50%. Dropping the rater who turns out to
disagree most, *after* discovering they disagree most, is sampling until the number improves.
Every rater is dropped in turn in the table above precisely so that reading is available to a
reader without being privileged by the analysis.

**The obvious explanation was tested and does not hold.** MM finished faster than MA, which
suggested haste. The timestamps say otherwise:

| rater | total | median per item | answers under 2s |
|---|---|---|---|
| MM | 26.1 min | **8.85 s** | 0 |
| MA | 35.3 min | 8.60 s | 0 |
| IM | **25.2 min** | **5.27 s** | 0 |

IM was the *fastest* rater and agrees most; MM has the *longest* median per item of the
three. MM's shorter total than MA comes from MA taking occasional long pauses, not from MM
rushing. Nobody answered anything in under two seconds. There is no haste signature in this
data and the hypothesis is dropped.

What is left is that MM read the tone scale differently: all four of their missed checks were
wrong-register items scored exactly **3** — "neither" where the others committed to a low
score — at a median of nearly nine seconds each. That is a scale-usage difference, not
inattention.

Since all three were blind and unconnected, the honest reading is that MA and IM happening to
share an intuition is as plausible as MM being the odd one out. Two raters agreeing is
evidence that they are similar, not that they are right. **0.579 and 0.684 stand as the
study's numbers.**

### The artifact ablation clears, on the instrument that scores the study

The registered check had never actually run: `runs/classifier/report.json` reports a gap of
exactly 0.0000 because it was computed on TF-IDF, whose tokenizer discards punctuation before
it sees the text. Redone on the frozen DistilRoBERTa, where stripping changes 98–100% of the
input:

| level | raw | stripped | gap |
|---|---|---|---|
| instrument, 5-fold CV | 0.8279 | 0.8214 | **+0.0065** |
| arms, worst case (`V1_paired5`) | 0.8747 | 0.8366 | **+0.0381** |
| arms, every other arm | — | — | −0.0009 to +0.0123 |

**The instrument does not read punctuation** — 0.65 of a point. The raw accuracies stand as
primary and §4.1's fallback is not triggered. The exception is `V1_paired5`, where nearly four
points of accuracy is formatting: the sixth measure on which V1 comes last.

### A path-formatting bug destroyed three completed runs

`Path.relative_to` raises when the path is outside the root, and every script here has an
output flag that can point anywhere — on Kaggle they always do, since outputs go to
`/kaggle/working` while the clone sits in `/kaggle/working/EmoCap`. Used only to shorten a
path for a log line, it took down:

* twenty complete baseline runs, reported as failures by a print statement;
* a P4 run that had trained and saved judge S (own-corpus CV 0.8565);
* this artifact ablation, after all five folds had finished **and the results file had been
  written**.

Each time the work was done and the crash was in the line formatting the path. The ablation
numbers above were recovered from the kernel log rather than recomputed, and
`results/artifact_ablation.json` records that provenance.

It was fixed twice as one-offs, which is exactly why it happened a third time. There is now
one `emocap.runtime.rel()` and no `relative_to(ROOT)` anywhere in `scripts/`.

The same run also exposed a second one: `finetune_classifier` and `train_final_classifier`
resolved their device as "mps if available else cpu". Correct on the laptop; on a T4 there is
no MPS, so both trained DistilRoBERTa on the host CPU with an idle GPU beside them. That is
why P4 appeared hung after 800 seconds without finishing a fold.

### Still open

* **P4 has never completed.** Two attempts: one wedged on MPS overnight, one crashed on the
  path bug after training judge S. Both bugs are fixed; it needs one clean run.
* **Two human items remain**, and both are narrower than the evaluation that just closed:
  the legibility sample widened to >=300, and 250 hand-labelled captions not written by
  Gemini. P5 waits on the first.

---

## 2026-08-25 (later) — P4 completes, and every registered prediction but one is resolved

Third attempt. The first wedged on MPS overnight, the second crashed formatting a path, the
third ran clean on a T4 in about 17 minutes.

### P4 is falsified on its registered comparison, and the comparison is vacuous

    S captions under the H-judge   0.2027
    H captions under the S-judge   0.2027
    asymmetry (S drop - H drop)   -0.0237  -> "symmetric, or reversed" -> FALSIFIED

The two foreign-judge numbers are **identical**. The −0.0237 comes entirely from the
own-judge baselines differing, 0.2415 against 0.2178, which is a gap between two
chance-level numbers. All three arms P4 is registered over sit at 0.20–0.24. So this is
falsified on noise, for the same structural reason as P1, P2 and P6: at 4,390 cells the
captioners learn no register, and there is nothing for a judge to transfer.

### The control is informative, and the asymmetry is real — in the corpora, not the models

| judge | own corpus | other corpus | drop |
|---|---|---|---|
| S-trained | 0.8565 | 0.3157 on human text | **54.1 pts** |
| H-trained | 0.7241 | 0.4294 on synthetic text | **29.5 pts** |

A judge trained on our synthetic captions loses 54 points moving to human text; a judge
trained on human text loses 29 moving to synthetic. **Synthetic register is far easier to
read than human register**, which is the same fact the keyword anchor reports (0.507 for
ours against 0.450 for human) and TF-IDF reports (0.81 against 0.61), now measured a third
way with a transformer.

So P4's *mechanism* holds in the data while P4 *as worded* — a claim about captioners —
fails, and it fails because the captioners it names are at floor. That is the fourth
prediction closed by the same diagnosis, and at this point the diagnosis is itself a finding:
**a 4,390-cell arm cannot support any provenance comparison in this design.**

**Reporting these as drops rather than raw accuracies is load-bearing.** Raw, the H-judge
looks better at foreign text (0.4294 against 0.3157) — but it is 13 points worse on its own
corpus. `p4_transfer.py` prints a warning when the two judges differ by more than 10 points
for exactly this reason, and it fired. Without the drop framing the finding inverts.

The judges are also the registered `per_arm_classifier` robustness item, which closes with
this run. The frozen instrument was not touched; both judges are separate and hashed.

### Every registered prediction, resolved

| | outcome |
|---|---|
| **P1** | H beats S by 3.75 accuracy points, below the 0.07 smallest effect of interest — **underpowered**, reported as such, not as a null |
| **P2** | **falsified** on its own condition: margin difference −0.0021, CI [−0.0232, +0.0185] |
| **P3a** | **confirmed**, +69.9 accuracy points |
| **P3b** | **confirmed**, margin +0.1360, CI [+0.1190, +0.1528] — the only registered magnitude criterion met |
| **P4** | **falsified**, and vacuous: symmetric at chance |
| **P5** | **open** — blocked on the widened legibility sample |
| **P6** | **not confirmed**: both directions right, magnitude 2.3 points against ≥7 |

P5 is the last one, and it is the only registered prediction still unmeasured.

### A fourth pinning failure, and the guard that should have existed from the start

The second P4 crash was the `relative_to` bug again — not because the fix was wrong, but
because the notebook pins its checkout to the predictions dataset's provenance, that
provenance was `ac10078`, and the fix landed in `8a43ea5` two commits later. The dataset was
never re-pushed, so the notebook faithfully checked out code predating its own fix.

That is three distinct pin failures: a missing script, missing CLI flags, and a bug fixed
after the last push. The `--help` guard added for the second cannot see internal changes.
Both notebooks now fetch and ask `git log COMMIT..origin` for the specific scripts they run,
refusing to start if anything touched them, naming the commits and the one command that
fixes it. Verified against the failing state: it reports `8a43ea5`, exactly the missing
commit.

Pinning to a dataset's provenance is right for reproducibility and wrong every time code is
fixed afterwards. The guard does not remove that coupling; it makes it visible in ten seconds
instead of eight minutes.

---

## 2026-08-25 (evening) — the off-distribution check passes: the classifier is not a Gemini detector

148 of the registered 250 hand-written captions, from three blind writers who had never seen
the corpus. The registered criticism was that the frozen classifier learned Gemini's house
style rather than emotional register — it saw 13,170 captions, 8,780 of them Gemini's, and
every arm was then trained on Gemini text, so the whole study would look identical either way.

| | |
|---|---|
| in-distribution CV, its own pool | 0.8279 |
| **148 human-written captions** | **0.7770** [0.7095, **0.8446**] |
| keyword anchor, same captions | **0.2782** |
| chance | 0.2000 |

**The interval contains the in-distribution figure**, so there is no detectable degradation.
A style detector meeting unfamiliar text would land near chance; this lands 58 points above
it.

**The same data answers a second accusation.** On these captions a top-25 keyword rule
reaches 0.2782 — barely above chance — while the classifier reaches 0.7770, a margin of
+0.4988. So it works on text where the lexical shortcut essentially does not exist, which
rules out "it is only a keyword detector". Two criticisms, one measurement.

That also says something the registered design never asked: **the keyword-reachability this
whole study measures is a property of machine-generated captions, not of the task.** The
anchor is 0.7722 on `S_paired25`'s output and 0.2782 on human captions describing the same
kind of photograph. The shortcut lives in the generator, not in the register taxonomy.

### What cannot be claimed

The interval runs [0.7095, 0.8446]. It contains 0.8279, but also values roughly twelve points
lower, so a modest real drop is compatible with this sample — it simply cannot be detected at
n=148. The point estimate sits 5.1 points below and is the number to quote as the
instrument's **measurement error**, not as a demonstrated gap.

It matters little for the study: every arm comparison uses the same instrument on the same
kind of text, so a uniform measurement error cancels out of a comparison. It bears only on
how absolute accuracies are described, and there the honest framing was already the margin.

### Three writers, and the one that looked wrong was not

| chunk | n | accuracy |
|---|---|---|
| 1 | 50 | 0.8400 |
| 2 | 50 | **0.6400** |
| 3 | 48 | 0.8542 |

At two chunks the pooled figure was 0.7400 and looked like a real drop. The third settles it:
chunks 1 and 3 agree within a point, chunk 2 sits twenty below. **Chunk 2 is retained** — it
is dropped from nothing, for the same reason MM was retained in the human evaluation.
Removing the observation that moves a number the wrong way is not an analysis.

Chunks 1 and 2 were the same writer, so that gap is within-person. The timestamps rule out
haste: median 42.1 s per caption in chunk 1 and 42.6 s in chunk 2, 100 captions back to back
over 1.4 hours at an identical pace. That is the second time here a haste hypothesis has died
on the timestamps, after MM's.

### The hard registers are the same ones humans disagree about

| register | ch1 | ch2 | ch3 | all |
|---|---|---|---|---|
| joyful | 1.00 | 1.00 | 1.00 | **1.00** |
| humorous | 1.00 | 0.80 | 0.70 | 0.83 |
| sad | 0.80 | 0.60 | 0.80 | 0.73 |
| romantic | 0.70 | 0.40 | 1.00 | 0.68 |
| tense | 0.70 | 0.40 | 0.80 | **0.63** |

`joyful` is perfect for all three writers. `tense` and `romantic` are the weak pair — and they
are **the same pair the human raters agreed least on**, the pair that held tone-match alpha to
0.579 while grounding reached 0.684. Two independent measurements, one about writing and one
about reading, converging on which registers are genuinely hard.

### A verdict line that contradicted its own interval

`score_offdist.py` printed "drops materially" at 0.7770 [0.7095, 0.8446] — an interval that
contains 0.8279. It was thresholding the point estimate and ignoring the confidence interval
it had just computed. Fixed to consult the interval first. That sentence would have gone into
the paper saying the opposite of what the data shows, and it is the third reporting bug this
week caught by reading output rather than trusting it.

### Two skipped captions are a good sign, not a shortfall

The third writer wrote 48 of 50 and skipped two, saying they could not work out what to write.
Registers stay at 30/30/30/30/28. A writer who declines rather than forcing a caption they do
not believe in produces a cleaner test set than one who fills every box.

### Still open

* **Closed at 148 of the registered 250**, decided 2026-08-25 and logged in
  `docs/deviations.md`. Completing it would tighten the interval from about ±0.067 to ±0.052
  and cannot change the verdict, since 0.8279 sits comfortably inside either. The order
  matters and is on the record: at 100 captions the pooled figure had *fallen* to 0.7400 and
  collection continued; it was stopped only after the third chunk lifted it to 0.7770.
* **P5** is measurable but not like-for-like — the human read corpus captions and the
  classifier read generated ones. The judge half of that gap is closed; the text half is not,
  and the paper must say so.
