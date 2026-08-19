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
4. **v5** closed those two. Defects 4.3% → 2.3%, best content recall of any version.

Root cause of the v4 regression, found by a blind Opus review of the prompt text: two
sections contradicted each other. `USE THE IMAGE FOR MOOD` granted the model "light,
weather, colour, crowding, posture, expression"; `WHAT MUST STAY TRUE` forbade adding
"weather, light, time of day, emotion on a face". **The model resolved my contradiction
by adding light.** Rewritten so the image informs word choice and never named content.

### The overlap metric was misleading on its own

Overlap kept rising as the captions got more honest, which read as a loss until the
Sonnet reference landed: **overlap 0.411 with keyword-rule 0.347**. High overlap, low
keyword-detectability is the *good* regime — registers separated by structure rather
than vocabulary, which is exactly what the study needs. v5 at **0.418 / 0.395** is in
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
