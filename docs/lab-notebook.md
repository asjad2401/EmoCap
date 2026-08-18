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
