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
