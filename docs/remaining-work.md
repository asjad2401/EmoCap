# Remaining work

Status as of 2026-08-25. **All measurement is complete and closed.** The one registered
quantity delivered short — off-distribution validation, 148 of 250 — was closed deliberately
and the decision is logged in [`deviations.md`](deviations.md). What remains is reporting and
the writing. Everything above the "Optional" heading is registered in
`configs/prereg.lock.yaml` or `docs/preregistration.md` and is not new scope; the Optional
section is explicitly not, and says so.

## GPU — the registered sweep is complete

All **36 registered runs** have landed: 6 arms x 5 folds, plus one negative control per arm.
`emocap-s25a`, `emocap-s25b` and `emocap-v1` finished on 2026-08-24. Every control passes
its 0.25 ceiling and every probed run changed 100% of its captions with the image blanked.

- [x] `emocap-posthoc` — done. `S_paired_matched` x6 and all 35 baselines pulled and
      scored.
- [x] `emocap-unpaired-scaled` — done. `S_unpaired_scaled` x6; see the Optional section.
- [x] `emocap-p4` — done. Two provenance judges trained on a T4 in ~17 minutes, on the
      third attempt: one run wedged on MPS overnight, one crashed formatting a log path.

**No GPU work remains.** Every number the machines can produce is produced.

## Analysis — no GPU needed

- [x] **P4, asymmetric transfer — run. FALSIFIED, and vacuously.**
      `scripts/p4_transfer.py` trains two single-provenance judges and reports transfer as
      a **drop from own-corpus accuracy**, because a raw cross-provenance number cannot
      separate "transfer fails" from "that judge is weak". It also measures corpus-level
      transfer on the reference captions as a control.

      On generated captions the two sides come out **identical** at 0.2027, so the
      registered condition — symmetric, or reversed — is met, and met vacuously: all three
      arms it names sit at chance. That is the fourth registered prediction closed by one
      diagnosis.

      The **corpus-level** asymmetry is real and large. The synthetic-trained judge loses
      **54.1 points** moving to human text (0.8565 → 0.3157); the human-trained judge loses
      **29.5** moving the other way (0.7241 → 0.4294). Synthetic register is far easier to
      read — the same fact the keyword anchor reports at 0.507 against 0.450 and TF-IDF at
      0.81 against 0.61, now measured a third time with a transformer. Reporting drops
      rather than raw scores is load-bearing here: raw, the human-trained judge looks
      *better* at foreign text while being 13 points worse on its own corpus.
- [x] **Secondary metrics — computed on all 48 runs.** `scripts/secondary_metrics.py`.
      Distinctiveness earned its place: it turned `S_paired_matched`'s floor result from an
      uninformative null into a diagnosis (self-BLEU 0.8631 — one caption per image reused
      across all five registers). All 48 runs are covered.
- [x] **Artifact ablation — redone on the instrument that scores the study.** The
      registered version reported a gap of exactly 0.0000 because it ran on TF-IDF, whose
      tokenizer discards punctuation: a test that could not fail. On the frozen
      DistilRoBERTa, stripping changes 100% of captions and the gaps are −0.0009 to
      +0.0123 — except `V1_paired5` at **+0.0381**, three times any other arm. See
      `results/artifact_ablation.json`. The instrument's own CV on stripped text is also
      done — **0.8279 → 0.8214, a gap of +0.0065** — so it does not read typography and the
      raw accuracies stand as primary. §4.1's fallback is not triggered.

- [x] **Per-arm classifier robustness.** Closed by P4's two single-provenance judges,
      `models/judge_S` and `models/judge_H`. The frozen instrument was not touched and its
      hash is unchanged.

## Human work — the critical path

These need other people and are the slowest items. Nothing else in this file can sink the
study; these can.

- [x] **Human evaluation — the task is BUILT and ready to send.**
      `HUMAN-EVAL.html`, 8.0 MB, self-contained: 150 scored items (S_paired25, S_paired5,
      V1_paired5, 50 each, 10 per register) plus 12 attention checks, every photograph
      embedded, no photograph shown twice. Built by `scripts/make_human_eval.py`; the
      answer key is `runs/human-eval/key.json` and the page never loads it. The arm
      selection, the check design and the pass rule were all fixed before any rating
      existed — see `docs/deviations.md`, 2026-08-24.

      Registered as **run once, on final models only**, so a second build after seeing
      results would be a deviation.

- [x] **Three raters, done.** All external, none connected to the study. Krippendorff's
      alpha (ordinal): **grounding 0.684**, above the 0.667 bar for tentative conclusions;
      **tone-match 0.579**, below it. Per-arm tone: S_paired25 4.19, S_paired5 4.15,
      V1_paired5 **3.88** — V1 last at one rater, two and three.

      Attention checks IM 12/12, MA 12/12, MM 8/12. **MM is retained**: they passed the
      rule fixed before any data existed, and dropping the rater who turns out to disagree
      most, after discovering that, is sampling until the number improves. Leave-one-out is
      reported post-hoc and drops every rater in turn, not only that one. The haste
      hypothesis was tested against the timestamps and does not hold — the *fastest* rater
      agreed most.

- [x] **Legibility widened to 300 items, and the gate PASSES.** A fresh reader, not the
      three above, since a reader trained on the task gives a higher ceiling and the gate
      asks whether a human *exceeds* the anchor. FCE **0.864 ± 0.020** against a corpus
      anchor of **0.507** — about eighteen standard errors, where the old 100-item sitting
      passed by two.

      The earlier figures (0.55 and 0.592) were measured on `captions_v10_35` and
      `captions_v10_37flash`, which are prompt candidates and **not** the shipped corpus:
      on 130 shared images, zero of their caption sets match `captions_corpus.jsonl`. So
      0.592 was never a measurement of the final corpus and 0.864 is the first one.

- [x] **A like-for-like human/classifier comparison**, added post-hoc and free. The frozen
      classifier was run over the identical 300 captions the reader saw: **0.8503**, CI
      [0.8095, 0.8912], against the reader's 0.8640. Indistinguishable, κ **0.718**, same
      label on 77.5% of items, complementary errors. This closes the *judge* half of P5's
      comparability problem — the instrument reads corpus captions at human level.

- [x] **Off-distribution validation — 148 of the registered 250.** Three blind writers.
      **0.7770**, CI [0.7095, **0.8446**] — the interval contains the instrument's own
      0.8279, so no degradation is detectable. On the same captions a keyword rule reaches
      only **0.2782**, so the classifier works where the lexical shortcut has essentially
      vanished: it is neither a Gemini detector nor a keyword detector.

      **CLOSED at 148 on 2026-08-25.** The shortfall of 102 is reported, not absorbed —
      see `docs/deviations.md` for the decision and why it is not optional stopping.
      Completing it would tighten the interval from about ±0.067 to ±0.052 and cannot move
      the verdict, since 0.8279 sits inside either. A real drop of up to roughly twelve
      points cannot be excluded at this n; the 5.1-point point difference is the
      instrument's measurement error, not a demonstrated gap.

      Note the order of events, because it is what makes the stop defensible: at 100
      captions the pooled figure had *fallen* to 0.7400 and collection continued anyway.
      It was stopped only after the third chunk lifted it to 0.7770 — the harder direction
      to stop in.

- [ ] **P5 — measurable now, and the comparison needs a caveat in the paper.** The human
      blind-guess score on the shipped corpus is **0.864**. `S_paired25` at **0.9119** and
      `V1_paired5` at **0.8748** exceed it; the other arms do not. P5 is a bound reported
      either way, so crossing it is not a failure.

      But it is **not like-for-like** and must be written as such: the human read *corpus*
      captions and the classifier read *generated* captions, so the text differs even
      though the judge no longer does. Nothing in the study closes that half.

      This also supersedes the prereg's own reasoning on P5, which says "on this corpus a
      reader scores 0.592 and a keyword bag 0.510, so a model at 0.85 is prima facie
      evidence of a shortcut". That rests on 0.592, measured on a corpus that was never
      shipped. On the one that was, a reader scores 0.864, so a model at 0.85–0.91 is near
      human rather than prima facie shortcutting.

## Reporting obligations

- [ ] Report **every** registered hypothesis with its outcome. Final tally: **P3a and
      P3b confirmed** (P3b the only registered magnitude criterion met, by a factor of
      two) · **P2 and P4 falsified** · **P6 not confirmed** — both directions right, the
      margin gap significant at +0.0232 but 2.3 points against a required 7 ·
      **P1 underpowered** at +3.75, below the 0.07 smallest effect of interest and reported
      as underpowered rather than as a null · **P5** a bound, crossed by two arms with the
      like-for-like caveat above.
      They are frozen in the public `prereg-v2` tag. Framing the paper around the
      confirmed results is ordinary scientific writing; omitting registered predictions is
      the selective reporting the registration exists to prevent.
- [ ] Report the **shared diagnosis** rather than four separate disappointments. P1, P2,
      P4 and P6 all rest on 4,390-cell arms, and no arm of that size learned any register:
      accuracy 0.2018–0.2457 against chance 0.200. One measured fact closes four
      predictions, and it belongs in the paper as a result. Then report P6 with its own
      diagnosis: the corpora differ by 21 anchor
      points and their generated captions by 3.1, because at 4,390 cells no model learns
      enough register vocabulary for provenance to transfer. The finding that a
      three-corpus provenance comparison needs paired data — and that the human corpus
      cannot supply it — stands on its own.
- [ ] State that the P6-shaped reading of `S_paired5` vs `V1_paired5` is **exploratory**.
- [ ] **Quote the recomputed anchor gradient, not the registered one, and give P6's true
      scoping reason.** The registered `ours 0.507` was a forecast and reproduces from no
      corpus; the v10 corpus that trains every S arm is **0.5890** at 4,486 cells. The
      gradient's ordering holds (v1 0.7171 > ours 0.5890 > human 0.4501) but the two anchor
      gaps swap, so P6 was scoped to the matched pair by the **image-distribution confound**,
      not by the anchor gap the registration cites. Also report that the anchor depends on
      **cells per image** as well as n. Full record: `docs/deviations.md`, 2026-08-25;
      numbers: `results/anchor_corpus_gradient.json`.
      That pair is a confirmatory comparison in its own right, but P6 was registered over
      the unpaired pair.

## Optional — decided after the tag, to be labelled post-hoc in the paper

Neither of these is registered. Both were decided on 2026-08-23, after the first arm
results existed, and the paper must say so where they are reported. **Both are now
implemented**, in code kept deliberately apart from the registered path so a later reader
can tell the two apart: `src/emocap/data/posthoc.py`, `scripts/build_posthoc_arms.py`,
`scripts/baseline_prior.py`, and one notebook, `notebooks/02f_posthoc.ipynb` (~2 h on a
T4). `configs/prereg.lock.yaml`'s `arms:` list and `data/arms/manifest.json` are untouched.
The full rationale is in `docs/deviations.md`, 2026-08-24.

- [x] **`S_paired_matched` — built.** 4,390 cells, 878 images, 878 per register, folds
      [820, 950, 905, 790, 925], sha256 `1b012fdd77862bf0…`. Identical in size to
      `S_unpaired`, differing only in structure, so a margin here where `S_unpaired` has
      none isolates pairing from volume. Its images are a strict subset of `S_unpaired`'s,
      its cells a strict subset of `S_paired5`'s, and every `S_unpaired` cell on a shared
      image reappears in it — the build asserts all three and refuses to write otherwise,
      which is what makes "the post-hoc arm drew easier data" unavailable as an
      explanation. Nothing new was sampled and no generation was needed.

      As registered, P3 confounds three things at once: `S_paired5` has 9x the cells of
      `S_unpaired`, roughly twice the images, AND the paired structure. The prereg calls
      P3a a "sanity check on the extra data", so as registered the claim is about volume.
      This is what turns P3 into a claim about *parallel* data that extends past image
      captioning — to controllable generation, persona dialogue, emotional TTS,
      attribute-conditioned editing. The style-transfer literature already distinguishes
      parallel from non-parallel corpora, so the contribution is the quantification
      against a lexical anchor, not the idea.

- [x] **Baselines — written.** `scripts/baseline_prior.py`, two modes, no training:
      the same frozen GPT-2 the arms use, the same frozen `DecodeConfig`, scored by the
      same frozen classifier. Output drops straight into `scripts/score_arm.py`.

      `--mode prior` is the register word alone — the baseline this file originally named.
      It is **degenerate by construction**: beam search is deterministic and the prompt
      depends only on the register, so the whole held-out set decodes to **five captions**.
      Its accuracy is a five-outcome draw, the same trap the blanked probe walked into at
      0.58. The script prints all five strings, and they must be published beside the
      number. `--mode text` adds the neutral Flickr8k source caption: per-cell variation,
      still no image, and it answers the sharper question — how much accuracy is reachable
      from the source text with the photograph removed.

      Phrase any systems claim in **margins, not raw accuracy**. The paper argues that
      accuracy overstates conditioning because 89% of it is keyword-reachable; leaning on
      that same raw number to praise the model contradicts its own central finding.

- [x] **Kaggle dataset re-pushed** with both post-hoc arms, and a third dataset
      `emocap-v2-predictions` added: generated captions with references stripped, so
      GPU-side analysis of model output needs no third-party corpus text.

- [x] **`S_unpaired_scaled` — trained, scored, and it ANSWERS P3 against the hypothesis.**
      40,235 cells, 8,047 images, five source captions per image at one register each.
      All six runs clean, control passes at 0.1942, probe changed 100% of captions.

      | arm | cells | images | accuracy | anchor | margin |
      |---|---|---|---|---|---|
      | `S_paired5` | 40,235 | 8,047 | 0.7685 | 0.6826 | **+0.0860** |
      | `S_unpaired_scaled` | 40,235 | 8,047 | 0.7758 | 0.6357 | **+0.1401** |

      At identical volume, identical images and identical cells-per-image, the arm with
      **no** register pairing scores **5.3 margin points higher**: paired comparison
      −0.0529, CI [−0.0648, −0.0409], p 0.0001, negative on 5/5 folds, and the fold mean
      (−0.0541) agrees with the bootstrap.

      **The parallel-data claim is contradicted, not merely unsupported.** Removing the
      pairing *improved* conditioning. `S_unpaired_scaled` also matches `S_paired25`'s
      margin (+0.1401 vs +0.1397) on a fifth of the cells.

      Both components move together: accuracy is marginally higher (0.7758 vs 0.7685) and
      the anchor is much lower (0.6357 vs 0.6826) — the unpaired arm's captions are
      substantially less keyword-separable.

      **A mechanism worth testing, offered as hypothesis and not finding.** When five cells
      share one source caption and differ only by register, the cheapest thing to learn is
      to swap register keywords into a fixed sentence — exactly what the keyword anchor
      detects. Five *different* source captions at one register force varied vocabulary
      instead. If that holds, paired data actively teaches the lexical shortcut.

      **Everything this file previously claimed for this arm is withdrawn.** It said "if
      the paired arm produces a margin over its anchor and the unpaired one does not,
      pairing is isolated from volume", and that the result would extend "to controllable
      generation, persona dialogue, emotional TTS". The measurement went the other way and
      those sentences do not stand.

- [x] **Probe verdicts recorded for all 48 runs**, none excluded. `compare_arms.py`
      prints FINAL rather than PROVISIONAL.

- [ ] **A stronger baseline, not written.** A published emotion-captioning model decoded on
      the same held-out cells would beat both modes above as a comparison point, and costs
      considerably more than an afternoon.
