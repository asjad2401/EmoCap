# Remaining work

Status as of 2026-08-24. Everything above the "Optional" heading is registered in
`configs/prereg.lock.yaml` or `docs/preregistration.md` and is not new scope; the Optional
section is explicitly not, and says so.

## GPU — the registered sweep is complete

All **36 registered runs** have landed: 6 arms x 5 folds, plus one negative control per arm.
`emocap-s25a`, `emocap-s25b` and `emocap-v1` finished on 2026-08-24. Every control passes
its 0.25 ceiling and every probed run changed 100% of its captions with the image blanked.

- [ ] `emocap-posthoc` — `S_paired_matched` x6 and the frozen-LM baselines (~2 h). See the
      Optional section; needs a Kaggle dataset re-push first.

## Analysis — no GPU needed

- [ ] **P4, asymmetric transfer.** Train two more classifiers, one on H only and one on S
      only, and score each arm under both. P4 is about judges, not generators, so no arm
      re-runs are involved. `src/emocap/eval/register_classifier.py` already has the
      training entry point.
- [ ] **Secondary metrics.** CIDEr-D, BLEU-4, CLIPScore and distinctiveness are registered
      in `metrics.secondary` and have not been computed on any arm output. A model can
      score well on register while writing bad captions, and nothing currently rules that
      out. `scripts/clipscore_*.py` exist but were written for corpus work.
- [ ] **Per-arm classifier robustness.** Registered in `metrics.classifier.
      robustness_reported`. Not run.

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

- [ ] **Send it to 3 raters and collect their exports.** THIS is the critical path and it
      is the only item left that needs other people. Each rater opens the file, works
      through it (progress saves in their browser), and sends back one JSON. Drop those
      into `runs/human-eval/` and run:

          uv run python scripts/score_human_eval.py

      It reports attention-check pass rates per rater first, then Krippendorff's alpha
      (ordinal) on both scales, then per-arm means with cluster-bootstrap CIs. A rater
      failing more than half their checks is FLAGGED and still included — excluding one is
      a judgement to write down, not something the script does.
- [ ] **Widen the legibility sample to >=300 items.** The current human number is one
      unblinded 100-item self-measurement at roughly 1.7 SE, and the prereg registers the
      gate with that weakness stated. The gate cannot be applied until this is done.
- [ ] **Off-distribution validation, 250 hand-labelled captions not written by Gemini.**
      Registered in `metrics.classifier.robustness_reported`. Without it, the classifier
      can be argued to have learned one generator's habits rather than emotional register.

- [ ] **P5 depends on the widened sample.** It is a bound reported either way, not a
      hypothesis. Worth flagging early: `S_paired5` scores 0.7685 against a superseded
      human reference of 0.726, so the bound may well be crossed. That is a reportable
      result in either direction, but the number is meaningless until the 300-item
      measurement exists.

## Reporting obligations

- [ ] Report **every** registered hypothesis with its outcome, including P1, P2 and P6.
      They are frozen in the public `prereg-v2` tag. Framing the paper around the
      confirmed results is ordinary scientific writing; omitting registered predictions is
      the selective reporting the registration exists to prevent.
- [ ] Report P6 **with its diagnosis**, not as a bare null: the corpora differ by 21 anchor
      points and their generated captions by 3.1, because at 4,390 cells no model learns
      enough register vocabulary for provenance to transfer. The finding that a
      three-corpus provenance comparison needs paired data — and that the human corpus
      cannot supply it — stands on its own.
- [ ] State that the P6-shaped reading of `S_paired5` vs `V1_paired5` is **exploratory**.
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

- [ ] **Still to do:** re-push the Kaggle dataset so `S_paired_matched.jsonl` is in it
      (`uv run python scripts/push_kaggle.py`), then run `02f_posthoc.ipynb`. Notebook
      cell 4 checks for the arm file and fails immediately if the attached dataset predates
      it, rather than 20 minutes in.

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

- [ ] **Probe verdicts for the 6 `S_unpaired_scaled` runs.** All six changed 100% of
      captions with the image blanked. `compare_arms.py` prints PROVISIONAL until they are
      recorded in `runs/probe_verdicts.json`.

- [ ] **A stronger baseline, not written.** A published emotion-captioning model decoded on
      the same held-out cells would beat both modes above as a comparison point, and costs
      considerably more than an afternoon.
