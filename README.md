# EmoCap — Emotion-Conditioned Image Captioning

A **pre-registered** study of what emotion-conditioning accuracy measures when the training
captions are LLM-synthesised — on Flickr8k, over five registers (`joyful`, `sad`, `tense`,
`romantic`, `humorous`).

> **Status — measurement complete.** All 48 runs trained and scored, every registered
> hypothesis resolved, the human evaluation closed, and the instrument validated four ways.
> What remains is reporting and writing: see
> [`docs/remaining-work.md`](docs/remaining-work.md).
>
> **v2 rebuild.** The v1 pilot lives in [`archive/v1-pilot/`](archive/v1-pilot/) and is
> documented in [`docs/v1-pilot-postmortem.md`](docs/v1-pilot-postmortem.md). Its results
> were not usable; the reasons are what shaped this design.

## The question

Emotion-conditioned captioning is normally scored by asking whether a classifier can recover
the requested emotion from the caption. This study asks **what that number measures when the
captions were written by a prompt applied 8,000 times.**

The motivating measurement, taken before registration: on our first corpus a DistilRoBERTa
classifier scored **0.775** while a human reading the same captions scored **0.310** against a
0.200 chance level — and a bag of 25 keywords per register scored **0.441**, *above* the
human. The classifier was recovering the generator's fingerprint, not a register a reader
could perceive.

So: **how much of emotion-conditioning accuracy reflects a register a reader can perceive, and
how much reflects lexical provenance? And does training on human-written emotive captions
change the answer?**

### What was found

**Most of the metric needs no model.** On the strongest arm the frozen classifier scores
**0.9119** while a 25-word-per-register rule, refitted on those same captions, scores
**0.7722** — the rule recovers **85%** of the headline number. The study's claim is therefore
never the accuracy but the **margin** between them, and its scoring script refuses to print
one without the other.

**And that shortcut is a property of machine-written captions, not of the task.** On 148
captions written by people who had never seen the corpus, the same keyword rule reaches
**0.2782** against a chance level of 0.200 — it essentially stops working — while the
classifier still scores **0.7770**. This was a robustness check, not a hypothesis, and it is
the sharpest result in the study.

**Data volume decides everything; structure barely registers.** Every arm with 40,235 cells
or more clears its anchor. Every arm with 4,390 sits at chance, whoever wrote the text and
whatever its structure. That one line closes **four** registered predictions at once (P1, P2,
P4, P6) — not four disappointments but one measured fact reported four times.

**Parallel data is not the mechanism, and the study says so against its own hope.** Two arms
were added after the tag to separate paired structure from volume. Pairing at 4,390 cells
lands on the floor; volume *without* pairing (`S_unpaired_scaled`, +0.1401) **beats** the
paired arm of identical size and images (`S_paired5`, +0.0860) by 5.3 margin points.

| registered prediction | outcome |
|---|---|
| **P3a** — more data raises accuracy | **confirmed**, +69.9 points |
| **P3b** — and raises the margin by ≥7 pts | **confirmed**, +0.1360 [+0.1190, +0.1528] — the only magnitude criterion met, cleared twofold |
| P2 — the human advantage exceeds vocabulary | **falsified** on its own condition |
| P4 — asymmetric transfer between judges | **falsified**, and vacuously: both sides identical at chance |
| P6 — accuracy tracks stereotypy, not legibility | **not confirmed** — both directions right, 2.3 points against a required 7 |
| P1 — human text beats synthetic on accuracy | **underpowered**, +3.75 below the 0.07 smallest effect of interest |
| P5 — no model beats the human blind-guess score | a bound; crossed by two arms, with a stated caveat |

The instrument was validated four ways rather than asserted: cross-validated **0.8279**;
**0.8503** against a human's **0.8640** on the *same* 300 captions (κ 0.718); **0.7770**
off-distribution, an interval that contains its own in-distribution figure; and a punctuation
ablation moving it **0.0065**. Full numbers in [`docs/lab-notebook.md`](docs/lab-notebook.md);
every figure here is read from `runs/` and `results/`, not transcribed.

### Design — eight data arms, one decoder

| arm | captions from | structure | cells | images |
|---|---|---|---|---|
| **S_paired25** | ours — prompt v10, gemini-3.7-flash | 25 / image | 201,175 | 8,047 |
| **S_paired5** | ours, one source caption | 5 / image | 40,235 | 8,047 |
| **V1_paired5** | the retired v1 pilot corpus | 5 / image | 40,235 | 8,047 |
| **S_unpaired** | ours, subsampled | 1 / image | 4,390 | 4,390 |
| **V1_unpaired** | the v1 pilot, subsampled | 1 / image | 4,390 | 4,390 |
| **H_unpaired** | Personality-Captions (human) | 1 / image | 4,390 | 4,390 |
| **S_unpaired_scaled** † | ours, five source captions | 5 captions, 1 register | 40,235 | 8,047 |
| **S_paired_matched** † | ours, one source caption | 5 / image | 4,390 | 878 |

† decided after the `prereg-v2` tag and labelled post-hoc wherever reported. Both are strict
*views* of cells the registered arms already trained on — `scripts/build_posthoc_arms.py`
refuses to write either unless it can prove the containment — so "the extra arm drew easier
data" is not available as an explanation of what they show.

Cell counts are the **arm** sizes from `data/arms/manifest.json`, which carries a sha256 per
arm and is the authoritative source. They are smaller than the corpus totals because an image
qualifies only if our corpus, the archived v1 pilot and the safety exclusions all admit it,
which leaves **8,047 of 8,076**.

Three structure classes, so provenance is compared at matched shape and never across it. The
1-caption-per-image arms are **4,390** (878 × 5), set by the scarcest register under the strict
trait mapping; matching on count as well as structure keeps P1 from confounding provenance
with data volume.

The decoder — frozen CLIP ViT-B/32 → mapping network → GPT-2 + LoRA — is a **control, held
fixed**. The trainable parameter count comes out identical to the digit, 8,696,064, in all
48 runs, because the arm name selects a data file and changes nothing else. So every
difference in the results belongs to the data. Folds split by `image_id`, never by caption.

The **v1 pilot corpus is an arm** for two reasons: the pilot's decoder faults are fixed here,
so its data can be tested apart from its code; and its keyword anchor is **0.718** against
0.507 for ours and 0.450 for human text — a three-point gradient in lexical stereotypy, which
is the axis the study is about.

Full design, hypotheses and gates: [`docs/preregistration.md`](docs/preregistration.md).
Everything that changed between the draft and the tag:
[`docs/deviations.md`](docs/deviations.md).

## Provenance

The registration is immutable and the working tree proves it:

```bash
git diff prereg-v2 HEAD -- docs/preregistration.md configs/prereg.lock.yaml   # empty
```

Both files are byte-identical to the `prereg-v2` tag (`892358f`, superseding `prereg-v1` at
`3a61cde`; both public, neither ever moved). They therefore still carry three arm sizes as
*drafted* — 8,075 images, 201,875 and 40,375 cells — arithmetic done before the arms existed.
The measured values are 8,047 / 201,175 / 40,235, they are authoritative in
`data/arms/manifest.json` with a sha256 per arm, and the correction is logged in
[`docs/deviations.md`](docs/deviations.md).

That disagreement is deliberate. Corrections belong in an amendments log, not in the
registration — a registration edited to agree with its results is worth less than one that
was not. The same discipline runs through the rest of the record: **four retractions** are
written up rather than quietly fixed, including one that had briefly reported the study's own
thesis arriving from the wrong direction.

## Layout

```
src/emocap/     the package — all logic, unit-tested on CPU
tests/          run with `uv run pytest`, no GPU, no Kaggle, seconds
configs/        every threshold and hyperparameter; prereg.lock.yaml is frozen
notebooks/      thin Kaggle runners: 02a–02e the registered arms, 02f–02g the post-hoc
                arms and baselines, 02h–02i the P4 judges and the ablation.
                Save & Run All, every run output preserved for export, no stage depending
                on a session surviving (hence no checkpointing).
docs/           preregistration, deviations log, lab notebook, remaining work
runs/           one directory per run — manifest, metrics, generations. Never overwritten.
results/        final tables and figures, regenerable from runs/
archive/        the v1 pilot, untouched
```

## Setup

```bash
uv sync --group dev      # Python 3.11 + CPU torch + transformers
uv run pytest            # must pass before anything is trusted
```

## Pipeline, as it actually ran

Corpus generation and analysis run locally; anything needing a GPU is one Kaggle notebook,
sized for a single *Save & Run All*, chained by Kaggle Dataset version. No stage depends on a
session surviving, so nothing checkpoints.

| stage | what | where | entry point |
|---|---|---|---|
| corpus | generate 201,900 captions, prompt v5 on gemini-3.7-flash, Vertex batch | local | `scripts/run_stage02.py` |
| audit | keyword anchor, grounding defects, hand audit of a fixed sample | local | `scripts/audit_captions.py`, `AUDIT.md` |
| arms | materialise the 6 registered arms, sha256 each | local | `scripts/build_arms.py` |
| features | frozen CLIP ViT-B/32 once per image, memory-mapped | GPU | `scripts/extract_features.py` |
| instrument | train the register classifier, then **freeze and hash** it | GPU | `scripts/train_classifier.py` |
| training | 48 runs — 8 arms × 5 folds + 8 controls | GPU | `notebooks/02a`–`02g` |
| scoring | accuracy, anchor at its own n, margin, cluster bootstrap, probe | local | `scripts/score_arm.py` |
| comparisons | Holm over exactly the 4 registered pairs; reference and post-hoc uncorrected | local | `scripts/compare_arms.py` |
| secondary | CIDEr-D, BLEU-4, CLIPScore, distinctiveness | local | `scripts/secondary_metrics.py` |
| instrument checks | P4 judges, punctuation ablation, off-distribution, human-vs-classifier | GPU + local | `notebooks/02h`–`02i`, `scripts/score_offdist.py` |
| human work | 150-item evaluation, 300-item legibility, 250-caption writing task | people | `scripts/make_human_eval.py`, `make_guess_task.py`, `make_writing_task.py` |

The instrument is frozen **before** the first captioner trains, and its hash travels inside
every `score.json` — which turns "all 48 runs used the same judge" from a claim into something
a reader can check.

## Where to start reading

| you want | read |
|---|---|
| what was claimed in advance | [`docs/preregistration.md`](docs/preregistration.md) — and `git show prereg-v2:docs/preregistration.md` is the same bytes |
| what happened, in order, mistakes included | [`docs/lab-notebook.md`](docs/lab-notebook.md) |
| every decision that departed from the plan | [`docs/deviations.md`](docs/deviations.md) |
| what is left | [`docs/remaining-work.md`](docs/remaining-work.md) |
| where any given number comes from | [`FILE-MAP.md`](FILE-MAP.md) |
| the numbers themselves | `runs/arms/*/score.json`, `results/*.json` |

## Rules of the road

- **No number lives in prose.** Thresholds and hyperparameters live in `configs/`, are
  read once, and are echoed into each run's manifest. v1's notebook headers advertised
  filters its code had overridden.
- **Nothing is quotable without a manifest.** Every run writes git SHA, config hash, seeds,
  package versions, and wall time next to its metrics.
- **Decode settings are pre-registered.** One `DecodeConfig` for every arm, so no arm can be
  advantaged by decode tuning.
- **Poor performance is never grounds for exclusion.** See `docs/preregistration.md`.
