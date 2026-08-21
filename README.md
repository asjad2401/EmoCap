# EmoCap — Emotion-Conditioned Image Captioning

A **pre-registered** study of what emotion-conditioning accuracy measures when the training
captions are LLM-synthesised — on Flickr8k, over five registers (`joyful`, `sad`, `tense`,
`romantic`, `humorous`).

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

### Design — six data arms, one decoder

| arm | captions from | structure | captions |
|---|---|---|---|
| **S-paired25** | ours — prompt v10, gemini-3.7-flash | 25/image | 201,900 |
| **S-paired5** | ours, 1 source caption | Flickr8k | 5/image | 40,240 |
| **S-unpaired** | ours, subsampled | 1/image | 4,390 |
| **V1-paired5** | the retired v1 pilot corpus | 5/image | 40,240 |
| **V1-unpaired** | the v1 pilot, subsampled | 1/image | 4,390 |
| **H-unpaired** | Personality-Captions (human) | 1/image | 4,390 |

Three structure classes, so provenance is compared at matched shape and never across it. The 1-caption-per-image arms are **4,390** (878 x 5), set by the scarcest register under the strict trait mapping; matching on count as well as structure keeps P1 from confounding provenance with data volume. The
decoder — frozen CLIP ViT-B/32 → mapping network → GPT-2 + LoRA — is a **control, held
fixed**, so differences are attributable to the data. **36 runs**: 6 arms × 5 folds, plus a
shuffled-label negative control per arm. Folds split by `image_id`, never by caption.

The **v1 pilot corpus is an arm** for two reasons: the pilot's decoder faults are fixed here,
so its data can be tested apart from its code; and its keyword anchor is **0.718** against
0.507 for ours and 0.450 for human text — a three-point gradient in lexical stereotypy, which
is the axis the study is about.

Full design, hypotheses and gates: [`docs/preregistration.md`](docs/preregistration.md).
Everything that changed between the draft and the tag:
[`docs/deviations.md`](docs/deviations.md).

## Layout

```
src/emocap/     the package — all logic, unit-tested on CPU
tests/          run with `uv run pytest`, no GPU, no Kaggle, seconds
configs/        every threshold and hyperparameter; prereg.lock.yaml is frozen
notebooks/      thin Kaggle runners, one per pipeline stage — PLANNED, not yet written.
                Save & Run All, every run output preserved for export, no stage depending
                on a session surviving (hence no checkpointing).
docs/           preregistration, deviations log, lab notebook, postmortem
runs/           one directory per run — manifest, metrics, generations. Never overwritten.
results/        final tables and figures, regenerable from runs/
archive/        the v1 pilot, untouched
```

## Setup

```bash
uv sync --group dev      # Python 3.11 + CPU torch + transformers
uv run pytest            # must pass before anything is trusted
```

## Pipeline

Each stage is one notebook, sized for a single Kaggle *Save & Run All*, chained by Kaggle
Dataset version. No stage depends on a session surviving, so no checkpointing is needed.

| # | Stage | Where | Output |
|---|-------|-------|--------|
| 00 | `env_smoke` | CPU | — |
| 01 | `data_flickr8k_splits` | CPU | `emocap-splits` |
| 02 | `captions_generate` — prompt v10, gemini-3.7-flash, Vertex batch | local + Vertex | `emocap-captions-raw` |
| 03 | `captions_qa` | local + Gemini | `emocap-captions` |
| 04 | `features_clip` | GPU | `emocap-clip-feats` |
| 05 | `tokenize_vocab` | CPU | `emocap-tokenized` |
| 06 | `instrument_emotion_clf` | GPU | `emocap-emo-clf` |
| 07 | `build_arms` — S-paired / S-unpaired / H-unpaired | CPU | `emocap-arms` |
| 08 | `train_clipcap` — one decoder, the six arms, five folds | GPU | `runs/` |
| 09 | `decode_testset` | GPU | `emocap-generations` |
| 10 | `metrics_and_anchors` | CPU | `results/` |
| 11 | `figures_and_tables` | CPU | `results/` |
| 12 | `human_eval_prep` | CPU | `results/` |

## Rules of the road

- **No number lives in prose.** Thresholds and hyperparameters live in `configs/`, are
  read once, and are echoed into each run's manifest. v1's notebook headers advertised
  filters its code had overridden.
- **Nothing is quotable without a manifest.** Every run writes git SHA, config hash, seeds,
  package versions, and wall time next to its metrics.
- **Decode settings are pre-registered.** One `DecodeConfig` for every arm, so no arm can be
  advantaged by decode tuning.
- **Poor performance is never grounds for exclusion.** See `docs/preregistration.md`.
