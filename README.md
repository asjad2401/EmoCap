# EmoCap — Emotion-Conditioned Image Captioning

A pre-registered ablation over **where** an emotion signal should be injected into a
caption decoder, on Flickr8k, across two decoder families.

> **v2 rebuild.** The v1 pilot lives in [`archive/v1-pilot/`](archive/v1-pilot/) and is
> documented in [`docs/v1-pilot-postmortem.md`](docs/v1-pilot-postmortem.md). Its results
> were not usable; the reasons are what shaped this design.

## The question

Given an image and a target emotion (`joyful`, `sad`, `tense`, `romantic`, `humorous`),
generate a caption that describes the image *and* reads in that register. The ablation asks
where the emotion should enter the decoder, and whether the answer depends on the decoder.

| Cond. | Track A — LSTM from scratch | Track B — ClipCap (GPT-2 + LoRA) |
|-------|------------------------------|-----------------------------------|
| `V0`  | no emotion input | visual prefix only |
| `V1`  | emotion concatenated into `h0` | emotion as first prefix slot |
| `V2`  | emotion appended to every word embedding | emotion broadcast across all prefix slots |
| `V3`  | emotion-queried cross-attention over patches | emotion-queried mapping network |
| `C`   | best variant, retrained on **shuffled** emotion labels | *(negative control)* |

3 seeds × 5 conditions × 2 tracks = **30 runs**.

## Layout

```
src/emocap/     the package — all logic, unit-tested on CPU
tests/          run with `uv run pytest`, no GPU, no Kaggle, seconds
configs/        every threshold and hyperparameter; prereg.lock.yaml is frozen
notebooks/      thin Kaggle runners, one pipeline stage each
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
| 02 | `captions_generate` | local + Gemini | `emocap-captions-raw` |
| 03 | `captions_qa` | local + Gemini | `emocap-captions` |
| 04 | `features_clip` | GPU | `emocap-clip-feats` |
| 05 | `tokenize_vocab` | CPU | `emocap-tokenized` |
| 06 | `instrument_emotion_clf` | GPU | `emocap-emo-clf` |
| 07 | `train_lstm` | GPU | `runs/` |
| 08 | `train_clipcap` | GPU | `runs/` |
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
- **Decode settings are pre-registered.** Both tracks read one `DecodeConfig`, so no
  variant can be advantaged by per-variant decode tuning.
- **Poor performance is never grounds for exclusion.** See `docs/preregistration.md`.
