"""The held-out register classifier — the study's primary instrument.

`docs/preregistration.md` §4 makes this a DistilRoBERTa trained on the training split
only, and §4's subsection insists it be validated as an instrument rather than trusted.
This module implements both the classifier and its validations, because every number it
produces feeds a pre-registered gate:

    ceiling            accuracy on the generated captions, expected >= 0.85.
                       Below that, the data lacks separable tone and §6 halts the study.
    floor              accuracy with labels randomly reassigned, expected ~= 0.20.
                       Above that, the evaluation is leaking and the metric is broken.
    artifact ablation  accuracy on punctuation-stripped, lowercased text. A large gap
                       means the classifier reads formatting, and the stripped number
                       becomes primary (§4.1).
    confusion matrix   always reported. Two registers collapsing into each other is a
                       finding about the taxonomy, not noise to average away (§4.3).

**Splitting is by ``image_id``, never by caption.** The 25 cells of one image are
rewrites of five near-identical source captions; splitting by row puts near-duplicate
text on both sides of the boundary and inflates every number here.

A TF-IDF + logistic-regression baseline is included deliberately. It is a stronger
relative of the lexical-shortcut anchor in :mod:`emocap.eval.anchors`: if it matches the
transformer, the task is being solved by vocabulary and the transformer is adding
nothing.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from typing import Mapping, Sequence

from emocap.data.prompt import EMOTIONS

__all__ = [
    "build_dataset",
    "folds_by_image",
    "tfidf_baseline",
    "finetune_classifier",
    "confusion_matrix",
    "strip_artifacts",
]

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def strip_artifacts(text: str) -> str:
    """Lowercase and remove punctuation, for the §4.1 artifact ablation.

    v1 captions differed by exclamation marks and clause structure as much as by tone.
    If accuracy survives this, the classifier is reading words rather than formatting.
    """
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


def build_dataset(records: Sequence[Mapping]) -> tuple[list[str], list[int], list[str]]:
    """(texts, label ids, image ids) from a caption store, in EMOTIONS order."""
    index = {e: i for i, e in enumerate(EMOTIONS)}
    texts: list[str] = []
    labels: list[int] = []
    images: list[str] = []
    for rec in records:
        for register, text in (rec.get("captions") or {}).items():
            if register not in index or not str(text).strip():
                continue
            texts.append(str(text).strip())
            labels.append(index[register])
            images.append(str(rec["image_id"]))
    return texts, labels, images


def folds_by_image(images: Sequence[str], *, folds: int = 5, seed: int = 42):
    """Yield (train_idx, test_idx) with every caption of an image on one side."""
    unique = sorted(set(images))
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    for fold in range(folds):
        held = set(shuffled[fold::folds])
        train = [i for i, img in enumerate(images) if img not in held]
        test = [i for i, img in enumerate(images) if img in held]
        if train and test:
            yield train, test


def tfidf_baseline(
    texts: Sequence[str],
    labels: Sequence[int],
    images: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 42,
) -> dict:
    """Word/char TF-IDF + logistic regression, cross-validated by image."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    correct = 0
    total = 0
    pairs: list[tuple[int, int]] = []
    for train, test in folds_by_image(images, folds=folds, seed=seed):
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
        )
        model.fit([texts[i] for i in train], [labels[i] for i in train])
        pred = model.predict([texts[i] for i in test])
        for i, p in zip(test, pred):
            pairs.append((labels[i], int(p)))
            correct += int(labels[i] == p)
            total += 1
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "n": total,
        "pairs": pairs,
    }


def finetune_classifier(
    texts: Sequence[str],
    labels: Sequence[int],
    images: Sequence[str],
    *,
    model_name: str = "distilroberta-base",
    folds: int = 5,
    seed: int = 42,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 48,
    device: str | None = None,
    duty: float = 1.0,
    fold_pause: float = 0.0,
    progress=None,
) -> dict:
    """Fine-tune ``model_name`` per fold and return pooled held-out accuracy.

    Deliberately plain: no early stopping, no per-fold tuning, no validation split
    carved out of train. Every fold gets identical treatment, so the number is a
    property of the data rather than of a search over configurations.

    ``duty`` trades wall clock for thermal load on a laptop: after each optimizer step,
    sleep long enough that the step occupies that fraction of the elapsed time. ``0.33``
    means compute a third of the time, so roughly 3x the runtime at roughly a third of the
    sustained power. The computation is **unchanged** -- same batches, same seed, same
    gradients, bit-identical result (verified to six decimals). Only the spacing differs,
    which is why this is preferable to shrinking the batch or cutting epochs: those alter
    the answer.

    Requires a device barrier per step, because MPS and CUDA are asynchronous. Without it
    the measured step time is CPU enqueue latency (~2 ms) rather than GPU compute
    (~400 ms), and the throttle silently does nothing -- which is how the first version of
    this behaved.

    ``fold_pause`` idles that many seconds between folds. ``duty`` lowers instantaneous
    load; a pause gives the machine a genuine cool-down window, which is what actually
    sheds accumulated heat on a laptop. Neither affects the result.
    """
    if not 0.0 < duty <= 1.0:
        raise ValueError("duty must be in (0, 1]")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _sync() -> None:
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(list(texts), truncation=True, max_length=max_length,
                    padding="max_length", return_tensors="pt")
    all_ids, all_mask = enc["input_ids"], enc["attention_mask"]
    all_labels = torch.tensor(list(labels))

    correct = 0
    total = 0
    pairs: list[tuple[int, int]] = []

    for fold_n, (train, test) in enumerate(folds_by_image(images, folds=folds, seed=seed)):
        torch.manual_seed(seed + fold_n)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(EMOTIONS)
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        tr = TensorDataset(all_ids[train], all_mask[train], all_labels[train])
        loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
        model.train()
        for _ in range(epochs):
            for ids, mask, y in loader:
                _t0 = time.perf_counter() if duty < 1.0 else 0.0
                opt.zero_grad()
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                            labels=y.to(device))
                out.loss.backward()
                opt.step()
                if duty < 1.0:
                    # MPS and CUDA queue work asynchronously: opt.step() returns once the
                    # kernels are ENQUEUED, so timing it without a barrier measures ~2 ms
                    # of CPU enqueue instead of ~400 ms of GPU compute, and the sleep below
                    # becomes a no-op. Synchronise first so `_busy` is the real cost.
                    _sync()
                    _busy = time.perf_counter() - _t0
                    time.sleep(_busy * (1.0 / duty - 1.0))

        model.eval()
        with torch.no_grad():
            for start in range(0, len(test), 64):
                chunk = test[start:start + 64]
                logits = model(input_ids=all_ids[chunk].to(device),
                               attention_mask=all_mask[chunk].to(device)).logits
                pred = logits.argmax(-1).cpu()
                for i, p in zip(chunk, pred.tolist()):
                    pairs.append((int(labels[i]), int(p)))
                    correct += int(labels[i] == p)
                    total += 1
        if progress:
            progress(fold_n, round(correct / max(1, total), 4))
        del model
        if device == "mps":
            torch.mps.empty_cache()
        if fold_pause > 0:
            time.sleep(fold_pause)

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "n": total,
        "pairs": pairs,
        "model": model_name,
        "device": device,
        "epochs": epochs,
    }


def confusion_matrix(pairs: Sequence[tuple[int, int]]) -> dict:
    """Row-normalised confusion, plus each register's own recall."""
    counts: Counter = Counter(pairs)
    rows: dict[str, dict[str, float]] = {}
    recall: dict[str, float] = {}
    for t, true_name in enumerate(EMOTIONS):
        row_total = sum(v for (a, _), v in counts.items() if a == t)
        if not row_total:
            continue
        rows[true_name] = {
            EMOTIONS[p]: round(counts.get((t, p), 0) / row_total, 3)
            for p in range(len(EMOTIONS))
        }
        recall[true_name] = round(counts.get((t, t), 0) / row_total, 3)
    return {"row_normalised": rows, "recall": recall}
