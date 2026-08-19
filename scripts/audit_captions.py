#!/usr/bin/env python
"""Reproduce the caption-quality tables for one or more caption stores.

Every figure quoted in docs/lab-notebook.md for the prompt-v5 iteration comes from here.
Run it to recompute them, or to check a new store before committing to a full run.

    uv run python scripts/audit_captions.py data/generated/captions_audit_v*.jsonl
    uv run python scripts/audit_captions.py data/generated/captions_raw.jsonl --json

Reported per store:
  defects        forbidden-shortcut rate (src/emocap/data/grounding.py)
  keyword rule   the pre-registered lexical-shortcut anchor (src/emocap/eval/anchors.py)
  overlap        mean pairwise similarity of one caption's five registers
  recall         fraction of the source caption's content words that survive
  strain         the model's own register-fit report, where present

Read `overlap` and `keyword rule` together, never separately. High overlap with LOW
keyword accuracy is the intended regime -- registers separated by structure rather than
vocabulary. The Sonnet reference set sits at 0.411 / 0.347. Either number alone points
the wrong way.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data.grounding import defect_counts, load_factual_captions  # noqa: E402
from emocap.data.quality import content_recall, novel_content, register_divergence  # noqa: E402
from emocap.eval.anchors import keyword_rule_accuracy  # noqa: E402
from emocap.eval.bootstrap import cluster_bootstrap_mean  # noqa: E402
from emocap.data.prompt import EMOTIONS  # noqa: E402


def load(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a truncated final line must not make the file unreadable
    return out


def audit(records: list[dict], factual: dict[str, str] | None) -> dict:
    clusters = [r["image_id"] for r in records]
    overlap = [register_divergence(r["captions"])["mean_similarity"] for r in records]
    recalls, recall_clusters, novel = [], [], []
    words: list[int] = []
    for r in records:
        for text in r["captions"].values():
            recalls.append(content_recall(r["source_caption"], text))
            recall_clusters.append(r["image_id"])
            novel.append(len(novel_content(r["source_caption"], text)))
            words.append(len(str(text).split()))

    strain_levels: Counter = Counter()
    strain_by_register: dict[str, list[int]] = defaultdict(list)
    for r in records:
        for register, level in (r.get("strain") or {}).items():
            strain_levels[level] += 1
            strain_by_register[register].append(level)

    out = {
        "records": len(records),
        "images": len({r["image_id"] for r in records}),
        "defects": defect_counts(records, factual),
        "anchor": keyword_rule_accuracy(records),
        "overlap": cluster_bootstrap_mean(overlap, clusters, n_resamples=4000),
        "recall": cluster_bootstrap_mean(recalls, recall_clusters, n_resamples=4000),
        "added_words_per_caption": round(statistics.mean(novel), 2),
        "words": {
            "mean": round(statistics.mean(words), 1),
            "under_min": sum(1 for w in words if w < 8),
            "over_max": sum(1 for w in words if w > 24),
        },
        "flagged_by_validator": sum(len(r.get("rejected") or {}) for r in records),
    }
    if strain_levels:
        out["strain"] = {
            "levels": {str(k): v for k, v in sorted(strain_levels.items())},
            "mean_by_register": {
                reg: round(statistics.mean(strain_by_register[reg]), 2)
                for reg in EMOTIONS
                if strain_by_register.get(reg)
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stores", nargs="+", help="caption JSONL files")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--no-factual", action="store_true",
                    help="skip the image-dependent defect classes")
    args = ap.parse_args()

    factual = None
    if not args.no_factual:
        try:
            factual = load_factual_captions(ROOT / "archive/v1-pilot/data/"
                                            "v1_moondream_factual_captions.csv.gz")
        except FileNotFoundError as exc:
            print(f"warning: {exc}\n", file=sys.stderr)

    results = {}
    for spec in args.stores:
        p = Path(spec)
        if not p.is_absolute():
            p = ROOT / p
        records = load(p)
        if not records:
            print(f"warning: {p} has no records, skipping", file=sys.stderr)
            continue
        results[p.name] = audit(records, factual)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"{'store':<34} {'cells':>6} {'defects':>9} {'keyword':>8} "
          f"{'overlap':>8} {'recall':>7} {'added':>6} {'<8w':>4}")
    print("-" * 90)
    for name, r in results.items():
        d = r["defects"]
        print(f"{name:<34} {d['cells']:>6} "
              f"{d['defects']:>4} {100 * d['rate']:>4.1f}% "
              f"{r['anchor']['accuracy']:>8.3f} "
              f"{r['overlap']['mean']:>8.3f} "
              f"{r['recall']['mean']:>7.3f} "
              f"{r['added_words_per_caption']:>6.2f} "
              f"{r['words']['under_min']:>4}")
    print(f"\nchance for the keyword rule: {next(iter(results.values()))['anchor']['chance']}")
    for name, r in results.items():
        d = r["defects"]
        if d["defects"]:
            print(f"\n{name} defects by class: {d['by_class']}")
        if "strain" in r:
            print(f"{name} strain levels: {r['strain']['levels']}")
            print(f"{name} mean strain by register: {r['strain']['mean_by_register']}")


if __name__ == "__main__":
    main()
