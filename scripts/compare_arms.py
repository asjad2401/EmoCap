#!/usr/bin/env python
"""Compare the arms. The script that turns 36 runs into the study's results table.

    uv run python scripts/compare_arms.py
    uv run python scripts/compare_arms.py --allow-incomplete   # partial sweep, clearly marked

Everything registered is enforced here rather than assembled by hand:

    * the FOUR confirmatory comparisons come from `configs/prereg.lock.yaml`, not from this
      file, so the set cannot drift from the registration
    * Holm-Bonferroni over exactly those four, and over nothing else
    * reference and exploratory comparisons reported uncorrected and labelled as such
    * cluster bootstrap by image_id, 10,000 resamples, 95% CI -- the registered estimator
    * every arm's margin over ITS OWN anchor, recomputed on ITS OWN captions at ITS OWN n

**Runs excluded by human review never reach the numbers.** §7 makes a failed
visual-dependence probe one of only three grounds for excluding a run, and that call is a
human judgement on caption pairs. This script reads those verdicts from a file and drops
failed runs before anything is computed. A run with no verdict is not silently included:
the results are marked PROVISIONAL and the missing verdicts are listed.

**Why comparisons are paired at the cell level.** All four confirmatory comparisons are
between arms built on the SAME images -- that is why they are confirmatory and the
cross-dataset ones are not. Pairing (image, register) cells across two arms and bootstrapping
by image keeps that matching intact, which is both the registered clustering and a much
tighter interval than comparing two independent means.

**Why accuracy and margin are treated differently.** Accuracy is a per-cell quantity, so it
gets the full 10,000-resample cluster bootstrap. The anchor is not per-cell -- it is a
cross-validated rule fitted to a whole caption set -- so a margin difference cannot be
resampled the same way without refitting the anchor inside every resample. Margins are
therefore reported as the mean over folds with the fold spread beside them, and labelled
n_folds rather than dressed up as a bootstrap interval.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.eval.bootstrap import cluster_bootstrap_mean  # noqa: E402
from emocap.runtime import load_config  # noqa: E402

VERDICT_HELP = """
Create it once the reviewers have read the probe_review.txt files:

    {
      "S_paired5-f0":  {"probe": "pass", "reviewer": "AA", "date": "2026-08-24"},
      "H_unpaired-f0": {"probe": "fail", "reviewer": "AA", "date": "2026-08-24",
                        "note": "captions unchanged with the image blanked"}
    }

Only "fail" excludes a run. Anything else is treated as reviewed-and-kept.
"""


def load_runs(runs_dir: Path) -> dict[str, dict]:
    """Every scored run, keyed by tag. Requires cells.jsonl -- re-score if it is missing."""
    out: dict[str, dict] = {}
    for d in sorted(runs_dir.iterdir()):
        score = d / "score.json"
        if not score.is_dir() and score.exists():
            s = json.loads(score.read_text())
            cells_path = d / "cells.jsonl"
            if not cells_path.exists():
                raise SystemExit(
                    f"{d.name} has score.json but no cells.jsonl -- it was scored before "
                    f"per-cell output existed. Re-run:\n"
                    f"    uv run python scripts/score_arm.py --run {d}")
            s["cells"] = [json.loads(l) for l in cells_path.read_text().splitlines() if l]
            out[d.name] = s
    return out


def apply_exclusions(runs: dict[str, dict], verdicts: dict) -> tuple[dict, list, list]:
    """Drop runs the reviewers failed. Returns kept runs, excluded tags, unreviewed tags."""
    kept, excluded, unreviewed = {}, [], []
    for tag, r in runs.items():
        v = verdicts.get(tag, {})
        if str(v.get("probe", "")).lower() == "fail":
            excluded.append(tag)
            continue
        if "probe" not in v:
            unreviewed.append(tag)
        kept[tag] = r
    return kept, excluded, sorted(unreviewed)


def arm_folds(runs: dict[str, dict], arm: str) -> dict[int, dict]:
    """Fold number -> run, for the real folds of one arm. Negative controls excluded:
    they are a gate on the design, never a data point in a comparison."""
    return {r["fold"]: r for r in runs.values()
            if r.get("arm") == arm and not r.get("negative_control")
            and r.get("fold") is not None}


def paired_cells(a: dict, b: dict) -> tuple[list[float], list[float], list[str]]:
    """Correctness for the (image, register) cells the two runs share.

    A cell can appear more than once in an arm -- S_paired25 holds five source captions per
    (image, register) -- so duplicates are averaged before pairing. That makes one cell one
    observation in both arms, which is what "matched images" has to mean for the comparison
    to be about the arms rather than about how many rows each one happens to have.
    """
    def collapse(run: dict) -> dict[tuple[str, str], float]:
        acc: dict[tuple[str, str], list[float]] = {}
        for c in run["cells"]:
            acc.setdefault((c["image_id"], c["emotion"]), []).append(float(c["correct"]))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    ca, cb = collapse(a), collapse(b)
    keys = sorted(set(ca) & set(cb))
    return [ca[k] for k in keys], [cb[k] for k in keys], [k[0] for k in keys]


def _all_cells(run: dict, fold: int) -> tuple[list[float], list[str]]:
    vals = [float(c["correct"]) for c in run["cells"]]
    clus = [f"{fold}:{c['image_id']}" for c in run["cells"]]
    return vals, clus


def _pvalue(samples: list[float]) -> float | None:
    """Two-sided bootstrap p-value: how often the resampled difference lands on the other
    side of zero from the observed one. Doubled, and floored at 1/n_resamples -- a p of
    exactly 0 would be an artifact of finite resampling, not a finding."""
    if not samples:
        return None
    n = len(samples)
    tail = min(sum(1 for s in samples if s <= 0), sum(1 for s in samples if s >= 0))
    return max(2.0 * tail / n, 1.0 / n)


def compare(runs: dict[str, dict], left: str, right: str, boot: dict) -> dict:
    """One arm-vs-arm comparison, pooled over the folds both arms completed.

    Paired at the cell level when the arms share images, which is the case for all four
    confirmatory comparisons and is exactly why they are the confirmatory ones. The
    reference comparisons cross datasets and share no images at all, so they fall back to
    two independent bootstraps differenced -- a wider interval, correctly, because nothing
    is matched.
    """
    fl, fr = arm_folds(runs, left), arm_folds(runs, right)
    folds = sorted(set(fl) & set(fr))
    if not folds:
        return {"left": left, "right": right, "folds": [], "error": "no shared folds"}

    la, ra, clusters, fold_margin = [], [], [], []
    la_all, ra_all, lc_all, rc_all = [], [], [], []
    for f in folds:
        a, b = fl[f], fr[f]
        x, y, imgs = paired_cells(a, b)
        la += x
        ra += y
        # Fold number joins the image id so the same image in different folds is not
        # resampled as one cluster.
        clusters += [f"{f}:{i}" for i in imgs]
        fold_margin.append((a["accuracy"] - a["anchor"]) - (b["accuracy"] - b["anchor"]))
        xa, xc = _all_cells(a, f)
        ya, yc = _all_cells(b, f)
        la_all += xa; lc_all += xc
        ra_all += ya; rc_all += yc

    is_paired = bool(la)
    if is_paired:
        diffs = [x - y for x, y in zip(la, ra)]
        bs = cluster_bootstrap_mean(diffs, clusters, n_resamples=boot["resamples"],
                                    ci=boot["ci"], seed=42, return_samples=True)
        samples = bs.pop("samples", [])
        left_acc, right_acc = sum(la) / len(la), sum(ra) / len(ra)
        n_paired = len(diffs)
    else:
        # No shared images: bootstrap each arm on its own, then difference the two
        # distributions. Separate seeds, so the two resamplings are independent rather
        # than sharing a draw sequence and quietly narrowing the interval.
        bl = cluster_bootstrap_mean(la_all, lc_all, n_resamples=boot["resamples"],
                                    ci=boot["ci"], seed=42, return_samples=True)
        br = cluster_bootstrap_mean(ra_all, rc_all, n_resamples=boot["resamples"],
                                    ci=boot["ci"], seed=43, return_samples=True)
        sl, sr = bl.pop("samples", []), br.pop("samples", [])
        # cluster_bootstrap_mean returns its resamples SORTED. Differencing two sorted
        # arrays element-wise subtracts matched quantiles, which cancels nearly all the
        # variance and produced an interval an order of magnitude too narrow. Shuffling one
        # side restores the independent pairing the two bootstraps actually have.
        rng = random.Random(4242)
        rng.shuffle(sl)
        rng.shuffle(sr)
        samples = [a - b for a, b in zip(sl, sr)]
        samples.sort()
        alpha = (1.0 - boot["ci"]) / 2.0
        left_acc, right_acc = bl["mean"], br["mean"]
        bs = {"mean": left_acc - right_acc,
              "lo": round(samples[int(alpha * (len(samples) - 1))], 4),
              "hi": round(samples[int((1 - alpha) * (len(samples) - 1))], 4),
              "n_clusters": bl["n_clusters"] + br["n_clusters"]}
        n_paired = 0
    p = _pvalue(samples)

    n_f = len(fold_margin)
    mean_margin = sum(fold_margin) / n_f
    sd_margin = (sum((m - mean_margin) ** 2 for m in fold_margin) / (n_f - 1)) ** 0.5 \
        if n_f > 1 else None

    return {
        "left": left, "right": right, "folds": folds, "paired": is_paired,
        "n_cells_paired": n_paired,
        "accuracy_diff": round(bs["mean"], 4), "ci": {"lo": bs["lo"], "hi": bs["hi"]},
        "n_clusters": bs["n_clusters"], "p_bootstrap": p,
        "crosses_zero": None if bs["lo"] is None else bool(bs["lo"] <= 0 <= bs["hi"]),
        "margin_diff": round(mean_margin, 4),
        "margin_sd_across_folds": round(sd_margin, 4) if sd_margin is not None else None,
        "margin_by_fold": [round(m, 4) for m in fold_margin],
        "left_accuracy": round(left_acc, 4),
        "right_accuracy": round(right_acc, 4),
    }


def holm(results: list[dict], alpha: float = 0.05) -> None:
    """Holm-Bonferroni, in place. Applied to the confirmatory set and nothing else."""
    live = [r for r in results if r.get("p_bootstrap") is not None]
    m = len(live)
    for rank, r in enumerate(sorted(live, key=lambda r: r["p_bootstrap"])):
        r["holm_threshold"] = round(alpha / (m - rank), 5)
        r["holm_rank"] = rank + 1
    # Holm is a step-down procedure: once one test fails, every larger p fails too,
    # regardless of its own threshold. Comparing each p to its own threshold in isolation
    # is the usual way this gets implemented wrong, and it inflates the error rate.
    stopped = False
    for r in sorted(live, key=lambda r: r["p_bootstrap"]):
        if stopped or r["p_bootstrap"] > r["holm_threshold"]:
            stopped = True
            r["significant_holm"] = False
        else:
            r["significant_holm"] = True


def gate_negative_controls(runs: dict[str, dict], cap: float) -> list[dict]:
    out = []
    for tag, r in sorted(runs.items()):
        if r.get("negative_control"):
            out.append({"run": tag, "arm": r.get("arm"), "accuracy": r["accuracy"],
                        "max": cap, "passes": r["accuracy"] <= cap})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/arms")
    ap.add_argument("--verdicts", default="runs/probe_verdicts.json",
                    help="human pass/fail on the visual-dependence probe")
    ap.add_argument("--out", default="results/comparisons.json")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="report on a partial sweep; results are marked PROVISIONAL")
    args = ap.parse_args()

    lock = load_config("prereg.lock")
    study, boot = lock["study"], lock["metrics"]["bootstrap"]
    confirmatory = [tuple(c) for c in study["confirmatory_comparisons"]]
    reference = [tuple(c) for c in study["reference_comparisons"]]
    soi = lock["metrics"]["smallest_effect_of_interest"]

    runs_dir = ROOT / args.runs
    runs = load_runs(runs_dir)
    vpath = ROOT / args.verdicts
    verdicts = json.loads(vpath.read_text()) if vpath.exists() else {}
    kept, excluded, unreviewed = apply_exclusions(runs, verdicts)

    expected = len(study["arms"]) * study["folds"] + len(study["arms"])
    complete = len(runs) >= expected and not unreviewed
    provisional = not complete

    print(f"runs found        {len(runs)} of {expected} registered")
    print(f"excluded by probe {len(excluded)}" + (f"  {excluded}" if excluded else ""))
    if not vpath.exists():
        print(f"\n  NO VERDICT FILE at {vpath.relative_to(ROOT)}")
        print(VERDICT_HELP)
    if unreviewed:
        print(f"unreviewed        {len(unreviewed)} run(s) have no probe verdict:")
        for t in unreviewed:
            print(f"                    {t}")
    if provisional and not args.allow_incomplete:
        raise SystemExit(
            "\nRefusing to write final results. Either finish the sweep and record the probe\n"
            "verdicts, or pass --allow-incomplete to get numbers marked PROVISIONAL.")

    banner = "PROVISIONAL -- incomplete sweep or unreviewed runs" if provisional else "FINAL"
    print(f"\n{'=' * 78}\n{banner}\n{'=' * 78}")

    # ── gates ────────────────────────────────────────────────────────────────
    cap = lock["gates"]["manipulation_check"]["negative_control_accuracy_max"]
    controls = gate_negative_controls(kept, cap)
    print(f"\nMANIPULATION CHECK  negative control accuracy must be <= {cap}")
    if not controls:
        print("  none run yet -- the gate is untested and no result below is interpretable")
    for c in controls:
        print(f"  {c['run']:<26} {c['accuracy']:.4f}  "
              f"{'PASS' if c['passes'] else 'FAIL -- results not interpretable'}")

    # ── confirmatory ─────────────────────────────────────────────────────────
    conf = [compare(kept, a, b, boot) for a, b in confirmatory]
    conf = [c for c in conf if not c.get("error")]
    holm(conf)

    def show(rows: list[dict], title: str, corrected: bool) -> None:
        print(f"\n{title}")
        if not rows:
            print("  (no comparison has runs on both sides yet)")
            return
        for r in rows:
            lo, hi = r["ci"]["lo"], r["ci"]["hi"]
            how = (f"{r['n_cells_paired']:,} paired cells" if r["paired"]
                   else "UNPAIRED -- no shared images, wider interval by construction")
            print(f"\n  {r['left']} vs {r['right']}   folds {r['folds']}  {how}")
            print(f"    accuracy  {r['left_accuracy']:.4f} vs {r['right_accuracy']:.4f}   "
                  f"diff {r['accuracy_diff']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
            if corrected:
                verdict = ("significant" if r.get("significant_holm")
                           else "not significant")
                print(f"    p {r['p_bootstrap']:.4f}  Holm threshold "
                      f"{r['holm_threshold']:.5f}  -> {verdict} after correction")
            else:
                print(f"    p {r['p_bootstrap']:.4f}  (uncorrected -- not a confirmatory test)")
            sd = r["margin_sd_across_folds"]
            print(f"    margin    diff {r['margin_diff']:+.4f}"
                  + (f"  sd {sd:.4f} across {len(r['folds'])} folds" if sd else ""))
            print(f"    by fold   {r['margin_by_fold']}")
            if abs(r["accuracy_diff"]) < soi:
                print(f"    NOTE      |diff| below the registered smallest effect of "
                      f"interest ({soi}); report as underpowered, not as no difference")

    show(conf, f"CONFIRMATORY -- {len(conf)} comparison(s), Holm-Bonferroni corrected", True)

    ref = [compare(kept, a, b, boot) for a, b in reference]
    ref = [r for r in ref if not r.get("error")]
    show(ref, "REFERENCE -- cross-dataset, confounded by construction, UNCORRECTED", False)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "status": "provisional" if provisional else "final",
        "runs_found": len(runs), "runs_expected": expected,
        "excluded_by_probe": excluded, "unreviewed": unreviewed,
        "manipulation_check": controls,
        "confirmatory": conf, "reference": ref,
        "smallest_effect_of_interest": soi,
        "bootstrap": boot, "correction": lock["metrics"]["multiple_comparisons"],
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
