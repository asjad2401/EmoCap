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

**The margin gets an interval too, and that took decomposing the anchor.** The study claims
the margin, not the accuracy, and every registered magnitude criterion -- P2, P3b, P6 -- is
stated in margin points. Yet the anchor was only ever a whole-corpus number, so a margin
difference could be reported as a fold mean with a fold spread and nothing more. A criterion
of ">=7 points" cannot be judged against a quantity with no interval on it.

`keyword_rule_cell_scores` now decomposes the SAME registered estimator to the cell -- same
keyword fitting, same cross-validation by image, same fractional tie credit, walking the same
generator as the point estimate -- and `score_arm.py` writes that credit into `cells.jsonl`
beside the classifier verdict. The margin is therefore a per-cell quantity here, resampled by
the registered cluster bootstrap exactly like accuracy. The fold mean and fold spread are
still reported next to it, because the two are computed from different weightings of the same
estimator and a disagreement between them is worth seeing.

**Post-hoc comparisons are computed here but corrected nowhere.** The two arms decided
after the tag -- `S_paired_matched` and `S_unpaired_scaled` -- carry the comparisons that
actually separate paired structure from data volume, which the registered P3 cannot. They
get the identical estimator every registered comparison gets, because a result reported as
two eyeballed means when an interval was available is weaker than it needs to be. They do
NOT enter the Holm family: the registration fixes that family at exactly four comparisons,
and quietly growing it to six would change every threshold in the table. Their pairs come
from `data/arms/posthoc_manifest.json`, so this file cannot invent one.

**Which metric the Holm correction attaches to is NOT settled by the registration.**
`metrics.primary` is `emotion_accuracy`, and `multiple_comparisons: holm_bonferroni` names
the family of four comparisons without naming a metric. Rather than pick one after seeing the
results, both are corrected over the same four comparisons and both are reported. Where they
disagree, that disagreement is the finding and belongs in the paper.
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
from emocap.runtime import load_config, rel# noqa: E402

#: Post-hoc comparisons, with the reading of each fixed here rather than after the numbers.
#: Left vs right, and what a positive margin difference would mean.
POSTHOC_PAIRS = [
    (("S_paired_matched", "S_unpaired"),
     "volume matched at 4,390 cells; LEFT is paired, RIGHT is not. A positive margin gap "
     "means paired structure helps at a small budget."),
    (("S_paired5", "S_unpaired_scaled"),
     "volume matched at 40,235 cells and the same 8,047 images; LEFT varies register "
     "within an image, RIGHT varies source text. A positive margin gap ISOLATES pairing "
     "from volume -- this is the comparison P3 could not make."),
]

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


def _collapse(run: dict) -> dict[tuple[str, str], tuple[float, float]]:
    """``(image, register) -> (classifier correctness, anchor credit)`` for one run.

    A cell can appear more than once in an arm -- S_paired25 holds five source captions per
    (image, register) -- so duplicates are averaged. That makes one cell one observation in
    both arms, which is what "matched images" has to mean for the comparison to be about the
    arms rather than about how many rows each one happens to have.

    The anchor is keyed by (image, register) upstream, so every duplicate row carries the
    same anchor credit and averaging it is a no-op. That is forced by the estimator's own
    data structure, which holds one caption per (image, register): in S_paired25 the anchor
    side is measured on one representative caption per key while the classifier side uses
    all five. Worth stating in the paper; it is not something this script can fix.
    """
    acc: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for c in run["cells"]:
        acc.setdefault((c["image_id"], c["emotion"]), []).append(
            (float(c["correct"]), float(c.get("anchor", 0.0))))
    return {k: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v))
            for k, v in acc.items()}


def paired_cells(a: dict, b: dict):
    """Correctness, anchor credit and image id for the cells the two runs share."""
    ca, cb = _collapse(a), _collapse(b)
    keys = sorted(set(ca) & set(cb))
    return ([ca[k] for k in keys], [cb[k] for k in keys], [k[0] for k in keys])


def _all_cells(run: dict, fold: int):
    """Every row of one run: correctness, margin (correct - anchor), and its cluster."""
    vals = [float(c["correct"]) for c in run["cells"]]
    marg = [float(c["correct"]) - float(c.get("anchor", 0.0)) for c in run["cells"]]
    clus = [f"{fold}:{c['image_id']}" for c in run["cells"]]
    return vals, marg, clus


def _has_anchor_cells(runs: dict[str, dict]) -> list[str]:
    """Tags whose cells.jsonl predates the per-cell anchor. They cannot carry a margin CI."""
    return sorted(t for t, r in runs.items()
                  if r["cells"] and "anchor" not in r["cells"][0])


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
    lm, rm = [], []
    la_all, ra_all, lm_all, rm_all, lc_all, rc_all = [], [], [], [], [], []
    for f in folds:
        a, b = fl[f], fr[f]
        x, y, imgs = paired_cells(a, b)
        la += [v for v, _ in x]
        ra += [v for v, _ in y]
        # Per-cell margin: the classifier's verdict minus the keyword rule's credit on the
        # SAME cell. This is what makes a margin difference resamplable at all.
        lm += [v - anc for v, anc in x]
        rm += [v - anc for v, anc in y]
        # Fold number joins the image id so the same image in different folds is not
        # resampled as one cluster.
        clusters += [f"{f}:{i}" for i in imgs]
        fold_margin.append((a["accuracy"] - a["anchor"]) - (b["accuracy"] - b["anchor"]))
        xa, xm, xc = _all_cells(a, f)
        ya, ym, yc = _all_cells(b, f)
        la_all += xa; lm_all += xm; lc_all += xc
        ra_all += ya; rm_all += ym; rc_all += yc

    def _boot_paired(diffs: list[float]) -> tuple[dict, list[float]]:
        bs = cluster_bootstrap_mean(diffs, clusters, n_resamples=boot["resamples"],
                                    ci=boot["ci"], seed=42, return_samples=True)
        return bs, bs.pop("samples", [])

    def _boot_unpaired(lv, lc, rv, rc) -> tuple[dict, list[float]]:
        """Two independent bootstraps, differenced. Wider, correctly: nothing is matched."""
        bl = cluster_bootstrap_mean(lv, lc, n_resamples=boot["resamples"],
                                   ci=boot["ci"], seed=42, return_samples=True)
        br = cluster_bootstrap_mean(rv, rc, n_resamples=boot["resamples"],
                                   ci=boot["ci"], seed=43, return_samples=True)
        sl, sr = bl.pop("samples", []), br.pop("samples", [])
        # cluster_bootstrap_mean returns its resamples SORTED. Differencing two sorted
        # arrays element-wise subtracts matched quantiles, which cancels nearly all the
        # variance and produced an interval an order of magnitude too narrow. Shuffling
        # restores the independent pairing the two bootstraps actually have.
        rng = random.Random(4242)
        rng.shuffle(sl)
        rng.shuffle(sr)
        diffed = sorted(a - b for a, b in zip(sl, sr))
        alpha = (1.0 - boot["ci"]) / 2.0
        return ({"mean": bl["mean"] - br["mean"],
                 "lo": round(diffed[int(alpha * (len(diffed) - 1))], 4),
                 "hi": round(diffed[int((1 - alpha) * (len(diffed) - 1))], 4),
                 "n_clusters": bl["n_clusters"] + br["n_clusters"],
                 "left": bl["mean"], "right": br["mean"]}, diffed)

    is_paired = bool(la)
    if is_paired:
        bs, samples = _boot_paired([x - y for x, y in zip(la, ra)])
        mbs, msamples = _boot_paired([x - y for x, y in zip(lm, rm)])
        left_acc, right_acc = sum(la) / len(la), sum(ra) / len(ra)
        n_paired = len(la)
    else:
        bs, samples = _boot_unpaired(la_all, lc_all, ra_all, rc_all)
        mbs, msamples = _boot_unpaired(lm_all, lc_all, rm_all, rc_all)
        left_acc, right_acc = bs["left"], bs["right"]
        n_paired = 0
    p = _pvalue(samples)
    mp = _pvalue(msamples)

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
        # The margin, bootstrapped by the registered estimator on per-cell anchor credit.
        # `margin_diff_bootstrap` weights every cell equally; `margin_diff` averages the
        # five folds' registered point estimates. They answer the same question under two
        # weightings, and both are reported so a gap between them is visible.
        "margin_diff_bootstrap": round(mbs["mean"], 4),
        "margin_ci": {"lo": mbs["lo"], "hi": mbs["hi"]},
        "margin_p_bootstrap": mp,
        "margin_crosses_zero": None if mbs["lo"] is None else bool(mbs["lo"] <= 0 <= mbs["hi"]),
        "margin_diff": round(mean_margin, 4),
        "margin_sd_across_folds": round(sd_margin, 4) if sd_margin is not None else None,
        "margin_by_fold": [round(m, 4) for m in fold_margin],
        "left_accuracy": round(left_acc, 4),
        "right_accuracy": round(right_acc, 4),
    }


def holm(results: list[dict], alpha: float = 0.05, *, field: str = "p_bootstrap",
         prefix: str = "") -> None:
    """Holm-Bonferroni, in place. Applied to the confirmatory set and nothing else.

    ``field``/``prefix`` exist because the registration fixes the family of four
    comparisons but never says which metric the correction attaches to: `metrics.primary`
    is accuracy, while every magnitude criterion is written in margin points. Both are
    corrected over the same four comparisons rather than choosing one after seeing results.
    """
    live = [r for r in results if r.get(field) is not None]
    m = len(live)
    for rank, r in enumerate(sorted(live, key=lambda r: r[field])):
        r[f"{prefix}holm_threshold"] = round(alpha / (m - rank), 5)
        r[f"{prefix}holm_rank"] = rank + 1
    # Holm is a step-down procedure: once one test fails, every larger p fails too,
    # regardless of its own threshold. Comparing each p to its own threshold in isolation
    # is the usual way this gets implemented wrong, and it inflates the error rate.
    stopped = False
    for r in sorted(live, key=lambda r: r[field]):
        if stopped or r[field] > r[f"{prefix}holm_threshold"]:
            stopped = True
            r[f"{prefix}significant_holm"] = False
        else:
            r[f"{prefix}significant_holm"] = True


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
    crit_pts = study["criterion_points"]

    runs_dir = ROOT / args.runs
    runs = load_runs(runs_dir)
    vpath = ROOT / args.verdicts
    verdicts = json.loads(vpath.read_text()) if vpath.exists() else {}
    kept, excluded, unreviewed = apply_exclusions(runs, verdicts)

    # A margin interval needs the per-cell anchor credit. Runs scored before that existed
    # would silently contribute anchor 0.0 and inflate every margin they touch, so this
    # stops rather than reporting a number built from two different definitions.
    stale = _has_anchor_cells(kept)
    if stale:
        raise SystemExit(
            f"{len(stale)} run(s) have a cells.jsonl with no per-cell anchor, so no margin\n"
            f"interval can be computed. Re-score them:\n"
            + "".join(f"    uv run python scripts/score_arm.py --run {runs_dir/t}\n"
                      for t in stale))

    expected = len(study["arms"]) * study["folds"] + len(study["arms"])
    complete = len(runs) >= expected and not unreviewed
    provisional = not complete

    extra = len(runs) - expected
    print(f"runs found        {len(runs)}"
          + (f"  ({expected} registered + {extra} post-hoc)" if extra > 0
             else f" of {expected} registered"))
    print(f"excluded by probe {len(excluded)}" + (f"  {excluded}" if excluded else ""))
    if not vpath.exists():
        print(f"\n  NO VERDICT FILE at {rel(vpath)}")
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
    holm(conf, field="margin_p_bootstrap", prefix="margin_")

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
            mlo, mhi = r["margin_ci"]["lo"], r["margin_ci"]["hi"]
            sd = r["margin_sd_across_folds"]
            print(f"    margin    diff {r['margin_diff_bootstrap']:+.4f}  "
                  f"95% CI [{mlo:+.4f}, {mhi:+.4f}]")
            if corrected:
                mverdict = ("significant" if r.get("margin_significant_holm")
                            else "not significant")
                print(f"    p {r['margin_p_bootstrap']:.4f}  Holm threshold "
                      f"{r['margin_holm_threshold']:.5f}  -> {mverdict} after correction")
            else:
                print(f"    p {r['margin_p_bootstrap']:.4f}  (uncorrected)")
            print(f"    fold mean {r['margin_diff']:+.4f}"
                  + (f"  sd {sd:.4f} across {len(r['folds'])} folds" if sd else "")
                  + f"   by fold {r['margin_by_fold']}")
            # The registered criteria for P2, P3b and P6 are all "the margin gap is at
            # least `criterion_points`". A gap that clears zero but not the criterion is a
            # real effect that is nonetheless smaller than the study said was worth
            # claiming, and those two verdicts must not be reported as one.
            crit = "criterion" if abs(r["margin_diff_bootstrap"]) >= crit_pts else None
            if crit:
                print(f"    CRITERION |margin| {abs(r['margin_diff_bootstrap']):.4f} "
                      f">= registered {crit_pts} -- criterion met")
            elif not r["margin_crosses_zero"]:
                print(f"    CRITERION |margin| {abs(r['margin_diff_bootstrap']):.4f} "
                      f"< registered {crit_pts}, but the CI excludes zero -- a real gap "
                      f"BELOW the magnitude the prereg set as worth claiming")
            else:
                print(f"    CRITERION |margin| {abs(r['margin_diff_bootstrap']):.4f} "
                      f"< registered {crit_pts} and the CI includes zero")
            if abs(r["accuracy_diff"]) < soi:
                print(f"    NOTE      |accuracy diff| below the registered smallest effect "
                      f"of interest ({soi}); report as underpowered, not as no difference")

    show(conf, f"CONFIRMATORY -- {len(conf)} comparison(s), Holm-Bonferroni corrected", True)

    ref = [compare(kept, a, b, boot) for a, b in reference]
    ref = [r for r in ref if not r.get("error")]
    show(ref, "REFERENCE -- cross-dataset, confounded by construction, UNCORRECTED", False)

    # Post-hoc. Same estimator, deliberately outside the Holm family -- see the module
    # docstring. Only pairs whose arms are named in the post-hoc manifest are attempted, so
    # a typo here cannot silently invent a comparison.
    ph_manifest = ROOT / "data/arms/posthoc_manifest.json"
    known = set(json.loads(ph_manifest.read_text())["arms"]) if ph_manifest.exists() else set()
    posthoc, notes = [], {}
    for (left, right), reading in POSTHOC_PAIRS:
        if not (known & {left, right}):
            continue
        r = compare(kept, left, right, boot)
        if r.get("error"):
            print(f"\n  {left} vs {right}: {r['error']} -- not yet trained?")
            continue
        r["post_hoc"] = True
        r["reading"] = reading
        posthoc.append(r)
        notes[f"{left} vs {right}"] = reading
    if posthoc:
        show(posthoc, "POST-HOC -- decided after the tag, NOT in the Holm family", False)
        for r in posthoc:
            print(f"\n  {r['left']} vs {r['right']}: {r['reading']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "status": "provisional" if provisional else "final",
        "runs_found": len(runs), "runs_expected": expected,
        "excluded_by_probe": excluded, "unreviewed": unreviewed,
        "manipulation_check": controls,
        "confirmatory": conf, "reference": ref,
        "post_hoc": posthoc, "post_hoc_readings": notes,
        "post_hoc_note": "same estimator as the confirmatory comparisons and deliberately "
                         "outside the Holm family, which the registration fixes at four",
        "smallest_effect_of_interest": soi, "criterion_points": crit_pts,
        "bootstrap": boot, "correction": lock["metrics"]["multiple_comparisons"],
    }, indent=2))
    print(f"\nwrote {rel(out)}")


if __name__ == "__main__":
    main()
