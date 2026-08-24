#!/usr/bin/env python
"""Score the registered human evaluation. Reliability first, then the arm means.

    uv run python scripts/score_human_eval.py
    uv run python scripts/score_human_eval.py --results runs/human-eval --key runs/human-eval/key.json

Reads every rater export in ``--results`` and the answer key the raters never saw.

**Reliability is reported before the arm means, and that ordering is deliberate.** Three
people rating 150 captions produce numbers whether or not they agree with each other. If
Krippendorff's alpha is near zero the scales did not measure a shared construct, and the
per-arm means underneath are three people's private impressions averaged together. So alpha
comes first, on both scales, and no threshold is applied here: the lock registers
``reliability: krippendorff_alpha`` and asks for it to be reported, not to gate anything.

**Alpha is computed with the ordinal difference function**, not nominal. On a 1-5 scale a
3-vs-4 disagreement is smaller than a 1-vs-5 disagreement, and nominal alpha treats them as
identical -- which understates agreement badly on rating data and is the usual way this
statistic gets misreported.

**Attention checks never enter the arm means.** They are scored separately, per rater, under
the rule fixed in ``make_human_eval.py`` before any rating existed: a check passes when the
scale it targets is <= 2. A rater failing more than half of theirs is FLAGGED and their
numbers are still reported. Dropping a rater is a judgement to be made and written down, not
something a script does silently -- the same discipline the visual-dependence probe uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.eval.bootstrap import cluster_bootstrap_mean  # noqa: E402
from emocap.eval.human_eval import krippendorff_alpha  # noqa: E402
from emocap.runtime import load_config  # noqa: E402

SCALES = ("tone", "ground")
LABEL = {"tone": "tone-match", "ground": "grounding"}


def load_results(results_dir: Path, build: str) -> dict[str, dict]:
    """Rater id -> answers. Refuses exports built from a different task file."""
    out: dict[str, dict] = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "key.json":
            continue
        d = json.loads(p.read_text())
        if "answers" not in d or "rater" not in d:
            continue
        if d.get("build") != build:
            raise SystemExit(
                f"{p.name} was produced from task build {d.get('build')!r}, but the key is "
                f"{build!r}. Those are different item sets and must not be pooled.")
        rid = str(d["rater"]).strip()
        if rid in out:
            raise SystemExit(f"two exports claim rater {rid!r}; rename one")
        out[rid] = d["answers"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/human-eval")
    ap.add_argument("--key", default="runs/human-eval/key.json")
    ap.add_argument("--out", default="results/human_eval.json")
    args = ap.parse_args()

    key_path = ROOT / args.key
    if not key_path.exists():
        raise SystemExit(f"no key at {key_path} -- run scripts/make_human_eval.py first")
    keyf = json.loads(key_path.read_text())
    key, build = keyf["key"], keyf["build"]

    raters = load_results(ROOT / args.results, build)
    if not raters:
        raise SystemExit(
            f"no rater exports in {args.results}/ (key.json aside).\n"
            f"Collect each rater's downloaded JSON into that folder and re-run.")

    lock = load_config("prereg.lock")
    spec, boot = lock["human_eval"], lock["metrics"]["bootstrap"]
    scored = {k: v for k, v in key.items() if v["kind"] == "real"}
    checks = {k: v for k, v in key.items() if v["kind"] != "real"}

    print(f"task build {build}   {len(scored)} scored items, {len(checks)} checks")
    print(f"raters found {len(raters)} of {spec['raters']} registered: "
          f"{', '.join(sorted(raters))}")
    for rid, ans in sorted(raters.items()):
        done = sum(1 for k in scored if ans.get(k, {}).get("tone")
                   and ans.get(k, {}).get("ground"))
        print(f"  {rid:<10} {done}/{len(scored)} scored items complete")

    # ── attention checks, before anything else is believed ───────────────────
    print(f"\nATTENTION CHECKS   pass = the targeted scale is <= 2 "
          f"({keyf['check_pass_rule']})")
    flagged = []
    check_report = {}
    for rid, ans in sorted(raters.items()):
        rated = [(k, v) for k, v in checks.items() if ans.get(k, {}).get(v["expect_low"])]
        passed = sum(1 for k, v in rated if ans[k][v["expect_low"]] <= 2)
        rate = passed / len(rated) if rated else None
        bad = rate is not None and rate <= 0.5
        if bad:
            flagged.append(rid)
        check_report[rid] = {"rated": len(rated), "passed": passed, "rate": rate}
        s = "n/a" if rate is None else f"{passed}/{len(rated)} = {rate:.0%}"
        print(f"  {rid:<10} {s}" + ("   FLAGGED -- fails more than half" if bad else ""))
    if flagged:
        print(f"  {len(flagged)} rater(s) flagged. Their ratings are still included below.")
        print("  Excluding a rater is a judgement to record in docs/deviations.md, not "
              "something this script does.")

    # ── reliability ──────────────────────────────────────────────────────────
    print("\nRELIABILITY   Krippendorff's alpha, ordinal difference function")
    alphas = {}
    for sc in SCALES:
        units = []
        for k in scored:
            vals = [raters[r][k][sc] for r in raters
                    if k in raters[r] and raters[r][k].get(sc)]
            units.append(vals)
        a = krippendorff_alpha(units, [1, 2, 3, 4, 5])
        alphas[sc] = a
        n_multi = sum(1 for u in units if len(u) >= 2)
        if a is None:
            print(f"  {LABEL[sc]:<12} undefined -- too few doubly-rated items "
                  f"({n_multi}), or no variance")
        else:
            print(f"  {LABEL[sc]:<12} alpha {a:+.3f}   over {n_multi} items rated by "
                  f"2+ raters")
    if len(raters) < 2:
        print("  Only one rater: alpha needs at least two and nothing above is agreement.")

    # ── per-arm means ────────────────────────────────────────────────────────
    print("\nPER-ARM MEANS   averaged across raters per item, then bootstrapped by image")
    arms = sorted({v["arm"] for v in scored.values()})
    per_arm: dict[str, dict] = {}
    for arm in arms:
        ids = [k for k, v in scored.items() if v["arm"] == arm]
        row = {"n_items": len(ids)}
        for sc in SCALES:
            vals, clus = [], []
            for k in ids:
                got = [raters[r][k][sc] for r in raters
                       if k in raters[r] and raters[r][k].get(sc)]
                if got:
                    vals.append(sum(got) / len(got))
                    clus.append(scored[k]["image_id"])
            if not vals:
                continue
            bs = cluster_bootstrap_mean(vals, clus, n_resamples=boot["resamples"],
                                        ci=boot["ci"], seed=42)
            row[sc] = {"mean": round(bs["mean"], 3), "lo": bs["lo"], "hi": bs["hi"],
                       "n_rated": len(vals)}
        per_arm[arm] = row

    hdr = "".join(f"{LABEL[s]:>26}" for s in SCALES)
    print(f"  {'arm':<18}{'n':>5}{hdr}")
    for arm in arms:
        r = per_arm[arm]
        cells = ""
        for sc in SCALES:
            c = r.get(sc)
            cells += (f"{c['mean']:>10.2f} [{c['lo']:.2f},{c['hi']:.2f}]"
                      if c else f"{'--':>26}")
        print(f"  {arm:<18}{r['n_items']:>5}{cells}")

    # Pairwise differences. Not registered as confirmatory comparisons -- the lock names
    # human_eval as a confirmatory METRIC without naming arm pairs -- so these are printed
    # uncorrected and labelled, never Holm-corrected as if they were the registered four.
    print("\n  pairwise differences (uncorrected -- the lock names no human-eval pairs)")
    pairs = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            line = f"  {a} - {b}   "
            rec = {"left": a, "right": b}
            for sc in SCALES:
                ida = [k for k, v in scored.items() if v["arm"] == a]
                idb = [k for k, v in scored.items() if v["arm"] == b]

                def mean_of(ids, sc=sc):
                    out = []
                    for k in ids:
                        got = [raters[r][k][sc] for r in raters
                               if k in raters[r] and raters[r][k].get(sc)]
                        if got:
                            out.append(sum(got) / len(got))
                    return sum(out) / len(out) if out else None

                ma, mb = mean_of(ida), mean_of(idb)
                d = None if ma is None or mb is None else round(ma - mb, 3)
                rec[sc] = d
                line += f"{LABEL[sc]} {d:+.2f}   " if d is not None else f"{LABEL[sc]} --   "
            pairs.append(rec)
            print(line)
    print("  Arms do not share photographs here by construction, so these are unpaired "
          "differences of independent item sets.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "build": build, "registered": {k: spec[k] for k in sorted(spec)},
        "raters": sorted(raters), "n_raters": len(raters),
        "attention_checks": check_report, "flagged_raters": flagged,
        "krippendorff_alpha_ordinal": {k: (round(v, 4) if v is not None else None)
                                       for k, v in alphas.items()},
        "per_arm": per_arm, "pairwise_uncorrected": pairs,
        "bootstrap": boot,
    }, indent=2))
    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {rel}")


if __name__ == "__main__":
    main()
