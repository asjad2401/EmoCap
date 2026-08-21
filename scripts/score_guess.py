#!/usr/bin/env python
"""The human-legibility gate: score a blind register-guess export against its key.

    uv run python scripts/score_guess.py ~/Downloads/emocap-guess-text-20260826.json \
        --key runs/guess/text_v10mix_key.json --anchor 0.507

§6 halts the study if the corpus is not more legible to a reader than to a bag of 25
keywords per register. That gate had no implementation; this is it.

**Three scorings, because the raw number alone misleads.** Tasks from 2026-08-20 onward
offer a "no register" option, so a reader can decline rather than guess:

* ``neutral rate``  -- share of captions carrying no register at all. The forced-choice
  tasks hid this: a blank caption got a random guess and was right 20% of the time.
* ``decided-only``  -- accuracy on the captions the reader was willing to place.
* ``forced-choice-equivalent`` -- ``acc_dec x (1 - nr) + 0.2 x nr``, reconstructing what
  the reader would have scored under the old rules. **This is the only figure comparable
  to scores from before the neutral option existed.**

The gate is evaluated on the FCE, against the same corpus's anchor at its own n.

**The comparison is not like-for-like and the output says so.** The anchor is a *trained*
rule doing 5-fold CV over thousands of cells; the reader is zero-shot on a hundred items.
"Keywords beat a human" is therefore not by itself proof of leakage.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTERS = ("joyful", "sad", "tense", "romantic", "humorous")
CHANCE = 1 / len(REGISTERS)


def score(answers: dict, key: dict) -> dict:
    """Three scorings plus per-register recall, for one set of items."""
    ids = [i for i in answers if i in key]
    n = len(ids)
    if not n:
        return {"n": 0}
    neutral = [i for i in ids if answers[i] == "neutral"]
    decided = [i for i in ids if answers[i] != "neutral"]
    ok = sum(1 for i in decided if answers[i] == key[i]["truth"])
    nr = len(neutral) / n
    acc = ok / len(decided) if decided else 0.0
    fce = acc * (1 - nr) + CHANCE * nr
    per: dict[str, dict] = {}
    for r in REGISTERS:
        d = [i for i in decided if key[i]["truth"] == r]
        nn = sum(1 for i in neutral if key[i]["truth"] == r)
        per[r] = {"recall": round(sum(1 for i in d if answers[i] == r) / len(d), 3)
                  if d else None, "decided": len(d), "neutral": nn}
    return {"n": n, "neutral_rate": round(nr, 4), "decided_n": len(decided),
            "acc_decided": round(acc, 4), "raw": round(ok / n, 4),
            "fce": round(fce, 4), "se": round(math.sqrt(fce * (1 - fce) / n), 4),
            "per_register": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="emocap-guess-*.json from the task page")
    ap.add_argument("--key", required=True, help="runs/guess/*_key.json")
    ap.add_argument("--anchor", type=float, required=True,
                    help="the SAME corpus's keyword anchor, at its own n")
    ap.add_argument("--anchor-n", type=int, default=None, help="cells the anchor used")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exp = json.loads(Path(args.export).read_text())
    kf = json.loads(Path(args.key).read_text())
    # A mismatched key silently scores against the wrong answers, so refuse rather than warn.
    if (exp.get("seed"), exp.get("source")) != (kf.get("seed"), kf.get("source")):
        raise SystemExit(f"export/key mismatch: export seed={exp.get('seed')} "
                         f"source={exp.get('source')} vs key seed={kf.get('seed')} "
                         f"source={kf.get('source')}")
    key = kf["key"]
    answers = exp["answers"]

    arms = {v.get("arm") for v in key.values()}
    groups = ({a: {i: v for i, v in key.items() if v.get("arm") == a} for a in sorted(arms)}
              if arms != {None} else {exp.get("source", "corpus"): key})

    print(f"export   {Path(args.export).name}")
    print(f"key      {Path(args.key).name}   source {kf.get('source')}")
    print(f"anchor   {args.anchor:.3f}" + (f" at n={args.anchor_n:,} cells" if args.anchor_n
                                           else "  (n NOT given -- §4 requires it)"))
    result: dict = {"export": Path(args.export).name, "anchor": args.anchor,
                    "anchor_n": args.anchor_n, "arms": {}}
    worst_pass = True
    for name, sub in groups.items():
        s = score(answers, sub)
        if not s["n"]:
            continue
        passes = s["fce"] > args.anchor
        worst_pass &= passes
        s["passes_legibility_gate"] = passes
        result["arms"][name] = s
        print(f"\n{name}")
        print(f"  neutral rate   {s['neutral_rate']:.3f}  ({int(s['neutral_rate']*s['n'])}/{s['n']})")
        print(f"  decided-only   {s['acc_decided']:.3f}  (n={s['decided_n']})")
        print(f"  FCE            {s['fce']:.3f} +-{s['se']:.3f}   <- comparable across tasks")
        print(f"  vs anchor      {s['fce'] - args.anchor:+.3f}   "
              f"{'PASSES' if passes else 'FAILS -- HALT'}")
        print("  per register   " + "  ".join(
            f"{r[:3]} {v['recall'] if v['recall'] is not None else '-'}" for r, v in
            s["per_register"].items()))

    print(f"\n{'=' * 60}")
    print("GATE: " + ("PASSES" if worst_pass else "FAILS -- HALT"))
    print("Caveat, always reported: the anchor is a TRAINED rule (5-fold CV over thousands\n"
          "of cells); the reader is zero-shot on ~100 items. A keyword rule beating a reader\n"
          "is not by itself proof of leakage. §6 commits to >=300 items before this gate is\n"
          "applied to the finished corpus.")
    result["passes"] = worst_pass
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
