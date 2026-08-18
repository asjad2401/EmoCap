#!/usr/bin/env python
"""Cost the full stage-02 run from measured per-call token counts.

Separate from the probe so costs can be recomputed when prices change, or when a
probe has already run, without spending another call. Reads prices from
`configs/data.yaml` -- never hardcoded, because they move (3.6-flash's introductory
rate ends 2026-12-31 and then doubles).

Two things this makes visible that a naive estimate misses:

* **Thinking tokens are billed at the output rate and are not counted in
  ``candidates_token_count``.** They are a surcharge on top of visible output, and
  only the *lite* models honour ``thinking_budget=0``.
* **Option B's saving is mostly image tokens.** The image costs a flat 1,064 tokens
  regardless of resolution, so sending it once per image rather than once per caption
  removes 4/5 of the largest single input component.

Usage:
    uv run python scripts/cost_report.py                      # from probe measurements
    uv run python scripts/cost_report.py --batch              # apply the batch discount
    uv run python scripts/cost_report.py --thinking 96        # override thinking/call
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.runtime import load_config  # noqa: E402

# Measured 2026-08-18 with the API's own token counter (scripts/cost_probe.py).
MEASURED = {
    "prompt_text_single": 834,    # multimodal wording, one source caption
    "prompt_text_batch": 975,     # five source captions + batch instructions (est.)
    "image_tokens": 1064,         # FLAT across 224px / 384px / 512px / native
    "output_per_caption": 27,     # 5 captions ~= 133 tokens/call, measured
}
CAPTIONS_PER_IMAGE = 25           # 5 human captions x 5 registers
N_IMAGES = 8091
N_CAPTION_ROWS = 40455            # 8,091 x 5 human captions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true", help="apply the batch-API discount")
    ap.add_argument("--thinking", type=int, default=None,
                    help="thinking tokens per call to assume (default: per-model measured)")
    ap.add_argument("--standard-rates", action="store_true",
                    help="use post-introductory rates where a model has them")
    args = ap.parse_args()

    cfg = load_config("data")
    pricing = cfg["pricing"]
    models = pricing["models"]
    discount = pricing["batch_api_discount"] if args.batch else 0.0

    # Measured on a trivial prompt, 2026-08-18. Lite honours thinking_budget=0.
    measured_thinking = {
        "gemini-3.5-flash": 77,
        "gemini-3.6-flash": 96,
        "gemini-3.1-flash-lite": 0,
    }

    total_captions = N_IMAGES * CAPTIONS_PER_IMAGE
    out_per_caption = MEASURED["output_per_caption"]

    plans = {
        "A (one call per caption)": {
            "calls": N_CAPTION_ROWS,
            "in_per_call": MEASURED["prompt_text_single"] + MEASURED["image_tokens"],
            "out_per_call": 5 * out_per_caption,
        },
        "B (one call per image)": {
            "calls": N_IMAGES,
            "in_per_call": MEASURED["prompt_text_batch"] + MEASURED["image_tokens"],
            "out_per_call": 25 * out_per_caption,
        },
    }

    print(f"prices as of {pricing['as_of']} ({pricing['source']})")
    print(f"batch discount: {'applied ' + str(int(discount*100)) + '%' if discount else 'not applied'}")
    print(f"target: {total_captions:,} captions over {N_IMAGES:,} images\n")

    rows = []
    for plan_name, p in plans.items():
        in_tok = p["calls"] * p["in_per_call"]
        for m, price in models.items():
            rates = price.get("standard", price) if args.standard_rates else price
            think = args.thinking if args.thinking is not None else measured_thinking.get(m, 0)
            out_visible = p["calls"] * p["out_per_call"]
            out_think = p["calls"] * think
            out_tok = out_visible + out_think
            cost_in = in_tok / 1e6 * rates["input"]
            cost_out = out_tok / 1e6 * rates["output"]
            total = (cost_in + cost_out) * (1 - discount)
            rows.append({
                "plan": plan_name, "model": m, "calls": p["calls"],
                "in_tok": in_tok, "out_visible": out_visible, "out_think": out_think,
                "cost_in": cost_in * (1 - discount), "cost_out": cost_out * (1 - discount),
                "total": total,
                "think_share": cost_out and (out_think / max(1, out_tok)),
            })

    hdr = (f"{'plan':<26}{'model':<24}{'calls':>8}{'in Mtok':>9}{'out Mtok':>10}"
           f"{'(think)':>9}{'$in':>9}{'$out':>9}{'TOTAL':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["total"]):
        print(f"{r['plan']:<26}{r['model']:<24}{r['calls']:>8,}"
              f"{r['in_tok']/1e6:>9.1f}{(r['out_visible']+r['out_think'])/1e6:>10.1f}"
              f"{r['out_think']/1e6:>9.1f}{r['cost_in']:>9.2f}{r['cost_out']:>9.2f}"
              f"{r['total']:>10.2f}")

    cheapest = min(rows, key=lambda x: x["total"])
    dearest = max(rows, key=lambda x: x["total"])
    print(f"\ncheapest: {cheapest['plan']} / {cheapest['model']} = ${cheapest['total']:,.2f}")
    print(f"dearest : {dearest['plan']} / {dearest['model']} = ${dearest['total']:,.2f}"
          f"  ({dearest['total']/cheapest['total']:.0f}x)")

    # What the levers are actually worth, holding the model fixed.
    print("\nlevers, holding the model fixed:")
    for m in models:
        a = next(r for r in rows if r["model"] == m and r["plan"].startswith("A"))
        b = next(r for r in rows if r["model"] == m and r["plan"].startswith("B"))
        print(f"  {m:<24} batching A->B saves ${a['total']-b['total']:>8,.2f}"
              f"  ({(1-b['total']/a['total'])*100:.0f}%)")
    print("\n  image tokens are a flat 1,064 per call at ANY resolution, so downscaling")
    print("  saves nothing -- batching is the only real input-side lever.")
    for m, t in measured_thinking.items():
        if t and m in models:
            r = next(x for x in rows if x["model"] == m and x["plan"].startswith("B"))
            share = r["out_think"] / (r["out_visible"] + r["out_think"]) * 100
            print(f"  {m:<24} thinking is {share:.0f}% of billed output "
                  f"({t} tok/call, cannot be disabled)")


if __name__ == "__main__":
    main()
