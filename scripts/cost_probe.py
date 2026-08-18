#!/usr/bin/env python
"""Stage-02 configuration probe: decide the generation config before spending on it.

Runs a small paired matrix of real Gemini calls and reports everything needed to
choose, defensibly, between the options we cannot settle by reasoning:

  1. **Does the image matter at all?**  arm `A-text` vs `A-384`. If multimodal output
     barely differs from text-only, the multimodal decision is unjustified and drops
     5x the input tokens.
  2. **What resolution?**  `A-224` vs `A-384` vs `A-512`. If outputs are near-identical
     across resolutions, take the cheapest.
  3. **One call per caption, or one per image?**  `A-384` (5 outputs) vs `B-384` (25).
     Decided by completion and rejection rate *by output position* -- the specific way
     a 25-field response is expected to fail.
  4. **Does a response schema help?**  `B-384-schema` vs `B-384-noschema`.

Every arm runs on the *same* images and captions, so all comparisons are paired.
`max_attempts=1` throughout, so measured cost is the base cost; the rejection rate is
reported separately as the retry tail.

Prices are a user-supplied input, never hardcoded -- a stale rate in a research repo
is worse than no rate.

Usage:
    uv run python scripts/cost_probe.py --n 8
    uv run python scripts/cost_probe.py --n 8 --in-price 0.10 --out-price 0.40
    uv run python scripts/cost_probe.py --n 8 --throughput 12
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data import drop_malformed, read_captions, read_image_list  # noqa: E402
from emocap.data.generate import (  # noqa: E402
    gemini_llm,
    generate_image_batch,
    generate_one,
    load_image_bytes,
    validate_all,
)
from emocap.data.prompt import EMOTIONS, build_batch_prompt, build_prompt  # noqa: E402
from emocap.data.quality import (  # noqa: E402
    config_divergence,
    grounding_report,
    length_stats,
    pairwise_similarity,
    register_divergence,
)
from emocap.eval import cluster_bootstrap_mean, cluster_bootstrap_paired_diff, is_conclusive  # noqa: E402
from emocap.runtime import Manifest, load_config  # noqa: E402

PROBE_ROOT = ROOT / "results" / "probe"


def load_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not key:
        sys.exit("No API key. Put GEMINI_API_KEY=... in .env (gitignored), or export it.")
    return key


def summarise(records: list[dict], label: str, full_calls: int) -> dict:
    """Collapse one arm's per-call records into the numbers that decide things."""
    ok = [r for r in records if not r.get("error")]
    caps = [c for r in ok for c in r["captions"].values() if c]
    rejects = [why for r in ok for why in r["rejects"].values()]

    pin = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    pout = [r["output_tokens"] for r in ok if r.get("output_tokens")]
    lat = [r["latency_s"] for r in ok if r.get("latency_s")]

    expected = sum(r["expected_captions"] for r in ok)
    by_rule: Counter = Counter()
    for why in rejects:
        by_rule[why.split(",")[0].split(":")[0][:34]] += 1

    recalls = [r["grounding"]["content_recall"] for r in ok if r.get("grounding")]
    novel = [r["grounding"]["novel_rate"] for r in ok if r.get("grounding")]
    divs = [r["divergence"]["mean_similarity"] for r in ok
            if r.get("divergence") and r["divergence"].get("mean_similarity") is not None]

    # Clustered by image: captions from one image share a picture and a call, so
    # per-caption resampling would report intervals several times too narrow.
    comp_vals, comp_cl, rej_vals, rej_cl = [], [], [], []
    for r in ok:
        img = r.get("image_id")
        exp = r["expected_captions"]
        got = len([c for c in r["captions"].values() if c])
        comp_vals.append(got / exp if exp else 0.0)
        comp_cl.append(img)
        for e, c in r["captions"].items():
            if c:
                rej_vals.append(1.0 if e in r["rejects"] else 0.0)
                rej_cl.append(img)
    comp_ci = cluster_bootstrap_mean(comp_vals, comp_cl, n_resamples=4000)
    rej_ci = cluster_bootstrap_mean(rej_vals, rej_cl, n_resamples=4000)

    out = {
        "arm": label,
        "calls": len(records),
        "completion_ci": comp_ci,
        "rejection_ci": rej_ci,
        "errors": len(records) - len(ok),
        "captions_returned": len(caps),
        "captions_expected": expected,
        "completion_rate": round(len(caps) / expected, 4) if expected else 0.0,
        "rejection_rate": round(len(rejects) / max(1, len(caps)), 4),
        "rejections_by_rule": dict(by_rule.most_common()),
        "truncated": sum(1 for r in ok if r.get("finish_reason")
                         and "MAX_TOKEN" in str(r["finish_reason"]).upper()),
        "input_tokens_mean": round(stats.mean(pin), 1) if pin else None,
        "output_tokens_mean": round(stats.mean(pout), 1) if pout else None,
        "latency_mean_s": round(stats.mean(lat), 2) if lat else None,
        "latency_p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else None,
        "length": length_stats(caps),
        "content_recall_mean": round(stats.mean(recalls), 4) if recalls else None,
        "novel_rate_mean": round(stats.mean(novel), 4) if novel else None,
        "register_similarity_mean": round(stats.mean(divs), 4) if divs else None,
    }
    if pin and pout:
        out["full_run_input_tokens"] = int(stats.mean(pin) * full_calls)
        out["full_run_output_tokens"] = int(stats.mean(pout) * full_calls)
        out["full_run_calls"] = full_calls
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="images for the A arms (each contributes 5 calls)")
    ap.add_argument("--n-batch", type=int, default=None,
                    help="images for the B arms; defaults to 2x --n because B's "
                         "per-output-position statistics need more images")
    ap.add_argument("--models", default=None,
                    help="comma-separated model ids. The first is the reference; any "
                         "others are compared at A-384 on identical inputs.")
    ap.add_argument("--list-models", action="store_true",
                    help="query the API for available models and exit")
    ap.add_argument("--validate-models", action="store_true",
                    help="make one real call per candidate model and exit. Listing is "
                         "not usability -- retired models still appear in models.list()")
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="0 disables thinking (default). Thinking tokens are billed as "
                         "output and count against max_output_tokens")
    ap.add_argument("--model", default=None, help="alias for a single --models entry")
    ap.add_argument("--in-price", type=float, default=None, help="USD / 1M input tokens")
    ap.add_argument("--out-price", type=float, default=None, help="USD / 1M output tokens")
    ap.add_argument("--throughput", type=int, default=0,
                    help="if >0, fire this many concurrent calls to measure achieved rate")
    ap.add_argument("--arms", default="A-text,A-224,A-384,A-512,B-384-schema,B-384-noschema",
                    help="comma-separated arms. Each runs at the reference model unless "
                         "it carries its own '@model' suffix, e.g. "
                         "'B-384-schema@gemini-3.1-flash-lite'.")
    ap.add_argument("--out-tag", default=None,
                    help="write to results/probe/<tag>/ instead of results/probe/. "
                         "Required to avoid clobbering an earlier run: probe_raw.jsonl "
                         "is opened in write mode.")
    args = ap.parse_args()

    cfg = load_config("data")
    if args.list_models or args.validate_models:
        _list_models(validate=args.validate_models)
        return
    models = [m.strip() for m in (args.models or args.model
                                  or cfg["generation"]["model"]).split(",") if m.strip()]
    model = models[0]
    n_batch = args.n_batch if args.n_batch is not None else args.n * 2
    images_dir = ROOT / cfg["paths"]["images_dir"]
    lo = cfg["data"]["target_caption_words"]["min"]
    hi = cfg["data"]["target_caption_words"]["max"]

    if not images_dir.exists():
        sys.exit(f"images not found at {images_dir} -- unzip Flickr8k_Dataset.zip first")

    rows = drop_malformed(read_captions(ROOT / "data/flickr8k/Flickr8k.token.txt"))
    train = set(read_image_list(ROOT / "data/flickr8k/Flickr_8k.trainImages.txt"))
    by_img: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.image_id in train and (images_dir / r.image_id).exists():
            by_img[r.image_id].append(r)
    # Probe train images only: never spend val/test on calibration.
    eligible = [i for i in sorted(by_img) if len(by_img[i]) == 5]
    images = eligible[: args.n]
    # B's decisive statistic is completion *by output position*, with one observation
    # per image per position -- so it needs more images than the A arms.
    images_b = eligible[: n_batch]
    if not images:
        sys.exit("no probe images found")

    full_calls_A = len(rows)                      # one call per (image, caption)
    full_calls_B = len({r.image_id for r in rows})  # one call per image

    # Each run gets its own directory. probe_raw.jsonl is opened in write mode, so a
    # second run into the same path would silently destroy the first run's results.
    OUT_DIR = PROBE_ROOT / args.out_tag if args.out_tag else PROBE_ROOT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(run_id="stage02-probe", stage="02_probe",
                        config={"n": args.n, "n_batch": n_batch, "models": models,
                                "arms": args.arms})
    manifest.save(OUT_DIR)

    key = load_key()
    # Build a client for every model any arm names, not just those in --models.
    arm_models = {a.split("@")[1] for a in args.arms.split(",") if "@" in a}
    llms = {
        m: gemini_llm(key, model=m, max_output_tokens=4096,
                      thinking_budget=args.thinking_budget)
        for m in set(models) | arm_models
    }
    # Config arms at the reference model; each additional model compared at A-384.
    # An arm may pin its own model with '@'; otherwise it runs at the reference.
    arms = [
        a.strip() if "@" in a else f"{a.strip()}@{model}"
        for a in args.arms.split(",") if a.strip()
    ]
    # Additional --models are compared at A-384 unless already named explicitly.
    for m in models[1:]:
        if f"A-384@{m}" not in arms:
            arms.append(f"A-384@{m}")

    print(f"models      : {', '.join(models)}  (reference: {model})")
    print(f"A arms      : {len(images)} images -> {len(images)*5} calls, "
          f"{len(images)*25} captions each")
    print(f"B arms      : {min(n_batch, len(images_b))} images -> "
          f"{min(n_batch, len(images_b))} calls, {min(n_batch, len(images_b))*25} captions each")
    print(f"full run    : A={full_calls_A:,} calls   B={full_calls_B:,} calls")
    print(f"arms        : {', '.join(arms)}")
    print()

    raw_path = OUT_DIR / "probe_raw.jsonl"
    raw_f = raw_path.open("w", encoding="utf-8")
    per_arm: dict[str, list[dict]] = {}
    # (arm, image, idx, emotion) -> caption, for paired comparisons
    flat: dict[str, dict[tuple, str]] = defaultdict(dict)

    for arm in arms:
        kind, res, schema, arm_model = _parse_arm(arm)
        llm = llms[arm_model]
        pool = images if kind == "A" else images_b
        print(f"── {arm} ({len(pool)} images) ──")
        records: list[dict] = []
        usage_before = len(llm.usage)  # type: ignore[attr-defined]

        for img_id in pool:
            caps_rows = sorted(by_img[img_id], key=lambda r: r.caption_idx)
            img_bytes = None
            if res is not None:
                img_bytes = load_image_bytes(images_dir / img_id, max_dim=res)

            if kind == "A":
                for r in caps_rows:
                    rec = _call_A(llm, r, img_bytes, lo, hi, arm)
                    records.append(rec)
                    for e, c in rec["captions"].items():
                        flat[arm][(img_id, r.caption_idx, e)] = c
            else:
                rec = _call_B(llm, img_id, caps_rows, img_bytes, lo, hi, arm, schema)
                records.append(rec)
                for idx, caps in rec["matrix"].items():
                    for e, c in caps.items():
                        flat[arm][(img_id, int(idx), e)] = c

        per_arm[arm] = records
        u = llm.usage[usage_before:]  # type: ignore[attr-defined]
        for rec, um in zip([r for r in records if not r.get("error")], u):
            rec.update({k: um.get(k) for k in
                        ("prompt_tokens", "output_tokens", "thinking_tokens",
                         "latency_s", "finish_reason")})
            rec["model"] = arm_model
        # Write only after usage is attached, so probe_raw.jsonl carries the token
        # counts. Writing inside the loop above left the raw file with no telemetry,
        # which meant cost could not be recomputed from it without re-calling.
        for rec in records:
            raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        raw_f.flush()
        done = sum(len([c for c in r.get("captions", {}).values() if c]) for r in records)
        print(f"   {len(records)} calls, {done}/{len(pool)*25} captions\n")

    raw_f.close()

    full_calls = {"A": full_calls_A, "B": full_calls_B}
    summaries = [summarise(per_arm[a], a, full_calls[_parse_arm(a)[0]]) for a in arms]

    report = _render(summaries, per_arm, flat, arms, images, images_b, args, models,
                     full_calls_A, full_calls_B, lo, hi)
    (OUT_DIR / "probe_report.md").write_text(report, encoding="utf-8")
    json.dump({"summaries": summaries}, (OUT_DIR / "probe_summary.json").open("w"), indent=2)
    manifest.finalise(OUT_DIR)

    print(report)
    print(f"\nraw calls  -> {raw_path}")
    print(f"report     -> {OUT_DIR / 'probe_report.md'}")

    if args.throughput:
        _throughput(llm, images, by_img, images_dir, args.throughput, lo, hi)


def _parse_arm(arm: str) -> tuple[str, int | None, bool, str]:
    """``A-384@gemini-2.0-flash`` -> ("A", 384, True, "gemini-2.0-flash")."""
    spec, _, model = arm.partition("@")
    kind = spec.split("-")[0]
    res = None if "text" in spec else int(spec.split("-")[1])
    schema = "noschema" not in spec
    return kind, res, schema, model


def _list_models(validate: bool = False) -> None:
    """Ask the API what exists, rather than trusting a hardcoded list.

    Model availability and naming change; picking from a stale recollection is how
    a run fails 30,000 calls in.

    **Listing is not usability.** `models.list()` returns models this key cannot
    call: `gemini-2.5-flash` was listed and then returned
    404 "no longer available to new users" on every request. With ``validate``, each
    candidate gets one real 8-token call, which is the only trustworthy check.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=load_key())
    rows = []
    for m in client.models.list():
        methods = getattr(m, "supported_actions", None) or []
        if methods and "generateContent" not in methods:
            continue
        rows.append((m.name.replace("models/", ""),
                     getattr(m, "input_token_limit", "?"),
                     getattr(m, "output_token_limit", "?")))

    if not validate:
        print(f"{'model':<44}{'in limit':>12}{'out limit':>12}")
        for name, i, o in sorted(rows):
            print(f"{name:<44}{str(i):>12}{str(o):>12}")
        print(f"\n{len(rows)} models are listed for generateContent -- but listing is "
              f"not usability.\nRe-run with --validate-models to make one real call "
              f"per candidate and see which actually work.")
        return

    # Text-generation candidates only: skip tts / image / robotics / embedding heads.
    skip = ("tts", "image", "robotics", "embedding", "lyria", "banana", "computer-use")
    cands = [n for n, _, _ in sorted(rows)
             if n.startswith("gemini") and not any(s in n for s in skip)]
    cfg = types.GenerateContentConfig(max_output_tokens=256)
    print(f"validating {len(cands)} candidates with one real call each\n")
    print(f"{'model':<40}{'status':<8}{'s':>7}{'think':>7}  note")
    ok_models = []
    for m in cands:
        t0 = time.time()
        try:
            r = client.models.generate_content(
                model=m, contents="Reply with the word ok.", config=cfg)
            dt = time.time() - t0
            um = r.usage_metadata
            think = getattr(um, "thoughts_token_count", None) or 0
            print(f"{m:<40}{'OK':<8}{dt:7.1f}{think:7}  out={um.candidates_token_count}")
            ok_models.append((m, dt, think))
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            note = ("retired for this key" if "no longer available" in msg
                    else msg.split("{")[0][:52].strip())
            print(f"{m:<40}{'FAIL':<8}{time.time()-t0:7.1f}{'-':>7}  {note}")

    print(f"\n{len(ok_models)} usable. Fastest first:")
    for m, dt, think in sorted(ok_models, key=lambda x: x[1])[:8]:
        est = dt * 40455 / 8 / 3600
        print(f"  {m:<40}{dt:6.1f}s/call  thinking={think:<4} "
              f"~{est:.1f}h for 40,455 calls at concurrency 8")


def _call_A(llm, row, img_bytes, lo, hi, arm) -> dict:
    try:
        caps, attempts, rejects = generate_one(
            llm, row.caption, image_bytes=img_bytes,
            min_words=lo, max_words=hi, max_attempts=1,
        )
        g = [grounding_report(row.caption, c) for c in caps.values() if c]
        return {
            "arm": arm, "kind": "A", "image_id": row.image_id,
            "caption_idx": row.caption_idx, "source": row.caption,
            "captions": caps, "rejects": rejects, "expected_captions": len(EMOTIONS),
            "grounding": {
                "content_recall": round(stats.mean(x["content_recall"] for x in g), 4) if g else 0.0,
                "novel_rate": round(stats.mean(x["novel_rate"] for x in g), 4) if g else 0.0,
            },
            "divergence": register_divergence(caps),
            "novel_words": sorted({w for x in g for w in x["novel_words"]}),
        }
    except Exception as exc:  # noqa: BLE001
        return {"arm": arm, "kind": "A", "image_id": row.image_id,
                "caption_idx": row.caption_idx, "error": f"{type(exc).__name__}: {exc}",
                "captions": {}, "rejects": {}, "expected_captions": len(EMOTIONS)}


def _call_B(llm, img_id, caps_rows, img_bytes, lo, hi, arm, schema) -> dict:
    sources = [r.caption for r in caps_rows]
    try:
        matrix, attempts, rejects = generate_image_batch(
            llm, sources, image_bytes=img_bytes,
            min_words=lo, max_words=hi, max_attempts=1, use_schema=schema,
        )
        gs, divs = [], []
        for i, caps in matrix.items():
            for c in caps.values():
                if c:
                    gs.append(grounding_report(sources[i], c))
            divs.append(register_divergence(caps))
        per_pos = {
            str(i): {
                "returned": len(matrix.get(i, {})),
                "rejected": len(rejects.get(i, {})),
            }
            for i in range(len(sources))
        }
        return {
            "arm": arm, "kind": "B", "image_id": img_id, "sources": sources,
            "matrix": {str(k): v for k, v in matrix.items()},
            "captions": {f"{i}:{e}": c for i, caps in matrix.items() for e, c in caps.items()},
            "rejects": {f"{i}:{e}": w for i, r in rejects.items() for e, w in r.items()},
            "expected_captions": len(sources) * len(EMOTIONS),
            "by_position": per_pos,
            "grounding": {
                "content_recall": round(stats.mean(x["content_recall"] for x in gs), 4) if gs else 0.0,
                "novel_rate": round(stats.mean(x["novel_rate"] for x in gs), 4) if gs else 0.0,
            },
            "divergence": {
                "mean_similarity": round(stats.mean(
                    d["mean_similarity"] for d in divs if d.get("mean_similarity") is not None
                ), 4) if any(d.get("mean_similarity") is not None for d in divs) else None
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"arm": arm, "kind": "B", "image_id": img_id,
                "error": f"{type(exc).__name__}: {exc}", "captions": {}, "rejects": {},
                "matrix": {}, "expected_captions": len(sources) * len(EMOTIONS)}


def _throughput(llm, images, by_img, images_dir, n, lo, hi) -> None:
    """Fire n calls concurrently to see what rate the account actually sustains."""
    from concurrent.futures import ThreadPoolExecutor

    print(f"\n── throughput test: {n} concurrent calls ──")
    rows = [r for i in images for r in by_img[i]][:n]
    img = load_image_bytes(images_dir / rows[0].image_id, max_dim=384)

    def one(r):
        t0 = time.time()
        try:
            generate_one(llm, r.caption, image_bytes=img, min_words=lo, max_words=hi,
                         max_attempts=1)
            return time.time() - t0, None
        except Exception as exc:  # noqa: BLE001
            return time.time() - t0, f"{type(exc).__name__}"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(one, rows))
    wall = time.time() - t0
    errs = Counter(e for _, e in res if e)
    print(f"   wall {wall:.1f}s for {len(rows)} calls -> {len(rows)/wall:.2f} calls/s")
    print(f"   per-call latency mean {stats.mean(l for l, _ in res):.2f}s")
    print(f"   errors: {dict(errs) if errs else 'none'}")
    if errs:
        print("   ^ rate limits reached; lower concurrency for the full run")


def _resolve(spec: str, arms: list[str]) -> str | None:
    """Match a bare arm spec (``A-384``) to the full arm name including its model."""
    if spec in arms:
        return spec
    for a in arms:
        if a.split("@")[0] == spec:
            return a
    return None


def _render(summaries, per_arm, flat, arms, images, images_b, args, models,
            fa, fb, lo, hi) -> str:
    L: list[str] = []
    L.append("# Stage-02 Configuration Probe\n")
    L.append(f"- models: {', '.join(f'`{m}`' for m in models)} (reference `{models[0]}`)")
    L.append(f"- A arms: {len(images)} train images -> {len(images)*5} calls, "
             f"{len(images)*25} captions")
    L.append(f"- B arms: {len(images_b)} train images -> {len(images_b)} calls, "
             f"{len(images_b)*25} captions "
             f"(more images because per-position stats need them)")
    L.append(f"- full run: **A = {fa:,} calls**, **B = {fb:,} calls** (same {fa*5:,} target captions)")
    L.append(f"- word target: {lo}-{hi}; `max_attempts=1` so cost shown is base cost\n")

    L.append("## Per-arm results\n")
    hdr = ("| arm | calls | complete (95% CI) | reject (95% CI) | trunc | in tok | "
           "out tok | lat s | words | in-target | recall | novel | reg-sim |")
    L.append(hdr)
    L.append("|" + "---|" * 13)

    def pct_ci(c: dict) -> str:
        if c.get("mean") is None:
            return "-"
        if c.get("lo") is None:
            return f"{c['mean']*100:.0f}%"
        return f"{c['mean']*100:.0f}% [{c['lo']*100:.0f}-{c['hi']*100:.0f}]"

    for s in summaries:
        ln = s["length"]
        L.append(
            f"| `{s['arm']}` | {s['calls']} | {pct_ci(s['completion_ci'])} | "
            f"{pct_ci(s['rejection_ci'])} | {s['truncated']} | "
            f"{s['input_tokens_mean'] or '-'} | {s['output_tokens_mean'] or '-'} | "
            f"{s['latency_mean_s'] or '-'} | {ln.get('mean','-')} | "
            f"{ln.get('in_target',0)*100:.0f}% | {s['content_recall_mean']} | "
            f"{s['novel_rate_mean']} | {s['register_similarity_mean']} |"
        )
    L.append("")
    L.append("`recall` = fraction of source content words kept. `novel` = fraction of "
             "generated content words absent from the source. `reg-sim` = mean pairwise "
             "similarity of the five registers (lower is more distinguishable).\n")

    L.append("## Rejections by rule\n")
    for s in summaries:
        if s["rejections_by_rule"]:
            L.append(f"- `{s['arm']}`: " + ", ".join(
                f"{k} ({v})" for k, v in s["rejections_by_rule"].items()))
        else:
            L.append(f"- `{s['arm']}`: none")
    L.append("")

    # Option B: the measurement that decides batching
    L.append("## Option B: quality by output position\n")
    L.append("The specific failure mode for a 25-field response is decay toward the end.\n")
    any_b = False
    for arm in arms:
        if not arm.startswith("B"):
            continue
        any_b = True
        pos: dict[str, list[int]] = defaultdict(list)
        rej: dict[str, list[int]] = defaultdict(list)
        for r in per_arm[arm]:
            for k, v in (r.get("by_position") or {}).items():
                pos[k].append(v["returned"])
                rej[k].append(v["rejected"])
        L.append(f"**`{arm}`** ({len(per_arm[arm])} images, so {len(per_arm[arm])} "
                 f"observations per position)\n")
        L.append("| caption index | returned /5 (95% CI) | rejected /5 |")
        L.append("|---|---|---|")
        for k in sorted(pos, key=int):
            ci = cluster_bootstrap_mean(pos[k], list(range(len(pos[k]))), n_resamples=4000)
            band = (f"{ci['mean']:.2f} [{ci['lo']:.2f}-{ci['hi']:.2f}]"
                    if ci.get("lo") is not None else f"{ci['mean']:.2f}")
            L.append(f"| {k} | {band} | {stats.mean(rej[k]):.2f} |")
        L.append("")
        first, last = sorted(pos, key=int)[0], sorted(pos, key=int)[-1]
        d = cluster_bootstrap_paired_diff(pos[first], pos[last],
                                          list(range(len(pos[first]))), n_resamples=4000)
        verdict = ("no detectable decay" if d.get("crosses_zero")
                   else f"DECAY: position {last} returns {d['diff']:.2f} fewer than {first}")
        L.append(f"- first-vs-last position difference: {d['diff']:.3f} "
                 f"[{d['lo']}, {d['hi']}] -> **{verdict}**\n")
    if not any_b:
        L.append("_no B arms run_\n")

    L.append("## Paired comparisons\n")
    L.append("Same images, same captions, same registers -- so these isolate one factor.\n")

    # SIMILARITY_SAME: above this, two configs are producing the same captions and
    # the cheaper one wins. Chosen as a judgement call, stated so it can be argued with.
    SAME = 0.85

    def cmp(a_spec: str, b_spec: str, question: str) -> None:
        a = _resolve(a_spec, arms)
        b = _resolve(b_spec, arms)
        if not a or not b:
            return
        keys = sorted(set(flat[a]) & set(flat[b]))
        if not keys:
            L.append(f"- **{a}** vs **{b}**: no overlapping captions\n")
            return
        left = [flat[a][k] for k in keys]
        right = [flat[b][k] for k in keys]
        d = config_divergence(left, right)
        sims = [pairwise_similarity(x, y) for x, y in zip(left, right)]
        clusters = [k[0] for k in keys]          # cluster by image
        ci = cluster_bootstrap_mean(sims, clusters, n_resamples=4000)

        L.append(f"**{a}** vs **{b}** — _{question}_\n")
        L.append(f"- paired captions: {d['n']} over {ci['n_clusters']} images")
        L.append(f"- identical strings: {d['identical_rate']*100:.1f}%")
        band = (f"{ci['mean']} [{ci['lo']}, {ci['hi']}]"
                if ci.get("lo") is not None else str(ci["mean"]))
        L.append(f"- mean content similarity: {band}")
        if ci.get("hi") is not None and ci["lo"] > SAME:
            L.append(f"- **verdict: equivalent** (CI entirely above {SAME}) -> "
                     f"prefer the cheaper option")
        elif ci.get("hi") is not None and ci["hi"] < SAME:
            L.append(f"- **verdict: materially different** (CI entirely below {SAME}) -> "
                     f"choose on quality, not cost")
        else:
            L.append(f"- **verdict: inconclusive** (CI spans {SAME}) -> "
                     f"raise --n before deciding")
        L.append("")

    cmp("A-text", "A-384", "does the image change the output at all? equivalence here "
        "means multimodal is not earning its 5x input cost")
    cmp("A-224", "A-384", "does resolution matter?")
    cmp("A-384", "A-512", "does more resolution matter?")
    cmp("A-384", "B-384-schema", "does batching change caption content?")
    cmp("B-384-schema", "B-384-noschema", "does the schema change content, or only reliability?")
    for m in models[1:]:
        cmp(f"A-384@{models[0]}", f"A-384@{m}", f"does {m} produce different captions "
            f"than {models[0]}?")

    L.append("## Cost extrapolation\n")
    L.append("| arm | full-run calls | input tok | output tok | cost |")
    L.append("|---|---|---|---|---|")
    for s in summaries:
        if not s.get("full_run_input_tokens"):
            continue
        i, o = s["full_run_input_tokens"], s["full_run_output_tokens"]
        if args.in_price is not None and args.out_price is not None:
            c = f"${i/1e6*args.in_price + o/1e6*args.out_price:,.2f}"
        else:
            c = "_no prices given_"
        L.append(f"| `{s['arm']}` | {s['full_run_calls']:,} | {i/1e6:,.1f}M | {o/1e6:,.1f}M | {c} |")
    if args.in_price is None:
        L.append("\nRerun with `--in-price` / `--out-price` (USD per 1M tokens) for dollar "
                 "figures. Rates are deliberately not hardcoded.\n")

    L.append("\n## Sample output\n")
    for arm in arms[:2]:
        recs = [r for r in per_arm[arm] if not r.get("error") and r.get("captions")]
        if not recs:
            continue
        r = recs[0]
        L.append(f"**`{arm}`** — source: _{r.get('source') or r.get('sources', [''])[0]}_\n")
        for k, v in list(r["captions"].items())[:5]:
            L.append(f"- `{k}`: {v}")
        if r.get("novel_words"):
            L.append(f"\n  novel words vs source: {', '.join(r['novel_words'][:18])}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
