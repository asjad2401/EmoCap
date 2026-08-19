#!/usr/bin/env python
"""Stage 02 -- generate the emotional caption corpus via the Gemini Batch API.

Config comes from `configs/data.yaml`; nothing important is a CLI default. Every run
writes a manifest, and the caption store is append-only so any part can be stopped
and resumed.

    uv run python scripts/run_stage02.py --part audit          # 50 train images
    uv run python scripts/run_stage02.py --part test
    uv run python scripts/run_stage02.py --part val
    uv run python scripts/run_stage02.py --part train --limit 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data import (  # noqa: E402
    assign_official_splits,
    drop_malformed,
    read_captions,
)
from emocap.data.batch import run_batch_generation  # noqa: E402
from emocap.data.exclusions import load_exclusions, record_exclusion  # noqa: E402
from emocap.data.generate import completed_keys, read_records  # noqa: E402
from emocap.runtime import Manifest, assert_matches_lock, load_config  # noqa: E402


def load_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                key = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not key:
        sys.exit("No API key. Put GEMINI_API_KEY=... in .env (gitignored).")
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["audit", "train", "val", "test"],
                    help="'audit' is 50 train images for the pre-registered hand audit")
    ap.add_argument("--limit", type=int, default=None, help="cap images this invocation")
    ap.add_argument("--audit-offset", type=int, default=0,
                    help="skip this many train images before taking the audit slice, so a "
                         "second audit lands on images the first never saw")
    ap.add_argument("--out", default=None,
                    help="override the caption store path; use a separate file when the run "
                         "is not meant to be pooled with the main corpus (e.g. a prompt A/B)")
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be generated and stop")
    args = ap.parse_args()

    # The prompt has been through five measured versions; a manifest that records the
    # model but not the prompt cannot answer "which prompt produced this caption". Hash
    # the rendered prompt so every record's provenance is exact and comparable.
    import hashlib

    from emocap.data.prompt import build_batch_prompt

    prompt_sha = hashlib.sha256(
        build_batch_prompt(["provenance probe"] * 5, emphasise_distinctness=True).encode()
    ).hexdigest()

    cfg = load_config("data")
    # The generation settings are pre-registered; drift must be deliberate.
    assert_matches_lock(cfg, require_sections=("data",))
    gen = cfg["generation"]

    captions = ROOT / "data/flickr8k/Flickr8k.token.txt"
    images_dir = ROOT / cfg["paths"]["images_dir"]
    out_path = Path(args.out) if args.out else ROOT / cfg["paths"]["raw_generations"]
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = drop_malformed(read_captions(captions))
    sources: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda x: (x.image_id, x.caption_idx)):
        sources.setdefault(r.image_id, []).append(r.caption)
    sources = {k: v for k, v in sources.items() if len(v) == 5}

    d = ROOT / "data/flickr8k"
    splits = assign_official_splits(
        sources.keys(),
        train_list=d / "Flickr_8k.trainImages.txt",
        val_list=d / "Flickr_8k.devImages.txt",
        test_list=d / "Flickr_8k.testImages.txt",
        seed=cfg["data"]["split_seed"],
    )

    if args.part == "audit":
        n = gen["audit_sample"] // 4
        train_ids = [i for i in sorted(sources) if splits[i] == "train"]
        ids = train_ids[args.audit_offset: args.audit_offset + n]
    else:
        ids = [i for i in sorted(sources) if splits[i] == args.part]

    done = completed_keys(out_path)
    # Images the generator refuses are recorded, not retried. Without this an image the
    # model will never process is pending forever, and preflighting on pending[0] aborts
    # every subsequent run before it submits anything.
    excluded = load_exclusions()
    pending = [i for i in ids
               if i not in excluded and any((i, k) not in done for k in range(5))]

    print(f"part          : {args.part}"
          + (f"  offset {args.audit_offset}" if args.part == "audit" else ""))
    print(f"model         : {gen['model']}   plan {gen['plan']}   "
          f"schema={gen['use_response_schema']}   thinking={gen['thinking_budget']}")
    print(f"images in part: {len(ids):,}")
    n_excluded = sum(1 for i in ids if i in excluded)
    print(f"already done  : {len(ids) - len(pending) - n_excluded:,}")
    if n_excluded:
        print(f"excluded      : {n_excluded:,}  (generator refusals; "
              f"see data/generated/excluded_images.json)")
    print(f"to generate   : {len(pending):,} images -> {len(pending)*25:,} captions")
    print(f"chunk size    : {args.chunk_size}  ({-(-len(pending)//args.chunk_size)} jobs)")
    print(f"store         : {out_path}")
    print(f"prompt sha256 : {prompt_sha[:16]}  (v{cfg['generation'].get('prompt_version','?')})")

    P = cfg["pricing"]["models"][gen["model"]]
    # Per-image tokens MEASURED on the v5 audit run (49 images: 219,120 in / 49,957 out).
    # The earlier 2152/672 constants predate both the prompt rewrite and the `strain`
    # field, and understated the bill by ~40%.
    est = (len(pending) * 4472 / 1e6 * P["input"]
           + len(pending) * 1019 / 1e6 * P["output"]) * (1 - cfg["pricing"]["batch_api_discount"])
    print(f"est. cost     : ${est:.2f} (batch)")

    if args.dry_run or not pending:
        print("\n(dry run)" if args.dry_run else "\nnothing to do")
        return

    from google import genai

    client = genai.Client(api_key=load_key())

    # Preflight: one real realtime call with the exact config, before committing a
    # whole batch to it. A dry run caught the config still naming gemini-3.6-flash,
    # which 400s on thinking_budget=0 -- every request in the job would have failed.
    # Preflight: one real realtime call with the exact config, before committing a
    # whole batch to it. A dry run once caught the config still naming gemini-3.6-flash,
    # which 400s on thinking_budget=0 -- every request in the job would have failed.
    #
    # It walks the pending list rather than testing only the first image: a refusal (zero
    # output tokens) is a property of that image, not of the config, so it is registered
    # and the next candidate tried. Testing only pending[0] meant one refused image needed
    # one whole invocation to record, and 12 of them needed 12.
    print("\npreflight: live call(s) with this exact config...")
    from emocap.data.generate import gemini_llm, generate_image_batch, load_image_bytes
    _llm = gemini_llm(load_key(), model=gen["model"], temperature=gen["temperature"],
                      max_output_tokens=gen["max_output_tokens"],
                      thinking_budget=gen["thinking_budget"], max_retries=2)
    _refused: list[str] = []
    _ok = False
    for _probe_id in list(pending)[:25]:
        _m, _, _rej = generate_image_batch(
            _llm, sources[_probe_id],
            image_bytes=load_image_bytes(images_dir / _probe_id,
                                         max_dim=gen["image_max_dim"]),
            min_words=cfg["data"]["target_caption_words"]["min"],
            max_words=cfg["data"]["target_caption_words"]["max"],
            max_attempts=1, use_schema=gen["use_response_schema"],
            emphasise_distinctness=True,
        )
        _got = sum(len(v) for v in _m.values())
        _u = _llm.usage[-1]
        print(f"  {_probe_id}: {_got}/25 captions  in={_u['prompt_tokens']} "
              f"out={_u['output_tokens']} think={_u['thinking_tokens']}", flush=True)
        if _got == 0 and _u["output_tokens"] == 0:
            # Zero output tokens is a refusal, not a misconfiguration: the model returned
            # nothing at all. Register it and try the next image.
            record_exclusion(_probe_id, "generator_safety_refusal",
                             "zero output tokens at preflight")
            _refused.append(_probe_id)
            continue
        if _got >= 20:
            _ok = True
            break
        sys.exit(f"preflight returned only {_got}/25 captions -- fix the config before "
                 f"spending on a batch")

    if _refused:
        print(f"  registered {len(_refused)} refusal(s) as excluded")
        pending = [i for i in pending if i not in set(_refused)]
    if not _ok:
        if not pending:
            print("\nevery pending image was refused; nothing left to generate")
            return
        sys.exit("preflight exhausted 25 candidates without a success -- check the config")
    print("  preflight OK\n")

    run_dir = ROOT / "runs" / f"stage02-{args.part}-{time.strftime('%Y%m%d-%H%M%S')}"
    man = Manifest(run_id=run_dir.name, stage="02_captions_generate",
                   config={**gen, "part": args.part, "chunk_size": args.chunk_size,
                           "prompt_sha256": prompt_sha,
                           "n_images": len(pending), "audit_offset": args.audit_offset,
                           "out_path": str(out_path.relative_to(ROOT)),
                           "image_ids": pending})
    man.save(run_dir)

    def progress(ev: dict) -> None:
        if ev.get("event") == "poll":
            print(f"    {ev['job']}  {ev['state']}  {ev['elapsed']:.0f}s", flush=True)
        elif ev.get("event") == "chunk_done":
            print(f"  {ev['job']} done in {ev['wall']:.0f}s  "
                  f"images {ev['images_written']}  captions {ev['captions_written']}  "
                  f"rejections {ev['rejections']}", flush=True)

    t0 = time.time()
    try:
        stats = run_batch_generation(
            client, gen["model"], pending, sources, out_path,
            images_dir=images_dir, chunk_size=args.chunk_size,
            image_max_dim=gen["image_max_dim"],
            min_words=cfg["data"]["target_caption_words"]["min"],
            max_words=cfg["data"]["target_caption_words"]["max"],
            emphasise_distinctness=True,
            temperature=gen["temperature"],
            max_output_tokens=gen["max_output_tokens"],
            thinking_budget=gen["thinking_budget"],
            limit_images=args.limit,
            progress=progress,
        )
    except Exception as exc:
        man.record_check("batch_generation", False, str(exc)[:400])
        man.finalise(run_dir, status="failed")
        raise

    u = stats["usage"]
    cost = (u["prompt"] / 1e6 * P["input"]
            + (u["output"] + u["thinking"]) / 1e6 * P["output"]) * 0.5
    stats["measured_cost_usd"] = round(cost, 4)
    man.record_check("all_captions_returned",
                     stats["captions_written"] == stats["captions_expected"],
                     f"{stats['captions_written']}/{stats['captions_expected']}")
    # Statistics go in their own file. `checks` is for pass/fail gates only.
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    man.finalise(run_dir)

    print(f"\n{'='*60}")
    print(f"jobs            {stats['jobs']}")
    print(f"images written  {stats['images_written']:,}")
    print(f"captions        {stats['captions_written']:,} / {stats['captions_expected']:,}"
          f"  ({stats['captions_written']/max(1,stats['captions_expected'])*100:.1f}%)")
    print(f"rejections      {stats['rejections']:,}  {stats['rejection_reasons']}")
    print(f"skipped (no img){stats['images_skipped_no_file']}")
    print(f"tokens          in {u['prompt']:,}  out {u['output']:,}  think {u['thinking']:,}")
    print(f"measured cost   ${cost:.4f} (batch rate)")
    print(f"wall            {time.time()-t0:.0f}s")
    print(f"manifest        {run_dir}/manifest.json")
    print(f"total in store  {len(read_records(out_path)):,} records")


if __name__ == "__main__":
    main()
