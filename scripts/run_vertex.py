#!/usr/bin/env python
"""Generate captions through Vertex AI, for the generator-capability arm.

Two uses, and the control must pass before the arm is worth paying for:

    # control: our production model via Vertex, on images already generated via AI Studio
    uv run python scripts/run_vertex.py --model gemini-3.1-flash-lite --n 100 \
        --out data/generated/captions_vertex_control.jsonl

    # arm: a stronger generator, same prompt, same images
    uv run python scripts/run_vertex.py --model gemini-3.5-flash --n 1000 \
        --out data/generated/captions_vertex_35flash.jsonl

**The control exists to kill a confound.** Without it, any difference in the stronger-model
arm could be "Vertex differs from AI Studio" rather than "the model is better". Running our
own production model through Vertex on the same images, with the same prompt and config,
settles that for pennies.

**`location` must be `global`.** Gemini 3.x is not served from us-central1, us-east4,
us-west1 or europe-west4 -- all 404 there while appearing in `models.list()`. Listing is
not usability; every model here was validated by a real call first.

Realtime calls, not batch: Vertex batch needs GCS staging and a different request format,
and at pilot sizes realtime costs about twice batch on a bill of tens of dollars. Not worth
the engineering until the arm is justified.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data import drop_malformed, read_captions  # noqa: E402
from emocap.data.exclusions import load_exclusions  # noqa: E402
from emocap.data.generate import (  # noqa: E402
    GenerationRecord,
    append_record,
    completed_keys,
    gemini_llm,
    generate_image_batch,
    load_image_bytes,
    vertex_client,
)
from emocap.runtime import Manifest, load_config  # noqa: E402

PROJECT = "gen-lang-client-0673022159"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100, help="images to generate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--location", default="global",
                    help="MUST be 'global' for Gemini 3.x; regional endpoints 404")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="New GCP projects have low Vertex quotas. Concurrency 16 drew a "
                         "sustained 429 RESOURCE_EXHAUSTED on 24%% of calls even with 3 "
                         "retries, so the default is deliberately low. Raise it only after "
                         "confirming the project's quota.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config("data")
    gen = cfg["generation"]
    images_dir = ROOT / cfg["paths"]["images_dir"]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Same images the AI Studio corpus already covers, so the comparison is paired at the
    # image level even though temperature 0.9 makes individual captions differ.
    rows = drop_malformed(read_captions(ROOT / "data/flickr8k/Flickr8k.token.txt"))
    sources: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda x: (x.image_id, x.caption_idx)):
        sources.setdefault(r.image_id, []).append(r.caption)
    sources = {k: v for k, v in sources.items() if len(v) == 5}

    existing = {json.loads(l)["image_id"] for l in
                (ROOT / cfg["paths"]["raw_generations"]).read_text().splitlines() if l.strip()}
    excluded = load_exclusions()
    pool = sorted(i for i in sources if i in existing and i not in excluded)
    random.Random(args.seed).shuffle(pool)
    chosen = pool[: args.n]

    done = completed_keys(out_path)
    pending = [i for i in chosen if any((i, k) not in done for k in range(5))]

    prices = cfg["pricing"]["models"].get(args.model)
    print(f"model       : {args.model}   location {args.location}")
    print(f"images      : {len(chosen)}   pending {len(pending)}")
    print(f"store       : {out_path}")
    print(f"prompt      : v{gen.get('prompt_version', '?')}")
    if prices:
        # Realtime: no batch discount. Measured AI Studio rates per image.
        est = len(pending) * (4472 / 1e6 * prices["input"] + 1019 / 1e6 * prices["output"])
        print(f"est. cost   : ${est:.2f} (realtime, no batch discount)")
    if args.dry_run or not pending:
        print("\n(dry run)" if args.dry_run else "\nnothing to do")
        return

    key = next((ROOT / "secrets").glob("*.json"), None)
    if key is None:
        sys.exit("no service-account JSON in secrets/")
    client = vertex_client(PROJECT, args.location, service_account_file=key)
    llm = gemini_llm(client=client, model=args.model,
                     temperature=gen["temperature"],
                     max_output_tokens=gen["max_output_tokens"],
                     thinking_budget=gen["thinking_budget"], max_retries=3)

    run_dir = ROOT / "runs" / f"vertex-{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    man = Manifest(run_id=run_dir.name, stage="02b_vertex_generation",
                   config={**gen, "model": args.model, "platform": "vertex",
                           "location": args.location, "n_images": len(pending),
                           "out_path": str(out_path.relative_to(ROOT)),
                           "image_ids": pending})
    man.save(run_dir)

    stats = {"images": 0, "captions": 0, "expected": 0, "rejections": 0,
             "errors": 0, "refused": 0, "error_kinds": {}}
    t0 = time.time()

    def one(img_id: str) -> tuple[str, dict, dict, dict | None]:
        try:
            b = load_image_bytes(images_dir / img_id, max_dim=gen["image_max_dim"])
            m, _, rej = generate_image_batch(
                llm, sources[img_id], image_bytes=b,
                min_words=cfg["data"]["target_caption_words"]["min"],
                max_words=cfg["data"]["target_caption_words"]["max"],
                max_attempts=1, use_schema=gen["use_response_schema"],
                emphasise_distinctness=True,
            )
            return img_id, m, rej, None
        except Exception as exc:  # noqa: BLE001
            return img_id, {}, {}, {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool_ex:
        futs = {pool_ex.submit(one, i): i for i in pending}
        for n, fut in enumerate(as_completed(futs), 1):
            img_id, matrix, rejects, err = fut.result()
            stats["expected"] += len(sources[img_id]) * 5
            if err:
                stats["errors"] += 1
                # Record WHAT failed, not just how many. The first control run reported
                # "24 errors" with no detail, and the cause (429 quota, not a model or
                # code fault) had to be rediscovered by hand.
                kind = err["error"].split("{")[0].strip()[:80]
                stats["error_kinds"][kind] = stats["error_kinds"].get(kind, 0) + 1
            got = 0
            for idx, src in enumerate(sources[img_id]):
                caps = matrix.get(idx, {})
                if not caps:
                    continue
                append_record(out_path, GenerationRecord(
                    image_id=img_id, caption_idx=idx, source_caption=src,
                    captions=caps, model=args.model, attempts=1,
                    rejected=rejects.get(idx, {}),
                ))
                got += len([c for c in caps.values() if c])
                stats["rejections"] += len(rejects.get(idx, {}))
            stats["captions"] += got
            if got:
                stats["images"] += 1
            elif not err:
                stats["refused"] += 1
            if n % 25 == 0 or n == len(pending):
                print(f"  {n}/{len(pending)}  captions {stats['captions']:,}  "
                      f"errors {stats['errors']}  refused {stats['refused']}", flush=True)

    u = {"prompt": sum(x["prompt_tokens"] for x in llm.usage),
         "output": sum(x["output_tokens"] for x in llm.usage),
         "thinking": sum(x.get("thinking_tokens", 0) for x in llm.usage)}
    stats["usage"] = u
    if prices:
        stats["measured_cost_usd"] = round(
            u["prompt"] / 1e6 * prices["input"]
            + (u["output"] + u["thinking"]) / 1e6 * prices["output"], 4)
    stats["wall_seconds"] = round(time.time() - t0, 1)
    man.record_check("all_captions_returned",
                     stats["captions"] == stats["expected"],
                     f"{stats['captions']}/{stats['expected']}")
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    man.finalise(run_dir)

    print(f"\n{'=' * 58}")
    print(f"images      {stats['images']}   refused {stats['refused']}   "
          f"errors {stats['errors']}")
    if stats["error_kinds"]:
        for k, v in sorted(stats["error_kinds"].items(), key=lambda kv: -kv[1]):
            print(f"  x{v}  {k}")
    print(f"captions    {stats['captions']:,} / {stats['expected']:,}")
    print(f"rejections  {stats['rejections']:,}")
    print(f"tokens      in {u['prompt']:,}  out {u['output']:,}  think {u['thinking']:,}")
    if "measured_cost_usd" in stats:
        print(f"cost        ${stats['measured_cost_usd']:.4f} (realtime)")
    print(f"wall        {stats['wall_seconds']:.0f}s")
    print(f"manifest    {run_dir}")


if __name__ == "__main__":
    main()
