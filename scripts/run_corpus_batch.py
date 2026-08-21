#!/usr/bin/env python
"""Generate the corpus through Vertex batch, in resumable chunks.

    uv run python scripts/run_corpus_batch.py --n 8076 --chunk 100

Chunked because a batch job is all-or-nothing: one bad job loses its chunk, not the run.
100 rows is the measured sweet spot -- 7.3 MB, ~220 s, and *faster* per image than 25 rows
because the queueing overhead is fixed.

Resume is a set difference over the append-only store, so re-running skips finished images.
Rows that fail inside a successful job are collected into `failed_images` and retried on the
next pass -- Vertex reports `JOB_STATE_SUCCEEDED` for jobs in which every row failed, so the
job state is never trusted on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from google.cloud import storage  # noqa: E402
from google.oauth2 import service_account  # noqa: E402

from emocap.data import drop_malformed, read_captions  # noqa: E402
from emocap.data.exclusions import load_exclusions  # noqa: E402
from emocap.data.generate import completed_keys, vertex_client  # noqa: E402
from emocap.data.vertex_batch import run_vertex_batch  # noqa: E402
from emocap.runtime import Manifest, load_config  # noqa: E402

PROJECT = "gen-lang-client-0673022159"
BUCKET = f"emocap-batch-{PROJECT}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8076, help="images to cover")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--out", default="data/generated/captions_corpus.jsonl")
    ap.add_argument("--cap-carry", default="runs/cap_carry_corpus.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config("data")
    gen = cfg["generation"]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = drop_malformed(read_captions(ROOT / "data/flickr8k/Flickr8k.token.txt"))
    sources: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda x: (x.image_id, x.caption_idx)):
        sources.setdefault(r.image_id, []).append(r.caption)
    sources = {k: v for k, v in sources.items() if len(v) == 5}
    excluded = load_exclusions()
    pool = sorted(i for i in sources if i not in excluded)[: args.n]

    done = completed_keys(out)
    pending = [i for i in pool if any((i, k) not in done for k in range(5))]
    chunks = [pending[i:i + args.chunk] for i in range(0, len(pending), args.chunk)]

    print(f"model     {gen['model']}   prompt v{gen['prompt_version']}")
    print(f"images    {len(pool):,} in scope, {len(pending):,} pending")
    print(f"chunks    {len(chunks)} x {args.chunk}")
    print(f"store     {out}")
    if args.dry_run or not pending:
        print("\n(dry run)" if args.dry_run else "\nnothing to do")
        return

    kf = next((ROOT / "secrets").glob("*.json"))
    creds = service_account.Credentials.from_service_account_file(
        str(kf), scopes=["https://www.googleapis.com/auth/cloud-platform"])
    gcs = storage.Client(project=PROJECT, credentials=creds)
    cli = vertex_client(PROJECT, "global", service_account_file=kf)

    carry = ROOT / args.cap_carry
    ban = json.loads(carry.read_text()) if carry.exists() else {}

    run_dir = ROOT / "runs" / f"corpus-batch-{time.strftime('%Y%m%d-%H%M%S')}"
    man = Manifest(run_id=run_dir.name, stage="02c_corpus_batch",
                   config={**gen, "chunk": args.chunk, "n_images": len(pending),
                           "out_path": str(out.relative_to(ROOT)), "bucket": BUCKET})
    man.save(run_dir)

    tot = {"images": 0, "captions": 0, "row_errors": 0, "failed": []}
    t0 = time.time()
    for n, imgs in enumerate(chunks, 1):
        st = run_vertex_batch(
            client=cli, client_storage=gcs, model=gen["model"], image_ids=imgs,
            sources=sources, images_dir=ROOT / cfg["paths"]["images_dir"], out_path=out,
            bucket=BUCKET, prefix=f"{run_dir.name}/c{n:03d}",
            image_max_dim=gen["image_max_dim"], temperature=gen["temperature"],
            max_output_tokens=gen["max_output_tokens"],
            thinking_budget=gen["thinking_budget"],
            min_words=cfg["data"]["target_caption_words"]["min"],
            max_words=cfg["data"]["target_caption_words"]["max"],
            max_sentences=1, banned_by_register=ban or None)
        tot["images"] += st["images"]
        tot["captions"] += st["captions"]
        tot["row_errors"] += st["row_errors"]
        tot["failed"] += st["failed_images"]
        print(f"  chunk {n}/{len(chunks)}  images {st['images']:3d}  "
              f"captions {st['captions']:4d}  row_errors {st['row_errors']:3d}  "
              f"[{(time.time() - t0) / 60:.0f} min]", flush=True)

    tot["wall_minutes"] = round((time.time() - t0) / 60, 1)
    (run_dir / "stats.json").write_text(json.dumps(tot, indent=2))
    man.finalise(run_dir)
    print(f"\nimages {tot['images']:,}  captions {tot['captions']:,}  "
          f"row_errors {tot['row_errors']}  wall {tot['wall_minutes']} min")
    if tot["failed"]:
        print(f"{len(tot['failed'])} images failed inside successful jobs — "
              f"re-run this command to retry them.")


if __name__ == "__main__":
    main()
