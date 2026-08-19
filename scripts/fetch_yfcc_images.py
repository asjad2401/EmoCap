#!/usr/bin/env python
"""Fetch the YFCC images referenced by the Personality-Captions subset we use.

Personality-Captions ships `image_hash` values only. The images live in the
`multimedia-commons` bucket, an AWS Registry of Open Data public dataset, readable
without credentials. See docs/data-attribution.md for licence and citation; the short
version is that every image is Creative Commons licensed with a per-image variant, so
this project computes CLIP features and **never redistributes an image**.

    uv run python scripts/fetch_yfcc_images.py --design grouped
    uv run python scripts/fetch_yfcc_images.py --design grouped --dry-run

Only the images for traits mapped to our five registers are fetched -- ~20,515 of
191,858, so ~2.4 GB rather than 23 GB. Resumable: an existing non-empty file is skipped,
so re-running costs only the misses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PREFIX = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images"
#: Transport failures worth retrying. A 404 is not one: the image is simply gone, and
#: retrying it three times just triples the wait before recording the miss.
_RETRY_STATUS = (408, 429, 500, 502, 503, 504)


def url_for(image_hash: str) -> str:
    h = image_hash
    return f"{PREFIX}/{h[:3]}/{h[3:6]}/{h}.jpg"


def fetch_one(image_hash: str, dest_dir: Path, *, attempts: int = 3,
              timeout: int = 30) -> tuple[str, str, int]:
    """Return (image_hash, status, bytes). status is 'ok', 'skip', or a reason."""
    import urllib.error
    import urllib.request

    dest = dest_dir / f"{image_hash}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return image_hash, "skip", dest.stat().st_size

    delay = 2.0
    last = "unknown"
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url_for(image_hash),
                headers={"User-Agent": "EmoCap-research/1.0 (feature extraction only)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            if not body:
                last = "empty"
            else:
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(body)
                tmp.replace(dest)          # atomic, so an interrupt cannot leave a
                return image_hash, "ok", len(body)   # truncated file that "exists"
        except urllib.error.HTTPError as e:
            last = f"http{e.code}"
            if e.code not in _RETRY_STATUS:
                break
        except Exception as e:  # noqa: BLE001
            last = type(e).__name__
        if attempt < attempts:
            time.sleep(delay)
            delay *= 2
    return image_hash, last, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["strict", "grouped"], default="grouped")
    ap.add_argument("--pc-root", default="data/external/personality_captions")
    ap.add_argument("--out", default="data/external/yfcc_images")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cmp", ROOT / "scripts/compare_human_ceiling.py")
    cmp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp_mod)

    mapping = cmp_mod.STRICT_TRAITS if args.design == "strict" else cmp_mod.GROUPED_TRAITS
    pc_root = ROOT / args.pc_root if not Path(args.pc_root).is_absolute() else Path(args.pc_root)
    _, _, hashes = cmp_mod.load_personality_captions(pc_root, mapping)

    # One image may carry more than one caption; fetch each image once.
    unique = sorted(set(hashes))
    if args.limit:
        unique = unique[: args.limit]

    dest_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    dest_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in dest_dir.glob("*.jpg") if p.stat().st_size > 0}
    todo = [h for h in unique if h not in have]

    print(f"design      : {args.design}")
    print(f"images      : {len(unique):,} referenced")
    print(f"already have: {len(unique) - len(todo):,}")
    print(f"to fetch    : {len(todo):,}  (~{len(todo) * 117 / 1024:.1f} MB at 117 KB mean)")
    print(f"dest        : {dest_dir}")
    print(f"workers     : {args.workers}")
    if args.dry_run or not todo:
        print("\n(dry run)" if args.dry_run else "\nnothing to fetch")
        return

    t0 = time.time()
    ok = skip = 0
    total_bytes = 0
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, h, dest_dir): h for h in todo}
        done = 0
        for fut in as_completed(futures):
            h, status, size = fut.result()
            done += 1
            if status == "ok":
                ok += 1
                total_bytes += size
            elif status == "skip":
                skip += 1
            else:
                failures[h] = status
            if done % 500 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el if el else 0
                left = (len(todo) - done) / rate if rate else 0
                print(f"  {done:,}/{len(todo):,}  ok {ok:,}  fail {len(failures):,}  "
                      f"{rate:.0f} img/s  eta {left / 60:.0f}m", flush=True)

    el = time.time() - t0
    manifest = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": PREFIX,
        "source_registry": "https://registry.opendata.aws/multimedia-commons/",
        "design": args.design,
        "images_referenced": len(unique),
        "downloaded": ok,
        "already_present": skip,
        "failed": len(failures),
        "bytes": total_bytes,
        "wall_seconds": round(el, 1),
        "failures": dict(sorted(failures.items())[:200]),
        "licence_note": (
            "Every YFCC100M image was published on Flickr under a Creative Commons "
            "licence, with the variant differing per image. This project computes CLIP "
            "features and does not redistribute any image. See docs/data-attribution.md."
        ),
    }
    out_man = ROOT / "runs/yfcc-fetch/manifest.json"
    out_man.parent.mkdir(parents=True, exist_ok=True)
    out_man.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 60}")
    print(f"downloaded  {ok:,}   already present {skip:,}   failed {len(failures):,}")
    print(f"bytes       {total_bytes / 1e9:.2f} GB")
    print(f"wall        {el / 60:.1f} min  ({ok / el:.0f} img/s)")
    if failures:
        from collections import Counter
        print(f"failure kinds: {dict(Counter(failures.values()))}")
        print("re-run to retry; a 404 means the image is gone and will not recover")
    print(f"manifest    {out_man}")


if __name__ == "__main__":
    main()
