#!/usr/bin/env python
"""Retrieve a Gemini batch job whose run was interrupted.

A submitted batch job runs and bills on Google's side whether or not our process is
still watching it. `httpx.RemoteProtocolError: Server disconnected` during polling
therefore does not stop the work -- it only stops us collecting it, and the captions
are paid for either way. This recovers them.

    # what jobs exist, and which of ours are unclaimed
    uv run python scripts/recover_batch.py --list

    # collect one, using a run's manifest for the image order
    uv run python scripts/recover_batch.py \
        --job batches/xxxx --manifest runs/stage02-audit-.../manifest.json

The image order matters and cannot be guessed: responses come back positionally, so
they are matched against `config.image_ids` from the manifest that submitted them. If
the counts disagree this refuses to write rather than mis-attributing captions to
images -- the same guard `run_batch_generation` uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emocap.data import assign_official_splits, drop_malformed, read_captions  # noqa: E402
from emocap.data.batch import _responses  # noqa: E402
from emocap.data.generate import (  # noqa: E402
    EMOTIONS,
    GenerationRecord,
    append_record,
    completed_keys,
    parse_batch_response,
    validate_all,
)
from emocap.runtime import load_config  # noqa: E402


def load_key() -> str:
    import os

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                key = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not key:
        sys.exit("No API key. Put GEMINI_API_KEY=... in .env (gitignored).")
    return key


def sources_by_image() -> dict[str, list[str]]:
    rows = drop_malformed(read_captions(ROOT / "data/flickr8k/Flickr8k.token.txt"))
    out: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda x: (x.image_id, x.caption_idx)):
        out.setdefault(r.image_id, []).append(r.caption)
    return {k: v for k, v in out.items() if len(v) == 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list recent batch jobs and exit")
    ap.add_argument("--job", help="batch job name, e.g. batches/xxxx")
    ap.add_argument("--manifest", help="manifest.json of the run that submitted it")
    ap.add_argument("--out", help="override the output path from the manifest")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds to wait if running")
    args = ap.parse_args()

    from google import genai

    client = genai.Client(api_key=load_key())

    if args.list:
        print(f"{'job':<46} {'state':<26} created")
        for j in client.batches.list(config={"page_size": 20}):
            state = str(j.state).replace("JobState.", "")
            print(f"{j.name:<46} {state:<26} {getattr(j, 'create_time', '')}")
        return

    if not args.job or not args.manifest:
        sys.exit("--job and --manifest are both required (or use --list)")

    man_path = Path(args.manifest)
    man = json.loads(man_path.read_text())
    cfg_block = man.get("config", {})
    image_ids: list[str] = cfg_block.get("image_ids") or []
    if not image_ids:
        sys.exit(f"{man_path} has no config.image_ids -- cannot establish response order")

    out_path = Path(args.out) if args.out else ROOT / cfg_block["out_path"]
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config("data")
    min_words = cfg["data"]["target_caption_words"]["min"]
    max_words = cfg["data"]["target_caption_words"]["max"]
    model = cfg_block.get("model", cfg["generation"]["model"])

    def poll(attempts: int = 8):
        """Fetch job state, retrying transport faults.

        The connection to Google drops mid-request often enough that a bare
        `batches.get` loses a job that is running fine on the server. The job is
        billing either way, so giving up on a network blip is the one outcome worth
        engineering against. Only transport-level faults are retried; a real API
        error still raises.
        """
        delay = 3.0
        for attempt in range(1, attempts + 1):
            try:
                return client.batches.get(name=args.job)
            except Exception as exc:  # noqa: BLE001
                transient = exc.__class__.__name__ in (
                    "RemoteProtocolError", "ConnectError", "ConnectTimeout",
                    "ReadTimeout", "ReadError", "WriteError", "PoolTimeout",
                    "ServerDisconnectedError",
                )
                if not transient or attempt == attempts:
                    raise
                print(f"  transport fault ({exc.__class__.__name__}), retry "
                      f"{attempt}/{attempts - 1} in {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    job = poll()
    t0 = time.time()
    while str(job.state).endswith(("PENDING", "RUNNING")):
        if time.time() - t0 > args.timeout:
            sys.exit(f"still {job.state} after {args.timeout}s -- rerun later")
        print(f"  {args.job}  {str(job.state).replace('JobState.', '')}  "
              f"{time.time() - t0:.0f}s", flush=True)
        time.sleep(20)
        job = poll()

    state = str(job.state).replace("JobState.", "")
    if not state.endswith("SUCCEEDED"):
        sys.exit(f"job ended {state} -- nothing to recover")

    resp = _responses(job)
    if len(resp) != len(image_ids):
        sys.exit(
            f"{len(resp)} responses for {len(image_ids)} image_ids in the manifest -- "
            f"refusing to guess which image each belongs to"
        )

    sources = sources_by_image()
    done = completed_keys(out_path)
    stats = {"recovered_from": args.job, "manifest": str(man_path),
             "images_written": 0, "captions_written": 0, "captions_expected": 0,
             "skipped_already_present": 0, "rejections": 0, "rejection_reasons": {},
             "strain": {}, "usage": {"prompt": 0, "output": 0, "thinking": 0}}

    for img_id, rr in zip(image_ids, resp):
        srcs = sources.get(img_id)
        if not srcs:
            continue
        stats["captions_expected"] += len(srcs) * len(EMOTIONS)
        inner = getattr(rr, "response", None)
        if inner is None:
            continue
        um = getattr(inner, "usage_metadata", None)
        if um is not None:
            stats["usage"]["prompt"] += getattr(um, "prompt_token_count", 0) or 0
            stats["usage"]["output"] += getattr(um, "candidates_token_count", 0) or 0
            stats["usage"]["thinking"] += getattr(um, "thoughts_token_count", 0) or 0

        strain: dict[int, dict[str, int]] = {}
        matrix = parse_batch_response(inner.text or "", len(srcs), strain=strain)
        wrote_any = False
        for idx, srctext in enumerate(srcs):
            if (img_id, idx) in done:
                stats["skipped_already_present"] += 1
                continue
            caps = matrix.get(idx, {})
            if not caps:
                continue
            rejects = validate_all(caps, min_words=min_words, max_words=max_words)
            for reason in rejects.values():
                kind = reason.split(",")[0].split(":")[0][:40]
                stats["rejection_reasons"][kind] = stats["rejection_reasons"].get(kind, 0) + 1
            stats["rejections"] += len(rejects)
            append_record(out_path, GenerationRecord(
                image_id=img_id, caption_idx=idx, source_caption=srctext,
                captions=caps, model=model, attempts=1, rejected=rejects,
                strain=strain.get(idx, {}),
            ))
            stats["captions_written"] += len([c for c in caps.values() if c])
            for v in strain.get(idx, {}).values():
                stats["strain"][str(v)] = stats["strain"].get(str(v), 0) + 1
            wrote_any = True
        if wrote_any:
            stats["images_written"] += 1

    P = cfg["pricing"]["models"][model]
    u = stats["usage"]
    cost = (u["prompt"] / 1e6 * P["input"]
            + (u["output"] + u["thinking"]) / 1e6 * P["output"]) * 0.5
    stats["measured_cost_usd"] = round(cost, 4)

    (man_path.parent / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    man.setdefault("checks", {})["recovered_batch"] = {
        "passed": stats["captions_written"] == stats["captions_expected"],
        "detail": f"{stats['captions_written']}/{stats['captions_expected']}",
    }
    man["status"] = "recovered"
    man_path.write_text(json.dumps(man, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"recovered from  {args.job}")
    print(f"images written  {stats['images_written']:,}")
    print(f"captions        {stats['captions_written']:,} / {stats['captions_expected']:,}")
    print(f"already present {stats['skipped_already_present']:,}")
    print(f"rejections      {stats['rejections']:,}  {stats['rejection_reasons']}")
    print(f"strain          {stats['strain'] or 'none reported'}")
    print(f"tokens          in {u['prompt']:,}  out {u['output']:,}  think {u['thinking']:,}")
    print(f"measured cost   ${cost:.4f} (batch rate, already billed)")
    print(f"store           {out_path}")


if __name__ == "__main__":
    main()
