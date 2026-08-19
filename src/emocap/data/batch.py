"""Stage 02 via the Gemini Batch API -- half the price of realtime.

Verified 2026-08-18: 5 inlined requests carrying an image and a `response_schema`
returned `JOB_STATE_SUCCEEDED` in 112s with token counts matching realtime.

Shape of the run
----------------
One request per **image**, returning all 25 captions for it (5 human captions x 5
registers). The image is sent once rather than five times, which is where most of
the saving comes from -- input drops from 77.6M tokens to 17.4M.

Requests are chunked into jobs because the payload is ~65 KB each (a base64 image
plus a ~1,100-token prompt): 250 per job is ~16.7 MB, comfortably inside inline
limits, and gives 33 resumable jobs over the corpus.

Everything lands in the same append-only JSONL the realtime path writes, keyed by
`(image_id, caption_idx)`, so resume is a set difference and the two transports are
interchangeable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from emocap.data.generate import (
    GenerationRecord,
    append_record,
    completed_keys,
    load_image_bytes,
    parse_batch_response,
    validate_all,
)
from emocap.data.prompt import EMOTIONS, batch_response_schema, build_batch_prompt

__all__ = ["BatchChunk", "build_requests", "submit_and_wait", "run_batch_generation"]

_TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")


class BatchChunk:
    """One batch job's worth of images, and the mapping back to their captions."""

    def __init__(self, image_ids: Sequence[str], sources: dict[str, list[str]]):
        self.image_ids = list(image_ids)
        self.sources = sources          # image_id -> [5 human captions]


def build_requests(chunk: BatchChunk, *, images_dir: str | Path, image_max_dim: int,
                   min_words: int, max_words: int, emphasise_distinctness: bool,
                   temperature: float, max_output_tokens: int,
                   thinking_budget: int | None):
    """Build one InlinedRequest per image. Returns (requests, kept_image_ids).

    Images that cannot be read are skipped and reported rather than silently
    producing a text-only request -- an ungrounded row with no record that it
    differs is worse than a missing one.
    """
    from google.genai import types

    reqs, kept = [], []
    extra: dict = {}
    if thinking_budget is not None:
        try:
            extra["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        except (AttributeError, TypeError):
            pass

    for img_id in chunk.image_ids:
        srcs = chunk.sources[img_id]
        try:
            data = load_image_bytes(Path(images_dir) / img_id, max_dim=image_max_dim)
        except Exception:  # noqa: BLE001
            continue
        prompt = build_batch_prompt(
            srcs, min_words=min_words, max_words=max_words,
            multimodal=True, emphasise_distinctness=emphasise_distinctness,
        )
        reqs.append(types.InlinedRequest(
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=data, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt),
            ])],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_schema=batch_response_schema(len(srcs)),
                **extra,
            ),
        ))
        kept.append(img_id)
    return reqs, kept


def submit_and_wait(client, model: str, requests, *, display_name: str,
                    poll_seconds: int = 20, timeout_seconds: int = 7200,
                    on_poll: Callable[[str, float], None] | None = None):
    """Submit one batch job and block until it reaches a terminal state.

    Raises on a non-SUCCEEDED terminal state or on timeout, so a failed job cannot
    be mistaken for a job that produced nothing.
    """
    from google.genai import types

    job = client.batches.create(
        model=model, src=requests,
        config=types.CreateBatchJobConfig(display_name=display_name),
    )
    t0 = time.time()
    while True:
        job = client.batches.get(name=job.name)
        state = str(job.state).rsplit(".", 1)[-1]
        if any(state.endswith(t) for t in _TERMINAL):
            break
        if time.time() - t0 > timeout_seconds:
            raise TimeoutError(f"batch {job.name} still {state} after {timeout_seconds}s")
        if on_poll:
            on_poll(state, time.time() - t0)
        time.sleep(poll_seconds)

    if not state.endswith("SUCCEEDED"):
        raise RuntimeError(f"batch {job.name} ended {state}: {job.error}")
    return job, time.time() - t0


def _responses(job) -> list:
    dest = getattr(job, "dest", None)
    return list(getattr(dest, "inlined_responses", None) or []) if dest else []


def run_batch_generation(
    client,
    model: str,
    image_ids: Iterable[str],
    sources: dict[str, list[str]],
    out_path: str | Path,
    *,
    images_dir: str | Path,
    chunk_size: int = 250,
    image_max_dim: int = 384,
    min_words: int = 8,
    max_words: int = 24,
    emphasise_distinctness: bool = True,
    temperature: float = 0.9,
    max_output_tokens: int = 4096,
    thinking_budget: int | None = 0,
    limit_images: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Generate captions for `image_ids`, resuming from `out_path`.

    An image counts as done only when all 5 of its `(image_id, caption_idx)` keys are
    present and complete, so a partially-returned image is retried whole.
    """
    done = completed_keys(out_path)
    pending = [
        i for i in image_ids
        if any((i, idx) not in done for idx in range(len(sources.get(i, []))))
    ]
    if limit_images is not None:
        pending = pending[:limit_images]

    stats = {
        "images_already_done": len({k[0] for k in done}),
        "images_pending": len(pending),
        "jobs": 0, "jobs_failed": 0,
        "images_written": 0, "captions_written": 0,
        "captions_expected": 0, "images_skipped_no_file": 0,
        "rejections": 0, "rejection_reasons": {},
        "records_already_present": 0,
        # counts of the model's own strain report: "0" natural, "1" strained,
        # "2" no honest reading exists. Recorded, not acted on.
        "strain": {},
        "wall_seconds": 0.0, "usage": {"prompt": 0, "output": 0, "thinking": 0},
    }

    for start in range(0, len(pending), chunk_size):
        batch_ids = pending[start:start + chunk_size]
        chunk = BatchChunk(batch_ids, sources)
        reqs, kept = build_requests(
            chunk, images_dir=images_dir, image_max_dim=image_max_dim,
            min_words=min_words, max_words=max_words,
            emphasise_distinctness=emphasise_distinctness,
            temperature=temperature, max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
        )
        stats["images_skipped_no_file"] += len(batch_ids) - len(kept)
        if not reqs:
            continue

        tag = f"emocap-stage02-{start // chunk_size:03d}"
        job, wall = submit_and_wait(
            client, model, reqs, display_name=tag,
            on_poll=(lambda st, el, t=tag: progress and progress(
                {"event": "poll", "job": t, "state": st, "elapsed": el})),
        )
        stats["jobs"] += 1
        stats["wall_seconds"] += wall

        resp = _responses(job)
        if len(resp) != len(kept):
            raise RuntimeError(
                f"{tag}: {len(resp)} responses for {len(kept)} requests -- refusing to "
                f"guess which image each belongs to"
            )

        for img_id, rr in zip(kept, resp):
            srcs = sources[img_id]
            stats["captions_expected"] += len(srcs) * len(EMOTIONS)
            inner = getattr(rr, "response", None)
            if inner is None:
                stats["jobs_failed"] += 0  # per-response error, not a job failure
                continue
            um = getattr(inner, "usage_metadata", None)
            if um is not None:
                stats["usage"]["prompt"] += getattr(um, "prompt_token_count", 0) or 0
                stats["usage"]["output"] += getattr(um, "candidates_token_count", 0) or 0
                stats["usage"]["thinking"] += getattr(um, "thoughts_token_count", 0) or 0

            strain: dict[int, dict[str, int]] = {}
            matrix = parse_batch_response(inner.text or "", len(srcs), strain=strain)
            for idx, srctext in enumerate(srcs):
                # An image is retried WHOLE, because the API cannot be asked for a
                # single caption_idx -- but only the MISSING records may be written.
                # Without this, topping up one absent record re-appends the four that
                # already exist, and the store gains duplicate (image_id, caption_idx)
                # keys that every downstream count then double-counts.
                if (img_id, idx) in done:
                    stats["records_already_present"] += 1
                    continue
                caps = matrix.get(idx, {})
                rejects = validate_all(caps, min_words=min_words, max_words=max_words)
                for reason in rejects.values():
                    kind = reason.split(",")[0].split(":")[0][:40]
                    stats["rejection_reasons"][kind] = \
                        stats["rejection_reasons"].get(kind, 0) + 1
                stats["rejections"] += len(rejects)
                if not caps:
                    continue
                append_record(out_path, GenerationRecord(
                    image_id=img_id, caption_idx=idx, source_caption=srctext,
                    captions=caps, model=model, attempts=1, rejected=rejects,
                    strain=strain.get(idx, {}),
                ))
                stats["captions_written"] += len([c for c in caps.values() if c])
                for v in strain.get(idx, {}).values():
                    stats["strain"][str(v)] = stats["strain"].get(str(v), 0) + 1
            stats["images_written"] += 1

        if progress:
            progress({"event": "chunk_done", "job": tag, "wall": wall, **stats})

    return stats
