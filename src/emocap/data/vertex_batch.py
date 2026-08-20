"""Stage 02 through the Vertex AI Batch API -- half the token rate, and no realtime quota.

Two reasons this exists, and the second matters more than the discount:

* **Cost.** Batch submission is billed at 50% of the realtime token rate. On measured
  tokens (7,741 in / 1,040 out per image on 3.7-flash) the 7,937-image corpus is ~$39
  batch against ~$77 realtime.
* **Quota.** Realtime 3.7-flash collapsed at concurrency 4: 57 of 100 images failed, the
  rate *climbing* through the run (7 -> 22 -> 39 -> 57) as the limiter tightened. Four
  sequential calls then succeeded, so it was throttling rather than a model fault, and
  concurrency 2 completed the remaining 57 with zero errors -- at ~30 hours for a full
  corpus. Batch queues server-side and does not compete for that quota at all.

Why this is separate from :mod:`emocap.data.batch`
--------------------------------------------------
That module targets the **Gemini Developer API**, which accepts a list of inlined
requests. Vertex accepts **only** ``gs://`` or ``bq://`` sources, so the requests have to
be staged as JSONL in Cloud Storage and the results collected from an output prefix. The
request shape differs too: Vertex wraps each row in a ``{"request": {...}}`` envelope.

Everything still lands in the same append-only JSONL keyed by ``(image_id, caption_idx)``,
so batch and realtime remain interchangeable and a half-finished batch resumes as a set
difference.

**Row order is not guaranteed.** Vertex may return rows in any order, so each request
carries the ``image_id`` in a ``labels``-style key echoed back in the response, and rows
are matched by that rather than by position. Position-matching is how a batch run
mis-attributes every caption to the wrong photograph while looking perfectly healthy.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from emocap.data.generate import (
    GenerationRecord,
    append_record,
    load_image_bytes,
    parse_batch_response,
    validate_all,
)
from emocap.data.prompt import batch_response_schema, build_batch_prompt

__all__ = ["BatchRequest", "build_jsonl", "upload", "submit", "wait", "collect",
           "run_vertex_batch"]

_TERMINAL = ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
             "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED")


@dataclass
class BatchRequest:
    """One image's worth of work: the prompt, the picture, and how to get back."""

    image_id: str
    sources: list[str]
    prompt: str
    image_b64: str
    mime: str = "image/jpeg"


def build_jsonl(
    requests: Sequence[BatchRequest],
    *,
    temperature: float,
    max_output_tokens: int,
    thinking_budget: int | None,
) -> str:
    """Serialise requests into the JSONL Vertex batch expects.

    The ``image_id`` is threaded through as the first text part of a *system*
    instruction rather than a request field, because Vertex echoes the request back in
    each response row and this is the only place that survives round-trip unmodified in
    every SDK version tried. It is read back by :func:`collect` to match rows to images.
    """
    lines = []
    for r in requests:
        gen_cfg: dict = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": batch_response_schema(len(r.sources)),
        }
        if thinking_budget is not None:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        lines.append(json.dumps({
            "request": {
                "contents": [{
                    "role": "user",
                    "parts": [
                        {"text": f"[[image_id:{r.image_id}]]\n{r.prompt}"},
                        {"inlineData": {"mimeType": r.mime, "data": r.image_b64}},
                    ],
                }],
                "generationConfig": gen_cfg,
            },
        }))
    return "\n".join(lines) + "\n"


def upload(client_storage, bucket_name: str, blob_path: str, text: str,
           *, location: str = "us-central1") -> str:
    """Write ``text`` to ``gs://bucket/blob_path``, creating the bucket if needed."""
    try:
        bucket = client_storage.get_bucket(bucket_name)
    except Exception:
        bucket = client_storage.create_bucket(bucket_name, location=location)
    bucket.blob(blob_path).upload_from_string(text, content_type="application/json")
    return f"gs://{bucket_name}/{blob_path}"


def submit(client, *, model: str, src_uri: str, dest_uri: str, display_name: str):
    """Create the batch job. Returns the job object; poll it with :func:`wait`."""
    from google.genai import types as gt

    return client.batches.create(
        model=model,
        src=src_uri,
        config=gt.CreateBatchJobConfig(display_name=display_name, dest=dest_uri),
    )


def wait(client, job, *, poll_s: float = 20.0, timeout_s: float = 86_400.0,
         on_poll: Callable[[str, float], None] | None = None):
    """Poll until the job reaches a terminal state.

    A submitted job runs and **bills** on Google's side whether or not this process is
    still watching, so a crash here loses the collection, not the work. The job name is
    printed by the caller for exactly that reason.
    """
    t0 = time.time()
    name = job.name
    while True:
        try:
            job = client.batches.get(name=name)
        except Exception as exc:  # noqa: BLE001
            # A dropped socket must never end the wait. The job runs and BILLS on
            # Google's side regardless, so an exception here loses the collection, not
            # the work -- a bare poll loop died on one RemoteProtocolError and orphaned
            # a job that was already halfway through.
            if on_poll:
                on_poll(f"poll error {type(exc).__name__}, retrying", time.time() - t0)
            if time.time() - t0 > timeout_s:
                raise
            time.sleep(poll_s)
            continue
        state = str(job.state)
        if on_poll:
            on_poll(state, time.time() - t0)
        if any(s in state for s in _TERMINAL):
            return job
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"batch {name} still {state} after {timeout_s}s")
        time.sleep(poll_s)


def _iter_result_rows(client_storage, dest_uri: str) -> Iterable[dict]:
    """Yield parsed rows from every prediction file under the output prefix."""
    assert dest_uri.startswith("gs://")
    bucket_name, _, prefix = dest_uri[5:].partition("/")
    bucket = client_storage.bucket(bucket_name)
    for blob in client_storage.list_blobs(bucket, prefix=prefix):
        if not blob.name.endswith((".jsonl", ".json")):
            continue
        for line in blob.download_as_text().splitlines():
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _row_image_id(row: dict) -> str | None:
    """Recover the image_id echoed in the request half of a result row."""
    try:
        parts = row["request"]["contents"][0]["parts"]
    except (KeyError, IndexError, TypeError):
        return None
    for p in parts:
        t = p.get("text") if isinstance(p, dict) else None
        if t and t.startswith("[[image_id:"):
            return t[len("[[image_id:"):].split("]]", 1)[0]
    return None


def _row_text(row: dict) -> str:
    for key in ("response", "candidates"):
        if key in row:
            node = row[key]
            break
    else:
        return ""
    try:
        cands = node["candidates"] if isinstance(node, dict) else node
        return cands[0]["content"]["parts"][0].get("text", "")
    except (KeyError, IndexError, TypeError):
        return ""


def collect(client_storage, dest_uri: str, *, sources: dict[str, list[str]],
            out_path: Path, model: str, min_words: int, max_words: int,
            max_sentences: int = 1,
            banned_by_register: dict[str, Sequence[str]] | None = None) -> dict:
    """Parse the output prefix and append records. Returns counts.

    Rows are matched to images by the echoed ``image_id``, never by position -- see the
    module docstring.
    """
    stats: dict = {"rows": 0, "unmatched": 0, "images": 0, "captions": 0,
                   "rejections": 0, "strain": {}, "row_errors": 0,
                   "row_error_kinds": {}, "failed_images": []}
    for row in _iter_result_rows(client_storage, dest_uri):
        stats["rows"] += 1
        img = _row_image_id(row)
        if img is None or img not in sources:
            stats["unmatched"] += 1
            continue
        # A row can fail while the JOB reports SUCCEEDED, and the failure can be
        # transient: an identical schema was rejected in one job and accepted in the
        # next 20 minutes later. So per-row status is checked and the image recorded
        # for resubmission -- counting only the job state would silently drop captions.
        if (st := row.get("status")):
            try:
                msg = json.loads(st).get("message", str(st))
            except (json.JSONDecodeError, TypeError):
                msg = str(st)
            if not (row.get("response") or {}).get("candidates"):
                stats["row_errors"] += 1
                kind = msg.split(" with status")[0][:90]
                stats["row_error_kinds"][kind] = stats["row_error_kinds"].get(kind, 0) + 1
                stats["failed_images"].append(img)
                continue
        srcs = sources[img]
        strain: dict[int, dict[str, int]] = {}
        matrix = parse_batch_response(_row_text(row), len(srcs), strain=strain)
        wrote = 0
        for idx, src in enumerate(srcs):
            caps = matrix.get(idx, {})
            if not caps:
                continue
            rej = validate_all(caps, min_words=min_words, max_words=max_words,
                               max_sentences=max_sentences,
                               banned_by_register=banned_by_register)
            append_record(out_path, GenerationRecord(
                image_id=img, caption_idx=idx, source_caption=src,
                captions=caps, model=model, attempts=1, rejected=rej,
                strain=strain.get(idx, {}),
            ))
            wrote += len([c for c in caps.values() if c])
            stats["rejections"] += len(rej)
            for v in strain.get(idx, {}).values():
                stats["strain"][str(v)] = stats["strain"].get(str(v), 0) + 1
        if wrote:
            stats["images"] += 1
            stats["captions"] += wrote
    return stats


def run_vertex_batch(
    *,
    client,
    client_storage,
    model: str,
    image_ids: Sequence[str],
    sources: dict[str, list[str]],
    images_dir: Path,
    out_path: Path,
    bucket: str,
    prefix: str,
    image_max_dim: int,
    temperature: float,
    max_output_tokens: int,
    thinking_budget: int | None,
    min_words: int,
    max_words: int,
    max_sentences: int = 1,
    banned_by_register: dict[str, Sequence[str]] | None = None,
    location: str = "us-central1",
    on_poll: Callable[[str, float], None] | None = None,
) -> dict:
    """Stage, submit, wait, and collect one batch of images."""
    reqs = []
    for img in image_ids:
        b = load_image_bytes(images_dir / img, max_dim=image_max_dim)
        reqs.append(BatchRequest(
            image_id=img, sources=sources[img],
            prompt=build_batch_prompt(
                sources[img], min_words=min_words, max_words=max_words,
                emphasise_distinctness=True,
                banned_by_register=banned_by_register),
            image_b64=base64.b64encode(b).decode(),
        ))
    payload = build_jsonl(reqs, temperature=temperature,
                          max_output_tokens=max_output_tokens,
                          thinking_budget=thinking_budget)
    src_uri = upload(client_storage, bucket, f"{prefix}/input.jsonl", payload,
                     location=location)
    dest_uri = f"gs://{bucket}/{prefix}/out"
    job = submit(client, model=model, src_uri=src_uri, dest_uri=dest_uri,
                 display_name=prefix.replace("/", "-")[:60])
    job = wait(client, job, on_poll=on_poll)
    state = str(job.state)
    stats = collect(client_storage, dest_uri, sources=sources, out_path=out_path,
                    model=model, min_words=min_words, max_words=max_words,
                    max_sentences=max_sentences, banned_by_register=banned_by_register)
    stats["job"] = job.name
    stats["state"] = state
    stats["src"] = src_uri
    stats["dest"] = dest_uri
    return stats
