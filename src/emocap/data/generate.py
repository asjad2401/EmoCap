"""Stage 02 -- generate emotional rewrites with Gemini.

Runs locally against the API key; it is CPU-bound and there is no reason to spend
Kaggle GPU hours on it.

Storage is **append-only JSONL keyed by (image_id, caption_idx)**, so resuming is a
set difference and regeneration is a filter over keys. v1 stored a CSV and its
repair loop re-read and rewrote all 35,000 rows once per image -- quadratic, and
abandoned after image 1 of 1,827. Those 1,827 flagged images stayed in the
training data.

The LLM is injected as a plain ``Callable[[str], str]``, so every test here runs
offline against a stub.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from emocap.data.prompt import (
    BANNED_ADVERBS,
    EMOTIONS,
    batch_response_schema,
    build_batch_prompt,
    build_prompt,
    single_response_schema,
)

__all__ = [
    "GenerationRecord",
    "LLM",
    "validate_caption",
    "validate_all",
    "parse_response",
    "parse_batch_response",
    "generate_one",
    "generate_image_batch",
    "load_image_bytes",
    "completed_keys",
    "append_record",
    "read_records",
    "run_generation",
    "gemini_llm",
    "vertex_client",
]

#: A generation backend: ``llm(prompt, image_bytes=None, response_schema=None) -> raw_text``.
#: Injected so every test here runs offline against a stub, and so nothing else in the
#: codebase knows which provider is in use.
LLM = Callable[..., str]

_BANNED_RE = re.compile(r"\b(" + "|".join(BANNED_ADVERBS) + r")\b", re.I)
#: Abstraction frames v1 leaked constantly, in the appositive form
#: "<noun phrase>, a <modifier> <abstraction>" -- e.g. "the sled a cold reminder",
#: "a tender ballet", "the house a silent, distant witness". The modifiers sit
#: between the article and the noun, which is why a bare "a reminder of" pattern
#: misses nearly all of it.
#:
#: This is a cheap deterministic pre-filter tuned against the actual v1 output
#: (see tests/test_data_generate.py). The LLM-judge rubric in stage 03 is the real
#: style gate; this only catches the obvious cases for free.
#: Two tiers, because some of these words name real things in Flickr8k photographs.
#:
#: Tier 1 is figurative in this corpus essentially always -- "the sled a cold
#: reminder", "a tender ballet", "the house a silent, distant witness". Flagged in the
#: appositive form: article, up to two modifiers, then the noun.
_ABSTRACTION_ALWAYS = (
    "testament reminder echo symphony tapestry whisper whispers ballet witness poem "
    "melody metaphor tribute void abyss"
).split()

#: Tier 2 words are frequently literal here: Flickr8k has painted canvases, beaches
#: with tides, and people embracing. Part 0 flagged five captions for "a canvas" on
#: images that genuinely contain a painted canvas, and one for "a gentle embrace"
#: describing a couple holding their newborn. Those are false positives, so tier 2 is
#: only flagged in the unambiguously figurative "X of Y" frame -- "a canvas of light"
#: is metaphor, "a canvas with a rainbow" is a description.
_ABSTRACTION_IF_OF = (
    "canvas tide dance embrace caress portrait beacon promise mirror shadow"
).split()

_ABSTRACTION_RE = re.compile(
    r"\b(?:a|an|the)\s+(?:[a-z]+,?\s+){0,2}(?:" + "|".join(_ABSTRACTION_ALWAYS) + r")\b"
    r"|\b(?:a|an|the)\s+(?:[a-z]+,?\s+){0,2}(?:" + "|".join(_ABSTRACTION_IF_OF) + r")\s+of\b"
    # Similes are forbidden by the prompt outright, literal or not.
    r"|\bas if\b|\blike a\b|\bas though\b",
    re.I,
)
_SENTENCE_END_RE = re.compile(r"[.!?]+")


@dataclass
class GenerationRecord:
    """One API call's worth of output: five rewrites of one human caption."""

    image_id: str
    caption_idx: int
    source_caption: str
    captions: dict[str, str]
    model: str
    attempts: int = 1
    rejected: dict[str, str] = field(default_factory=dict)
    #: emotion -> the model's own report of how well the register fits this scene:
    #: 0 natural, 1 strained, 2 no honest reading exists. RECORDED ONLY. Whether
    #: strain-2 cells are excluded from training is a pre-registration decision and
    #: has not been made -- see docs/deviations.md. Absent on records generated
    #: before 2026-08-19, so always treat a missing key as "unknown", never as 0.
    strain: dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @property
    def key(self) -> tuple[str, int]:
        return (self.image_id, self.caption_idx)

    @property
    def complete(self) -> bool:
        return all(self.captions.get(e) for e in EMOTIONS)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ── validation ──────────────────────────────────────────────────────────────


def validate_caption(text: str, *, min_words: int = 8, max_words: int = 24) -> str | None:
    """Return a rejection reason, or ``None`` if the caption passes.

    Deterministic checks only. Grounding and register quality need the source
    caption and a judge, and live in stage 03.
    """
    if not text or not text.strip():
        return "empty"
    text = text.strip()

    n = len(text.split())
    if n < min_words:
        return f"{n} words, needs {min_words}-{max_words}"
    if n > max_words:
        return f"{n} words, needs {min_words}-{max_words}"

    if m := _BANNED_RE.search(text):
        return f'names the emotion with "{m.group(0)}"'

    if m := _ABSTRACTION_RE.search(text):
        return f'figurative/abstract phrasing: "{m.group(0)}"'

    # One sentence: at most one terminal punctuation run, and it must be at the end.
    ends = list(_SENTENCE_END_RE.finditer(text))
    if len(ends) > 1 or (ends and ends[0].end() != len(text)):
        return "more than one sentence"

    return None


def validate_all(
    captions: dict[str, str], *, min_words: int = 8, max_words: int = 24
) -> dict[str, str]:
    """Rejection reasons per emotion. Empty dict means everything passed."""
    out: dict[str, str] = {}
    for emotion in EMOTIONS:
        text = captions.get(emotion, "")
        if (reason := validate_caption(text, min_words=min_words, max_words=max_words)):
            out[emotion] = reason

    # Rule 6: the five rewrites must be distinguishable.
    seen: dict[str, str] = {}
    for emotion in EMOTIONS:
        norm = re.sub(r"[^a-z0-9 ]", "", captions.get(emotion, "").lower()).strip()
        if norm and norm in seen:
            out.setdefault(emotion, f"identical to the {seen[norm]} rewrite")
        elif norm:
            seen[norm] = emotion
    return out


# ── response parsing ────────────────────────────────────────────────────────


def _cell(value: object) -> tuple[str, int | None]:
    """Read one register's output, which comes in either of two shapes.

    A bare string is the original schema. ``{"text": ..., "strain": 0|1|2}`` is the
    schema from 2026-08-19 onward, where the model also reports how well the register
    fits the scene. Both are accepted so that a schema change cannot silently discard
    captions, and so records written under either shape stay readable.
    """
    if isinstance(value, dict):
        text = str(value.get("text") or "").strip()
        raw = value.get("strain")
        strain: int | None = None
        if isinstance(raw, bool):
            strain = int(raw)
        elif isinstance(raw, (int, float)):
            strain = int(raw)
        elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
            strain = int(raw.strip())
        if strain is not None and strain not in (0, 1, 2):
            strain = None          # out-of-range is unknown, not clamped to a value
        return text, strain
    return str(value or "").strip(), None


def parse_response(
    raw: str, *, strain: dict[str, int] | None = None
) -> dict[str, str]:
    """Extract the five captions from a model response.

    Tolerates a ```json fence, prose around the object, and single-key-per-line
    output. Returns whatever it found -- validation decides if that is enough.

    ``strain``, if given, is filled in as a side channel: emotion -> 0|1|2 for every
    register that reported one. It is a side channel rather than part of the return
    value so that every existing caller and test keeps working unchanged; the captions
    remain the function's contract.
    """
    if not raw:
        return {}
    text = raw.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()

    for candidate in (text, *(m.group(0) for m in re.finditer(r"\{.*?\}", text, re.S))):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            found: dict[str, str] = {}
            got: dict[str, int] = {}
            for e in EMOTIONS:
                if data.get(e) is None:
                    continue
                t, s = _cell(data[e])
                if t:
                    found[e] = t
                    if s is not None:
                        got[e] = s
            if found:
                if strain is not None:
                    strain.update(got)
                return found

    # Last resort: pull "key": "value" pairs out of malformed JSON.
    out: dict[str, str] = {}
    for emotion in EMOTIONS:
        m = re.search(rf'"{emotion}"\s*:\s*"([^"]+)"', text, re.I)
        if m:
            out[emotion] = m.group(1).strip()
    return out


# ── one call, with retry-and-feedback ───────────────────────────────────────


def parse_batch_response(
    raw: str, n_sources: int, *, strain: dict[int, dict[str, int]] | None = None
) -> dict[int, dict[str, str]]:
    """Extract an ``{caption_idx: {emotion: text}}`` matrix from a batch response.

    Missing indices and missing registers are simply absent from the result -- the
    caller reports the completion rate rather than this function guessing. That
    per-position completeness is the measurement that decides whether batching is
    safe, so it must not be papered over here.
    """
    if not raw:
        return {}
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()

    data = None
    for candidate in (text, *(m.group(0) for m in re.finditer(r"\{.*\}", text, re.S))):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
    if data is None:
        return {}

    out: dict[int, dict[str, str]] = {}
    for i in range(n_sources):
        block = data.get(str(i), data.get(i))
        if not isinstance(block, dict):
            continue
        caps: dict[str, str] = {}
        got: dict[str, int] = {}
        for e in EMOTIONS:
            if block.get(e) is None:
                continue
            text, s_ = _cell(block[e])
            if text:
                caps[e] = text
                if s_ is not None:
                    got[e] = s_
        if caps:
            out[i] = caps
            if strain is not None and got:
                strain[i] = got
    return out


def load_image_bytes(path: str | Path, *, max_dim: int | None = 512) -> bytes:
    """Read an image as JPEG bytes, optionally downscaled.

    Gemini charges images as tokens by tile, so downscaling is the main cost lever.
    512px is well above CLIP ViT-B/32's 224px input, so nothing the captioning model
    could learn is lost by capping here.
    """
    from io import BytesIO

    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max_dim and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_one(
    llm: LLM,
    source_caption: str,
    *,
    image_bytes: bytes | None = None,
    min_words: int = 8,
    max_words: int = 24,
    max_attempts: int = 3,
    on_error: Callable[[Exception, int], None] | None = None,
) -> tuple[dict[str, str], int, dict[str, str]]:
    """Generate five rewrites, retrying with targeted feedback on rejection.

    Returns ``(captions, attempts_used, remaining_rejections)``. The best attempt
    is kept: a later attempt only replaces an earlier one if it has strictly fewer
    rejections, so a retry can never make the result worse.
    """
    best: dict[str, str] = {}
    best_rejects: dict[str, str] = {e: "not generated" for e in EMOTIONS}
    failures: dict[str, tuple[str, str]] | None = None
    attempts = 0
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        prompt = build_prompt(
            source_caption, min_words=min_words, max_words=max_words,
            failures=failures, multimodal=image_bytes is not None,
        )
        try:
            raw = llm(prompt, image_bytes, single_response_schema())
        except Exception as exc:  # noqa: BLE001 -- transport errors are expected
            last_exc = exc
            if on_error:
                on_error(exc, attempt)
            continue

        captions = parse_response(raw)
        rejects = validate_all(captions, min_words=min_words, max_words=max_words)

        if len(rejects) < len(best_rejects):
            best, best_rejects = captions, rejects
        if not rejects:
            break

        failures = {e: (captions.get(e, ""), why) for e, why in rejects.items()}

    # A call that never succeeded must not be reported as "not generated". That is
    # how 150 consecutive `thinking_budget=0` rejections from gemini-3.6-flash
    # appeared in a probe as empty captions with `error: None` -- a systematic config
    # failure wearing the costume of a model-quality result.
    if not best and last_exc is not None:
        raise RuntimeError(
            f"all {attempts} attempt(s) failed with no response: {last_exc}"
        ) from last_exc

    return best, attempts, best_rejects


def generate_image_batch(
    llm: LLM,
    source_captions: Sequence[str],
    *,
    image_bytes: bytes | None = None,
    min_words: int = 8,
    max_words: int = 24,
    max_attempts: int = 1,
    use_schema: bool = True,
    emphasise_distinctness: bool = False,
    on_error: Callable[[Exception, int], None] | None = None,
) -> tuple[dict[int, dict[str, str]], int, dict[int, dict[str, str]]]:
    """Option B: rewrite every source caption of one image in a single call.

    Returns ``(matrix, attempts, rejects)`` where ``matrix`` maps caption index to
    ``{emotion: text}`` and ``rejects`` maps caption index to ``{emotion: reason}``.

    Retries request the whole matrix again -- there is no partial-repair path here on
    purpose. Repairing one register of one caption is what Option A is for, and the
    probe measures whether that fallback is needed often enough to matter.
    """
    n = len(source_captions)
    best: dict[int, dict[str, str]] = {}
    best_rejects: dict[int, dict[str, str]] = {i: {e: "not generated" for e in EMOTIONS}
                                               for i in range(n)}
    attempts = 0
    last_exc: Exception | None = None

    def n_bad(rej: dict[int, dict[str, str]]) -> int:
        return sum(len(v) for v in rej.values())

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        prompt = build_batch_prompt(
            source_captions, min_words=min_words, max_words=max_words,
            multimodal=image_bytes is not None,
            emphasise_distinctness=emphasise_distinctness,
        )
        schema = batch_response_schema(n) if use_schema else None
        try:
            raw = llm(prompt, image_bytes, schema)
        except Exception as exc:  # noqa: BLE001 -- transport errors are expected
            last_exc = exc
            if on_error:
                on_error(exc, attempt)
            continue

        matrix = parse_batch_response(raw, n)
        rejects: dict[int, dict[str, str]] = {}
        for i in range(n):
            caps = matrix.get(i, {})
            bad = validate_all(caps, min_words=min_words, max_words=max_words)
            if bad:
                rejects[i] = bad

        if n_bad(rejects) < n_bad(best_rejects):
            best, best_rejects = matrix, rejects
        if not rejects:
            break

    if not best and last_exc is not None:
        raise RuntimeError(
            f"all {attempts} attempt(s) failed with no response: {last_exc}"
        ) from last_exc

    return best, attempts, best_rejects


# ── append-only JSONL store ─────────────────────────────────────────────────


def read_records(path: str | Path) -> list[dict]:
    """Read the store, skipping a truncated final line.

    A 40,000-call run will be interrupted at some point; a half-written last line
    must not make the whole file unreadable.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated tail from an interrupted write
    return out


def completed_keys(path: str | Path, *, require_complete: bool = True) -> set[tuple[str, int]]:
    """Keys already generated, for resume.

    With ``require_complete``, a record missing any of the five emotions is *not*
    counted as done, so it gets another attempt on the next run.
    """
    keys: set[tuple[str, int]] = set()
    for rec in read_records(path):
        try:
            key = (rec["image_id"], int(rec["caption_idx"]))
        except (KeyError, TypeError, ValueError):
            continue
        if require_complete:
            caps = rec.get("captions") or {}
            if not all(caps.get(e) for e in EMOTIONS):
                continue
        keys.add(key)
    return keys


def append_record(path: str | Path, record: GenerationRecord) -> None:
    """Append one record and flush. Never rewrites the file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")
        f.flush()


# ── the runner ──────────────────────────────────────────────────────────────


def run_generation(
    llm: LLM,
    rows: Iterable,
    out_path: str | Path,
    *,
    model: str = "gemini-2.0-flash",
    images_dir: str | Path | None = None,
    max_image_dim: int | None = 512,
    min_words: int = 8,
    max_words: int = 24,
    max_attempts: int = 3,
    limit: int | None = None,
    max_error_rate: float = 0.10,
    min_before_abort: int = 20,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Generate rewrites for ``rows`` (``CaptionRow``-like), resuming from disk.

    ``rows`` items need ``.image_id``, ``.caption_idx``, ``.caption``.
    Returns a summary suitable for the run manifest.
    """
    done = completed_keys(out_path)
    pending = [r for r in rows if (r.image_id, int(r.caption_idx)) not in done]
    if limit is not None:
        pending = pending[:limit]

    stats = {
        "already_done": len(done),
        "attempted": 0,
        "written": 0,
        "incomplete": 0,
        "total_attempts": 0,
        "missing_images": 0,
        "errors_hard": 0,
        "rejection_reasons": {},
    }

    # One image is reused across its 5 source captions, so cache the encode.
    image_cache: dict[str, bytes | None] = {}

    for row in pending:
        img_bytes = None
        if images_dir is not None:
            if row.image_id not in image_cache:
                path = Path(images_dir) / row.image_id
                try:
                    image_cache[row.image_id] = load_image_bytes(path, max_dim=max_image_dim)
                except Exception:
                    image_cache[row.image_id] = None
            img_bytes = image_cache[row.image_id]
            if img_bytes is None:
                stats["missing_images"] += 1
                continue

        try:
            captions, attempts, rejects = generate_one(
                llm, row.caption, image_bytes=img_bytes,
                min_words=min_words, max_words=max_words, max_attempts=max_attempts,
            )
        except Exception as exc:  # noqa: BLE001
            stats["errors_hard"] += 1
            stats.setdefault("last_error", str(exc)[:300])
            # Abort instead of grinding through thousands of doomed calls. A broken
            # model id or argument fails every call identically; discovering that at
            # call 8,000 costs the whole run.
            if (stats["attempted"] >= min_before_abort
                    and stats["errors_hard"] / max(1, stats["attempted"] + stats["errors_hard"])
                    > max_error_rate):
                raise RuntimeError(
                    f"aborting: {stats['errors_hard']} hard failures in "
                    f"{stats['attempted'] + stats['errors_hard']} calls "
                    f"(> {max_error_rate:.0%}). Last: {exc}"
                ) from exc
            continue
        stats["attempted"] += 1
        stats["total_attempts"] += attempts

        for reason in rejects.values():
            kind = reason.split(",")[0].split(":")[0][:40]
            stats["rejection_reasons"][kind] = stats["rejection_reasons"].get(kind, 0) + 1

        record = GenerationRecord(
            image_id=row.image_id,
            caption_idx=int(row.caption_idx),
            source_caption=row.caption,
            captions=captions,
            model=model,
            attempts=attempts,
            rejected=rejects,
        )
        if captions:
            append_record(out_path, record)
            stats["written"] += 1
            if not record.complete:
                stats["incomplete"] += 1

        if progress:
            progress({"key": record.key, "attempts": attempts, "rejected": rejects, **stats})

    return stats


# ── the real client ─────────────────────────────────────────────────────────


#: HTTP statuses worth retrying: rate limits, and transient server faults.
#: Everything else (404 for a retired model, 400 for a bad request, 403 for a bad
#: key) is permanent -- retrying only delays the error and buries the message.
_RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504)


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code in _RETRYABLE_STATUS
    text = str(exc)
    if any(str(s) in text for s in _RETRYABLE_STATUS):
        return True
    # Transport-level failures carry no status but are worth another attempt.
    return any(
        w in text.lower()
        for w in ("timeout", "timed out", "connection", "temporarily", "unavailable retry")
    ) and "no longer available" not in text.lower()


def vertex_client(project: str, location: str = "us-central1",
                  service_account_file: str | Path | None = None,
                  timeout_s: float = 180.0):
    """A genai client routed through Vertex AI rather than AI Studio.

    Vertex bills to a Google Cloud project, so promotional GCP credits apply. Two ways
    to authenticate, and the service-account route is used here because it needs no
    `gcloud` install and no browser:

    * ``service_account_file`` -- a JSON key. Long-lived, so it MUST stay out of the
      repository; `.gitignore` covers `*.json` under `secrets/`.
    * otherwise Application Default Credentials, which is what
      `gcloud auth application-default login` writes.

    Note the model catalogue differs from AI Studio's: names and availability are not
    the same, so validate before committing a run. `gemini-2.5-flash` for instance is
    refused by AI Studio for new projects but may exist on Vertex.

    ``timeout_s`` is **not optional in practice.** With no client timeout the SDK inherits
    no socket deadline, so a connection dropped mid-request never returns and never raises.
    The retry wrapper in :func:`gemini_llm` classifies *exceptions*, so a call that hangs
    is invisible to it: a run stalled for six hours at 139/170 images with 0.03s of CPU
    consumed and no error in the log, and only a stale file mtime revealed it. A request
    that has produced nothing in three minutes is dead; failing it lets the retry work.
    """
    from google import genai

    creds = None
    if service_account_file:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            str(service_account_file),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    from google.genai import types as _gt

    return genai.Client(vertexai=True, project=project, location=location,
                        credentials=creds,
                        # milliseconds, per the SDK's HttpOptions contract
                        http_options=_gt.HttpOptions(timeout=int(timeout_s * 1000)))


def gemini_llm(
    api_key: str | None = None,
    *,
    model: str,
    temperature: float = 0.9,
    max_output_tokens: int = 4096,
    thinking_budget: int | None = 0,
    max_retries: int = 4,
    base_delay: float = 2.0,
    client=None,
) -> LLM:
    """A multimodal ``llm(prompt, image_bytes=None) -> str`` over Gemini.

    The returned callable carries a ``.usage`` list -- one dict per successful call
    with ``prompt_tokens``, ``output_tokens`` and ``total_tokens`` read from the
    API's own ``usage_metadata``. That is what makes the cost of a full run
    measurable from a small probe batch instead of estimated from list prices.

    Pass ``client`` to supply a pre-built client -- e.g. :func:`vertex_client` -- so the
    same generation code runs against AI Studio or Vertex without duplicating it. Exactly
    one of ``api_key`` or ``client`` is required.
    """
    from google import genai
    from google.genai import types

    if client is None:
        if not api_key:
            raise ValueError("pass either api_key or client")
        client = genai.Client(api_key=api_key)
    usage: list[dict] = []

    def call(
        prompt: str,
        image_bytes: bytes | None = None,
        response_schema: dict | None = None,
    ) -> str:
        # Thinking tokens are billed as output and count against
        # max_output_tokens. Measured on a trivial prompt, flash models spend
        # 26-29 of them by default -- so a small budget can be consumed entirely by
        # thinking, returning empty text that looks like a model-quality problem
        # rather than a configuration one. This task is constrained rewriting, not
        # reasoning, so thinking defaults to off; the probe measures whether that
        # costs any caption quality.
        extra: dict = {}
        if response_schema:
            extra["response_schema"] = response_schema
        if thinking_budget is not None:
            try:
                extra["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
            except (AttributeError, TypeError):
                pass  # SDK or model without thinking control; harmless
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            **extra,
        )
        parts: list = []
        if image_bytes is not None:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text=prompt))

        last: Exception | None = None
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=cfg,
                )
                latency = time.time() - t0
                um = getattr(resp, "usage_metadata", None)
                finish = None
                try:
                    finish = str(resp.candidates[0].finish_reason)
                except Exception:  # noqa: BLE001 -- telemetry only
                    pass
                usage.append({
                    "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                    "thinking_tokens": getattr(um, "thoughts_token_count", 0) or 0,
                    "total_tokens": getattr(um, "total_token_count", 0) or 0,
                    "latency_s": round(latency, 3),
                    "finish_reason": finish,
                    "had_image": image_bytes is not None,
                    "image_bytes": len(image_bytes) if image_bytes else 0,
                    "schema": bool(response_schema),
                })
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001
                last = exc
                # Retrying a permanent error wastes the backoff and hides the cause.
                # A retired model returns 404 in 0.7s; four retries with exponential
                # backoff turned that into a silent ~14s hang per call, and the real
                # message ("no longer available to new users") never surfaced.
                if not _is_retryable(exc):
                    raise RuntimeError(f"Gemini call failed permanently: {exc}") from exc
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))
        raise RuntimeError(
            f"Gemini call failed after {max_retries} attempts: {last}"
        ) from last

    def count_text_tokens(prompt: str) -> int:
        """Tokens for the prompt text alone, via the API's own counter.

        Subtracting this from a multimodal call's prompt_token_count isolates the
        image's token cost, which is the quantity the resolution decision turns on.
        """
        r = client.models.count_tokens(model=model, contents=prompt)
        return int(getattr(r, "total_tokens", 0) or 0)

    call.usage = usage  # type: ignore[attr-defined]
    call.count_text_tokens = count_text_tokens  # type: ignore[attr-defined]
    return call
