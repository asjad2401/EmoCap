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
from typing import Callable, Iterable

from emocap.data.prompt import BANNED_ADVERBS, EMOTIONS, build_prompt

__all__ = [
    "GenerationRecord",
    "LLM",
    "validate_caption",
    "validate_all",
    "parse_response",
    "generate_one",
    "completed_keys",
    "append_record",
    "read_records",
    "run_generation",
    "gemini_llm",
]

LLM = Callable[[str], str]

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
_ABSTRACTION_NOUNS = (
    "testament reminder echo symphony dance tapestry canvas whisper whispers tide "
    "ballet witness poem melody metaphor promise portrait tribute embrace caress "
    "void abyss beacon"
).split()
_ABSTRACTION_RE = re.compile(
    # "a" / "an" / "the", up to two comma-separated modifiers, then an abstraction noun
    r"\b(?:a|an|the)\s+(?:[a-z]+,?\s+){0,2}(?:" + "|".join(_ABSTRACTION_NOUNS) + r")\b"
    # or an explicit simile
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


def parse_response(raw: str) -> dict[str, str]:
    """Extract the five captions from a model response.

    Tolerates a ```json fence, prose around the object, and single-key-per-line
    output. Returns whatever it found -- validation decides if that is enough.
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
            found = {e: str(data[e]).strip() for e in EMOTIONS if data.get(e)}
            if found:
                return found

    # Last resort: pull "key": "value" pairs out of malformed JSON.
    out: dict[str, str] = {}
    for emotion in EMOTIONS:
        m = re.search(rf'"{emotion}"\s*:\s*"([^"]+)"', text, re.I)
        if m:
            out[emotion] = m.group(1).strip()
    return out


# ── one call, with retry-and-feedback ───────────────────────────────────────


def generate_one(
    llm: LLM,
    source_caption: str,
    *,
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

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        prompt = build_prompt(
            source_caption, min_words=min_words, max_words=max_words, failures=failures
        )
        try:
            raw = llm(prompt)
        except Exception as exc:  # noqa: BLE001 -- transport errors are expected
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
    min_words: int = 8,
    max_words: int = 24,
    max_attempts: int = 3,
    limit: int | None = None,
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
        "rejection_reasons": {},
    }

    for row in pending:
        captions, attempts, rejects = generate_one(
            llm, row.caption,
            min_words=min_words, max_words=max_words, max_attempts=max_attempts,
        )
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


def gemini_llm(
    api_key: str,
    *,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.9,
    max_output_tokens: int = 512,
    max_retries: int = 4,
    base_delay: float = 2.0,
) -> LLM:
    """A ``Callable[[str], str]`` over Gemini, with exponential backoff.

    Kept behind the same interface the tests stub, so nothing else in the codebase
    knows which provider is being used.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )

    def call(prompt: str) -> str:
        last: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=cfg
                )
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001 -- retry any transport failure
                last = exc
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))
        raise RuntimeError(f"Gemini call failed after {max_retries} attempts") from last

    return call
