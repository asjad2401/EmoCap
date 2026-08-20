"""Tests for stage 02 -- caption generation.

Everything here runs offline: the LLM is a plain `Callable[[str], str]`, so the
tests stub it. The style filter is pinned against captions the v1 pilot actually
produced, which is the only honest way to know it catches what went wrong.
"""

from __future__ import annotations

import json

import pytest

from emocap.data.generate import (
    GenerationRecord,
    append_record,
    completed_keys,
    generate_one,
    parse_response,
    read_records,
    run_generation,
    validate_all,
    validate_caption,
)
from emocap.data.prompt import EMOTIONS

GOOD = "A child in a pink dress bounds up the entryway stairs, one hand out for balance."


def _five(text=GOOD):
    return {e: f"{text[:-1]} ({e})." for e in EMOTIONS}


def _json_reply(captions):
    return json.dumps(captions)


class StubLLM:
    """Returns queued replies in order, recording the prompts it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.images: list[bytes | None] = []
        self.schemas: list[dict | None] = []

    def __call__(self, prompt: str, image_bytes: bytes | None = None,
                 response_schema: dict | None = None) -> str:
        self.prompts.append(prompt)
        self.images.append(image_bytes)
        self.schemas.append(response_schema)
        if not self.replies:
            raise AssertionError("StubLLM ran out of replies")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class Row:
    def __init__(self, image_id, caption_idx, caption):
        self.image_id, self.caption_idx, self.caption = image_id, caption_idx, caption


# ── validation, pinned against real v1 output ───────────────────────────────

# Verbatim references from notebooka1ac23db48.ipynb cell 45. The v1 prompt forbade
# metaphor; Gemini produced these anyway, and v1's 13-regex filter passed them all.
V1_LEAKAGE = [
    "the blue and pink sled a cold reminder against the stark, white snow.",
    "two dogs, collars of blue and brown, dance through the yard, a tender ballet beneath the house.",
    "the red wooden house a silent, distant witness to the running dogs in the yard.",
    "the woman with dark hair holds the clear mug, her presence a quiet poem in the light.",
    "man in sunglasses points at the woman's neck tattoo, a melody in ink on her skin.",
    "the blue and pink sled a soft, sweet echo on the snowy hill this afternoon.",
]

CLEAN = [
    "A child in a pink dress bounds up the entryway stairs, one hand out for balance.",
    "A black dog and a spotted dog lunge at each other in the street, teeth bared.",
    "Two brown dachshunds push through the tall grass, one carrying a red toy.",
    "A girl in a red life jacket climbs the white inflatable wall above the water.",
    "The woman in the white graphic tee holds a clear mug of something brown.",
]


@pytest.mark.parametrize("text", V1_LEAKAGE)
def test_rejects_the_figurative_style_v1_leaked(text):
    reason = validate_caption(text, min_words=8, max_words=30)
    assert reason is not None, f"v1 leakage slipped through: {text!r}"
    assert "figurative" in reason


@pytest.mark.parametrize("text", CLEAN)
def test_accepts_plain_grounded_captions(text):
    assert validate_caption(text, min_words=8, max_words=30) is None


def test_rejects_too_short():
    assert "needs 8-24" in validate_caption("A dog runs fast.")


def test_rejects_too_long():
    assert "needs 8-24" in validate_caption(" ".join(["word"] * 40) + ".")


def test_rejects_emotion_naming_adverb():
    r = validate_caption("A child joyfully climbs the entryway stairs with one hand out today.")
    assert "names the emotion" in r and "joyfully" in r


def test_rejects_simile():
    r = validate_caption("A child climbs the entryway stairs like a small determined machine today.")
    assert "figurative" in r


def test_rejects_two_sentences():
    r = validate_caption("A child climbs the stairs. She wears a pink dress and holds on tight.")
    assert r == "more than one sentence"


def test_accepts_internal_punctuation_that_is_not_a_sentence_break():
    assert validate_caption(
        "A child in a pink dress climbs the entryway stairs, slowly, one hand out."
    ) is None


def test_rejects_empty():
    assert validate_caption("") == "empty"
    assert validate_caption("   ") == "empty"


def test_validate_all_passes_a_clean_set():
    assert validate_all(_five()) == {}


def test_validate_all_flags_identical_rewrites():
    caps = _five()
    caps["sad"] = caps["joyful"]
    out = validate_all(caps)
    assert "sad" in out and "identical to" in out["sad"]


def test_validate_all_flags_a_missing_emotion():
    caps = _five()
    del caps["tense"]
    assert validate_all(caps)["tense"] == "empty"


# ── response parsing ────────────────────────────────────────────────────────


def test_parses_plain_json():
    caps = _five()
    assert parse_response(_json_reply(caps)) == caps


def test_parses_fenced_json():
    caps = _five()
    assert parse_response(f"```json\n{_json_reply(caps)}\n```") == caps


def test_parses_json_wrapped_in_prose():
    caps = _five()
    assert parse_response(f"Sure, here you go:\n{_json_reply(caps)}\nHope that helps!") == caps


def test_recovers_pairs_from_malformed_json():
    got = parse_response('{"joyful": "a bright thing", "sad": "a dim thing",,}')
    assert got == {"joyful": "a bright thing", "sad": "a dim thing"}


def test_parses_empty_and_junk_without_raising():
    assert parse_response("") == {}
    assert parse_response("I cannot help with that.") == {}
    assert parse_response("[1, 2, 3]") == {}


# ── generate_one ────────────────────────────────────────────────────────────


def test_succeeds_on_the_first_attempt():
    llm = StubLLM([_json_reply(_five())])
    caps, attempts, rejects = generate_one(llm, "A child climbs the stairs.")
    assert attempts == 1 and rejects == {} and len(caps) == 5


def test_retries_and_tells_the_model_what_to_fix():
    bad = _five()
    bad["sad"] = "Too short."
    llm = StubLLM([_json_reply(bad), _json_reply(_five())])

    caps, attempts, rejects = generate_one(llm, "A child climbs the stairs.")

    assert attempts == 2 and rejects == {}
    assert "REJECTED" in llm.prompts[1]
    assert "sad" in llm.prompts[1]
    assert "Too short." in llm.prompts[1], "feedback must quote the rejected text"
    assert "REJECTED" not in llm.prompts[0]


def test_keeps_the_best_attempt_not_the_last():
    """A retry must never make the result worse."""
    two_bad = _five()
    two_bad["sad"] = "Short."
    two_bad["tense"] = "Also short."
    three_bad = _five()
    three_bad["sad"] = "Short."
    three_bad["tense"] = "Also short."
    three_bad["joyful"] = "Short too."

    llm = StubLLM([_json_reply(two_bad), _json_reply(three_bad), _json_reply(three_bad)])
    caps, attempts, rejects = generate_one(llm, "A child climbs.", max_attempts=3)

    assert attempts == 3
    assert len(rejects) == 2, f"kept the worse attempt: {rejects}"
    assert caps["joyful"] != "Short too."


def test_survives_a_transport_error_and_retries():
    llm = StubLLM([RuntimeError("503"), _json_reply(_five())])
    seen: list[int] = []
    caps, attempts, rejects = generate_one(
        llm, "A child climbs.", on_error=lambda e, a: seen.append(a)
    )
    assert seen == [1] and attempts == 2 and rejects == {}


def test_gives_up_after_max_attempts():
    bad = _json_reply({e: "Short." for e in EMOTIONS})
    llm = StubLLM([bad, bad])
    caps, attempts, rejects = generate_one(llm, "A child climbs.", max_attempts=2)
    assert attempts == 2 and len(rejects) == 5


# ── the append-only store ───────────────────────────────────────────────────


def _rec(image_id="a.jpg", idx=0, caps=None):
    return GenerationRecord(
        image_id=image_id, caption_idx=idx, source_caption="A child climbs.",
        captions=caps if caps is not None else _five(), model="stub",
    )


def test_store_is_append_only(tmp_path):
    """Earlier bytes must never change. v1 rewrote a 35k-row CSV per image."""
    p = tmp_path / "gen.jsonl"
    append_record(p, _rec("a.jpg", 0))
    first = p.read_bytes()
    append_record(p, _rec("b.jpg", 0))
    assert p.read_bytes().startswith(first)


def test_resume_skips_completed_keys(tmp_path):
    p = tmp_path / "gen.jsonl"
    append_record(p, _rec("a.jpg", 0))
    append_record(p, _rec("a.jpg", 1))
    assert completed_keys(p) == {("a.jpg", 0), ("a.jpg", 1)}


def test_incomplete_records_are_not_treated_as_done(tmp_path):
    """A record missing an emotion must be retried, not silently accepted."""
    p = tmp_path / "gen.jsonl"
    partial = {e: GOOD for e in EMOTIONS if e != "tense"}
    append_record(p, _rec("a.jpg", 0, caps=partial))
    assert completed_keys(p) == set()
    assert completed_keys(p, require_complete=False) == {("a.jpg", 0)}


def test_truncated_final_line_is_tolerated(tmp_path):
    """A 40k-call run will be interrupted mid-write at some point."""
    p = tmp_path / "gen.jsonl"
    append_record(p, _rec("a.jpg", 0))
    with p.open("a", encoding="utf-8") as f:
        f.write('{"image_id": "b.jpg", "caption_id')  # cut off
    assert len(read_records(p)) == 1
    assert completed_keys(p) == {("a.jpg", 0)}


def test_missing_store_reads_as_empty(tmp_path):
    assert read_records(tmp_path / "nope.jsonl") == []
    assert completed_keys(tmp_path / "nope.jsonl") == set()


# ── the runner ──────────────────────────────────────────────────────────────


def test_run_generation_writes_one_record_per_row(tmp_path):
    p = tmp_path / "gen.jsonl"
    rows = [Row("a.jpg", i, f"Caption {i}.") for i in range(3)]
    llm = StubLLM([_json_reply(_five())] * 3)

    stats = run_generation(llm, rows, p, model="stub")

    assert stats["written"] == 3 and stats["attempted"] == 3
    assert completed_keys(p) == {("a.jpg", 0), ("a.jpg", 1), ("a.jpg", 2)}


def test_run_generation_resumes_and_does_not_recall_the_api(tmp_path):
    p = tmp_path / "gen.jsonl"
    rows = [Row("a.jpg", i, f"Caption {i}.") for i in range(3)]

    run_generation(StubLLM([_json_reply(_five())] * 3), rows, p, model="stub")
    # Only one reply queued: a second call would raise "ran out of replies".
    stats = run_generation(StubLLM([_json_reply(_five())]), rows, p, model="stub")

    assert stats["already_done"] == 3
    assert stats["attempted"] == 0


def test_run_generation_honours_limit(tmp_path):
    p = tmp_path / "gen.jsonl"
    rows = [Row("a.jpg", i, f"Caption {i}.") for i in range(10)]
    stats = run_generation(
        StubLLM([_json_reply(_five())] * 2), rows, p, model="stub", limit=2
    )
    assert stats["attempted"] == 2 and stats["written"] == 2


def test_run_generation_records_rejection_reasons(tmp_path):
    """The audit needs to know *why* rows failed, per rule, not just how many."""
    p = tmp_path / "gen.jsonl"
    bad = _five()
    bad["sad"] = "Short."
    llm = StubLLM([_json_reply(bad)] * 3)
    stats = run_generation(
        StubLLM([_json_reply(bad)] * 3), [Row("a.jpg", 0, "c")], p,
        model="stub", max_attempts=3,
    )
    assert stats["incomplete"] == 0
    assert sum(stats["rejection_reasons"].values()) >= 1


# ── multimodal path ─────────────────────────────────────────────────────────


def _tiny_jpeg(tmp_path, name="a.jpg", size=(64, 48), colour=(120, 90, 60)):
    from PIL import Image

    p = tmp_path / name
    Image.new("RGB", size, colour).save(p, format="JPEG")
    return p


def test_load_image_bytes_returns_jpeg(tmp_path):
    from emocap.data.generate import load_image_bytes

    b = load_image_bytes(_tiny_jpeg(tmp_path))
    assert b[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_load_image_bytes_downscales_above_max_dim(tmp_path):
    """Image tokens are the main cost lever, so the cap must actually apply."""
    from io import BytesIO

    from PIL import Image

    from emocap.data.generate import load_image_bytes

    big = _tiny_jpeg(tmp_path, "big.jpg", size=(1600, 1200))
    small = load_image_bytes(big, max_dim=256)
    assert max(Image.open(BytesIO(small)).size) == 256

    unchanged = load_image_bytes(big, max_dim=None)
    assert max(Image.open(BytesIO(unchanged)).size) == 1600


def test_image_is_passed_to_the_llm_and_prompt_switches_to_multimodal(tmp_path):
    from emocap.data.generate import load_image_bytes

    img = load_image_bytes(_tiny_jpeg(tmp_path))
    llm = StubLLM([_json_reply(_five())])
    generate_one(llm, "A child climbs the stairs.", image_bytes=img)

    assert llm.images == [img]
    assert "USE THE IMAGE FOR MOOD" in llm.prompts[0]


def test_text_only_prompt_when_no_image_given():
    llm = StubLLM([_json_reply(_five())])
    generate_one(llm, "A child climbs the stairs.")
    assert llm.images == [None]
    assert "USE THE IMAGE FOR MOOD" not in llm.prompts[0]


def test_runner_loads_images_and_reuses_them_across_source_captions(tmp_path):
    """Five source captions share one image, so it must be encoded once."""
    from emocap.data import generate as gen_mod

    _tiny_jpeg(tmp_path, "a.jpg")
    rows = [Row("a.jpg", i, f"Caption {i}.") for i in range(5)]
    llm = StubLLM([_json_reply(_five())] * 5)

    calls = {"n": 0}
    real = gen_mod.load_image_bytes

    def counting(path, **kw):
        calls["n"] += 1
        return real(path, **kw)

    gen_mod.load_image_bytes = counting
    try:
        stats = run_generation(llm, rows, tmp_path / "gen.jsonl",
                               model="stub", images_dir=tmp_path)
    finally:
        gen_mod.load_image_bytes = real

    assert stats["written"] == 5
    assert calls["n"] == 1, f"image encoded {calls['n']} times, expected 1"
    assert all(b is not None for b in llm.images)


def test_rows_with_a_missing_image_are_skipped_not_silently_text_only(tmp_path):
    """Falling back to a text-only call would put ungrounded rows in the dataset
    with no record that they differ."""
    rows = [Row("absent.jpg", 0, "Caption.")]
    llm = StubLLM([_json_reply(_five())])
    stats = run_generation(llm, rows, tmp_path / "gen.jsonl",
                           model="stub", images_dir=tmp_path)
    assert stats["missing_images"] == 1
    assert stats["written"] == 0
    assert llm.prompts == []


# ── Option B: one call per image, 25 outputs ────────────────────────────────


SOURCES = [
    "A black dog and a spotted dog are fighting in the street",
    "A black dog and a tri-colored dog play with each other on the road",
    "Two dogs of different breeds look at each other on the road",
    "Two dogs on pavement move toward each other slowly",
    "A black dog and a white dog with brown spots stare at each other",
]


def _matrix_reply(n=5, bad_index=None, drop_index=None):
    """A well-formed batch response, optionally with a defect at one index."""
    out = {}
    for i in range(n):
        if i == drop_index:
            continue
        caps = {e: f"{SOURCES[i][:60]} in a {e} register today here now." for e in EMOTIONS}
        if i == bad_index:
            caps["sad"] = "Short."
        out[str(i)] = caps
    return json.dumps(out)


def test_parse_batch_response_reads_the_full_matrix():
    from emocap.data.generate import parse_batch_response

    m = parse_batch_response(_matrix_reply(), 5)
    assert set(m) == set(range(5))
    assert all(set(v) == set(EMOTIONS) for v in m.values())


def test_parse_batch_response_tolerates_a_code_fence():
    from emocap.data.generate import parse_batch_response

    assert len(parse_batch_response(f"```json\n{_matrix_reply()}\n```", 5)) == 5


def test_parse_batch_response_reports_missing_indices_by_omission():
    """A dropped index must be visibly absent, not silently filled -- completion by
    position is the measurement that decides whether batching is safe."""
    from emocap.data.generate import parse_batch_response

    m = parse_batch_response(_matrix_reply(drop_index=3), 5)
    assert 3 not in m
    assert len(m) == 4


def test_parse_batch_response_on_junk():
    from emocap.data.generate import parse_batch_response

    assert parse_batch_response("sorry, I cannot", 5) == {}
    assert parse_batch_response("", 5) == {}


def test_generate_image_batch_returns_matrix_and_per_index_rejections():
    from emocap.data.generate import generate_image_batch

    llm = StubLLM([_matrix_reply(bad_index=2)])
    matrix, attempts, rejects = generate_image_batch(llm, SOURCES, max_attempts=1)
    assert set(matrix) == set(range(5))
    assert attempts == 1
    assert 2 in rejects and "sad" in rejects[2]
    assert 0 not in rejects


def test_generate_image_batch_passes_the_schema_when_enabled():
    from emocap.data.generate import generate_image_batch

    llm = StubLLM([_matrix_reply()])
    generate_image_batch(llm, SOURCES, max_attempts=1, use_schema=True)
    schema = llm.schemas[0]
    assert schema is not None
    # slot_N, not 0..N: Vertex BATCH coerces numeric object keys to integers and then
    # rejects every row. See emocap.data.prompt.SLOT_PREFIX.
    assert schema["required"] == ["slot_0", "slot_1", "slot_2", "slot_3", "slot_4"]
    assert schema["properties"]["slot_0"]["required"] == list(EMOTIONS)


def test_generate_image_batch_omits_the_schema_when_disabled():
    from emocap.data.generate import generate_image_batch

    llm = StubLLM([_matrix_reply()])
    generate_image_batch(llm, SOURCES, max_attempts=1, use_schema=False)
    assert llm.schemas[0] is None


def test_generate_image_batch_uses_the_batch_prompt_with_every_source():
    from emocap.data.generate import generate_image_batch

    llm = StubLLM([_matrix_reply()])
    generate_image_batch(llm, SOURCES, max_attempts=1)
    prompt = llm.prompts[0]
    for s in SOURCES:
        assert s in prompt
    assert "25 captions in total" in prompt


def test_generate_image_batch_keeps_the_best_attempt():
    from emocap.data.generate import generate_image_batch

    worse = _matrix_reply(bad_index=0, drop_index=4)
    better = _matrix_reply(bad_index=0)
    llm = StubLLM([better, worse])
    matrix, attempts, rejects = generate_image_batch(llm, SOURCES, max_attempts=2)
    assert attempts == 2
    assert set(matrix) == set(range(5)), "a retry must never lose an index"


def test_generate_image_batch_survives_a_transport_error():
    from emocap.data.generate import generate_image_batch

    llm = StubLLM([RuntimeError("503"), _matrix_reply()])
    matrix, attempts, rejects = generate_image_batch(llm, SOURCES, max_attempts=2)
    assert attempts == 2 and rejects == {}


def test_batch_prompt_switches_to_text_only_without_an_image():
    from emocap.data.generate import generate_image_batch

    llm = StubLLM([_matrix_reply()])
    generate_image_batch(llm, SOURCES, max_attempts=1)
    assert "USE THE IMAGE FOR MOOD" not in llm.prompts[0]

    llm2 = StubLLM([_matrix_reply()])
    generate_image_batch(llm2, SOURCES, image_bytes=b"\xff\xd8fake", max_attempts=1)
    assert "USE THE IMAGE FOR MOOD" in llm2.prompts[0]


def test_single_call_path_also_sends_a_schema():
    llm = StubLLM([_json_reply(_five())])
    generate_one(llm, "A child climbs the stairs.")
    assert llm.schemas[0] is not None
    assert llm.schemas[0]["required"] == list(EMOTIONS)


# ── the strain flag (added 2026-08-19) ──────────────────────────────────────
#
# The model reports, per register, how well that register fits the scene: 0 natural,
# 1 strained, 2 no honest reading exists. It exists because the prompt cannot make an
# impossible cell possible -- closing one invention route only opened the next -- so a
# declared strain turns a silent invention into a filterable signal.
#
# Both response shapes must parse: a bare string (pre-2026-08-19 records and any model
# that ignores the schema) and {"text": ..., "strain": n}. A schema change must never
# silently discard captions.


def _cell_reply(**caps) -> str:
    import json as _json
    return _json.dumps({e: {"text": t, "strain": s} for e, (t, s) in caps.items()})


def test_parse_response_reads_text_and_strain_from_the_object_shape():
    from emocap.data.generate import parse_response

    raw = _cell_reply(joyful=("a bright thing happens here today", 0),
                      sad=("a dim thing happens here today", 2))
    strain: dict[str, int] = {}
    caps = parse_response(raw, strain=strain)
    assert caps == {"joyful": "a bright thing happens here today",
                    "sad": "a dim thing happens here today"}
    assert strain == {"joyful": 0, "sad": 2}


def test_parse_response_still_reads_bare_strings_and_reports_no_strain():
    from emocap.data.generate import parse_response

    strain: dict[str, int] = {}
    caps = parse_response('{"joyful": "a bright thing", "sad": "a dim thing"}', strain=strain)
    assert caps == {"joyful": "a bright thing", "sad": "a dim thing"}
    assert strain == {}, "a bare string reports no strain; it must not default to 0"


def test_strain_is_absent_not_zero_when_the_model_omits_it():
    from emocap.data.generate import parse_response

    strain: dict[str, int] = {}
    parse_response('{"joyful": {"text": "a bright thing happens today"}}', strain=strain)
    assert strain == {}, "missing strain is unknown, never 0"


def test_out_of_range_strain_is_treated_as_unknown():
    from emocap.data.generate import parse_response

    strain: dict[str, int] = {}
    caps = parse_response('{"joyful": {"text": "a bright thing happens today", "strain": 7}}',
                          strain=strain)
    assert caps["joyful"] == "a bright thing happens today", "the caption survives"
    assert strain == {}, "7 is not a strain level; it must not be clamped to 2"


def test_parse_batch_response_collects_strain_per_caption_index():
    import json as _json

    from emocap.data.generate import parse_batch_response

    raw = _json.dumps({
        "0": {e: {"text": f"caption zero for {e} register here", "strain": 0} for e in EMOTIONS},
        "1": {e: {"text": f"caption one for {e} register here", "strain": 2} for e in EMOTIONS},
    })
    strain: dict[int, dict[str, int]] = {}
    m = parse_batch_response(raw, 2, strain=strain)
    assert len(m) == 2
    assert strain[0]["joyful"] == 0
    assert strain[1]["sad"] == 2


def test_generation_record_round_trips_strain():
    import json as _json

    from emocap.data.generate import GenerationRecord

    rec = GenerationRecord(image_id="a.jpg", caption_idx=0, source_caption="x",
                           captions={"joyful": "y"}, model="m", strain={"joyful": 1})
    assert _json.loads(rec.to_json())["strain"] == {"joyful": 1}


def test_a_record_written_before_the_strain_field_still_loads():
    from emocap.data.generate import read_records

    import json as _json
    from pathlib import Path as _P
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = _P(d) / "old.jsonl"
        p.write_text(_json.dumps({
            "image_id": "a.jpg", "caption_idx": 0, "source_caption": "x",
            "captions": {"joyful": "y"}, "model": "m", "attempts": 1,
            "rejected": {}, "created_at": "2026-08-18T00:00:00",
        }) + "\n")
        recs = read_records(p)
    # read_records yields plain dicts, so an older record simply has no "strain" key.
    # Every consumer must therefore use .get("strain", {}) -- never ["strain"].
    assert len(recs) == 1
    assert "strain" not in recs[0]
    assert recs[0].get("strain", {}) == {}
