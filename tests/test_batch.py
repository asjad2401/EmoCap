"""Tests for the Batch API path, against a stubbed client.

The batch path writes the dataset everything downstream depends on, so the failure
modes that must be loud are: a job that ends in a non-SUCCEEDED state, and a
response count that does not match the request count (which would silently attach
captions to the wrong image).
"""

from __future__ import annotations

import json
import types as pytypes

import pytest

from emocap.data.batch import BatchChunk, run_batch_generation, submit_and_wait
from emocap.data.generate import completed_keys, read_records
from emocap.data.prompt import EMOTIONS

SRCS = {
    "a.jpg": [f"A dog runs across the field number {i}" for i in range(5)],
    "b.jpg": [f"A child climbs the wooden stairs number {i}" for i in range(5)],
}


def _matrix_json(n=5, bad_idx=None, drop_idx=None):
    out = {}
    for i in range(n):
        if i == drop_idx:
            continue
        caps = {e: f"A dog runs across the open field in a {e} way today" for e in EMOTIONS}
        if i == bad_idx:
            caps["sad"] = "Short."
        out[str(i)] = caps
    return json.dumps(out)


def _resp(text, prompt=2200, out=672, think=0):
    um = pytypes.SimpleNamespace(prompt_token_count=prompt, candidates_token_count=out,
                                thoughts_token_count=think)
    return pytypes.SimpleNamespace(response=pytypes.SimpleNamespace(text=text, usage_metadata=um))


class StubBatches:
    def __init__(self, states, responses, error=None):
        self.states = list(states)
        self.responses = responses
        self.error = error
        self.created = []

    def create(self, *, model, src, config=None):
        self.created.append({"model": model, "n": len(src),
                             "display_name": getattr(config, "display_name", None)})
        return pytypes.SimpleNamespace(name="batches/stub", state="JobState.JOB_STATE_PENDING")

    def get(self, *, name):
        state = self.states.pop(0) if self.states else "JobState.JOB_STATE_SUCCEEDED"
        dest = pytypes.SimpleNamespace(inlined_responses=self.responses)
        return pytypes.SimpleNamespace(name=name, state=state, dest=dest, error=self.error)


class StubClient:
    def __init__(self, batches):
        self.batches = batches


@pytest.fixture
def imgdir(tmp_path):
    from PIL import Image
    for name in SRCS:
        Image.new("RGB", (64, 48), (100, 120, 140)).save(tmp_path / name, format="JPEG")
    return tmp_path


# ── request building ────────────────────────────────────────────────────────


def test_one_request_per_image_not_per_caption(imgdir):
    from emocap.data.batch import build_requests

    reqs, kept = build_requests(
        BatchChunk(list(SRCS), SRCS), images_dir=imgdir, image_max_dim=384,
        min_words=8, max_words=24, emphasise_distinctness=True,
        temperature=0.9, max_output_tokens=4096, thinking_budget=0,
    )
    assert len(reqs) == 2 and kept == ["a.jpg", "b.jpg"], "image must be sent once per image"


def test_requests_carry_image_prompt_and_schema(imgdir):
    from emocap.data.batch import build_requests

    reqs, _ = build_requests(
        BatchChunk(["a.jpg"], SRCS), images_dir=imgdir, image_max_dim=384,
        min_words=8, max_words=24, emphasise_distinctness=True,
        temperature=0.9, max_output_tokens=4096, thinking_budget=0,
    )
    parts = reqs[0].contents[0].parts
    assert parts[0].inline_data is not None, "image part missing"
    assert "MOST COMMON FAILURE" in parts[1].text, "revised prompt not used"
    assert reqs[0].config.response_schema is not None


def test_missing_image_is_skipped_not_sent_text_only(imgdir):
    """A text-only request would produce an ungrounded row with no record that it
    differs from the rest of the dataset."""
    from emocap.data.batch import build_requests

    srcs = dict(SRCS, **{"missing.jpg": ["x"] * 5})
    reqs, kept = build_requests(
        BatchChunk(list(srcs), srcs), images_dir=imgdir, image_max_dim=384,
        min_words=8, max_words=24, emphasise_distinctness=True,
        temperature=0.9, max_output_tokens=4096, thinking_budget=0,
    )
    assert "missing.jpg" not in kept and len(reqs) == 2


# ── job lifecycle ───────────────────────────────────────────────────────────


def test_waits_through_running_then_succeeds():
    c = StubClient(StubBatches(
        ["JobState.JOB_STATE_RUNNING", "JobState.JOB_STATE_SUCCEEDED"], [_resp(_matrix_json())]))
    job, wall = submit_and_wait(c, "m", ["r"], display_name="t", poll_seconds=0)
    assert wall >= 0


def test_failed_job_raises_rather_than_returning_nothing():
    c = StubClient(StubBatches(["JobState.JOB_STATE_FAILED"], [], error="quota"))
    with pytest.raises(RuntimeError, match="ended.*FAILED"):
        submit_and_wait(c, "m", ["r"], display_name="t", poll_seconds=0)


def test_cancelled_job_raises():
    c = StubClient(StubBatches(["JobState.JOB_STATE_CANCELLED"], []))
    with pytest.raises(RuntimeError, match="CANCELLED"):
        submit_and_wait(c, "m", ["r"], display_name="t", poll_seconds=0)


def test_timeout_raises():
    c = StubClient(StubBatches(["JobState.JOB_STATE_RUNNING"] * 50, []))
    with pytest.raises(TimeoutError, match="still"):
        submit_and_wait(c, "m", ["r"], display_name="t", poll_seconds=0, timeout_seconds=-1)


# ── end to end ──────────────────────────────────────────────────────────────


def _run(tmp_path, imgdir, responses, **kw):
    c = StubClient(StubBatches(["JobState.JOB_STATE_SUCCEEDED"], responses))
    return c, run_batch_generation(
        c, "gemini-3.1-flash-lite", list(SRCS), SRCS, tmp_path / "gen.jsonl",
        images_dir=imgdir, chunk_size=250, **kw)


def test_writes_one_record_per_caption_index(tmp_path, imgdir):
    _, st = _run(tmp_path, imgdir, [_resp(_matrix_json()), _resp(_matrix_json())])
    assert st["images_written"] == 2
    assert st["captions_written"] == 50           # 2 images x 5 captions x 5 registers
    assert st["captions_expected"] == 50
    assert completed_keys(tmp_path / "gen.jsonl") == {
        (i, idx) for i in SRCS for idx in range(5)
    }


def test_records_carry_source_caption_and_model(tmp_path, imgdir):
    _run(tmp_path, imgdir, [_resp(_matrix_json()), _resp(_matrix_json())])
    recs = read_records(tmp_path / "gen.jsonl")
    r = next(x for x in recs if x["image_id"] == "a.jpg" and x["caption_idx"] == 3)
    assert r["source_caption"] == SRCS["a.jpg"][3]
    assert r["model"] == "gemini-3.1-flash-lite"


def test_response_count_mismatch_raises(tmp_path, imgdir):
    """Zipping N responses onto M requests would attach captions to the wrong image."""
    with pytest.raises(RuntimeError, match="refusing to guess"):
        _run(tmp_path, imgdir, [_resp(_matrix_json())])   # 1 response, 2 requests


def test_rejections_are_counted_by_rule(tmp_path, imgdir):
    _, st = _run(tmp_path, imgdir,
                 [_resp(_matrix_json(bad_idx=2)), _resp(_matrix_json())])
    assert st["rejections"] >= 1
    assert sum(st["rejection_reasons"].values()) == st["rejections"]


def test_dropped_caption_index_lowers_completion(tmp_path, imgdir):
    _, st = _run(tmp_path, imgdir,
                 [_resp(_matrix_json(drop_idx=4)), _resp(_matrix_json())])
    assert st["captions_written"] == 45
    assert ("a.jpg", 4) not in completed_keys(tmp_path / "gen.jsonl")


def test_usage_is_accumulated(tmp_path, imgdir):
    _, st = _run(tmp_path, imgdir,
                 [_resp(_matrix_json(), prompt=2200, out=672),
                  _resp(_matrix_json(), prompt=2100, out=650)])
    assert st["usage"]["prompt"] == 4300
    assert st["usage"]["output"] == 1322


def test_resume_skips_completed_images(tmp_path, imgdir):
    _run(tmp_path, imgdir, [_resp(_matrix_json()), _resp(_matrix_json())])
    # Second run: no responses queued at all. If it tried to call, it would fail.
    c = StubClient(StubBatches(["JobState.JOB_STATE_SUCCEEDED"], []))
    st = run_batch_generation(c, "m", list(SRCS), SRCS, tmp_path / "gen.jsonl",
                              images_dir=imgdir, chunk_size=250)
    assert st["images_pending"] == 0
    assert c.batches.created == []


def test_limit_images_caps_work(tmp_path, imgdir):
    _, st = _run(tmp_path, imgdir, [_resp(_matrix_json())], limit_images=1)
    assert st["images_pending"] == 1 and st["images_written"] == 1


def test_chunking_creates_multiple_jobs(tmp_path, imgdir):
    c = StubClient(StubBatches(["JobState.JOB_STATE_SUCCEEDED"] * 10,
                               [_resp(_matrix_json())]))
    st = run_batch_generation(c, "m", list(SRCS), SRCS, tmp_path / "gen.jsonl",
                              images_dir=imgdir, chunk_size=1)
    assert st["jobs"] == 2
    assert [x["n"] for x in c.batches.created] == [1, 1]
