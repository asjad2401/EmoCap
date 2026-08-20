"""Tests for the corpus-level vocabulary cap."""

from __future__ import annotations

import pytest

from emocap.data.generate import validate_all
from emocap.data.prompt import EMOTIONS, build_batch_prompt, build_prompt
from emocap.data.vocab_cap import (
    DEFAULT_CAPS,
    banned_hit,
    content_words,
    over_cap_words,
    cap_report,
    register_frequencies,
    stem,
)


def _records(n: int, *, joyful_word: str = "bright", share: float = 1.0) -> list[dict]:
    """`n` records where `joyful_word` appears in `share` of the joyful captions."""
    out = []
    for i in range(n):
        caps = {e: f"the {e} sentence number {i} about a field" for e in EMOTIONS}
        if i < int(n * share):
            caps["joyful"] = f"a {joyful_word} field with number {i} in it"
        out.append({"image_id": f"img{i}.jpg", "caption_idx": 0, "captions": caps})
    return out


def test_content_words_are_a_set_so_repeats_count_once():
    # Document frequency, not term frequency: one verbose caption must not look like a
    # corpus-wide pattern.
    assert content_words("bright bright bright field") == {"bright", "field"}


def test_inflections_share_a_stem():
    # The model rebuilt `romantic` on `gentle`/`softly` after `soft` and `gently` were
    # capped. Measuring exact forms while banning exact forms made that free.
    assert stem("softly") == stem("soft") == stem("softer")
    assert stem("gently") == stem("gentle")       # English -le / -ly pair
    assert stem("emptied") == stem("empty")       # y -> i before a suffix
    assert stem("brightest") == stem("bright")


def test_stem_keeps_unrelated_words_apart():
    for a, b in (("close", "closet"), ("bright", "brighton"), ("high", "highway"),
                 ("soft", "software"), ("gently", "gentleman")):
        assert stem(a) != stem(b), (a, b)


def test_banned_hit_catches_inflections_and_nothing_else():
    B = ["empty", "gently", "soft", "bright", "high"]
    for text in ("the field is emptied", "a gentle hand", "softly played",
                 "the brightest day", "higher still"):
        assert banned_hit(text, B) is not None, text
    for text in ("Brighton pier", "a highway sign", "software update", "closet door",
                 "a gentleman waits", "nothing at all here"):
        assert banned_hit(text, B) is None, text


def test_banned_hit_is_empty_safe():
    assert banned_hit("anything at all", []) is None


def test_stopwords_and_short_words_dropped():
    got = content_words("a man and the dog on it")
    assert got == set(), got


def test_register_frequencies_counts_per_register():
    freq, totals = register_frequencies(_records(10))
    assert totals["joyful"] == 10
    assert freq["joyful"][stem("bright")] == 10
    assert freq["sad"][stem("bright")] == 0


def test_word_over_cap_in_one_register_is_banned_there_only():
    banned = over_cap_words(_records(60), min_support=40)
    assert "bright" in banned["joyful"]
    for other in EMOTIONS:
        if other != "joyful":
            assert "bright" not in banned[other]


def test_below_min_support_bans_nothing():
    # 20 captions per register: a single unlucky draw would read as a 5% pattern.
    banned = over_cap_words(_records(20), min_support=40)
    assert all(v == [] for v in banned.values()), banned


def test_register_neutral_word_is_never_banned_however_common():
    # "field" is in every caption of every register -- exactly the scene vocabulary the
    # captions are supposed to be about. High frequency, zero lift, must survive.
    banned = over_cap_words(_records(60), min_support=40)
    assert all("field" not in v for v in banned.values()), banned


def test_lift_threshold_protects_shared_vocabulary():
    recs = []
    for i in range(60):
        # "warm" in 100% of romantic AND 100% of joyful -> lift 1.0, not register-specific
        caps = {e: f"a warm scene number {i} here" for e in EMOTIONS}
        recs.append({"image_id": f"i{i}.jpg", "caption_idx": 0, "captions": caps})
    banned = over_cap_words(recs, min_support=40)
    assert all("warm" not in v for v in banned.values()), banned


def test_caps_are_per_register_and_sad_is_the_most_permissive():
    # The gap between classifier and human accuracy is smallest for `sad`, so its
    # vocabulary is doing real work and must be capped most gently. If this ordering is
    # ever changed, the reasoning in vocab_cap's docstring has to change with it.
    assert DEFAULT_CAPS["sad"] > DEFAULT_CAPS["joyful"]
    assert DEFAULT_CAPS["joyful"] == min(DEFAULT_CAPS.values())


def test_per_register_cap_is_actually_applied():
    # A word at ~12% in-register: over joyful's 0.08 cap, under sad's 0.15.
    for reg, expect_banned in (("joyful", True), ("sad", False)):
        recs = []
        for i in range(100):
            caps = {e: f"plain sentence number {i} about a field" for e in EMOTIONS}
            if i < 12:
                caps[reg] = f"a zzq sentence number {i} about a field"
            recs.append({"image_id": f"i{i}.jpg", "caption_idx": 0, "captions": caps})
        banned = over_cap_words(recs, min_support=40)
        assert ("zzq" in banned[reg]) is expect_banned, (reg, banned[reg])


def test_max_per_register_bounds_the_list():
    # An unbounded ban list is how v5 happened -- deny every signal and the registers
    # become interchangeable.
    recs = []
    for i in range(60):
        caps = {e: f"plain number {i} field" for e in EMOTIONS}
        # purely alphabetic: the tokenizer strips digits, so "wq0" would become "wq"
        # and fall under the 3-character floor.
        caps["joyful"] = " ".join("wq" + chr(ord("a") + j) for j in range(20)) + f" num {i}"
        recs.append({"image_id": f"i{i}.jpg", "caption_idx": 0, "captions": caps})
    banned = over_cap_words(recs, min_support=40, max_per_register=12)
    assert len(banned["joyful"]) == 12, banned["joyful"]


def test_missing_registers_do_not_crash():
    recs = [{"image_id": "a.jpg", "caption_idx": 0, "captions": {"joyful": "one two three"}}]
    assert over_cap_words(recs, min_support=1)["sad"] == []


def test_cap_report_records_the_numbers_the_ban_rests_on():
    rows = cap_report(_records(60), min_support=40)
    row = next(r for r in rows if r["word"] == "bright")
    assert row["register"] == "joyful"
    assert row["share"] == pytest.approx(1.0)
    assert row["elsewhere"] == 0.0
    assert row["lift"] is None          # infinite lift serialises as null
    assert row["cap"] == DEFAULT_CAPS["joyful"]


# ── prompt and validator wiring ─────────────────────────────────────────────


def test_prompt_omits_the_block_entirely_when_nothing_is_capped():
    assert "ALREADY USED" not in build_prompt("a dog runs")
    assert "ALREADY USED" not in build_prompt("a dog runs", banned_by_register={})
    # A register present but empty must not emit a dangling line either.
    assert "ALREADY USED" not in build_prompt("a dog runs",
                                              banned_by_register={"joyful": []})


def test_prompt_states_the_banned_words_for_the_right_register():
    p = build_prompt("a dog runs", banned_by_register={"joyful": ["bright", "wide"]})
    block = p.split("ALREADY USED")[1].split("REGISTERS")[0]
    assert "bright" in block and "wide" in block
    assert "sad" not in block


def test_batch_prompt_forwards_the_ban_list():
    p = build_batch_prompt(["a dog runs"] * 5, banned_by_register={"sad": ["empty"]})
    assert "ALREADY USED" in p and "empty" in p


def test_validator_rejects_a_banned_word_for_its_own_register_only():
    caps = {e: f"a plain {e} sentence about the field and nothing else here"
            for e in EMOTIONS}
    caps["joyful"] = "a bright sentence about the field and nothing else at all"
    caps["humorous"] = "a bright sentence about the field and nothing much else here"
    bad = validate_all(caps, banned_by_register={"joyful": ["bright"]})
    assert "joyful" in bad and "bright" in bad["joyful"]
    assert "humorous" not in bad


def test_validator_catches_an_inflection_of_a_banned_word():
    # "brightly" IS caught: it is the same habit with a different ending, which is how
    # `romantic` evaded the first version of this cap.
    caps = {e: f"a plain {e} sentence about the field and nothing else here"
            for e in EMOTIONS}
    caps["joyful"] = "a brightly lit sentence about the field and nothing else here"
    bad = validate_all(caps, banned_by_register={"joyful": ["bright"]})
    assert bad.get("joyful", "").endswith('"brightly"'), bad


def test_validator_does_not_fire_on_a_word_that_merely_starts_the_same():
    caps = {e: f"a plain {e} sentence about the field and nothing else here"
            for e in EMOTIONS}
    caps["joyful"] = "a software update sits on the desk beside the field and the road"
    assert validate_all(caps, banned_by_register={"joyful": ["soft"]}) == {}


def test_validator_unchanged_when_no_cap_passed():
    caps = {e: f"a bright plain {e} sentence about the field and nothing else"
            for e in EMOTIONS}
    assert validate_all(caps) == {}


def test_deterministic_validation_still_takes_priority():
    # A caption that is too short should report its length, not a banned word: the
    # rejection reason is fed back to the model and must name the primary defect.
    caps = {e: f"a plain {e} sentence about the field and nothing else here"
            for e in EMOTIONS}
    caps["joyful"] = "bright field"
    bad = validate_all(caps, banned_by_register={"joyful": ["bright"]})
    assert "words" in bad["joyful"]


def test_config_caps_match_the_code_defaults():
    """`configs/data.yaml` and `DEFAULT_CAPS` must not drift apart.

    The caps are a pre-registered corpus-construction rule, so two copies of them is
    already a risk; this makes a silent divergence a test failure rather than a corpus
    built under caps nobody wrote down.
    """
    from emocap.runtime import load_config

    cfg = load_config("data")["vocab_cap"]
    assert cfg["caps"] == DEFAULT_CAPS
    assert cfg["min_lift"] == 2.0
    assert cfg["max_per_register"] == 12
    assert cfg["min_support"] == 40
