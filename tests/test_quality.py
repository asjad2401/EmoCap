"""Tests for the caption quality metrics.

These decide the stage-02 generation configuration and gate stage 03, so a metric
that silently misbehaves would propagate into every downstream number.
"""

from __future__ import annotations

from emocap.data.quality import (
    config_divergence,
    content_recall,
    content_words,
    grounding_report,
    length_stats,
    novel_content,
    pairwise_similarity,
    register_divergence,
)

SRC = "A child in a pink dress is climbing up a set of stairs in an entry way."


def test_content_words_drops_stopwords_and_short_tokens():
    got = content_words("A child in the pink dress is on a set of stairs")
    assert "child" in got and "pink" in got and "stairs" in got
    assert "the" not in got and "is" not in got and "of" not in got


def test_content_words_is_case_insensitive():
    assert content_words("Child Stairs") == content_words("child stairs")


# ── grounding ───────────────────────────────────────────────────────────────


def test_content_recall_is_one_for_a_verbatim_copy():
    assert content_recall(SRC, SRC) == 1.0


def test_content_recall_falls_when_content_is_dropped():
    assert content_recall(SRC, "A child climbs.") < 0.5


def test_content_recall_of_empty_source_is_zero_not_an_error():
    assert content_recall("the of and", "anything") == 0.0


def test_novel_content_finds_only_words_absent_from_the_source():
    novel = novel_content(SRC, "A child in a pink dress bounds up the stairs, sunlight bright")
    assert "sunlight" in novel and "bounds" in novel
    assert "child" not in novel and "pink" not in novel


def test_grounding_report_shape():
    r = grounding_report(SRC, "A child in a pink dress bounds up the entryway stairs")
    assert 0.0 <= r["content_recall"] <= 1.0
    assert 0.0 <= r["novel_rate"] <= 1.0
    assert isinstance(r["novel_words"], list)
    assert r["n_content_words"] > 0


# ── register divergence ─────────────────────────────────────────────────────


def test_identical_registers_score_maximum_similarity():
    caps = {e: "the same caption every time here" for e in
            ("joyful", "sad", "tense", "romantic", "humorous")}
    d = register_divergence(caps)
    assert d["mean_similarity"] == 1.0
    assert d["max_similarity"] == 1.0


def test_distinct_registers_score_low_similarity():
    caps = {
        "joyful": "children race across bright green grass",
        "sad": "an empty bench faces grey water",
        "tense": "knuckles whiten on a steel railing",
        "romantic": "two hands meet beneath warm lamplight",
        "humorous": "a pigeon has claimed the entire pavement",
    }
    assert register_divergence(caps)["mean_similarity"] < 0.15


def test_register_divergence_names_the_most_similar_pair():
    caps = {
        "joyful": "children race across bright grass",
        "sad": "children race across bright grass",
        "tense": "knuckles whiten on steel railing",
    }
    d = register_divergence(caps)
    assert set(d["most_similar_pair"]) == {"joyful", "sad"}
    assert d["max_similarity"] == 1.0


def test_register_divergence_handles_too_few_captions():
    d = register_divergence({"joyful": "only one"})
    assert d["n"] == 1 and d["mean_similarity"] is None


# ── length ──────────────────────────────────────────────────────────────────


def test_length_stats_reports_in_target_fraction():
    texts = ["one two three four five six seven eight",      # 8, in
             "one two three",                                 # 3, out
             " ".join(["w"] * 30)]                            # 30, out
    st = length_stats(texts, lo=8, hi=24)
    assert st["n"] == 3
    assert st["in_target"] == round(1 / 3, 4)
    assert st["min"] == 3 and st["max"] == 30


def test_length_stats_on_empty_input():
    assert length_stats([])["n"] == 0


# ── configuration divergence ────────────────────────────────────────────────


def test_identical_configs_report_full_agreement():
    a = ["a child climbs the stairs", "two dogs run"]
    d = config_divergence(a, list(a))
    assert d["identical_rate"] == 1.0
    assert d["mean_similarity"] == 1.0


def test_different_configs_report_disagreement():
    a = ["a child climbs the entryway stairs"]
    b = ["knuckles whiten on a steel railing"]
    d = config_divergence(a, b)
    assert d["identical_rate"] == 0.0
    assert d["mean_similarity"] < 0.2


def test_config_divergence_ignores_pure_rewording_in_similarity():
    """Content similarity should stay high when only word order changes -- that is
    the point of comparing content words rather than strings."""
    d = config_divergence(["a child climbs the stairs"], ["the stairs a child climbs"])
    assert d["identical_rate"] == 0.0
    assert d["mean_similarity"] == 1.0


def test_config_divergence_skips_missing_pairs():
    d = config_divergence(["a", "", "two dogs run"], ["a", "b", "two dogs run"])
    assert d["n"] == 2


def test_config_divergence_on_empty_input():
    assert config_divergence([], [])["n"] == 0


def test_pairwise_similarity_edge_cases():
    assert pairwise_similarity("", "") == 1.0
    assert pairwise_similarity("the of and", "the of and") == 1.0   # no content words
    assert pairwise_similarity("dogs", "cats") == 0.0
