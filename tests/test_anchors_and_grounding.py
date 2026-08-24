"""The lexical-shortcut anchor and the grounding-defect detectors.

Both produce numbers that are quoted in configs/prereg.lock.yaml and docs/. They were
ad-hoc scripts while the prompt was being iterated, which made the headline figures
unreproducible from the repo; these tests pin the properties that make them meaningful.
"""

from __future__ import annotations

import re

import pytest

from emocap.data.grounding import caption_defects, defect_counts
from emocap.data.prompt import EMOTIONS
from emocap.eval.anchors import (ESTIMATOR_ID, keyword_rule_accuracy,
                                 keyword_rule_cell_scores, top_keywords)


def _rec(image_id: str, caps: dict, source: str = "a person stands in a field") -> dict:
    return {"image_id": image_id, "caption_idx": 0, "source_caption": source,
            "captions": caps}


def _uninformative(n_images: int = 20) -> list[dict]:
    """Every register gets the same wording, so keywords cannot separate them."""
    return [
        _rec(f"img{i}.jpg", {e: f"a person stands quietly in the open field here {i}"
                             for e in EMOTIONS})
        for i in range(n_images)
    ]


def _perfectly_marked(n_images: int = 20) -> list[dict]:
    """Each register carries a unique marker word, the degenerate keyword case."""
    marker = {"joyful": "zzjoy", "sad": "zzsad", "tense": "zztense",
              "romantic": "zzrom", "humorous": "zzfun"}
    return [
        _rec(f"img{i}.jpg", {e: f"a person stands in the open field {marker[e]} here"
                             for e in EMOTIONS})
        for i in range(n_images)
    ]


def test_estimator_id_matches_the_preregistration_lock():
    from emocap.runtime import load_config

    lock = load_config("prereg.lock")
    assert lock["anchors"]["lexical_shortcut_estimator"] == ESTIMATOR_ID, (
        "the lock names an estimator that this module does not implement -- the anchor "
        "would be unverifiable"
    )


def test_lock_stores_a_recompute_rule_not_a_bare_anchor_value():
    """Regression guard for the 2026-08-19 retractions.

    The lock used to store `lexical_shortcut_value: 0.394` as though the anchor were a
    single number. It is not: it falls from 0.440 at 4,486 cells to 0.332 at 201,900,
    because a keyword rule fitted on few images transfers well inside that narrow pool and
    degrades as the pool widens. Three findings were retracted for comparing an anchor at
    one n against accuracy at another. The lock now stores a RULE, and a bare value must
    never come back.
    """
    from emocap.runtime import load_config

    anchors = load_config("prereg.lock")["anchors"]
    assert "lexical_shortcut_value" not in anchors, (
        "a bare anchor value is back in the lock; it is meaningless without its n"
    )
    assert anchors["lexical_shortcut_rule"] == (
        "recompute_on_each_evaluation_set_at_its_own_n"
    )
    assert anchors["lexical_shortcut_estimator"] == ESTIMATOR_ID


def test_every_reference_anchor_is_labelled_with_its_sample_size():
    """A reference figure without its n is the exact error that caused the retractions."""
    from emocap.runtime import load_config

    refs = load_config("prereg.lock")["anchors"]["lexical_shortcut_reference"]
    assert refs, "reference values should be recorded for context"
    for key, value in refs.items():
        assert re.search(r"\d", key), f"reference {key!r} does not state its sample size"
        assert 0.2 <= value <= 1.0, f"{key} = {value} is not a plausible accuracy"


def test_the_anchor_falls_as_the_sample_grows():
    """The property that invalidated three claims, pinned so it cannot be forgotten."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    store = root / "data/generated/captions_raw.jsonl"
    if not store.exists():
        pytest.skip("corpus not present")
    records = [json.loads(l) for l in store.read_text().splitlines() if l.strip()]
    if sum(len(r.get("captions") or {}) for r in records) < 60000:
        pytest.skip("corpus too small to show the effect")

    from emocap.eval.anchors import keyword_rule_matched

    small = keyword_rule_matched(records, n_cells=4000, n_subsamples=2)
    large = keyword_rule_matched(records, n_cells=50000, n_subsamples=2)
    assert large["mean"] < small["mean"] - 0.02, (
        f"anchor did not fall with n ({small['mean']} -> {large['mean']}); the estimator's "
        f"sample-size dependence is the basis of a reported finding"
    )


def test_marker_words_drive_the_anchor_far_above_chance():
    result = keyword_rule_accuracy(_perfectly_marked())
    assert result["accuracy"] > 0.9, result
    assert result["chance"] == pytest.approx(0.2)


def test_identical_registers_score_at_chance_not_above():
    """With nothing to discriminate, every register ties -- fractional credit gives 1/5."""
    result = keyword_rule_accuracy(_uninformative())
    assert result["accuracy"] == pytest.approx(0.2, abs=0.02), result


def test_ties_are_credited_fractionally_not_in_full():
    """A rule that discriminates nothing must not be flattered to 1.0."""
    assert keyword_rule_accuracy(_uninformative())["accuracy"] < 0.5


def test_top_keywords_is_deterministic():
    """Non-deterministic selection is what made the first recorded figures un-repeatable."""
    recs = _perfectly_marked()
    assert top_keywords(recs) == top_keywords(recs)


def test_anchor_folds_by_image_never_by_row():
    """Splitting by row would leak near-duplicate text across the fold boundary."""
    with pytest.raises(ValueError, match="folds"):
        keyword_rule_accuracy(_perfectly_marked(n_images=3), folds=5)


# ── the per-cell decomposition ──────────────────────────────────────────────
#
# The study claims the MARGIN, and every registered magnitude criterion is written in
# margin points -- so the margin needs a confidence interval, which needs the anchor as a
# per-cell quantity. `keyword_rule_cell_scores` provides that from the same estimator.
# If it ever drifts from `keyword_rule_accuracy`, every margin interval in the paper is
# surrounding a number computed a different way, which is worse than having no interval.


def test_cell_scores_cover_every_cell_exactly_once():
    recs = _perfectly_marked()
    scores = keyword_rule_cell_scores(recs)
    assert set(scores) == {(r["image_id"], e) for r in recs for e in r["captions"]}


def test_cell_scores_pool_to_the_point_estimate():
    """The two weightings must agree when the folds are equal in size.

    20 images over 5 folds is exactly 4 images each, so the unweighted fold mean and the
    pooled cell mean are the same quantity. This is the check that would catch a drift
    between the two code paths.
    """
    for recs in (_perfectly_marked(), _uninformative()):
        scores = keyword_rule_cell_scores(recs)
        pooled = sum(scores.values()) / len(scores)
        assert pooled == pytest.approx(
            keyword_rule_accuracy(recs)["accuracy"], abs=5e-4)


def test_cell_scores_are_bounded_credit():
    """A score is a mean of fractional tie credits, so it lives in [0, 1].

    With nothing to discriminate, all five registers tie on every cell and each earns
    exactly 1/5 -- the same fractional-credit rule the point estimate uses, visible per
    cell rather than averaged away.
    """
    assert all(0.0 <= v <= 1.0
               for v in keyword_rule_cell_scores(_perfectly_marked()).values())
    assert all(v == pytest.approx(0.2)
               for v in keyword_rule_cell_scores(_uninformative()).values())


def test_cell_scores_separate_the_two_degenerate_corpora():
    marked = keyword_rule_cell_scores(_perfectly_marked())
    flat = keyword_rule_cell_scores(_uninformative())
    assert sum(marked.values()) / len(marked) > 0.9
    assert sum(flat.values()) / len(flat) == pytest.approx(0.2, abs=0.02)


def test_cell_scores_are_deterministic():
    recs = _perfectly_marked()
    assert keyword_rule_cell_scores(recs) == keyword_rule_cell_scores(recs)


# ── grounding defects ───────────────────────────────────────────────────────


def test_solitude_flagged_only_against_a_multi_subject_scene():
    assert caption_defects("a single tent sits alone on the ice",
                           source_caption="a tent is being set up on the ice",
                           factual_caption="Two people are standing near a blue tent.") \
        == ["solitude"]
    assert caption_defects("a single tent sits alone on the ice",
                           source_caption="a tent is being set up on the ice",
                           factual_caption="A blue tent sits on a frozen lake.") == []


def test_several_non_person_objects_do_not_count_as_multiple_subjects():
    """The bug this scoping fixes: "several small bowls" read as several people."""
    assert caption_defects("a girl sits alone in the grass",
                           source_caption="a girl sits in the grass",
                           factual_caption="The girl sits on grass. There are several "
                                           "small bowls of paint around her.") == []


def test_pace_flagged_when_a_running_subject_is_slowed():
    assert caption_defects("the dog moves slowly across the sand",
                           source_caption="a dog on the beach",
                           factual_caption="A dog runs on a sandy beach.") == ["pace"]


def test_pace_not_flagged_when_the_motion_is_kept():
    assert caption_defects("the dog runs the length of the wet sand",
                           source_caption="a dog on the beach",
                           factual_caption="A dog runs on a sandy beach.") == []


def test_light_and_contact_are_judged_against_the_source_caption_alone():
    got = caption_defects("a couple lean into each other as the light fades",
                          source_caption="several people sitting on a ledge")
    assert set(got) == {"contact", "light"}


def test_a_detail_already_in_the_source_caption_is_not_a_defect():
    assert caption_defects("two people embrace in the warm sunlight",
                           source_caption="two people embrace in the sunlight") == []


def test_image_dependent_classes_are_skipped_without_a_factual_caption():
    got = caption_defects("one lone figure moves slowly", source_caption="a figure")
    assert "solitude" not in got and "pace" not in got


def test_defect_counts_reports_rate_and_class_breakdown():
    recs = [_rec("a.jpg", {"sad": "a couple lean into each other in the fading light",
                           "joyful": "a person stands in a field"})]
    out = defect_counts(recs)
    assert out["cells"] == 2
    assert out["by_class"]["contact"] == 1 and out["by_class"]["light"] == 1
    assert out["by_register"]["sad"] == 2 and "joyful" not in out["by_register"]
    assert out["rate"] == pytest.approx(1.0)


# ── the register classifier, as an instrument ───────────────────────────────


def test_folds_never_split_one_image_across_the_boundary():
    """The property the ceiling number depends on: no near-duplicate leakage.

    The 25 cells of one image are rewrites of five near-identical source captions. If a
    fold puts some on each side, the classifier sees the test text during training and
    every accuracy here is inflated.
    """
    from emocap.eval.register_classifier import folds_by_image

    images = [f"img{i // 5}.jpg" for i in range(50)]
    for train, test in folds_by_image(images, folds=5, seed=1):
        train_images = {images[i] for i in train}
        test_images = {images[i] for i in test}
        assert not (train_images & test_images)


def test_every_image_is_held_out_exactly_once():
    from emocap.eval.register_classifier import folds_by_image

    images = [f"img{i}.jpg" for i in range(20)]
    held = [img for _, test in folds_by_image(images, folds=5, seed=1)
            for img in {images[i] for i in test}]
    assert sorted(held) == sorted(set(images))


def test_build_dataset_uses_the_locked_emotion_order():
    from emocap.eval.register_classifier import build_dataset

    recs = [{"image_id": "a.jpg", "source_caption": "x",
             "captions": {e: f"caption for {e}" for e in EMOTIONS}}]
    texts, labels, images = build_dataset(recs)
    assert [EMOTIONS[l] for l in labels] == [e for e in EMOTIONS if e in recs[0]["captions"]]
    assert images == ["a.jpg"] * len(texts)


def test_build_dataset_skips_blank_and_unknown_registers():
    from emocap.eval.register_classifier import build_dataset

    recs = [{"image_id": "a.jpg", "source_caption": "x",
             "captions": {"joyful": "a real caption", "sad": "   ", "bogus": "ignore me"}}]
    texts, labels, _ = build_dataset(recs)
    assert texts == ["a real caption"] and len(labels) == 1


def test_strip_artifacts_removes_punctuation_and_case():
    from emocap.eval.register_classifier import strip_artifacts

    assert strip_artifacts("A Dog runs -- fast!") == "a dog runs fast"


def test_confusion_matrix_rows_sum_to_one_and_report_recall():
    from emocap.eval.register_classifier import confusion_matrix

    pairs = [(0, 0), (0, 0), (0, 1), (1, 1), (1, 0)]
    cm = confusion_matrix(pairs)
    assert cm["recall"]["joyful"] == pytest.approx(2 / 3, abs=0.01)
    assert sum(cm["row_normalised"]["joyful"].values()) == pytest.approx(1.0, abs=0.01)
