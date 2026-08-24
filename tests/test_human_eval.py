"""The human evaluation is registered as CONFIRMATORY, so its statistic gets tested.

Krippendorff's alpha decides whether the per-arm means underneath it mean anything at all:
if the three raters did not agree, those means are three private impressions averaged
together. A quietly wrong alpha would not look wrong -- it would land somewhere plausible
and nobody would catch it.

The expected values below were produced by the reference ``krippendorff`` PyPI package and
then hard-coded, so the check survives without adding a dependency the study does not
otherwise need. Every value here matched it to five decimal places.
"""

from __future__ import annotations

import random

import pytest

from emocap.eval.human_eval import krippendorff_alpha

SCALE = [1, 2, 3, 4, 5]


def test_perfect_agreement_is_one():
    units = [[3, 3, 3], [1, 1, 1], [5, 5, 5], [2, 2, 2], [4, 4, 4]] * 4
    assert krippendorff_alpha(units, SCALE) == pytest.approx(1.0)


def test_matches_the_reference_implementation_on_the_canonical_example():
    """Krippendorff's 3-observer, 12-unit example, with its missing values.

    Units rated by only one observer carry no agreement information and are dropped -- the
    last two entries exist to check that they are.
    """
    units = [[1, 1], [2, 2, 3], [3, 3, 3], [3, 3, 3], [2, 2, 2], [1, 2, 3],
             [4, 4, 4], [1, 1, 2], [2, 2, 2], [5, 5], [1], [3]]
    assert krippendorff_alpha(units, SCALE) == pytest.approx(0.8049, abs=5e-4)


def test_independent_raters_land_near_zero():
    """Alpha is chance-corrected, so unrelated ratings must not score as agreement."""
    rng = random.Random(0)
    units = [[rng.randint(1, 5) for _ in range(3)] for _ in range(60)]
    assert krippendorff_alpha(units, SCALE) == pytest.approx(-0.0593, abs=5e-4)


def test_raters_who_agree_within_one_point_score_high_but_not_perfect():
    rng = random.Random(0)
    units = []
    for _ in range(60):
        v = rng.randint(1, 5)
        units.append([v, min(5, v + 1), max(1, v - 1)])
    assert krippendorff_alpha(units, SCALE) == pytest.approx(0.6489, abs=5e-4)


def test_ordinal_beats_nominal_on_near_miss_disagreement():
    """A 3-vs-4 disagreement must count as smaller than a 1-vs-5 one.

    This is the whole reason the ordinal difference function is used. Under a nominal
    metric both are simply "different" and rating data comes out looking far less reliable
    than it is.
    """
    near = [[3, 4] for _ in range(20)] + [[1, 1], [5, 5], [2, 2], [1, 2]]
    far = [[1, 5] for _ in range(20)] + [[1, 1], [5, 5], [2, 2], [1, 2]]
    assert krippendorff_alpha(near, SCALE) > krippendorff_alpha(far, SCALE)


def test_too_little_data_returns_none_rather_than_a_number():
    assert krippendorff_alpha([[1, 2]], SCALE) is None
    assert krippendorff_alpha([[1], [2], [3]], SCALE) is None


def test_no_variance_returns_none_not_one():
    """Every rater gave every unit the same value: agreement is undefined, not perfect.

    Expected disagreement is zero, so alpha is 1 - 0/0. Returning 1.0 would report a
    degenerate task as maximally reliable.
    """
    assert krippendorff_alpha([[3, 3, 3]] * 10, SCALE) is None
