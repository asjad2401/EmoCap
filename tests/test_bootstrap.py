"""Tests for the cluster bootstrap.

This decides whether a probe comparison is called conclusive, and stage 10 uses it
for the pre-registered analysis. An interval that is too narrow would make us act on
noise -- which is the specific failure it exists to prevent.
"""

from __future__ import annotations

import random

import pytest

from emocap.eval import (
    cluster_bootstrap_mean,
    cluster_bootstrap_paired_diff,
    is_conclusive,
)


def test_mean_is_the_point_estimate():
    vals = [1.0, 2.0, 3.0, 4.0]
    r = cluster_bootstrap_mean(vals, ["a", "a", "b", "b"], n_resamples=500)
    assert r["mean"] == 2.5


def test_interval_brackets_the_mean():
    random.seed(0)
    vals = [random.gauss(5, 1) for _ in range(100)]
    cl = [i // 10 for i in range(100)]
    r = cluster_bootstrap_mean(vals, cl, n_resamples=2000)
    assert r["lo"] <= r["mean"] <= r["hi"]
    assert r["n_clusters"] == 10


def test_clustering_widens_the_interval_when_data_is_correlated():
    """The whole point: 200 correlated captions are not 200 independent ones.

    Resampling individual captions would report an interval several times too
    narrow, and we would act on noise.
    """
    random.seed(1)
    vals, cl = [], []
    for img in range(8):
        base = random.gauss(0.5, 0.25)
        for _ in range(25):
            vals.append(base + random.gauss(0, 0.02))
            cl.append(img)

    clustered = cluster_bootstrap_mean(vals, cl, n_resamples=2000)
    naive = cluster_bootstrap_mean(vals, list(range(len(vals))), n_resamples=2000)
    assert clustered["width"] > 2 * naive["width"]


def test_more_clusters_narrows_the_interval():
    """Precision must improve with more images -- this is what justifies raising n."""
    random.seed(2)

    def width(n_images):
        vals, cl = [], []
        for img in range(n_images):
            base = random.gauss(0.5, 0.25)
            for _ in range(25):
                vals.append(base + random.gauss(0, 0.02))
                cl.append(img)
        return cluster_bootstrap_mean(vals, cl, n_resamples=2000)["width"]

    assert width(40) < width(8)


def test_single_cluster_reports_no_interval_rather_than_a_fake_one():
    r = cluster_bootstrap_mean([1.0, 2.0], ["a", "a"], n_resamples=100)
    assert r["n_clusters"] == 1
    assert r["lo"] is None and r["hi"] is None


def test_empty_input():
    r = cluster_bootstrap_mean([], [], n_resamples=100)
    assert r["mean"] is None and r["n_clusters"] == 0


def test_bootstrap_is_deterministic_given_a_seed():
    vals = [float(i) for i in range(40)]
    cl = [i // 4 for i in range(40)]
    a = cluster_bootstrap_mean(vals, cl, n_resamples=500, seed=7)
    b = cluster_bootstrap_mean(vals, cl, n_resamples=500, seed=7)
    assert a == b


# ── paired differences ──────────────────────────────────────────────────────


def test_paired_diff_detects_a_real_shift():
    random.seed(3)
    cl = [i // 5 for i in range(100)]
    left = [random.gauss(0.9, 0.05) for _ in range(100)]
    right = [x - 0.30 for x in left]
    r = cluster_bootstrap_paired_diff(left, right, cl, n_resamples=2000)
    assert r["diff"] == pytest.approx(0.30, abs=0.01)
    assert r["crosses_zero"] is False


def test_paired_diff_reports_no_difference_when_configs_agree():
    """The case that matters for the probe: if two resolutions produce the same
    output, the interval must contain zero so we take the cheaper one."""
    random.seed(4)
    cl = [i // 5 for i in range(100)]
    left = [random.gauss(0.5, 0.1) for _ in range(100)]
    right = list(left)
    r = cluster_bootstrap_paired_diff(left, right, cl, n_resamples=2000)
    assert r["diff"] == 0.0
    assert r["crosses_zero"] is True


def test_paired_diff_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        cluster_bootstrap_paired_diff([1.0, 2.0], [1.0], ["a", "b"])


# ── conclusiveness ──────────────────────────────────────────────────────────


def test_a_difference_smaller_than_the_threshold_is_not_conclusive():
    """A statistically non-zero difference that is too small to matter is not a
    reason to pay more for a configuration."""
    r = {"lo": 0.01, "hi": 0.03, "crosses_zero": False}
    assert is_conclusive(r, threshold=0.10) is False
    assert is_conclusive(r, threshold=0.005) is True


def test_an_interval_crossing_zero_is_not_conclusive():
    assert is_conclusive({"lo": -0.05, "hi": 0.20, "crosses_zero": True},
                         threshold=0.01) is False


def test_missing_interval_returns_none_not_a_verdict():
    assert is_conclusive({"lo": None, "hi": None}, threshold=0.1) is None
