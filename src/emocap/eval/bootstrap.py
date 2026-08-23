"""Cluster bootstrap confidence intervals.

Used by the stage-02 probe to say whether a configuration comparison is
*conclusive*, and by stage 10 for the pre-registered analysis. Same machinery in
both places on purpose: the probe should be held to the standard the study is.

**Why clustered.** The five registers of one source caption, and the five source
captions of one image, are not independent observations -- they share a picture and
a generator call. Resampling individual captions would treat 200 correlated
captions as 200 independent ones and report intervals several times too narrow.
Resampling whole images respects the correlation. ``configs/prereg.lock.yaml``
specifies ``cluster_by: image_id`` for the same reason.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

__all__ = [
    "cluster_bootstrap_mean",
    "cluster_bootstrap_paired_diff",
    "ci_width",
    "is_conclusive",
]


def _group(values: Sequence[float], clusters: Sequence) -> list[list[float]]:
    by: dict[object, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        by[c].append(float(v))
    return list(by.values())


def cluster_bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence,
    *,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
    return_samples: bool = False,
) -> dict:
    """Mean of ``values`` with a cluster-bootstrap CI.

    ``clusters[i]`` is the cluster (e.g. image id) that ``values[i]`` belongs to.
    Returns mean, low, high, the number of clusters, and the interval width.
    """
    groups = _group(values, clusters)
    if not groups:
        return {"mean": None, "lo": None, "hi": None, "n_clusters": 0, "width": None}

    flat = [v for g in groups for v in g]
    point = sum(flat) / len(flat)
    if len(groups) < 2:
        return {"mean": round(point, 4), "lo": None, "hi": None,
                "n_clusters": len(groups), "width": None}

    rng = random.Random(seed)
    k = len(groups)
    means: list[float] = []
    for _ in range(n_resamples):
        picked = [groups[rng.randrange(k)] for _ in range(k)]
        vals = [v for g in picked for v in g]
        if vals:
            means.append(sum(vals) / len(vals))
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo = means[int(alpha * (len(means) - 1))]
    hi = means[int((1 - alpha) * (len(means) - 1))]
    out = {
        "mean": round(point, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "n_clusters": k,
        "width": round(hi - lo, 4),
    }
    # The resample distribution, for callers that need a bootstrap p-value rather than an
    # interval. Exposed rather than recomputed elsewhere so every p-value in the study comes
    # from the registered estimator -- same clustering, same seed, same resample count.
    if return_samples:
        out["samples"] = means
    return out


def cluster_bootstrap_paired_diff(
    left: Sequence[float],
    right: Sequence[float],
    clusters: Sequence,
    *,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """CI for ``mean(left) - mean(right)`` on paired observations.

    Pairs must be aligned: ``left[i]`` and ``right[i]`` are the same unit measured
    under two configurations. Resampling clusters keeps the pairing intact, which
    is what makes the interval narrow enough to be useful on a small probe.

    ``crosses_zero`` is the decision-relevant field: if the interval contains zero,
    the probe has not distinguished the two configurations.
    """
    if len(left) != len(right) or len(left) != len(clusters):
        raise ValueError("left, right and clusters must be the same length")
    diffs = [float(a) - float(b) for a, b in zip(left, right)]
    out = cluster_bootstrap_mean(diffs, clusters, n_resamples=n_resamples, ci=ci, seed=seed)
    out["diff"] = out.pop("mean")
    out["crosses_zero"] = (
        None if out["lo"] is None else bool(out["lo"] <= 0.0 <= out["hi"])
    )
    return out


def ci_width(result: dict) -> float | None:
    return result.get("width")


def is_conclusive(result: dict, *, threshold: float) -> bool | None:
    """Whether a paired difference is both non-zero and larger than ``threshold``.

    ``threshold`` is the smallest difference worth acting on. A statistically
    non-zero difference that is smaller than the threshold is not a reason to pay
    more for a configuration.
    """
    if result.get("lo") is None:
        return None
    if result.get("crosses_zero"):
        return False
    return min(abs(result["lo"]), abs(result["hi"])) >= threshold
