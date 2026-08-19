"""Minimum detectable effect for the arm comparisons.

The study's central claim (P2) is a **null**: that the margin over the keyword anchor
differs by less than 5 points between human and synthetic training data. A null's force
depends entirely on what the design could have detected — a 1.2-point observed difference
means something very different against a +/-0.5-point interval than against +/-3. Stating
the MDE in advance turns a bounded null into a result and an unbounded one into a shrug.

**Two variance components, and only one is measurable before training.**

1. *Evaluation noise* — how precisely one arm's accuracy can be measured on a finite test
   set whose cells are clustered by image. Measurable now, and computed here from the
   intra-cluster correlation of an actual classifier's per-cell correctness.

2. *Seed noise* — how much the same arm moves across training seeds. Unknowable until
   models exist. So the MDE is reported as a function of seed SD, and the registration
   commits to the rule already in the pre-registration: a gap smaller than the observed
   seed spread is reported as null regardless of its p-value.

The clustering matters more than it looks. The five register cells of one image share a
source caption and an image, so they are not independent observations. Treating 5,000
cells as 5,000 independent draws understates the standard error by the square root of the
design effect — the same error that made the naive bootstrap five times too narrow
earlier in this project.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

__all__ = ["icc_from_clusters", "design_effect", "mde_two_arms", "mde_table"]

#: Two-sided alpha and the z multipliers for common power levels.
_Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


def icc_from_clusters(correct: Sequence[float], clusters: Sequence) -> dict:
    """Intra-cluster correlation of per-cell correctness, by one-way ANOVA.

    ``correct`` is 1.0/0.0 per evaluated cell; ``clusters`` the image id each belongs to.
    Returns the ICC plus the pieces, so a suspiciously high or low value can be traced
    rather than trusted.
    """
    by: dict[object, list[float]] = defaultdict(list)
    for c, k in zip(correct, clusters):
        by[k].append(float(c))
    groups = [v for v in by.values() if v]
    n_total = sum(len(v) for v in groups)
    k = len(groups)
    if k < 2 or n_total <= k:
        raise ValueError("need at least two non-empty clusters")

    grand = sum(sum(v) for v in groups) / n_total
    # mean cluster size, harmonic-ish correction for unequal sizes
    m = n_total / k
    ss_between = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in groups)
    ss_within = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in groups)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)
    denom = ms_between + (m - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0.0
    return {
        "icc": round(max(0.0, icc), 4),
        "clusters": k,
        "cells": n_total,
        "mean_cluster_size": round(m, 2),
        "ms_between": round(ms_between, 6),
        "ms_within": round(ms_within, 6),
        "mean_correct": round(grand, 4),
    }


def design_effect(icc: float, cluster_size: float) -> float:
    """DEFF = 1 + (m - 1) * ICC. Effective n is cells / DEFF."""
    return 1.0 + (cluster_size - 1.0) * max(0.0, icc)


def mde_two_arms(
    p: float,
    n_cells_per_arm: int,
    *,
    icc: float,
    cluster_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
    n_comparisons: int = 1,
    seed_sd: float = 0.0,
    n_seeds: int = 3,
) -> dict:
    """Smallest difference in accuracy two arms could reliably show.

    ``n_comparisons`` applies a Bonferroni split of alpha, matching the
    Holm-Bonferroni correction the pre-registration commits to (Holm is uniformly at
    least as powerful, so this is conservative).

    ``seed_sd`` is the between-seed SD of one arm's accuracy. Averaging ``n_seeds`` runs
    reduces its contribution by sqrt(n_seeds); it is added in quadrature with the
    evaluation variance because the two are independent sources.
    """
    deff = design_effect(icc, cluster_size)
    n_eff = n_cells_per_arm / deff
    var_eval = p * (1.0 - p) / n_eff                 # per arm
    var_seed = (seed_sd ** 2) / max(1, n_seeds)      # per arm
    se_diff = math.sqrt(2.0 * (var_eval + var_seed))  # two independent arms

    alpha_adj = alpha / max(1, n_comparisons)
    z_alpha = _z_two_sided(alpha_adj)
    z_power = _Z.get(power)
    if z_power is None:
        raise ValueError(f"power must be one of {sorted(_Z)}")
    return {
        "mde": round((z_alpha + z_power) * se_diff, 4),
        "se_of_difference": round(se_diff, 5),
        "design_effect": round(deff, 3),
        "effective_n_per_arm": round(n_eff, 1),
        "cells_per_arm": n_cells_per_arm,
        "icc": icc,
        "power": power,
        "alpha_per_comparison": round(alpha_adj, 5),
        "n_comparisons": n_comparisons,
        "seed_sd": seed_sd,
        "n_seeds": n_seeds,
        "eval_share_of_variance": round(var_eval / (var_eval + var_seed), 3)
        if (var_eval + var_seed) > 0 else None,
    }


def _z_two_sided(alpha: float) -> float:
    """Inverse normal CDF at 1 - alpha/2, via a rational approximation."""
    return _ppf(1.0 - alpha / 2.0)


def _ppf(q: float) -> float:
    """Acklam's inverse normal CDF. Accurate to ~1e-9, ample here."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
               ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    if q > phigh:
        r = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
                ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    r = q - 0.5
    s = r * r
    return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / \
           (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)


def mde_table(
    p: float,
    n_cells_per_arm: int,
    *,
    icc: float,
    cluster_size: float,
    seed_sds: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03),
    n_comparisons: int = 3,
    power: float = 0.80,
    n_seeds: int = 3,
) -> list[dict]:
    """MDE across plausible seed SDs, since seed noise is unknown pre-training."""
    return [
        mde_two_arms(p, n_cells_per_arm, icc=icc, cluster_size=cluster_size,
                     power=power, n_comparisons=n_comparisons,
                     seed_sd=sd, n_seeds=n_seeds)
        for sd in seed_sds
    ]
