"""Reliability for the registered human evaluation.

`configs/prereg.lock.yaml` registers `reliability: krippendorff_alpha` for a 150-item,
3-rater, blinded evaluation on two 5-point scales. A statistic named in a lock needs an
implementation in the repository for the same reason the lexical anchor does: a number with
no estimator behind it cannot be checked or recomputed.

**Ordinal, not nominal.** On a 1-5 rating scale a 3-vs-4 disagreement is smaller than a
1-vs-5 disagreement. Nominal alpha treats them as identical, which understates agreement
badly on rating data and is the usual way this statistic is misreported.

Verified against the reference ``krippendorff`` PyPI package to five decimal places on four
cases -- perfect agreement, Krippendorff's own 12-unit example with missing values,
independent raters, and raters agreeing within one point. Those expected values are pinned
in `tests/test_human_eval.py` so the check survives without the extra dependency.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["krippendorff_alpha"]


def krippendorff_alpha(
    units: Sequence[Sequence[int]], values: Sequence[int]
) -> float | None:
    """Krippendorff's alpha with the ordinal difference function.

    ``units`` is one list of ratings per unit -- however many raters happened to rate it,
    so raters who skipped items need no placeholder. Units with fewer than two ratings
    carry no information about agreement and are dropped, which is the standard treatment
    rather than a convenience.

    ``values`` is the ordered scale, e.g. ``[1, 2, 3, 4, 5]``. Order matters: the ordinal
    difference between ranks ``c`` and ``k`` is

        ( sum of n_g for g between c and k inclusive, minus (n_c + n_k) / 2 ) ** 2

    where ``n_g`` is the marginal frequency of value ``g`` in the coincidence matrix. It
    depends on the observed marginals, so it is computed after that matrix rather than
    from the scale alone.

    Returns ``None`` -- never a number -- when alpha is undefined: fewer than two usable
    units, or no variance at all. A degenerate task reported as ``1.0`` would read as
    maximal reliability.
    """
    usable = [list(u) for u in units if len(u) >= 2]
    if len(usable) < 2:
        return None

    idx = {v: i for i, v in enumerate(values)}
    m = len(values)
    o = [[0.0] * m for _ in range(m)]
    for u in usable:
        mu = len(u)
        for a in range(mu):
            for b in range(mu):
                if a != b:
                    o[idx[u[a]]][idx[u[b]]] += 1.0 / (mu - 1)

    n_c = [sum(row) for row in o]
    n = sum(n_c)
    if n < 2:
        return None

    def delta2(c: int, k: int) -> float:
        lo, hi = (c, k) if c <= k else (k, c)
        s = sum(n_c[lo:hi + 1]) - (n_c[c] + n_c[k]) / 2.0
        return s * s

    d2 = [[delta2(c, k) for k in range(m)] for c in range(m)]
    do = sum(o[c][k] * d2[c][k] for c in range(m) for k in range(m)) / n
    de = sum(n_c[c] * n_c[k] * d2[c][k]
             for c in range(m) for k in range(m)) / (n * (n - 1))
    if de == 0:
        return None
    return 1.0 - do / de
