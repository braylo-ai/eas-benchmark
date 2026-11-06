"""Small statistics helpers: Wilson intervals and bootstrap AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95):
    """Point estimate and 95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return dict(p=float("nan"), lo=float("nan"), hi=float("nan"), k=0, n=0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(
        p=p, lo=max(0.0, centre - half), hi=min(1.0, centre + half), k=int(k), n=int(n)
    )


def auc_ci(y, score, rng, n_boot=200, max_n=100_000):
    """AUC with a 95% percentile bootstrap CI (bootstrap on a subsample of ≤ max_n for speed)."""
    y = np.asarray(y)
    score = np.asarray(score)
    auc = roc_auc_score(y, score)
    idx_all = rng.choice(len(y), size=min(max_n, len(y)), replace=False)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if y[idx].min() == y[idx].max():
            continue
        boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(auc=float(auc), lo=float(lo), hi=float(hi))
