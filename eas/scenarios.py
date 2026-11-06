"""Labeled scenario generators for the Monte Carlo study.

IMPORTANT: these distributions are *assumptions*, not measurements. They describe hypothetical
sub-detector outputs (drift, hallucination, misalignment, anchor distance, argument lengths) for
six scenario classes. Results computed from them describe how the scoring engine *aggregates*
such inputs. They say nothing about how well any real detector works. That is why every
headline number is also reported across a sweep of detector quality (`sep`) and noise.

Each sub-score is logit-normal:  score = sigmoid(mu_class + sep·delta_class + noise·z).
"""

from __future__ import annotations

import numpy as np

from .engine import PHI

CLASSES = [
    "benign",
    "harmful",
    "legal_risk",
    "resource_risk",
    "self_modification",
    "financial",
]
UNSAFE = CLASSES[1:]

BASE_LOGIT = {"drift": -2.5, "halluc": -2.5, "misalign": -2.5, "anchor": -2.0}
# Per-class logit shift per unit of detector separation (assumed profiles).
DELTA = {
    "benign": (0.0, 0.0, 0.0, 0.0),
    "harmful": (1.0, 0.5, 1.0, 1.0),
    "legal_risk": (0.5, 0.5, 1.0, 1.0),
    "resource_risk": (0.5, 0.3, 0.5, 0.5),
    "self_modification": (1.5, 1.5, 1.0, 1.5),
    "financial": (0.8, 1.0, 0.8, 0.8),
}
# Disproportionate-escalation multiplier for the argument-length ratio (unsafe classes).
ESCALATION = {
    "benign": 0.0,
    "harmful": 0.5,
    "legal_risk": 0.3,
    "resource_risk": 0.3,
    "self_modification": 0.8,
    "financial": 0.5,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_per_class: int,
    rng: np.random.Generator,
    sep: float = 1.0,
    noise: float = 1.0,
    covert_frac: float = 0.0,
    arg_model: str = "phi_hypothesis",
):
    """Return dict of arrays for n_per_class trials of every class.

    covert_frac: share of unsafe trials whose surface sub-scores are drawn from the *benign*
        profile -- the deck's own "mismatch problem" (polite surface, unsafe intent).
    arg_model:
        "phi_hypothesis" -- generous to the source: benign successive-argument ratios centre on
            φ; unsafe ratios escalate above φ.
        "steady" -- benign arguments keep roughly constant length (ratio ≈ 1), a common real
            pattern; unsafe ratios escalate above 1.
    """
    rows = []
    for ci, c in enumerate(CLASSES):
        n = n_per_class
        d = np.array(DELTA[c])
        covert = np.zeros(n, bool)
        if c != "benign" and covert_frac > 0:
            covert = rng.random(n) < covert_frac
        shift = np.where(covert[:, None], 0.0, sep * d[None, :])
        base = np.array(
            [BASE_LOGIT[k] for k in ("drift", "halluc", "misalign", "anchor")]
        )
        z = rng.standard_normal((n, 4))
        scores = _sigmoid(base[None, :] + shift + noise * z)
        centre = PHI if arg_model == "phi_hypothesis" else 1.0
        esc = np.where(covert, 0.0, sep * ESCALATION[c])
        log_ratio = np.log(centre) + esc + 0.25 * noise * rng.standard_normal(n)
        arg_prev = rng.poisson(40, n) + 1
        arg_n = np.maximum(1, np.round(arg_prev * np.exp(log_ratio)))
        rows.append(
            dict(
                cls=np.full(n, ci),
                unsafe=np.full(n, c != "benign"),
                covert=covert,
                drift=scores[:, 0],
                halluc=scores[:, 1],
                misalign=scores[:, 2],
                anchor=scores[:, 3],
                arg_prev=arg_prev,
                arg_n=arg_n,
            )
        )
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    # action_id / state strings for the hash gate: unique per trial, random state nonce.
    nonces = rng.integers(0, 2**63, len(out["cls"]))
    out["action_id"] = np.array(
        [f"{CLASSES[c]}-{i}" for i, c in enumerate(out["cls"])], dtype=object
    )
    out["state"] = np.array([f"{x:016x}" for x in nonces], dtype=object)
    return out
