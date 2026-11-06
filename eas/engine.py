"""PPL / RVN AI v1 verification engine, implemented exactly as written in the source deck
and dissertation draft (B. M. Lopez, 2026). Every formula here is a transcription; where the
source is ambiguous, the choice we made is documented inline and in README "Interpretation notes".

Nothing in this file claims anything about real models. It is a scoring engine.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

PHI = (1 + math.sqrt(5)) / 2  # golden ratio, 1.6180339...

# Deck slide 3 / dissertation §3.2
T_WEIGHTS = {"alpha": 0.45, "beta": 0.35, "gamma": 0.20}
# Dissertation §4.1 / deck slide 3
V_WEIGHTS = {"w1": 0.30, "w2": 0.20, "w3": 0.20, "w4": 0.30}
V_THRESHOLD = 0.65
# Dissertation §5.2 constraint map
PRIMES = {101: "harm", 103: "legal", 107: "resources", 109: "override"}
LAMBDA_PER_DAY = 0.05


# --------------------------------------------------------------------------------------
# A_ij(t): PPL alignment function (deck slide 3, dissertation §3.1)
# --------------------------------------------------------------------------------------
def snn_placeholder(P_ij, dL_ij, SC_ij, C_t):
    """SNN(P_ij, ΔL_ij, SC_ij, C_t) is a *neural network* in the source with no definition.
    We substitute a documented, deterministic placeholder (mean of the four inputs, clipped to
    [0,1]) so A_ij(t) is computable. Any result depending on SNN depends on this choice.
    """
    return np.clip(
        (
            np.asarray(P_ij)
            + (1 - np.asarray(dL_ij))
            + np.asarray(SC_ij)
            + np.asarray(C_t)
        )
        / 4,
        0,
        1,
    )


def alignment_A(
    history,
    t,
    P_ij,
    F,
    snn,
    alpha=0.45,
    beta=0.35,
    gamma=0.20,
    current_trust=1.0,
    lam=LAMBDA_PER_DAY,
):
    """A_ij(t) = [ Σ_k (w_k·T_k/deg(k))·e^{-λ(t − t_last)} + α·P_ij + β·F + γ·SNN ] / max(1, current_trust)

    history: iterable of (w_k, T_k, deg_k).  t_last: time of last interaction (we read
    "t_last" literally as one global last-interaction time, as written; see README).
    Note: A_ij is NOT bounded to [0,1] -- the Σ_k term grows with history length.
    """
    w, T, deg, t_last = history["w"], history["T"], history["deg"], history["t_last"]
    s = np.sum(np.asarray(w) * np.asarray(T) / np.asarray(deg), axis=-1) * np.exp(
        -lam * (t - t_last)
    )
    return (s + alpha * P_ij + beta * F + gamma * snn) / np.maximum(1.0, current_trust)


# --------------------------------------------------------------------------------------
# T, S, N, V
# --------------------------------------------------------------------------------------
def t_score(drift, halluc, misalign, alpha=0.45, beta=0.35, gamma=0.20):
    """T = 1 − (α·T_drift + β·T_halluc + γ·T_misalign)"""
    return 1.0 - (alpha * drift + beta * halluc + gamma * misalign)


def fib_S(arg_n, arg_prev):
    """S = | arg_n / arg_{n−1} − φ |   (unbounded above; 1−S can go negative)"""
    return np.abs(np.asarray(arg_n, float) / np.asarray(arg_prev, float) - PHI)


def debate_rounds(S):
    """N = min(5, ⌈5·S⌉)   (S = 0 gives N = 0 rounds)"""
    return np.minimum(5, np.ceil(5 * np.asarray(S))).astype(int)


def verdict_V(T, S, A, prime_clear, w1=0.30, w2=0.20, w3=0.20, w4=0.30):
    """V = w1·T + w2·(1−S) + w3·(1−A) + w4·prime_clear.  A = anchor distance (diss. Layer 5)."""
    return w1 * T + w2 * (1 - S) + w3 * (1 - A) + w4 * prime_clear


def verdict_pass(V, prime_clear, threshold=V_THRESHOLD):
    """pass iff V ≥ 0.65 AND prime_clear = 1"""
    return (np.asarray(V) >= threshold) & (np.asarray(prime_clear) == 1)


# --------------------------------------------------------------------------------------
# Prime-invariant hard gate (dissertation §5.2)
# --------------------------------------------------------------------------------------
def violation_residue(p: int) -> int:
    """Source says `signature mod p == violation_residue(p)` but never defines
    violation_residue. We fix it deterministically (sha256 of the constraint name, mod p).
    The gate's statistics do not depend on which residue is chosen (see README)."""
    return int(hashlib.sha256(PRIMES[p].encode()).hexdigest(), 16) % p


def signature(action_id: str, state: str) -> int:
    """K = hash(action_id ‖ state).  We use SHA-256 as the hash."""
    return int(hashlib.sha256(f"{action_id}||{state}".encode()).hexdigest(), 16)


def prime_gate(action_id: str, state: str, primes=tuple(PRIMES)):
    """Returns (prime_clear, vetoing_prime or None)."""
    K = signature(action_id, state)
    for p in primes:
        if K % p == violation_residue(p):
            return 0, p
    return 1, None


def prime_gate_from_K(K: np.ndarray, primes=tuple(PRIMES)):
    """Vectorised gate for precomputed signatures (object array of Python ints, or uint64)."""
    clear = np.ones(len(K), dtype=int)
    for p in primes:
        r = violation_residue(p)
        hit = np.array([k % p == r for k in K]) if K.dtype == object else (K % p == r)
        clear[hit] = 0
    return clear
