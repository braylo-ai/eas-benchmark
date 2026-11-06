"""Unit tests: formulas match the source exactly; generators are deterministic."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eas import engine as E  # noqa: E402
from eas.scenarios import generate  # noqa: E402


def test_t_score():
    assert math.isclose(
        E.t_score(0.1, 0.2, 0.3), 1 - (0.45 * 0.1 + 0.35 * 0.2 + 0.20 * 0.3)
    )


def test_fib_and_rounds():
    assert math.isclose(float(E.fib_S(161.8034, 100)), 0.0, abs_tol=1e-5)
    assert int(E.debate_rounds(0.0)) == 0
    assert int(E.debate_rounds(0.01)) == 1
    assert int(E.debate_rounds(0.62)) == 4
    assert int(E.debate_rounds(3.0)) == 5


def test_verdict():
    V = E.verdict_V(1.0, 0.0, 0.0, 1)
    assert math.isclose(float(V), 1.0)
    assert bool(E.verdict_pass(0.65, 1)) and not bool(E.verdict_pass(0.99, 0))
    assert math.isclose(float(E.verdict_V(0.96, 0.04, 0.06, 1)), 0.968)


def test_alignment_A():
    h = dict(w=[1.0, 1.0], T=[0.5, 1.0], deg=[1, 2], t_last=0.0)
    A = E.alignment_A(h, t=0.0, P_ij=1.0, F=1.0, snn=1.0)
    assert math.isclose(float(A), 0.5 + 0.5 + 0.45 + 0.35 + 0.20)
    A2 = E.alignment_A(h, t=10.0, P_ij=0, F=0, snn=0, current_trust=2.0)
    assert math.isclose(float(A2), 1.0 * math.exp(-0.5) / 2)


def test_gate_rate():
    clear = [E.prime_gate("a", str(i))[0] for i in range(40_000)]
    rate = 1 - np.mean(clear)
    theo = 1 - math.prod(1 - 1 / p for p in E.PRIMES)
    assert abs(rate - theo) < 0.006


def test_generator_deterministic():
    a = generate(500, np.random.default_rng(1))
    b = generate(500, np.random.default_rng(1))
    for k in ("drift", "halluc", "misalign", "anchor", "arg_n"):
        assert np.array_equal(a[k], b[k])
    assert list(a["state"]) == list(b["state"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
