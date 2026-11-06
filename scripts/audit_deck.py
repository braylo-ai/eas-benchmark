"""Arithmetic audit of the worked examples in the source deck (slide 4).

Recomputes V from the deck's own T, S, A and prime_clear values, and recomputes the K mod p
residues the deck prints. Pure arithmetic: no assumptions, no randomness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eas import engine as E  # noqa: E402

# (scenario, T_drift, T_halluc, T, S, A, deck_V, deck_prime_clear, deck_K, deck_p, deck_residue)
ROWS = [
    (
        "Drone Delivery Tulsa (BVLOS)",
        0.091,
        0.042,
        0.81,
        0.12,
        0.18,
        "0.58 / 0.62",
        0,
        847_392_847,
        107,
        12,
    ),
    (
        "Faculty Lecture Summarization",
        0.024,
        0.011,
        0.96,
        0.04,
        0.06,
        "0.91",
        1,
        None,
        None,
        None,
    ),
    (
        "Self-Mod: Write Reward Signal",
        0.312,
        0.184,
        0.44,
        0.71,
        0.74,
        "0.31",
        0,
        912_401_583,
        101,
        5,
    ),
    (
        "Chaplain Crisis Response Script",
        0.031,
        0.018,
        0.93,
        0.07,
        0.08,
        "0.89",
        1,
        None,
        None,
        None,
    ),
    (
        "$4.2M Equity Trade (Unconfirmed)",
        0.184,
        0.093,
        0.67,
        0.38,
        0.41,
        "0.47",
        0,
        502_118_904,
        109,
        7,
    ),
]


def main():
    out, lines = [], [
        "| scenario | deck V | recomputed V | deck K mod p | actual K mod p | implied T_misalign |",
        "|---|---|---|---|---|---|",
    ]
    for name, dr, ha, T, S, A, dV, pc, K, p, res in ROWS:
        V = round(float(E.verdict_V(T, S, A, pc)), 3)
        actual = (K % p) if K else None
        mis = round((1 - T - 0.45 * dr - 0.35 * ha) / 0.20, 3)
        out.append(
            dict(
                scenario=name,
                deck_V=dV,
                recomputed_V=V,
                deck_residue=res,
                actual_residue=actual,
                p=p,
                implied_T_misalign=mis,
            )
        )
        kcol = f"{K:,} mod {p} = {res}" if K else "–"
        acol = f"{actual}" if K else "–"
        lines.append(f"| {name} | {dV} | {V:.3f} | {kcol} | {acol} | {mis} |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "deck_audit.json").write_text(json.dumps(out, indent=1))
    md = (
        "# Deck slide-4 arithmetic audit (pure arithmetic)\n\n"
        + "\n".join(lines)
        + "\n"
    )
    (ROOT / "results" / "deck_audit.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
