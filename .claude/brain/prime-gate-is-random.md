The PPL prime gate (hash mod p == residue) is a content-independent random filter.

- It vetoes 1 − Π(1 − 1/p) ≈ 3.76% of every input for p ∈ {101, 103, 107, 109}.
- Re-rolling any free field in `state` clears a veto in about one retry.
- Proposed fix: keep primes as constraint IDs only; use semantic predicates on the action; get tamper resistance from process isolation, signed policy and audit logs.
- Source numbers: results/montecarlo.json (prime_gate).
