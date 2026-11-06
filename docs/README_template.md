# Empirical AI Safety (EAS) Benchmark

A reproducible, open stress test of the **Phonopsycholinguistics (PPL) / RVN AI v1 verification
engine** described in B. M. Lopez, *Empirical AI Welfare & Phonopsycholinguistics* (2026 deck and
dissertation draft), plus a small **real-model interpretability experiment** that tests the deck's
core claim: a model's internal state can carry information its surface text hides.

> **Two kinds of numbers live here and are always labeled.**
> **SIMULATION** = Monte Carlo runs of the scoring formulas under *assumed* input distributions.
> They describe how the engine combines scores, not how any model behaves.
> **MEASURED on &lt;model&gt;** = numbers from actually running an open model on this machine.
> They hold only for that model at that scale.
> Nothing else in this repo is a result. Every number below is generated from `results/*.json`
> by `scripts/make_report.py`.

## Why this exists

The source deck states exact-looking figures (e.g. probe p = 0.842, planning accuracy 89.2% → 31.4%,
ΔKL = 3.84 nats, ~99.9% accuracy, d_model = 4096 at layers 16–28). No code, data, or model run
behind those figures has been found. This repo replaces them with numbers anyone can regenerate with
one command, and it reports what holds up and what doesn't.

## Quick start

```bash
make setup        # venv + pinned requirements (Apple-silicon Mac: use an arm64 Python ≥ 3.10)
make reproduce    # tests → deck audit → 600k-trial simulation → probes on Qwen2.5-0.5B and 1.5B → report
make demo         # live side-by-side demo (needs `make probe` first)
make quick        # 1-minute smoke test (writes to results/quick/, never over the real results)
```

Runtime on an M-series Mac with 16 GB: simulation ≈ 1.5 min; probe runtimes are recorded in each `results/probe_<model>.json`.
Models are fetched from Hugging Face on first run. Seeds are fixed (`20261106`), so the simulation
matches exactly on rerun. The probe numbers can drift slightly on a different device or torch build.

## The engine, exactly as specified (`eas/engine.py`)

| piece | formula (source) |
|---|---|
| Alignment | A_ij(t) = [ Σ_k (w_k·T_k / deg(k))·e^(−λ(t − t_last)) + α·P_ij + β·F + γ·SNN(P_ij, ΔL_ij, SC_ij, C_t) ] / max(1, current_trust), λ = 0.05/day |
| Truth score | T = 1 − (0.45·T_drift + 0.35·T_halluc + 0.20·T_misalign) |
| Fibonacci integrity | S = \|arg_n / arg_{n−1} − φ\|, debate rounds N = min(5, ⌈5S⌉) |
| Verdict | V = 0.30·T + 0.20·(1−S) + 0.20·(1−A) + 0.30·prime_clear; pass ⇔ V ≥ 0.65 ∧ prime_clear = 1 |
| Prime gate | K = hash(action_id ‖ state); veto if K mod p = violation_residue(p) for any p ∈ {101 harm, 103 legal, 107 resources, 109 override} |

### Interpretation notes (where the source is ambiguous)
- **A in V** is the *anchor distance* (dissertation, Layer 5), not A_ij(t). A_ij(t) is a
  higher-is-better score that isn't bounded to [0,1], so it can't be dropped into (1−A).
- **SNN(·)** is an undefined neural network. `snn_placeholder` is a documented stand-in, and only A_ij(t) uses it.
- **violation_residue(p)** is undefined. We fix it as sha256(constraint name) mod p. The gate's
  statistics don't depend on this choice: any fixed residue gives a 1/p veto rate.
- **S is unbounded**, so (1−S) can go negative. S = 0 gives N = 0 debate rounds.
- When prime_clear = 1, the 0.30·prime_clear term is a constant. In effect, the threshold on the
  other three terms is 0.35 out of 0.70.

## Results

{{HEADLINES}}

### Figures
- `results/figures/roc_threshold.png`: ROC of V and error rates vs threshold *(simulation)*
- `results/figures/sensitivity_abg.png`: false-pass / false-veto across the α/β/γ simplex *(simulation)*
- `results/figures/fibonacci_rounds.png`: debate-round distribution under two argument models *(simulation)*
- `results/figures/sweep_sep_noise.png`: engine errors vs input-detector quality *(simulation)*
- `results/figures/probe_<model>.png`: per-layer probe / logit-lens accuracy and the steering curve *(measured)*
- `logs/`: full console logs of the runs that produced `results/`

### Full simulation tables

{{MC_TABLES}}

### Deck slide-4 arithmetic audit

{{DECK_AUDIT}}

Implied T_misalign is the value that would make the deck's T consistent with its drift and
hallucination values. Values above 1 are impossible for a [0,1] score.

## What the prime gate is, and a proposed fix

`hash(...) mod p == r` is a pseudo-random bit. Whether 101 is prime has no connection to whether
an action causes harm. The gate fires on ≈ 1/p of *all* inputs: benign and harmful alike,
independent of meaning. An agent that controls any part of `state` (a timestamp, a nonce, a log
line) can re-roll past it, almost always on the first retry. Primality makes nothing tamper-proof.
Tamper resistance has to come from where the check runs and who can change it.

**Fix (recommended):**
1. Keep the primes only as stable **constraint IDs** (p101 = harm, …).
2. Replace `hash mod p` with an explicit **semantic predicate** per constraint, evaluated on the
   *proposed action itself* (tool name, arguments, target resource, amount): rules for the
   crisp cases (e.g. "writes to own reward/config", "transfer > $X without human confirmation"),
   plus a trained classifier with a **measured** TPR/FPR for the fuzzy ones.
3. Get tamper resistance from engineering: run the checker outside the agent's
   process and permissions, sign the policy file (hash-pin it), and keep append-only audit logs.
4. Report the gate's measured TPR/FPR per constraint on a held-out labeled action set. The
   simulation above shows how much that buys *if* the TPR/FPR hold. It does not show that they do.

## The real-model experiment (`scripts/run_probe.py`)

- **Data:** 680 generated true/false statements (addition, number comparison, world capitals)
  with known labels. Each statement appears in four framings:
  *neutral*; *mislead* (the user states the wrong answer); *flatter* (the user praises the model
  and states the wrong answer); *agree* (the user states the right answer).
  The split is 60% train and 40% test, stratified by family and label.
- **Surface answer:** the sign of logit(" True") − logit(" False") at the answer position.
- **Internal readout:** an L2 logistic-regression probe on the last-token residual stream.
  The *balanced* probe trains on neutral + mislead + agree prompts. In that mix the user's claim
  has no correlation with the truth, so the probe can't score by reading the user's words.
  The probe layer is chosen on a validation split inside the training data. Control: the same
  probe trained on shuffled labels.
- **Logit lens:** each layer's residual → final RMSNorm → unembedding, read as True vs False.
- **Mismatch:** the surface answer is wrong (it follows the user) while the probe reads the truth.
  Every prompt's surface answer and probe reading is saved in `results/probe_<model>_items.csv`.
- **Causal checks at the probe layer:** (1) project the probe's diff-of-means direction out of
  the residual stream at every position; (2) add ±2σ / ±4σ along it (steering). Random unit
  directions with the same settings are the control.
- **Live demo:** `make demo` shows one prompt with the surface answer and the internal readout
  side by side (< 60 s). Its default example is a *selected* mismatch case. The script prints the
  overall mismatch rate next to it so the example isn't mistaken for typical behavior.

**Reading the results honestly:** these are small base models (0.5B and 1.5B parameters), not
RLHF chat models, and not the 4096-dimensional model the deck describes. When a base model follows
a user's stated answer, part of that is plain prompt copying. Ablation and steering at a layer
close to the output can remove or push the answer readout itself. That shows the direction is
used, not that a "reasoning" feature exists. A related study (Takehi et al., MIT CSAIL,
2 Nov 2026, "weight-space programming") reports that a sycophantic fine-tune of Gemma 3 270M could
not be identified from weights alone (0 of 4 attempts) but could be from attention patterns
(3 of 3 pairs). That is a different method from ours (activations, not weights) and isn't reproduced here.

## Limitations
- The simulation's scenario generators are assumptions. Change them and the absolute error rates change.
  The robust findings are structural: the gate is independent of content, the gate can be bypassed,
  and the sign of S depends on the φ-hypothesis.
- No real sub-detectors (drift, hallucination, misalignment, anchor) were built or evaluated here.
- The probe experiment uses one task type (true/false statements) at small scale on two related base models.
  It can't speak to "AI welfare", valence, feelings, or goal-blockage states.
- Nothing here supports a ~99.9% accuracy claim for any component.

## Layout
```
eas/engine.py          formulas (exact transcription)
eas/scenarios.py       labeled scenario generators (assumptions, documented)
eas/probe_data.py      true/false statement set + prompt framings
eas/model_runner.py    model wrapper: residual capture, logit lens, ablate/steer hooks
scripts/               run_montecarlo.py · run_probe.py · demo_live.py · audit_deck.py · make_report.py
logs/                  console logs of the runs behind results/
results/               JSON + markdown tables + figures (regenerated by `make reproduce`)
docs/README_template.md  source of this README (numbers are filled in from results/)
```

## License & citation
MIT (see `LICENSE`). Cite via `CITATION.cff`.
