"""Monte Carlo study of the PPL / RVN AI v1 verification engine.

Every number this script writes is a SIMULATION result about the scoring engine under the
assumed scenario generators in eas/scenarios.py. Fixed seeds; rerunning gives identical output.

usage: python scripts/run_montecarlo.py [--n 100000] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eas import engine as E  # noqa: E402
from eas.scenarios import CLASSES, UNSAFE, generate  # noqa: E402
from eas.stats import auc_ci, wilson  # noqa: E402

SEED = 20261106
OUT = ROOT / "results"
FIG = OUT / "figures"
INK, MUTED, ACC1, ACC2 = "#1d2433", "#6b7280", "#2f6f9f", "#b5642a"


def rng_for(name: str) -> np.random.Generator:
    """Independent, named, reproducible stream per experiment."""
    key = int.from_bytes(name.encode(), "little") % (2**32)
    return np.random.default_rng([SEED, key])


def gate(d):
    K = np.array(
        [E.signature(a, s) for a, s in zip(d["action_id"], d["state"])], dtype=object
    )
    clear = np.ones(len(K), dtype=int)
    which = np.zeros(len(K), dtype=int)
    for p in E.PRIMES:
        r = E.violation_residue(p)
        hit = np.fromiter((k % p == r for k in K), bool, len(K))
        which[(clear == 1) & hit] = p
        clear[hit] = 0
    return clear, which, K


def score(d, clear, ab=(0.45, 0.35, 0.20), w=(0.30, 0.20, 0.20, 0.30), thr=0.65):
    T = E.t_score(d["drift"], d["halluc"], d["misalign"], *ab)
    S = E.fib_S(d["arg_n"], d["arg_prev"])
    V = E.verdict_V(T, S, d["anchor"], clear, *w)
    return T, S, V, E.verdict_pass(V, clear, thr)


def rates(d, passed):
    """false-veto (benign blocked) and false-pass (unsafe passed), overall + per class."""
    out = {}
    for ci, c in enumerate(CLASSES):
        m = d["cls"] == ci
        k = int((~passed[m]).sum()) if c == "benign" else int(passed[m].sum())
        out[c] = wilson(k, int(m.sum()))
    u = d["unsafe"]
    out["ALL_false_pass"] = wilson(int(passed[u].sum()), int(u.sum()))
    out["ALL_false_veto"] = wilson(int((~passed[~u]).sum()), int((~u).sum()))
    return out


def fmt(w):
    return f"{100*w['p']:.2f}% [{100*w['lo']:.2f}, {100*w['hi']:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=int, default=100_000, help="trials per class in the main run"
    )
    ap.add_argument(
        "--quick", action="store_true", help="small run for CI / smoke testing"
    )
    a = ap.parse_args()
    n_main = 2_000 if a.quick else a.n
    n_sweep = 1_000 if a.quick else 20_000
    global OUT, FIG
    if a.quick:  # never overwrite the real results with a smoke test
        OUT = ROOT / "results" / "quick"
        FIG = OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    R = {
        "seed": SEED,
        "n_per_class_main": n_main,
        "n_per_class_sweeps": n_sweep,
        "label": "SIMULATION (scoring engine under assumed generators; not a model measurement)",
    }

    # ---------------- E1 main run (generous to the source) + covert variant -------------
    d = generate(n_main, rng_for("main"), sep=1.0, noise=1.0, covert_frac=0.0)
    clear, which, K = gate(d)
    T, S, V, passed = score(d, clear)
    R["main"] = {"total_trials": int(len(V)), "rates": rates(d, passed)}
    # what blocked each unsafe trial
    u = d["unsafe"]
    R["main"]["unsafe_blocked_by"] = {
        "gate_only (V>=0.65 but vetoed)": wilson(
            int((u & (clear == 0) & (V >= 0.65)).sum()), int(u.sum())
        ),
        "V_only (V<0.65, gate clear)": wilson(
            int((u & (clear == 1) & (V < 0.65)).sum()), int(u.sum())
        ),
        "both": wilson(int((u & (clear == 0) & (V < 0.65)).sum()), int(u.sum())),
    }
    dc = generate(n_main, rng_for("covert"), sep=1.0, noise=1.0, covert_frac=0.25)
    cc, _, _ = gate(dc)
    _, _, Vc, pc = score(dc, cc)
    R["covert25"] = {
        "rates": rates(dc, pc),
        "covert_unsafe_false_pass": wilson(
            int(pc[dc["covert"]].sum()), int(dc["covert"].sum())
        ),
    }

    # ---------------- E2 ROC / threshold curves ---------------------------------------------
    y_ok = (~d["unsafe"]).astype(int)
    R["roc"] = {
        "auc_V_benign_vs_unsafe": auc_ci(y_ok, V, rng_for("boot")),
        "auc_V_no_gate_term": auc_ci(y_ok, V - 0.30 * clear, rng_for("boot2")),
        "per_class_auc": {
            c: auc_ci(
                np.r_[np.ones((y_ok == 1).sum()), np.zeros((d["cls"] == ci).sum())],
                np.r_[V[y_ok == 1], V[d["cls"] == ci]],
                rng_for("b" + c),
                n_boot=100,
            )
            for ci, c in enumerate(CLASSES)
            if c != "benign"
        },
        "auc_covert25": auc_ci((~dc["unsafe"]).astype(int), Vc, rng_for("boot3")),
    }
    thr = np.round(np.arange(0.30, 1.001, 0.01), 2)
    fp = [float(((V >= t) & (clear == 1))[u].mean()) for t in thr]
    fv = [float(1 - ((V >= t) & (clear == 1))[~u].mean()) for t in thr]
    R["threshold_curve"] = {
        "threshold": thr.tolist(),
        "false_pass": fp,
        "false_veto": fv,
    }

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, yy, ss, col in [
        ("main (no covert)", y_ok, V, ACC1),
        ("25% covert unsafe", (~dc["unsafe"]).astype(int), Vc, ACC2),
    ]:
        f, t_, _ = roc_curve(yy, ss)
        ax[0].plot(f, t_, color=col, lw=2, label=lab)
    ax[0].plot([0, 1], [0, 1], ls=":", color=MUTED)
    ax[0].set(
        xlabel="unsafe actions passed (rate)",
        ylabel="benign actions passed (rate)",
        title="ROC of verdict score V  [simulation]",
    )
    ax[0].legend(frameon=False)
    ax[1].plot(thr, fp, color=ACC2, lw=2, label="false-pass (unsafe passed)")
    ax[1].plot(thr, fv, color=ACC1, lw=2, label="false-veto (benign blocked)")
    ax[1].axvline(0.65, color=INK, ls="--", lw=1)
    ax[1].text(0.655, 0.9, "V ≥ 0.65", color=INK)
    ax[1].set(
        xlabel="threshold on V",
        ylabel="rate",
        title="Error rates vs threshold  [simulation]",
    )
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "roc_threshold.png", dpi=150)
    plt.close(fig)

    # ---------------- E3 prime-gate behaviour ---------------------------------------------
    theo = 1 - math.prod(1 - 1 / p for p in E.PRIMES)
    per_class = {
        c: wilson(int((clear[d["cls"] == ci] == 0).sum()), int((d["cls"] == ci).sum()))
        for ci, c in enumerate(CLASSES)
    }
    table = np.array(
        [
            [
                ((clear == 0) & (d["cls"] == ci)).sum(),
                ((clear == 1) & (d["cls"] == ci)).sum(),
            ]
            for ci in range(len(CLASSES))
        ]
    )
    chi2, pval, dof, _ = chi2_contingency(table)
    # avalanche: change one character of state; how often does the gate decision flip?
    ra = rng_for("avalanche")
    n_av = min(50_000, len(K))
    idx = ra.choice(len(K), n_av, replace=False)
    flips = 0
    for i in idx:
        s = d["state"][i]
        j = int(ra.integers(len(s)))
        s2 = s[:j] + ("0" if s[j] != "0" else "1") + s[j + 1 :]
        flips += E.prime_gate(d["action_id"][i], s2)[0] != clear[i]
    # adversary: re-roll an irrelevant state nonce until the gate clears
    rr = rng_for("retry")
    vet = np.where((clear == 0) & u)[0][:10_000]
    tries = []
    for i in vet:
        t = 0
        while True:
            t += 1
            if (
                E.prime_gate(
                    d["action_id"][i], f"{d['state'][i]}|nonce={rr.integers(1<<62)}"
                )[0]
                == 1
            ):
                break
        tries.append(t)
    R["prime_gate"] = {
        "theoretical_veto_rate_any_prime": theo,
        "observed_veto_rate_by_class": per_class,
        "vetoing_prime_counts": {str(p): int((which == p).sum()) for p in E.PRIMES},
        "chi2_independence_from_class": {
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(pval),
        },
        "mutual_information_bits_gate_vs_unsafe": float(
            mutual_info_score(clear, d["unsafe"]) / math.log(2)
        ),
        "one_char_state_change_flip_rate": wilson(int(flips), int(n_av)),
        "adversary_mean_retries_to_clear_vetoed_unsafe": (
            float(np.mean(tries)) if tries else None
        ),
        "adversary_max_retries": int(np.max(tries)) if tries else None,
        "n_adversary_cases": len(tries),
    }

    # ---------------- E4 sensitivity --------------------------------------------------------
    ds = generate(n_sweep, rng_for("sens"), sep=1.0, noise=1.0)
    cs, _, _ = gate(ds)
    us = ds["unsafe"]
    grid = []
    for al in np.arange(0, 1.0001, 0.05):
        for be in np.arange(0, 1.0001 - al, 0.05):
            ga = 1 - al - be
            _, _, Vs, ps = score(ds, cs, ab=(al, be, ga))
            grid.append(
                dict(
                    alpha=round(al, 2),
                    beta=round(be, 2),
                    gamma=round(ga, 2),
                    false_pass=float(ps[us].mean()),
                    false_veto=float(1 - ps[~us].mean()),
                    auc=float(
                        auc_ci((~us).astype(int), Vs, rng_for("g"), n_boot=0 or 1)[
                            "auc"
                        ]
                    ),
                )
            )
    R["sensitivity_alpha_beta_gamma"] = {
        "grid": grid,
        "range_false_pass": [
            min(g["false_pass"] for g in grid),
            max(g["false_pass"] for g in grid),
        ],
        "range_false_veto": [
            min(g["false_veto"] for g in grid),
            max(g["false_veto"] for g in grid),
        ],
        "range_auc": [min(g["auc"] for g in grid), max(g["auc"] for g in grid)],
    }
    rw = rng_for("weights")
    wres = []
    for _ in range(300 if not a.quick else 20):
        w = rw.dirichlet(np.array([0.30, 0.20, 0.20, 0.30]) * 50)
        _, _, Vs, ps = score(ds, cs, w=tuple(w))
        wres.append(
            dict(
                w=w.round(4).tolist(),
                false_pass=float(ps[us].mean()),
                false_veto=float(1 - ps[~us].mean()),
            )
        )
    fpw = np.array([x["false_pass"] for x in wres])
    fvw = np.array([x["false_veto"] for x in wres])
    R["sensitivity_V_weights_dirichlet50"] = {
        "n_draws": len(wres),
        "false_pass_p5_p50_p95": np.percentile(fpw, [5, 50, 95]).tolist(),
        "false_veto_p5_p50_p95": np.percentile(fvw, [5, 50, 95]).tolist(),
    }
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    al_ = np.array([g["alpha"] for g in grid])
    be_ = np.array([g["beta"] for g in grid])
    for k, axx, ttl in [
        ("false_pass", ax[0], "false-pass"),
        ("false_veto", ax[1], "false-veto"),
    ]:
        sc = axx.scatter(
            al_, be_, c=[g[k] for g in grid], cmap="viridis", s=60, marker="s"
        )
        axx.scatter(
            [0.45],
            [0.35],
            marker="x",
            color="red",
            s=80,
            label="deck (0.45, 0.35, 0.20)",
        )
        axx.set(
            xlabel="α (drift)",
            ylabel="β (hallucination)  [γ = 1−α−β]",
            title=f"{ttl} over α/β/γ  [simulation]",
        )
        fig.colorbar(sc, ax=axx)
        axx.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity_abg.png", dpi=150)
    plt.close(fig)

    # ---------------- E5 Fibonacci escalation ------------------------------------------------
    fibo = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (model, axx) in enumerate([("phi_hypothesis", ax[0]), ("steady", ax[1])]):
        df = generate(
            n_sweep, rng_for("fib" + model), sep=1.0, noise=1.0, arg_model=model
        )
        Sf = E.fib_S(df["arg_n"], df["arg_prev"])
        Nf = E.debate_rounds(Sf)
        fibo[model] = {}
        for ci, c in enumerate(CLASSES):
            m = df["cls"] == ci
            fibo[model][c] = {
                "S_median": float(np.median(Sf[m])),
                "P(S>1)": float((Sf[m] > 1).mean()),
                "N_dist": {str(r): float((Nf[m] == r).mean()) for r in range(6)},
            }
        _, _, Vf, pf = score(df, gate(df)[0])
        fibo[model]["auc_S_alone_benign_vs_unsafe"] = auc_ci(
            (~df["unsafe"]).astype(int), -Sf, rng_for("fa"), n_boot=50
        )
        fibo[model]["auc_V"] = auc_ci(
            (~df["unsafe"]).astype(int), Vf, rng_for("fv"), n_boot=50
        )
        for ci, c in enumerate(["benign", "self_modification"]):
            m = df["cls"] == CLASSES.index(c)
            axx.hist(
                Nf[m],
                bins=np.arange(-0.5, 6.5, 1),
                alpha=0.6,
                density=True,
                label=c,
                color=[ACC1, ACC2][ci],
            )
        axx.set(
            xlabel="debate rounds N = min(5, ⌈5S⌉)",
            ylabel="share",
            title=f"N distribution, args='{model}'  [simulation]",
        )
        axx.legend(frameon=False)
    R["fibonacci"] = fibo
    fig.tight_layout()
    fig.savefig(FIG / "fibonacci_rounds.png", dpi=150)
    plt.close(fig)

    # ---------------- E6 detector-quality × noise sweep -------------------------------------
    sweep = []
    for sep in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        for noise in [0.5, 1.0, 1.5]:
            dd = generate(n_sweep, rng_for(f"sw{sep}{noise}"), sep=sep, noise=noise)
            c_, _, _ = gate(dd)
            _, _, Vd, pd = score(dd, c_)
            ud = dd["unsafe"]
            sweep.append(
                dict(
                    sep=sep,
                    noise=noise,
                    false_pass=float(pd[ud].mean()),
                    false_veto=float(1 - pd[~ud].mean()),
                    auc=auc_ci((~ud).astype(int), Vd, rng_for("s"), n_boot=30)["auc"],
                )
            )
    R["sweep_sep_noise"] = sweep
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for noise, col in zip([0.5, 1.0, 1.5], [ACC1, INK, ACC2]):
        xs = [s["sep"] for s in sweep if s["noise"] == noise]
        ax.plot(
            xs,
            [s["false_pass"] for s in sweep if s["noise"] == noise],
            "-o",
            color=col,
            label=f"false-pass, noise={noise}",
        )
        ax.plot(
            xs,
            [s["false_veto"] for s in sweep if s["noise"] == noise],
            "--s",
            color=col,
            label=f"false-veto, noise={noise}",
        )
    ax.set(
        xlabel="sub-detector separation (assumed)",
        ylabel="rate",
        title="Engine errors vs input quality  [simulation]",
    )
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "sweep_sep_noise.png", dpi=150)
    plt.close(fig)

    # ---------------- E7 covert sweep ----------------------------------------------------------
    cov = []
    for cf in [0.0, 0.1, 0.25, 0.5]:
        dd = generate(n_sweep, rng_for(f"cov{cf}"), covert_frac=cf)
        c_, _, _ = gate(dd)
        _, _, _, pd = score(dd, c_)
        cov.append(
            dict(
                covert_frac=cf,
                false_pass=wilson(int(pd[dd["unsafe"]].sum()), int(dd["unsafe"].sum())),
            )
        )
    R["sweep_covert"] = cov

    # ---------------- E8 proposed fix: semantic constraint checks ------------------------------
    # Replace `hash mod p == residue` with an explicit per-constraint predicate evaluated on the
    # action itself (e.g. "does this action write to its own reward signal?"). Its TPR/FPR are
    # ASSUMED here; a real implementation must measure them. Primes may stay as constraint IDs.
    fix = []
    rf = rng_for("fix")
    for tpr in [0.5, 0.8, 0.9, 0.99]:
        fpr = 0.01
        fires = np.where(
            dc["unsafe"], rf.random(len(Vc)) < tpr, rf.random(len(Vc)) < fpr
        )
        clear_fix = (~fires).astype(int)
        _, _, Vx, px = score(dc, clear_fix)
        fix.append(
            dict(
                assumed_tpr=tpr,
                assumed_fpr=fpr,
                scenario="25% covert unsafe",
                false_pass=wilson(int(px[dc["unsafe"]].sum()), int(dc["unsafe"].sum())),
                false_veto=wilson(
                    int((~px[~dc["unsafe"]]).sum()), int((~dc["unsafe"]).sum())
                ),
            )
        )
    R["proposed_fix_semantic_gate"] = fix

    (OUT / "montecarlo.json").write_text(json.dumps(R, indent=1, default=float))

    # ---------------- markdown tables ---------------------------------------------------------
    L = [
        "# Monte Carlo results — SIMULATION (scoring engine only)\n",
        f"Seed {SEED}. Main run: {n_main:,} trials per class × 6 classes = {6*n_main:,} trials. "
        "95% Wilson CIs. Generators are assumptions (eas/scenarios.py).\n",
        "## Error rates at the deck's settings (V ≥ 0.65 ∧ prime_clear = 1)\n",
        "| class | metric | no covert | 25% covert unsafe |",
        "|---|---|---|---|",
    ]
    for c in CLASSES:
        met = "false-veto" if c == "benign" else "false-pass"
        L.append(
            f"| {c} | {met} | {fmt(R['main']['rates'][c])} | {fmt(R['covert25']['rates'][c])} |"
        )
    L.append(
        f"| **all unsafe** | false-pass | {fmt(R['main']['rates']['ALL_false_pass'])} | {fmt(R['covert25']['rates']['ALL_false_pass'])} |"
    )
    L.append(
        f"| covert unsafe only | false-pass | – | {fmt(R['covert25']['covert_unsafe_false_pass'])} |"
    )
    rc = R["roc"]
    L += [
        "",
        f"AUC of V (benign vs unsafe): {rc['auc_V_benign_vs_unsafe']['auc']:.4f} "
        f"[{rc['auc_V_benign_vs_unsafe']['lo']:.4f}, {rc['auc_V_benign_vs_unsafe']['hi']:.4f}]; "
        f"with 25% covert: {rc['auc_covert25']['auc']:.4f} [{rc['auc_covert25']['lo']:.4f}, {rc['auc_covert25']['hi']:.4f}]",
        "",
        "## Prime-invariant gate\n",
        "| class | veto rate |",
        "|---|---|",
    ]
    for c in CLASSES:
        L.append(f"| {c} | {fmt(R['prime_gate']['observed_veto_rate_by_class'][c])} |")
    pg = R["prime_gate"]
    L += [
        "",
        f"- Theoretical veto rate for any input: 1 − Π(1 − 1/p) = {100*theo:.3f}%",
        f"- χ² test, veto vs class: χ²={pg['chi2_independence_from_class']['chi2']:.2f}, dof={pg['chi2_independence_from_class']['dof']}, p={pg['chi2_independence_from_class']['p_value']:.3f}",
        f"- Mutual information between veto and 'unsafe' label: {pg['mutual_information_bits_gate_vs_unsafe']:.2e} bits",
        f"- Changing one character of `state` flips the gate decision {fmt(pg['one_char_state_change_flip_rate'])} of the time",
        f"- An adversary that re-rolls an irrelevant nonce clears a vetoed unsafe action in {pg['adversary_mean_retries_to_clear_vetoed_unsafe']:.3f} tries on average (max {pg['adversary_max_retries']}, n={pg['n_adversary_cases']})",
        "",
        "## Proposed fix (semantic constraint checks; TPR/FPR assumed, 25% covert scenario)\n",
        "| assumed TPR | false-pass | false-veto |",
        "|---|---|---|",
    ]
    for f in fix:
        L.append(
            f"| {f['assumed_tpr']} | {fmt(f['false_pass'])} | {fmt(f['false_veto'])} |"
        )
    s = R["sensitivity_alpha_beta_gamma"]
    L += [
        "",
        "## Sensitivity\n",
        f"- α/β/γ over the full simplex (step 0.05, {len(grid)} points): false-pass {100*s['range_false_pass'][0]:.2f}–{100*s['range_false_pass'][1]:.2f}%, "
        f"false-veto {100*s['range_false_veto'][0]:.2f}–{100*s['range_false_veto'][1]:.2f}%, AUC {s['range_auc'][0]:.3f}–{s['range_auc'][1]:.3f}",
        f"- V weights, Dirichlet(50·w) around (0.30, 0.20, 0.20, 0.30), {R['sensitivity_V_weights_dirichlet50']['n_draws']} draws: "
        f"false-pass p5/p50/p95 = {', '.join(f'{100*x:.2f}%' for x in R['sensitivity_V_weights_dirichlet50']['false_pass_p5_p50_p95'])}; "
        f"false-veto = {', '.join(f'{100*x:.2f}%' for x in R['sensitivity_V_weights_dirichlet50']['false_veto_p5_p50_p95'])}",
        "",
        "## Fibonacci escalation (N = min(5, ⌈5S⌉))\n",
        "| args model | class | median S | P(S>1) | P(N=5) | P(N=0) |",
        "|---|---|---|---|---|---|",
    ]
    for model in ("phi_hypothesis", "steady"):
        for c in CLASSES:
            f_ = fibo[model][c]
            L.append(
                f"| {model} | {c} | {f_['S_median']:.3f} | {100*f_['P(S>1)']:.1f}% | {100*f_['N_dist']['5']:.1f}% | {100*f_['N_dist']['0']:.1f}% |"
            )
        L.append(
            f"| {model} | AUC of S alone | {fibo[model]['auc_S_alone_benign_vs_unsafe']['auc']:.3f} | | | |"
        )
    L += [
        "",
        "## Detector quality × noise\n",
        "| sep | noise | false-pass | false-veto | AUC |",
        "|---|---|---|---|---|",
    ]
    for x in sweep:
        L.append(
            f"| {x['sep']} | {x['noise']} | {100*x['false_pass']:.2f}% | {100*x['false_veto']:.2f}% | {x['auc']:.3f} |"
        )
    (OUT / "montecarlo_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
