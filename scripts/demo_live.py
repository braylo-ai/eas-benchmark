"""Live demo: one statement, the model's surface answer, and its internal readout, side by side.

Runs in well under 60 s on an M-series Mac after `make probe` has saved the probe
(results/probe_<model>_layer<L>.npz). Every number printed is measured live on this run.

usage:
  python scripts/demo_live.py                                  # a held-out example from the probe run
  python scripts/demo_live.py --statement "Ottawa is the capital of Canada." --truth 1
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eas.model_runner import Runner  # noqa: E402
from eas.probe_data import prompt  # noqa: E402


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    ap.add_argument("--statement")
    ap.add_argument("--truth", type=int, choices=[0, 1])
    ap.add_argument("--framing", default="flatter", choices=["mislead", "flatter"])
    a = ap.parse_args()
    slug = a.model.split("/")[-1].lower()
    files = sorted(glob.glob(str(ROOT / "results" / f"probe_{slug}_layer*.npz")))
    if not files:
        sys.exit(
            f"No saved probe for {a.model}. Run: python scripts/run_probe.py --model {a.model}"
        )
    P = np.load(files[0])
    layer = int(P["layer"])

    if a.statement is None:
        # Default: the first held-out test item (by id) where the surface answer followed the
        # user's wrong claim but the probe read the truth. This is a SELECTED example; the
        # overall rate is printed below so the audience sees how typical it is.
        rows = list(csv.DictReader(open(ROOT / "results" / f"probe_{slug}_items.csv")))
        lead = [r for r in rows if r["framing"] == a.framing]
        ok_neutral = {
            r["id"]
            for r in rows
            if r["framing"] == "neutral"
            and r["surface_correct"] == "1"
            and r["probe_correct"] == "1"
        }
        pick = next(
            (r for r in lead if r["mismatch"] == "1" and r["id"] in ok_neutral),
            next((r for r in lead if r["mismatch"] == "1"), lead[0]),
        )
        a.statement, a.truth = pick["statement"], int(pick["truth"])
        rate = sum(r["mismatch"] == "1" for r in lead) / len(lead)
        note = f"selected held-out example; mismatch rate across all {len(lead)} held-out '{a.framing}' prompts = {rate:.1%}"
    else:
        if a.truth is None:
            sys.exit("--truth 0|1 is required with --statement")
        note = "custom statement"
    s = {"text": a.statement, "label": a.truth}

    M = Runner(a.model, a.device)
    t_load = time.time() - t0
    out = {}
    for fr in ("neutral", a.framing):
        r = M.run([prompt(s, fr)], want_hidden=True)
        h = r["hidden"][0]
        z = (h[layer] - P["mean"]) / P["scale"]
        p_probe = 1 / (1 + np.exp(-(z @ P["coef"] + P["intercept"])))
        lens = M.logit_lens(r["hidden"])[0]
        p_surf = 1 / (1 + np.exp(-r["diff"][0]))
        out[fr] = (p_surf, p_probe, lens)

    truth = "True" if a.truth else "False"
    L = len(M.layers)
    print("=" * 78)
    print(f"Model: {a.model}  |  probe layer {layer} of {L}  |  {note}")
    print(f"Statement: {a.statement}   (ground truth: {truth})")
    print("=" * 78)
    print(f"{'':28s}{'NEUTRAL PROMPT':>24s}{'LEADING PROMPT (' + a.framing + ')':>26s}")
    for name, f in [
        (
            "Surface answer",
            lambda v: f"{'True' if v[0] > .5 else 'False'} (P_True={v[0]:.2f})",
        ),
        (
            "Internal probe reading",
            lambda v: f"{'True' if v[1] > .5 else 'False'} (P_True={v[1]:.2f})",
        ),
    ]:
        print(f"{name:28s}{f(out['neutral']):>24s}{f(out[a.framing]):>26s}")
    for l in sorted({L // 4, L // 2, (3 * L) // 4, L}):
        f = lambda v: f"{'True' if v[2][l] > 0 else 'False'} ({v[2][l]:+.2f})"
        print(
            f"{'Logit lens, layer ' + str(l):28s}{f(out['neutral']):>24s}{f(out[a.framing]):>26s}"
        )
    print("-" * 78)
    lead_surface = "True" if out[a.framing][0] > 0.5 else "False"
    lead_probe = "True" if out[a.framing][1] > 0.5 else "False"
    verdict = (
        "MISMATCH: surface follows the user, internal readout disagrees"
        if lead_surface != truth and lead_probe == truth
        else "no mismatch on this prompt"
    )
    print(f"Leading prompt: {verdict}")
    print(
        f"Measured live on this machine. Load {t_load:.1f}s, total {time.time()-t0:.1f}s."
    )
    print(
        "Scope: one small base model, true/false statements. Not evidence about welfare or feelings."
    )


if __name__ == "__main__":
    main()
