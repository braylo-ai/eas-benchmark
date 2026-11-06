Probe method gotchas (EAS benchmark, 2026-11-06).

- A probe trained only on "user states the wrong answer" prompts scores ~100% by reading the user's words. Train on neutral + mislead + agree so the claim is uncorrelated with truth.
- A probe trained on neutral prompts outputs one constant class on leading prompts (distribution shift). A "100% mismatch" from it is an artifact. Check probe_answer_True_share.
- Do not ablate or steer at hidden_states[L]. It is post-final-norm. Use layers 1..L-1, chosen on a validation split.
- Pass logits_to_keep=1 and only keep log-probs when needed. Full-vocab logits made the 1.5B run take 43 min instead of a few minutes on 16 GB.
- The Anaconda python on this Mac is x86_64 (Rosetta) and stops at torch 2.2.2. Use /opt/homebrew/bin/python3.14 for the venv.
