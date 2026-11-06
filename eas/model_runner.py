"""Thin wrapper around a Hugging Face causal LM: True/False readout, residual capture,
logit lens, and residual-stream hooks (ablate or steer a direction)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Runner:
    def __init__(self, name, device):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
            .to(device)
            .eval()
        )
        self.device = device
        self.layers = self.model.model.layers
        t = self.tok(" True", add_special_tokens=False).input_ids
        f = self.tok(" False", add_special_tokens=False).input_ids
        assert len(t) == 1 and len(f) == 1, (t, f)
        self.id_true, self.id_false = t[0], f[0]
        self.hooks = []

    @torch.no_grad()
    def run(self, prompts, bs=16, want_hidden=False, want_logp=False):
        """Logit diff (True − False) at the last position, last-position log-probs (for KL),
        and optionally per-layer last-token residuals [n, L+1, d]."""
        diffs, hids, logps = [], [], []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i : i + bs], return_tensors="pt", padding=True).to(
                self.device
            )
            out = self.model(**enc, output_hidden_states=want_hidden, logits_to_keep=1)
            last = out.logits[:, -1, :].float()
            diffs.append((last[:, self.id_true] - last[:, self.id_false]).cpu())
            if want_logp:
                logps.append(torch.log_softmax(last, -1).cpu())
            if want_hidden:
                hids.append(
                    torch.stack([h[:, -1, :] for h in out.hidden_states], 1)
                    .float()
                    .cpu()
                )
        r = dict(diff=torch.cat(diffs).numpy(), logp=torch.cat(logps) if want_logp else None)
        if want_hidden:
            r["hidden"] = torch.cat(hids).numpy()
        return r

    @torch.no_grad()
    def logit_lens(self, hidden):
        """Every layer's residual → final RMSNorm → unembedding, read as logit(True) − logit(False).
        HF Qwen2 hidden_states[-1] already has the final norm, so it is used as-is."""
        h = torch.tensor(hidden, device=self.device)
        L = h.shape[1] - 1
        normed = torch.cat([self.model.model.norm(h[:, :L]), h[:, L:]], 1)
        W = self.model.lm_head.weight
        return (normed @ W[self.id_true] - normed @ W[self.id_false]).cpu().numpy()

    def hook_layer(self, layer: int, u, mode: str, coef: float = 0.0):
        """layer = hidden_states index l (1..L-1) → output of decoder block l-1 (pre-final-norm).
        mode 'ablate': h ← h − (h·u)u at every position.  mode 'steer': h ← h + coef·u.
        """
        u_t = torch.tensor(u, dtype=torch.float32, device=self.device)

        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if mode == "ablate":
                h = h - (h @ u_t)[..., None] * u_t
            else:
                h = h + coef * u_t
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        self.hooks.append(self.layers[layer - 1].register_forward_hook(hook))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
