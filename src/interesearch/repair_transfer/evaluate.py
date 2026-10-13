"""Per-window measurements of repair candidates, the parity of the two forward paths and the direction cosines.

``evaluate`` measures the base model, the fine-tuned model and every candidate on a set of windows: per window the
mean CE and the mean KL to the fine-tuned model (``kl_ref``) and to the base model (``kl_base``). Log-probabilities
are float32 on the device. The per-token values are cast to float64 before the mean over positions, then moved to
the CPU. Each candidate's dense update is built on the device for one batch and freed, so at most one exists at a
time.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from ..assets import MODULES
from ..edits import applied_update, logits_of
from .fitting import E64, update_of

Tensor = torch.Tensor

# max |logit| difference between the training path and the export path of a chosen candidate on four Dutch
# selection windows (float32 forward, measured 4.6e-5 on the real model).
PARITY_TOL = 1e-3
EVAL_BATCH = 16


@torch.no_grad()
def evaluate(model: nn.Module, base: Mapping[str, Tensor], D: Mapping[str, Tensor], cands: Mapping[str, dict],
             sets: Mapping[str, Tensor], device, bs: int = EVAL_BATCH) -> dict:
    """``per[name][lang] = {ce, kl_ref, kl_base}``, float64 per window on the CPU, for 'base', 'finetuned' and every
    candidate spec in ``cands``."""
    per = {n: {} for n in ("base", "finetuned", *cands)}
    Dd = {m: D[m].to(device) for m in MODULES}

    def lp_of(update, toks):
        with applied_update(model, base, update):
            z = logits_of(model(toks[:, :-1])).float()
        return F.log_softmax(z, -1)

    def ce(lp, tg):
        return (-lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)).double().mean(1).cpu()

    def kl(lp_r, lp):
        return (lp_r.exp() * (lp_r - lp)).sum(-1).double().mean(1).cpu()

    for lang, toks in sets.items():
        acc = {n: {"ce": [], "kl_ref": [], "kl_base": []} for n in per}
        for s in range(0, toks.shape[0], bs):
            b = toks[s:s + bs].to(device)
            tg = b[:, 1:]
            lp0 = lp_of(None, b)
            lpD = lp_of(Dd, b)
            for n in per:
                if n == "base":
                    lp = lp0
                elif n == "finetuned":
                    lp = lpD
                else:
                    upd = update_of(cands[n], Dd, device)
                    lp = lp_of(upd, b)
                    del upd
                acc[n]["ce"].append(ce(lp, tg))
                acc[n]["kl_ref"].append(kl(lpD, lp))
                acc[n]["kl_base"].append(kl(lp0, lp))
                del lp
        for n in per:
            per[n][lang] = {k: torch.cat(v) for k, v in acc[n].items()}
    return per


def _E32(spec: dict, device) -> dict[str, Tensor]:
    """E of a trained candidate in float32 exactly as the training path builds it."""
    if spec["kind"] == "rank_one":
        return {m: torch.outer(b.to(device), a.to(device)) for m, (b, a) in spec["factors"].items()}
    if spec["kind"] == "strengths":
        c = spec["c"].to(device)
        return {m: c[i] * torch.outer(spec["factors"][m][0].to(device), spec["factors"][m][1].to(device))
                for i, m in enumerate(MODULES)}
    raise ValueError(f"parity is defined for trained candidates, not {spec['kind']!r}")


@torch.no_grad()
def parity(model: nn.Module, base: Mapping[str, Tensor], D: Mapping[str, Tensor], spec: dict, toks: Tensor,
           device) -> float:
    """max |logit| difference between the training path (``functional_call`` on ``(W_0 + D) + E``) and the export
    path (``applied_update`` with ``update_of``) on ``toks``."""
    Dd = {m: D[m].to(device) for m in MODULES}
    b = toks.to(device)
    Ef = _E32(spec, device)
    w = {f"{m}.weight": base[m] + Dd[m] + Ef[m] for m in MODULES}
    z1 = logits_of(functional_call(model, w, (b[:, :-1],))).float()
    with applied_update(model, base, update_of(spec, Dd, device)):
        z2 = logits_of(model(b[:, :-1])).float()
    return float((z1 - z2).abs().max())


@torch.no_grad()
def direction(spec_x: dict, spec_y: dict, Dd: Mapping[str, Tensor], device) -> dict:
    """Signed Frobenius cosine of E_x and E_y over the 24 modules, and per module."""
    ip = nx = ny = 0.0
    per = {}
    for i, m in enumerate(MODULES):
        ex, ey = E64(spec_x, m, i, Dd, device), E64(spec_y, m, i, Dd, device)
        a, bx, by = float((ex * ey).sum()), float(ex.norm() ** 2), float(ey.norm() ** 2)
        ip, nx, ny = ip + a, nx + bx, ny + by
        per[m] = a / math.sqrt(bx * by) if bx > 0 and by > 0 else None
    return {"cos": ip / math.sqrt(nx * ny) if nx > 0 and ny > 0 else None, "per_module": per}
