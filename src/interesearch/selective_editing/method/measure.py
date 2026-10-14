"""Measurements of a weight edit on token windows: the per-window CE of the edited model and its KL to the original.
Rules: papers/selective-editing/index.qmd#sec-calibration and papers/selective-editing/index.qmd#sec-analysis.

Per window: the log-softmax of the logits in their own dtype, the mean CE over the window's predictions on the
device, then float64 on the CPU. ``KL(p_0 || p)`` summed over the vocabulary, averaged over the predictions, then
float64. The edit is written into the live weights (``applied_update``). The reference forward of the same batch
runs inside it with ``W_0`` (``ActiveEdit.reference``).
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from interesearch.edits import applied_update, logits_of
BATCH = 16

Tensor = torch.Tensor


@torch.no_grad()
def doc_ce(model: nn.Module, tokens: Tensor, device, bs: int = BATCH) -> Tensor:
    """Mean CE of each window's predictions (log-softmax in the logits' dtype, mean on the device), float64 on the
    CPU."""
    out = []
    for s in range(0, tokens.shape[0], bs):
        w = tokens[s:s + bs].to(device)
        lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
        out.append((-lp.gather(-1, w[:, 1:].unsqueeze(-1)).squeeze(-1)).mean(1).double().cpu())
    return torch.cat(out)


@torch.no_grad()
def ce_docs(model: nn.Module, base: Mapping[str, Tensor], tokens: Tensor, update: Mapping[str, Tensor] | None,
            device, bs: int = BATCH) -> Tensor:
    """Per-window CE of the model with ``W_0 + update`` (``update=None``: the original model)."""
    with applied_update(model, base, update):
        return doc_ce(model, tokens, device, bs)


@torch.no_grad()
def edit_readout(model: nn.Module, base: Mapping[str, Tensor], update: Mapping[str, Tensor],
                 sets: Mapping[str, Tensor], device, base_ce: Mapping[str, Tensor], bs: int = BATCH) -> dict[str, dict]:
    """Per set: per-window CE of the edited model, its change against ``base_ce`` and ``KL(p_0 || p)``, with their
    means."""
    out = {}
    for name, tokens in sets.items():
        ce, kl = [], []
        with applied_update(model, base, update) as edit:
            for s in range(0, tokens.shape[0], bs):
                w = tokens[s:s + bs].to(device)
                lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
                ce.append((-lp.gather(-1, w[:, 1:].unsqueeze(-1)).squeeze(-1)).mean(1).double().cpu())
                with edit.reference():
                    lp0 = F.log_softmax(logits_of(model(w[:, :-1])), -1)
                kl.append((lp0.exp() * (lp0 - lp)).sum(-1).mean(1).double().cpu())
        ce, kl = torch.cat(ce), torch.cat(kl)
        d = ce - base_ce[name]
        out[name] = {"ce_doc": ce.tolist(), "delta_ce_doc": d.tolist(), "kl_doc": kl.tolist(),
                     "ce_base": float(base_ce[name].mean()), "ce": float(ce.mean()), "delta_ce": float(d.mean()),
                     "kl": float(kl.mean())}
    return out


def update_norm(update: Mapping[str, Tensor]) -> float:
    """Frobenius norm of an edit over all its modules (float64)."""
    return float(sum(float(torch.as_tensor(D).double().pow(2).sum()) for D in update.values()) ** 0.5)
