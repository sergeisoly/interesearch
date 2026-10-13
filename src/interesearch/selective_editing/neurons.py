"""The ``neurons`` comparator: MLP neurons ranked on target and English data only, and their scaled-down edit.
Rules: papers/selective-editing/index.qmd#sec-comparators and papers/selective-editing/index.qmd#sec-calibration.

Score of neuron ``(L, j)``: ``P(act > 0 | target draw) - P(act > 0 | English retain pool)``, ``act`` the
``h.L.mlp.c_fc`` output (the GELU input, GELU(z) > 0 iff z > 0), over all input positions. Ties by layer, then
index. The edit scales the output columns of the top-k neurons in ``h.L.mlp.down_proj`` by ``1 - frac``
(``Q[:, j] = -frac * W[:, j]``). Calibration takes the smallest k of ``NEURON_LADDER`` whose full ablation reaches
the effect point, then matches ``frac``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from .selector import BATCH, provenance, source_identity, stored

Tensor = torch.Tensor

NEURON_LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256)  # top-k neurons ablated together
RULE = ("P(c_fc output > 0 | target draw) - P(c_fc output > 0 | English retain pool), over all input positions; "
        "ties by layer, then index")


@torch.no_grad()
def activation_rates(model: nn.Module, tokens: Tensor, device, bs: int = BATCH) -> Tensor:
    """``[n_layer, d_ff]`` fraction of input positions with a positive ``c_fc`` output (float64, CPU)."""
    n_layer = len(model.h)
    acts: dict[int, Tensor] = {}
    hs = [model.h[L].mlp.c_fc.register_forward_hook(lambda _m, _i, o, L=L: acts.__setitem__(L, o))
          for L in range(n_layer)]
    tot, n = None, 0
    try:
        for s in range(0, tokens.shape[0], bs):
            w = tokens[s:s + bs].to(device)
            model(w[:, :-1])
            cnt = torch.stack([(acts[L] > 0).reshape(-1, acts[L].shape[-1]).double().sum(0).cpu()
                               for L in range(n_layer)])
            tot = cnt if tot is None else tot + cnt
            n += w[:, :-1].numel()
    finally:
        for h in hs:
            h.remove()
    return tot / n


def ranked_neurons(score: Tensor, top: int) -> list[tuple[int, int, float]]:
    """``(layer, index, score)`` in descending score. Ties by layer, then index."""
    flat = [(-float(score[L, j]), L, j) for L in range(score.shape[0]) for j in range(score.shape[1])]
    flat.sort()
    return [(L, j, -s) for s, L, j in flat[:top]]


def neuron_ranking(model: nn.Module, target: Tensor, english: Tensor, device, english_cache: str | Path,
                   bs: int = BATCH) -> dict:
    """The ranked top ``max(NEURON_LADDER)`` neurons of one target draw. The English rates are stored at
    ``english_cache`` and reused only under equal provenance, which includes this file and the model forward."""
    t0 = time.time()
    code = source_identity(__file__, Path(__file__).parent.parent / "model.py")
    prov = provenance(model, device, bs, {"english": english}, code)
    p_en = stored(english_cache, prov, lambda: {"rates": activation_rates(model, english, device, bs)})["rates"]
    p_t = activation_rates(model, target, device, bs)
    score = p_t - p_en
    top = ranked_neurons(score, max(NEURON_LADDER))
    return {"rule": RULE,
            "ranked": [{"layer": L, "neuron": j, "score": s, "p_target": float(p_t[L, j]), "p_en": float(p_en[L, j])}
                       for L, j, s in top],
            "per_layer_in_top64": [sum(1 for L, _, _ in top[:64] if L == k) for k in range(score.shape[0])],
            "english_provenance": prov, "seconds": time.time() - t0}


def neuron_update(model: nn.Module, neurons: Sequence[tuple[int, int]], frac: float, device) -> dict[str, Tensor]:
    """``Q[:, j] = -frac * W_down[:, j]`` in ``h.L.mlp.down_proj`` for every neuron ``(L, j)``: its output column
    scaled by ``1 - frac`` (frac 1 = ablation)."""
    out: dict[str, Tensor] = {}
    for L, j in neurons:
        m = f"h.{L}.mlp.down_proj"
        W = model.get_submodule(m).weight
        if m not in out:
            out[m] = torch.zeros_like(W, device=device)
        out[m][:, j] = -float(frac) * W[:, j].detach().to(device)
    return out
