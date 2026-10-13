"""Dictionaries for the two additional atom comparators, using the main selection rule.
Rules: papers/selective-editing/index.qmd#sec-extended-comparisons.

* ``svd_atom``: the rank-one SVD terms ``sigma_i u_i v_i^T`` of each weight. They reconstruct W exactly, whereas
  the VPD atoms leave a decomposition residual that VPD edits keep.
* ``whitened_rank1``: one atom per module, without a dictionary: the rank-one direction that maximizes the squared
  target gradient against the K-FAC curvature of the English retain pool. It borrows the Frobenius norm of the
  cell's VPD atom and the main selector's ridge floor eps. Its walk has its own grid (``WHITENED_GRID``, 1/64 to 8
  in units of that norm).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..edits import logits_of
from .dictionary import Dictionary, ModuleAtoms
from .selector import BATCH, FISHER_K, SELECTION_SALT, Capture, sample_labels, sha_int

Tensor = torch.Tensor

WHITENED_GRID = tuple(0.0625 * 2 ** (k / 2) for k in range(-4, 15))  # 0.0156 ... 8, x sqrt(2)
FACTORS_SEED = sha_int(f"{SELECTION_SALT}|kfac") % (2**31)  # label sets of the K-FAC output factor


def svd_dictionary(weights: Mapping[str, Tensor]) -> Dictionary:
    """Rank-one SVD atoms of each weight (float64, CPU), oriented as ``Dictionary`` (U ``[C, d_out]``, V
    ``[d_in, C]``). Seed name ``svd``."""
    tensors = {}
    for m, W in weights.items():
        U, sv, Vh = torch.linalg.svd(W.detach().double().cpu(), full_matrices=False)
        tensors[m] = ((U * sv).T, Vh.T)
    return Dictionary.from_tensors(tensors, {"name": "svd", "source": "SVD of the target weights"})


def kfac_factors(model: nn.Module, modules: Sequence[str], tokens: Tensor, device, K: int = FISHER_K,
                 seed: int = FACTORS_SEED, bs: int = BATCH) -> dict[str, tuple[Tensor, Tensor]]:
    """Per module and prediction: ``A = mean x x^T`` of its input and ``B = mean dy dy^T`` of the gradient at its
    output under K label sets sampled from the model (``selector.sample_labels``). float64, CPU."""
    A = dict.fromkeys(modules, 0.0)
    B = dict.fromkeys(modules, 0.0)
    n = 0
    with Capture(model, modules) as cap:
        for s in range(0, tokens.shape[0], bs):
            w = tokens[s:s + bs].to(device)
            nb, T = w.shape[0], w.shape[1] - 1
            lp = F.log_softmax(logits_of(model(w[:, :-1])), -1)
            probs = lp.detach().exp()
            for m in modules:
                x = cap.x[m].reshape(-1, cap.x[m].shape[-1]).double()
                A[m] = A[m] + x.T @ x
            for k in range(K):
                y = sample_labels(probs, range(s, s + nb), k, seed)
                (-lp.gather(-1, y.unsqueeze(-1)).sum()).backward(retain_graph=k < K - 1)
                for m in modules:
                    dy = cap.dy[m].reshape(-1, cap.dy[m].shape[-1]).double()
                    B[m] = B[m] + dy.T @ dy
            n += nb * T
            del lp, probs
    out = {m: ((A[m] / n).cpu(), (B[m] / (n * K)).cpu()) for m in modules}
    bad = [m for m, ab in out.items() if not all(bool(torch.isfinite(t).all()) for t in ab)]
    if bad:
        raise FloatingPointError(f"technical: non-finite K-FAC factors in {bad[:4]} ({len(bad)} module(s))")
    return out


def _inv_sqrt(M: Tensor) -> Tensor:
    e, Q = torch.linalg.eigh(M)
    return (Q * e.clamp_min(1e-300).rsqrt()) @ Q.T


def whitened_atom(G: Tensor, A: Tensor, B: Tensor, gamma: float) -> tuple[Tensor, Tensor, float]:
    """The rank-one maximizer of ``(u^T G v)^2 / ((u^T B' u)(v^T A' v))`` under factored Tikhonov damping of
    ``gamma`` (``A' = A + pi sqrt(gamma) I``, ``B' = B + sqrt(gamma)/pi I``, ``pi^2 = (tr A / d_in) / (tr B /
    d_out)``, Martens and Grosse 2015, sec. 6.3): the top singular pair of ``B'^-1/2 G A'^-1/2``, mapped back.
    Unit ``u``, ``v`` and the maximum."""
    d_out, d_in = G.shape
    pi = math.sqrt((float(A.trace()) / d_in) / (float(B.trace()) / d_out))
    ra = _inv_sqrt(A + pi * math.sqrt(gamma) * torch.eye(d_in, dtype=A.dtype))
    rb = _inv_sqrt(B + math.sqrt(gamma) / pi * torch.eye(d_out, dtype=B.dtype))
    Uw, sw, Vw = torch.linalg.svd(rb @ G @ ra, full_matrices=False)
    u, v = rb @ Uw[:, 0], ra @ Vw[0]
    return u / u.norm(), v / v.norm(), float(sw[0]) ** 2


def whitened_dictionary(G: Mapping[str, Tensor], factors: Mapping[str, tuple[Tensor, Tensor]], gamma: float,
                        norm: float, name: str) -> Dictionary:
    """One atom per module at Frobenius norm ``norm``. ``name`` (``kfac_<target>_d<draw>``) seeds its q."""
    mods, best = {}, {}
    for m, (A, B) in factors.items():
        u, v, best[m] = whitened_atom(G[m], A, B, gamma)
        mods[m] = ModuleAtoms(m, (norm * u)[None, :], v[:, None], torch.ones(1, dtype=torch.bool))
    return Dictionary(mods, {m: [] for m in mods},
                      {"name": name, "source": "K-FAC whitened target gradient", "whitened_max": best})
