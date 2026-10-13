"""Fine-tuning primitives shared by both studies.

Window convention: a window ``w`` has ``T+1`` tokens. The model reads ``w[:, :-1]`` and its ``T`` logits predict
``w[:, 1:]``. Dtypes and the order of casts and reductions are those of the original runs: CE and KL are computed in
float32 and reduced per window on the device. ``per_window_ce`` moves the result to float64 on the CPU.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .edits import logits_of

Tensor = torch.Tensor


def disable_tf32() -> dict:
    """Disable CUDA TF32 and return the precision flags."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


# Original training seed namespace, retained to reproduce the published data order.
TRAINING_SEED_SALT = "vpd-ft"


def seeded_generator(seed: int, purpose: str) -> torch.Generator:
    """A CPU generator per (seed, purpose). The original study salt preserves the published data order."""
    h = int(hashlib.sha256(f"{TRAINING_SEED_SALT}|{seed}|{purpose}".encode()).hexdigest(), 16) % (2**63 - 1)
    return torch.Generator().manual_seed(h)


class EpochSampler:
    """Draws row indices without replacement, reshuffling at each epoch boundary (deterministic in the generator)."""

    def __init__(self, n: int, generator: torch.Generator) -> None:
        self.n, self.gen, self.queue, self.epoch = int(n), generator, [], 0

    def take(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if not self.queue:
                self.queue = torch.randperm(self.n, generator=self.gen).tolist()
                self.epoch += 1
            out.append(self.queue.pop(0))
        return out


# --------------------------------------------------------------------------- matrix-only update
def _key(module: str) -> str:
    return module.replace(".", "-")


class DenseUpdate(nn.Module):
    """Matrix-only fine-tuning: one zero-initialised ``D`` per module, trained directly (``W = W_0 + D``)."""

    kind = "matrix"

    def __init__(self, shapes: Mapping[str, tuple[int, int]], dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.module_names = list(shapes)
        self.D = nn.ParameterDict({_key(m): nn.Parameter(torch.zeros(shapes[m], dtype=dtype)) for m in shapes})

    def delta(self, module: str) -> Tensor:
        return self.D[_key(module)]

    @torch.no_grad()
    def update(self) -> dict[str, Tensor]:
        """``D`` per module on the CPU, in the training dtype (float32 for the real model): exact, no subtraction."""
        return {m: self.D[_key(m)].detach().cpu().clone() for m in self.module_names}

    def state(self) -> dict:
        return {}


def merged_forward(model: nn.Module, base: Mapping[str, Tensor], upd: nn.Module) -> Callable[[Tensor], Tensor]:
    """Forward with ``W_0 + upd.delta(m)`` for every module in ``upd.module_names``, through ``functional_call``: the
    live weights are not written, and gradients reach the update."""

    def fwd(tokens: Tensor) -> Tensor:
        params = {f"{m}.weight": base[m] + upd.delta(m) for m in upd.module_names}
        return logits_of(functional_call(model, params, (tokens,)))

    return fwd


# --------------------------------------------------------------------------- window losses
def window_ce(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean next-token CE per window. ``logits [B, T, V]`` from ``w[:, :-1]``, ``targets = w[:, 1:]``."""
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction="none")
    return ce.view(targets.shape[0], -1).mean(dim=1)


def window_kl(lp_ref: Tensor, logits: Tensor) -> Tensor:
    """``KL(p_ref || p)`` averaged over positions, per window. ``lp_ref`` are log-probs ``[B, T, V]``."""
    lp = F.log_softmax(logits.float(), dim=-1)
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1).mean(-1)


def hinge(g: Tensor, margin: float) -> Tensor:
    """``max(0, margin - g)``: no incentive once a window's CE has risen by ``margin`` nats."""
    return torch.clamp(margin - g, min=0.0)


@torch.no_grad()
def per_window_ce(forward_fn: Callable[[Tensor], Tensor], tokens: Tensor, device,
                  batch_size: int = 8) -> Tensor:
    """``window_ce`` of every window in batches, float64 on the CPU (MPS has no float64)."""
    out = []
    for s in range(0, tokens.shape[0], batch_size):
        b = tokens[s:s + batch_size].to(device)
        out.append(window_ce(forward_fn(b[:, :-1]), b[:, 1:]).cpu().double())
    return torch.cat(out)
