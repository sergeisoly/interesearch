"""Date bootstrap, intervals and finiteness checks shared by every cell and contrast of a target.
Rules: papers/repair-transfer/index.qmd#sec-app-methods.

One bootstrap draw resamples the readout dates jointly for every cell and metric of a target (5,000 draws, seed
10610). Intervals are 95% percentile intervals. A non-finite draw is never dropped: the interval is then
(nan, nan) and the caller marks the cell or contrast technical_invalid.
"""

from __future__ import annotations

import math

import torch

CHECKED = ("r_nl", "r_kl", "rho_de", "phi_de", "d_nl", "d_de")


def boot_indices(n: int, n_boot: int, seed: int) -> torch.Tensor:
    """[n_boot, n] resampled date rows. One draw resamples dates jointly for every cell and metric."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.arange(n)[torch.randint(0, n, (n_boot, n), generator=g)]


def ci(v: torch.Tensor, quantiles: tuple[float, float] | list[float]) -> tuple[float, float]:
    """Percentile interval at the recorded quantiles. Any non-finite draw makes it (nan, nan)."""
    if v.numel() == 0 or not bool(torch.isfinite(v).all()):
        return (math.nan, math.nan)
    return (float(torch.quantile(v, quantiles[0])), float(torch.quantile(v, quantiles[1])))


def r_nl_draws(per: dict, name: str, rows: torch.Tensor) -> torch.Tensor:
    W0, D, X = per["base"]["nl"]["ce"], per["finetuned"]["nl"]["ce"], per[name]["nl"]["ce"]
    return 1.0 - (X[rows] - W0[rows]).mean(1) / (D[rows] - W0[rows]).mean(1)


def nonfinite(per: dict, names) -> list[str]:
    return [f"non-finite values: {n}/{lg}/{k}" for n in names for lg, v in per[n].items() for k, t in v.items()
            if not bool(torch.isfinite(t).all())]


def nonfinite_point(p: dict) -> list[str]:
    """Point estimates that are not finite. Only r_nl with zero fine-tune damage is excused: it is undefined
    there, and the damage floor already marks the target not evaluable."""
    bad = [k for k in CHECKED if not math.isfinite(p[k]) and not (k == "r_nl" and p["d_nl_D"] == 0)]
    bad += [f"retain[{g}]" for g, v in p["retain"].items() if not math.isfinite(v)]
    return [f"non-finite estimate: {', '.join(bad)}"] if bad else []
