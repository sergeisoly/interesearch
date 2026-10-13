"""Fitting of repair candidates: the rank-one families, their training, the full-batch fit loss and the polish of
the strengths family (papers/repair-transfer/index.qmd#sec-setup).

A repair is an explicit edit ``E`` of the 24 modules on top of the fine-tuned target, one rank-one term per module.
Families:

    strengths          E_m = c_m b_m a_m^T on the donor repair's frozen factors. The 24 c_m are trained from c0
    donor_init         E_m = b_m a_m^T from b = c0 b_donor, a = a_donor, both trained (the strengths start exactly)
    donor_init_output  as donor_init with a frozen: only the output (write) vectors b are trained
    random_init        donor_init from the norm-matched randomized donor (seed 20261011, random_init_seed2: 20261012)
    fresh              b = 0, a ~ N(0, 1/d_in) with generator seed 1 (fresh_seed2: seed 2)

with c0 = min(1, cap / ||(b_m a_m^T)_m||_F) and cap = ||D||_F of the fine-tune. Loss of every family, on fit
windows of German (de), Dutch (nl) and the four retain languages::

    KL(p_D || p_R) on de + KL(p_W0 || p_R) on nl + KL(p_D || p_R) on retain

each a mean over all predicted tokens of its group. Adam (0.9, 0.999), linear warm-up over 10 steps, grad clip 1.0.
After every step ``E`` is projected onto ``||E||_F <= cap`` (one common factor). Every step draws its batch from one
generator (seed 0) in the order de, nl, en, fr, es, it.

Two forward paths are kept apart. Training runs ``(W_0 + D) + E`` through ``functional_call`` with ``E`` in float32
as the parameters build it. Evaluation and export run ``W_0 + (D + E)`` through ``edits.applied_update``, with ``E``
built per module in float64 and cast to float32 before it is added to ``D`` (``update_of``). Their agreement is
checked against ``FIT_PATH_TOL`` (fit loss) and ``evaluate.PARITY_TOL`` (logits).

A candidate is a compact spec. Its dense ``E`` exists only inside ``E64``/``update_of``/``fro_of``, one module at a
time:

    {"kind": "rank_one", "factors": {m: (b, a)}}                 E_m = b a^T
    {"kind": "strengths", "c": [24], "factors": {m: (b, a)}}     E_m = c_m b a^T
    {"kind": "scaled", "k": k, "factors": {m: (b, a)}}           E_m = k b a^T (the donor repair, raw or capped)
    {"kind": "zero"}                                             E = 0 (the fine-tuned model itself)
"""

from __future__ import annotations

import math
import time
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from ..assets import MODULES
from ..edits import applied_update, logits_of

Tensor = torch.Tensor
Factors = dict[str, tuple[Tensor, Tensor]]

RETAIN = ("en", "fr", "es", "it")
GROUPS = ("de", "nl", "ret")
BATCH = {"de": 8, "nl": 8, "ret_per_lang": 2}
# Fit budgets in unique predicted tokens: (windows, tokens per window) per group. Every budget takes prefixes (rows
# [:n], tokens [:L+1]) of the same fit windows, so the smaller budgets are nested in the larger ones.
BUDGETS = {
    "256": {"nl": (1, 128), "de": (1, 64), "ret": (1, 16)},   # 128 + 64 + 4*16
    "512": {"nl": (1, 256), "de": (1, 128), "ret": (1, 32)},  # 256 + 128 + 4*32
    "2K": {"nl": (4, 256), "de": (2, 256), "ret": (1, 128)},  # 1024 + 512 + 4*128
}
WARMUP = 10
SAMPLER_SEED = 0
INIT_SEED = {"fresh": 1, "fresh_seed2": 2}
RAND_SEED = {"random_init": 20261011, "random_init_seed2": 20261012}
LR_FACTOR = (1e-4, 3e-4, 1e-3, 3e-3)
LRS = {"strengths": (3e-3, 1e-2, 3e-2, 1e-1), **{f: LR_FACTOR for f in (
    "donor_init", "donor_init_output", "random_init", "fresh", "fresh_seed2", "random_init_seed2")}}
# |fit loss on the training path - fit loss on the export path| at the polish start, nat/token: the two paths differ
# only by the float64 -> float32 rounding of E and the order of the weight sum.
FIT_PATH_TOL = 1e-5
# ||E||_F / cap after the projection may exceed 1 by float rounding of the radial scaling only.
CAP_TOL = 1e-6
# Agreement of an initialization with the strengths start (relative max |diff| of E, or of ||E||).
INIT_TOL = 1e-6
POLISH = {"max_iter": 300, "armijo": 1e-4, "rel_tol": 1e-7, "patience": 5, "pg_tol": 1e-4, "alpha_min": 1e-10,
          "lam_range": 1e6, "lam0_frac": 0.1, "floor_stationarity": 1e-3}


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# --------------------------------------------------------------------------- fit windows
def fit_sets(windows: Mapping[str, Tensor], budget: str | None) -> dict:
    """``{'de': [n, L+1], 'nl': [...], 'ret': {lang: [...]}}``: the budget's prefixes of the fit windows, or every
    window whole for ``budget=None``. Fit windows shorter than the budget's prefixes are refused (a shortage would
    otherwise be cut silently under the budget's name)."""
    if budget is None:
        return {"de": windows["de"], "nl": windows["nl"], "ret": {lg: windows[lg] for lg in RETAIN}}
    B = BUDGETS[budget]
    need = {"de": B["de"], "nl": B["nl"], **{lg: B["ret"] for lg in RETAIN}}
    short = {lg: tuple(windows[lg].shape) for lg, (n, L) in need.items()
             if windows[lg].shape[0] < n or windows[lg].shape[1] < L + 1}
    if short:
        raise ValueError(f"fit windows {short} are shorter than the prefixes of budget {budget} {B}")
    out = {g: windows[g][:B[g][0], :B[g][1] + 1] for g in ("de", "nl")}
    n, L = B["ret"]
    out["ret"] = {lg: windows[lg][:n, :L + 1] for lg in RETAIN}
    return out


def fit_accounting(fs: dict) -> dict:
    pred = {g: int(fs[g].shape[0] * (fs[g].shape[1] - 1)) for g in ("de", "nl")}
    pred["ret"] = int(sum(t.shape[0] * (t.shape[1] - 1) for t in fs["ret"].values()))
    windows = {g: int(fs[g].shape[0]) for g in ("de", "nl")}
    windows["ret"] = {lg: int(t.shape[0]) for lg, t in fs["ret"].items()}
    nwin = windows["de"] + windows["nl"] + sum(windows["ret"].values())
    return {"unique_predicted_tokens": pred, "total_unique_predicted_tokens": sum(pred.values()),
            "total_unique_input_tokens": sum(pred.values()) + nwin, "windows": windows}


# --------------------------------------------------------------------------- parameterisations
def randomized(donor: Factors, seed: int) -> Factors:
    """The donor's factors replaced by Gaussian directions of the same norms (float64 draws, b then a per module)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m in MODULES:
        b, a = donor[m]
        gb = torch.randn(b.shape[0], generator=g, dtype=torch.float64)
        ga = torch.randn(a.shape[0], generator=g, dtype=torch.float64)
        out[m] = ((gb / gb.norm() * b.double().norm()).float(), (ga / ga.norm() * a.double().norm()).float())
    return out


def c0_of(factors: Factors, cap: float) -> float:
    """The strengths start factor min(1, cap / ||(b_m a_m^T)_m||_F), norms in float64."""
    n0 = math.sqrt(sum(float(b.double().norm() ** 2 * a.double().norm() ** 2) for b, a in factors.values()))
    return min(1.0, cap / n0) if n0 > 0 else 0.0


class Method:
    """``E`` (per module, float32 on the device) as a function of the trainable parameters ``params``.

    ``rank_one``: E_m = b_m a_m^T with vectors ``b``, ``a``. ``strengths``: E_m = c_m b_m a_m^T with the frozen
    factors ``fb``, ``fa`` and the 24 strengths ``c``. ``c0`` is the start factor of a donor-based start (None for
    fresh)."""

    def __init__(self, kind: str, cap: float, params: list[Tensor], c0: float | None = None):
        self.kind, self.cap, self.params, self.c0 = kind, cap, params, c0

    @classmethod
    def rank_one(cls, b: dict[str, Tensor], a: dict[str, Tensor], cap: float, c0: float | None = None,
                 train_a: bool = True) -> "Method":
        params = [t for m in MODULES for t in (b[m], a[m])] if train_a else [b[m] for m in MODULES]
        meth = cls("rank_one", cap, params, c0)
        meth.b, meth.a = b, a
        return meth

    @classmethod
    def strengths(cls, donor: Factors, cap: float, device) -> "Method":
        """Strengths on the donor's factors from c0 (computed from the factors on the device)."""
        fb = {m: donor[m][0].to(device) for m in MODULES}
        fa = {m: donor[m][1].to(device) for m in MODULES}
        fn2 = {m: float(fb[m].double().norm() ** 2 * fa[m].double().norm() ** 2) for m in MODULES}
        n0 = math.sqrt(sum(fn2.values()))
        c0 = min(1.0, cap / n0) if n0 > 0 else 0.0
        c = torch.full((len(MODULES),), c0, device=device, requires_grad=True)
        meth = cls("strengths", cap, [c], c0)
        meth.fb, meth.fa, meth.fn2, meth.c = fb, fa, fn2, c
        return meth

    def E(self) -> dict[str, Tensor]:
        if self.kind == "rank_one":
            return {m: torch.outer(self.b[m], self.a[m]) for m in MODULES}
        return {m: self.c[i] * torch.outer(self.fb[m], self.fa[m]) for i, m in enumerate(MODULES)}

    def norm(self) -> float:
        with torch.no_grad():
            if self.kind == "rank_one":
                return math.sqrt(sum(float(self.b[m].double().norm() ** 2 * self.a[m].double().norm() ** 2)
                                     for m in MODULES))
            return math.sqrt(sum(float(self.c[i].double()) ** 2 * self.fn2[m] for i, m in enumerate(MODULES)))

    @torch.no_grad()
    def project(self) -> tuple[float, float]:
        """Radial projection onto ``||E||_F <= cap``. Returns the norm before and after."""
        n = self.norm()
        if n <= self.cap or n == 0:
            return n, n
        k = self.cap / n
        if self.kind == "rank_one":
            for m in MODULES:
                self.b[m].mul_(k)
        else:
            self.c.mul_(k)
        return n, self.norm()

    @torch.no_grad()
    def snapshot(self) -> dict:
        """The trainable state on the CPU: ``{'factors': {m: (b, a)}}`` or ``{'c': [24]}``."""
        if self.kind == "rank_one":
            return {"factors": {m: (self.b[m].detach().cpu().clone(), self.a[m].detach().cpu().clone())
                                for m in MODULES}}
        return {"c": self.c.detach().cpu().clone()}

    def spec(self, snap: dict, donor: Factors | None = None) -> dict:
        """The candidate spec of a snapshot of this method (strengths need the donor's factors)."""
        if self.kind == "rank_one":
            return {"kind": "rank_one", "factors": snap["factors"]}
        return {"kind": "strengths", "c": snap["c"], "factors": donor}


def make_method(family: str, shapes: Mapping[str, tuple[int, int]], device, cap: float, donor: Factors,
                randomized_donors: Mapping[str, Factors]) -> Method:
    """A family's parameterisation at its start. ``randomized_donors``: random_init and random_init_seed2."""
    if family == "strengths":
        return Method.strengths(donor, cap, device)
    if family in ("fresh", "fresh_seed2"):
        gd = torch.Generator().manual_seed(INIT_SEED[family])
        b = {m: torch.zeros(shapes[m][0], device=device, requires_grad=True) for m in MODULES}
        a = {}
        for m in MODULES:
            d_in = shapes[m][1]
            a[m] = (torch.randn(d_in, generator=gd) / math.sqrt(d_in)).to(device).requires_grad_(True)
        return Method.rank_one(b, a, cap)
    src = donor if family in ("donor_init", "donor_init_output") else randomized_donors[family]
    c0 = c0_of(src, cap)  # from the factors on the CPU
    train_a = family != "donor_init_output"
    b = {m: (c0 * src[m][0].double()).float().to(device).requires_grad_(True) for m in MODULES}
    a = {m: src[m][1].clone().to(device).requires_grad_(train_a) for m in MODULES}
    return Method.rank_one(b, a, cap, c0, train_a)


@torch.no_grad()
def init_check(family: str, meth: Method, E_start: Mapping[str, Tensor], cap: float) -> dict:
    """fresh: E = 0 exactly. donor_init(_output): E equals the strengths start c0 b a^T (relative max |diff| <=
    INIT_TOL). random_init(_seed2): ||E|| equals the strengths start's norm (relative INIT_TOL). ``E_start``: the
    strengths start in float64."""
    E = {m: e.detach().double() for m, e in meth.E().items()}
    if family in ("fresh", "fresh_seed2"):
        v = max(float(e.abs().max()) for e in E.values())
        return {"zero_edit_max_abs": v, "ok": v == 0.0}
    if family == "strengths":
        return {"c0": meth.c0, "norm_over_cap": meth.norm() / cap if cap > 0 else None, "ok": math.isfinite(meth.c0)}
    if family in ("donor_init", "donor_init_output"):
        diff = max(float((E[m] - E_start[m]).abs().max()) for m in E)
        scale = max(float(E_start[m].abs().max()) for m in E_start)
        v = diff / scale if scale > 0 else math.inf
        return {"c0": meth.c0, "max_rel_diff_vs_strengths_start": v, "ok": v <= INIT_TOL}
    n = math.sqrt(sum(float(e.norm() ** 2) for e in E.values()))
    nr = math.sqrt(sum(float(e.norm() ** 2) for e in E_start.values()))
    v = abs(n / nr - 1.0) if nr > 0 else math.inf
    return {"c0": meth.c0, "norm_ratio_minus_1_vs_strengths_start": v, "ok": v <= INIT_TOL}


# --------------------------------------------------------------------------- candidate specs
def E64(spec: dict, m: str, i: int, Dd: Mapping[str, Tensor], device) -> Tensor:
    """E_m of one candidate, float64 on the device."""
    kind = spec["kind"]
    if kind == "zero":
        return torch.zeros(Dd[m].shape, dtype=torch.float64, device=device)
    b, a = spec["factors"][m]
    E = torch.outer(b.to(device).double(), a.to(device).double())
    if kind == "rank_one":
        return E
    if kind == "strengths":
        return float(spec["c"][i]) * E
    if kind == "scaled":
        return spec["k"] * E
    raise ValueError(kind)


@torch.no_grad()
def update_of(spec: dict, Dd: Mapping[str, Tensor], device) -> dict[str, Tensor]:
    """The float32 update ``D + E`` of one candidate on the device (E in float64, cast before the sum)."""
    upd = {}
    for i, m in enumerate(MODULES):
        e = E64(spec, m, i, Dd, device)
        upd[m] = Dd[m] + e.to(torch.float32)
        del e
    return upd


@torch.no_grad()
def fro_of(spec: dict, Dd: Mapping[str, Tensor], device) -> float:
    """``||E||_F`` over the 24 modules (float64, one module at a time, Python's sum over the modules)."""
    return math.sqrt(sum(float(E64(spec, m, i, Dd, device).norm() ** 2) for i, m in enumerate(MODULES)))


def nonfinite_snapshot(snap: dict) -> bool:
    if "c" in snap:
        return not bool(torch.isfinite(snap["c"]).all())
    return not all(bool(torch.isfinite(b).all()) and bool(torch.isfinite(a).all()) for b, a in snap["factors"].values())


# --------------------------------------------------------------------------- training
def _kl(lp_t: Tensor, lp_s: Tensor) -> Tensor:
    """KL(p_t || p_s), mean over every predicted token of the batch."""
    return (lp_t.exp() * (lp_t - lp_s)).sum(-1).mean()


def train(model: nn.Module, base: Mapping[str, Tensor], D: Mapping[str, Tensor], meth: Method, fs: dict, lr: float,
          device, log: Callable[[str], None], tag: str, steps: int, every: int) -> tuple[dict, list, dict]:
    """Train one family at one learning rate. Returns (snapshots by step, one log row per step, status). A
    non-finite loss or gradient stops the run before its update."""
    g = torch.Generator().manual_seed(SAMPLER_SEED)
    WD = {m: (base[m] + D[m].to(device)).detach() for m in MODULES}
    Dd = {m: D[m].to(device) for m in MODULES}
    opt = torch.optim.Adam(meth.params, lr=lr, betas=(0.9, 0.999))
    ckpts, logs, status = {}, [], {"finite": True, "steps_planned": steps}
    nb = {"de": min(BATCH["de"], fs["de"].shape[0]), "nl": min(BATCH["nl"], fs["nl"].shape[0]),
          "ret": min(BATCH["ret_per_lang"], fs["ret"]["en"].shape[0])}
    nw = {"de": int(fs["de"].shape[0]), "nl": int(fs["nl"].shape[0]), "ret": int(fs["ret"]["en"].shape[0])}
    status["batch"], status["unique_windows"] = nb, nw
    status["batches_presented"] = 0  # forward/backward passes made, a non-finite last one included

    def lp_teacher(update, toks):
        with torch.no_grad(), applied_update(model, base, update):
            return F.log_softmax(logits_of(model(toks[:, :-1])).float(), -1)

    def pick(t: Tensor, n: int) -> Tensor:
        return t[torch.randperm(t.shape[0], generator=g)[:min(n, t.shape[0])]]

    for step in range(1, steps + 1):
        de = pick(fs["de"], BATCH["de"]).to(device)
        nl = pick(fs["nl"], BATCH["nl"]).to(device)
        ret = torch.cat([pick(fs["ret"][lg], BATCH["ret_per_lang"]) for lg in RETAIN]).to(device)
        status["batches_presented"] = step
        t_de, t_ret, t_nl = lp_teacher(Dd, de), lp_teacher(Dd, ret), lp_teacher(None, nl)
        E = meth.E()
        out = {}
        for key, toks, t in (("de", de, t_de), ("nl", nl, t_nl), ("ret", ret, t_ret)):
            weights = {f"{m}.weight": WD[m] + E[m] for m in MODULES}
            z = logits_of(functional_call(model, weights, (toks[:, :-1],))).float()
            out[key] = _kl(t, F.log_softmax(z, -1))
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, step / WARMUP)
        loss = out["de"] + out["nl"] + out["ret"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(meth.params, 1.0)
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            status["finite"] = False
            status["nonfinite_step"] = step
            log(f"{tag} lr={lr:g}: non-finite loss/grad at step {step}. Run stopped")
            break
        opt.step()
        n_pre, n_post = meth.project()
        logs.append({"step": step, "loss": float(loss.detach()), **{k: float(v.detach()) for k, v in out.items()},
                     "grad_norm": float(gn), "norm_pre": n_pre, "norm_post": n_post})
        if step % every == 0:
            ckpts[step] = meth.snapshot()
            L = logs[-1]
            log(f"{tag} lr={lr:g} step {step}: loss {L['loss']:.5f} (de {L['de']:.5f} nl {L['nl']:.5f} "
                f"ret {L['ret']:.5f}) |E| {n_post:.5f}/{meth.cap:.5f}")
    status["updates_applied"] = len(logs)
    status["presentations_per_unique_window"] = {k: status["batches_presented"] * nb[k] / nw[k] for k in nb}
    if logs:
        status["first_loss"], status["last_loss"] = logs[0]["loss"], logs[-1]["loss"]
        status["max_post_over_cap"] = max(r["norm_post"] for r in logs) / meth.cap if meth.cap > 0 else None
    return ckpts, logs, status


# --------------------------------------------------------------------------- full-batch fit loss and polish
class FitLoss:
    """The training loss on a budget's whole fit set (at these budgets every step presents every window, and the
    order of rows does not change the mean), with the teachers computed once.

    ``train_path(E)``: ``functional_call`` on ``(W_0 + D) + E`` as in ``train`` (differentiable).
    ``export(spec)``: ``update_of`` and ``applied_update`` as in ``evaluate``, per group and in total."""

    def __init__(self, model: nn.Module, base: Mapping[str, Tensor], D: Mapping[str, Tensor], fs: dict, device):
        self.model, self.base, self.device = model, base, device
        self.Dd = {m: D[m].to(device) for m in MODULES}
        self.WD = {m: (base[m] + self.Dd[m]).detach() for m in MODULES}
        self.groups = {"de": fs["de"].to(device), "nl": fs["nl"].to(device),
                       "ret": torch.cat([fs["ret"][lg] for lg in RETAIN]).to(device)}
        self.t = {}
        with torch.no_grad():
            for g, toks in self.groups.items():
                with applied_update(model, base, None if g == "nl" else self.Dd):
                    self.t[g] = F.log_softmax(logits_of(model(toks[:, :-1])).float(), -1)

    def _kl(self, g: str, z: Tensor) -> Tensor:
        return _kl(self.t[g], F.log_softmax(z, -1))

    def train_path(self, E: Mapping[str, Tensor]) -> dict[str, Tensor]:
        out = {}
        for g, toks in self.groups.items():
            w = {f"{m}.weight": self.WD[m] + E[m] for m in MODULES}
            out[g] = self._kl(g, logits_of(functional_call(self.model, w, (toks[:, :-1],))).float())
        return out

    @torch.no_grad()
    def export(self, spec: dict) -> dict[str, float]:
        upd = update_of(spec, self.Dd, self.device)
        out = {}
        for g, toks in self.groups.items():
            with applied_update(self.model, self.base, upd):
                z = logits_of(self.model(toks[:, :-1])).float()
            out[g] = float(self._kl(g, z))
        del upd
        out["total"] = out["de"] + out["nl"] + out["ret"]
        return out


def polish(fl: FitLoss, meth: Method, c_start: Tensor, cap: float, max_iter: int, log: Callable[[str], None],
           tag: str, ref: dict | None = None) -> tuple[dict, dict | None]:
    """Spectral projected gradient (Barzilai-Borwein step, monotone Armijo backtracking) on the full-batch fit loss of
    strengths ``meth``, in e_m = c_m ||b_m|| ||a_m|| coordinates (||E||_F = ||e||_2, so the cap is a ball and the
    radial scaling is its exact projection). Returns (record, snapshot ``{'c': [24]}`` or None if not finite).

    converged: finite, path parity <= FIT_PATH_TOL and a stop on the projected-gradient or relative-decrease test,
    or on the line-search floor with stationarity_end <= floor_stationarity x stationarity_ref. finite is False when
    any loss or gradient the polish computed, a line-search trial included, is non-finite. There is then no
    snapshot. ``ref``: ``{'stat', 'lam0'}`` of the polish from c0 (run first), so that both polishes measure
    stationarity on one scale: the projected-gradient norm at step lam0, relative to its value at c0. Without
    ``ref`` the run's own start is the reference."""
    fn = torch.tensor([math.sqrt(meth.fn2[m]) for m in MODULES], dtype=torch.float64)
    rec = {"tag": tag, "params": POLISH, "max_iter": max_iter, "finite": True, "trace": []}
    factors = {m: (meth.fb[m].detach().cpu(), meth.fa[m].detach().cpu()) for m in MODULES}
    if not bool((fn > 0).all()):
        rec.update(finite=False, stop="a donor factor has zero norm", converged=False)
        return rec, None

    def c_of(e):
        return (e / fn).to(torch.float32)

    def f(e, grad: bool):
        with torch.enable_grad() if grad else torch.no_grad():
            cc = c_of(e).to(fl.device).requires_grad_(grad)
            E = {m: cc[i] * torch.outer(meth.fb[m], meth.fa[m]) for i, m in enumerate(MODULES)}
            comps = fl.train_path(E)
            L = comps["de"] + comps["nl"] + comps["ret"]
            if not grad:
                return float(L), None
            L.backward()
            return float(L.detach()), cc.grad.detach().double().cpu() / fn

    def proj(e):
        n = float(e.norm())
        return e * (cap / n) if n > cap else e

    t0 = time.time()
    e = proj(c_start.detach().double().cpu() * fn)
    L, g = f(e, True)
    evals = 1
    rec["L_start"] = L
    L_exp = fl.export({"kind": "strengths", "c": c_of(e), "factors": factors})["total"]
    rec["path_parity_abs"] = abs(L - L_exp) if math.isfinite(L) and math.isfinite(L_exp) else math.inf
    if not (math.isfinite(L) and bool(torch.isfinite(g).all())):
        rec.update(finite=False, stop="non-finite loss or gradient at the start", converged=False,
                   seconds=time.time() - t0)
        return rec, None
    gn0 = float(g.norm())
    lam0 = POLISH["lam0_frac"] * max(float(e.norm()), 1e-12) / gn0 if gn0 > 0 else 1.0
    if ref is not None and finite(ref.get("lam0")) and ref["lam0"] > 0 and finite(ref.get("stat")):
        lam0 = ref["lam0"]
    else:
        ref = None
    lam_lo, lam_hi = lam0 / POLISH["lam_range"], lam0 * POLISH["lam_range"]

    def pg(e_, g_):  # projected-gradient norm at the scale lam0 (= ||g|| strictly inside the ball)
        return float((proj(e_ - lam0 * g_) - e_).norm()) / lam0

    rec["stationarity_start"] = pg(e, g)
    rec["lam0"] = lam0
    rec["stationarity_ref"] = s_ref = ref["stat"] if ref is not None else rec["stationarity_start"]
    rec["ref"] = "c0 polish" if ref is not None else "own start"
    lam, it, small, stop, rel = lam0, 0, 0, "max_iter", math.nan
    while it < max_iter:
        if s_ref == 0 or pg(e, g) <= POLISH["pg_tol"] * s_ref:
            stop = "pg_tol"
            break
        d = proj(e - lam * g) - e
        gd = float(g.dot(d))
        alpha, ok = 1.0, False
        while alpha >= POLISH["alpha_min"]:
            L_try, _ = f(e + alpha * d, False)
            evals += 1
            if not math.isfinite(L_try):  # a numerical failure, not a rejected step
                rec["finite"] = False
                rec["nonfinite_trial"] = {"iter": it + 1, "alpha": alpha, "L_try": L_try}
                break
            if L_try <= L + POLISH["armijo"] * alpha * gd:
                ok = True
                break
            alpha *= 0.5
        if not rec["finite"]:
            stop = "non-finite loss in the line search"
            break
        if not ok:
            stop = "line_search_floor"  # no sufficient decrease at float precision along the projected direction
            break
        e_new = e + alpha * d
        it += 1
        L_new, g_new = f(e_new, True)
        evals += 1
        if not (math.isfinite(L_new) and bool(torch.isfinite(g_new).all())):
            rec["finite"] = False
            stop = "non-finite loss or gradient"
            break
        rel = (L - L_new) / abs(L) if L != 0 else 0.0
        s, y = e_new - e, g_new - g
        sy = float(s.dot(y))
        lam = min(max(float(s.dot(s)) / sy, lam_lo), lam_hi) if sy > 0 else lam_hi
        e, L, g = e_new, L_new, g_new
        if it <= 3 or it % 10 == 0:
            rec["trace"].append({"iter": it, "L": L, "alpha": alpha, "lam": lam, "rel": rel})
        small = small + 1 if rel < POLISH["rel_tol"] else 0
        if small >= POLISH["patience"]:
            stop = "rel_tol"
            break
    c = c_of(e)
    rec.update(L_end=L, iters=it, evals=evals, stop=stop, rel_last=rel, stationarity_end=pg(e, g),
               norm_over_cap=float(e.norm()) / cap if cap > 0 else None, seconds=time.time() - t0)
    if not bool(torch.isfinite(c).all()):
        rec["finite"] = False
    floor_ok = stop == "line_search_floor" and rec["stationarity_end"] <= POLISH["floor_stationarity"] * s_ref
    rec["converged"] = bool(rec["finite"] and rec["path_parity_abs"] <= FIT_PATH_TOL
                            and (stop in ("pg_tol", "rel_tol") or floor_ok))
    log(f"polish strengths from {tag}: L {rec['L_start']:.6f} -> {L:.6f} in {it} iterations ({evals} evaluations), "
        f"stop {stop}, stationarity {rec['stationarity_start']:.3g} -> {rec['stationarity_end']:.3g}, |e|/cap "
        f"{rec['norm_over_cap']:.6f}, parity {rec['path_parity_abs']:.2e}, converged {rec['converged']}, "
        f"{rec['seconds']:.1f}s")
    return rec, ({"c": c.cpu().clone()} if rec["finite"] else None)
