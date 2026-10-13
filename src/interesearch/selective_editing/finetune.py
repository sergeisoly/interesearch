"""The learned baselines and the historical LoRA condition, trained on a target draw with English retention only.
Rules: papers/selective-editing/index.qmd#sec-comparators (learned baselines, reference rows) and
papers/selective-editing/index.qmd#sec-access.

* ``matrix``: full-matrix fine-tune of the 24 modules (``finetune_24``, and the update ``omp_finetune`` reads).
* ``lora``: rank-1 LoRA in one module, the module of the selected atom of the same target and draw
  (``lora_rank1``).
* the historical LoRA condition, read only by ``omp_lora``: rank-1 LoRA on the 24 modules, lr 1e-4, seed = draw,
  checkpoints 50/100/150/200 and the historical checkpoint rule (``checkpoint_rule``).

Objective: ``max(0, 1 - dCE)`` per target window plus ``lambda KL(p_0 || p)`` per English retain window, lambda 1.
Adam, linear warm-up over 10 steps, gradient clip 1.0, 8 target and 8 retain windows per step in micro-batches of
4, 200 steps. The data order and the LoRA initialisation come from CPU generators seeded by (seed, purpose). The
matrix and LoRA baselines use seed 100 + draw. The learning rate is chosen once per target and condition on draw 0
(``choose_lr``). The dose is set later by scaling the final update into (0, 1].
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..checksums import sha256_file
from ..edits import assert_frozen, capture_base, logits_of, snapshot_params
from ..training import (DenseUpdate, EpochSampler, _key, hinge, merged_forward, per_window_ce, seeded_generator,
                        window_kl)
from .calibration import MAX_BISECT, TOL, scan_and_match
from .data import write_json
from .measure import ce_docs, edit_readout, update_norm
from .selector import TAUS, tokens_identity

Tensor = torch.Tensor

LEARNING_RATES = {"matrix": (1e-6, 3e-6, 1e-5), "lora": (1e-4, 3e-4, 1e-3, 3e-3)}
STEPS = 200
SEED_OFFSET = 100  # matrix and LoRA baselines: seed = 100 + draw
SCALE_GRID = tuple(0.0625 * 2 ** (k / 2) for k in range(9))  # 0.0625 ... 1.0: a learned update is never amplified
VAL_BATCH = 16
HISTORICAL = "historical_lora"
HISTORICAL_LR = 1e-4
CHECKPOINT_BAND = (0.5, 1.5)  # usable target validation rise at a checkpoint
CHECKPOINT_PREFERRED = 1.0
CHECKPOINT_RETAIN_MAX = 0.1
CHECKPOINT_RULE = (f"usable: target val rise in [{CHECKPOINT_BAND[0]}, {CHECKPOINT_BAND[1]}] and English val rise <= "
                   f"{CHECKPOINT_RETAIN_MAX}; closest to {CHECKPOINT_PREFERRED}, then lower English KL, then earlier "
                   "step")
CHOICE_RULE = "matched at the upper effect point on target val; lowest English val KL; then smaller lr"


@dataclass
class FineTuneConfig:
    condition: str  # "matrix" or "lora"
    lr: float
    seed: int
    steps: int = STEPS
    warmup: int = 10
    checkpoints: tuple[int, ...] = (50, 100, 150, 200)
    lam: float = 1.0  # weight of the retention KL
    margin: float = 1.0  # hinge cap in nats per target window
    n_target: int = 8  # target windows per step
    n_retain: int = 8  # English retain windows per step
    micro_batch: int = 4  # windows per forward pass
    rank: int = 1
    alpha: float | None = None  # LoRA alpha. None: alpha = rank
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.checkpoints = tuple(sorted({int(s) for s in self.checkpoints if 0 < int(s) <= self.steps}))
        if self.steps not in self.checkpoints:
            self.checkpoints += (self.steps,)

    @property
    def lora_scale(self) -> float:
        return float(self.alpha if self.alpha is not None else self.rank) / self.rank

    def to_dict(self) -> dict:
        return dataclasses.asdict(self) | {"lora_scale": self.lora_scale}


def config(condition: str, draw: int, lr: float | None = None, reduced: bool = False) -> FineTuneConfig:
    """The protocol's configuration of ``condition`` (``matrix``, ``lora`` or ``historical_lora``) for one draw.
    ``reduced``: 3 steps, warm-up 1, one checkpoint (code-path checks only)."""
    if condition == HISTORICAL:
        condition, lr, seed = "lora", HISTORICAL_LR, draw
    else:
        seed = SEED_OFFSET + draw
    short = {"steps": 3, "warmup": 1, "checkpoints": (3,)} if reduced else {}
    return FineTuneConfig(condition=condition, lr=lr, seed=seed, **short)


# --------------------------------------------------------------------------- LoRA
class LoRAUpdate(nn.Module):
    """Rank-``r`` LoRA per module: ``D = scale * B @ A``, ``A`` random, ``B`` zero."""

    kind = "lora"

    def __init__(self, shapes: Mapping[str, tuple[int, int]], rank: int, scale: float, generator: torch.Generator,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.module_names = list(shapes)
        self.rank, self.scale = int(rank), float(scale)
        A, B = {}, {}
        for m in shapes:  # module order fixes the draws from the generator
            d_out, d_in = shapes[m]
            bound = 1.0 / math.sqrt(d_in)  # kaiming_uniform_(a=sqrt(5)) on an (r, d_in) weight
            a = (torch.rand(self.rank, d_in, generator=generator, dtype=torch.float32) * 2 - 1) * bound
            A[_key(m)] = nn.Parameter(a.to(dtype))
            B[_key(m)] = nn.Parameter(torch.zeros(d_out, self.rank, dtype=dtype))
        self.A, self.B = nn.ParameterDict(A), nn.ParameterDict(B)

    def delta(self, module: str) -> Tensor:
        k = _key(module)
        return self.scale * (self.B[k] @ self.A[k])

    @torch.no_grad()
    def update(self) -> dict[str, Tensor]:
        """``D`` per module on the CPU, formed in float64 from the factors, returned in the training dtype."""
        out = {}
        for m in self.module_names:
            k = _key(m)
            B, A = self.B[k].detach().cpu().double(), self.A[k].detach().cpu().double()
            out[m] = (self.scale * (B @ A)).to(self.B[k].dtype)
        return out

    @torch.no_grad()
    def state(self) -> dict:
        return {"rank": self.rank, "scale": self.scale,
                "A": {m: self.A[_key(m)].detach().cpu().clone() for m in self.module_names},
                "B": {m: self.B[_key(m)].detach().cpu().clone() for m in self.module_names}}


def lora_dense(state: Mapping) -> dict[str, Tensor]:
    """``D`` from saved LoRA factors, formed exactly as ``LoRAUpdate.update`` does."""
    return {m: (float(state["scale"]) * (state["B"][m].double() @ state["A"][m].double())).to(state["B"][m].dtype)
            for m in state["A"]}


@contextmanager
def lora_hooks(model: nn.Module, upd: LoRAUpdate) -> Iterator[None]:
    """Adds ``scale * B (A x)`` to the output of every adapted module."""
    handles = []
    for m in upd.module_names:
        A, B = upd.A[_key(m)], upd.B[_key(m)]

        def hook(_mod, inp, out, A=A, B=B):
            return out + upd.scale * F.linear(F.linear(inp[0], A), B)

        handles.append(model.get_submodule(m).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def adapter_forward(model: nn.Module, upd: LoRAUpdate) -> Callable[[Tensor], Tensor]:
    def fwd(tokens: Tensor) -> Tensor:
        with lora_hooks(model, upd):
            return logits_of(model(tokens))

    return fwd


def training_forward(model: nn.Module, base: Mapping[str, Tensor],
                     upd: DenseUpdate | LoRAUpdate) -> Callable[[Tensor], Tensor]:
    """LoRA trains through output hooks on the factors, the matrix update through ``W_0 + D``."""
    return adapter_forward(model, upd) if isinstance(upd, LoRAUpdate) else merged_forward(model, base, upd)


def make_update(cfg: FineTuneConfig, shapes: Mapping[str, tuple[int, int]],
                dtype: torch.dtype = torch.float32) -> DenseUpdate | LoRAUpdate:
    if cfg.condition == "matrix":
        return DenseUpdate(shapes, dtype)
    return LoRAUpdate(shapes, cfg.rank, cfg.lora_scale, seeded_generator(cfg.seed, "lora-init"), dtype)


# --------------------------------------------------------------------------- training
@torch.no_grad()
def validation(fwd: Callable[[Tensor], Tensor], tokens: Tensor, device, ce0: Tensor | None = None,
               ref_fwd: Callable[[Tensor], Tensor] | None = None, bs: int = VAL_BATCH) -> tuple[dict, Tensor]:
    """Mean CE of ``fwd`` on ``tokens``, its change against ``ce0`` and ``KL(p_ref || p)`` against ``ref_fwd``,
    with the per-window CE (float64)."""
    ce, kl = [], []
    for s in range(0, tokens.shape[0], bs):
        w = tokens[s:s + bs].to(device)
        lp = F.log_softmax(fwd(w[:, :-1]).float(), -1)
        ce.append((-lp.gather(-1, w[:, 1:].unsqueeze(-1)).squeeze(-1)).mean(1).double().cpu())
        if ref_fwd is not None:
            lp0 = F.log_softmax(ref_fwd(w[:, :-1]).float(), -1)
            kl.append((lp0.exp() * (lp0 - lp)).sum(-1).mean(1).double().cpu())
    ce = torch.cat(ce)
    out = {"ce": float(ce.mean())}
    if ce0 is not None:
        out["delta_ce"] = float((ce - ce0).mean())
    if kl:
        out["kl"] = float(torch.cat(kl).mean())
    return out, ce


def train(model: nn.Module, modules: Sequence[str], target: Tensor, retain: Tensor, val_t: Tensor, val_en: Tensor,
          cfg: FineTuneConfig, device, log: Callable[[str], None] = print, keep_adapters: bool = False) -> dict:
    """One fine-tune of ``modules`` on the target windows with English retention. Validation on target and English
    validation at every checkpoint. A non-finite loss, gradient norm or final update raises ``FloatingPointError``
    (a technical failure)."""
    t_start = time.time()
    device = torch.device(device)
    model.eval().requires_grad_(False)
    base = capture_base(model, modules)
    frozen = snapshot_params(model)
    shapes = {m: tuple(base[m].shape) for m in modules}
    upd = make_update(cfg, shapes, next(iter(base.values())).dtype).to(device)
    fwd = training_forward(model, base, upd)
    bfwd = lambda x: logits_of(model(x))  # noqa: E731
    probe = val_t[:2].to(device)
    with torch.no_grad():
        z = float((fwd(probe[:, :-1]) - bfwd(probe[:, :-1])).abs().max())
    if z != 0.0:
        raise RuntimeError(f"zero-update identity failed: {z}")
    ce0_train = per_window_ce(bfwd, target, device).to(torch.float32)
    v0_t, ce0_vt = validation(bfwd, val_t, device)
    v0_e, ce0_ve = validation(bfwd, val_en, device)
    val_by_step = {0: {"target": {**v0_t, "delta_ce": 0.0}, "en": {**v0_e, "delta_ce": 0.0, "kl": 0.0}}}
    samp_t = EpochSampler(target.shape[0], seeded_generator(cfg.seed, "order-target"))
    samp_r = EpochSampler(retain.shape[0], seeded_generator(cfg.seed, "order-retain-en"))
    opt = torch.optim.Adam(upd.parameters(), lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=0.0)
    mb, n_ret = max(1, int(cfg.micro_batch)), cfg.n_retain
    log_rows, step_times, adapters = [], [], {}
    for step in range(1, cfg.steps + 1):
        t = time.time()
        lr_now = cfg.lr * min(1.0, step / cfg.warmup) if cfg.warmup > 0 else cfg.lr
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        idx_t, idx_r = samp_t.take(cfg.n_target), samp_r.take(n_ret)
        h_sum, g_sum, kl_sum = 0.0, 0.0, 0.0
        for s in range(0, len(idx_t), mb):
            ii = idx_t[s:s + mb]
            x = target[ii].to(device)
            logits = fwd(x[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1), reduction="none")
            g_ = ce.view(len(ii), -1).mean(1) - ce0_train[ii].to(device)
            term = hinge(g_, cfg.margin)
            (term.sum() / cfg.n_target).backward()
            h_sum += float(term.detach().sum())
            g_sum += float(g_.detach().sum())
        for s in range(0, len(idx_r), mb):
            x = retain[idx_r[s:s + mb]].to(device)
            with torch.no_grad():
                lp0 = F.log_softmax(bfwd(x[:, :-1]).float(), -1)
            kl = window_kl(lp0, fwd(x[:, :-1]))
            (cfg.lam * kl.sum() / n_ret).backward()
            kl_sum += float(kl.detach().sum())
        gnorm = float(torch.nn.utils.clip_grad_norm_(upd.parameters(), cfg.grad_clip))
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t)
        rec = {"step": step, "hinge": h_sum / cfg.n_target, "g_mean": g_sum / cfg.n_target,
               "kl_retain": kl_sum / n_ret, "grad_norm": gnorm}
        if not all(math.isfinite(v) for v in rec.values()):
            raise FloatingPointError(f"technical: non-finite values at step {step}: {rec}")
        log_rows.append(rec)
        if step == 1 or step % 25 == 0:
            log(f"step {step:4d} hinge {rec['hinge']:.4f} dCE_t {rec['g_mean']:+.4f} KL_en {rec['kl_retain']:.5f} "
                f"|g| {gnorm:.3f}")
        if step in cfg.checkpoints:
            vt, _ = validation(fwd, val_t, device, ce0_vt)
            ve, _ = validation(fwd, val_en, device, ce0_ve, ref_fwd=bfwd)
            val_by_step[step] = {"target": vt, "en": ve}
            if keep_adapters:
                adapters[step] = upd.state()
            log(f"[val {step}] target {vt['delta_ce']:+.4f}  en {ve['delta_ce']:+.4f} (KL {ve['kl']:.5f})")
    assert_frozen(model, frozen)
    update = upd.update()
    bad = [m for m, v in update.items() if not bool(torch.isfinite(v).all())]
    if bad:
        raise FloatingPointError(f"technical: non-finite update in {bad[:4]} ({len(bad)} module(s))")
    out = {"update": update, "val_by_step": val_by_step, "log": log_rows,
           "timings": {"mean_step_s": sum(step_times) / len(step_times), "total_s": time.time() - t_start},
           "checks": {"zero_update_max_abs_logit_diff": z, "frozen_parameters_unchanged": True}}
    if cfg.condition == "lora":
        out["adapter"] = upd.state()
    if keep_adapters:
        out["adapters_by_step"] = adapters
    return out


def checkpoint_rule(val_by_step: Mapping) -> int | None:
    """The historical checkpoint rule with English retention (``CHECKPOINT_RULE``). None when no checkpoint is
    usable. A non-finite validation value at any checkpoint leaves the rule undetermined: ``FloatingPointError``
    (technical), never "no checkpoint" and never an arbitrary pick."""
    rows = []
    for step, v in val_by_step.items():
        if int(step) == 0:
            continue
        vals = {"target val rise": v["target"]["delta_ce"], "English val rise": v["en"]["delta_ce"],
                "English val KL": v["en"]["kl"]}
        bad = {k: x for k, x in vals.items() if not math.isfinite(float(x))}
        if bad:
            raise FloatingPointError(f"technical: non-finite validation at checkpoint {step} ({bad}). The historical "
                                     "checkpoint rule is undetermined")
        t, e = v["target"]["delta_ce"], v["en"]["delta_ce"]
        if CHECKPOINT_BAND[0] <= t <= CHECKPOINT_BAND[1] and e <= CHECKPOINT_RETAIN_MAX:
            rows.append((abs(t - CHECKPOINT_PREFERRED), v["en"]["kl"], int(step)))
    return min(rows)[2] if rows else None


# --------------------------------------------------------------------------- a training run on disk
def train_run(directory: str | Path, model: nn.Module, modules: Sequence[str], target: Tensor, retain: Tensor,
              val_t: Tensor, val_en: Tensor, condition: str, cfg: FineTuneConfig, device, reduced: bool = False,
              header: Mapping | None = None, log: Callable[[str], None] = print) -> Path:
    """Train ``condition`` and write ``update.pt``, then ``train.json`` with the update's sha256 (written last). The
    historical condition keeps only the adapter of the checkpoint its rule selects, and no update when the rule
    selects none (a reduced run then keeps the last step, recorded as a fallback)."""
    d = Path(directory)
    historical = condition == HISTORICAL
    res = train(model, modules, target, retain, val_t, val_en, cfg, device, log, keep_adapters=historical)
    d.mkdir(parents=True, exist_ok=True)
    extra, update = {}, res["update"]
    if historical:
        step = checkpoint_rule(res["val_by_step"])
        extra = {"selected_step": step, "checkpoint_rule": CHECKPOINT_RULE}
        if step is None and reduced:
            step = max(res["adapters_by_step"])
            extra |= {"selected_step": step, "reduced_fallback_last_step": True}
        update = None if step is None else lora_dense(res["adapters_by_step"][step])
        if step is not None:
            torch.save({"modules": None, "adapter": res["adapters_by_step"][step], "config": cfg.to_dict(),
                        "step": step}, d / "update.pt")
    else:
        torch.save({"modules": update, "adapter": res.get("adapter"), "config": cfg.to_dict()}, d / "update.pt")
    rec = {**(header or {}), "condition": condition, "lr": cfg.lr, "modules": list(modules), **extra,
           "config": cfg.to_dict(), "inputs": {k: tokens_identity(v) for k, v in
                                               (("target", target), ("retain", retain), ("val_target", val_t),
                                                ("val_en", val_en))},
           "val_by_step": res["val_by_step"], "log": res["log"], "timings": res["timings"], "checks": res["checks"],
           "reduced": reduced, "update_norm": None if update is None else update_norm(update),
           "update_sha256": None if update is None else sha256_file(d / "update.pt")}
    write_json(d / "train.json", rec)  # last: the run is complete once train.json exists
    return d


def load_update(directory: str | Path) -> dict[str, Tensor]:
    """The update of a training run, refused unless the bytes of ``update.pt`` match the sha256 its ``train.json``
    recorded at training (papers/selective-editing/index.qmd#sec-analysis)."""
    d = Path(directory)
    want = json.loads((d / "train.json").read_text())["update_sha256"]
    got = sha256_file(d / "update.pt")
    if got != want:
        raise ValueError(f"{d}/update.pt: sha256 {got} does not match train.json ({want})")
    rec = torch.load(d / "update.pt", map_location="cpu", weights_only=True)
    return lora_dense(rec["adapter"]) if rec["modules"] is None else rec["modules"]


def scaled(update: Mapping[str, Tensor], s: float, device) -> dict[str, Tensor]:
    """``s * D`` per module, float32 on ``device``: the dose of a learned update."""
    return {m: (float(s) * D).to(device=device, dtype=torch.float32) for m, D in update.items()}


def update_rank(update: Mapping[str, Tensor]) -> int:
    """Largest numerical rank over the update's modules (float64, tolerance ``max(shape) * eps`` of its dtype)."""
    return max(int(torch.linalg.matrix_rank(v.double(), rtol=max(v.shape) * torch.finfo(v.dtype).eps))
               for v in update.values())


# --------------------------------------------------------------------------- learning-rate choice
def choose_lr(model: nn.Module, updates: Mapping[float, Mapping[str, Tensor]], val_t: Tensor, val_en: Tensor,
              device, bs: int = VAL_BATCH) -> dict:
    """The draw-0 choice of one target and condition: each rate's final update is matched to the upper effect point
    on target validation by the calibration rule on ``SCALE_GRID``. Among the matched rates, the lowest English
    validation KL at the match wins, then the smaller rate. ``chosen_lr`` None when no rate is matched.

    A non-finite target rise anywhere in a rate's scan, or a non-finite English KL or rise at its match, leaves the
    choice undetermined: ``FloatingPointError`` (technical), never "no rate reached" and never a pick ordered by
    NaN."""
    tau = max(TAUS)
    rows = []
    for lr, D in updates.items():
        base = capture_base(model, list(D))
        ce0 = ce_docs(model, base, val_t, None, device, bs)
        cache: dict[float, float] = {}

        def delta(s, D=D, base=base, ce0=ce0, cache=cache):
            if s not in cache:
                cache[s] = float((ce_docs(model, base, val_t, scaled(D, s, device), device, bs) - ce0).mean())
            return cache[s]

        mt = scan_and_match(delta, tau, SCALE_GRID, max_bisect=MAX_BISECT, tol=TOL)
        bad = sorted(s for s, x in cache.items() if not math.isfinite(x))
        if bad:
            raise FloatingPointError(f"technical: lr {lr:g}: non-finite target val rise at scale {bad[0]:g} "
                                     f"({len(bad)} of {len(cache)} evaluated). The rate choice is undetermined")
        row = {"lr": lr, "status": mt["status"], "scale": mt["scale"], "achieved": mt["achieved"]}
        if mt["status"] == "matched":
            ce0e = ce_docs(model, base, val_en, None, device, bs)
            r = edit_readout(model, base, scaled(D, mt["scale"], device), {"en": val_en}, device, {"en": ce0e},
                             bs)["en"]
            if not (math.isfinite(r["kl"]) and math.isfinite(r["delta_ce"])):
                raise FloatingPointError(f"technical: lr {lr:g}: non-finite English val readout at the match (KL "
                                         f"{r['kl']}, rise {r['delta_ce']}). The rate choice is undetermined")
            row |= {"en_kl": r["kl"], "en_delta_ce": r["delta_ce"]}
        rows.append(row)
    ok = [r for r in rows if r["status"] == "matched"]
    pick = min(ok, key=lambda r: (r["en_kl"], r["lr"])) if ok else None
    return {"rows": rows, "chosen_lr": None if pick is None else pick["lr"], "tau": tau, "rule": CHOICE_RULE}
