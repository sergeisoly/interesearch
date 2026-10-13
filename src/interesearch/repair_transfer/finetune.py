"""The target fine-tune of the repair-transfer study: a full-matrix update D of the 24 modules, its checkpoint rule
and its gates (papers/repair-transfer/index.qmd#sec-app-methods).

The trainable tensor is D itself, zero-initialised in float32. The training forward is ``W_0 + D`` through
``functional_call``, so D is exact and the model's own parameters never change (asserted). Objective for German
windows G and retention windows C (en, fr, es, it in equal shares)::

    L = mean_{x in G} max(0, margin - (CE_theta(x) - CE_0(x))) + lam * mean_{x in C} KL(p_0(.|x) || p_theta(.|x))

with CE_0 computed once with the base model and p_0 the base model run on the fly. Adam (0.9, 0.999), eps 1e-8,
global grad-norm clip 1.0, 10 warm-up steps then a constant learning rate, 200 steps, 8 German + 8 retention windows
per step in micro-batches of 4, validation at steps 50/100/150/200. The data order comes from generators salted by
the seed (``training.seeded_generator``). No global seed is set, nothing else draws random numbers.

Checkpoint rule (validation only): usable if the German CE rises by >= 0.5 nats and every retention language's CE
by <= 0.1 nats. Among usable checkpoints the German rise closest to 1.0, then the lower mean retention KL, then the
earlier step. Gates: the zero update reproduces the base model exactly, every model parameter is unchanged, and the
evaluation path (``edits.applied_update``) reproduces the per-window German validation CE of the training forward
at the saved checkpoint within 1e-4 nats. The first two stop the run with an exception, before any record is
written. A failed evaluation-path gate keeps the records and exits with code 3.

Validation reduces per window in float32 on the device, then moves to float64 on the CPU (``window_ce_of``,
``window_kl_of``). These are the original runs' reductions, kept apart from the core's.

Outputs: ``update.pt`` (``{"modules": {module: D float32}, "meta": ...}``, the selected checkpoint, or the last one
when none is usable), ``val.json`` (validation at step 0 and at every checkpoint, the rule's rows and choice),
``train_log.jsonl`` (one line per step) and ``run.json`` (configuration, input hashes, checks, gates, timings).

    uv run python -m interesearch.repair_transfer.finetune train --lr 3e-6 --seed 14 --assets assets \\
        --train data/repair_transfer/windows/finetune_train/primary_target \\
        --val data/repair_transfer/windows/finetune_val --out runs/finetune/primary_target
    uv run python -m interesearch.repair_transfer.finetune select runs/finetune/donor_lr3e-6 runs/finetune/donor_lr1e-5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import assets
from ..checksums import sha256_file
from ..edits import applied_update, assert_frozen, capture_base, logits_of, snapshot_params
from ..training import (DenseUpdate, EpochSampler, disable_tf32, hinge, merged_forward, per_window_ce,
                        seeded_generator, window_ce, window_kl)
from .data import load_windows, window_role

Tensor = torch.Tensor

TARGET_LANG = "de"
RETAIN = ("en", "fr", "es", "it")
TRAIN_LANGS = (TARGET_LANG, *RETAIN)
VAL_LANGS = (TARGET_LANG, *RETAIN, "nl", "pt")  # nl and pt are validated and reported, never selected on
TRAIN_ROLE, VAL_ROLE = "finetune_train", "finetune_val"
GATES = {"eval_path_max_abs_ce_doc_diff": 1e-4}
RULE_TEXT = ("usable: German val CE rise >= {min_t} nats and every retention language's ({langs}) CE rise <= {max_r} "
             "nats; among usable: German rise closest to {pref}, then lower mean retention KL(p_0||p), then earlier "
             "step")


@dataclass
class FTConfig:
    lr: float
    seed: int
    steps: int = 200
    warmup: int = 10
    checkpoints: tuple[int, ...] = (50, 100, 150, 200)
    lam: float = 1.0  # retention KL weight
    margin: float = 1.0  # hinge cap in nats per German window
    n_target: int = 8  # German windows per optimizer step
    n_retain_per_lang: int = 2
    micro_batch: int = 4  # windows per forward pass (gradient accumulation)
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    min_target_effect: float = 0.5
    max_retain_effect: float = 0.1
    preferred_target_effect: float = 1.0
    condition: str = "matrix"  # the only condition here. Recorded for the target's binding check

    def __post_init__(self) -> None:
        self.checkpoints = tuple(sorted({int(s) for s in self.checkpoints if 0 < int(s) <= self.steps}))
        if self.steps not in self.checkpoints:
            self.checkpoints = self.checkpoints + (self.steps,)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "FTConfig":
        """Load a configuration, ignoring unknown keys."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names})


@dataclass
class FTData:
    """Token windows ``[N, T+1]``. The model reads ``w[:, :-1]`` and predicts ``w[:, 1:]``."""

    target: Tensor  # German training windows
    retain: dict[str, Tensor]  # language -> training windows
    val: dict[str, Tensor]  # language -> validation windows, in VAL_LANGS order


@dataclass
class FTResult:
    config: dict
    val_by_step: dict[int, dict]
    selection: dict
    updates: dict[int, dict[str, Tensor]]  # checkpoint step -> D (float32, CPU)
    log: list[dict]
    checks: dict
    timings: dict

    @property
    def saved_step(self) -> int:
        """The selected checkpoint, or the last one when no checkpoint is usable."""
        return self.selection["selected_step"] if self.selection["selected_step"] is not None else max(self.updates)


# --------------------------------------------------------------------------- per-window reductions
def base_forward(model: nn.Module) -> Callable[[Tensor], Tensor]:
    return lambda tokens: logits_of(model(tokens))


@torch.no_grad()
def seq_logprobs(forward_fn: Callable[[Tensor], Tensor], window: Tensor) -> Tensor:
    """Log-softmax ``[B, T, V]`` (float32) of the predictions for ``window [B, T+1]``."""
    return F.log_softmax(forward_fn(window[:, :-1]).float(), dim=-1)


def window_ce_of(lp: Tensor, targets: Tensor) -> Tensor:
    """Mean CE per window: float32 on the device, then float64 on the CPU."""
    return (-lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).mean(dim=1).cpu().double()


def window_kl_of(lp_ref: Tensor, lp: Tensor) -> Tensor:
    """``KL(p_ref || p)`` averaged over positions per window: float32 on the device, then float64 on the CPU."""
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1).mean(dim=1).cpu().double()


# --------------------------------------------------------------------------- validation and the checkpoint rule
@torch.no_grad()
def validate(model: nn.Module, forward_fn: Callable[[Tensor], Tensor], val: Mapping[str, Tensor], device,
             base_ce: Mapping[str, Tensor] | None = None, batch_size: int = 8) -> dict:
    """Per language: mean CE and the per-window CE. With ``base_ce`` also the mean CE change and the mean
    ``KL(p_0 || p_theta)``."""
    bfwd = base_forward(model)
    out = {}
    for lang, toks in val.items():
        ce, kl = [], []
        for s in range(0, toks.shape[0], batch_size):
            b = toks[s:s + batch_size].to(device)
            lp = seq_logprobs(forward_fn, b)
            ce.append(window_ce_of(lp, b[:, 1:]))
            if base_ce is not None:
                lp0 = seq_logprobs(bfwd, b)
                kl.append(window_kl_of(lp0, lp))
                del lp0
            del lp
        ce_doc = torch.cat(ce)
        row = {"ce": float(ce_doc.mean()), "ce_doc": ce_doc.tolist()}
        if base_ce is not None:
            row["delta_ce"] = float((ce_doc - base_ce[lang]).mean())
            row["kl_base"] = float(torch.cat(kl).mean())
        out[lang] = row
    return out


def _rule_row(step, v: Mapping[str, dict], cfg: FTConfig, run: str | None = None) -> dict:
    dt = v[TARGET_LANG]["delta_ce"]
    dr = {lang: v[lang]["delta_ce"] for lang in RETAIN}
    other = {lang: v[lang]["delta_ce"] for lang in v if lang != TARGET_LANG and lang not in RETAIN}
    kl = sum(v[lang]["kl_base"] for lang in RETAIN) / len(RETAIN)
    usable = dt >= cfg.min_target_effect and all(d <= cfg.max_retain_effect for d in dr.values())
    row = {"step": int(step), "delta_target": dt, "delta_retain": dr, "max_delta_retain": max(dr.values()),
           "delta_other": other, "max_delta_other": max(other.values()) if other else None,
           "mean_retain_kl": kl, "usable": usable, "distance_to_preferred": abs(dt - cfg.preferred_target_effect)}
    if run is not None:
        row["run"] = run
    return row


def _pick(rows: list[dict]) -> dict | None:
    usable = [r for r in rows if r["usable"]]
    if not usable:
        return None
    # distances are compared after rounding to 1e-9 nats so that float noise does not decide a tie
    return min(usable, key=lambda r: (round(r["distance_to_preferred"], 9), r["mean_retain_kl"], r["step"]))


def rule_text(cfg: FTConfig) -> str:
    return RULE_TEXT.format(min_t=cfg.min_target_effect, max_r=cfg.max_retain_effect,
                            pref=cfg.preferred_target_effect, langs="/".join(RETAIN))


def select_checkpoint(val_by_step: Mapping[int, Mapping[str, dict]], cfg: FTConfig) -> dict:
    """The checkpoint rule within one run."""
    rows = [_rule_row(s, v, cfg) for s, v in sorted(val_by_step.items(), key=lambda kv: int(kv[0]))
            if int(s) in cfg.checkpoints]
    pick = _pick(rows)
    return {"rule": rule_text(cfg), "rows": rows, "selected_step": pick["step"] if pick else None,
            "usable": pick is not None}


def select_across(runs: Mapping[str, dict]) -> dict:
    """The same rule over every (run, checkpoint) pair of several runs (``{name: val.json record}``): the donor
    fine-tune's learning rate was chosen this way."""
    rows, cfg = [], None
    for name, val in runs.items():
        cfg = FTConfig.from_dict(val["config"])
        rows += [_rule_row(s, v, cfg, run=name) for s, v in val["by_step"].items() if int(s) in cfg.checkpoints]
    return {"rows": rows, "selected": _pick(rows), "rule": rule_text(cfg) if cfg else None}


# --------------------------------------------------------------------------- training
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def train(model: nn.Module, modules: Sequence[str], data: FTData, cfg: FTConfig, device: str | torch.device = "cpu",
          log: Callable[[str], None] = print, val_batch_size: int = 8) -> FTResult:
    """Fine-tune ``modules`` of ``model`` (whose own parameters are never changed)."""
    t_start = time.time()
    device = torch.device(device)
    model.eval().requires_grad_(False)
    modules = list(modules)
    base = capture_base(model, modules)
    frozen = snapshot_params(model)
    shapes = {m: tuple(base[m].shape) for m in modules}
    upd = DenseUpdate(shapes, next(iter(base.values())).dtype).to(device)
    fwd, bfwd = merged_forward(model, base, upd), base_forward(model)
    checks: dict = {}
    timings: dict = {}

    # zero-update identity: at initialisation the training forward equals the base model exactly
    probe = data.val[TARGET_LANG][: min(2, data.val[TARGET_LANG].shape[0])].to(device)
    with torch.no_grad():
        z = (fwd(probe) - bfwd(probe)).abs().max().item()
    checks["zero_update_max_abs_logit_diff"] = z
    if z != 0.0:
        raise RuntimeError(f"zero-update identity failed: max |logit diff| = {z}")

    # base CE of the German training windows (the hinge reference) and the base validation metrics
    t = time.time()
    ce0_target = per_window_ce(bfwd, data.target, device).to(torch.float32)
    val0 = validate(model, bfwd, data.val, device, base_ce=None, batch_size=val_batch_size)
    base_ce = {lang: torch.tensor(v["ce_doc"], dtype=torch.float64) for lang, v in val0.items()}
    for lang in val0:
        val0[lang]["delta_ce"], val0[lang]["kl_base"] = 0.0, 0.0
    timings["base_pass_s"] = time.time() - t
    val_by_step: dict[int, dict] = {0: val0}
    log("base val CE: " + ", ".join(f"{lang} {v['ce']:.4f}" for lang, v in val0.items()))

    samp_t = EpochSampler(data.target.shape[0], seeded_generator(cfg.seed, "order-target"))
    samp_r = {lang: EpochSampler(data.retain[lang].shape[0], seeded_generator(cfg.seed, f"order-retain-{lang}"))
              for lang in RETAIN}
    opt = torch.optim.Adam(upd.parameters(), lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=0.0)
    n_ret = cfg.n_retain_per_lang * len(RETAIN)
    mb = cfg.micro_batch
    train_log: list[dict] = []
    updates: dict[int, dict[str, Tensor]] = {}
    step_times, val_times = [], []

    for step in range(1, cfg.steps + 1):
        t = time.time()
        lr_now = cfg.lr * min(1.0, step / cfg.warmup) if cfg.warmup > 0 else cfg.lr
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        idx_t = samp_t.take(cfg.n_target)
        rows_r = [(lang, i) for lang in RETAIN for i in samp_r[lang].take(cfg.n_retain_per_lang)]

        h_sum, g_sum, kl_sum = 0.0, 0.0, 0.0
        for s in range(0, len(idx_t), mb):
            ii = idx_t[s:s + mb]
            x = data.target[ii].to(device)
            logits = fwd(x[:, :-1])
            g_ = window_ce(logits, x[:, 1:]) - ce0_target[ii].to(device)
            term = hinge(g_, cfg.margin)
            (term.sum() / cfg.n_target).backward()
            h_sum += float(term.detach().sum())
            g_sum += float(g_.detach().sum())
            del logits
        for s in range(0, len(rows_r), mb):
            chunk = rows_r[s:s + mb]
            x = torch.stack([data.retain[lang][i] for lang, i in chunk]).to(device)
            with torch.no_grad():
                lp0 = F.log_softmax(bfwd(x[:, :-1]).float(), dim=-1)
            kl = window_kl(lp0, fwd(x[:, :-1]))
            (cfg.lam * kl.sum() / n_ret).backward()
            kl_sum += float(kl.detach().sum())
            del lp0
        gnorm = float(torch.nn.utils.clip_grad_norm_(upd.parameters(), cfg.grad_clip))
        opt.step()
        _sync(device)
        dt = time.time() - t
        step_times.append(dt)
        rec = {"step": step, "lr": lr_now, "hinge": h_sum / cfg.n_target, "g_mean": g_sum / cfg.n_target,
               "kl_retain": kl_sum / n_ret, "loss": h_sum / cfg.n_target + cfg.lam * kl_sum / n_ret,
               "grad_norm": gnorm, "clipped": gnorm > cfg.grad_clip, "step_s": dt}
        if not all(math.isfinite(v) for v in (rec["loss"], gnorm)):
            raise FloatingPointError(f"non-finite loss or gradient at step {step}: {rec}")
        train_log.append(rec)
        if step == 1 or step % 10 == 0:
            log(f"step {step:4d}  loss {rec['loss']:.4f}  hinge {rec['hinge']:.4f}  dCE_de(batch) {rec['g_mean']:+.4f}"
                f"  KL_ret {rec['kl_retain']:.5f}  |g| {gnorm:.3f}  {dt:.2f}s")
        if step in cfg.checkpoints:
            t = time.time()
            v = validate(model, fwd, data.val, device, base_ce=base_ce, batch_size=val_batch_size)
            _sync(device)
            val_times.append(time.time() - t)
            val_by_step[step] = v
            updates[step] = upd.update()
            log(f"[val step {step}] " + ", ".join(f"{lang} {r['delta_ce']:+.4f} (KL {r['kl_base']:.4f})"
                                                  for lang, r in v.items()))

    assert_frozen(model, frozen)
    checks["frozen_parameters_unchanged"] = True
    for s, u in updates.items():
        for m, D in u.items():
            assert tuple(D.shape) == shapes[m] and D.dtype == base[m].dtype, (s, m)
    timings.update({"steps": cfg.steps, "mean_step_s": sum(step_times) / len(step_times),
                    "median_step_s": sorted(step_times)[len(step_times) // 2], "first_step_s": step_times[0],
                    "train_s": sum(step_times), "val_pass_s": sum(val_times) / len(val_times),
                    "total_s": time.time() - t_start})
    return FTResult(cfg.to_dict(), val_by_step, select_checkpoint(val_by_step, cfg), updates, train_log, checks,
                    timings)


# --------------------------------------------------------------------------- gates and records
@torch.no_grad()
def eval_path_check(model: nn.Module, modules: Sequence[str], update: Mapping[str, Tensor], tokens: Tensor,
                    recorded_ce_doc: Sequence[float], device, batch_size: int = 8) -> dict:
    """Re-evaluate ``W_0 + D`` through ``edits.applied_update`` (the path every later stage uses) and compare the
    per-window CE with what the training forward recorded at the same checkpoint."""
    base = capture_base(model, modules)
    ce = []
    with applied_update(model, base, dict(update)):
        for s in range(0, tokens.shape[0], batch_size):
            b = tokens[s:s + batch_size].to(device)
            ce.append(window_ce_of(seq_logprobs(base_forward(model), b), b[:, 1:]))
    ce = torch.cat(ce)
    rec = torch.tensor(list(recorded_ce_doc), dtype=torch.float64)
    return {"max_abs_ce_doc_diff": float((ce - rec).abs().max()), "mean_ce_eval_path": float(ce.mean()),
            "mean_ce_training_path": float(rec.mean())}


def apply_gates(checks: Mapping) -> dict:
    failed = []
    if checks.get("zero_update_max_abs_logit_diff") != 0.0:
        failed.append("zero_update_identity")
    if not checks.get("frozen_parameters_unchanged"):
        failed.append("frozen_parameters")
    ep = checks.get("eval_path_vs_training_path_de_val", {}).get("max_abs_ce_doc_diff", math.inf)
    if not ep <= GATES["eval_path_max_abs_ce_doc_diff"]:
        failed.append("eval_path")
    return {"tolerances": GATES, "failed": failed, "passed": not failed}


def target_model_identity(assets_dir: str | Path) -> tuple[dict, list[str]]:
    """sha256 of the target's weights and model config, and what differs from the pinned assets."""
    ident, errors = {}, []
    for a in assets.ASSETS:
        if a.path in ("target/model_step_99999.pt", "target/model_config.yaml"):
            problem = assets.check(assets_dir, a)
            ident[a.path] = a.sha256 if problem is None else None
            if problem:
                errors.append(problem)
    return ident, errors


def environment(device) -> dict:
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
           "device": str(device), "threads": torch.get_num_threads()}
    if torch.device(device).type == "cuda":
        env.update({"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                    "cuda_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9})
    return env


def code_identity() -> dict:
    """sha256 of every source file of the package, by path inside it."""
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root.parent)): sha256_file(p) for p in sorted(root.rglob("*.py"))}


def save_result(res: FTResult, out: Path, modules: Sequence[str], model_identity: dict) -> dict:
    """Write ``update.pt`` (the saved checkpoint), ``train_log.jsonl`` and ``val.json``."""
    for s, u in res.updates.items():
        if set(u) != set(modules):
            raise KeyError(f"checkpoint {s}: modules {sorted(set(u) ^ set(modules))} differ from the trained set")
    step = res.saved_step
    meta = {"step": step, "selected_by_rule": res.selection["selected_step"] is not None,
            "condition": res.config["condition"], "seed": res.config["seed"], "lr": res.config["lr"],
            "D": "W_FT - W_0, float32, nn.Linear [d_out, d_in] layout", "target_model": model_identity}
    torch.save({"modules": {m: res.updates[step][m].to(torch.float32) for m in modules}, "meta": meta},
               out / "update.pt")
    with open(out / "train_log.jsonl", "w") as f:
        for r in res.log:
            f.write(json.dumps(r) + "\n")
    val = {"config": res.config, "by_step": {str(s): v for s, v in res.val_by_step.items()},
           "selection": res.selection, "saved_step": step}
    (out / "val.json").write_text(json.dumps(val, indent=1))
    return {"written": ["update.pt", "train_log.jsonl", "val.json"], "saved_step": step,
            "update_sha256": sha256_file(out / "update.pt")}


def load_update(ft_dir: str | Path) -> tuple[dict[str, Tensor], dict]:
    """D (float32) of a fine-tune directory and its run record. Refuses an ``update.pt`` whose sha256 differs from
    the one the record wrote."""
    ft_dir = Path(ft_dir)
    run = json.loads((ft_dir / "run.json").read_text())
    if sha256_file(ft_dir / "update.pt") != run["saved"]["update_sha256"]:
        raise ValueError(f"{ft_dir}/update.pt: sha256 differs from run.json")
    obj = torch.load(ft_dir / "update.pt", map_location="cpu", weights_only=True)
    return {m: D.to(torch.float32) for m, D in obj["modules"].items()}, run


# --------------------------------------------------------------------------- CLI
def cmd_train(a) -> int:
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"REFUSED: {out} is not empty (no resume, use a new directory)", file=sys.stderr)
        return 3
    model_id, errors = target_model_identity(a.assets)
    errors += [f"{d}: role {window_role(d)!r}, expected {role!r}"
               for d, role in ((a.train, TRAIN_ROLE), (a.val, VAL_ROLE)) if window_role(d) != role]
    if errors:
        print("REFUSED: inputs\n  " + "\n  ".join(errors), file=sys.stderr)
        return 3
    cfg = FTConfig(lr=a.lr, seed=a.seed, steps=a.steps, checkpoints=tuple(int(x) for x in a.checkpoints.split(",")))
    torch.set_num_threads(a.threads)
    precision = disable_tf32()
    t0 = time.time()
    model = assets.load_target(a.assets, a.device)
    train_w = load_windows(a.train, TRAIN_LANGS, role=TRAIN_ROLE)
    val_w = load_windows(a.val, VAL_LANGS, role=VAL_ROLE)
    data = FTData(target=train_w.tokens[TARGET_LANG], retain={lg: train_w.tokens[lg] for lg in RETAIN},
                  val=val_w.tokens)
    t_load = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    tag = f"[{out.name}]"
    print(f"{tag} matrix lr={cfg.lr} seed={cfg.seed} device={a.device}, loaded in {t_load:.1f}s", flush=True)
    res = train(model, assets.MODULES, data, cfg, a.device, lambda s: print(f"{tag} {s}", flush=True),
                a.val_batch_size)
    step = res.saved_step
    t = time.time()
    res.checks["eval_path_vs_training_path_de_val"] = eval_path_check(
        model, assets.MODULES, res.updates[step], data.val[TARGET_LANG], res.val_by_step[step][TARGET_LANG]["ce_doc"],
        a.device, batch_size=a.val_batch_size)
    res.timings.update({"eval_path_check_s": time.time() - t, "load_s": t_load})
    gates = apply_gates(res.checks)
    saved = save_result(res, out, assets.MODULES, model_id)
    run = {"config": res.config, "val_batch_size": a.val_batch_size,
           "selection": {k: v for k, v in res.selection.items() if k != "rows"}, "completed": True, "gates": gates,
           "saved": saved, "checks": res.checks, "timings": res.timings, "precision": precision,
           "environment": environment(a.device), "code": code_identity(), "target_model": model_id,
           "data": {"train": {"role": train_w.role, "source": train_w.source, "sha256": train_w.sha256},
                    "val": {"role": val_w.role, "dates": val_w.dates, "sha256": val_w.sha256}},
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    sel = res.selection
    print(f"{tag} completed, selected step {sel['selected_step']} (usable={sel['usable']}), saved step {step}, gates "
          f"{'passed' if gates['passed'] else 'FAILED ' + str(gates['failed'])}, total {res.timings['total_s']:.0f}s",
          flush=True)
    return 0 if gates["passed"] else 3


def cmd_select(a) -> int:
    res = select_across({str(d): json.loads(Path(d, "val.json").read_text()) for d in a.runs})
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["selected"]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a repair-transfer target (full-matrix update of the 24 "
                                             "modules) or apply the checkpoint rule across runs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--lr", type=float, required=True)
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--assets", required=True, help="assets directory with target/")
    tr.add_argument("--train", required=True, help=f"window directory of role {TRAIN_ROLE} ({', '.join(TRAIN_LANGS)})")
    tr.add_argument("--val", required=True, help=f"window directory of role {VAL_ROLE} ({', '.join(VAL_LANGS)})")
    tr.add_argument("--out", required=True)
    tr.add_argument("--steps", type=int, default=200)
    tr.add_argument("--checkpoints", default="50,100,150,200")
    tr.add_argument("--val-batch-size", type=int, default=8)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--threads", type=int, default=4)
    tr.set_defaults(func=cmd_train)
    se = sub.add_parser("select")
    se.add_argument("runs", nargs="+")
    se.add_argument("--json", default=None)
    se.set_defaults(func=cmd_select)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
