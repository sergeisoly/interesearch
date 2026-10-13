"""Runs of the repair-transfer study on the target model: one measurement block of a target, and the donor repair.

    block   --target primary_target|replication_target --block main|nested
        main    the 2K budget. Every family of fitting.py. The polish of strengths (strengths_converged) and the
                group strengths_best = strengths and its polished versions.
        nested  the 256 and 512 budgets (nested prefixes of the 2K windows). The first five families.
        Both: step-capped groups <family>@<budget>:le<cap> of the first five families (the checkpoints at steps <=
        cap of the same runs).
    donor   the donor repair: a fresh rank-one repair of the donor fine-tune on its own fit windows (three learning
            rates, 200 steps, a checkpoint every 25), chosen on the fine-tune's validation windows and read once on
            the donor readout windows with its gate. Writes donor_repair.pt, the input of every block: the pinned one
            (DONOR_REPAIR_SHA256), or a new one whose run.json beside it records a full, successful donor run that
            passed its gate, on the same target model, with this file as its chosen candidate.

Steps of a block: input checks. The pre-readout gate (the fine-tune is usable and passed its gates, and its Dutch
damage on the selection windows is at least 0.05 nat/token). Training of every family, budget and learning rate
(600 steps, a checkpoint every 50). The full-batch fit loss of every candidate and the polish. Evaluation of every
candidate on the selection windows and the selection of every group (selection.py). Parity and direction cosines
of the chosen candidates. Evaluation of the chosen candidates on the readout windows, once.

Every run writes ``run.json``: status, exit code and reasons, the input hashes (with the role, checked hashes,
dates and rows of every window directory it read, recorded when it read it), the identity of this code, the
effective configuration with its deviations from the frozen one, and the run's records. The measurement files of a
block (``block.json``, ``selection.npz``, ``readout.npz``, ``SHA256SUMS``, data.load_block) are written only when
the run ends with code 0. Exit codes: 0 ok. 2 bad arguments. 3 refused before anything is written (a non-empty
output directory, an input that differs from its pin, its recipe or the windows it was prepared from, a window
directory of the wrong role, fit windows shorter than a budget). 8 a planned family saved no checkpoint. 9 technical
failure (a non-finite value, a failed initialization, parity or cap check, a non-finite polish), which takes
precedence. 10 the target is not evaluable before any repair is trained. A donor run whose repair fails its gate or
that has no feasible checkpoint ends with 0 and ``gate.pass`` false.

Reduced verification runs (``--steps``, ``--every``, ``--step-caps``, ``--rows``, ``--polish-iters`` away from the
frozen values) take ``--target standin``, and a stand-in block is always reduced. A reduced run refuses every
readout role (data.READOUT_ROLES) before it reads any window, reads its readout from stand-in windows, and its
block can be loaded only with ``allow_reduced``.

    uv run python -m interesearch.repair_transfer.run block --target primary_target --block main --assets assets \\
        --finetune runs/finetune/primary_target \\
        --finetune-train data/repair_transfer/windows/finetune_train/primary_target \\
        --finetune-val data/repair_transfer/windows/finetune_val \\
        --donor-repair data/repair_transfer/donor/donor_repair.pt --fit data/repair_transfer/windows/fit \\
        --selection data/repair_transfer/windows/selection --readout data/repair_transfer/windows/readout \\
        --out runs/blocks/primary_target/main
    uv run python -m interesearch.repair_transfer.run donor --assets assets --finetune runs/finetune/donor \\
        --fit data/repair_transfer/windows/donor_fit --selection data/repair_transfer/windows/finetune_val \\
        --readout data/repair_transfer/windows/donor_readout --out runs/donor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping

import torch

from .. import assets
from ..assets import MODULES
from ..checksums import sha256_file
from ..edits import capture_base
from ..training import disable_tf32
from . import stats
from .data import DONOR_FILES, READOUT_ROLES, load_windows, refused_roles, window_role, write_block
from .evaluate import EVAL_BATCH, PARITY_TOL, direction, evaluate, parity
from .finetune import TRAIN_ROLE, VAL_ROLE, code_identity, environment, load_update, target_model_identity
from .fitting import (BUDGETS, CAP_TOL, LRS, POLISH, RAND_SEED, RETAIN, FitLoss, Method, finite, fit_accounting,
                      fit_sets, fro_of, init_check, make_method, nonfinite_snapshot, polish, randomized, train)
from .selection import german_feasible, point, select

LANGS = ("de", "nl", "en", "fr", "es", "it", "pt")
FIT_LANGS = ("de", "nl", *RETAIN)
TARGETS = ("primary_target", "replication_target", "standin")
FAMILIES = ("strengths", "donor_init", "donor_init_output", "random_init", "fresh", "fresh_seed2",
            "random_init_seed2")
CAPPED = FAMILIES[:5]
BLOCKS = {"main": {"budgets": ("2K",), "polish": True, "families": FAMILIES},
          "nested": {"budgets": ("256", "512"), "polish": False, "families": FAMILIES[:5]}}
STEPS, EVERY, STEP_CAPS = 600, 50, (50, 100, 200)
DAMAGE_FLOOR = 0.05  # Dutch CE damage of the fine-tune on the selection windows, nat/token
N_READOUT_MIN = 64
RC_OK, RC_REFUSED, RC_FAILED_FAMILY, RC_TECHNICAL, RC_PRE_READOUT_STOP = 0, 3, 8, 9, 10
STATUS = {RC_OK: "ok", RC_FAILED_FAMILY: "failed_family", RC_TECHNICAL: "technical",
          RC_PRE_READOUT_STOP: "pre_readout_stop"}
POLISH_KEYS = ("finite", "converged", "stop", "iters", "evals", "L_start", "L_end", "stationarity_start",
               "stationarity_end", "stationarity_ref", "path_parity_abs", "norm_over_cap", "seconds")
DIRECTION_PAIRS = (("donor_init", "fresh"), ("donor_init_output", "fresh"), ("donor_init", "donor_init_output"),
                   ("random_init", "fresh"), ("donor_init", "random_init"), ("fresh_seed2", "fresh"),
                   ("donor_init", "strengths_best"))
# The donor repair of the published blocks (the output of the donor command on the donor fine-tune). A block also
# accepts a donor repair the donor command wrote anew, through that run's record (donor_repair_errors).
DONOR_REPAIR_SHA256 = "603b62c776856654c925fecd37b7bb43d8dc98994334e5582bb5838c5a9371d7"
# The target fine-tunes: the common recipe, the seed and the rows of the fine-tune text each one is bound to.
FT_RECIPE = {"condition": "matrix", "lr": 3e-6, "steps": 200}
FT_BINDING = {"primary_target": {"seed": 14, "rows": [0, 64]}, "replication_target": {"seed": 15, "rows": [64, 128]}}
FT_SOURCE_ROLE = "donor_readout"  # the fine-tune text: rows of the donor stage's readout windows (128 dates)

DONOR_STEPS, DONOR_EVERY = 200, 25
DONOR_LRS = (3e-4, 1e-3, 3e-3)
DONOR_GATE_R_NL = 0.80  # plus German feasibility, r_kl > 0 and the retain check, on the primary readout dates
DONOR_DAMAGE_CI_LO = 0.025  # lower end of the 95% interval of the donor's Dutch readout damage, nat/token
N_DONOR_READOUT = 128
# The first 16 donor readout dates were evaluated by an earlier technical check. They are disclosed as previewed and
# excluded from the donor gate (the full 128 are reported as a sensitivity row).
PREVIEWED_READOUT = ("1998-01-16", "2006-05-17", "1998-04-02", "1996-12-10", "2006-12-18", "2005-09-07", "2002-03-14",
                     "2003-09-03", "2006-12-12", "1999-09-17", "2010-01-18", "2001-07-05", "1998-10-08", "2002-04-25",
                     "2000-02-14", "2001-12-11")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def logger():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)
    return log, t0


def setup(a):
    """The target model on the device with W_0 captured. Threads and precision as recorded."""
    torch.set_num_threads(a.threads)
    precision = disable_tf32()
    model = assets.load_target(a.assets, a.device)
    return model, capture_base(model, MODULES), precision


def finish(out: Path, info: dict, rc: int, log, reasons: list[str], export=None) -> int:
    """Complete run.json, write it and log the end. ``export(record_sha256)`` writes the files of a successful run
    first, naming the hash of the record's exact bytes: a record that says ok exists only after its files."""
    info.update(status=STATUS[rc], rc=rc, reasons=reasons, total_s=time.time() - info.pop("_t0"), utc_end=utc())
    if info["config"]["device"] == "cuda":
        info["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    text = json.dumps(info, indent=1, default=str)
    if export is not None:
        export(hashlib.sha256(text.encode()).hexdigest())
    (out / "run.json").write_text(text)
    log(f"{info['command']} done: {info['status']} (exit {rc}){': ' + '; '.join(reasons) if reasons else ''}")
    return rc


# --------------------------------------------------------------------------- input checks
def role_errors(dirs: Mapping[str, tuple[str, str | None]], reduced: bool) -> list[str]:
    """Each window directory must declare its expected role (None: a stand-in of any role). A reduced run opens no
    readout role. Reads only windows.json."""
    err = []
    for arg, (d, want) in dirs.items():
        role = window_role(d)
        if reduced and refused_roles(role, READOUT_ROLES):
            err.append(f"--{arg} {d}: role {role!r} may not be opened by a reduced run")
        elif want is not None and role != want:
            err.append(f"--{arg} {d}: role {role!r}, expected {want!r}")
    return err


def window_dates(d: str | Path) -> list[str]:
    return json.loads((Path(d) / "windows.json").read_text())["dates"]


def binding_errors(target: str, ft: dict, train_dir: str | None, val_dir: str | None) -> list[str]:
    """A target's fine-tune against its recipe and seed, and against the window directories it was prepared from
    (--finetune-train, --finetune-val): each of its role and matching its SHA256SUMS, the hashes the fine-tune
    recorded equal to the directory's, and its text drawn from this target's rows of the donor readout windows."""
    want = {**FT_RECIPE, "seed": FT_BINDING[target]["seed"]}
    got = {k: ft["config"].get(k) for k in want}
    err = [] if got == want else [f"fine-tune config {got} is not the recipe of {target} {want}"]
    draw = {"role": FT_SOURCE_ROLE, "rows": FT_BINDING[target]["rows"]}
    for part, d, role in (("train", train_dir, TRAIN_ROLE), ("val", val_dir, VAL_ROLE)):
        if d is None:
            err.append(f"--finetune-{part} is required for {target}")
            continue
        try:
            w = load_windows(d, role=role)
        except ValueError as e:
            err.append(str(e))
            continue
        rec = ft.get("data", {}).get(part) or {}
        if rec.get("role") != w.role or rec.get("sha256") != w.sha256:
            err.append(f"fine-tune {part} data: the recorded role and hashes are not those of {d}")
        if part == "train" and ({k: w.source.get(k) for k in draw} != draw or rec.get("source") != w.source):
            err.append(f"{d}: fine-tune text from {w.source} (recorded: {rec.get('source')}) is not the draw of "
                       f"{target} {draw}")
    return err


def finetune_errors(ft_dir: str | Path) -> tuple[dict, list[str]]:
    ft = json.loads((Path(ft_dir) / "run.json").read_text())
    s = sha256_file(Path(ft_dir) / "update.pt")
    return ft, [] if s == ft["saved"]["update_sha256"] else [f"{ft_dir}/update.pt: sha256 differs from its run.json"]


def donor_repair_errors(path: str | Path, model_id: dict) -> tuple[dict, list[str]]:
    """The donor repair's identity: the pinned file, or a file of a donor run (its run.json beside it) that was not
    reduced, ended ok, passed its gate, loaded the same target model, and recorded this file's sha256, its chosen
    candidate and its fine-tune."""
    path = Path(path)
    s = sha256_file(path)
    if s == DONOR_REPAIR_SHA256:
        return {"sha256": s, "source": "pinned"}, []
    rec_path = path.parent / "run.json"
    if not rec_path.exists():
        return {"sha256": s}, [f"{path}: sha256 {s} is not the pinned donor repair's {DONOR_REPAIR_SHA256}, and no "
                               f"donor run record is beside it"]
    rec = json.loads(rec_path.read_text())
    ident = {"sha256": s, "source": "donor run", "run_record_sha256": sha256_file(rec_path)}
    got = {k: rec.get(k) for k in ("command", "status", "reduced", "donor_repair_sha256")}
    want = {"command": "donor", "status": "ok", "reduced": False, "donor_repair_sha256": s}
    err = [] if got == want else [f"{rec_path}: {got} is not a full, successful donor run that wrote {path.name}"]
    if rec.get("gate", {}).get("pass") is not True:
        err.append(f"{rec_path}: the donor repair did not pass its gate")
    if rec.get("inputs", {}).get("target_model") != model_id:
        err.append(f"{rec_path}: the donor run loaded another target model")
    if not err:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if obj.get("name") != rec["selection"]["selected"]:
            err.append(f"{path}: candidate {obj.get('name')!r} is not the run's choice "
                       f"{rec['selection']['selected']!r}")
        if obj.get("donor_update_sha256") != rec["inputs"]["finetune"]["update_sha256"]:
            err.append(f"{path}: fine-tune {obj.get('donor_update_sha256')} is not the run's fine-tune")
    return ident, err


def ft_record(ft: dict) -> dict:
    return {"update_sha256": ft["saved"]["update_sha256"], "selection": ft["selection"], "gates": ft["gates"],
            "lr": ft["config"]["lr"], "seed": ft["config"]["seed"], "saved_step": ft["saved"]["saved_step"]}


def block_inputs(a) -> tuple[dict, list[str]]:
    model_id, err = target_model_identity(a.assets)
    donor, e = donor_repair_errors(a.donor_repair, model_id)
    err += e
    ft, e = finetune_errors(a.finetune)
    err += e
    if a.target in FT_BINDING:
        err += binding_errors(a.target, ft, a.finetune_train, a.finetune_val)
    err += role_errors({"fit": (a.fit, "fit"), "selection": (a.selection, "selection"),
                        "readout": (a.readout, None if a.reduced else "readout")}, a.reduced)
    need = max(BUDGETS[B][g][0] for B in BLOCKS[a.block]["budgets"] for g in ("de", "nl", "ret"))
    if not err and len(window_dates(a.fit)) < need:  # fitting.fit_sets: the budgets' prefixes, never fewer windows
        err.append(f"--fit {a.fit}: fewer than the {need} windows of budgets {BLOCKS[a.block]['budgets']}")
    if not a.reduced and not err and len(window_dates(a.readout)) < N_READOUT_MIN:
        err.append(f"--readout {a.readout}: fewer than {N_READOUT_MIN} dates")
    return {"target_model": model_id, "donor_repair": donor, "finetune": ft_record(ft)}, err


def donor_inputs(a) -> tuple[dict, list[str]]:
    model_id, err = target_model_identity(a.assets)
    ft, e = finetune_errors(a.finetune)
    err += e
    if ft.get("target_model") != model_id:
        err.append(f"{a.finetune}: the fine-tune was trained on another target model")
    err += role_errors({"fit": (a.fit, "donor_fit"), "selection": (a.selection, VAL_ROLE),
                        "readout": (a.readout, None if a.reduced else "donor_readout")}, a.reduced)
    if not a.reduced and not err:
        dates = window_dates(a.readout)
        if len(dates) != N_DONOR_READOUT or tuple(dates[:len(PREVIEWED_READOUT)]) != PREVIEWED_READOUT:
            err.append(f"--readout {a.readout}: not the {N_DONOR_READOUT} donor readout dates with the previewed "
                       f"dates first")
    return {"target_model": model_id, "finetune": ft_record(ft)}, err


def windows(a, info: dict, arg: str, langs: tuple[str, ...], want: str | None, rows: int | None = None,
            prefixes: dict | None = None):
    """Load a window directory (role contract first, then SHA256SUMS) and record what was read in
    ``info['inputs']['windows'][arg]``: role, checked hashes, dates, source, rows and the fit prefixes (windows,
    tokens per window) of each budget, or None for whole windows."""
    w = load_windows(getattr(a, arg), langs, rows, role=want, refused=READOUT_ROLES if a.reduced else ())
    info["inputs"].setdefault("windows", {})[arg] = {
        "role": w.role, "sha256": w.sha256, "dates": w.dates, "source": w.source, "languages": list(w.tokens),
        "rows": len(w.dates), "prefixes": prefixes}
    return w


# --------------------------------------------------------------------------- shared helpers
def nonfinite_runs(runs: dict) -> dict:
    """The runs that stopped on a non-finite loss or gradient, with where they stopped and what they saved."""
    return {n: {k: r["status"].get(k) for k in ("nonfinite_step", "updates_applied", "batches_presented",
                                                 "checkpoints_saved")}
            for n, r in runs.items() if r["status"].get("finite") is False}


def failed_family_reason(fam: str, runs: dict, every: int) -> str:
    """Why a planned family has no candidate: the status of each of its runs (run names start with 'family:')."""
    parts = []
    for n, r in runs.items():
        if n.split(":")[0] != fam:
            continue
        st = r["status"]
        parts.append(f"{n.split(':', 1)[1]}: " + (f"non-finite at step {st.get('nonfinite_step')} after "
                                                    f"{st.get('updates_applied')} updates" if st.get("finite") is False
                                                    else "no checkpoint saved"))
    return (f"no checkpoint saved by any run (first checkpoint at step {every}): "
            + ("; ".join(parts) if parts else "no run"))


def nonfinite_log(lg: list) -> list[str]:
    return [f"step {r['step']}: {k}" for r in lg for k, v in r.items() if k != "step" and not finite(v)][:5]


def nonfinite_selection(vper: dict, names, sidx) -> dict:
    """Candidates with a non-finite per-window value or point estimate on the selection windows (the selection rule
    would drop them silently)."""
    bad = {}
    for n in names:
        why = [f"{lg}/{k}" for lg, vals in vper[n].items() for k, t in vals.items()
               if not bool(torch.isfinite(t[sidx]).all())]
        p = point(vper, n, sidx)
        why += [k for k in ("r_nl", "r_kl", "rho_de", "phi_de") if not finite(p.get(k))]
        why += [f"retain[{g}]" for g, v in (p.get("retain") or {}).items() if not finite(v)]
        if why:
            bad[n] = why[:5]
    return bad


def load_donor_repair(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


# --------------------------------------------------------------------------- one block of a target
def cmd_block(a) -> int:
    log, t0 = logger()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"REFUSED: {out} is not empty (no resume, use a new directory)", file=sys.stderr)
        return RC_REFUSED
    inputs, err = block_inputs(a)
    if err:
        print("REFUSED: inputs\n  " + "\n  ".join(err), file=sys.stderr)
        return RC_REFUSED
    out.mkdir(parents=True, exist_ok=True)
    blk = BLOCKS[a.block]
    fams, budgets = list(blk["families"]), list(blk["budgets"])
    model, base, precision = setup(a)
    D, ft = load_update(a.finetune)
    info = {"_t0": t0, "command": "block", "target": a.target, "block": a.block, "status": None, "rc": None,
            "reasons": [], "reduced": a.reduced, "deviations": a.deviations,
            "config": {"steps": a.steps, "every": a.every, "step_caps": list(a.step_caps), "rows": a.rows,
                       "polish_iters": a.polish_iters, "budgets": budgets, "families": fams,
                       "lrs": {f: LRS[f] for f in fams}, "eval_batch_size": a.eval_batch_size, "device": a.device,
                       "threads": a.threads},
            "inputs": inputs, "code": code_identity(), "precision": precision, "environment": environment(a.device),
            "utc_start": utc(), "budgets": {}, "runs": {}, "technical": {}, "checks": {"init": {}, "parity": {}}}
    tech = info["technical"]  # key -> reasons. Any entry makes the exit code 9

    def flag(key, why):
        tech.setdefault(key, []).append(why)
        log(f"TECHNICAL {key}: {why}")

    if not a.reduced and ft["target_model"] != inputs["target_model"]:
        flag("target", f"the fine-tune was trained on {ft['target_model']}, this run loads {inputs['target_model']}")
    D_finite = all(bool(torch.isfinite(D[m]).all()) for m in MODULES)
    if not D_finite:
        flag("target", "non-finite value in the fine-tune update D")
    cap = math.sqrt(sum(float(D[m].double().norm() ** 2) for m in MODULES))
    info["norm_cap"] = cap
    if not (math.isfinite(cap) and cap > 0):
        flag("target", f"cap ||D|| = {cap}")
    dobj = load_donor_repair(a.donor_repair)
    donor = {m: (b.float(), a_.float()) for m, (b, a_) in dobj["factors"].items()}
    if not all(bool(torch.isfinite(b).all()) and bool(torch.isfinite(a_).all()) for b, a_ in donor.values()):
        flag("target", "non-finite donor factor")
    rdonors = {f: randomized(donor, seed) for f, seed in RAND_SEED.items()}
    info["donor_repair"] = {**inputs["donor_repair"], "name": dobj.get("name"),
                            "cap_donor": dobj.get("cap_donor")}
    Dd = {m: D[m].to(a.device) for m in MODULES}
    raw_fro = fro_of({"kind": "scaled", "k": 1.0, "factors": donor}, Dd, a.device)
    k_fixed = min(1.0, cap / raw_fro)
    info["donor_raw_fro"], info["donor_fixed_factor"] = raw_fro, k_fixed
    info["donor_raw_over_cap"] = raw_fro / cap if cap > 0 else math.nan  # cap 0 is flagged above: the run ends at 9
    fixed = {"donor_raw": {"kind": "scaled", "k": 1.0, "factors": donor},
             "donor_fixed": {"kind": "scaled", "k": k_fixed, "factors": donor}}

    # ---- pre-readout gate: the fine-tune is usable and passed its gates. Damage floor on the selection windows
    sel_w = windows(a, info, "selection", LANGS, "selection", a.rows)
    sel_sets = sel_w.tokens
    sidx = torch.arange(sel_sets["de"].shape[0])
    stop = []
    if D_finite:  # a non-finite D cannot be applied (edits.applied_update). It is already technical
        t = time.time()
        vref = evaluate(model, base, D, {}, sel_sets, a.device, a.eval_batch_size)
        p_D = point(vref, "finetuned", sidx)
        info["damage_selection"] = {"d_nl_D": p_D["d_nl_D"], "d_de_D": p_D["d_de_D"],
                                    "floor_ok": bool(finite(p_D["d_nl_D"]) and p_D["d_nl_D"] >= DAMAGE_FLOOR),
                                    "eval_s": time.time() - t}
        nf = stats.nonfinite(vref, ("base", "finetuned"))
        if nf:
            flag("target", f"base/finetuned on the selection windows: {nf[:5]}")
    if not inputs["finetune"]["selection"].get("usable"):
        stop.append("the fine-tune has no usable checkpoint (finetune.py checkpoint rule)")
    if not inputs["finetune"]["gates"].get("passed"):
        stop.append(f"fine-tune gates failed: {inputs['finetune']['gates'].get('failed')}")
    if D_finite and not info["damage_selection"]["floor_ok"]:
        stop.append(f"selection damage floor: d_nl(D) = {info['damage_selection']['d_nl_D']} < {DAMAGE_FLOOR}")
    if tech:  # a technical failure takes precedence over the stop reasons: exit 9, never 10
        info["pre_readout_halt"] = {"rc": RC_TECHNICAL, "technical": dict(tech), "stop_reasons_not_applied": stop}
        return finish(out, info, RC_TECHNICAL, log, [f"technical before training: {sorted(tech)}"])
    if stop:
        info["pre_readout_stop"] = stop
        return finish(out, info, RC_PRE_READOUT_STOP, log, stop)

    # ---- training: every budget, family and learning rate
    fit_w = windows(a, info, "fit", FIT_LANGS, "fit", prefixes={B: BUDGETS[B] for B in budgets})
    shapes = {m: tuple(base[m].shape) for m in MODULES}
    meth_S = Method.strengths(donor, cap, a.device)
    E_start = {m: e.detach().double() for m, e in meth_S.E().items()}
    cands, cand_meta, fsets, planned, saved = {}, {}, {}, [], set()
    for B in budgets:
        fs = fit_sets(fit_w.tokens, B)
        fsets[B] = fs
        info["budgets"][B] = fit_accounting(fs)
        for fam in fams:
            planned.append(f"{fam}@{B}")
            for lr in LRS[fam]:
                meth = make_method(fam, shapes, a.device, cap, donor, rdonors)
                if fam not in info["checks"]["init"]:
                    ic = init_check(fam, meth, E_start, cap)
                    info["checks"]["init"][fam] = ic
                    if not ic["ok"]:
                        flag(fam, f"init check failed: {ic}")
                t = time.time()
                ck, lg, st = train(model, base, D, meth, fs, lr, a.device, log, f"{fam}@{B}", a.steps, a.every)
                st["train_s"] = time.time() - t
                st["checkpoints_saved"] = sorted(ck)
                st["c_init"] = meth.c0
                run = f"{fam}@{B}:lr={lr:g}"
                info["runs"][run] = {"status": st, "log": lg}
                if st.get("finite") is False:
                    flag(f"{fam}@{B}", f"{run}: non-finite loss/gradient at step {st.get('nonfinite_step')}")
                bad_log = nonfinite_log(lg)
                if bad_log:
                    flag(f"{fam}@{B}", f"{run}: non-finite logged values {bad_log}")
                mpc = st.get("max_post_over_cap")
                if mpc is not None and not (finite(mpc) and mpc <= 1 + CAP_TOL):
                    flag(f"{fam}@{B}", f"{run}: ||E|| / cap after projection = {mpc}")
                for step, s in ck.items():
                    saved.add(f"{fam}@{B}")
                    if nonfinite_snapshot(s):  # cannot be applied. Flagged and left out of every evaluation
                        flag(f"{fam}@{B}", f"{run}: non-finite parameters at checkpoint {step}")
                        continue
                    n = f"{run}:step={step}"
                    cands[n] = meth.spec(s, donor)
                    cand_meta[n] = {"family": f"{fam}@{B}", "budget": B, "lr": lr, "step": step}
    info["planned_families"] = planned
    info["failed_families"] = {f: failed_family_reason(f, info["runs"], a.every) for f in planned if f not in saved}
    for f, why in info["failed_families"].items():
        log(f"FAILED FAMILY {f}: {why}")
    info["nonfinite_runs"] = nonfinite_runs(info["runs"])

    # ---- exact fit loss of every candidate on its own budget's fit set. Polish of strengths (main block)
    t = time.time()
    fit, converged = {}, {}
    info["polish"] = {}
    exp_s, exp_n = {}, {}
    polished = blk["polish"] and "strengths" in fams
    for B in budgets:
        fl = FitLoss(model, base, D, fsets[B], a.device)
        names = [n for n in cands if cand_meta[n]["budget"] == B]
        for n in names:
            t_cand = time.time()
            fit[n] = fl.export(cands[n])
            grp = cand_meta[n]["family"]  # strengths@B: its count and seconds are the strengths_best start search
            exp_s[grp] = exp_s.get(grp, 0.0) + time.time() - t_cand
            exp_n[grp] = exp_n.get(grp, 0) + 1
        for n, spec in {**fixed, "finetuned": {"kind": "zero"}}.items():
            fit[f"{n}@{B}"] = fl.export(spec)
        if polished:
            s_names = [n for n in names if cand_meta[n]["family"] == f"strengths@{B}"
                       and all(finite(x) for x in fit[n].values())]
            starts = {"c0": torch.full((len(MODULES),), meth_S.c0)}
            if s_names:
                best = min(s_names, key=lambda n: (fit[n]["total"], s_names.index(n)))
                starts["best"] = cands[best]["c"]
                info["polish_best_start"] = best
            ref = None
            for tag, c_start in starts.items():  # c0 first: its start sets the stationarity scale of both
                rec, snap = polish(fl, meth_S, c_start, cap, a.polish_iters, log, tag, ref)
                info["polish"][tag] = rec
                if tag == "c0" and rec.get("finite"):
                    ref = {"stat": rec.get("stationarity_start"), "lam0": rec.get("lam0")}
                if not rec["finite"]:
                    flag(f"strengths_best@{B}", f"polish from {tag} not finite ({rec.get('stop')})")
                if snap is not None:
                    n = f"strengths_converged@{B}:start={tag}"
                    converged[n] = meth_S.spec(snap, donor)
                    cand_meta[n] = {"family": f"strengths_converged@{B}", "budget": B, "start": tag}
                    fit[n] = fl.export(converged[n])
        del fl
    info["fit_export_s"], info["fit_export_n"] = exp_s, exp_n
    info["s_adequacy_verified"] = bool(polished and len(info["polish"]) == 2
                                       and all(r.get("converged") for r in info["polish"].values()))
    if len(info["polish"]) == 2 and all(finite(r.get("L_end")) for r in info["polish"].values()):
        Ls = [r["L_end"] for r in info["polish"].values()]  # descriptive: do both starts reach the same loss?
        info["polish_agreement_rel"] = abs(Ls[0] - Ls[1]) / max(min(Ls), 1e-12)
    if polished and not info["s_adequacy_verified"]:
        log("strengths adequacy NOT verified (a polish did not converge, or the best-checkpoint start was missing)")
    info["fit_loss"] = fit
    info["fit_loss_s"] = time.time() - t
    for n, v in fit.items():
        if not all(finite(x) for x in v.values()):
            flag(cand_meta.get(n, {}).get("family", "target"), f"non-finite fit loss of {n}")
    cands.update(converged)

    # ---- selection
    allc = {**cands, **fixed}
    meta = {n: {**cand_meta.get(n, {"family": n}), "removed_fro": fro_of(spec, Dd, a.device)}
            for n, spec in allc.items()}
    removed_fro = {n: m["removed_fro"] for n, m in meta.items()}
    t = time.time()
    vper = evaluate(model, base, D, allc, sel_sets, a.device, a.eval_batch_size)
    info["selection_eval_s"] = time.time() - t
    log(f"selection evaluation of {len(allc)} candidates: {info['selection_eval_s']:.0f}s")
    nf = stats.nonfinite(vper, ("base", "finetuned"))
    if nf:
        flag("target", f"base/finetuned on the selection windows: {nf[:5]}")
    groups = {}
    for B in budgets:
        for fam in fams:
            groups[f"{fam}@{B}"] = [n for n in cands if cand_meta[n]["family"] == f"{fam}@{B}"]
        if polished:
            groups[f"strengths_best@{B}"] = groups[f"strengths@{B}"] + [n for n in converged
                                                                        if cand_meta[n]["budget"] == B]
        for fam in CAPPED:
            if fam in fams:
                for cap_s in a.step_caps:
                    if a.every <= cap_s < a.steps:  # a cap with at least one checkpoint and below the horizon
                        groups[f"{fam}@{B}:le{cap_s}"] = [n for n in groups[f"{fam}@{B}"]
                                                          if cand_meta[n]["step"] <= cap_s]
    info["selection_groups"] = {g: len(ns) for g, ns in groups.items()}
    bad_sel = nonfinite_selection(vper, list(cands), sidx)
    info["nonfinite_selection"] = bad_sel
    selections = {}
    for g, names in groups.items():
        parent = g.split(":")[0].replace("strengths_best@", "strengths@")
        bad = [n for n in names if n in bad_sel]
        if bad:
            flag(g, f"non-finite selection values: {len(bad)} of {len(names)} candidates (first {bad[0]}: "
                    f"{bad_sel[bad[0]]})")
        if parent in info["failed_families"] or not names:
            selections[g] = {"selected": None, "n_feasible": 0, "tried": [],
                             "failed": info["failed_families"].get(parent, "no candidate")}
        else:
            selections[g] = select(vper, removed_fro, names, sidx)
    info["selections"] = selections
    info["selection_points"] = {n: point(vper, n, sidx) for n in allc}

    # ---- checks of the chosen candidates and descriptive directions
    chosen = dict(fixed)
    chosen.update(converged)
    for g, sel in selections.items():
        if sel["selected"]:
            chosen[f"{g}::{sel['selected']}"] = cands[sel["selected"]]
            info["checks"]["parity"][g] = parity(model, base, D, cands[sel["selected"]], sel_sets["nl"][:4],
                                                 a.device)
    for n in converged:
        info["checks"]["parity"][n] = parity(model, base, D, cands[n], sel_sets["nl"][:4], a.device)
    for k, v in info["checks"]["parity"].items():
        if not (finite(v) and v <= PARITY_TOL):  # keyed by the group (a converged candidate: strengths_best)
            flag(k if k in selections else f"strengths_best@{cand_meta[k]['budget']}", f"parity {k}: {v}")
    info["direction"] = {}
    ref = {g: f"{g}::{sel['selected']}" for g, sel in selections.items() if sel["selected"]}
    for g, nm in ref.items():
        info["direction"][f"{g} vs donor_raw"] = direction(chosen[nm], fixed["donor_raw"], Dd, a.device)
    for B in budgets:
        for x, y in DIRECTION_PAIRS:
            gx, gy = f"{x}@{B}", f"{y}@{B}"
            if gx in ref and gy in ref:
                info["direction"][f"{gx} vs {gy}"] = direction(chosen[ref[gx]], chosen[ref[gy]], Dd, a.device)

    # ---- readout: once, after every selection is made
    rd_w = windows(a, info, "readout", LANGS, None if a.reduced else "readout", a.rows)
    t = time.time()
    rper = evaluate(model, base, D, chosen, rd_w.tokens, a.device, a.eval_batch_size)
    info["readout_eval_s"] = time.time() - t
    info["readout_role"], info["readout_dates"] = rd_w.role, rd_w.dates
    info["norms"] = {n: meta[n.split("::")[-1]]["removed_fro"] for n in chosen}
    for n in rper:  # keyed by the group whose readout is not finite (base and finetuned: the target)
        nf = stats.nonfinite(rper, (n,))
        if nf:
            flag(n.split("::")[0] if "::" in n else "target" if n in ("base", "finetuned")
                 else f"strengths_best@{cand_meta[n]['budget']}" if n in converged else n,
                 f"non-finite readout values: {nf[:5]} ({len(nf)} tensors)")

    if tech:
        return finish(out, info, RC_TECHNICAL, log, [f"technical: {sorted(tech)}"])
    if info["failed_families"]:
        return finish(out, info, RC_FAILED_FAMILY, log, [f"failed families: {sorted(info['failed_families'])}"])

    def export(record_sha256):
        block = {
            "format": 1, "study": "repair_transfer", "target": a.target, "block": a.block,
            "protocol": {"budgets": budgets, "families": fams, "capped_families": [f for f in CAPPED if f in fams],
                         "steps": a.steps, "every": a.every, "step_caps": list(a.step_caps), "polish": blk["polish"]},
            "candidates": [{"name": n, "family": m["family"], "removed_fro": m["removed_fro"],
                            **{k: m[k] for k in ("budget", "lr", "step", "start") if k in m}} for n, m in meta.items()],
            "selection": {"rows": int(sidx.numel()), "names": list(vper)},
            "readout": {"dates": rd_w.dates, "names": list(rper)},
            "recorded": {
                "selections": {g: {"selected": s["selected"], "n_feasible": s["n_feasible"]}
                               for g, s in selections.items()},
                "group_sizes": info["selection_groups"],
                "s_adequacy_verified": info["s_adequacy_verified"],
                "polish": {tag: {k: rec[k] for k in POLISH_KEYS} for tag, rec in info["polish"].items()},
                "polish_best_start": info.get("polish_best_start"),
                "polish_agreement_rel": info.get("polish_agreement_rel"),
                "selection_damage_d_nl": info["damage_selection"]["d_nl_D"],
                "fit_tokens": {b: v["total_unique_predicted_tokens"] for b, v in info["budgets"].items()},
                "fit_loss": fit,
                "fit_loss_evaluations": {g: {"n": exp_n[g], "s": exp_s[g]} for g in exp_n},
                "runs": {k: {"updates_applied": r["status"]["updates_applied"], "train_s": r["status"]["train_s"]}
                         for k, r in info["runs"].items()},
                "selection_eval_s": info["selection_eval_s"],
                "total_s": info["total_s"],
                "norm_cap": cap,
                "donor_fixed_factor": k_fixed,
                "direction_cos": {k: v["cos"] for k, v in info["direction"].items()},
                "finetune_update_sha256": inputs["finetune"]["update_sha256"],
            },
            "source": {"run_record_sha256": record_sha256},
            "run": {"reduced": a.reduced, "deviations": a.deviations},
        }
        write_block(out, block, vper, rper)

    return finish(out, info, RC_OK, log, [], export)


# --------------------------------------------------------------------------- the donor repair
def cmd_donor(a) -> int:
    log, t0 = logger()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"REFUSED: {out} is not empty (no resume, use a new directory)", file=sys.stderr)
        return RC_REFUSED
    inputs, err = donor_inputs(a)
    if err:
        print("REFUSED: inputs\n  " + "\n  ".join(err), file=sys.stderr)
        return RC_REFUSED
    out.mkdir(parents=True, exist_ok=True)
    model, base, precision = setup(a)
    D, ft = load_update(a.finetune)
    info = {"_t0": t0, "command": "donor", "status": None, "rc": None, "reasons": [], "reduced": a.reduced,
            "deviations": a.deviations,
            "config": {"steps": a.steps, "every": a.every, "rows": a.rows, "lrs": DONOR_LRS,
                       "eval_batch_size": a.eval_batch_size, "device": a.device, "threads": a.threads},
            "inputs": inputs, "code": code_identity(), "precision": precision, "environment": environment(a.device),
            "utc_start": utc(), "runs": {}}
    tech = []  # technical reasons (exit 9): the run stops at the stage that has one. No donor_repair.pt is written

    def technical():
        info["technical"] = {"donor": tech}
        return finish(out, info, RC_TECHNICAL, log, tech)

    if not all(bool(torch.isfinite(D[m]).all()) for m in MODULES):
        tech.append("non-finite value in the fine-tune update D")
        return technical()
    cap = math.sqrt(sum(float(D[m].double().norm() ** 2) for m in MODULES))
    fs = fit_sets(windows(a, info, "fit", FIT_LANGS, "donor_fit", a.rows).tokens, None)
    info.update(cap=cap, fit=fit_accounting(fs))
    shapes = {m: tuple(base[m].shape) for m in MODULES}
    cands, nonfinite = {}, []
    for lr in DONOR_LRS:
        meth = make_method("fresh", shapes, a.device, cap, {}, {})
        t = time.time()
        ck, lg, st = train(model, base, D, meth, fs, lr, a.device, log, "donor_repair", a.steps, a.every)
        st["checkpoints_saved"] = sorted(ck)
        info["runs"][f"{lr:g}"] = {"status": st, "train_s": time.time() - t, "log": lg}
        mpc = st.get("max_post_over_cap")
        if mpc is not None and not (finite(mpc) and mpc <= 1 + CAP_TOL):
            tech.append(f"run lr={lr:g}: ||E|| / cap after projection = {mpc}")
        for step, s in ck.items():
            n = f"donor_repair:lr={lr:g}:step={step}"
            if nonfinite_snapshot(s):  # cannot be applied: left out of the selection, as no non-finite one is chosen
                nonfinite.append(n)
            else:
                cands[n] = meth.spec(s)
    info["nonfinite_runs"] = nonfinite_runs(info["runs"])
    info["nonfinite_snapshots"] = nonfinite
    tech += [f"run lr={lr}: non-finite at step {r['nonfinite_step']}" for lr, r in info["nonfinite_runs"].items()]
    tech += [f"non-finite checkpoint {n}" for n in nonfinite]
    if tech:  # no selection, readout or donor_repair.pt
        return technical()
    info["failed_families"] = {}
    if not info["runs"] or not any(r["status"]["checkpoints_saved"] for r in info["runs"].values()):
        info["failed_families"]["donor_repair"] = failed_family_reason(
            "donor_repair", {f"donor_repair:lr={k}": r for k, r in info["runs"].items()}, a.every)
        log(f"FAILED FAMILY donor_repair: {info['failed_families']['donor_repair']}")
    # selection on the fine-tune's validation windows
    sel_sets = windows(a, info, "selection", LANGS, VAL_ROLE, a.rows).tokens
    Dd = {m: D[m].to(a.device) for m in MODULES}
    removed_fro = {n: fro_of(spec, Dd, a.device) for n, spec in cands.items()}
    t = time.time()
    vper = evaluate(model, base, D, cands, sel_sets, a.device, a.eval_batch_size)
    info["selection_eval_s"] = time.time() - t
    idx = torch.arange(sel_sets["de"].shape[0])
    tech += stats.nonfinite(vper, ("base", "finetuned"))[:5]
    info["nonfinite_selection"] = nonfinite_selection(vper, list(cands), idx)  # the rule would drop them silently
    tech += [f"non-finite selection values of {n}: {why}" for n, why in info["nonfinite_selection"].items()][:5]
    sel = select(vper, removed_fro, list(cands), idx)
    info["selection"] = sel
    info["selection_points"] = {n: point(vper, n, idx) for n in cands}
    p_dev = point(vper, "finetuned", idx)  # the donor's own Dutch damage, as for targets
    info["damage_selection"] = {"d_nl_D": p_dev["d_nl_D"], "d_de_D": p_dev["d_de_D"],
                                "floor_ok": p_dev["d_nl_D"] >= DAMAGE_FLOOR}
    log(f"donor repair selection: {sel['selected']}")
    if sel["selected"]:
        cspec = cands[sel["selected"]]
        info["parity_max_abs_logit"] = parity(model, base, D, cspec, sel_sets["nl"][:4], a.device)
        info["E_fro"] = removed_fro[sel["selected"]]
        info["E_over_cap"] = info["E_fro"] / cap
        if not (finite(info["parity_max_abs_logit"]) and info["parity_max_abs_logit"] <= PARITY_TOL):
            tech.append(f"parity of {sel['selected']}: {info['parity_max_abs_logit']}")
        if not (finite(info["E_over_cap"]) and info["E_over_cap"] <= 1 + CAP_TOL):
            tech.append(f"||E|| / cap of {sel['selected']}: {info['E_over_cap']}")
    if tech:  # before the readout: no readout and no donor_repair.pt
        return technical()
    if sel["selected"]:
        rd_w = windows(a, info, "readout", LANGS, None if a.reduced else "donor_readout", a.rows)
        t = time.time()
        rper = evaluate(model, base, D, {"donor_repair": cspec}, rd_w.tokens, a.device, a.eval_batch_size)
        info["readout_eval_s"] = time.time() - t
        tech += stats.nonfinite(rper, list(rper))[:5]
        if tech:
            return technical()
        prev = set(PREVIEWED_READOUT)
        pidx = torch.tensor([i for i, d in enumerate(rd_w.dates) if d not in prev])
        info["readout"] = {"role": rd_w.role, "dates": rd_w.dates,
                           "previewed_dates": [d for d in rd_w.dates if d in prev], "primary_idx": pidx.tolist()}
        p = point(rper, "donor_repair", pidx)  # primary: previewed dates excluded
        p_all = point(rper, "donor_repair", torch.arange(len(rd_w.dates)))  # disclosed sensitivity
        dD = rper["finetuned"]["nl"]["ce"] - rper["base"]["nl"]["ce"]
        bd = dD[pidx[stats.boot_indices(len(pidx))]].mean(1).sort().values
        lo = float(torch.quantile(bd, 0.025))
        info["damage_readout"] = {"d_nl_D": float(dD[pidx].mean()), "ci_lo": lo, "ci_lo_ok": lo >= DONOR_DAMAGE_CI_LO}
        # finite tensors can still give undefined estimates (a zero mean damage): the gate cannot be read
        tech += [f"non-finite readout estimate {tag}/{k}: {q[k]}" for tag, q in (("primary", p), ("all", p_all))
                 for k in stats.CHECKED if not finite(q[k])]
        tech += [f"non-finite readout estimate {tag}/retain[{g}]: {v}" for tag, q in (("primary", p), ("all", p_all))
                 for g, v in q["retain"].items() if not finite(v)]
        tech += [f"non-finite readout damage {k}: {v}" for k, v in info["damage_readout"].items()
                 if k != "ci_lo_ok" and not finite(v)]
        if tech:
            return technical()
        gate = {"r_nl": p["r_nl"] >= DONOR_GATE_R_NL, "german": german_feasible(p),
                "r_kl_pos": bool(math.isfinite(p["r_kl"]) and p["r_kl"] > 0), "retain": bool(p.get("retain_ok"))}
        damage = {"selection_floor": info["damage_selection"]["floor_ok"],
                  "readout_ci_lo": info["damage_readout"]["ci_lo_ok"]}
        ft_ok = {"donor_ft_gates_passed": bool(inputs["finetune"]["gates"].get("passed"))}
        info["gate"] = {"checks": gate, "finetune": ft_ok, "damage": damage,
                        "pass": all(gate.values()) and all(ft_ok.values()) and all(damage.values()),
                        "point": p, "point_all_incl_previewed": p_all, "n_primary": len(pidx)}
        torch.save({"factors": cspec["factors"], "name": sel["selected"], "cap_donor": cap,
                    "donor_update_sha256": inputs["finetune"]["update_sha256"]}, out / "donor_repair.pt")
        info["donor_repair_sha256"] = sha256_file(out / "donor_repair.pt")
        log(f"donor repair gate: {info['gate']['pass']} r_nl={p['r_nl']:.4f} rho={p['rho_de']:.4f} "
            f"phi={p['phi_de']:.5f} retain={p.get('retain')}")
    else:
        info["gate"] = {"pass": False, "reason": info["failed_families"].get("donor_repair")
                        or "no feasible donor repair checkpoint on the selection windows"}
    if info["failed_families"]:
        return finish(out, info, RC_FAILED_FAMILY, log, [info["failed_families"]["donor_repair"]])

    def export(record_sha256):  # the inputs of the donor repair's one-time cost (data.load_donor)
        if not sel["selected"] or a.reduced:
            return
        fit_keys = ("unique_predicted_tokens", "windows", "total_unique_predicted_tokens")
        rec = {"format": 1, "study": "repair_transfer", "record": "donor_repair_fit",
               "fit": {k: info["fit"][k] for k in fit_keys},
               "runs": {lr: {k: r["status"][k] for k in ("batch", "batches_presented")}
                        for lr, r in info["runs"].items()},
               "total_s": info["total_s"], "source": {"run_record_sha256": record_sha256}}
        (out / "fit_record.json").write_text(json.dumps(rec, indent=1) + "\n")
        (out / "SHA256SUMS").write_text("".join(f"{sha256_file(out / f)}  {f}\n" for f in DONOR_FILES))

    return finish(out, info, RC_OK, log, [], export)


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fit and compare ways to reuse an earlier repair on a newly fine-tuned model.",
        epilog="Usage and outputs: papers/guides/repair-transfer.md")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("block", "donor"):
        q = sub.add_parser(name, help={
            "block": "compare repair strategies on a new fine-tuned model",
            "donor": "learn the earlier repair that later targets can reuse",
        }[name])
        q.add_argument("--assets", required=True, help="assets directory with target/")
        q.add_argument("--finetune", required=True, help="fine-tune directory (finetune.py train)")
        q.add_argument("--fit", required=True, help="window directory of the fit windows")
        q.add_argument("--selection", required=True, help="window directory of the selection windows")
        q.add_argument("--readout", required=True, help="window directory of the readout windows")
        q.add_argument("--out", required=True, help="new directory for measurements and the run record")
        q.add_argument("--device", default="cuda")
        q.add_argument("--threads", type=int, default=4)
        q.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH)
        q.add_argument("--steps", type=int, default=STEPS if name == "block" else DONOR_STEPS)
        q.add_argument("--every", type=int, default=EVERY if name == "block" else DONOR_EVERY)
        q.add_argument("--rows", type=int, default=None, help="reduced runs: first rows of the evaluation windows")
        if name == "block":
            q.add_argument("--target", required=True, choices=TARGETS)
            q.add_argument("--block", required=True, choices=sorted(BLOCKS))
            q.add_argument("--donor-repair", required=True, help="donor_repair.pt")
            q.add_argument("--finetune-train", default=None, help="the fine-tune's training window directory "
                           "(primary_target, replication_target)")
            q.add_argument("--finetune-val", default=None, help="the fine-tune's validation window directory "
                           "(primary_target, replication_target)")
            q.add_argument("--step-caps", default=",".join(map(str, STEP_CAPS)))
            q.add_argument("--polish-iters", type=int, default=POLISH["max_iter"])
    a = ap.parse_args(argv)
    frozen = {"steps": STEPS, "every": EVERY} if a.cmd == "block" else {"steps": DONOR_STEPS, "every": DONOR_EVERY}
    frozen["rows"] = None
    if a.cmd == "block":
        try:
            a.step_caps = tuple(sorted({int(x) for x in a.step_caps.split(",") if x}))
        except ValueError:
            ap.error("--step-caps must be comma-separated integers")
        frozen.update(step_caps=STEP_CAPS, polish_iters=POLISH["max_iter"])
    a.deviations = {k: getattr(a, k) for k, v in frozen.items() if getattr(a, k) != v}
    if a.cmd == "donor":
        a.reduced = bool(a.deviations)
        return cmd_donor(a)
    if a.deviations and a.target != "standin":
        ap.error(f"{', '.join(a.deviations)} away from the frozen values need --target standin")
    a.reduced = a.target == "standin"
    return cmd_block(a)


if __name__ == "__main__":
    raise SystemExit(main())
