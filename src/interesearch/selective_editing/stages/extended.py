"""Model stages for exploratory additional comparisons, after the main study's freeze.
Rules: papers/selective-editing/index.qmd#sec-extended-comparisons.

The additional comparisons use their own run folder. The runner reads the main study's frozen run folder (``--main``): the VPD selection
and its ridge floor, the calibrated VPD edit it measures again, and the main dev readout of that edit. Items, run
binding and status rows are those of ``run.Runner``.

Pre-freeze, in order:
    Compute q of the VPD dictionary (for the check) and of the SVD dictionary, then K-FAC factors on the English retain pool.
    Per cell: the VPD selection again (it must reproduce the main study's atom, reach points and, within 5 %, its
    ridge floor), the SVD selection, the whitened rank-one selection (its own q, the main ridge floor and the
    whitened grid).
    Per condition and target: draw 0, seed 0 over the long learning-rate grid, the choice (a rate without a training
    record or with a non-finite scan or readout is a listed technical row), then every draw and seed with the chosen
    rate.
    The calibration of a cell (the comparators only, the VPD edit is the main study's).
A reduced run (``--reduced``, with a reduced main run folder) checks the code path only, as in ``run``.

The learned updates of the original additional comparisons were not kept, so its learned edits cannot be read again: a replay
from the model trains them anew (``data/selective_editing/extended_comparisons`` holds the records of the original run).

    uv run python -m interesearch.selective_editing.stages.extended prefreeze assets --main runs/selective_editing \\
        --out runs/selective_editing_extended_comparisons --gate runs/selective_editing/fisher_check.json
    uv run python -m interesearch.selective_editing.stages.extended freeze --main runs/selective_editing \\
        --out runs/selective_editing_extended_comparisons --gate runs/selective_editing/fisher_check.json --note "..."
    uv run python -m interesearch.selective_editing.stages.extended readout assets --main runs/selective_editing \\
        --out runs/selective_editing_extended_comparisons
"""

from __future__ import annotations

import argparse
import dataclasses
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from interesearch.assets import MODULES
from interesearch.checksums import sha256_file
from interesearch.selective_editing.stages import support as R
from interesearch.selective_editing.stages.access import FREEZE, AccessError, verify_freeze, write_freeze
from interesearch.selective_editing.analysis.extended import analyze, load, method_of
from interesearch.selective_editing.method.calibration import MAX_BISECT, TOL
from interesearch.selective_editing.method.comparators import (
    WHITENED_GRID,
    kfac_factors,
    svd_dictionary,
    whitened_dictionary,
)
from interesearch.selective_editing.data import (
    expected_cells,
    load_dev,
    write_cell,
    write_json,
    write_npz,
    write_sums,
)
from interesearch.selective_editing.method.dictionary import label_key
from interesearch.selective_editing.method.dose import (
    GUARD_MAX,
    PAIRS,
    Family,
    calibrate,
    declared,
    learned_family,
    missing,
)
from interesearch.selective_editing.method.finetune import (
    SEED_OFFSET,
    SCALE_GRID,
    choose_lr,
    config,
    load_update,
    train_run,
)
from interesearch.selective_editing.stages.readout import dev_sets, read_dev, rebuild, standin_sets
from interesearch.selective_editing.method.selector import (
    ATOM_GRID,
    FISHER_K,
    TAUS,
    q_record,
    select,
    target_gradient,
)
from interesearch.selective_editing.method.runtime import require_gate, code_identity, provenance, stored

MAIN = "vpd_atom"
SELECTED = ("svd_atom", "whitened_rank1")
CONDITIONS = {"lora": "lora_rank1_long", "matrix": "finetune_24_long"}
LABELS = {
    "vpd_atom": "the main study's VPD atom edit at its calibrated scale, measured again with the additional comparisons",
    "svd_atom": "g^2/q selector, SVD atoms of the 24 weight matrices (same rule as the main selector)",
    "whitened_rank1": "dictionary-free rank-one selector: per matrix the top singular pair of the K-FAC whitened "
                      "target gradient, scaled to the norm of the cell's VPD atom; same score and walk",
    "lora_rank1_long": "rank-1 LoRA in the selected atom's module, 2000 steps, margin 1.5, three seeds",
    "finetune_24_long": "full-matrix fine-tune of all 24 modules, 2000 steps, margin 1.5, three seeds",
}
LEARNING_RATES = {"lora": (1e-3, 3e-3, 1e-2, 3e-2), "matrix": (1e-7, 3e-7, 1e-6, 3e-6)}
STEPS, EVERY, MARGIN = 2000, 100, 1.5
SEEDS = (0, 1, 2)
PLATEAU = (1500, 2000, 0.10)  # reported only
EPS_TOLERANCE = 0.05  # the VPD check: eps within the label-sampling SE of the curvature check, rounded up
BAND = 0.02
LOG_FIELDS = ("hinge", "g_mean", "kl_retain", "grad_norm")
NO_RECORD = "technical: no training record (failed, blocked or not run)"


# --------------------------------------------------------------------------- run folder layout
def run_name(t: str, d: int, condition: str, lr: float, k: int) -> str:
    return f"{t}_draw{d}_{CONDITIONS[condition]}_lr{lr:g}_s{k}"


def run_dir(out: Path, t: str, d: int, condition: str, lr: float, k: int) -> Path:
    return out / "finetune" / run_name(t, d, condition, lr, k)


def choice_path(out: Path, t: str, condition: str) -> Path:
    return out / "finetune" / f"{t}_{CONDITIONS[condition]}_lr_choice.json"


def labels_for(recipe: dict | None = None) -> dict:
    labels = dict(LABELS)
    if recipe is not None:
        ext, training = recipe["extended"], recipe["training"]
        seed_text = "three seeds" if len(ext["seeds"]) == 3 else f"{len(ext['seeds'])} seeds"
        labels["lora_rank1_long"] = (f"rank-{training['rank']} LoRA in the selected atom's module, "
            f"{ext['steps']} steps, margin {ext['margin']}, {seed_text}")
        labels["finetune_24_long"] = (f"full-matrix fine-tune of all 24 modules, {ext['steps']} steps, "
            f"margin {ext['margin']}, {seed_text}")
    return labels


def methods_of_cell(recipe: dict | None = None) -> list[str]:
    return [*SELECTED, *(f"{m}_s{k}" for m in CONDITIONS.values() for k in (SEEDS if recipe is None else recipe["extended"]["seeds"]))]


# --------------------------------------------------------------------------- training
def long_config(condition: str, d: int, lr: float, k: int, reduced: bool, *, recipe: dict | None = None):
    """The main study's configuration with this protocol's seed (100 + draw + 1000 k), length, margin and validation
    steps. ``reduced`` keeps the reduced length."""
    cfg = config(condition, d, lr, reduced, recipe=recipe)
    ext = {} if recipe is None else recipe["extended"]
    steps, every = ext.get("steps", STEPS), ext.get("validation_every", EVERY)
    long = {} if reduced else {"steps": steps, "margin": ext.get("margin", MARGIN),
                              "checkpoints": tuple(range(every, steps + 1, every))}
    offset = SEED_OFFSET if recipe is None else recipe["training"]["seed_offset"]
    return dataclasses.replace(cfg, seed=offset + d + ext.get("seed_stride", 1000) * k, **long)


def convergence(train: dict, *, recipe: dict | None = None) -> dict:
    """Reported only: plateau, falling or rising between the PLATEAU steps. Unstable when the last English val KL is
    more than twice its minimum from step 100 on. The largest gradient norm after warm-up."""
    ext = {} if recipe is None else recipe["extended"]
    a, b = ext.get("plateau_steps", PLATEAU[:2])
    tol = ext.get("plateau_relative_change", PLATEAU[2])
    start, factor = ext.get("unstable_from_step", 100), ext.get("unstable_factor", 2.0)
    kl = {int(s): v["en"]["kl"] for s, v in train["val_by_step"].items()}
    last = max(kl)
    late = [v for s, v in kl.items() if s >= min(start, last)]
    rec = {"rule": f"plateau: English val KL changes by less than {tol:.0%} from step {a} to {b}; unstable: last KL "
                   f"> {factor:g} x its minimum from step {start} on", "unstable": kl[last] > factor * min(late),
           "max_grad_norm_after_warmup": max((r["grad_norm"] for r in train["log"]
                                              if r["step"] > train["config"]["warmup"]), default=None)}
    if a not in kl or b not in kl:
        return {**rec, "plateau": None, "state": None}
    change = (kl[b] - kl[a]) / kl[a] if kl[a] > 0 else 0.0
    state = "plateau" if abs(change) < tol else ("falling" if change < 0 else "rising")
    return {**rec, "kl": [kl[a], kl[b]], "rel_change": change, "state": state, "plateau": state == "plateau"}


def train(r: R.Runner, t: str, d: int, condition: str, lr: float, k: int) -> None:
    S = r.sizes
    if condition == "lora":  # the module of the main study's selected atom of the cell
        rec = json.loads(R.select_path(r.main, t, d, "vpd").read_text())
        modules = [(rec["accepted"] or rec["walk"][0])["module"]]
    else:
        modules = list(MODULES)
    train_run(run_dir(r.out, t, d, condition, lr, k), r.ctx.model, modules, r.draw(t, d),
              r.windows("retain_en", "en")[:S.retain_rows], r.windows("val", t)[:S.val_rows],
              r.windows("val_en", "en")[:S.val_rows], condition, long_config(condition, d, lr, k, r.reduced, recipe=r.recipe),
              r.ctx.device, r.reduced, header={"target": t, "draw": d, "seed_index": k}, recipe=r.recipe)


def choose(r: R.Runner, t: str, condition: str) -> None:
    """The main study's rate rule on draw 0, seed 0. A rate that cannot be judged is a technical row, not a pick."""
    val_t, val_en = r.windows("val", t)[:r.sizes.val_rows], r.windows("val_en", "en")[:r.sizes.val_rows]
    rows = []
    for lr in (LEARNING_RATES if r.recipe is None else r.recipe["extended"]["learning_rates"])[condition][:r.sizes.n_rates]:
        d = run_dir(r.out, t, 0, condition, lr, 0)
        if not (d / "train.json").exists():
            rows.append({"lr": lr, "status": NO_RECORD, "technical_failure": True})
            continue
        try:
            rows += choose_lr(r.ctx.model, {lr: load_update(d)}, val_t, val_en, r.ctx.device, r.ctx.bs, recipe=r.recipe)["rows"]
        except FloatingPointError as e:
            rows.append({"lr": lr, "status": str(e), "technical_failure": True})
    ok = [x for x in rows if x["status"] == "matched"]
    pick = min(ok, key=lambda x: (x["en_kl"], x["lr"])) if ok else None
    write_json(choice_path(r.out, t, condition),
               {"target": t, "method": CONDITIONS[condition], "tau": max(TAUS if r.recipe is None else r.recipe["taus"]), "rows": rows,
                "chosen_lr": None if pick is None else pick["lr"],
                "rule": "draw 0, seed 0: matched at the upper effect point on target val; lowest English val KL, "
                        "then the smaller rate; a rate without a record or with a non-finite scan or readout is a "
                        "technical row"})


def learned(r: R.Runner, t: str, condition: str, draws: list[int]) -> None:
    # Once the choice is written the grid is closed: a rate that failed before stays a technical row, not a retry.
    closed = choice_path(r.out, t, condition).exists()
    for lr in (LEARNING_RATES if r.recipe is None else r.recipe["extended"]["learning_rates"])[condition][:r.sizes.n_rates]:
        if not closed:
            r.item(f"finetune_{run_name(t, 0, condition, lr, 0)}", run_dir(r.out, t, 0, condition, lr, 0) /
                   "train.json", lambda lr=lr: train(r, t, 0, condition, lr, 0))
    choice = f"lr_choice_{t}_{CONDITIONS[condition]}"
    r.item(choice, choice_path(r.out, t, condition), lambda: choose(r, t, condition))
    for d in draws:
        for k in (SEEDS if r.recipe is None else r.recipe["extended"]["seeds"]):  # draw 0, seed 0 at the chosen rate is its grid run
            name = f"finetune_{t}_draw{d}_{CONDITIONS[condition]}_s{k}"
            if r.blocked(name, [choice]):
                continue
            lr = json.loads(choice_path(r.out, t, condition).read_text())["chosen_lr"]
            if lr is None:
                r.declare(name, "no learning rate chosen on draw 0")
                continue
            r.item(name, run_dir(r.out, t, d, condition, lr, k) / "train.json",
                   lambda d=d, k=k, lr=lr: train(r, t, d, condition, lr, k))


# --------------------------------------------------------------------------- selection
def factors(r: R.Runner) -> dict:
    """The K-FAC factors of every module on the English retain pool, stored under equal provenance."""
    if "factors" not in r.state_cache:
        retain = r.windows("retain_en", "en")[:r.sizes.q_rows]
        ctx = r.ctx
        prov = provenance(ctx.model, ctx.device, ctx.bs, {"retain": retain}, code_identity(), fisher_k=FISHER_K if r.recipe is None else r.recipe["selection"]["fisher_k"],
                          what="K-FAC factors")
        r.state_cache["factors"] = stored(r.out / "cache" / "kfac_factors.pt", prov, lambda: {
            "factors": kfac_factors(ctx.model, MODULES, retain, ctx.device, bs=ctx.bs,
                                  K=FISHER_K if r.recipe is None else r.recipe["selection"]["fisher_k"])})["factors"]
    return r.state_cache["factors"]


def selection(r: R.Runner, t: str, d: int, method: str) -> None:
    ctx, out, S = r.ctx, r.out, r.sizes
    retain, qcheck = r.windows("retain_en", "en")[:S.q_rows], r.windows("qcheck_en", "en")[:S.q_rows]
    tokens, g_cache = r.draw(t, d), out / "cache" / f"G_{t}_draw{d}.pt"
    main_sel = json.loads(R.select_path(r.main, t, d, "vpd").read_text())
    main_eps = json.loads((r.main / "q" / "vpd.json").read_text())["eps"]
    grid, eps, extra = ATOM_GRID if r.recipe is None else r.recipe["selection"]["atom_grid"], None, {}
    if method == MAIN:
        dic, rule = ctx.vpd, "1st percentile of q/||A||^2 (VPD dictionary, computed again)"
    elif method == "svd_atom":
        dic, rule = r.svd(), "1st percentile of q/||A||^2 (SVD dictionary)"
    else:
        if main_sel["accepted"] is None:
            write_json(R.select_path(out, t, d, method), {"target": t, "draw": d, "method": method, "accepted": None,
                                                          "status": "no main atom: no norm to borrow"})
            return
        G, _ = target_gradient(ctx.model, MODULES, tokens, ctx.device, ctx.bs, g_cache)
        dic = whitened_dictionary(G, factors(r), main_eps, main_sel["accepted"]["norm"], f"kfac_{t}_d{d}")
        grid, eps, rule = (WHITENED_GRID if r.recipe is None else r.recipe["extended"]["whitened_grid"]), main_eps, "eps of the main selector (main study)"
        extra = {"whitened_max": dic.meta["whitened_max"]}
    q = q_record(out / "q" / f"{dic.name}.pt", ctx.model, dic, retain, qcheck, ctx.device, ctx.gate, ctx.bs, recipe=r.recipe)
    rec, _ = select(ctx.model, dic, q, tokens, ctx.device, ctx.gate, ctx.bs, g_cache, S.n_short, grid, eps, recipe=r.recipe)
    rec = {"target": t, "draw": d, "method": method, "seed_name": dic.name, "eps_rule": rule, **rec, **extra}
    acc = rec["accepted"]
    if method == MAIN:
        keys = ("module", "index", "sign", "reach_lo", "reach_hi")
        ref = main_sel["accepted"]
        same = rec["status"] == main_sel["status"] and (acc is ref is None or None not in (acc, ref) and all(
            acc[k] == ref[k] for k in keys))  # the same atom and reach points, or no atom for the same reason
        rec |= {"eps_rel_diff": rec["eps"] / main_eps - 1,
                "main_study": {"status": main_sel["status"], "accepted": ref and {k: ref[k] for k in keys},
                               "eps": main_eps}}
        rec |= {"reproduces_main": same and abs(rec["eps_rel_diff"]) < (EPS_TOLERANCE if r.recipe is None else r.recipe["extended"]["eps_tolerance"]),
                "reproduces_main_strict": same and abs(rec["eps_rel_diff"]) < 1e-4}
    elif acc is not None:
        m, c = acc["module"], acc["index"]
        p = R.select_path(out, t, d, method).with_suffix(".pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"module": m, "index": c, "u": dic[m].U[c].clone(), "v": dic[m].V[:, c].clone()}, p)
        rec["atom_sha256"] = sha256_file(p)
    write_json(R.select_path(out, t, d, method), rec)


def saved_atom(path: Path, sha256: str) -> dict:
    """A selected comparator atom (module, index, u, v), refused unless its bytes match the recorded sha256."""
    if sha256_file(path) != sha256:
        raise AccessError(f"{path}: sha256 does not match its record")
    return torch.load(path, weights_only=True)


# --------------------------------------------------------------------------- calibration
def families(r: R.Runner, t: str, d: int) -> dict[str, Family | tuple[str, bool]]:
    out, dev = r.out, r.ctx.device
    fam: dict[str, Family | tuple[str, bool]] = {}
    for method, grid in zip(SELECTED, (ATOM_GRID, WHITENED_GRID) if r.recipe is None else
                            (r.recipe["selection"]["atom_grid"], r.recipe["extended"]["whitened_grid"])):
        p = R.select_path(out, t, d, method)
        if not p.exists():
            fam[method] = missing(R.rel(out, p))
            continue
        rec = json.loads(p.read_text())
        acc = rec["accepted"]
        if acc is None:
            fam[method] = ((f"selector failed: {rec['status']}", True) if rec.get("technical_failure")
                           else declared(f"selector status {rec['status']!r}"))
            continue
        a = saved_atom(p.with_suffix(".pt"), rec["atom_sha256"])
        A, m, sign = torch.outer(a["u"], a["v"]), a["module"], acc["sign"]
        fam[method] = Family(lambda x, A=A, m=m, sign=sign: {m: (sign * x * A).to(dev, torch.float32)}, grid,
                             {"rank": 1, "modules": [m],
                              "saved_atom": {"file": R.rel(out, p.with_suffix(".pt")), "sha256": rec["atom_sha256"],
                                             "label": label_key(m, a["index"]), "sign": sign,
                                             "norm": float(A.norm())}},
                             {"source": R.rel(out, p), "status": rec["status"], "walk_rank": acc["rank"]})
    for condition, method in CONDITIONS.items():
        ch = choice_path(out, t, condition)
        for k in (SEEDS if r.recipe is None else r.recipe["extended"]["seeds"]):
            name = f"{method}_s{k}"
            if not ch.exists():
                fam[name] = missing(R.rel(out, ch))
                continue
            choice = json.loads(ch.read_text())
            lr = choice["chosen_lr"]
            if lr is None:
                tech = [f"{x['lr']:g}" for x in choice["rows"] if x.get("technical_failure")]
                fam[name] = ((f"technical: no rate chosen; technical rows for lr {tech}", True) if tech
                             else declared("no learning rate reached the upper effect point on draw 0"))
                continue
            run = run_dir(out, t, d, condition, lr, k)
            if not (run / "train.json").exists():
                fam[name] = missing(f"{R.rel(out, run)}/train.json")
                continue
            tr = json.loads((run / "train.json").read_text())
            f = learned_family(load_update(run), R.rel(out, run), tr, (LEARNING_RATES if r.recipe is None else r.recipe["extended"]["learning_rates"])[condition][:r.sizes.n_rates], dev, recipe=r.recipe)
            f.identity["learned_update"] |= {"seed_index": k, "training_run": run.name}
            f.search["convergence"] = convergence(tr, recipe=r.recipe)
            fam[name] = f
    return fam


def calibration(r: R.Runner, t: str, d: int) -> None:
    r.access.blind("calibration")
    n = r.sizes.val_rows
    labels = {m: labels_for(r.recipe)[method_of(m)] for m in methods_of_cell(recipe=r.recipe)}
    rec = calibrate(r.ctx.model, r.base, r.windows("val", t)[:n], r.windows("val_en", "en")[:n], families(r, t, d),
                    r.ctx.device, r.ctx.bs, labels=labels, recipe=r.recipe)
    write_json(R.cell_path(r.out, "calibration", t, d),
               {"target": t, "draw": d, "val_set": f"europarl_val_{t}", "reduced": r.reduced, **rec})


def prefreeze(r: R.Runner, cells: list[tuple[str, int]]) -> None:
    out = r.out
    r.access.blind("prefreeze")
    require_gate(r.ctx.gate, FISHER_K if r.recipe is None else r.recipe["selection"]["fisher_k"],
                 eps_percentile=1.0 if r.recipe is None else r.recipe["selection"]["eps_percentile"],
                 null_seed=101 if r.recipe is None else r.recipe["selection"]["null_seeds"][0])
    for t, d in cells:
        for method in (MAIN, *SELECTED):
            r.item(f"select_{t}_draw{d}_{method}", R.select_path(out, t, d, method),
                   lambda t=t, d=d, method=method: selection(r, t, d, method))
    targets = sorted({t for t, _ in cells}, key=R.TARGETS.index)
    for condition in CONDITIONS:
        for t in targets:
            learned(r, t, condition, [d for tt, d in cells if tt == t])
    for t, d in cells:
        r.item(f"calibration_{t}_draw{d}", R.cell_path(out, "calibration", t, d), lambda t=t, d=d: calibration(r, t, d),
               [f"select_{t}_draw{d}_{m}" for m in SELECTED])


def freeze(out: Path, gate: Path, cells: list[tuple[str, int]], allow_incomplete: bool, note: str) -> dict:
    """The additional comparisons’ FREEZE.json (``access.write_freeze``), refused unless every cell's VPD selection reproduced
    the main study's."""
    bad = [f"{t}:{d}" for t, d in cells if not (p := R.select_path(out, t, d, MAIN)).exists()
           or json.loads(p.read_text()).get("reproduces_main") is not True]
    if bad:
        raise AccessError(f"freeze refused: the VPD selection is missing or not reproduced in {bad}")
    return write_freeze(out, gate, cells, allow_incomplete, note)


# --------------------------------------------------------------------------- readout, export, analysis
def readout(r: R.Runner, cells: list[tuple[str, int]]) -> None:
    out, ctx, main = r.out, r.ctx, r.main
    verify_freeze(out)

    def make_of(m: str, rec: dict):
        if "saved_atom" in rec:
            a = saved_atom(out / rec["saved_atom"]["file"], rec["saved_atom"]["sha256"])
            A, s = torch.outer(a["u"], a["v"]), rec["saved_atom"]["sign"]
            return lambda x: {a["module"]: (s * x * A).to(ctx.device, torch.float32)}
        return rebuild(rec, main if m == MAIN else out, ctx.model, r.dictionary, ctx.device)

    def read(t: str, d: int) -> None:
        t0 = time.time()
        sets, ce0 = R.sets_for(r, ("dev", t), lambda: (standin_sets if r.access.standin else dev_sets)(r.access, t, recipe=r.recipe))
        cal = json.loads(R.cell_path(out, "calibration", t, d).read_text())
        vpd = json.loads(R.cell_path(main, "calibration", t, d).read_text())["methods"][MAIN]
        edits = read_dev(ctx.model, r.base, {**cal, "methods": {MAIN: vpd, **cal["methods"]}}, sets, ce0, make_of,
                         ctx.device, ctx.bs, recipe=r.recipe)
        before = json.loads(R.cell_path(main, "readout", t, d).read_text())["edits"]
        diff = [abs(a - b) for tau in cal["taus"] if edits[f"{MAIN}@{tau}"]["status"] == "matched"
                for n in sets.tokens for a, b in zip(edits[f"{MAIN}@{tau}"]["readout"][n]["delta_ce_doc"],
                                                     before[f"{MAIN}@{tau}"]["readout"][n]["delta_ce_doc"])]
        write_json(R.cell_path(out, "readout", t, d), R.readout_record(r, t, d, sets, ce0, edits, t0)
                   | {"vpd_vs_main_readout_max_abs": max(diff, default=None)})

    for t, d in cells:
        r.item(f"readout_{t}_draw{d}", R.cell_path(out, "readout", t, d), lambda t=t, d=d: read(t, d),
               needs=[R.cell_path(out, "calibration", t, d)])

    def analysis() -> None:  # always rerun
        export(out, out / "records", r.reduced, recipe=r.recipe)
        rec = analyze(load(out / "records", allow_reduced=r.reduced), load_dev(main / "records", r.reduced))
        write_json(out / "analysis" / "extended_comparisons_analysis.json", rec)
    r.item("extended_comparisons_analysis", None, analysis,
           needs=[R.cell_path(out, "readout", t, d) for t, d in expected_cells(json.loads((out / FREEZE).read_text()))])


def export(out: Path, root: Path, reduced: bool, *, recipe: dict | None = None) -> None:
    """The records of a frozen run folder for the additional comparisons in the format of ``data/selective_editing/extended_comparisons``."""
    frz = json.loads((out / FREEZE).read_text())
    cells = expected_cells(frz)
    trains, choices = {}, {}
    for t, d in cells:
        rd = json.loads(R.cell_path(out, "readout", t, d).read_text())
        pkey, tkey = rd["protected_key"], rd["target_key"]
        edits, measured, arrays = {}, [], {"delta": [], "target": [], "kl": []}
        for name, e in rd["edits"].items():
            if e["status"] != "matched":
                edits[name] = e
                continue
            rows = e["readout"]
            edits[name] = {k: e[k] for k in ("status", "scale", "guards", "usable", "target_test_delta")} | {
                "sets": {s: {k: v[k] for k in ("ce_base", "ce", "delta_ce", "kl")} for s, v in rows.items()}}
            measured.append(name)
            for key, s, f in (("delta", pkey, "delta_ce_doc"), ("target", tkey, "delta_ce_doc"),
                              ("kl", pkey, "kl_doc")):
                arrays[key].append(np.asarray(rows[s][f], dtype=np.float64))
        n_p, n_t = rd["sets"][pkey], rd["sets"][tkey]
        stack = lambda xs, n: np.stack(xs) if xs else np.zeros((0, n))  # noqa: E731
        write_cell(root / "cells" / f"{t}_draw{d}",
                   {"format": 1, "study": "selective_editing_extended_comparisons", "split": "dev", "target": t,
                    "protected": rd["protected"], "draw": d, "target_key": tkey, "protected_key": pkey, "sets": rd["sets"],
                    "groups": rd["groups"][pkey], "measured": measured, "edits": edits,
                    "vpd_vs_main_readout_max_abs": rd["vpd_vs_main_readout_max_abs"],
                    "source": {"readout_sha256": sha256_file(R.cell_path(out, "readout", t, d))}},
                   {"delta_ce_doc": stack(arrays["delta"], n_p)},
                   {"target_delta_ce_doc": stack(arrays["target"], n_t), "protected_kl_doc": stack(arrays["kl"], n_p)})
        p = R.cell_path(out, "calibration", t, d)
        write_json(root / "calibration" / f"{t}_draw{d}.json",
                   json.loads(p.read_text()) | {"source": {"calibration_sha256": sha256_file(p)}})
        sel, atoms = {}, {}
        for method in (MAIN, *SELECTED):
            rec = json.loads(R.select_path(out, t, d, method).read_text())
            sel[method] = rec
            if method != MAIN and rec["accepted"] is not None:
                a = saved_atom(R.select_path(out, t, d, method).with_suffix(".pt"), rec["atom_sha256"])
                atoms[f"{method}_u"], atoms[f"{method}_v"] = a["u"].numpy(), a["v"].numpy()
        write_json(root / "selection" / f"{t}_draw{d}.json", {"format": 1, "target": t, "draw": d, "selections": sel})
        write_npz(root / "selection" / f"{t}_draw{d}_atoms.npz", atoms)
        for condition, method in CONDITIONS.items():
            ch = choice_path(out, t, condition)
            if ch.exists():
                choices[f"{t}_{method}"] = json.loads(ch.read_text())
    for p in sorted((out / "finetune").glob("*/train.json")):
        tr = json.loads(p.read_text())
        trains[p.parent.name] = tr | {"convergence": convergence(tr, recipe=recipe)}
    chosen = {run_name(t, d, c, ch["chosen_lr"], k) for t, d in cells for c, m in CONDITIONS.items()
              if (ch := choices.get(f"{t}_{m}")) and ch["chosen_lr"] is not None for k in (SEEDS if recipe is None else recipe["extended"]["seeds"])}
    runs = [{"run": n, "target": tr["target"], "draw": tr["draw"], "method": CONDITIONS[tr["condition"]],
             "learning_rate": tr["lr"], "seed_index": tr["seed_index"], "chosen": n in chosen,
             **{k: tr[k] for k in ("modules", "config", "val_by_step", "convergence", "checks", "timings",
                                   "update_sha256", "update_norm")}} for n, tr in trains.items()]
    write_json(root / "training" / "records.json", {"format": 1, "runs": runs})
    write_json(root / "training" / "lr_choice.json", choices)
    write_npz(root / "training" / "logs.npz", {f: np.asarray([[x[f] for x in trains[r["run"]]["log"]] for r in runs],
                                                             dtype=np.float64) for f in LOG_FIELDS})
    write_json(root / "protocol.json", protocol(cells, sha256_file(out / FREEZE), reduced, recipe=recipe))
    write_sums(root)


def protocol(cells: list[tuple[str, int]], freeze_sha256: str, reduced: bool, *, recipe: dict | None = None) -> dict:
    a, b, tol = PLATEAU
    rec = {
        "format": 1, "study": "selective_editing_extended_comparisons", "split": "dev", "reduced": reduced,
        "expected_cells": [f"{t}:{d}" for t, d in cells], "targets": {t: PAIRS[t] for t, _ in cells},
        "taus": list(TAUS), "methods": LABELS, "main_method": MAIN, "comparators": [*SELECTED, *CONDITIONS.values()],
        "learned_methods": list(CONDITIONS.values()), "seeds": list(SEEDS), "guard_max": GUARD_MAX, "band": BAND,
        "bootstrap": {"resamples": 2000, "seed": 0, "level": 0.95, "statistic": "max |t| over contrasts"},
        "calibration": {"tol": TOL, "max_bisect": MAX_BISECT, "atom_grid": list(ATOM_GRID),
                        "whitened_grid": list(WHITENED_GRID), "scale_grid": list(SCALE_GRID)},
        "selection": {"eps_tolerance": EPS_TOLERANCE, "fisher_k": FISHER_K},
        "training": {"steps": STEPS, "margin": MARGIN, "validation_every": EVERY,
                     "learning_rate_grid": {CONDITIONS[c]: list(v) for c, v in LEARNING_RATES.items()},
                     "seed_rule": "100 + draw + 1000 x seed index"},
        "convergence": {"plateau": f"English val KL changes by less than {tol:.0%} from step {a} to {b}",
                        "unstable": "last English val KL > 2 x its minimum from step 100 on"},
        "source": {"freeze_sha256": freeze_sha256},
    }

    if recipe is not None:
        ext, cal, train = recipe["extended"], recipe["calibration"], recipe["training"]
        a, b = ext["plateau_steps"]
        tol = ext["plateau_relative_change"]
        labels = labels_for(recipe)
        rec.update(format=2, recipe=copy.deepcopy(recipe), targets={t: recipe["targets"][t] for t, _ in cells},
                   taus=list(recipe["taus"]), methods=labels, seeds=list(ext["seeds"]),
                   guard_max=recipe["readout"]["guard_max"], band=ext["band"],
                   bootstrap=dict(recipe["bootstrap"]) | {"statistic": "max |t| over contrasts"})
        rec["calibration"] = {k: cal[k] for k in ("tol", "max_bisect", "mono_tol")} | {
            "atom_grid": recipe["selection"]["atom_grid"], "whitened_grid": ext["whitened_grid"],
            "scale_grid": train["scale_grid"]}
        rec["selection"] = {"eps_tolerance": ext["eps_tolerance"], "fisher_k": recipe["selection"]["fisher_k"]}
        rec["training"] = {"steps": ext["steps"], "margin": ext["margin"], "validation_every": ext["validation_every"],
            "learning_rate_grid": {CONDITIONS[c]: list(v) for c, v in ext["learning_rates"].items()},
            "seed_rule": f"{train['seed_offset']} + draw + {ext['seed_stride']} x seed index"}
        rec["convergence"] = {"plateau": f"English val KL changes by less than {tol:.0%} from step {a} to {b}",
            "unstable": f"last English val KL > {ext['unstable_factor']:g} x its minimum from step {ext['unstable_from_step']} on"}
    return rec


# --------------------------------------------------------------------------- command line
def bind_main(out: Path, main: Path) -> None:
    """The additional comparisons start from one frozen main study. The first stage records the content hash of the main FREEZE
    in ``select/main_study.json`` (frozen with the additional comparisons). Every stage refuses another main folder, and one whose
    windows differ in a role both runs read."""
    verify_freeze(main)
    rec, p = {"freeze_sha256": sha256_file(main / FREEZE)}, out / "select" / "main_study.json"
    if not p.exists():
        write_json(p, rec)
    elif json.loads(p.read_text()) != rec:
        raise AccessError(f"{main} is not the main study run that {out} started from")
    records = [json.loads((f / R.RUN_FILE).read_text()) for f in (out, main)]
    if all("recipe" in rec for rec in records):
        shared = ("targets", "taus", "selection", "calibration", "readout")
        changed = [k for k in shared if records[0]["recipe"][k] != records[1]["recipe"][k]]
        if changed:
            raise AccessError(f"additional comparisons and main study differ in shared recipe fields: {changed}")
    ours, theirs = (rec["windows"] for rec in records)
    changed = sorted(k for k in ours.keys() & theirs.keys() if ours[k] != theirs[k])
    if changed:
        raise AccessError(f"{out} and {main} read other windows of {changed}")


class Runner(R.Runner):
    """``run.Runner`` with the main study's frozen run folder (``bind_main``) and the SVD dictionary of the model."""

    def __init__(self, ctx: R.Context, main: Path, stage: str, reduced: bool | None = None,
                 standin: bool = False) -> None:
        super().__init__(ctx, stage, reduced, standin)
        self.main, self.state_cache, self._svd = main, {}, None
        if json.loads((main / R.RUN_FILE).read_text())["reduced"] != self.reduced:
            raise AccessError(f"{main} and {ctx.out} differ in being reduced runs")
        bind_main(ctx.out, main)

    def svd(self):
        if self._svd is None:
            self._svd = svd_dictionary({m: self.ctx.model.get_submodule(m).weight for m in MODULES})
        return self._svd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one model stage of the exploratory additional comparisons.")
    ap.add_argument("stage", choices=("prefreeze", "freeze", "readout"))
    ap.add_argument("assets_dir", type=Path, nargs="?", help="assets directory (not for freeze)")
    ap.add_argument("--main", type=Path, required=True, help="the main study's frozen run folder")
    ap.add_argument("--out", type=Path, required=True, help="run folder for the additional comparisons")
    ap.add_argument("--windows", type=Path, default=Path("data/selective_editing/windows"))
    ap.add_argument("--gate", type=Path, help="curvature-check record (prefreeze, freeze)")
    ap.add_argument("--cells", help="t:d[,t:d...]. Default: every cell of the run")
    ap.add_argument("--reduced", action="store_true", help="prefreeze: code-path check sizes (a new run folder)")
    ap.add_argument("--standin", action="store_true", help="readout: stand-in sets of a reduced run")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--bs", type=int)
    ap.add_argument("--allow-incomplete", action="store_true", help="freeze: record technical gaps instead")
    ap.add_argument("--note", default="", help="freeze: note in FREEZE.json")
    ap.add_argument("--recipe", type=Path, help="complete scientific recipe (default: packaged published profile)")
    a = ap.parse_args(argv)
    a.invocation = R.start_invocation(a.out, a.stage, argv)
    R.prepare(a)
    if a.stage == "freeze":
        _, reduced = R.bind(a.out, None)
        bind_main(a.out, a.main)
        rec = freeze(a.out, a.gate, R.parse_cells(a.cells, a.recipe["reduced"]["draws"] if reduced else a.recipe["draws"], a.recipe),
                     a.allow_incomplete, a.note)
        R.finish_invocation(a.invocation, status="ok", outputs={FREEZE: {"reused": False}})
        print(f"froze {len(rec['artifacts'])} records of {a.out}. Technical gaps: {rec['technical_gaps']}")
        return 0
    if a.stage == "prefreeze" and not a.gate:
        ap.error("prefreeze needs --gate")
    a.no_ci = True  # run.context: the additional comparisons have no reference-environment step
    r = Runner(R.context(a), a.main, a.stage, a.reduced if a.stage == "prefreeze" else None, a.standin)
    if a.stage == "prefreeze":
        prefreeze(r, R.parse_cells(a.cells, r.recipe["reduced"]["draws"] if r.reduced else r.recipe["draws"], r.recipe))
    else:
        readout(r, R.parse_cells(a.cells, (), a.recipe) if a.cells else expected_cells(verify_freeze(a.out)))
    return r.finish(argv)


if __name__ == "__main__":
    raise SystemExit(main())
