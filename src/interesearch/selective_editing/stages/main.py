"""Selective-editing model stages: pre-freeze, freeze, dev readout and confirm readout.
Rules: papers/selective-editing/index.qmd#sec-access, papers/selective-editing/index.qmd#sec-calibration,
papers/selective-editing/index.qmd#sec-analysis.

A cell is one target and draw. A stage runs a fixed list of items in order. An item is skipped when its output exists
(outputs are written last, to a temporary sibling renamed into place, so an interrupted write leaves no output). The run
folder is bound (``RUN.json``) to its scientific configuration, runtime settings, and window identities.
Every stage invocation records its sources, including resume; changed source bytes alone do not block a stage.
An item is not started (``blocked``) when an item it depends on failed or was blocked
in this process. A non-finite value that decides something (``FloatingPointError``) or a failed reference-env subprocess
fails its item. Independent cells go on and the stage exits with status 3. Every item writes a row to ``status.jsonl``.
Every invocation writes its run record to ``runs/<stage>-<utc>.json``: item statuses, the windows it opened and the
code.

Pre-freeze, in order:
    Compute q of the VPD dictionary, then select per cell.
    Per draw, q of that draw's randomized dictionary (seed 101, 102, 103), then its selection per cell.
    Per condition (full matrix, then LoRA) and target: draw-0 training over the learning-rate grid, the choice, then
    the other draws with the chosen rate (no chosen rate: a declared outcome, nothing is trained). After the matrix
    condition, OMP top-1 of its update per cell.
    The historical LoRA condition and its OMP top-1 per cell.
    ``ci_difference`` per cell in the reference env (not run with ``--no-ci``: a recorded technical gap), the neuron
    ranking per cell.
    The calibration of a cell, only when no item it depends on failed or was blocked (a calibration made without
    ci_difference is made again once a rerun runs it).
``--cells`` limits the cells. The draw-0 rate choice of their targets runs with them. A reduced run (``--reduced``:
draws 0 and 1, 8 rows per draw and English pool, 4 per validation set, a shortlist of 4, 3 training steps and the
first rate only) checks the code path only. Its readouts read stand-in sets from selection roles (``--standin``),
and its records load only with ``allow_reduced``.

    uv run python -m interesearch.selective_editing.stages.main prefreeze assets --out runs/selective_editing \\
        --gate runs/selective_editing/fisher_check.json --ci-python vendor/param-decomp/.venv/bin/python
    uv run python -m interesearch.selective_editing.stages.main freeze --out runs/selective_editing \\
        --gate runs/selective_editing/fisher_check.json --note "..."
    uv run python -m interesearch.selective_editing.stages.main readout assets --out runs/selective_editing
    uv run python -m interesearch.selective_editing.stages.main confirm assets --out runs/selective_editing \\
        --confirm-windows CONFIRM_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from interesearch.assets import MODULES
from interesearch.checksums import sha256_file
from interesearch.selective_editing.analysis import confirm
from interesearch.selective_editing.method import fisher
from interesearch.selective_editing.stages.support import (
    prepare,
    start_invocation,
    finish_invocation,
    Runner,
    TARGETS,
    bind,
    cell_path,
    choice_path,
    context,
    null_name,
    parse_cells,
    readout_record,
    ref_path,
    rel,
    run_dir,
    select_path,
    sets_for,
)
from interesearch.selective_editing.stages.access import (
    CONFIRM_ROLE,
    FREEZE,
    SELECTION_ROLES,
    load_confirm,
    open_confirm,
    utc,
    verify_freeze,
    window_identity,
    write_freeze,
)
from interesearch.selective_editing.analysis.statistics import analyze
from interesearch.selective_editing.method.calibration import replay_trace
from interesearch.selective_editing.data import expected_cells, load_dev, write_cell, write_json, write_sums
from interesearch.selective_editing.method.dictionary import Dictionary, parse_label_key
from interesearch.selective_editing.method.dose import (
    HISTORICAL_ATOM,
    LEARNED,
    PAIRS,
    Family,
    atom_family,
    calibrate,
    declared,
    learned_family,
    missing,
    neuron_family,
    protocol,
    public_calibration,
)
from interesearch.selective_editing.method.finetune import (
    OMP_LORA,
    OMP_LORA_LR,
    LEARNING_RATES,
    choose_lr,
    config,
    load_update,
    load_update_record,
    train_run,
)
from interesearch.selective_editing.method.neurons import neuron_ranking
from interesearch.selective_editing.method.runtime import require_gate
from interesearch.selective_editing.stages.readout import (
    Sets,
    atlas,
    confirm_plan,
    confirm_sets,
    dev_sets,
    public_cell,
    read_confirm,
    read_dev,
    standin_confirm_sets,
    standin_sets,
)
from interesearch.selective_editing.method.references import omp_top1
from interesearch.selective_editing.method.selector import FISHER_K, q_record, select


# --------------------------------------------------------------------------- pre-freeze
def prefreeze(r: Runner, cells: list[tuple[str, int]]) -> None:
    ctx, out, S = r.ctx, r.out, r.sizes
    omp_lr = OMP_LORA_LR if r.recipe is None else r.recipe["omp_lora"]["lr"]
    r.access.blind("prefreeze")
    retain, qcheck = r.windows("retain_en", "en"), r.windows("qcheck_en", "en")
    if ctx.gate.resolve() == (out / "fisher_check.json").resolve():
        r.item("fisher_check", ctx.gate, lambda: write_json(ctx.gate, fisher.run(ctx.assets, retain, ctx.device, dictionary=ctx.vpd, recipe=r.recipe)))
    require_gate(ctx.gate, FISHER_K if r.recipe is None else r.recipe["selection"]["fisher_k"],
                 eps_percentile=1.0 if r.recipe is None else r.recipe["selection"]["eps_percentile"],
                 null_seed=101 if r.recipe is None else r.recipe["selection"]["null_seeds"][0])  # a hard gate: no selection without a passed curvature check of the current code
    targets = sorted({t for t, _ in cells}, key=TARGETS.index)
    draws = sorted({d for _, d in cells})
    vpd_cells = sorted(set(cells) | {(t, 0) for t in targets}, key=lambda c: (TARGETS.index(c[0]), c[1]))
    q_recs = {}

    def q_item(dic: Dictionary) -> None:
        p = out / "q" / f"{dic.name}.pt"

        def run() -> None:
            q_recs[dic.name] = q_record(p, ctx.model, dic, retain[:S.q_rows], qcheck[:S.q_rows], ctx.device,
                                        ctx.gate, ctx.bs, recipe=r.recipe)
        r.item(f"q_{dic.name}", p.with_suffix(".json"), run)

    def select_item(t: str, d: int, dic: Dictionary) -> None:
        p = select_path(out, t, d, dic.name)

        def run() -> None:
            if dic.name not in q_recs:  # q done by an earlier invocation: reloaded under equal provenance only
                q_recs[dic.name] = q_record(out / "q" / f"{dic.name}.pt", ctx.model, dic, retain[:S.q_rows],
                                            qcheck[:S.q_rows], ctx.device, ctx.gate, ctx.bs, recipe=r.recipe)
            rec, tensors = select(ctx.model, dic, q_recs[dic.name], r.draw(t, d), ctx.device, ctx.gate, ctx.bs,
                                  out / "cache" / f"G_{t}_draw{d}.pt", S.n_short, recipe=r.recipe)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensors, p.with_suffix(".pt"))
            write_json(p, {"target": t, "draw": d, **rec})
        r.item(f"select_{t}_draw{d}_{dic.name}", p, run, [f"q_{dic.name}"])

    q_item(ctx.vpd)
    for t, d in vpd_cells:
        select_item(t, d, ctx.vpd)
    for d in draws:
        dic = r.dictionary(null_name(d, recipe=r.recipe))
        q_item(dic)
        for t in targets:
            if (t, d) in cells:
                select_item(t, d, dic)
    for condition in ("matrix", "lora"):
        for t in targets:
            learned(r, t, condition, [d for tt, d in cells if tt == t and d > 0])
        if condition == "matrix":
            for t, d in cells:
                deps = [f"lr_choice_{t}_matrix"] + ([f"finetune_{t}_draw{d}_matrix"] if d else [])
                r.item(f"omp_finetune_{t}_draw{d}", ref_path(out, "omp_finetune", t, d),
                       lambda t=t, d=d: omp_finetune(r, t, d), deps)
    for t, d in cells:
        hist = f"finetune_{t}_draw{d}_{OMP_LORA}"
        r.item(hist, run_dir(out, t, d, OMP_LORA, omp_lr) / "train.json",
               lambda t=t, d=d: train(r, t, d, OMP_LORA, omp_lr))
        r.item(f"omp_lora_{t}_draw{d}", ref_path(out, "omp_lora", t, d), lambda t=t, d=d: omp_lora(r, t, d), [hist])
    for t, d in cells:
        if ctx.ci is not None:
            r.item(f"ci_difference_{t}_draw{d}", ref_path(out, "ci_difference", t, d), lambda t=t, d=d: ci(r, t, d))
        r.item(f"neurons_{t}_draw{d}", cell_path(out, "neurons", t, d), lambda t=t, d=d: neurons(r, t, d))
    for t, d in cells:
        deps = ["q_vpd", f"select_{t}_draw{d}_vpd", f"q_{null_name(d, recipe=r.recipe)}", f"select_{t}_draw{d}_{null_name(d, recipe=r.recipe)}",
                f"omp_finetune_{t}_draw{d}", f"finetune_{t}_draw{d}_{OMP_LORA}", f"omp_lora_{t}_draw{d}",
                f"neurons_{t}_draw{d}"]
        deps += [f"lr_choice_{t}_{c}" for c in LEARNED.values()]
        deps += [f"finetune_{t}_draw{d}_{c}" for c in LEARNED.values() if d]
        deps += [f"ci_difference_{t}_draw{d}"] if ctx.ci is not None else []
        p = cell_path(out, "calibration", t, d)
        r.item(f"calibration_{t}_draw{d}", None if ctx.ci is not None and lacks_ci(p) else p,
               lambda t=t, d=d: calibration(r, t, d), deps)


def lacks_ci(p: Path) -> bool:
    """True when the calibration at ``p`` exists but was made without the CI reference (an earlier run with
    ``--no-ci``): it is not complete once the reference runs, and is recomputed."""
    return p.exists() and json.loads(p.read_text())["methods"]["ci_difference"]["status"].startswith("missing input")


def learned(r: Runner, t: str, condition: str, later_draws: list[int]) -> None:
    """Draw 0 over the rate grid, the choice, then the other draws with the chosen rate."""
    out = r.out
    grid = (LEARNING_RATES if r.recipe is None else r.recipe["training"]["learning_rates"])[condition][:r.sizes.n_rates]
    sel0 = [f"select_{t}_draw0_vpd"] if condition == "lora" else []
    names = []
    for lr in grid:
        names.append(f"finetune_{t}_draw0_{condition}_lr{lr:g}")
        r.item(names[-1], run_dir(out, t, 0, condition, lr) / "train.json",
               lambda lr=lr: train(r, t, 0, condition, lr), sel0)
    choice = f"lr_choice_{t}_{condition}"

    def choose() -> None:
        val_t, val_en = r.windows("val", t)[:r.sizes.val_rows], r.windows("val_en", "en")[:r.sizes.val_rows]
        updates = {lr: load_update(run_dir(out, t, 0, condition, lr)) for lr in grid}
        rec = choose_lr(r.ctx.model, updates, val_t, val_en, r.ctx.device, r.ctx.bs, recipe=r.recipe)
        write_json(choice_path(out, t, condition), {"target": t, "condition": condition, "lr_grid": list(grid), **rec})
    r.item(choice, choice_path(out, t, condition), choose, names)
    for d in later_draws:
        name = f"finetune_{t}_draw{d}_{condition}"
        if r.blocked(name, [choice] + ([f"select_{t}_draw{d}_vpd"] if condition == "lora" else [])):
            continue
        lr = json.loads(choice_path(out, t, condition).read_text())["chosen_lr"]
        if lr is None:
            r.declare(name, "no learning rate reached the upper effect point on draw 0")
            continue
        r.item(name, run_dir(out, t, d, condition, lr) / "train.json", lambda d=d, lr=lr: train(r, t, d, condition, lr))


def train(r: Runner, t: str, d: int, condition: str, lr: float) -> None:
    S, ctx = r.sizes, r.ctx
    if condition == "lora":  # the module of the selected atom (else of the first walked atom) of the same cell
        rec = json.loads(select_path(r.out, t, d, "vpd").read_text())
        modules = [(rec["accepted"] or rec["walk"][0])["module"]]
    else:
        modules = list(MODULES)
    cfg = config(condition, d, None if condition == OMP_LORA else lr, r.reduced, recipe=r.recipe)
    train_run(run_dir(r.out, t, d, condition, lr), ctx.model, modules, r.draw(t, d),
              r.windows("retain_en", "en")[:S.retain_rows], r.windows("val", t)[:S.val_rows],
              r.windows("val_en", "en")[:S.val_rows], condition, cfg, ctx.device, r.reduced,
              header={"target": t, "draw": d}, recipe=r.recipe)


def _omp(r: Runner, t: str, d: int, rule: str, run: Path | None, status: str | None = None) -> None:
    method = "omp_finetune" if rule.startswith("full") else "omp_lora"
    rec = {"target": t, "draw": d, "rule": f"OMP top-1 (normalized atoms) of the {rule} update"}
    if status is not None:  # a declared outcome of the update's rule: written, no atom
        rec |= {"module": None, "status": status, "run": None if run is None else rel(r.out, run)}
    else:
        update, digest = load_update_record(run)
        rec |= {"run": rel(r.out, run), "update_sha256": digest, **omp_top1(r.ctx.vpd, update)}
    write_json(ref_path(r.out, method, t, d), rec)


def omp_finetune(r: Runner, t: str, d: int) -> None:
    lr = json.loads(choice_path(r.out, t, "matrix").read_text())["chosen_lr"]
    if lr is None:
        return _omp(r, t, d, "full-matrix", None, "no update: no learning rate reached the upper effect point on draw 0")
    _omp(r, t, d, "full-matrix", run_dir(r.out, t, d, "matrix", lr))


def omp_lora(r: Runner, t: str, d: int) -> None:
    run = run_dir(r.out, t, d, OMP_LORA, OMP_LORA_LR if r.recipe is None else r.recipe["omp_lora"]["lr"])
    if json.loads((run / "train.json").read_text())["update_sha256"] is None:
        return _omp(r, t, d, "historical LoRA", run, "no update: no checkpoint met the historical checkpoint rule")
    _omp(r, t, d, "historical LoRA", run)


def ci(r: Runner, t: str, d: int) -> None:
    cache = r.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    target, english = cache / f"ci_{t}_draw{d}.npy", cache / "ci_english.npy"
    np.save(target, r.draw(t, d)[:r.sizes.ci_rows].numpy().astype(np.int32))
    np.save(english, r.windows("retain_en", "en")[:r.sizes.ci_rows].numpy().astype(np.int32))
    out = ref_path(r.out, "ci_difference", t, d)
    tmp = out.with_name(f".{out.name}.tmp")  # the reference env writes here. Renamed only after it succeeded
    r.ctx.ci(target, english, cache / "ci_en.pt", tmp)
    os.replace(tmp, out)


def neurons(r: Runner, t: str, d: int) -> None:
    rec = neuron_ranking(r.ctx.model, r.draw(t, d), r.windows("retain_en", "en")[:r.sizes.neuron_rows], r.ctx.device,
                         r.out / "cache" / "rates_en.pt", r.ctx.bs, recipe=r.recipe)
    write_json(cell_path(r.out, "neurons", t, d), {"target": t, "draw": d, **rec})


def families(r: Runner, t: str, d: int) -> dict[str, Family | tuple[str, bool]]:
    """The edit family of every method of a cell, or why it has none (technical: an input is absent)."""
    out, dev, vpd = r.out, r.ctx.device, r.ctx.vpd
    fam: dict[str, Family | tuple[str, bool]] = {}
    for m, name in (("vpd_atom", "vpd"), ("null_atom", null_name(d, recipe=r.recipe))):
        p = select_path(out, t, d, name)
        if not p.exists():
            fam[m] = missing(rel(out, p))
            continue
        rec = json.loads(p.read_text())
        if rec["accepted"] is None:
            fam[m] = ((f"selector failed: {rec['status']}", True) if rec.get("technical_failure")
                      else declared(f"selector status {rec['status']!r}"))
            continue
        a = rec["accepted"]
        fam[m] = atom_family(r.dictionary(name), a["module"], a["index"], a["sign"], dev,
                             {"source": rel(out, p), "status": rec["status"], "n_candidates": rec["n_candidates"],
                              "n_forward_passes": rec["n_forward_passes"]}, recipe=r.recipe)
    for m in ("omp_finetune", "omp_lora", "ci_difference"):
        p = ref_path(out, m, t, d)
        if not p.exists():
            fam[m] = missing(rel(out, p))
            continue
        rec = json.loads(p.read_text())
        fam[m] = (declared(f"upstream status {rec['status']!r}") if rec["module"] is None else
                  atom_family(vpd, rec["module"], rec["index"], rec["sign"], dev, {"source": rel(out, p)}, recipe=r.recipe))
    if t in HISTORICAL_ATOM:
        label, sign = HISTORICAL_ATOM[t]
        fam["historical_atom"] = atom_family(vpd, *parse_label_key(label), sign, dev,
                                             {"source": "historical, fixed before this study; dev-selected"}, recipe=r.recipe)
    else:
        fam["historical_atom"] = ("not applicable (de only)", False)
    for m, condition in LEARNED.items():
        p = choice_path(out, t, condition)
        if not p.exists():
            fam[m] = missing(rel(out, p))
            continue
        lr = json.loads(p.read_text())["chosen_lr"]
        run = None if lr is None else run_dir(out, t, d, condition, lr)
        if lr is None:
            fam[m] = declared("no learning rate reached the upper effect point on draw 0")
        elif not (run / "train.json").exists():
            fam[m] = missing(f"{rel(out, run)}/train.json")
        else:
            fam[m] = learned_family(load_update(run), rel(out, run), json.loads((run / "train.json").read_text()),
                                    (LEARNING_RATES if r.recipe is None else r.recipe["training"]["learning_rates"])[condition][:r.sizes.n_rates], dev, recipe=r.recipe)
    p = cell_path(out, "neurons", t, d)
    fam["neurons"] = (missing(rel(out, p)) if not p.exists() else neuron_family(
        r.ctx.model, [(n["layer"], n["neuron"]) for n in json.loads(p.read_text())["ranked"]], rel(out, p), dev, recipe=r.recipe))
    return fam


def calibration(r: Runner, t: str, d: int) -> None:
    r.access.blind("calibration")
    n = r.sizes.val_rows
    rec = calibrate(r.ctx.model, r.base, r.windows("val", t)[:n], r.windows("val_en", "en")[:n], families(r, t, d),
                    r.ctx.device, r.ctx.bs, recipe=r.recipe)
    write_json(cell_path(r.out, "calibration", t, d),
               {"target": t, "draw": d, "val_set": f"europarl_val_{t}", "reduced": r.reduced, **rec})


# --------------------------------------------------------------------------- readouts


def export_records(r: Runner, root: Path, splits: Sequence[str]) -> None:
    """The exported records (data.load_dev / data.load_study format) of every expected cell that has them."""
    frz = json.loads((r.out / FREEZE).read_text())
    cells = expected_cells(frz)
    write_json(root / "protocol.json", protocol(cells, frz, sha256_file(r.out / FREEZE), r.reduced, recipe=r.recipe))
    for t, d in cells:
        p = cell_path(r.out, "calibration", t, d)
        if p.exists():
            write_json(root / "calibration" / f"{t}_draw{d}.json",
                       public_calibration(json.loads(p.read_text()), t, d, sha256_file(p)))
        for split in splits:
            p = cell_path(r.out, "readout" if split == "dev" else "confirm/readout", t, d)
            if p.exists():
                write_cell(root / split / f"{t}_draw{d}", *public_cell(json.loads(p.read_text()), split, sha256_file(p)))
    write_sums(root)


def readout(r: Runner, cells: list[tuple[str, int]]) -> None:
    out, ctx = r.out, r.ctx
    verify_freeze(out)

    def read(t: str, d: int) -> None:
        t0 = time.time()
        sets, ce0 = sets_for(r, ("dev", t), lambda: (standin_sets if r.access.standin else dev_sets)(r.access, t, recipe=r.recipe))
        cal = json.loads(cell_path(out, "calibration", t, d).read_text())
        edits = read_dev(ctx.model, r.base, cal, sets, ce0, r.make_of, ctx.device, ctx.bs, recipe=r.recipe)
        write_json(cell_path(out, "readout", t, d), readout_record(r, t, d, sets, ce0, edits, t0))

    def post_hoc(t: str, d: int) -> None:
        sets, _ = sets_for(r, ("dev", t), lambda: (standin_sets if r.access.standin else dev_sets)(r.access, t, recipe=r.recipe))
        sel = json.loads(select_path(out, t, d, "vpd").read_text())
        rec = atlas(ctx.model, r.base, ctx.vpd, sel, sets.tokens[f"europarl_val_{t}"], sets.tokens[sets.protected_key],
                    ctx.device, ctx.bs, r.sizes.atlas_top, recipe=r.recipe)
        write_json(cell_path(out, "atlas", t, d), {"target": t, "draw": d, "protected_set": sets.protected_key, **rec})

    for t, d in cells:
        r.item(f"readout_{t}_draw{d}", cell_path(out, "readout", t, d), lambda t=t, d=d: read(t, d),
               needs=[cell_path(out, "calibration", t, d)])
        r.item(f"atlas_{t}_draw{d}", cell_path(out, "atlas", t, d), lambda t=t, d=d: post_hoc(t, d),
               needs=[select_path(out, t, d, "vpd")])

    def analysis() -> None:  # always rerun: every expected cell, read or "missing readout"
        export_records(r, out / "records", ("dev",))
        study = load_dev(out / "records", allow_reduced=r.reduced)
        write_json(out / "analysis" / "dev_analysis.json", analyze(study, "dev"))
        write_json(out / "analysis" / "calibration_trace.json", replay_trace(study))
    r.item("dev_analysis", None, analysis)


def confirm_readout(r: Runner, cells: list[tuple[str, int]], windows: Path | None, resume: bool = False) -> None:
    """The one confirm readout: the opening record first, then the confirm windows, then the frozen dev edits."""
    out, ctx = r.out, r.ctx
    verify_freeze(out)
    root = out / "confirm"
    if r.access.standin:
        source = {"path": "stand-in: selection roles", "identity": {k: r.inputs[k] for k in SELECTION_ROLES}}
    else:  # the identity from the manifest, recorded before the first confirm window is read
        source = {"path": str(windows.resolve()), "identity": window_identity(windows, CONFIRM_ROLE)}
    open_confirm(root, {"utc": utc(), "code": r.code, "freeze_sha256": sha256_file(out / FREEZE),
                        "confirm_windows": source["identity"], "confirm_path": source["path"]}, resume)
    if r.access.standin:
        def sets_of(t: str) -> Sets:
            return standin_confirm_sets(r.access, t, recipe=r.recipe)
    else:
        pairs = PAIRS if r.recipe is None else r.recipe["targets"]
        langs = sorted({x for t, _ in cells for x in (t, pairs[t])} | {"en"})
        w = load_confirm(windows, root, langs)
        r.access.opened["confirm"] = {"rows": len(w.groups), "stems": list(w.tokens), "sha256": w.sha256}

        def sets_of(t: str) -> Sets:
            return confirm_sets(w, t, recipe=r.recipe)

    def read(t: str, d: int) -> None:
        t0 = time.time()
        sets, ce0 = sets_for(r, ("confirm", t), lambda: sets_of(t))
        cal = json.loads(cell_path(out, "calibration", t, d).read_text())
        plan = confirm_plan(json.loads(cell_path(out, "readout", t, d).read_text())["edits"], cal)
        edits = read_confirm(ctx.model, r.base, plan, cal, sets, ce0, r.make_of, ctx.device, ctx.bs, recipe=r.recipe)
        write_json(cell_path(out, "confirm/readout", t, d),
                   readout_record(r, t, d, sets, ce0, edits, t0) | {"confirm_windows": source})

    for t, d in cells:
        r.item(f"confirm_{t}_draw{d}", cell_path(out, "confirm/readout", t, d), lambda t=t, d=d: read(t, d),
               needs=[cell_path(out, "calibration", t, d), cell_path(out, "readout", t, d)])
    expected = expected_cells(json.loads((out / FREEZE).read_text()))

    def analysis() -> None:
        export_records(r, root / "records", ("dev", "confirm"))
        (root / "analysis").mkdir(exist_ok=True)
        for name, text in confirm.run(root / "records", allow_reduced=r.reduced).items():
            (root / "analysis" / name).write_text(text)
    r.item("confirm_analysis", None, analysis,
           needs=[cell_path(out, k, t, d) for t, d in expected for k in ("calibration", "readout", "confirm/readout")])


# --------------------------------------------------------------------------- command line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Select language-targeted edits, calibrate their effects, and measure changes in related languages.",
        epilog="Usage and outputs: papers/guides/selective-editing.md")
    ap.add_argument("stage", choices=("prefreeze", "freeze", "readout", "confirm"))
    ap.add_argument("assets_dir", type=Path, nargs="?", help="assets directory (not for freeze)")
    ap.add_argument("--out", type=Path, required=True, help="run folder")
    ap.add_argument("--windows", type=Path, default=Path("data/selective_editing/windows"))
    ap.add_argument("--gate", type=Path, help="curvature-check record (prefreeze, freeze)")
    ap.add_argument("--cells", help="t:d[,t:d...]. Default: every cell of the run")
    ap.add_argument("--reduced", action="store_true", help="prefreeze: code-path check sizes (a new run folder)")
    ap.add_argument("--standin", action="store_true", help="readout, confirm: stand-in sets of a reduced run")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--bs", type=int)
    ap.add_argument("--ci-python", help="prefreeze: python of the param-decomp reference env")
    ap.add_argument("--param-decomp", type=Path, default=Path("vendor/param-decomp"))
    ap.add_argument("--no-ci", action="store_true", help="prefreeze: do not run ci_difference (a technical gap)")
    ap.add_argument("--allow-incomplete", action="store_true", help="freeze: record technical gaps instead")
    ap.add_argument("--note", default="", help="freeze: note in FREEZE.json")
    ap.add_argument("--confirm-windows", type=Path, help="confirm: the confirm window directory")
    ap.add_argument("--resume", action="store_true", help="confirm: complete the cells of an opened confirm set")
    ap.add_argument("--recipe", type=Path, help="complete scientific recipe (default: packaged published profile)")
    a = ap.parse_args(argv)
    a.invocation = start_invocation(a.out, a.stage, argv)
    prepare(a)
    if a.stage == "freeze":
        _, reduced = bind(a.out, None)
        rec = write_freeze(a.out, a.gate, parse_cells(a.cells, a.recipe["reduced"]["draws"] if reduced else a.recipe["draws"], a.recipe),
                           a.allow_incomplete, a.note)
        finish_invocation(a.invocation, status="ok", outputs={FREEZE: {"reused": False}})
        print(f"froze {len(rec['artifacts'])} records of {a.out}. Technical gaps: {rec['technical_gaps']}")
        return 0
    if a.stage == "prefreeze" and not (a.gate and (a.no_ci or a.ci_python)):
        ap.error("prefreeze needs --gate and --ci-python (or --no-ci)")
    if a.stage == "confirm" and not (a.standin or a.confirm_windows):
        ap.error("confirm needs --confirm-windows (or --standin for a reduced run)")
    r = Runner(context(a), a.stage, a.reduced if a.stage == "prefreeze" else None, a.standin)
    if a.stage == "prefreeze":
        prefreeze(r, parse_cells(a.cells, r.recipe["reduced"]["draws"] if r.reduced else r.recipe["draws"], r.recipe))
    elif a.stage == "readout":
        readout(r, parse_cells(a.cells, (), a.recipe) if a.cells else expected_cells(verify_freeze(a.out)))
    else:
        confirm_readout(r, parse_cells(a.cells, (), a.recipe) if a.cells else expected_cells(verify_freeze(a.out)),
                        a.confirm_windows, a.resume)
    return r.finish(argv)


if __name__ == "__main__":
    raise SystemExit(main())
