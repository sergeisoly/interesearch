"""Target block: fit and select candidates before opening readout windows."""

from __future__ import annotations
import math
import json
import sys
import time
from pathlib import Path
import torch
from ...assets import MODULES
from ..analysis import stats
from ..data import write_block
from ..method.evaluate import direction, evaluate, parity
from ..method.finetune import code_identity, environment
from ..method.fitting import (
    FitLoss, Method, finite, fit_accounting, fit_sets, fro_of, init_check, make_method,
    nonfinite_snapshot, polish, randomized, train,
)
from ..method.selection import point, select

from .support import (
    LANGS, FIT_LANGS, CAPPED, RC_OK, RC_REFUSED, RC_FAILED_FAMILY, RC_TECHNICAL, RC_PRE_READOUT_STOP,
    POLISH_KEYS, DIRECTION_PAIRS, utc, logger, setup, finish, block_inputs, windows, nonfinite_runs,
    failed_family_reason, nonfinite_log, nonfinite_selection,
)

def train_candidates(a, info, model, base, D, donor, rdonors, cap, budgets, fams, log, flag):
    # ---- training: every budget, family and learning rate
    fit_w = windows(a, info, "fit", FIT_LANGS, "fit", prefixes={B: a.resolved_recipe["fitting"]["budgets"][B] for B in budgets})
    shapes = {m: tuple(base[m].shape) for m in MODULES}
    strengths_method = Method.strengths(donor, cap, a.device)
    E_start = {m: e.detach().double() for m, e in strengths_method.E().items()}
    cands, cand_meta, fsets, planned, saved = {}, {}, {}, [], set()
    for B in budgets:
        fs = fit_sets(fit_w.tokens, B, budgets=a.resolved_recipe["fitting"]["budgets"])
        fsets[B] = fs
        info["budgets"][B] = fit_accounting(fs)
        for fam in fams:
            planned.append(f"{fam}@{B}")
            for lr in a.resolved_recipe["fitting"]["lrs"][fam]:
                meth = make_method(fam, shapes, a.device, cap, donor, rdonors, init_seed=a.resolved_recipe["fitting"]["init_seed"])
                if fam not in info["checks"]["init"]:
                    ic = init_check(fam, meth, E_start, cap, init_tol=a.resolved_recipe["fitting"]["init_tol"])
                    info["checks"]["init"][fam] = ic
                    if not ic["ok"]:
                        flag(fam, f"init check failed: {ic}")
                t = time.time()
                checkpoints, train_log, train_status = train(
                    model, base, D, meth, fs, lr, a.device, log, f"{fam}@{B}", a.steps, a.every,
                    settings=a.resolved_recipe["fitting"])
                train_status["train_s"] = time.time() - t
                train_status["checkpoints_saved"] = sorted(checkpoints)
                train_status["c_init"] = meth.c0
                run = f"{fam}@{B}:lr={lr:g}"
                info["runs"][run] = {"status": train_status, "log": train_log}
                if train_status.get("finite") is False:
                    flag(f"{fam}@{B}", f"{run}: non-finite loss/gradient at step {train_status.get('nonfinite_step')}")
                bad_log = nonfinite_log(train_log)
                if bad_log:
                    flag(f"{fam}@{B}", f"{run}: non-finite logged values {bad_log}")
                mpc = train_status.get("max_post_over_cap")
                if mpc is not None and not (finite(mpc) and mpc <= 1 + a.resolved_recipe["fitting"]["cap_tol"]):
                    flag(f"{fam}@{B}", f"{run}: ||E|| / cap after projection = {mpc}")
                for step, snapshot in checkpoints.items():
                    saved.add(f"{fam}@{B}")
                    if nonfinite_snapshot(snapshot):  # cannot be applied. Flagged and left out of every evaluation
                        flag(f"{fam}@{B}", f"{run}: non-finite parameters at checkpoint {step}")
                        continue
                    candidate_name = f"{run}:step={step}"
                    cands[candidate_name] = meth.spec(snapshot, donor)
                    cand_meta[candidate_name] = {"family": f"{fam}@{B}", "budget": B, "lr": lr, "step": step}
    info["planned_families"] = planned
    info["failed_families"] = {f: failed_family_reason(f, info["runs"], a.every) for f in planned if f not in saved}
    for f, why in info["failed_families"].items():
        log(f"FAILED FAMILY {f}: {why}")
    info["nonfinite_runs"] = nonfinite_runs(info["runs"])

    return cands, cand_meta, fsets, strengths_method

def refine_candidates(a, info, model, base, D, donor, cap, budgets, fams, blk, cands, cand_meta, fsets, strengths_method, fixed, log, flag):
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
            starts = {"c0": torch.full((len(MODULES),), strengths_method.c0)}
            if s_names:
                best = min(s_names, key=lambda n: (fit[n]["total"], s_names.index(n)))
                starts["best"] = cands[best]["c"]
                info["polish_best_start"] = best
            ref = None
            for tag, c_start in starts.items():  # c0 first: its start sets the stationarity scale of both
                rec, snap = polish(fl, strengths_method, c_start, cap, a.polish_iters, log, tag, ref, params=a.resolved_recipe["polish"], fit_path_tol=a.resolved_recipe["fitting"]["fit_path_tol"])
                info["polish"][tag] = rec
                if tag == "c0" and rec.get("finite"):
                    ref = {"stat": rec.get("stationarity_start"), "lam0": rec.get("lam0")}
                if not rec["finite"]:
                    flag(f"strengths_best@{B}", f"polish from {tag} not finite ({rec.get('stop')})")
                if snap is not None:
                    n = f"strengths_converged@{B}:start={tag}"
                    converged[n] = strengths_method.spec(snap, donor)
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

    return fit, converged, exp_s, exp_n, polished

def select_candidates(a, info, model, base, D, Dd, cands, cand_meta, converged, fixed, sel_sets, sidx, budgets, fams, polished, log, flag):
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
    bad_sel = nonfinite_selection(vper, list(cands), sidx, a.resolved_recipe["selection"])
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
            selections[g] = select(vper, removed_fro, names, sidx, a.resolved_recipe["selection"])
    info["selections"] = selections
    info["selection_points"] = {n: point(vper, n, sidx, a.resolved_recipe["selection"]) for n in allc}

    return vper, selections, meta

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
    blk = a.resolved_recipe["block"]["profiles"][a.block]
    fams, budgets = list(blk["families"]), list(blk["budgets"])
    D, ft = a.finetune_update
    info = {"_t0": t0, "command": "block", "target": a.target, "block": a.block, "status": None, "rc": None,
            "reasons": [], "reduced": a.reduced, "deviations": a.deviations, "resolved_recipe": a.resolved_recipe,
            "config": {"steps": a.steps, "every": a.every, "step_caps": list(a.step_caps), "rows": a.rows,
                       "polish_iters": a.polish_iters, "budgets": budgets, "families": fams,
                       "lrs": {f: a.resolved_recipe["fitting"]["lrs"][f] for f in fams}, "eval_batch_size": a.eval_batch_size, "device": a.device,
                       "threads": a.threads},
            "inputs": inputs, "code": code_identity(), "precision": None, "environment": environment(a.device),
            "utc_start": utc(), "budgets": {}, "runs": {}, "technical": {}, "checks": {"init": {}, "parity": {}}}
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str))
    model, base, precision = setup(a)
    info["precision"] = precision
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
    dobj = a.donor_payload
    donor = {m: (b.float(), a_.float()) for m, (b, a_) in dobj["factors"].items()}
    if not all(bool(torch.isfinite(b).all()) and bool(torch.isfinite(a_).all()) for b, a_ in donor.values()):
        flag("target", "non-finite donor factor")
    rdonors = {f: randomized(donor, seed) for f, seed in a.resolved_recipe["fitting"]["random_seed"].items()}
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
        p_D = point(vref, "finetuned", sidx, a.resolved_recipe["selection"])
        info["damage_selection"] = {"d_nl_D": p_D["d_nl_D"], "d_de_D": p_D["d_de_D"],
                                    "floor_ok": bool(finite(p_D["d_nl_D"]) and p_D["d_nl_D"] >= a.resolved_recipe["block"]["damage_floor"]),
                                    "eval_s": time.time() - t}
        nf = stats.nonfinite(vref, ("base", "finetuned"))
        if nf:
            flag("target", f"base/finetuned on the selection windows: {nf[:5]}")
    if not inputs["finetune"]["selection"].get("usable"):
        stop.append("the fine-tune has no usable checkpoint (finetune.py checkpoint rule)")
    if not inputs["finetune"]["gates"].get("passed"):
        stop.append(f"fine-tune gates failed: {inputs['finetune']['gates'].get('failed')}")
    if D_finite and not info["damage_selection"]["floor_ok"]:
        stop.append(f"selection damage floor: d_nl(D) = {info['damage_selection']['d_nl_D']} < {a.resolved_recipe["block"]["damage_floor"]}")
    if tech:  # a technical failure takes precedence over the stop reasons: exit 9, never 10
        info["pre_readout_halt"] = {"rc": RC_TECHNICAL, "technical": dict(tech), "stop_reasons_not_applied": stop}
        return finish(out, info, RC_TECHNICAL, log, [f"technical before training: {sorted(tech)}"])
    if stop:
        info["pre_readout_stop"] = stop
        return finish(out, info, RC_PRE_READOUT_STOP, log, stop)

    cands, cand_meta, fsets, strengths_method = train_candidates(
        a, info, model, base, D, donor, rdonors, cap, budgets, fams, log, flag)

    fit, converged, exp_s, exp_n, polished = refine_candidates(
        a, info, model, base, D, donor, cap, budgets, fams, blk, cands, cand_meta, fsets, strengths_method, fixed, log, flag)

    vper, selections, meta = select_candidates(
        a, info, model, base, D, Dd, cands, cand_meta, converged, fixed, sel_sets, sidx, budgets, fams, polished, log, flag)

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
        if not (finite(v) and v <= a.resolved_recipe["evaluation"]["parity_tol"]):  # keyed by the group (a converged candidate: strengths_best)
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
            "protocol": {"resolved_recipe": a.resolved_recipe, "budgets": budgets, "families": fams, "capped_families": [f for f in CAPPED if f in fams],
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
