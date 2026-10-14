"""Fit the donor repair, select on validation, then evaluate its readout gate."""

from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
import torch
from ...assets import MODULES
from ...checksums import sha256_file
from ..analysis import stats
from ..data import DONOR_FILES
from ..method.evaluate import evaluate, parity
from ..method.finetune import VAL_ROLE, code_identity, environment
from ..method.fitting import finite, fit_accounting, fit_sets, fro_of, make_method, nonfinite_snapshot, train
from ..method.selection import german_feasible, point, select

from .support import (
    LANGS, FIT_LANGS, RC_OK, RC_REFUSED, RC_FAILED_FAMILY, RC_TECHNICAL, PREVIEWED_READOUT, utc, logger,
    setup, finish, donor_inputs, windows, nonfinite_runs, failed_family_reason, nonfinite_selection,
)

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
    D, ft = a.finetune_update
    info = {"_t0": t0, "command": "donor", "status": None, "rc": None, "reasons": [], "reduced": a.reduced,
            "deviations": a.deviations, "resolved_recipe": a.resolved_recipe,
            "config": {"steps": a.steps, "every": a.every, "rows": a.rows, "lrs": a.resolved_recipe["donor"]["lrs"],
                       "eval_batch_size": a.eval_batch_size, "device": a.device, "threads": a.threads},
            "inputs": inputs, "code": code_identity(), "precision": None, "environment": environment(a.device),
            "utc_start": utc(), "runs": {}}
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str))
    model, base, precision = setup(a)
    info["precision"] = precision
    tech = []  # technical reasons (exit 9): the run stops at the stage that has one. No donor_repair.pt is written

    def technical():
        info["technical"] = {"donor": tech}
        return finish(out, info, RC_TECHNICAL, log, tech)

    if not all(bool(torch.isfinite(D[m]).all()) for m in MODULES):
        tech.append("non-finite value in the fine-tune update D")
        return technical()
    cap = math.sqrt(sum(float(D[m].double().norm() ** 2) for m in MODULES))
    fs = fit_sets(windows(a, info, "fit", FIT_LANGS, "donor_fit", a.rows).tokens, None, budgets=a.resolved_recipe["fitting"]["budgets"])
    info.update(cap=cap, fit=fit_accounting(fs))
    shapes = {m: tuple(base[m].shape) for m in MODULES}
    cands, nonfinite = {}, []
    for lr in a.resolved_recipe["donor"]["lrs"]:
        meth = make_method("fresh", shapes, a.device, cap, {}, {}, init_seed=a.resolved_recipe["fitting"]["init_seed"])
        t = time.time()
        ck, lg, st = train(model, base, D, meth, fs, lr, a.device, log, "donor_repair", a.steps, a.every, settings=a.resolved_recipe["fitting"])
        st["checkpoints_saved"] = sorted(ck)
        info["runs"][f"{lr:g}"] = {"status": st, "train_s": time.time() - t, "log": lg}
        mpc = st.get("max_post_over_cap")
        if mpc is not None and not (finite(mpc) and mpc <= 1 + a.resolved_recipe["fitting"]["cap_tol"]):
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
    info["nonfinite_selection"] = nonfinite_selection(vper, list(cands), idx, a.resolved_recipe["selection"])  # the rule would drop them silently
    tech += [f"non-finite selection values of {n}: {why}" for n, why in info["nonfinite_selection"].items()][:5]
    sel = select(vper, removed_fro, list(cands), idx, a.resolved_recipe["selection"])
    info["selection"] = sel
    info["selection_points"] = {n: point(vper, n, idx, a.resolved_recipe["selection"]) for n in cands}
    p_dev = point(vper, "finetuned", idx, a.resolved_recipe["selection"])  # the donor's own Dutch damage, as for targets
    info["damage_selection"] = {"d_nl_D": p_dev["d_nl_D"], "d_de_D": p_dev["d_de_D"],
                                "floor_ok": p_dev["d_nl_D"] >= a.resolved_recipe["block"]["damage_floor"]}
    log(f"donor repair selection: {sel['selected']}")
    if sel["selected"]:
        cspec = cands[sel["selected"]]
        info["parity_max_abs_logit"] = parity(model, base, D, cspec, sel_sets["nl"][:4], a.device)
        info["E_fro"] = removed_fro[sel["selected"]]
        info["E_over_cap"] = info["E_fro"] / cap
        if not (finite(info["parity_max_abs_logit"]) and info["parity_max_abs_logit"] <= a.resolved_recipe["evaluation"]["parity_tol"]):
            tech.append(f"parity of {sel['selected']}: {info['parity_max_abs_logit']}")
        if not (finite(info["E_over_cap"]) and info["E_over_cap"] <= 1 + a.resolved_recipe["fitting"]["cap_tol"]):
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
        p = point(rper, "donor_repair", pidx, a.resolved_recipe["selection"])  # primary: previewed dates excluded
        p_all = point(rper, "donor_repair", torch.arange(len(rd_w.dates)), a.resolved_recipe["selection"])  # disclosed sensitivity
        dD = rper["finetuned"]["nl"]["ce"] - rper["base"]["nl"]["ce"]
        bd = dD[pidx[stats.boot_indices(len(pidx), a.resolved_recipe["bootstrap"]["n_boot"], a.resolved_recipe["bootstrap"]["seed"])]].mean(1).sort().values
        lo = float(torch.quantile(bd, a.resolved_recipe["bootstrap"]["quantiles"][0]))
        info["damage_readout"] = {"d_nl_D": float(dD[pidx].mean()), "ci_lo": lo, "ci_lo_ok": lo >= a.resolved_recipe["donor"]["damage_ci_lo"]}
        # finite tensors can still give undefined estimates (a zero mean damage): the gate cannot be read
        tech += [f"non-finite readout estimate {tag}/{k}: {q[k]}" for tag, q in (("primary", p), ("all", p_all))
                 for k in stats.CHECKED if not finite(q[k])]
        tech += [f"non-finite readout estimate {tag}/retain[{g}]: {v}" for tag, q in (("primary", p), ("all", p_all))
                 for g, v in q["retain"].items() if not finite(v)]
        tech += [f"non-finite readout damage {k}: {v}" for k, v in info["damage_readout"].items()
                 if k != "ci_lo_ok" and not finite(v)]
        if tech:
            return technical()
        gate = {"r_nl": p["r_nl"] >= a.resolved_recipe["donor"]["gate_r_nl"], "german": german_feasible(p, a.resolved_recipe["selection"]),
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
        rec = {"format": 2, "study": "repair_transfer", "record": "donor_repair_fit",
               "protocol": {"resolved_recipe": a.resolved_recipe},
               "fit": {k: info["fit"][k] for k in fit_keys},
               "runs": {lr: {k: r["status"][k] for k in ("batch", "batches_presented")}
                        for lr, r in info["runs"].items()},
               "total_s": info["total_s"], "source": {"run_record_sha256": record_sha256}}
        (out / "fit_record.json").write_text(json.dumps(rec, indent=1) + "\n")
        (out / "SHA256SUMS").write_text("".join(f"{sha256_file(out / f)}  {f}\n" for f in DONOR_FILES))

    return finish(out, info, RC_OK, log, [], export)
