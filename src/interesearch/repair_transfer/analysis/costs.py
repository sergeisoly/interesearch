"""In-sample fit, target-side cost of every cell, and the donor's one-time cost (papers/repair-transfer/index.qmd#sec-app-cost).

Target-side cost has three currencies:
    unique_tokens             the cell's unique fit tokens plus the selection windows every trained cell reads
    fit_token_presentations   fit tokens times the executed optimizer steps, polish evaluations and start-search
                              evaluations (each one pass over the cell's fit tokens)
    gpu_s                     training, polish and start-search seconds plus the cell's share of the selection
                              evaluation (by its number of candidates)
A step cap is the same runs stopped at the cap, so steps and seconds are prorated. strengths_best adds both
polishes and the fit-loss evaluations of every strengths checkpoint that pick the polish start. The donor
fine-tune is listed, not added to the donor cost: in use the donor model exists already.

The token currencies are the primary, hardware-free ones (papers/repair-transfer/index.qmd#sec-app-cost). gpu_s only estimates
allocated shared work: the selection share is allocated by candidate counts, the fixed base and fine-tuned readout
forwards and the pre-training gate are in no cell, and donor and target seconds come from different runs.
"""

from __future__ import annotations

import math

from ..method.selection import BEST, CONVERGED, STRENGTHS, groups

CURRENCIES = ("unique_tokens", "fit_token_presentations", "gpu_s")
FINETUNE_AND_REPAIR, NORMS_ONLY = "donor fine-tune and repair", "donor repair norms only"
DONOR_INPUTS = {STRENGTHS: FINETUNE_AND_REPAIR, "donor_init": FINETUNE_AND_REPAIR,
                "donor_init_output": FINETUNE_AND_REPAIR, "random_init": NORMS_ONLY, "random_init_seed2": NORMS_ONLY,
                "fresh": "none", "fresh_seed2": "none"}
DONOR_FINETUNE = ("2 fine-tuning runs (seed 11; learning rates 3e-6 and 1e-5; 200 steps; 64 dates), learning "
                  "rate chosen on the donor's selection dates")


def donor_cost(rec: dict, selection_tokens: int) -> dict:
    """C_donor from the donor repair's fit record: unique tokens = its fit tokens + the selection windows.
    Fit-token presentations = executed steps x the tokens of one step's minibatch (batch windows of German, of
    Dutch and of every retain language, window length = unique tokens / windows). GPU seconds = the stage's total
    time (training, selection and readout)."""
    up, win = rec["fit"]["unique_predicted_tokens"], rec["fit"]["windows"]
    wl = {"de": up["de"] / win["de"], "nl": up["nl"] / win["nl"], "ret": up["ret"] / sum(win["ret"].values())}
    per_step, steps, pres = set(), 0, 0
    for r in rec["runs"].values():
        b = r["batch"]
        k = b["de"] * wl["de"] + b["nl"] * wl["nl"] + len(win["ret"]) * b["ret"] * wl["ret"]
        per_step.add(k)
        steps += r["batches_presented"]
        pres += r["batches_presented"] * k
    return {"unique_tokens": int(rec["fit"]["total_unique_predicted_tokens"]) + selection_tokens,
            "fit_token_presentations": int(pres), "gpu_s": float(rec["total_s"]),
            "tokens_per_step": int(per_step.pop()) if len(per_step) == 1 else sorted(per_step),
            "steps": steps, "runs": len(rec["runs"]), "window_tokens": wl,
            "batch": {k: r["batch"] for k, r in rec["runs"].items()}, "donor_finetune_not_added": DONOR_FINETUNE}


def _r_kl_fit(v: dict, f0: dict) -> float:
    return 1 - v["nl"] / f0["nl"] if f0["nl"] > 0 else math.nan


def fit_and_cost(info: dict, selections: dict, selection_tokens: int) -> tuple[dict, dict, dict]:
    """In-sample fit of each selected candidate and of each family's best checkpoint (exact full-batch loss on
    the cell's own fit set, r_kl_fit_nl = 1 - KL_nl(X) / KL_nl(finetuned)), and the target-side cost of every
    cell of one block."""
    proto, rec = info["protocol"], info["recorded"]
    fit = rec["fit_loss"]
    fit_sel, fit_best, cost = {}, {}, {}
    for b in proto["budgets"]:
        f0 = fit[f"finetuned@{b}"]
        heads = [h for h in dict.fromkeys(n.split(":")[0] for n in fit if ":" in n) if h.endswith(f"@{b}")]
        members = {h: (h,) for h in heads}
        if f"{STRENGTHS}@{b}" in members:
            members[f"{BEST}@{b}"] = (f"{STRENGTHS}@{b}", f"{CONVERGED}@{b}")
        for g, hs in members.items():
            rows = {n: v for n, v in fit.items() if n.split(":")[0] in hs}
            n = min(rows, key=lambda k: (rows[k]["total"], k))
            fit_best[g] = {"best": n, **rows[n], "r_kl_fit_nl": _r_kl_fit(rows[n], f0)}
        for ref in ("donor_fixed", "finetuned"):
            v = fit[f"{ref}@{b}"]
            fit_best[f"{ref}@{b}"] = {"best": ref, **v, "r_kl_fit_nl": _r_kl_fit(v, f0)}

    sizes = {g: len(ns) for g, ns in groups(info).items()}
    step = {c["name"]: c.get("step") for c in info["candidates"]}  # converged candidates have no step
    n_all, sel_s = len(info["candidates"]), rec["selection_eval_s"]
    for g, sel in selections.items():
        n = sel["selected"]
        fam, b = g.split(":")[0].split("@")
        cap = int(g.split(":le")[1]) if ":le" in g else None
        if n:
            fit_sel[g] = {"selected": n, **fit[n], "r_kl_fit_nl": _r_kl_fit(fit[n], fit[f"finetuned@{b}"])}
        run_fam = STRENGTHS if fam == BEST else fam
        runs = [r for k, r in rec["runs"].items() if k.startswith(f"{run_fam}@{b}:")]
        tok = rec["fit_tokens"][b]
        steps, train_s = 0, 0.0
        for r in runs:
            u = r["updates_applied"]
            k = min(cap, u) if cap else u
            steps += k
            train_s += r["train_s"] * (k / u)
        polish_ev = start_ev = 0
        polish_s = start_s = 0.0
        if fam == BEST:
            pol = rec["polish"].values()
            polish_s = sum(p["seconds"] for p in pol)
            polish_ev = sum(p["evals"] + 1 for p in pol)  # each polish: its full-batch evaluations + 1 parity export
            start = rec["fit_loss_evaluations"][f"{STRENGTHS}@{b}"]
            start_ev, start_s = start["n"], start["s"]
        n_c = sizes[g]
        sel_share = sel_s * n_c / n_all
        cost[g] = {
            "budget": b, "step_cap": cap or proto["steps"], "unique_fit_tokens": tok,
            "unique_tokens": tok + selection_tokens, "runs": len(runs), "steps_executed": steps,
            "polish_evaluations": polish_ev, "start_search_evaluations": start_ev,
            "fit_token_presentations": (steps + polish_ev + start_ev) * tok,
            "step_of_choice": step[n] if n else None,
            "train_s": train_s, "polish_s": polish_s, "start_search_s": start_s, "selection_s_share": sel_share,
            "gpu_s": train_s + polish_s + start_s + sel_share,
            "selection_candidates": n_c, "selection_predicted_tokens": n_c * selection_tokens,
            "donor_inputs": DONOR_INPUTS[run_fam],
        }
    if info["block"] == "main":  # the fixed donor repair is a cell of the main block and fits nothing on the target
        cost["donor_fixed"] = {
            "budget": None, "step_cap": 0, "unique_fit_tokens": 0, "unique_tokens": 0, "runs": 0, "steps_executed": 0,
            "polish_evaluations": 0, "start_search_evaluations": 0, "fit_token_presentations": 0,
            "step_of_choice": None, "train_s": 0.0, "polish_s": 0.0, "start_search_s": 0.0, "selection_s_share": 0.0,
            "gpu_s": 0.0, "selection_candidates": 0, "selection_predicted_tokens": 0,
            "donor_inputs": FINETUNE_AND_REPAIR,
        }
    return fit_sel, fit_best, cost
