"""Replay of the repair-transfer analysis from the exported blocks.
Rules: papers/repair-transfer/index.qmd#sec-app-methods, papers/repair-transfer/index.qmd#sec-fixed and
papers/repair-transfer/index.qmd#sec-warm.

Each target has a main block (2K fit tokens) and a nested block (256 and 512). Selections are replayed on the
selection windows. Every chosen repair is read out on the same readout dates in both blocks, and one joint date
bootstrap (5,000 draws, seed 10610) is shared by every cell and contrast of the target, so contrasts across the
blocks are paired too. Recovery r_nl = 1 - d_nl(X) / d_nl(finetuned) is the share of the fine-tune's Dutch CE
damage that repair X removes.

Cell status, in order of precedence (a contrast takes the worse status of its two cells):
    technical_invalid      a non-finite input, estimate or bootstrap draw, a replayed selection that differs from
                           the recorded one, or a chosen repair missing from the readout
    non_evaluable_target   the fine-tune's readout damage is not clearly positive (CI lo < 0.025)
    no_choice              no candidate passed selection
    descriptive            donor_raw, the donor repair without the norm cap (a reference)
    ok

Primary contrast r_nl(donor_init@2K) - r_nl(strengths_best@2K), published margin delta = 0.03:
    basis limit supported      CI lo > delta
    strengths nearly suffice   CI hi < delta
    inconclusive               otherwise
    not assessable: <reason>   a primary cell is not ok or is outside the readout constraints, or the original run
                               did not verify that the strengths arm was optimized to convergence
Secondary contrasts read better (CI lo > 0), worse (CI hi < 0) or inconclusive. The pairs in EQUIV_PAIRS are also
checked for an interval inside +-0.02. Secondary readings are descriptive, without multiplicity correction. The
replication target is read with the same rules and is not a second primary. The reading keys are the protocol's.
The papers word them as in READING_TEXT.

    uv run python -m interesearch.repair_transfer.analysis.recovery data/repair_transfer \
        --out results/repair_transfer/analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .costs import donor_cost, fit_and_cost
from .curve import CURVE_FAMILIES, cost_curve, curve_contrasts
from ..data import load_block, load_donor
from ..method.selection import BEST, STRENGTHS, german_feasible, point, reselect
from .stats import boot_indices, ci, nonfinite, nonfinite_point, r_nl_draws
from ...recipes import load_recipe

ROLES = {"primary_target": "primary", "replication_target": "replication"}
PRECEDENCE = ("technical_invalid", "non_evaluable_target", "no_choice", "descriptive", "ok")
PRIMARY = ("donor_init@2K", f"{BEST}@2K")
MAIN_CONTRASTS = [("donor_init_output@2K", f"{BEST}@2K"), ("donor_init@2K", "donor_init_output@2K"),
                  ("donor_init@2K", "fresh@2K"), ("donor_init_output@2K", "fresh@2K"),
                  ("donor_init@2K", "random_init@2K"), ("random_init@2K", "fresh@2K"), (f"{BEST}@2K", "fresh@2K"),
                  (f"{BEST}@2K", "donor_fixed"), (f"{BEST}@2K", f"{STRENGTHS}@2K"), ("fresh_seed2@2K", "fresh@2K"),
                  ("random_init_seed2@2K", "random_init@2K"), ("donor_init@2K", "fresh_seed2@2K")]
NESTED_CONTRASTS = [(f"{x}@{b}", f"{y}@{b}") for b in ("256", "512") for x, y in (
    ("donor_init", "fresh"), ("donor_init_output", "fresh"), ("donor_init", "donor_init_output"),
    ("donor_init", STRENGTHS), (STRENGTHS, "fresh"), ("donor_init", "random_init"), ("random_init", "fresh"))]
CROSS_CONTRASTS = [(f"{f}@2K", f"{f}@{b}") for f in CURVE_FAMILIES for b in ("256", "512")]
EQUIV_PAIRS = {("donor_init@2K", "fresh@2K"), ("donor_init@2K", "donor_init_output@2K"),
               ("fresh_seed2@2K", "fresh@2K"), ("random_init_seed2@2K", "random_init@2K"),
               ("random_init@2K", "fresh@2K")}
POINT_KEYS = ("r_nl", "r_kl", "rho_de", "phi_de", "d_nl", "retain", "retain_ok")
# Public wording of each reading, fixed by a protocol amendment (papers/repair-transfer/index.qmd#sec-fixed):
# the keys stay, the scope is stated.
READING_TEXT = {
    "basis limit supported": (
        "Donor-initialized fitting exceeds the scalar-coefficient search including full-batch optimization "
        "by more than {pp} percentage points. This shows a practical limit of the scalar-coefficient "
        "optimization procedure (4 learning rates × 600 steps plus two full-batch optimizations, with 50 "
        "candidates in total). It does not show that no assignment of scalar coefficients over the donor "
        "directions can reach donor-initialized fitting."),
    "strengths nearly suffice": (
        "An advantage of donor-initialized fitting over the scalar-coefficient search including "
        "optimization to convergence of more than {pp} percentage points is excluded. This is not "
        "equivalence: no lower margin is set, and the scalar-coefficient search may be better. The reading "
        "is relative to donor-initialized fitting and says nothing about the scalar-coefficient search "
        "against fresh fitting or an absolute repair target."),
    "inconclusive": (
        "An advantage of donor-initialized fitting over the scalar-coefficient search including "
        "optimization to convergence of more than {pp} percentage points is neither established nor "
        "excluded."),
}


def worst(*statuses: str) -> str:
    return min(statuses, key=PRECEDENCE.index)


def cross_check(main, nested, tolerance: float) -> list[str]:
    """Main and nested blocks of one target: the same fine-tune update, dates and base/finetuned readouts."""
    err = []
    if main.info["recorded"]["finetune_update_sha256"] != nested.info["recorded"]["finetune_update_sha256"]:
        err.append("main and nested blocks loaded different fine-tune updates")
    if main.info["readout"]["dates"] != nested.info["readout"]["dates"]:
        err.append("main and nested readout dates differ")
    else:
        diff = max(float((nested.readout[r][lg][k] - main.readout[r][lg][k]).abs().max())
                   for r in ("base", "finetuned") for lg in main.readout[r] for k in main.readout[r][lg])
        if not diff <= tolerance:
            err.append(f"base/finetuned readouts differ between the blocks by {diff:.3g} > {tolerance}")
    return err


def replay_selections(block, selection_settings: dict) -> tuple[dict, dict]:
    replay = reselect(block, selection_settings)
    recorded = block.info["recorded"]["selections"]
    mismatches = sorted(g for g, r in replay.items()
                        if (r["selected"], r["n_feasible"]) != (recorded[g]["selected"], recorded[g]["n_feasible"]))
    return replay, {"groups": len(replay), "matching_recorded": len(replay) - len(mismatches),
                    "mismatches": mismatches}


def read_cells(block, replay: dict, mismatches: list, target_problems: list, damage: dict, idx, draws,
               selection_settings: dict, analysis_settings: dict, bootstrap_settings: dict):
    """Point estimates, intervals and status of every cell of one block. Returns the cells and their r_nl draws."""
    info, per = block.info, block.readout
    removed_fro = {c["name"]: c["removed_fro"] for c in info["candidates"]}
    entries = {g: (f"{g}::{r['selected']}" if r["selected"] else None, r["selected"], r["n_feasible"])
               for g, r in replay.items()}
    if info["block"] == "main":
        entries |= {ref: (ref, None, None) for ref in ("donor_fixed", "donor_raw")}
    cells, cell_draws = {}, {}
    for g, (name, selected, n_feasible) in entries.items():
        problems = list(target_problems)
        if g in mismatches:
            problems.append("replayed selection differs from the recorded one")
        status = ("technical_invalid" if problems else "no_choice" if name is None
                  else "descriptive" if g == "donor_raw" else "ok")
        cell = {"block": info["block"], "name": name, "selected": selected, "n_feasible": n_feasible,
                "problems": problems}
        if name is not None and name not in per:
            problems.append(f"{name} is missing from the readout")
            status = "technical_invalid"
        elif name is not None:
            p = point(per, name, idx, selection_settings)
            d = r_nl_draws(per, name, draws)
            bad = nonfinite(per, (name,)) + nonfinite_point(p)
            if not bool(torch.isfinite(d).all()):
                bad.append("non-finite bootstrap draws of r_nl")
            if bad:
                problems += bad
                status = "technical_invalid"
            cell_draws[g] = d
            cell["point"] = {k: p[k] for k in POINT_KEYS}
            cell["ci_r_nl"] = list(ci(d, bootstrap_settings["quantiles"]))
            cell["constraints_ok"] = bool(german_feasible(p, selection_settings)
                                          and math.isfinite(p["r_kl"]) and p["r_kl"] > 0 and p["retain_ok"])
            cell["norm"] = removed_fro[name.split("::")[-1]]
        if status != "technical_invalid" and not damage["clearly_positive"]:
            status = "non_evaluable_target"
            problems.append(f"readout damage CI lo {damage['ci'][0]} < {analysis_settings['damage_ci_lo']}")
        cell["status"] = status
        cells[g] = cell
    return cells, cell_draws


def contrast(cells: dict, cell_draws: dict, x: str, y: str, analysis_settings: dict,
             bootstrap_settings: dict, cross_errors=()) -> dict:
    cx, cy = cells[x], cells[y]
    out = {"x": x, "y": y, "status": worst(cx["status"], cy["status"]),
           "problems": sorted(set(cx["problems"] + cy["problems"]))}
    if cross_errors:
        out["status"] = "technical_invalid"
        out["problems"] += cross_errors
    if x in cell_draws and y in cell_draws:
        dd = cell_draws[x] - cell_draws[y]
        px, py = cx["point"]["r_nl"], cy["point"]["r_nl"]
        out["diff"] = px - py if math.isfinite(px) and math.isfinite(py) else math.nan
        out["ci"] = list(ci(dd, bootstrap_settings["quantiles"]))
        if not bool(torch.isfinite(dd).all()):
            out["status"] = "technical_invalid"
        if out["status"] == "ok":
            lo, hi = out["ci"]
            out["reading"] = "better" if lo > 0 else "worse" if hi < 0 else "inconclusive"
            if (x, y) in EQUIV_PAIRS:
                margin = analysis_settings["equivalence_margin"]
                out["equivalent_within"] = margin
                out["equivalent"] = bool(-margin < lo and hi < margin)
    return out


def primary_reading(cells: dict, contrasts: dict, main_recorded: dict, delta: float) -> dict:
    x, y = PRIMARY
    c = contrasts[f"{x} - {y}"]
    out = {"contrast": f"{x} - {y}", "delta": delta, "diff": c.get("diff"), "ci": c.get("ci")}
    why = []
    for g in PRIMARY:
        if cells[g]["status"] != "ok":
            why.append(f"{g} is {cells[g]['status']} ({'; '.join(cells[g]['problems'][:3])})")
        elif not cells[g]["constraints_ok"]:
            why.append(f"{g} is outside the readout constraints")
    if not main_recorded["s_adequacy_verified"]:
        why.append("the strengths arm was not verified to be optimized to convergence (" + ", ".join(
            f"polish from {k}: stop {v['stop']}, converged {v['converged']}"
            for k, v in main_recorded["polish"].items()) + ")")
    if c["status"] != "ok" and not why:
        why.append(f"contrast {c['status']}")
    if why:
        out["reading"] = "not assessable: " + "; ".join(why)
        return out
    lo, hi = c["ci"]
    out["reading"] = ("basis limit supported" if lo > delta else "strengths nearly suffice" if hi < delta
                      else "inconclusive")
    return out


def analyze_target(target: str, main, nested, donor: dict, recipe: dict) -> dict:
    """Every cell and contrast of one target, its primary reading, in-sample fit, costs and quality-cost curve."""
    selection_settings, analysis_settings = recipe["selection"], recipe["analysis"]
    bootstrap_settings = recipe["bootstrap"]
    per = main.readout
    n = len(main.info["readout"]["dates"])
    idx, draws = torch.arange(n), boot_indices(n, bootstrap_settings["n_boot"], bootstrap_settings["seed"])
    target_problems = nonfinite(per, ("base", "finetuned"))  # the target's references, read from the main block
    dD = per["finetuned"]["nl"]["ce"] - per["base"]["nl"]["ce"]
    lo, hi = ci(dD[draws].mean(1), bootstrap_settings["quantiles"])
    damage = {"selection_d_nl": main.info["recorded"]["selection_damage_d_nl"], "d_nl": float(dD.mean()),
              "ci": [lo, hi], "clearly_positive": math.isfinite(lo) and lo >= analysis_settings["damage_ci_lo"]}
    cross_errors = cross_check(main, nested, analysis_settings["cross_block_tol"])
    res = {"target": target, "role": ROLES[target], "n_dates": n, "damage": damage, "problems": target_problems,
           "cross_block": {"ok": not cross_errors, "errors": cross_errors}, "blocks": {}, "cells": {},
           "contrasts": {}, "fit": {}, "fit_best": {}, "cost": {}, "direction": {}}
    cell_draws = {}
    for block in (main, nested):
        info, rec = block.info, block.info["recorded"]
        replay, summary = replay_selections(block, selection_settings)
        cells, d = read_cells(block, replay, summary["mismatches"], target_problems, damage, idx, draws,
                              selection_settings, analysis_settings, bootstrap_settings)
        res["cells"] |= cells
        cell_draws |= d
        fit, fit_best, cost = fit_and_cost(info, replay, recipe["costs"]["selection_tokens"])
        res["fit"] |= fit
        res["fit_best"] |= fit_best
        res["cost"] |= cost
        res["direction"] |= rec["direction_cos"]
        res["blocks"][info["block"]] = {
            "selection_replay": summary, "s_adequacy_verified": rec["s_adequacy_verified"],
            "total_s": rec["total_s"], "norm_cap": rec["norm_cap"], "donor_fixed_factor": rec["donor_fixed_factor"]}
    for x, y in [PRIMARY, *MAIN_CONTRASTS, *NESTED_CONTRASTS, *curve_contrasts(recipe["curve"])]:
        if f"{x} - {y}" not in res["contrasts"]:
            res["contrasts"][f"{x} - {y}"] = contrast(
                res["cells"], cell_draws, x, y, analysis_settings, bootstrap_settings)
    for x, y in CROSS_CONTRASTS:
        res["contrasts"][f"{x} - {y}"] = contrast(
            res["cells"], cell_draws, x, y, analysis_settings, bootstrap_settings, cross_errors)
    rec = main.info["recorded"]
    res["primary"] = primary_reading(res["cells"], res["contrasts"], rec, analysis_settings["delta"])
    res["polish"] = rec["polish"]
    res["polish_best_start"] = rec["polish_best_start"]
    res["polish_agreement_rel"] = rec["polish_agreement_rel"]
    res["curve"] = cost_curve(res["cells"], cell_draws, res["cost"], res["cross_block"], donor,
                              recipe["curve"], analysis_settings, bootstrap_settings)
    return res


REPLAY_SECTIONS = ("selection", "analysis", "bootstrap", "curve", "costs")


def replay_recipe(info: dict, published_recipe: dict) -> dict:
    """Use the saved recipe; only the published format-1 exports have a legacy fallback."""
    snapshot = info.get("protocol", {}).get("resolved_recipe")
    if snapshot is not None:
        # New records must be complete. Archived records are not user YAML and are not passed through
        # the strict YAML loader; require the sections used here without inventing missing settings.
        for section in REPLAY_SECTIONS:
            for key in published_recipe[section]:
                if key not in snapshot.get(section, {}):
                    raise ValueError(f"saved recipe is missing {section}.{key}")
        return snapshot
    legacy = (info.get("format") == 1 and info.get("study") == "repair_transfer" and "run" not in info
              and (info.get("record") == "donor_repair_fit"
                   or (info.get("target") in ROLES and info.get("block") in ("main", "nested"))))
    if not legacy:
        raise ValueError("record has no resolved recipe and is not a published legacy export")
    return published_recipe


def _replay_settings(recipe: dict) -> dict:
    return {section: recipe[section] for section in REPLAY_SECTIONS}


def analyze(data_dir: str | Path, published_recipe: dict | None = None,
            expected_recipe: dict | None = None) -> dict:
    """Replay saved settings, with the packaged published recipe only for recognized legacy exports.

    ``expected_recipe`` optionally checks the requested analysis profile. It never replaces recorded values.
    ``published_recipe`` can be supplied by an entrypoint that has already loaded the packaged profile.
    """
    if published_recipe is None:
        published_recipe = load_recipe("interesearch.repair_transfer")
    data_dir = Path(data_dir)
    blocks = {t: tuple(load_block(data_dir / t / b) for b in ("main", "nested")) for t in ROLES}
    recipes = {t: tuple(replay_recipe(b.info, published_recipe) for b in bs) for t, bs in blocks.items()}
    recipe = recipes["primary_target"][0]
    settings = _replay_settings(recipe)
    if any(_replay_settings(r) != settings for rs in recipes.values() for r in rs):
        raise ValueError("blocks have different recorded replay settings")
    if expected_recipe is not None and _replay_settings(expected_recipe) != settings:
        raise ValueError("requested recipe differs from the saved replay settings")
    donor_record = load_donor(data_dir / "donor")
    donor_recipe = replay_recipe(donor_record, published_recipe)
    donor = donor_cost(donor_record, donor_recipe["costs"]["selection_tokens"])
    targets = {t: analyze_target(t, *bs, donor, recipe) for t, bs in blocks.items()}
    result = {"study": "repair_transfer", "n_boot": recipe["bootstrap"]["n_boot"],
              "boot_seed": recipe["bootstrap"]["seed"], "delta": recipe["analysis"]["delta"],
              "equivalence_margin": recipe["analysis"]["equivalence_margin"], "donor_cost": donor,
              "primary": targets["primary_target"]["primary"], "targets": targets}
    if any("resolved_recipe" in b.info.get("protocol", {}) for bs in blocks.values() for b in bs):
        result["resolved_recipe"] = recipe
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay the repair-transfer analysis from the exported blocks.")
    ap.add_argument("data_dir", type=Path, help="data/repair_transfer")
    ap.add_argument("--out", type=Path, required=True, help="results JSON")
    ap.add_argument("--recipe", type=Path,
                    help="verify the saved replay settings against this recipe; does not override them")
    a = ap.parse_args(argv)
    published = load_recipe("interesearch.repair_transfer")
    expected = load_recipe("interesearch.repair_transfer", a.recipe) if a.recipe else None
    res = analyze(a.data_dir, published, expected)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["primary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
