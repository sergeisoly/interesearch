"""Quality-cost curve and the cost-aware reading (papers/repair-transfer/index.qmd#sec-app-cost,
secondary, descriptive).

Families strengths, donor_init, donor_init_output, random_init and fresh are read on the grid budget
{256, 512, 2K} x step cap {50, 100, 200, none}. A cell reaches the reference fresh@2K (600 steps) when it is ok,
inside the readout constraints, and the paired interval of r_nl(cell) - r_nl(fresh@2K) has lo > -tolerance. The reach
cost C*(X) of a family in a currency is the cost of its cheapest reaching cell (ties: the first cell name).
Break-even K* = C_donor / (C*(fresh) - C*(X)) targets when C*(X) < C*(fresh): reuse costs C_donor + K C*(X),
fresh fitting K C*(fresh). The budget axis compares cells of both blocks, so the curve needs a passing
cross-block check. K* is computed in every currency. A gpu_s reading is an estimate (see costs) and supports a
saving or break-even statement only where the token currencies agree.
"""

from __future__ import annotations

import math

import torch

from .costs import CURRENCIES
from ..method.selection import BEST, STRENGTHS
from .stats import ci

CURVE_FAMILIES = (STRENGTHS, "donor_init", "donor_init_output", "random_init", "fresh")
CURVE_PAIRS = (("donor_init", "fresh"), ("donor_init", "random_init"), ("donor_init_output", "fresh"),
               (STRENGTHS, "fresh"))
EXTRA_CELLS = (f"{BEST}@2K", "fresh_seed2@2K", "random_init_seed2@2K", "donor_fixed")
REACH_FAMILIES = (STRENGTHS, BEST, "donor_init", "donor_init_output", "random_init", "fresh", "fresh_seed2",
                  "random_init_seed2", "donor_fixed")
BREAK_EVEN_FAMILIES = (STRENGTHS, BEST, "donor_init", "donor_init_output", "random_init", "random_init_seed2",
                       "donor_fixed")


def curve_name(fam: str, b: str, cap: int | None) -> str:
    return f"{fam}@{b}" + (f":le{cap}" if cap else "")


def curve_contrasts(settings: dict) -> list[tuple[str, str]]:
    return [(curve_name(x, b, c), curve_name(y, b, c))
            for b in settings["budgets"] for c in (*settings["step_caps"], None) for x, y in CURVE_PAIRS]


def cost_curve(cells: dict, draws: dict, cost: dict, cross_block: dict, donor: dict,
               curve_settings: dict, analysis_settings: dict, bootstrap_settings: dict) -> dict:
    reference = curve_settings["reference"]
    tolerance = analysis_settings["equivalence_margin"]
    out = {"reference": reference, "tolerance": tolerance, "currencies": list(CURRENCIES), "grid": {}, "reach": {},
           "break_even": {}}
    if not cross_block["ok"]:
        out["status"] = "not assessable: cross-block check failed (" + "; ".join(cross_block["errors"]) + ")"
        return out
    if cells[reference]["status"] != "ok":
        out["status"] = f"not assessable: reference {reference} is {cells[reference]['status']}"
        return out
    out["status"] = "ok"
    names = [curve_name(f, b, c) for f in CURVE_FAMILIES for b in curve_settings["budgets"]
             for c in (*curve_settings["step_caps"], None)]
    for g in names + list(EXTRA_CELLS):
        c = cells[g]
        row = {"status": c["status"], "r_nl": c.get("point", {}).get("r_nl"), "ci_r_nl": c.get("ci_r_nl"),
               "constraints_ok": c.get("constraints_ok"), "cost": cost[g], "reaches": None}
        if c["status"] == "ok":
            dd = draws[g] - draws[reference]
            lo, hi = ci(dd, bootstrap_settings["quantiles"])
            row["vs_reference_ci"] = [lo, hi]
            if bool(torch.isfinite(dd).all()):
                row["reaches"] = bool(math.isfinite(lo) and lo > -tolerance and c["constraints_ok"])
        out["grid"][g] = row

    for fam in REACH_FAMILIES:
        rows = {g: r for g, r in out["grid"].items() if g.split("@")[0] == fam}
        rec = {"cells": len(rows), "not_assessable": sum(r["reaches"] is None for r in rows.values()),
               "reaching": sorted(g for g, r in rows.items() if r["reaches"])}
        for cur in CURRENCIES:
            cand = [(r["cost"][cur], g) for g, r in rows.items() if r["reaches"]]
            rec[cur] = {"cost": min(cand)[0], "cell": min(cand)[1]} if cand else None
        out["reach"][fam] = rec

    fresh = out["reach"]["fresh"]
    for fam in BREAK_EVEN_FAMILIES:
        be = {}
        for cur in CURRENCIES:
            cx, cf = out["reach"][fam][cur], fresh[cur]
            if cx is None:
                be[cur] = {"reading": "does not reach the reference"}
            elif cf is None:
                be[cur] = {"reading": "fresh reach cost not available"}
            elif cx["cost"] >= cf["cost"]:
                be[cur] = {"reading": "no per-target saving", "reuse": cx["cost"], "fresh": cf["cost"]}
            else:
                k = donor[cur] / (cf["cost"] - cx["cost"])
                be[cur] = {"reading": f"reuse pays after K > {k:.3g} targets", "k_star": k, "reuse": cx["cost"],
                           "fresh": cf["cost"], "saving_per_target": cf["cost"] - cx["cost"]}
        out["break_even"][fam] = be
    return out
