"""Selection of repair candidates on the selection windows, as fixed in papers/repair-transfer/index.qmd#sec-app-methods.

Point readouts are means over windows of per-window token means, in float64. A candidate is feasible when it
keeps the fine-tune's German effect (rho_de in [0.9, 1.1]), adds little German drift (phi_de <= 0.05) and moves
the Dutch distribution toward the base model (r_kl > 0). Feasible candidates are ranked by Dutch recovery r_nl,
then by smaller removed Frobenius norm, then by listing order. The first of the top five that also keeps every
retain language within 0.01 nat/token of the fine-tuned model is selected.
"""

from __future__ import annotations

import math

import torch

STRENGTHS, CONVERGED, BEST = "strengths", "strengths_converged", "strengths_best"


def _mean(t: torch.Tensor, idx: torch.Tensor) -> float:
    return float(t[idx].to(torch.float64).mean())


def point(per: dict, name: str, idx: torch.Tensor, settings: dict) -> dict:
    """Point readouts of candidate `name` on rows `idx`, relative to the base model and the fine-tuned model."""
    X, W0, D = per[name], per["base"], per["finetuned"]
    d_nl_D = _mean(D["nl"]["ce"] - W0["nl"]["ce"], idx)
    d_nl = _mean(X["nl"]["ce"] - W0["nl"]["ce"], idx)
    kb_D = _mean(D["nl"]["kl_base"], idx)
    kb = _mean(X["nl"]["kl_base"], idx)
    d_de_D = _mean(D["de"]["ce"] - W0["de"]["ce"], idx)
    d_de = _mean(X["de"]["ce"] - W0["de"]["ce"], idx)
    B_de = _mean(W0["de"]["kl_ref"], idx)
    k_de = _mean(X["de"]["kl_ref"], idx)
    out = {
        "d_nl": d_nl, "d_nl_D": d_nl_D, "r_nl": (1.0 - d_nl / d_nl_D) if d_nl_D != 0 else math.nan,
        "kb_nl": kb, "kb_nl_D": kb_D, "r_kl": (1.0 - kb / kb_D) if kb_D > 0 else math.nan,
        "d_de": d_de, "d_de_D": d_de_D, "rho_de": d_de / d_de_D if d_de_D != 0 else math.nan,
        "kl_de": k_de, "B_de": B_de, "phi_de": k_de / B_de if B_de > 0 else math.nan,
    }
    retain = settings["retain_languages"]
    if all(g in X for g in retain):
        out["retain"] = {g: _mean(X[g]["ce"] - D[g]["ce"], idx) for g in retain}
        out["retain_ok"] = all(v <= settings["retain_tol"] for v in out["retain"].values())
    return out


def german_feasible(p: dict, settings: dict) -> bool:
    return settings["rho_lo"] <= p["rho_de"] <= settings["rho_hi"] and p["phi_de"] <= settings["phi_max"]


def feasible(p: dict, settings: dict) -> bool:
    return german_feasible(p, settings) and math.isfinite(p["r_kl"]) and p["r_kl"] > 0


def ranked_feasible(per: dict, removed_fro: dict, names: list[str], idx: torch.Tensor, settings: dict) -> list[str]:
    rows = []
    for order, n in enumerate(names):
        p = point(per, n, idx, settings)
        if feasible(p, settings) and math.isfinite(p["r_nl"]):
            rows.append((-p["r_nl"], removed_fro[n], order, n))
    return [r[-1] for r in sorted(rows)]


def select(per: dict, removed_fro: dict, names: list[str], idx: torch.Tensor, settings: dict) -> dict:
    ranking = ranked_feasible(per, removed_fro, names, idx, settings)
    tried = []
    for n in ranking[:settings["top_retain"]]:
        p = point(per, n, idx, settings)
        if "retain_ok" not in p:
            raise KeyError(f"{n}: retain languages are not measured")
        tried.append({"name": n, "r_nl": p["r_nl"], "r_kl": p["r_kl"], "retain": p["retain"], "retain_ok": p["retain_ok"]})
        if p["retain_ok"]:
            return {"selected": n, "n_feasible": len(ranking), "tried": tried}
    return {"selected": None, "n_feasible": len(ranking), "tried": tried}


def groups(info: dict) -> dict[str, list[str]]:
    """Candidate groups in listing order: one per family and budget, the best strengths checkpoints and their
    polished versions in blocks with polish, and step-capped subsets of the capped families."""
    proto, cands = info["protocol"], info["candidates"]

    def members(family: str) -> list[str]:
        return [c["name"] for c in cands if c["family"] == family]

    out: dict[str, list[str]] = {}
    for b in proto["budgets"]:
        for fam in proto["families"]:
            out[f"{fam}@{b}"] = members(f"{fam}@{b}")
        if proto["polish"] and STRENGTHS in proto["families"]:
            out[f"{BEST}@{b}"] = out[f"{STRENGTHS}@{b}"] + members(f"{CONVERGED}@{b}")
        step = {c["name"]: c.get("step") for c in cands}
        for fam in proto["capped_families"]:
            for cap in proto["step_caps"]:
                if proto["every"] <= cap < proto["steps"]:
                    out[f"{fam}@{b}:le{cap}"] = [n for n in out[f"{fam}@{b}"] if step[n] <= cap]
    for g, names in out.items():
        if not names:
            raise ValueError(f"group {g} has no candidates")
    return out


def reselect(block, settings: dict) -> dict[str, dict]:
    """Replay the selection of every group of a block on its selection windows."""
    idx = torch.arange(block.info["selection"]["rows"])
    removed_fro = {c["name"]: c["removed_fro"] for c in block.info["candidates"]}
    return {g: select(block.selection, removed_fro, names, idx, settings) for g, names in groups(block.info).items()}
