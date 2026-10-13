"""Frozen analysis of the selective-editing readouts, replayed on the dev and the confirm split with one code path.
Rules: papers/selective-editing/index.qmd#sec-analysis.

Each edit of a cell is first read by the guard rule: on dev an edit is usable when every English guard raises CE by
at most 0.1 nats. On confirm it must also have been usable on dev and stay within 0.1 nats on the confirm English
set. A target-test rise more than 0.1 nats from tau is flagged as a diagnostic, without recalibration. Each edit
then gets a state: "ok", "declared" (not matched, no edit by the method's declared rule, or unusable) or
"technical" (absent from the readout, no edit for a technical reason, or a matched edit's per-document readout
missing or holding a non-finite number). A non-finite readout is technical, not declared "unusable".

The per-document protected CE rise is computed here from the exported CE (ce_doc - base_ce, float64), converted
to float32 and averaged over the used draws in float32, as the frozen analyzer did. The bootstrap then runs in
float64.
"""

from __future__ import annotations

import math

import torch

from .data import Cell, Study, expected_cells

N_BOOT, SEED, LEVEL = 2000, 0, 0.95
HISTORICAL = "historical_atom"
FAMILIES = (("claim", "claim_comparators"), ("references", "reference_comparators"))


def _sd(B: torch.Tensor) -> torch.Tensor:
    """Sample SD (n - 1) of each column, by Welford's sequential update in draw order. The frozen analyzer used
    ``B.std(0)``. Its stored results were computed on x86, and this fixed order reproduces them bit for bit on the
    checked Apple M4 Mac. This changes the reduction. Bitwise agreement on other CPUs is not established."""
    out = []
    for col in B.T.tolist():
        m = s = 0.0
        for k, x in enumerate(col, 1):
            d = x - m
            m += d / k
            s += d * (x - m)
        out.append(math.sqrt(s / (len(col) - 1)))
    return torch.tensor(out, dtype=torch.float64)


def simultaneous(contrasts: dict[str, tuple[torch.Tensor, list]], n_boot: int = N_BOOT, seed: int = SEED,
                 level: float = LEVEL, sd=_sd) -> dict:
    """Per contrast: mean of per-document differences, bootstrap SE over date groups (shared resampling for all
    contrasts, so the intervals are simultaneous by the max-|t| method). ``sd`` reduces the bootstrap draws to SEs."""
    names = list(contrasts)
    groups = contrasts[names[0]][1]
    if any(contrasts[n][1] != groups for n in names):
        raise ValueError("all contrasts must share the bootstrap groups (paired documents)")
    vals = torch.stack([contrasts[n][0] for n in names]).double()  # [K, N]
    bad = [n for k, n in enumerate(names) if not bool(torch.isfinite(vals[k]).all())]
    if bad:  # backstop: analyze() excludes non-finite readouts as technical. One NaN would poison every interval
        raise ValueError(f"non-finite per-document contrasts: {bad}")
    members: dict = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    members = list(members.values())
    gen = torch.Generator().manual_seed(seed)
    est = vals.mean(1)
    boots = []
    for _ in range(n_boot):
        draw = torch.randint(0, len(members), (len(members),), generator=gen)
        idx = torch.tensor([i for j in draw.tolist() for i in members[j]])
        boots.append(vals[:, idx].mean(1))
    B = torch.stack(boots)  # [n_boot, K]
    se = sd(B)
    tmax = ((B - est) / se.clamp_min(1e-12)).abs().max(1).values
    crit = float(torch.quantile(tmax, level))
    return {n: {"estimate": float(est[k]), "se": float(se[k]), "lo": float(est[k] - crit * se[k]),
                "hi": float(est[k] + crit * se[k])} for k, n in enumerate(names)} | {"_critical_value": crit,
                                                                                    "_n_groups": len(members)}


def readings(protocol: dict, info: dict, dev_info: dict | None = None) -> dict[str, dict]:
    """Every edit of a cell with the guard rule applied: usable, target flag and, on confirm, the guard flag and the
    dev values it depends on."""
    out = {}
    for name, e in info["edits"].items():
        if e["status"] != "matched":
            out[name] = e
            continue
        guards_ok = all(v <= protocol["guard_max"] for v in e["guards"].values())
        tau = float(name.split("@")[1])
        r = e | {"target_test_flag": abs(e["target_test_delta"] - tau) > protocol["target_deviation_flag"]}
        if dev_info is None:
            out[name] = r | {"usable": guards_ok}
            continue
        dev = dev_info["edits"][name]
        dev_usable = all(v <= protocol["guard_max"] for v in dev["guards"].values())
        out[name] = r | {"usable": dev_usable and guards_ok, "guard_flag": not guards_ok, "dev_usable": dev_usable,
                         "dev_target_test_delta": dev["target_test_delta"]}
    return out


def delta(cell: Cell, name: str) -> torch.Tensor:
    """Per-document protected CE rise of a matched edit (float64)."""
    return cell.ce_doc[name] - cell.base_ce


def _unreadable(edit: dict, cell: Cell, name: str) -> str | None:
    """Why a matched edit's readout cannot be used as numbers. None when every value is there and finite."""
    if not all(math.isfinite(v) for v in [*edit["guards"].values(), edit["target_test_delta"]]):
        return "non-finite guard or target-test readout"
    key = cell.info["protected_key"]
    if name not in cell.ce_doc:
        return f"no per-document readout for {key}"
    doc = delta(cell, name)
    n_bad = int((~torch.isfinite(doc)).sum())
    return f"non-finite readout ({n_bad} of {len(doc)} documents of {key})" if n_bad else None


def edit_state(edit: dict | None, cell: Cell, name: str) -> tuple[str, str]:
    """(description, kind) of one edit in one draw. Kind is "ok", "declared" or "technical"."""
    if edit is None:
        return "absent from the readout", "technical"
    if edit["status"] != "matched":
        if edit["technical_failure"]:
            return f"no edit ({edit['status']})", "technical"
        return f"not matched ({edit['status']})", "declared"
    why = _unreadable(edit, cell, name)
    if why:
        return why, "technical"
    return ("ok", "ok") if edit["usable"] else ("unusable (guard)", "declared")


def analyze(study: Study, split: str) -> dict:
    """Contrast = protected CE rise of the main method minus that of the comparator, per document of the protected
    Europarl test set, averaged over the draws in which both edits are matched and usable. One simultaneous family
    (claim): 3 targets x 2 effect points x 4 comparators. The reference rows form a second, separately reported
    family with its own simultaneous correction. The historical atom is a reference row for de only.

    Composition is explicit: the expected cells (target x draw) are the protocol's. Each cell is "read" or "missing
    readout". Each contrast lists its used draws, every excluded draw with its reason, and a status: "complete"
    (every expected draw used), "conditional" (every expected draw read, some excluded by the declared rules: not
    matched, unusable, or the method's rule gave no edit), "incomplete" (a readout is missing, an edit is absent
    from it, a method has no edit for a technical reason, or a matched edit's per-document readout is missing or
    holds a non-finite number) or "empty" (no draw usable, declared reasons only, the contrast is then absent from
    its family). The top-level status is "complete" if every contrast is complete, "conditional" if none is
    incomplete and no cell is missing, else "incomplete". The exported data carry no job log, so a cell whose read
    failed in the original run is "missing readout" here too."""
    protocol, cells = study.protocol, study.cells[split]
    main = protocol["main_method"]
    exp = expected_cells(protocol)
    cell_status = {f"{t}:{d}": "read" if (t, d) in cells else "missing readout" for t, d in exp}
    reads = {c: readings(protocol, cell.info, study.cells["dev"][c].info if split == "confirm" else None)
             for c, cell in cells.items()}
    fam = {"claim": {}, "references": {}}
    composition = {"claim": {}, "references": {}}
    targets = [t for t in protocol["targets"] if any(tt == t for tt, _ in exp)]
    for t in targets:
        p = protocol["targets"][t]
        draws = [d for tt, d in exp if tt == t]
        for tau in protocol["taus"]:
            for family, others in FAMILIES:
                for o in protocol[others]:
                    if o == HISTORICAL and t not in protocol["historical_atom_targets"]:
                        continue
                    name = f"{t}->{p}|tau={tau}|{main}-{o}"
                    diffs, grp, used, excluded, technical = None, None, [], {}, False
                    for d in draws:
                        if (t, d) not in cells:
                            excluded[d] = cell_status[f"{t}:{d}"]
                            technical = True
                            continue
                        cell = cells[(t, d)]
                        na, nb = f"{main}@{tau}", f"{o}@{tau}"
                        (sa, ka), (sb, kb) = (edit_state(reads[(t, d)].get(n), cell, n) for n in (na, nb))
                        if ka != "ok" or kb != "ok":
                            excluded[d] = f"{main}: {sa}; {o}: {sb}"
                            technical = technical or "technical" in (ka, kb)
                            continue
                        diff = delta(cell, na).float() - delta(cell, nb).float()
                        diffs = diff if diffs is None else diffs + diff
                        if grp is not None and cell.info["groups"] != grp:
                            raise ValueError(f"{name}: draws disagree on the bootstrap groups")
                        grp = cell.info["groups"]
                        used.append(d)
                    if technical:
                        status = "incomplete"
                    elif not used:
                        status = "empty"
                    elif excluded:
                        status = "conditional"
                    else:
                        status = "complete"
                    composition[family][name] = {"expected_draws": draws, "used_draws": used,
                                                 "excluded": {str(k): v for k, v in excluded.items()}, "status": status}
                    if used:
                        fam[family][name] = (diffs / len(used), grp)
    missing = [c for c, v in cell_status.items() if v != "read"]
    statuses = [v["status"] for f in composition.values() for v in f.values()]
    if missing or "incomplete" in statuses:
        overall = "incomplete"
    elif all(s == "complete" for s in statuses):
        overall = "complete"
    else:
        overall = "conditional"
    out = {"rule": analyze.__doc__, "split": split, "method_labels": protocol["methods"], "status": overall,
           "expected_cells": protocol["expected_cells"], "cells": cell_status, "missing_cells": missing,
           "composition": composition,
           "incomplete_contrasts": sorted(k for f in composition.values() for k, v in f.items()
                                          if v["status"] == "incomplete"),
           "freeze": protocol["freeze"]}
    for family, cs in fam.items():
        out[family] = simultaneous(cs) if cs else {}
        for name in cs:  # an interval is read together with its composition status, never alone
            out[family][name]["status"] = composition[family][name]["status"]
        out[f"{family}_contrasts_absent"] = sorted(k for k, v in composition[family].items() if k not in cs)
    return out
