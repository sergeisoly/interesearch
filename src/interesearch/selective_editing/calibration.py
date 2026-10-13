"""Effect-matched calibration: the scan-and-bisect rule, and its replay on the recorded calibration points.

Every edit is a one-parameter family (an atom's signed multiple, a learned update's scale, the ablated fraction of
the top-k neurons). Its dose is set so that the mean target validation CE rises by tau: the whole grid is
evaluated, the first crossing of tau is bisected at most 5 times, and the dose is matched when the closest point
inside that bracket is within 0.02 nats of tau. Neurons first take the smallest k of the ladder whose full
ablation reaches tau, then scan the ablated fraction at that k. A non-finite effect anywhere on the evaluated points
makes the calibration a technical failure, never "not reached".

The trace replays every recorded decision from the recorded (dose, effect) points: the rule must ask for exactly
the recorded doses (an unrecorded dose is a KeyError), scan exactly when the record has a scan (a neuron decision
taken on the ladder alone has none) and give the recorded status, dose, achieved effect, k and technical failure.
It checks the decision rule only, not the effects themselves: the model evaluations are not replayed here.
"""
# Rules: papers/selective-editing/index.qmd#sec-calibration (kept out of the docstring above, which the trace records).

from __future__ import annotations

import math
from typing import Callable, Sequence

from .data import Study, expected_cells

MAX_BISECT = 5
TOL = 0.02  # nats
NOT_REACHED = "not reached on the evaluated grid"
TECHNICAL = "technical: non-finite effect in calibration"
DECISION = ("status", "scale", "achieved", "best_reached")


def scan_and_match(delta_fn: Callable[[float], float], target: float, grid: Sequence[float],
                   max_bisect: int = MAX_BISECT, tol: float = TOL, mono_tol: float = TOL) -> dict:
    """Evaluate ``delta_fn`` on all of ``grid``, pick the first crossing of ``target`` and bisect inside that
    bracket at most ``max_bisect`` times. ``delta(0) = 0`` by definition.

    Status: ``matched`` (within ``tol``), ``bracket found, tolerance not met`` (closest point inside the first
    bracket, never eligible for a matched comparison), or ``not reached on the evaluated grid`` (scale None, the
    best value reached is stored).
    """
    sgn = 1.0 if target >= 0 else -1.0
    evals: list[tuple[float, float]] = [(0.0, 0.0)]
    first = None
    for a in grid:
        evals.append((float(a), float(delta_fn(float(a)))))
        if first is None and sgn * (evals[-1][1] - target) >= 0:
            first = len(evals) - 1
    diffs = [sgn * (evals[i + 1][1] - evals[i][1]) for i in range(len(evals) - 1)]
    above = [sgn * (d - target) >= 0 for _, d in evals]
    out = {"target": target, "grid_evaluations": evals,
           "n_crossings": sum(above[i] != above[i + 1] for i in range(len(above) - 1)),
           "monotone": all(x >= -mono_tol for x in diffs), "max_drop": -min(diffs) if diffs else 0.0,
           "monotone_before_match": None, "bisections": []}
    if first is None:
        best = max(evals[1:] or evals, key=lambda e: sgn * e[1])  # scale 0 (the base) excluded
        return out | {"scale": None, "achieved": None, "best_reached": best[1], "best_scale": best[0],
                      "status": NOT_REACHED}
    out["monotone_before_match"] = all(x >= -mono_tol for x in diffs[:first])
    (a0, d0), (a1, d1) = evals[first - 1], evals[first]
    inside = [(a0, d0), (a1, d1)] if a0 > 0 else [(a1, d1)]
    if abs(d1 - target) > tol and not (a0 > 0 and abs(d0 - target) <= tol):
        for _ in range(max_bisect):
            am = 0.5 * (a0 + a1)
            dm = float(delta_fn(am))
            out["bisections"].append((am, dm))
            inside.append((am, dm))
            if abs(dm - target) <= tol:
                break
            if sgn * (dm - target) >= 0:
                a1, d1 = am, dm
            else:
                a0, d0 = am, dm
    best = min(inside, key=lambda e: (abs(e[1] - target), e[0]))
    return out | {"scale": best[0], "achieved": best[1],
                  "status": "matched" if abs(best[1] - target) <= tol else "bracket found, tolerance not met"}


def decide(delta_fn: Callable[[float], float], tau: float, grid: Sequence[float]) -> tuple[dict, dict]:
    """(scan, decision) of one family at one effect point."""
    mt = scan_and_match(delta_fn, tau, grid)
    decision = {k: mt[k] for k in DECISION if k in mt}
    if not all(math.isfinite(d) for _, d in mt["grid_evaluations"] + mt["bisections"]):
        decision |= {"status": TECHNICAL, "technical_failure": True, "scale": None, "achieved": None}
    return mt, decision


def decide_neurons(ladder: Sequence[tuple[int, float]], fraction_fn: Callable[[int], Callable[[float], float]],
                   tau: float, grid: Sequence[float]) -> tuple[dict | None, dict]:
    """The smallest k of the ladder (full ablation) that reaches tau, then the ablated fraction at that k."""
    if not all(math.isfinite(d) for _, d in ladder):
        return None, {"k": None, "status": TECHNICAL, "technical_failure": True, "scale": None, "achieved": None}
    k = next((k for k, d in ladder if d >= tau), None)
    if k is None:
        return None, {"status": NOT_REACHED, "scale": None, "k": None, "best_reached": max(d for _, d in ladder)}
    mt, decision = decide(fraction_fn(k), tau, grid)
    return mt, {"k": k} | decision


def _recorded(scan: dict) -> Callable[[float], float]:
    table = dict(map(tuple, scan["grid_evaluations"][1:] + scan["bisections"]))
    return lambda a: table[a]


def replay_trace(study: Study) -> dict:
    """Replay every recorded calibration decision of the dev cells present."""
    cal = study.protocol["calibration"]
    rows = []
    for t, d in expected_cells(study.protocol):
        if (t, d) not in study.calibration:  # a dev-only folder without this cell's calibration
            continue
        for method, rec in study.calibration[(t, d)]["methods"].items():
            grid = cal[cal["grid_of_method"][method]]
            for tau, row in rec["by_tau"].items():
                if method == "neurons":
                    mt, got = decide_neurons(rec["ladder"], lambda k, row=row: _recorded(row["scan"]), float(tau), grid)
                else:
                    mt, got = decide(_recorded(row["scan"]), float(tau), grid)
                scan = mt and {k: [list(p) for p in mt[k]] for k in ("grid_evaluations", "bisections")}
                same_points = scan == row["scan"]
                rows.append({"cell": f"{t}:{d}", "method": method, "tau": tau, "recorded": row["recorded"],
                             "replayed": got, "same_points": same_points,
                             "agrees": same_points and got == row["recorded"]})
    return {"rule": __doc__, "n_decisions": len(rows), "n_agree": sum(r["agrees"] for r in rows),
            "statuses": {s: sum(r["replayed"]["status"] == s for r in rows)
                         for s in sorted({r["replayed"]["status"] for r in rows})},
            "decisions": rows}
