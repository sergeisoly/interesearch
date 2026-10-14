"""Confirmatory reading of the selective-editing claim family.

The confirm split is read once with the dev edits unchanged (same doses, same k). The frozen analysis runs on it
as on dev. Each claim contrast present on dev gets a verdict by the reading table fixed before the confirm set was
opened (papers/selective-editing/index.qmd#sec-contrasts), in this order:
    not established          no confirm interval, or the contrast is incomplete (technical)
    effect not maintained    a used draw has a confirm target-test rise more than 0.1 nats from tau
    not established          a non-finite interval or a zero dev estimate
    reproduced on confirm    the simultaneous interval excludes 0 with the dev sign
    not reproduced           it excludes 0 with the opposite sign
    not established          it includes 0
The reference family is reported without verdicts. Achieved effects and English guards are summarized
descriptively (means over draws of matched, usable edits), next to the target and guard flags of every edit.

    uv run python -m interesearch.selective_editing.analysis.confirm data/selective_editing --out results/selective_editing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from interesearch.selective_editing.analysis.statistics import analyze, readings
from interesearch.selective_editing.method.calibration import replay_trace
from interesearch.selective_editing.data import load_study

VERDICTS = ("reproduced on confirm", "not reproduced", "not established", "effect not maintained")
GUARD = "confirm_en"


def verdict(iv: dict | None, status: str | None, dev_est: float, effect_flag: bool,
            target_deviation_flag: float = 0.1) -> tuple[str, str]:
    if iv is None:
        return "not established", f"no interval on confirm (status {status})"
    if status == "incomplete":
        return "not established", "status incomplete (technical)"
    if effect_flag:
        return ("effect not maintained",
                f"a method's confirm target rise is more than {target_deviation_flag} nats from tau in a used draw")
    lo, hi = iv["lo"], iv["hi"]
    dev_sign = (dev_est > 0) - (dev_est < 0)
    if not (math.isfinite(lo) and math.isfinite(hi)) or dev_sign == 0:
        return "not established", "non-finite interval or zero dev estimate"
    if (lo > 0 and dev_sign > 0) or (hi < 0 and dev_sign < 0):
        return "reproduced on confirm", "simultaneous CI excludes 0 with the dev sign"
    if (lo > 0 and dev_sign < 0) or (hi < 0 and dev_sign > 0):
        return "not reproduced", "simultaneous CI excludes 0 with the opposite sign"
    return "not established", "simultaneous CI includes 0"


def _parse(name: str) -> tuple[str, str, str, str, str]:
    """"de->nl|tau=0.5|vpd_atom-null_atom" -> (target, protected, tau, main method, comparator)."""
    pair, tau, methods = name.split("|")
    t, p = pair.split("->")
    main, other = methods.split("-", 1)
    return t, p, tau.removeprefix("tau="), main, other


def reading_table(a: dict, dev_a: dict, reads: dict, target_deviation_flag: float = 0.1) -> list[dict]:
    rows = []
    for name, dv in dev_a["claim"].items():
        if name.startswith("_"):
            continue
        t, _, tau, main, other = _parse(name)
        comp = a["composition"]["claim"].get(name, {})
        iv = a["claim"].get(name)
        used = comp.get("used_draws", [])
        effects, flag = {}, False
        for d in used:
            for m in (main, other):
                e = reads[(t, d)][f"{m}@{tau}"]
                effects[f"{m}@d{d}"] = e["target_test_delta"]
                flag = flag or e["target_test_flag"]
        v, why = verdict(iv, comp.get("status"), dv["estimate"], flag, target_deviation_flag)
        rows.append({"contrast": name, "verdict": v, "why": why, "status": comp.get("status"),
                     "confirm": None if iv is None else {k: iv[k] for k in ("estimate", "se", "lo", "hi")},
                     "dev": {k: dv[k] for k in ("estimate", "lo", "hi")}, "tau": float(tau),
                     "used_draws": used, "excluded": comp.get("excluded"), "confirm_target_delta": effects,
                     "effect_flag": flag})
    return rows


def descriptive(protocol: dict, reads: dict) -> list[dict]:
    """Mean over draws of matched, usable edits: confirm target, protected and English CE rise."""
    rows = []
    for t in protocol["targets"]:
        rs = [reads[k] for k in sorted(reads) if k[0] == t]
        for tau in map(str, protocol["taus"]):
            for m in protocol["methods"]:
                es = [e for e in (r.get(f"{m}@{tau}") for r in rs) if e and e["status"] == "matched" and e["usable"]]
                if not es:
                    continue
                mean = lambda f: sum(f(e) for e in es) / len(es)  # noqa: E731
                rows.append({"target": t, "tau": float(tau), "method": m, "n_draws": len(es),
                             "target_delta": mean(lambda e: e["target_test_delta"]),
                             "protected_delta": mean(lambda e: e["protected_delta"]),
                             "en_delta": mean(lambda e: e["guards"][GUARD]),
                             "max_abs_target_dev_from_tau": max(abs(e["target_test_delta"] - float(tau)) for e in es),
                             "dev_target_test_delta": mean(lambda e: e["dev_target_test_delta"])})
    return rows


def edit_flags(reads: dict) -> dict:
    out = {"n_matched": 0, "target_flags": [], "guard_flags": [], "max_abs_target_dev": 0.0, "max_en_delta": None}
    for (t, d), r in sorted(reads.items()):
        for name, e in r.items():
            if e["status"] != "matched":
                continue
            out["n_matched"] += 1
            tau = float(name.split("@")[1])
            out["max_abs_target_dev"] = max(out["max_abs_target_dev"], abs(e["target_test_delta"] - tau))
            g = e["guards"][GUARD]
            out["max_en_delta"] = g if out["max_en_delta"] is None else max(out["max_en_delta"], g)
            if e["target_test_flag"]:
                out["target_flags"].append(f"{t}:{d} {name} {e['target_test_delta']:.3f}")
            if e["guard_flag"]:
                out["guard_flags"].append(f"{t}:{d} {name} {g:.3f}")
    return out


def dumps(obj) -> str:
    return json.dumps(obj, indent=1) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def run(data_dir: str | Path, allow_reduced: bool = False) -> dict[str, str]:
    """Every result file as text: the dev and confirm analyses, the confirm result and the calibration trace.
    The confirm result names both analyses by the sha256 of these bytes. The historical fingerprints of the
    original analyses and freeze are carried separately. ``allow_reduced`` admits the records of a reduced run
    (code-path checks only)."""
    study = load_study(data_dir, allow_reduced)
    dev_a, a = analyze(study, "dev"), analyze(study, "confirm")
    dev_text, text = dumps(dev_a), dumps(a)
    reads = {c: readings(study.protocol, cell.info, study.cells["dev"][c].info)
             for c, cell in study.cells["confirm"].items()}
    table = reading_table(a, dev_a, reads, study.protocol["target_deviation_flag"])
    src = study.protocol["source"]
    res = {"label": "selective editing: confirmatory readout", "analysis_status": a["status"],
           "analysis_sha256": sha256_text(text), "cells": a["cells"], "missing_cells": a["missing_cells"],
           "incomplete_contrasts": a["incomplete_contrasts"], "claim_critical": a["claim"].get("_critical_value"),
           "n_groups": a["claim"].get("_n_groups"), "reading_table": table,
           "verdict_counts": {v: sum(r["verdict"] == v for r in table) for v in VERDICTS},
           "claim_absent": a["claim_contrasts_absent"],
           "references": {k: v for k, v in a["references"].items() if not k.startswith("_")},
           "references_critical": a["references"].get("_critical_value"),
           "descriptive": descriptive(study.protocol, reads), "edit_flags": edit_flags(reads),
           "dev_analysis_sha256": sha256_text(dev_text),
           "historical_analysis_sha256": src["confirm_analysis_sha256"],
           "historical_dev_analysis_sha256": src["dev_analysis_sha256"], "freeze_sha256": src["freeze_sha256"]}
    return {"dev_analysis.json": dev_text, "confirm_analysis.json": text, "confirm_result.json": dumps(res),
            "calibration_trace.json": dumps(replay_trace(study))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay the selective-editing analyses from the exported readouts.")
    ap.add_argument("data_dir", type=Path, help="data/selective_editing")
    ap.add_argument("--out", type=Path, required=True, help="results directory")
    a = ap.parse_args(argv)
    t0 = time.time()
    files = run(a.data_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (a.out / name).write_text(text)
    res, trace = json.loads(files["confirm_result.json"]), json.loads(files["calibration_trace.json"])
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} ({time.time() - t0:.1f} s): "
          f"confirm {res['analysis_status']}, verdicts {res['verdict_counts']}, "
          f"calibration {trace['n_agree']}/{trace['n_decisions']} decisions replayed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
