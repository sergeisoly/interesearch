"""Post-hoc analyses of the selective-editing readouts. They are not part of the frozen protocol: each was chosen
after the dev and confirm results were known, and none changes a frozen status or verdict.

dose agreement  Did vpd_atom and each claim comparator reach comparable target effects? Per split, one joint family
                of 54 coordinates: for each of the 18 assessable claim contrasts (3 targets x 2 tau x lora_rank1,
                finetune_24, neurons), vpd_atom minus comparator in protected CE rise, target CE rise ("target gap")
                and protected KL, each a per-date value averaged over the three draws. Session dates are resampled
                with replacement, jointly for all 54 coordinates (NumPy, seed 20261011, 20,000 resamples). SE is the
                SD of the resampled means and one max-|t| critical value gives simultaneous 95% intervals. A target
                gap whose whole interval lies inside +-0.1 nats counts as agreement. This band is a post-hoc criterion
                for the gap between two methods. The frozen 0.1-nat flag concerns one edit's target rise against tau.
english KL      How often a learned baseline's English KL lies below vpd_atom's: every target, draw, tau and English
                set, per split (dev: English validation and the three guards, confirm: confirm_en).
edits           The scalars of every matched edit, for the tables: dose, the atom and its mask coefficient 1 + alpha
                (alpha = sign x dose) or the learning rate of a learned update, edit norm, achieved target and
                protected CE rise (means over dates of the per-date rises, equal to the readout's recorded rises),
                English guards and KL, usability.

    uv run python -m interesearch.selective_editing.posthoc data/selective_editing --out results/selective_editing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import readings
from .data import Cell, Study, expected_cells, load_study

SEED, N_BOOT, LEVEL = 20261011, 20_000, 0.95
BAND = 0.1
COMPARATORS = ("lora_rank1", "finetune_24", "neurons")  # the claim comparators with edits (null_atom has none)
LEARNED = ("lora_rank1", "finetune_24")
METRICS = ("protected_ce", "target_gap", "protected_kl")


def joint_intervals(matrix: np.ndarray, names: list[str], seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Simultaneous intervals for the row means of ``matrix`` (coordinates x dates): the dates are resampled with
    replacement, the same resample for every row. SE = SD of the resampled means (denominator n_boot - 1). The
    critical value is the LEVEL quantile of the largest |resampled mean - mean| / SE over rows."""
    ids = np.random.default_rng(seed).integers(matrix.shape[1], size=(n_boot, matrix.shape[1]))
    means = np.stack([v[ids].mean(axis=1) for v in matrix], axis=1)
    point = matrix.mean(axis=1)
    se = means.std(axis=0, ddof=1)
    critical = np.quantile(np.abs((means - point) / np.maximum(se, 1e-12)).max(axis=1), LEVEL)
    return {"seed": seed, "n_boot": n_boot, "critical": float(critical),
            "rows": {n: {"mean": float(point[i]), "se": float(se[i]), "lo": float(point[i] - critical * se[i]),
                         "hi": float(point[i] + critical * se[i])} for i, n in enumerate(names)}}


def _draws(study: Study, t: str) -> list[int]:
    return [d for tt, d in expected_cells(study.protocol) if tt == t]


def _reads(study: Study, split: str) -> dict:
    dev = study.cells["dev"]
    return {c: readings(study.protocol, cell.info, dev[c].info if split == "confirm" else None)
            for c, cell in study.cells[split].items()}


def per_date(cell: Cell, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Protected CE rise, target CE rise and protected KL of one matched edit, per date (float64)."""
    i, ph = cell.info["measured"].index(name), cell.posthoc
    return (cell.ce_doc[name].numpy() - cell.base_ce.numpy(), ph["target_ce_doc"][i] - ph["target_base_ce"],
            ph["protected_kl_doc"][i])


def dose_agreement(study: Study, split: str) -> dict:
    main = study.protocol["main_method"]
    reads = _reads(study, split)
    dates = study.cells[split][expected_cells(study.protocol)[0]].info["groups"]
    if len(set(dates)) != len(dates):
        raise ValueError(f"{split}: repeated dates")
    rows, names = [], []
    for t, p in study.protocol["targets"].items():
        for tau in map(str, study.protocol["taus"]):
            mean = {}
            for m in (main, *COMPARATORS):
                draws = []
                for d in _draws(study, t):
                    cell, e = study.cells[split][(t, d)], reads[(t, d)][f"{m}@{tau}"]
                    if cell.info["groups"] != dates or not (e["status"] == "matched" and e["usable"]):
                        raise ValueError(f"{split} {t}:{d} {m}@{tau}: not a usable edit on the shared dates")
                    draws.append(per_date(cell, f"{m}@{tau}"))
                mean[m] = tuple(np.mean(x, axis=0) for x in zip(*draws))
            for other in COMPARATORS:
                for i, metric in enumerate(METRICS):
                    names.append(f"{t}->{p}|tau={tau}|{main}-{other}|{metric}")
                    rows.append(mean[main][i] - mean[other][i])
    out = joint_intervals(np.stack(rows), names)
    gaps = {n: r for n, r in out["rows"].items() if n.endswith("|target_gap")}
    outside = [n for n, r in gaps.items() if not (-BAND < r["lo"] and r["hi"] < BAND)]
    return {"n_dates": len(dates)} | out | {"summary": {
        "band": BAND, "target_gaps": len(gaps), "target_gap_within_band": len(gaps) - len(outside),
        "target_gap_outside_band": outside,
        **{f"{m}_below_zero": sum(r["hi"] < 0 for n, r in out["rows"].items() if n.endswith(f"|{m}"))
           for m in ("protected_ce", "protected_kl")}}}


def english_kl(study: Study, split: str) -> dict:
    main, out = study.protocol["main_method"], {}
    for other in LEARNED:
        ratios, lower = [], 0
        for (t, d), cell in study.cells[split].items():
            for tau in map(str, study.protocol["taus"]):
                a, b = (cell.info["edits"][f"{m}@{tau}"]["english_kl"] for m in (main, other))
                for s in a:
                    ratios.append(a[s] / b[s])
                    lower += b[s] < a[s]
        out[other] = {"n": len(ratios), "comparator_lower": lower, "ratio_min": min(ratios), "ratio_max": max(ratios)}
    return out


def edits(study: Study) -> list[dict]:
    rows = []
    for split, cells in study.cells.items():
        reads = _reads(study, split)
        for (t, d), cell in cells.items():
            cal = study.calibration[(t, d)]["methods"]
            for name in cell.info["measured"]:
                m, tau = name.split("@")
                e, rec = reads[(t, d)][name], cal[m]
                protected, target, _ = per_date(cell, name)
                atom, learned = rec.get("atom"), rec.get("learned_update")
                rows.append({
                    "split": split, "target": t, "draw": d, "method": m, "tau": float(tau), "scale": e["scale"],
                    "k": e["k"], "atom": None if atom is None else atom["label"],
                    "coefficient": None if atom is None else 1 + atom["sign"] * e["scale"],
                    "learning_rate": None if learned is None else learned["learning_rate"],
                    "edit_norm": rec["by_tau"][tau]["edit_norm"], "target_rise": float(target.mean()),
                    "protected_rise": float(protected.mean()),
                    "guards": e["guards"], "english_kl": e["english_kl"], "usable": e["usable"]})
    return rows


def run(data_dir: str | Path) -> dict:
    study = load_study(data_dir)
    return {"label": "selective editing: post-hoc analyses (not part of the frozen protocol)",
            "dose_agreement": {s: dose_agreement(study, s) for s in ("dev", "confirm")},
            "english_kl": {s: english_kl(study, s) for s in ("dev", "confirm")},
            "edits": edits(study)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Post-hoc analyses of the selective-editing readouts.")
    ap.add_argument("data_dir", type=Path, help="data/selective_editing")
    ap.add_argument("--out", type=Path, required=True, help="results directory")
    a = ap.parse_args(argv)
    res = run(a.data_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "posthoc.json").write_text(json.dumps(res, indent=1) + "\n")
    for s, r in res["dose_agreement"].items():
        print(f"{s}: {r['n_dates']} dates, critical {r['critical']:.3f}, target gaps within +-{BAND}: "
              f"{r['summary']['target_gap_within_band']}/{r['summary']['target_gaps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
