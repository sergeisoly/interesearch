"""Exploratory additional comparisons for the selective-editing study, replayed from its exported dev readouts.
Rules: papers/selective-editing/index.qmd#sec-extended-comparisons.

The additional comparisons measured the main study's VPD atom edits again next to stronger comparators on the dev dates: atoms of
an SVD dictionary and a dictionary-free whitened rank-one direction, both chosen by the main selector's rule, and the
two learned updates trained for 2000 steps with three seeds. Its analysis differs from the main one: each side of a
contrast is the mean over its matched and usable runs (draws, and seeds for a learned update), per document and in
float64, not a mean of paired per-draw differences. All estimable contrasts form one simultaneous family. The
learned-update contrasts are repeated without unstable runs as a second family. Every interval gets two independent
attributes, its direction and whether it lies inside the band of +-0.02 nats. A contrast without a usable run on one
side has neither.

    uv run python -m interesearch.selective_editing.analysis.extended data/selective_editing --out results/selective_editing
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from interesearch.checksums import verify_checksums
from interesearch.selective_editing.analysis.statistics import bootstrap_parameters, simultaneous
from interesearch.selective_editing.data import load_protocol, expected_cells, load_study, write_json

CELL_FILES = ("cell.json", "protected.npz", "posthoc.npz")
REFERENCE_200 = ("lora_rank1", "finetune_24")  # the main study's learned updates (200 steps, one seed)


@dataclass(frozen=True)
class ExtendedComparisons:
    protocol: dict
    cells: dict[tuple[str, int], tuple[dict, dict[str, list[float]]]]  # info, protected per-document rise by edit
    calibration: dict[tuple[str, int], dict]
    training: dict[str, dict]  # run -> record


def load(root: str | Path, allow_reduced: bool = False) -> ExtendedComparisons:
    """The exported additional comparisons (data/selective_editing/extended_comparisons, or new run records), checked against its
    SHA256SUMS."""
    root = Path(root)
    protocol = load_protocol(root, allow_reduced)
    cells = expected_cells(protocol)
    files = ["protocol.json", "training/records.json", "training/lr_choice.json", "training/logs.npz"]
    files += [f"calibration/{t}_draw{d}.json" for t, d in cells]
    files += [f"selection/{t}_draw{d}{f}" for t, d in cells for f in (".json", "_atoms.npz")]
    files += [f"cells/{t}_draw{d}/{f}" for t, d in cells for f in CELL_FILES]
    verify_checksums(root, tuple(files))
    out = {}
    for t, d in cells:
        info = json.loads((root / f"cells/{t}_draw{d}/cell.json").read_text())
        with np.load(root / f"cells/{t}_draw{d}/protected.npz", allow_pickle=False) as z:
            rise = z["delta_ce_doc"]
        if rise.dtype != np.float64 or rise.shape != (len(info["measured"]), len(info["groups"])):
            raise ValueError(f"{t}_draw{d}: protected.npz is {rise.dtype} {rise.shape}")
        out[(t, d)] = (info, {m: rise[i].tolist() for i, m in enumerate(info["measured"])})
    records = json.loads((root / "training/records.json").read_text())["runs"]
    return ExtendedComparisons(protocol, out,
                    {(t, d): json.loads((root / f"calibration/{t}_draw{d}.json").read_text()) for t, d in cells},
                    {r["run"]: r for r in records})


def method_of(run: str) -> str:
    """'lora_rank1_long_s2' -> 'lora_rank1_long'. An atom method is its own run."""
    return re.sub(r"_s\d+$", "", run)


def docs(edit: dict, rise: list[float] | None) -> list[float] | None:
    """Per-document protected rise of a matched, usable edit with finite values, else None."""
    if edit["status"] != "matched" or not edit["usable"] or rise is None:
        return None
    return rise if all(math.isfinite(x) for x in rise) else None


def contrast(main: list[list[float]], other: list[list[float]], groups: list) -> tuple[torch.Tensor, list]:
    """Per document: mean over the main runs minus mean over the comparator runs, float64."""
    mv = torch.tensor(main, dtype=torch.float64).mean(0)
    ov = torch.tensor(other, dtype=torch.float64).mean(0)
    return mv - ov, groups


def _sd(B: torch.Tensor) -> torch.Tensor:
    """Bootstrap SE using ``torch.std`` from the original additional comparisons (n - 1). The main study uses Welford order
    (analysis._sd). The two differ in the last bit."""
    return B.std(0)


def label(ci: dict, band: float) -> dict:
    """Two independent attributes of the interval of main - comparator: `direction` is below 0, above 0 or includes
    0. `within_band` is true when the whole interval lies in [-band, band], whether or not it includes 0."""
    lo, hi = ci["lo"], ci["hi"]
    return {"direction": "below 0" if hi < 0 else "above 0" if lo > 0 else "includes 0",
            "within_band": bool(-band <= lo and hi <= band)}


def unstable_runs(add: ExtendedComparisons, t: str, d: int) -> set[str]:
    """Learned runs of a cell whose English validation KL ended above twice its minimum (training records)."""
    out = set()
    for run, rec in add.calibration[(t, d)]["methods"].items():
        upd = rec.get("learned_update")
        if upd and add.training[upd["training_run"]]["convergence"]["unstable"]:
            out.add(run)
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def reference_200(study, protocol: dict) -> dict:
    """Mean protected rise of the main study's 200-step learned updates on the same dev documents (descriptive):
    every matched, usable draw, pooled over documents."""
    out = {}
    cells = expected_cells(protocol)
    for t in dict.fromkeys(t for t, _ in cells):
        for tau in protocol["taus"]:
            for m in REFERENCE_200:
                xs = []
                for d in [d for tt, d in cells if tt == t]:
                    cell = study.cells["dev"].get((t, d))
                    e = None if cell is None else cell.info["edits"].get(f"{m}@{tau}")
                    if e is None or e["status"] != "matched" or m + f"@{tau}" not in cell.ce_doc:
                        continue
                    if not all(v <= study.protocol["guard_max"] for v in e["guards"].values()):
                        continue
                    x = (cell.ce_doc[f"{m}@{tau}"] - cell.base_ce).tolist()
                    if all(math.isfinite(v) for v in x):
                        xs.append(x)
                out[f"{t}|tau={tau}|{m}"] = sum(map(sum, xs)) / sum(map(len, xs)) if xs else None
    return out


def analyze(add: ExtendedComparisons, study) -> dict:
    """Contrasts of the main edit against each comparator on protected Europarl test, each side averaged over its
    matched and usable runs (float64, per document), in one simultaneous family. A second family excludes
    unstable learned-update runs. Also reports descriptive rows per method, the main study's 200-step updates and the
    convergence of every training run."""
    pr = add.protocol
    bootstrap = bootstrap_parameters(pr)
    main, comparators, learned = pr["main_method"], pr["comparators"], set(pr["learned_methods"])
    contrasts, stable_contrasts, composition, rows = {}, {}, {}, {}
    cells = expected_cells(pr)
    for t in dict.fromkeys(t for t, _ in cells):
        p = pr["targets"][t]
        reads = {d: add.cells[(t, d)] for tt, d in cells if tt == t}
        groups = next(iter(reads.values()))[0]["groups"]
        if any(info["groups"] != groups for info, _ in reads.values()):
            raise ValueError(f"{t}: draws differ in their protected documents")
        unstable = {d: unstable_runs(add, t, d) for d in reads}
        for tau in pr["taus"]:
            runs: dict[str, list] = {}
            for d, (info, rise) in reads.items():
                for name, e in info["edits"].items():
                    run, at = name.rsplit("@", 1)
                    if float(at) == tau:
                        runs.setdefault(method_of(run), []).append(
                            (d, run, docs(e, rise.get(name)), e, run in unstable[d]))
            for m, lst in runs.items():
                used = [x for _, _, x, _, _ in lst if x is not None]
                e_used = [e for _, _, x, e, _ in lst if x is not None]
                e_stable = [e for _, _, x, e, u in lst if x is not None and not u]
                rows[f"{t}|tau={tau}|{m}"] = {
                    "runs": len(lst), "used": len(used), "unstable": [f"draw {d} {r}" for d, r, _, _, u in lst if u],
                    "excluded": [f"draw {d} {r}: {e['status']}{'' if e.get('usable', True) else ' (guard)'}"
                                 for d, r, x, e, _ in lst if x is None],
                    "protected_mean": sum(map(sum, used)) / sum(map(len, used)) if used else None,
                    "target_test_mean": _mean([e["target_test_delta"] for e in e_used]),
                    "target_test_mean_stable": _mean([e["target_test_delta"] for e in e_stable]),
                    "guard_max": max((max(e["guards"].values()) for e in e_used), default=None),
                    "per_run_protected": [sum(x) / len(x) for x in used]}
            mains = [x for _, _, x, _, _ in runs.get(main, []) if x is not None]
            for m in comparators:
                lst = runs.get(m, [])
                other = [x for _, _, x, _, _ in lst if x is not None]
                stable = [x for _, _, x, _, u in lst if x is not None and not u]
                name = f"{t}->{p}|tau={tau}|{main}-{m}"
                composition[name] = {"main_runs": len(mains), "comparator_runs": len(other),
                                     "comparator_runs_stable": len(stable)}
                if mains and other:
                    contrasts[name] = contrast(mains, other, groups)
                if m in learned and mains and stable:
                    stable_contrasts[name] = contrast(mains, stable, groups)
    fam = simultaneous(contrasts, sd=_sd, **bootstrap) if contrasts else {}
    fam_stable = simultaneous(stable_contrasts, sd=_sd, **bootstrap) if stable_contrasts else {}
    band = pr["band"]
    labels = {n: label(fam[n], band) for n in contrasts}
    labels_stable = {n: label(fam_stable[n], band) if n in stable_contrasts else "absent (no stable run)"
                     for n in contrasts if n.rsplit("-", 1)[1] in learned}
    stable_diff = {n: (["direction", "within_band"] if isinstance(s, str)
                       else [k for k in ("direction", "within_band") if s[k] != labels[n][k]])
                   for n, s in labels_stable.items()}
    conv = {run: rec["convergence"] for run, rec in add.training.items()}
    return {"rule": analyze.__doc__, "label_rule": label.__doc__,
            "expected_cells": pr["expected_cells"], "band": band,
            "all_runs": fam, "labels": labels,
            "stable_runs": fam_stable, "labels_stable_runs": labels_stable,
            "stable_changes_a_label": {n: x for n, x in sorted(stable_diff.items()) if x},
            "composition": composition, "contrasts_absent": sorted(k for k in composition if k not in contrasts),
            "methods": rows, "main_study_200_step": reference_200(study, pr), "convergence": conv,
            "plateau_count": f"{sum(1 for v in conv.values() if v.get('plateau'))}/{len(conv)}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay exploratory additional comparisons from their exported readouts.")
    ap.add_argument("data_dir", type=Path, help="data/selective_editing")
    ap.add_argument("--out", type=Path, required=True, help="results directory")
    a = ap.parse_args(argv)
    res = analyze(load(a.data_dir / "extended_comparisons"), load_study(a.data_dir))
    write_json(a.out / "extended_comparisons_analysis.json", res)
    n = len(res["labels"])
    print(f"additional comparisons: {n} contrasts in all_runs, {len(res['labels_stable_runs'])} in stable_runs, "
          f"{len(res['contrasts_absent'])} without an estimate, plateau {res['plateau_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
