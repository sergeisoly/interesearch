"""Descriptive checks from the confirmation windows and the earlier German shortlist scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interesearch.selective_editing.analysis.posthoc import per_date
from interesearch.selective_editing.data import Study, load_study


def window_effects(study: Study) -> list[dict]:
    """Summarize per-window CE changes once per fixed selected edit."""
    main = study.protocol["main_method"]
    rows = []
    for target, protected in study.protocol["targets"].items():
        cells = [cell for (language, _), cell in study.cells["confirm"].items() if language == target]
        dates = cells[0].info["groups"]
        if len(set(dates)) != len(dates) or any(cell.info["groups"] != dates for cell in cells[1:]):
            raise ValueError(f"{target}: draws must share the same unique dates")
        for tau in study.protocol["taus"]:
            edit = f"{main}@{tau}"
            changes = [per_date(cell, edit)[:2] for cell in cells]
            protected_ce, target_ce = changes[0]
            if any(not np.array_equal(values, expected)
                   for draw in changes[1:] for values, expected in zip(draw, changes[0])):
                raise ValueError(f"{target} at tau={tau}: per-date CE changes differ across draws")
            rows.append({"target": target, "protected_language": protected, "tau": tau, "n_dates": len(dates),
                         "target_ce_change": summarize(target_ce), "protected_ce_change": summarize(protected_ce)})
    return rows


def summarize(changes: np.ndarray) -> dict:
    """Share of the sum of positive changes contributed by the six largest changes."""
    if not np.isfinite(changes).all():
        raise ValueError("window CE changes must be finite")
    positive = np.sort(changes[changes > 0])
    return {
        "mean": float(changes.mean()),
        "max": float(changes.max()),
        "positive_windows": int(positive.size),
        "largest_six_share_of_positive": float(positive[-6:].sum() / positive.sum()) if positive.size else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="data/selective_editing")
    args = parser.parse_args(argv)
    windows = window_effects(load_study(args.data_dir))
    german = json.loads((args.data_dir / "supplementary" / "german_shortlist.json").read_text())
    print(json.dumps({"confirmation_windows": windows, "german_shortlist": german}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
