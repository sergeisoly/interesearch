"""Post hoc diagnostics on the saved Repair Transfer readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..method.selection import BEST
from .curve import cost_curve
from .recovery import load_replay, replay_target
from .stats import ci

REFERENCES = ("fresh@2K", "fresh_seed2@2K")
MARGINS_PP = (1, 2, 3)
FAMILIES = ("donor_init", "fresh")
PORTUGUESE_CELLS = (f"{BEST}@2K", "donor_init@2K", "fresh@2K")
COST_UNITS = ("fit_token_presentations", "unique_tokens")


def reach_sensitivity(reference: str, replay: dict, cell_draws: dict, donor: dict, recipe: dict) -> dict:
    """Vary the reach threshold on the same candidate grid and with the same readout constraints."""
    by_margin = {}
    for margin in MARGINS_PP:
        curve = cost_curve(
            replay["cells"], cell_draws, replay["cost"], replay["cross_block"], donor,
            recipe["curve"] | {"reference": reference},
            recipe["analysis"] | {"equivalence_margin": margin / 100}, recipe["bootstrap"],
        )
        if curve["status"] != "ok":
            return {"status": curve["status"]}
        by_margin[str(margin)] = {
            family: {unit: curve["reach"][family][unit] for unit in COST_UNITS} for family in FAMILIES
        }
    return {"status": "ok", "by_margin_pp": by_margin}


def portuguese_delta(cell: dict, readout: dict, draws: torch.Tensor, quantiles: list[float]) -> dict:
    """Mean CE change from the fine-tune and its paired interval over dates."""
    if cell["status"] != "ok":
        return {"status": cell["status"]}
    delta = readout[cell["name"]]["pt"]["ce"] - readout["finetuned"]["pt"]["ce"]
    if not bool(torch.isfinite(delta).all()):
        return {"status": "technical_invalid"}
    return {"status": "ok", "delta_ce": float(delta.mean()), "ci": list(ci(delta[draws].mean(1), quantiles))}


def analyze(data_dir: str | Path) -> dict:
    """Replay both diagnostics using the saved selections, costs and bootstrap settings."""
    blocks, recipe, donor = load_replay(data_dir)
    bootstrap = recipe["bootstrap"]
    targets = {}
    for target, (main, nested) in blocks.items():
        replay, cell_draws, draws = replay_target(target, main, nested, recipe)
        targets[target] = {
            "n_dates": replay["n_dates"],
            "reach": {ref: reach_sensitivity(ref, replay, cell_draws, donor, recipe) for ref in REFERENCES},
            "portuguese_ce": {
                cell: portuguese_delta(replay["cells"][cell], main.readout, draws, bootstrap["quantiles"])
                for cell in PORTUGUESE_CELLS
            },
        }
    return {
        "study": "repair_transfer",
        "analysis": "post hoc",
        "bootstrap": bootstrap,
        "reach_cost_units": list(COST_UNITS),
        "portuguese_ce_unit": "nats/token",
        "portuguese_ce_contrast": "repair minus fine-tuned model",
        "targets": targets,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay the post hoc reach and Portuguese CE checks on CPU.")
    parser.add_argument("data_dir", type=Path, help="data/repair_transfer")
    parser.add_argument("--out", type=Path, required=True, help="post hoc results JSON")
    args = parser.parse_args(argv)
    result = analyze(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
