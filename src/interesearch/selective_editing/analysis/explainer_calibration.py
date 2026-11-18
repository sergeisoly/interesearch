"""Export the saved VPD calibration points used by the interactive summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export(data: Path, posthoc: Path) -> dict:
    protocol = json.loads((data / "protocol.json").read_text())
    method = protocol["main_method"]
    edits = json.loads(posthoc.read_text())["edits"]
    selected = {}
    checked = set()
    for edit in edits:
        if edit["split"] != "confirm" or edit["method"] != method or not edit["usable"]:
            continue
        target, draw, tau = (edit[key] for key in ("target", "draw", "tau"))
        path = data / "calibration" / f"{target}_draw{draw}.json"
        calibration = json.loads(path.read_text())["methods"][method]
        atom = calibration["atom"]
        record = calibration["by_tau"][str(tau)]
        decision = record["recorded"]
        if (decision["status"], decision["scale"], atom["label"]) != ("matched", edit["scale"], edit["atom"]):
            raise ValueError(f"{target} draw {draw} at {tau}: validation and confirmation edits disagree")

        points = {}
        for dose, rise in record["scan"]["grid_evaluations"] + record["scan"]["bisections"]:
            points[1 + atom["sign"] * dose] = rise
        coefficient = edit["coefficient"]
        if points.get(coefficient) != decision["achieved"]:
            raise ValueError(f"{target} draw {draw} at {tau}: missing selected validation measurement")
        row = {
            "target": target, "tau": tau, "atom": atom["label"], "coefficient": coefficient,
            "validation_rise": decision["achieved"], "points": sorted(points.items()),
        }
        if selected.setdefault((target, tau), row) != row:
            raise ValueError(f"{target} at {tau}: calibration differs across draws")
        checked.add((f"{target}:{draw}", tau))

    expected = {(cell, tau) for cell in protocol["expected_cells"] for tau in protocol["taus"]}
    if checked != expected:
        raise ValueError("Expected a usable calibration for every draw and level")
    return {
        "sources": ["data/selective_editing/calibration", "results/selective_editing/posthoc.json#edits"],
        "tolerance": protocol["calibration"]["tol"],
        "calibrations": list(selected.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Selective-editing study data directory")
    parser.add_argument("posthoc", type=Path, help="Saved post hoc analysis with confirmation edits")
    args = parser.parse_args()
    print(json.dumps(export(args.data, args.posthoc), indent=2))


if __name__ == "__main__":
    main()
