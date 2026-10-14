"""Resolve a study recipe once at an entrypoint and bind its applied values to a run."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from interesearch.recipes import load_recipe

PACKAGE = "interesearch.selective_editing"


def validate(recipe: dict) -> dict:
    """Domain checks for counts and ordered scientific grids; schema/types are checked by the loader."""
    positive = {"draw_rows", "fisher_k", "max_walk", "tol",  "atlas_top", "standin_rows",
                "steps", "n_target", "n_retain", "micro_batch", "rank", "grad_clip", "eps", "resamples",
                "q_rows", "n_short", "retain_rows", "val_rows", "ci_rows", "neuron_rows", "n_rates",
                "validation_every", "seed_stride", "eps_tolerance", "unstable_factor"}
    def walk(value, path="recipe"):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            here = f"{path}.{key}"
            if key in positive and item <= 0:
                raise ValueError(f"{here} must be positive")
            if isinstance(item, dict):
                walk(item, here)
            elif isinstance(item, list) and (key.endswith("grid") or key in ("taus", "neuron_ladder")):
                if not item or any(x <= 0 for x in item) or item != sorted(set(item)):
                    raise ValueError(f"{here} must be nonempty, positive and increasing")
    walk(recipe)
    if len(recipe["taus"]) != 2:
        raise ValueError("recipe.taus must contain the lower and upper effect points")
    if not 0 <= recipe["selection"]["eps_percentile"] <= 100:
        raise ValueError("selection.eps_percentile must be between 0 and 100")
    if recipe["readout"]["atlas_top"] > 64:
        raise ValueError("readout.atlas_top cannot exceed the saved top 64 candidates")
    for key in ("bootstrap", "posthoc"):
        if not 0 < recipe[key]["level"] < 1:
            raise ValueError(f"{key}.level must be between 0 and 1")
    if any(d < 0 or d >= len(recipe["selection"]["null_seeds"])
           for d in (*recipe["draws"], *recipe["reduced"]["draws"])):
        raise ValueError("each draw needs a randomized-dictionary seed")
    if not recipe["draws"] or 0 not in recipe["draws"]:
        raise ValueError("draws must include draw 0 for learning-rate choice")
    for key in ("max_bisect", "mono_tol"):
        if recipe["calibration"][key] < 0:
            raise ValueError(f"calibration.{key} must be nonnegative")
    if not recipe["extended"]["seeds"] or 0 not in recipe["extended"]["seeds"]:
        raise ValueError("extended.seeds must include seed 0 for learning-rate choice")
    if any(k < 0 for k in recipe["extended"]["seeds"]):
        raise ValueError("extended seed indices must be nonnegative")
    reduced_draws = recipe["reduced"]["draws"]
    if not reduced_draws or len(set(reduced_draws)) != len(reduced_draws) or not set(reduced_draws) <= set(recipe["draws"]):
        raise ValueError("reduced.draws must be a nonempty unique subset of draws")
    for grid in (*recipe["training"]["learning_rates"].values(), *recipe["extended"]["learning_rates"].values()):
        if not grid or any(x <= 0 for x in grid):
            raise ValueError("training learning rates must be positive")
    return recipe


def resolve(path: str | Path | None = None) -> dict:
    return validate(load_recipe(PACKAGE, path))


def saved_recipe(out: Path, recipe: dict, explicit: bool) -> dict:
    """Use a run's full snapshot by default, and reject explicit changes to an existing run."""
    p = out / "RUN.json"
    if not p.exists():
        return recipe
    record = json.loads(p.read_text())
    if record.get("format") == 2 and "recipe" not in record:
        raise ValueError(f"{p}: a new run requires its complete recipe snapshot")
    saved = record.get("recipe")
    if saved is None:
        if record.get("format") not in (None, 1):
            raise ValueError(f"{p}: missing recipe snapshot")
        if explicit:
            raise ValueError(f"{p}: a legacy run has no recipe snapshot; use a fresh folder for an explicit recipe")
        return recipe  # legacy runs use the packaged published profile
    if explicit and saved != recipe:
        raise ValueError(f"{p}: the scientific recipe differs; use a fresh output folder")
    return copy.deepcopy(saved)
