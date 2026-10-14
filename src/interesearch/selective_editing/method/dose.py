"""Effect-matched calibration of every method of one target and draw on target validation, and the method list.
Rules: papers/selective-editing/index.qmd#sec-calibration and papers/selective-editing/index.qmd#sec-comparators.

Each edit is a one-parameter family, calibrated by ``calibration.decide`` at every effect point tau:

* atoms (vpd_atom, null_atom, omp_finetune, omp_lora, ci_difference, historical_atom): ``W + sign a A_c``, ``a`` on
  the atom grid (in units of the atom's own norm, up to 2).
* neurons: the smallest k of the ladder whose full ablation reaches tau, then the ablated fraction on the neuron grid
  (``calibration.decide_neurons``).
* learned updates (lora_rank1, finetune_24): ``s D``, ``s`` on the scale grid (never amplified).

Recorded per method and tau: the scan (evaluated points and bisections) and the decision. At a matched dose also the
edit norm and the English validation CE rise and KL, a calibration-time warning above 0.1 nats (the binding guards
are the English test sets read after the freeze). A method without an edit records why: ``technical_failure`` true
when an input is absent (an upstream item failed, was blocked or did not run), false for a declared outcome of the
rules (no feasible atom, no learning rate reached the upper effect point, no checkpoint met the historical rule).
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch
import torch.nn as nn

from interesearch.selective_editing.method.calibration import MAX_BISECT, TOL, decide, decide_neurons
from interesearch.selective_editing.method.dictionary import Dictionary, atom_update, label_key
from interesearch.selective_editing.method.finetune import (
    OMP_LORA_LR,
    OMP_LORA_RULE,
    SCALE_GRID,
    SEED_OFFSET,
    checkpoint_rule_text,
    scaled,
    update_rank,
)
from interesearch.selective_editing.method.measure import ce_docs, edit_readout, update_norm
from interesearch.selective_editing.method.neurons import NEURON_LADDER, neuron_update
from interesearch.selective_editing.method.selector import ATOM_GRID, BATCH, TAUS

Tensor = torch.Tensor

METHODS = ("vpd_atom", "null_atom", "omp_finetune", "omp_lora", "ci_difference", "historical_atom", "neurons",
           "lora_rank1", "finetune_24")
LABELS = {
    "vpd_atom": "g^2/q selector, VPD dictionary (main)",
    "null_atom": "g^2/q selector, randomized dictionary (same rule)",
    "omp_finetune": "OMP top-1 of the full-matrix fine-tune of all 24 modules (same target and draw)",
    "omp_lora": "OMP top-1 of the historical LoRA condition (rank-1 LoRA on all 24 modules, lr 1e-4, historical "
                "checkpoint rule, English retention)",
    "ci_difference": "CI top-1 (adaptation of Bushnaq's top-16 rule)",
    "historical_atom": "historical atom 0.attn.o:268, ablation side (historical, de only, dev-selected)",
    "neurons": "top-k activation neurons (target vs English rate), ablation",
    "lora_rank1": "rank-1 LoRA in the selected atom's module",
    "finetune_24": "full-matrix fine-tune of all 24 modules",
}
MAIN = "vpd_atom"
CLAIM = ("null_atom", "lora_rank1", "finetune_24", "neurons")  # the claim: vpd_atom against each of these
REFERENCES = ("omp_finetune", "omp_lora", "ci_difference", "historical_atom")  # reported, not part of the claim
ATOMS = METHODS[:6]
LEARNED = {"lora_rank1": "lora", "finetune_24": "matrix"}  # method -> fine-tune condition
HISTORICAL_ATOM = {"de": ("0.attn.o:268", -1.0)}  # label, sign: fixed before this study, on the de target only
PAIRS = {"de": "nl", "es": "pt", "fr": "it"}  # target -> protected language
NEURON_GRID = SCALE_GRID  # ablated fraction (1.0 = output columns zeroed)
GRID_OF_METHOD = {m: "atom_grid" if m in ATOMS else "neuron_grid" if m == "neurons" else "scale_grid"
                  for m in METHODS}
GUARD_MAX = 0.1  # nats: an edit raising CE by more on any English guard set is unusable
TEST_DEVIATION_FLAG = 0.1  # nats: a target-test rise this far from tau is flagged (no re-matching on test)


def labels_for(recipe: Mapping | None = None) -> dict[str, str]:
    """Method names for the applied settings; the published profile retains its exact labels."""
    labels = dict(LABELS)
    if recipe is None:
        return labels
    training, omp = recipe["training"], recipe["omp_lora"]
    rank = training["rank"]
    if rank != 1:
        for method in ("lora_rank1", "omp_lora"):
            labels[method] = labels[method].replace("rank-1 LoRA", f"rank-{rank} LoRA")
    if omp["lr"] != OMP_LORA_LR:
        labels["omp_lora"] = labels["omp_lora"].replace("lr 1e-4", f"lr {omp['lr']:g}")
    rule = checkpoint_rule_text(recipe)
    if rule != OMP_LORA_RULE:
        labels["omp_lora"] = labels["omp_lora"].replace("historical checkpoint rule", f"checkpoint rule: {rule}")
    omp_details = []
    if omp["seed_offset"] != 0:
        omp_details.append(f"seed = {omp['seed_offset']} + draw")
    checkpoints = sorted({int(s) for s in training["checkpoints"] if 0 < int(s) <= training["steps"]}
                         | {training["steps"]})
    if tuple(checkpoints) != (50, 100, 150, 200):
        omp_details.append("checkpoints " + "/".join(map(str, checkpoints)))
    if omp_details:
        labels["omp_lora"] = labels["omp_lora"][:-1] + "; " + "; ".join(omp_details) + ")"
    if training["seed_offset"] != SEED_OFFSET:
        for method in ("lora_rank1", "finetune_24", "omp_finetune"):
            labels[method] += f"; seed = {training['seed_offset']} + draw"
    return labels


def missing(what: str) -> tuple[str, bool]:
    """Status of a method whose input is absent: technical."""
    return f"missing input: {what} (failed, blocked or not run)", True


def declared(what: str) -> tuple[str, bool]:
    """Status of a method that has no edit by its declared rule."""
    return f"no edit by the declared rule: {what}", False


@dataclass
class Family:
    """A one-parameter edit family: ``make(dose)`` (neurons: ``make(fraction, k)``) gives the update."""

    make: Callable[..., dict[str, Tensor]]
    grid: tuple[float, ...]
    identity: dict  # the method's edit: rank, modules and the atom or learned update it scales
    search: dict = field(default_factory=dict)  # how the edit was chosen
    ladder: tuple[int, ...] | None = None  # neurons: the k ladder


def atom_family(dictionary: Dictionary, module: str, c: int, sign: float, device, search: Mapping | None = None,
                *, recipe: Mapping | None = None) -> Family:
    grid = tuple(recipe["selection"]["atom_grid"]) if recipe is not None else ATOM_GRID
    return Family(lambda a: atom_update(dictionary, module, c, sign * a, device), grid,
                  {"rank": 1, "modules": [module], "dictionary": dictionary.name,
                   "atom": {"label": label_key(module, c), "sign": sign, "norm": float(dictionary[module].norms[c])}},
                  dict(search or {}))


def learned_family(update: Mapping[str, Tensor], run: str, train: Mapping, lr_grid, device, *,
                   recipe: Mapping | None = None) -> Family:
    """``s D`` of the final update of a training run (``run`` its directory in the run folder, ``train`` its
    train.json)."""
    grid = tuple(recipe["training"]["scale_grid"]) if recipe is not None else SCALE_GRID
    return Family(lambda s: scaled(update, s, device), grid,
                  {"rank": update_rank(update), "modules": list(update), "run": run,
                   "update_sha256": train["update_sha256"], "train_seconds": train["timings"]["total_s"],
                   "learned_update": {"learning_rate": train["lr"], "learning_rate_grid": list(lr_grid),
                                      "chosen_on": "draw 0", "norm": update_norm(update)}},
                  {"lr_grid": list(lr_grid), "lr": train["lr"], "lr_choice_on": "draw 0",
                   "trainings_for_choice": len(lr_grid)})


def neuron_family(model: nn.Module, neurons: list[tuple[int, int]], source: str, device, *,
                  recipe: Mapping | None = None) -> Family:
    """Ablation of the top-k neurons of the ranking at ``source`` (``neurons[:k]``, fraction ``frac``)."""
    cal = recipe["calibration"] if recipe is not None else {}
    grid = tuple(cal.get("neuron_grid", NEURON_GRID))
    ladder = tuple(cal.get("neuron_ladder", NEURON_LADDER))
    return Family(lambda frac, k: neuron_update(model, neurons[:k], frac, device), grid,
                  {"source": source, "n_ranked": len(neurons)}, {}, ladder)


def _scan(mt: dict | None) -> dict | None:
    return mt and {k: [list(p) for p in mt[k]] for k in ("grid_evaluations", "bisections")}


@torch.no_grad()
def calibrate(model: nn.Module, base: Mapping[str, Tensor], val_t: Tensor, val_en: Tensor,
              families: Mapping[str, Family | tuple[str, bool]], device, bs: int = BATCH,
              taus=None, labels: Mapping[str, str] | None = None, *, recipe: Mapping | None = None) -> dict:
    """Calibration of every method of ``labels`` (method -> label, the main study's by default): ``families[m]`` is
    its edit family or, without an edit, its (status, technical_failure)."""
    t0 = time.time()
    if labels is None:
        labels = labels_for(recipe)
    if taus is None:
        taus = recipe["taus"] if recipe is not None else TAUS
    cal = recipe["calibration"] if recipe is not None else {}
    parameters = {"max_bisect": cal.get("max_bisect", MAX_BISECT), "tol": cal.get("tol", TOL),
                  "mono_tol": cal.get("mono_tol", TOL)}
    guard_max = recipe["readout"]["guard_max"] if recipe is not None else GUARD_MAX
    ce0 = ce_docs(model, base, val_t, None, device, bs)
    ce0_en = ce_docs(model, base, val_en, None, device, bs)

    def english(upd: dict) -> dict:
        r = edit_readout(model, base, upd, {"en": val_en}, device, {"en": ce0_en}, bs)["en"]
        return {"edit_norm": update_norm(upd),
                "english_val": {"delta_ce": r["delta_ce"], "kl": r["kl"], "warning": r["delta_ce"] > guard_max}}

    methods = {}
    for m, label in labels.items():
        fam = families[m]
        rec = {"label": label}
        if isinstance(fam, tuple):
            methods[m] = rec | {"search": {}, "status": fam[0], "technical_failure": fam[1], "by_tau": {}}
            continue
        cache: dict = {}

        def effect(x: float, k: int | None = None, fam=fam, cache=cache) -> float:
            key = float(x) if k is None else (k, float(x))
            if key not in cache:
                upd = fam.make(float(x)) if k is None else fam.make(float(x), k)
                cache[key] = float((ce_docs(model, base, val_t, upd, device, bs) - ce0).mean())
            return cache[key]

        rec |= {"search": fam.search, **fam.identity, "by_tau": {}}
        if fam.ladder is not None:
            rec["ladder"] = [(k, effect(1.0, k)) for k in fam.ladder]
        for tau in taus:
            if fam.ladder is None:
                mt, got = decide(effect, tau, fam.grid, **parameters)
            else:
                mt, got = decide_neurons(rec["ladder"], lambda k: (lambda fr, k=k: effect(fr, k)), tau, fam.grid,
                                         **parameters)
            row = {"scan": _scan(mt), "recorded": got}
            if got["status"] == "matched":
                upd = fam.make(got["scale"]) if fam.ladder is None else fam.make(got["scale"], got["k"])
                row |= english(upd)
                if fam.ladder is not None:
                    row |= {"rank": got["k"], "modules": sorted(upd)}
            rec["by_tau"][str(tau)] = row
        rec["val_forward_passes"] = len(cache)
        methods[m] = rec
    recorded_parameters = {k: v for k, v in parameters.items() if recipe is not None or k != "mono_tol"}
    return {"taus": list(taus), **recorded_parameters, "n_val_windows": int(val_t.shape[0]),
            "n_val_en_windows": int(val_en.shape[0]), "batch_size": bs, "methods": methods,
            "seconds": time.time() - t0}


# --------------------------------------------------------------------------- public records
RECORDED = ("status", "scale", "achieved", "best_reached", "k", "technical_failure")
IDENTITY = ("atom", "learned_update")


def public_calibration(rec: dict, target: str, draw: int, sha256: str) -> dict:
    """The exported calibration record of one cell: per method with an edit family, its identity (the atom or the
    learned update its edits scale), the neuron ladder, and per tau the scan, the decision and the edit norm of a
    matched decision."""
    methods = {}
    for m, r in rec["methods"].items():
        if not r["by_tau"]:
            continue
        methods[m] = ({k: r[k] for k in IDENTITY if k in r} | ({"ladder": r["ladder"]} if "ladder" in r else {})
                      | {"by_tau": {tau: {"scan": row["scan"],
                                          "recorded": {k: row["recorded"][k] for k in RECORDED if k in row["recorded"]},
                                          "edit_norm": row.get("edit_norm")}
                                    for tau, row in r["by_tau"].items()}})
    return {"format": 1, "study": "selective_editing", "target": target, "draw": draw, "methods": methods,
            "source": {"calibration_sha256": sha256}}


def protocol(cells: list[tuple[str, int]], freeze: dict, freeze_sha256: str, reduced: bool, *,
             recipe: Mapping | None = None) -> dict:
    """The exported protocol of a run (the records of a ``reduced`` run say so and load only with
    ``allow_reduced``)."""
    cal = recipe["calibration"] if recipe is not None else {}
    readout = recipe["readout"] if recipe is not None else {}
    rec = {"format": 2 if recipe is not None else 1, "study": "selective_editing",
           "expected_cells": [f"{t}:{d}" for t, d in cells],
           "freeze": {k: freeze[k] for k in ("technical_gaps", "allow_incomplete")},
           "targets": dict(recipe["targets"] if recipe is not None else PAIRS),
           "taus": list(recipe["taus"] if recipe is not None else TAUS), "methods": labels_for(recipe), "main_method": MAIN,
           "claim_comparators": list(CLAIM), "reference_comparators": list(REFERENCES),
           "historical_atom_targets": sorted(HISTORICAL_ATOM), "guard_max": readout.get("guard_max", GUARD_MAX),
           "target_deviation_flag": readout.get("target_deviation_flag", TEST_DEVIATION_FLAG),
           "calibration": {"tol": cal.get("tol", TOL), "max_bisect": cal.get("max_bisect", MAX_BISECT),
                           "atom_grid": list(recipe["selection"]["atom_grid"] if recipe is not None else ATOM_GRID),
                           "scale_grid": list(recipe["training"]["scale_grid"] if recipe is not None else SCALE_GRID),
                           "neuron_ladder": list(cal.get("neuron_ladder", NEURON_LADDER)),
                           "neuron_grid": list(cal.get("neuron_grid", NEURON_GRID)),
                           "grid_of_method": dict(GRID_OF_METHOD)},
           "source": {"freeze_sha256": freeze_sha256, "dev_analysis_sha256": None, "confirm_analysis_sha256": None}}
    if recipe is not None:
        rec["recipe"] = deepcopy(dict(recipe))
        rec["calibration"]["mono_tol"] = cal["mono_tol"]
        rec["bootstrap"] = deepcopy(recipe["bootstrap"])
        rec["posthoc"] = deepcopy(recipe["posthoc"])
    return rec | ({"reduced": True} if reduced else {})
