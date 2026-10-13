"""After the freeze: the dev readout of every matched edit, the confirm readout of the same edits, the post-hoc atlas
and the exported cell records.
Rules: papers/selective-editing/index.qmd#sec-analysis.

Dev sets of a target t with protected language p: target Europarl validation and test, English validation, the
guards (Europarl en test, WikiText-2 test, Pile test documents), protected Europarl validation and test and, for nl
and it, aligned NTREX. Every matched edit of the frozen calibration is rebuilt at its recorded dose (an atom's signed
multiple, a learned update's scale, the top-k neurons' ablated fraction, learned updates load only when their bytes
match the hash of their training record) and measured on every set: per-document CE, its rise against the original
model and KL to it. An edit is unusable when any guard rises by more than 0.1 nats. A target-test rise more than 0.1
nats from tau is flagged, never re-matched. An edit without a match keeps the calibration's status and its
technical or declared kind.

The confirm readout rebuilds the matched dev edits only, at the dose and k of the dev readout (which must agree with
the calibration), on ``confirm_{t}``, ``confirm_{p}`` and ``confirm_en``. An edit is usable when it was usable on dev
and ``confirm_en`` rises by at most 0.1 nats. Edits not matched on dev keep their dev status.

A stand-in readout (a reduced run that has no frozen protocol data) reads every set from selection roles: 4 rows of
target validation, target train rows 124-127 in the target-test slot, English validation rows 0-3 and, as guards,
English validation rows 4-7 (Europarl) and retain rows 128-131 (WikiText-2) and 192-195 (Pile), and retain rows 0-3
(English Europarl train) in the protected slot. Its confirm readout reads target train rows 120-123 as
``confirm_{t}``, retain rows 4-7 as ``confirm_{p}`` and English validation rows 8-11 as ``confirm_en``. Its numbers
say nothing about the study.

The atlas (post hoc, outside the protocol, descriptive only) matches each of the top atoms of the frozen g^2/q
ranking to the lower effect point on target validation and reports its protected CE rise.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from .access import Access, Windows
from .calibration import decide
from .dictionary import Dictionary, atom_update, parse_label_key
from .dose import GUARD_MAX, PAIRS, TEST_DEVIATION_FLAG
from .finetune import load_update, scaled
from .measure import ce_docs, edit_readout
from .neurons import neuron_update
from .selector import ATOM_GRID, BATCH, TAUS

Tensor = torch.Tensor

NTREX_LANGS = ("nl", "it")  # aligned NTREX exists for these protected languages only
STANDIN_ROWS = 4
CONFIRM_EN = "confirm_en"
PROTECTED_TECHNICAL = "technical: non-finite protected rise in the atlas"


@dataclass
class Sets:
    tokens: dict[str, Tensor]
    groups: dict[str, list]  # per set: the bootstrap group of every row (date or source row)
    target_key: str
    protected_key: str
    guards: list[str]
    english: list[str]  # English sets whose KL the exported record carries
    sources: dict[str, str] | None = None  # stand-in: where each set comes from


def _add(sets: dict, groups: dict, name: str, w: Windows, stem: str, rows: slice = slice(None)) -> None:
    sets[name], groups[name] = w.tokens[stem][rows], w.groups[rows]


def dev_sets(access: Access, t: str) -> Sets:
    """The dev readout sets of target ``t`` (readout roles: only after a verified freeze)."""
    p = PAIRS[t]
    sets, groups = {}, {}
    _add(sets, groups, f"europarl_val_{t}", access.windows("val", [t]), t)
    _add(sets, groups, f"europarl_test_{t}", access.windows("test", [t]), t)
    _add(sets, groups, "val_en", access.windows("val_en"), "en")
    guards = {"europarl_test_en": "guard_europarl_en", "wikitext2_test": "guard_wikitext",
              "pile_docs_test": "guard_pile"}
    for name, role in guards.items():
        _add(sets, groups, name, access.windows(role), "en")
    _add(sets, groups, f"europarl_val_{p}", access.windows("protected_europarl_val", [p]), p)
    _add(sets, groups, f"europarl_test_{p}", access.windows("protected_europarl_test", [p]), p)
    if p in NTREX_LANGS:
        _add(sets, groups, f"ntrex_all_{p}", access.windows("protected_ntrex", [p]), p)
    return Sets(sets, groups, f"europarl_test_{t}", f"europarl_test_{p}", list(guards), ["val_en", *guards])


def standin_sets(access: Access, t: str) -> Sets:
    """Stand-in readout sets of target ``t``, from selection roles only (reduced runs)."""
    n, p = STANDIN_ROWS, PAIRS[t]
    val, train, ven, ret = (access.windows(r, langs) for r, langs in
                            (("val", [t]), ("train", [t]), ("val_en", None), ("retain_en", None)))
    spec = {f"europarl_val_{t}": (val, t, 0), f"standin_train_{t}": (train, t, 128 - n), "val_en": (ven, "en", 0),
            "standin_guard_europarl_val_en": (ven, "en", n), "standin_guard_wikitext2_val": (ret, "en", 128),
            "standin_guard_pile_val": (ret, "en", 192), f"standin_{p}": (ret, "en", 0)}
    sets, groups, sources = {}, {}, {}
    for name, (w, stem, a) in spec.items():
        _add(sets, groups, name, w, stem, slice(a, a + n))
        sources[name] = f"{w.role}/{stem}[{a}:{a + n}]"
    guards = [k for k in spec if k.startswith("standin_guard_")]
    return Sets(sets, groups, f"standin_train_{t}", f"standin_{p}", guards, ["val_en", *guards], sources)


def standin_confirm_sets(access: Access, t: str) -> Sets:
    """Stand-in confirm sets of target ``t``, from selection roles only (reduced runs)."""
    n, p = STANDIN_ROWS, PAIRS[t]
    spec = {f"confirm_{t}": (access.windows("train", [t]), t, 128 - 2 * n),
            f"confirm_{p}": (access.windows("retain_en"), "en", n),
            CONFIRM_EN: (access.windows("val_en"), "en", 2 * n)}
    sets, groups, sources = {}, {}, {}
    for name, (w, stem, a) in spec.items():
        _add(sets, groups, name, w, stem, slice(a, a + n))
        sources[name] = f"{w.role}/{stem}[{a}:{a + n}]"
    return Sets(sets, groups, f"confirm_{t}", f"confirm_{p}", [CONFIRM_EN], [CONFIRM_EN], sources)


def confirm_sets(w: Windows, t: str) -> Sets:
    """``confirm_{t}``, ``confirm_{p}`` and ``confirm_en`` of the opened confirm windows ``w``."""
    p = PAIRS[t]
    sets, groups = {}, {}
    for name, stem in ((f"confirm_{t}", t), (f"confirm_{p}", p), (CONFIRM_EN, "en")):
        _add(sets, groups, name, w, stem)
    return Sets(sets, groups, f"confirm_{t}", f"confirm_{p}", [CONFIRM_EN], [CONFIRM_EN])


# --------------------------------------------------------------------------- edits
def rebuild(method: Mapping, out: str | Path, model: nn.Module, dictionary_of: Callable[[str], Dictionary],
            device) -> Callable[..., dict[str, Tensor]]:
    """The edit family of a calibrated method (its record in the frozen calibration of the run folder ``out``):
    ``make(dose)``, for neurons ``make(fraction, k)``."""
    out = Path(out)
    if "atom" in method:
        dic = dictionary_of(method["dictionary"])
        m, c = parse_label_key(method["atom"]["label"])
        sign = method["atom"]["sign"]
        return lambda a: atom_update(dic, m, c, sign * a, device)
    if "learned_update" in method:
        D = load_update(out / method["run"])
        return lambda s: scaled(D, s, device)
    ranked = json.loads((out / method["source"]).read_text())["ranked"]
    pairs = [(r["layer"], r["neuron"]) for r in ranked]
    return lambda frac, k: neuron_update(model, pairs[:k], frac, device)


def _update(make: Callable, method: str, scale: float, k: int | None) -> dict[str, Tensor]:
    return make(scale, k) if method == "neurons" else make(scale)


def base_ce(model: nn.Module, base: Mapping[str, Tensor], sets: Sets, device, bs: int = BATCH) -> dict[str, Tensor]:
    return {n: ce_docs(model, base, x, None, device, bs) for n, x in sets.tokens.items()}


@torch.no_grad()
def read_dev(model: nn.Module, base: Mapping[str, Tensor], calibration: Mapping, sets: Sets,
             ce0: Mapping[str, Tensor], make_of: Callable[[str, Mapping], Callable], device,
             bs: int = BATCH) -> dict[str, dict]:
    """Every method@tau of a cell's calibration: matched edits measured on every set, the others with their status."""
    rows = {}
    for m, rec in calibration["methods"].items():
        if not rec["by_tau"]:  # no edit family: the calibration's reason and its kind, at every tau
            for tau in calibration["taus"]:
                rows[f"{m}@{tau}"] = {"status": rec["status"], "technical_failure": rec["technical_failure"]}
            continue
        for tau, row in rec["by_tau"].items():
            got = row["recorded"]
            if got["status"] != "matched":
                rows[f"{m}@{tau}"] = {"status": got["status"],
                                      "technical_failure": got.get("technical_failure", False)}
                continue
            r = edit_readout(model, base, _update(make_of(m, rec), m, got["scale"], got.get("k")), sets.tokens,
                             device, ce0, bs)
            guards = {n: r[n]["delta_ce"] for n in sets.guards}
            tt = r[sets.target_key]["delta_ce"]
            rows[f"{m}@{tau}"] = {"status": "matched", "scale": got["scale"], "k": got.get("k"), "readout": r,
                                  "usable": all(v <= GUARD_MAX for v in guards.values()), "guards": guards,
                                  "target_test_delta": tt,
                                  "target_test_flag": abs(tt - float(tau)) > TEST_DEVIATION_FLAG}
    return rows


def confirm_plan(dev_edits: Mapping[str, dict], calibration: Mapping) -> list[dict]:
    """The frozen edits of a cell: matched dev edits with their dose and k (which must be the calibration's), the
    others with their dev status."""
    plan = []
    for name, e in dev_edits.items():
        m, tau = name.split("@")
        if e["status"] != "matched":
            plan.append({"name": name, "method": m, "matched": False, "status": e["status"],
                         "technical_failure": e["technical_failure"]})
            continue
        got = calibration["methods"][m]["by_tau"][tau]["recorded"]
        if (got["status"], got["scale"], got.get("k")) != ("matched", e["scale"], e["k"]):
            raise ValueError(f"{name}: the dev readout and the calibration disagree on the edit")
        plan.append({"name": name, "method": m, "tau": float(tau), "matched": True, "scale": e["scale"], "k": e["k"],
                     "dev_usable": e["usable"], "dev_target_test_delta": e["target_test_delta"]})
    return plan


@torch.no_grad()
def read_confirm(model: nn.Module, base: Mapping[str, Tensor], plan: list[dict], calibration: Mapping, sets: Sets,
                 ce0: Mapping[str, Tensor], make_of: Callable[[str, Mapping], Callable], device,
                 bs: int = BATCH) -> dict[str, dict]:
    rows = {}
    for e in plan:
        if not e["matched"]:
            rows[e["name"]] = {"status": e["status"], "technical_failure": e["technical_failure"]}
            continue
        m = e["method"]
        r = edit_readout(model, base, _update(make_of(m, calibration["methods"][m]), m, e["scale"], e["k"]),
                         sets.tokens, device, ce0, bs)
        guards = {n: r[n]["delta_ce"] for n in sets.guards}
        ok = all(v <= GUARD_MAX for v in guards.values())
        tt = r[sets.target_key]["delta_ce"]
        rows[e["name"]] = {"status": "matched", "scale": e["scale"], "k": e["k"], "readout": r,
                           "dev_usable": e["dev_usable"], "guards": guards, "usable": e["dev_usable"] and ok,
                           "guard_flag": not ok, "target_test_delta": tt,
                           "target_test_flag": abs(tt - e["tau"]) > TEST_DEVIATION_FLAG,
                           "dev_target_test_delta": e["dev_target_test_delta"]}
    return rows


@torch.no_grad()
def atlas(model: nn.Module, base: Mapping[str, Tensor], dictionary: Dictionary, selection: Mapping, val_t: Tensor,
          protected: Tensor, device, bs: int = BATCH, n_top: int = 20) -> dict:
    """POST HOC, descriptive: the top ``n_top`` atoms of the frozen ranking, each matched to the lower effect point on
    target validation (sign of g, the atom grid), and the protected CE rise at the match. As in the calibration, an
    atom with a non-finite evaluated effect or protected rise is technical and is not ranked."""
    tau = min(TAUS)
    ce0 = ce_docs(model, base, val_t, None, device, bs)
    ce0_p = ce_docs(model, base, protected, None, device, bs)
    rows = []
    for rank, (m, c, score, label, g, _q, _norm) in enumerate(selection["top64"][:n_top]):
        sign = 1.0 if g > 0 else -1.0

        def rise(a: float, tokens: Tensor = val_t, ref: Tensor = ce0, m=m, c=c, sign=sign) -> float:
            return float((ce_docs(model, base, tokens, atom_update(dictionary, m, c, sign * a, device), device, bs)
                          - ref).mean())

        _, dec = decide(rise, tau, ATOM_GRID)
        row = {"rank": rank, "label_key": label, "score": score, "status": dec["status"], "scale": dec["scale"]}
        if dec["status"] == "matched":
            p = rise(dec["scale"], protected, ce0_p)
            row |= ({"protected_delta_ce": p} if math.isfinite(p) else
                    {"status": PROTECTED_TECHNICAL, "technical_failure": True})
        elif "technical_failure" in dec:
            row["technical_failure"] = True
        rows.append(row)
    matched = [r for r in rows if "protected_delta_ce" in r]
    best = min(matched, key=lambda r: r["protected_delta_ce"]) if matched else None
    return {"label": "POST HOC (descriptive, after the freeze)", "tau": tau, "rows": rows,
            "most_selective_rank": None if best is None else best["rank"]}


# --------------------------------------------------------------------------- exported records
def public_cell(readout: Mapping, split: str, sha256: str) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(cell.json, protected.npz, posthoc.npz) of one readout record: the scalars of every edit, and per matched edit
    the protected per-document CE and KL and the target per-document CE (float64, document order, the target and
    protected documents are paired by position)."""
    pkey, tkey = readout["protected_key"], readout["target_key"]
    n = len(readout["base_ce"][pkey])
    edits, measured, ce, tce, kl = {}, [], [], [], []
    for name, e in readout["edits"].items():
        if e["status"] != "matched":
            edits[name] = {"status": e["status"], "technical_failure": e["technical_failure"]}
            continue
        r = e["readout"]
        edits[name] = {"status": "matched", "scale": e["scale"], "k": e["k"], "guards": e["guards"],
                       "target_test_delta": e["target_test_delta"],
                       "english_kl": {s: r[s]["kl"] for s in readout["english_keys"]}}
        if split == "confirm":
            edits[name]["protected_delta"] = r[pkey]["delta_ce"]
        measured.append(name)
        ce.append(r[pkey]["ce_doc"])
        tce.append(r[tkey]["ce_doc"])
        kl.append(r[pkey]["kl_doc"])

    def docs(rows: list) -> np.ndarray:
        return np.asarray(rows, dtype=np.float64).reshape(len(rows), n)

    info = {"format": 1, "study": "selective_editing", "split": split, "target": readout["target"],
            "protected": readout["protected"], "draw": readout["draw"], "target_key": tkey, "protected_key": pkey,
            "groups": readout["groups"][pkey], "measured": measured, "edits": edits,
            "source": {"readout_sha256": sha256}}
    return (info, {"base_ce": np.asarray(readout["base_ce"][pkey], dtype=np.float64), "ce_doc": docs(ce)},
            {"target_base_ce": np.asarray(readout["base_ce"][tkey], dtype=np.float64), "target_ce_doc": docs(tce),
             "protected_kl_doc": docs(kl)})
