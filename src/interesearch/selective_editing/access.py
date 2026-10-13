"""Data roles, the freeze and the one-time opening of the confirm set in the selective-editing model path.
Rules: papers/selective-editing/index.qmd#sec-access and papers/selective-editing/index.qmd#sec-analysis.

Token windows come from window directories, one data role each: ``windows.json`` with the role, the file stems
(``languages``) and, per row, its session date (``dates``: Europarl sets, row i the same date in every stem) or its
provenance (``rows``: the other pools, a guard or NTREX entry names the document or block of the row). One int32
array ``<stem>.npy`` of shape [rows, 257] per stem, and ``SHA256SUMS`` over all of them. The date or ``rows`` entry
of a row is its bootstrap group.

    Selection roles  train, val (target Europarl), retain_en, qcheck_en, val_en: read before the freeze.
    Readout roles    test, guard_europarl_en, guard_wikitext, guard_pile, protected_europarl_val,
                     protected_europarl_test, protected_ntrex: read only after a verified freeze, and never by a
                     stand-in run, which reads its readout from selection roles.
    Confirm          read only by the confirm readout, after its opening record (``open_confirm``).

The declared role of a directory, and every role of a comma-separated list, is checked before any of its files is
hashed or read. The content identity of a directory is the sha256 of its ``SHA256SUMS`` (``window_identity``). Selection, training and calibration refuse to run once the freeze exists or a readout role was
opened in the process.

The freeze (``FREEZE.json``) records the sha256 of every pre-freeze record (``*.json`` under ``FROZEN_DIRS``), the
code identity, the curvature-check record, the expected cells and the technical gaps (items that failed or were
blocked, missing calibrations, methods calibrated as technical). It refuses gaps unless ``allow_incomplete``, which
records them. A readout role opens only while the frozen records and the code are unchanged.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..checksums import sha256_file, verify_checksums
from .data import write_json
from .selector import require_gate

SELECTION_ROLES = ("train", "val", "retain_en", "qcheck_en", "val_en")
READOUT_ROLES = ("test", "guard_europarl_en", "guard_wikitext", "guard_pile", "protected_europarl_val",
                 "protected_europarl_test", "protected_ntrex")
CONFIRM_ROLE = "confirm"
WINDOWS_FILE = "windows.json"
WINDOW_TOKENS = 257  # 256 predictions per window
FREEZE = "FREEZE.json"
FROZEN_DIRS = ("q", "select", "finetune", "references", "neurons", "calibration")
STATUS_FILE = "status.jsonl"
NOT_OK = ("failed", "blocked")  # item statuses that are technical gaps. "declared" and "not run" are not
CONFIRM_OPENED = "CONFIRM_OPENED.json"


class AccessError(PermissionError):
    """A data access the protocol forbids at this point."""


class AlreadyOpened(AccessError):
    """The confirm set was opened before: a second readout is refused."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def package_identity() -> dict[str, str]:
    """sha256 of every source file of the package, by path inside it."""
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root.parent)): sha256_file(p) for p in sorted(root.rglob("*.py"))}


# --------------------------------------------------------------------------- window directories
@dataclass(frozen=True)
class Windows:
    role: str
    groups: list  # per row: the session date, or the rows entry
    tokens: dict[str, torch.Tensor]  # stem -> int64 [rows, 257]
    sha256: dict[str, str]  # file -> sha256, as SHA256SUMS lists it


def load_windows(directory: str | Path, role: str, refused: Sequence[str] = (),
                 langs: Sequence[str] | None = None) -> Windows:
    """The windows of ``role`` (the stems ``langs``, default all). A directory that declares a role in ``refused``,
    also inside a comma-separated list, or another role than ``role`` is refused before any file is hashed or read.
    Then the files must match SHA256SUMS."""
    directory = Path(directory)
    meta = json.loads((directory / WINDOWS_FILE).read_text())
    bad = [r for r in meta["role"].split(",") if r in refused]
    if bad:
        raise AccessError(f"{directory}: role {meta['role']!r} may not be opened here ({', '.join(bad)})")
    if meta["role"] != role:
        raise AccessError(f"{directory}: role {meta['role']!r}, expected {role!r}")
    verify_checksums(directory, (WINDOWS_FILE, *sorted({f"{s}.npy" for s in (*meta["languages"], *(langs or ()))})))
    groups = meta["dates"] if "dates" in meta else meta["rows"]
    tokens = {}
    for s in langs or meta["languages"]:
        arr = np.load(directory / f"{s}.npy", allow_pickle=False)
        if arr.dtype != np.int32 or arr.shape != (len(groups), WINDOW_TOKENS):
            raise ValueError(f"{directory}/{s}.npy: {arr.dtype} {arr.shape}, expected int32 "
                             f"{(len(groups), WINDOW_TOKENS)}")
        tokens[s] = torch.from_numpy(arr.astype(np.int64))
    sums = dict(line.split("  ", 1)[::-1] for line in (directory / "SHA256SUMS").read_text().splitlines())
    return Windows(meta["role"], list(groups), tokens, sums)


def window_identity(directory: str | Path, role: str) -> str:
    """The content identity of a window directory of ``role``: the sha256 of its ``SHA256SUMS`` (which lists every
    file of it, checked when the windows are loaded). Only ``windows.json`` (for the role) and the manifest are read."""
    directory = Path(directory)
    declared = json.loads((directory / WINDOWS_FILE).read_text())["role"]
    if declared != role:
        raise AccessError(f"{directory}: role {declared!r}, expected {role!r}")
    return sha256_file(directory / "SHA256SUMS")


class Access:
    """The roles one run may open from ``windows_dir/<role>``: selection roles before and after the freeze. Readout
    roles after a verified freeze of ``out``, never with ``standin``. The confirm role never (``load_confirm``)."""

    def __init__(self, windows_dir: str | Path, out: str | Path, standin: bool = False) -> None:
        self.windows_dir, self.out, self.standin = Path(windows_dir), Path(out), standin
        self.readout_opened = False
        self.opened: dict[str, dict] = {}  # role -> what was read, for the run record

    def windows(self, role: str, langs: Sequence[str] | None = None) -> Windows:
        if role in READOUT_ROLES:
            if self.standin:
                raise AccessError(f"a stand-in run reads no readout role ({role})")
            verify_freeze(self.out)
            self.readout_opened = True
            refused = (CONFIRM_ROLE,)
        else:
            refused = (*READOUT_ROLES, CONFIRM_ROLE)
        w = load_windows(self.windows_dir / role, role, refused, langs)
        stems = sorted(set(self.opened.get(role, {}).get("stems", [])) | set(w.tokens))
        self.opened[role] = {"rows": len(w.groups), "stems": stems, "sha256": w.sha256}
        return w

    def blind(self, stage: str) -> None:
        """Selection, training and calibration run only before the freeze and before any readout role is read."""
        if self.readout_opened:
            raise AccessError(f"{stage}: a readout role was opened in this process")
        if (self.out / FREEZE).exists():
            raise AccessError(f"{stage}: {self.out / FREEZE} exists. Selection, training and calibration are closed")


# --------------------------------------------------------------------------- item status and the freeze
def status_rows(out: str | Path) -> dict[str, dict]:
    """The latest status row of every item of the run directory."""
    p = Path(out) / STATUS_FILE
    last = {}
    if p.exists():
        for line in p.read_text().splitlines():
            row = json.loads(line)
            last[row["item"]] = row
    return last


def frozen_artifacts(out: str | Path) -> dict[str, str]:
    out = Path(out)
    return {str(p.relative_to(out)): sha256_file(p) for d in FROZEN_DIRS for p in sorted((out / d).rglob("*.json"))}


def technical_gaps(out: str | Path, cells: Sequence[tuple[str, int]]) -> list[str]:
    """Items that failed or were blocked, missing calibrations of ``cells`` and methods or effect points that the
    calibration marked technical. Declared outcomes (no feasible atom, no rate reached tau, a dose not reached) are
    not gaps."""
    out = Path(out)
    gaps = [f"item {n}: {r['status']} ({r['reason']})" for n, r in status_rows(out).items() if r["status"] in NOT_OK]
    for t, d in cells:
        p = out / "calibration" / f"{t}_draw{d}.json"
        if not p.exists():
            gaps.append(f"calibration {t}:{d}: missing")
            continue
        for m, rec in json.loads(p.read_text())["methods"].items():
            if rec.get("technical_failure"):
                gaps.append(f"calibration {t}:{d} {m}: {rec['status']}")
            for tau, row in rec["by_tau"].items():
                if row["recorded"].get("technical_failure"):
                    gaps.append(f"calibration {t}:{d} {m}@{tau}: {row['recorded']['status']}")
    return gaps


def write_freeze(out: str | Path, gate: str | Path, cells: Sequence[tuple[str, int]], allow_incomplete: bool = False,
                 note: str = "") -> dict:
    """FREEZE.json of the run directory ``out``. Refused when it exists, without a passed curvature check of the
    current code (``gate``), with nothing to freeze, and with technical gaps unless ``allow_incomplete``."""
    out = Path(out)
    if (out / FREEZE).exists():
        raise AccessError(f"{out / FREEZE} exists already")
    gate_rec = require_gate(gate)
    arts = frozen_artifacts(out)
    if not arts:
        raise ValueError(f"nothing to freeze under {out}")
    gaps = technical_gaps(out, cells)
    if gaps and not allow_incomplete:
        raise AccessError(f"freeze refused: {len(gaps)} technical gap(s) in {out}: {gaps[:8]} (re-run the failed "
                          "items, or freeze with allow_incomplete to record them)")
    rec = {"created_utc": utc(), "note": note, "artifacts": arts, "code": package_identity(),
           "gate": {"path": str(gate), "sha256": sha256_file(gate), "passed": gate_rec["passed"]},
           "expected_cells": [f"{t}:{d}" for t, d in cells], "technical_gaps": gaps,
           "allow_incomplete": bool(allow_incomplete)}
    write_json(out / FREEZE, rec)
    return rec


def verify_freeze(out: str | Path) -> dict:
    """The freeze record of ``out``, refused when absent or when a frozen record or the code changed since."""
    p = Path(out) / FREEZE
    if not p.exists():
        raise AccessError(f"no {p}: readout roles stay closed until the freeze")
    rec = json.loads(p.read_text())
    now = frozen_artifacts(out)
    if now != rec["artifacts"]:
        changed = sorted(set(now.items()) ^ set(rec["artifacts"].items()))
        raise AccessError(f"pre-freeze records changed after the freeze: {changed[:6]}")
    code = package_identity()
    if code != rec["code"]:
        changed = sorted(k for k in set(code) | set(rec["code"]) if code.get(k) != rec["code"].get(k))
        raise AccessError(f"code changed after the freeze: {changed}")
    return rec


# --------------------------------------------------------------------------- the confirm set
def open_confirm(out: str | Path, record: dict, resume: bool = False) -> dict:
    """Write ``CONFIRM_OPENED.json`` in ``out`` (exclusive create) before the first confirm window is read. ``record``
    carries the content identity of the confirm windows (``confirm_windows``). A second opening is refused. ``resume``
    with the same code, freeze and confirm-window identity only completes the cells without a readout, and is
    logged."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / CONFIRM_OPENED
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        prev = json.loads(p.read_text())
        same = all(prev[k] == record[k] for k in ("code", "freeze_sha256", "confirm_windows"))
        if not (resume and same):
            raise AlreadyOpened(f"{p} exists ({prev['utc']}): the confirm set was opened already. Only a resume "
                                "with the same code, freeze and windows may complete missing cells") from None
        with open(out / "resume_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return prev
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(record, indent=1))
    return record


def load_confirm(directory: str | Path, out: str | Path, langs: Sequence[str]) -> Windows:
    """The confirm windows of ``langs``, only after the opening record exists in ``out``."""
    if not (Path(out) / CONFIRM_OPENED).exists():
        raise AccessError(f"the confirm windows are read only after {Path(out) / CONFIRM_OPENED} is written")
    return load_windows(directory, CONFIRM_ROLE, READOUT_ROLES, langs)
