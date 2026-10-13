"""Exported measurement blocks of the repair-transfer study.
Data roles: papers/repair-transfer/index.qmd#sec-app-methods.

A block holds per-window measurements of every candidate on the selection windows and of the chosen repairs on
the readout dates, together with the protocol settings and the records of the original run (decisions, fit loss,
timings, polish, direction cosines). Values are float64, unrounded and in the original row order. Measurements
are indexed as per[name][language][metric] -> 1-D tensor, with metrics ce, kl_ref = KL(p_finetuned || p_X) and
kl_base = KL(p_base || p_X). The donor's fit record holds the inputs of the donor repair's one-time cost.

The model path reads token windows from window directories: ``windows.json`` (the data role, the session dates in
row order, the languages and, for fine-tune text, where its rows come from), one int32 array ``<lang>.npy`` of
shape [dates, 257] per language, and ``SHA256SUMS`` over all of them. Row i of every language is date i.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..checksums import sha256_file, verify_checksums

BLOCK_FILES = ("block.json", "readout.npz", "selection.npz")
RUN_RECORD = "run.json"
DONOR_FILES = ("donor_repair.pt", "fit_record.json")
WINDOWS_FILE = "windows.json"
WINDOW_TOKENS = 257  # 256 predictions per window
# Roles a reduced run may never open (papers/repair-transfer/index.qmd#sec-app-methods): this study's readout
# dates and the donor stage's readout text. Rows of the latter are the targets' fine-tune text. That input has its
# own role, finetune_train, and is not refused by this rule.
READOUT_ROLES = ("readout", "donor_readout")


@dataclass(frozen=True)
class Block:
    info: dict
    selection: dict
    readout: dict


def _measurements(path: Path, names: list[str], rows: int) -> dict:
    per: dict = {n: {} for n in names}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            lang, metric = key.split("/")
            arr = z[key]
            if arr.dtype != np.float64 or arr.shape != (len(names), rows):
                raise ValueError(f"{path.name}:{key}: {arr.dtype} {arr.shape}, expected float64 {(len(names), rows)}")
            for i, n in enumerate(names):
                per[n].setdefault(lang, {})[metric] = torch.from_numpy(arr[i].copy())
    return per


def load_block(block_dir: str | Path, allow_reduced: bool = False) -> Block:
    """A block for analysis. Refused: a directory whose run record (``run.json``, written by ``run.py``) reports an
    unsuccessful run, and a block of a reduced verification run unless ``allow_reduced`` (code-path checks only)."""
    block_dir = Path(block_dir)
    if (block_dir / RUN_RECORD).exists():
        status = json.loads((block_dir / RUN_RECORD).read_text())["status"]
        if status != "ok":
            raise ValueError(f"{block_dir}: the run that wrote it ended with status {status!r}. Not an analysis block")
    verify_checksums(block_dir, BLOCK_FILES)
    info = json.loads((block_dir / "block.json").read_text())
    if info.get("run", {}).get("reduced") and not allow_reduced:
        raise ValueError(f"{block_dir}: a reduced verification run ({info['run']['deviations']}), not an analysis "
                         f"block")
    return Block(
        info=info,
        selection=_measurements(block_dir / "selection.npz", info["selection"]["names"], info["selection"]["rows"]),
        readout=_measurements(block_dir / "readout.npz", info["readout"]["names"], len(info["readout"]["dates"])),
    )


def _stack(per: dict, names: list[str]) -> dict[str, np.ndarray]:
    """``{'<lang>/<metric>': float64 [names, rows]}`` in the given name order."""
    first = per[names[0]]
    return {f"{lg}/{k}": np.stack([per[n][lg][k].numpy() for n in names]) for lg in first for k in first[lg]}


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """NPZ with a fixed member order and timestamp, so equal arrays give equal bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for key in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())


def write_block(block_dir: str | Path, info: dict, selection: dict, readout: dict) -> None:
    """The inverse of ``load_block``: measurements in the name order of ``info``, then ``SHA256SUMS``."""
    block_dir = Path(block_dir)
    for per, names in ((selection, info["selection"]["names"]), (readout, info["readout"]["names"])):
        if list(per) != names:
            raise ValueError("measurements are not in the block's name order")
    _write_npz(block_dir / "selection.npz", _stack(selection, info["selection"]["names"]))
    _write_npz(block_dir / "readout.npz", _stack(readout, info["readout"]["names"]))
    (block_dir / "block.json").write_text(json.dumps(info, indent=1) + "\n")
    (block_dir / "SHA256SUMS").write_text("".join(f"{sha256_file(block_dir / f)}  {f}\n" for f in BLOCK_FILES))


@dataclass(frozen=True)
class Windows:
    role: str
    dates: list[str]
    tokens: dict[str, torch.Tensor]  # language -> int64 [rows, 257]
    source: dict  # fine-tune text: the role and rows it was drawn from. Empty otherwise
    sha256: dict[str, str]  # file -> sha256, as SHA256SUMS lists it


def window_role(directory: str | Path) -> str:
    """The role a window directory declares (several roles concatenated in order are joined by commas)."""
    return json.loads((Path(directory) / WINDOWS_FILE).read_text())["role"]


def refused_roles(role: str, refused: tuple[str, ...]) -> list[str]:
    return [r for r in role.split(",") if r in refused]


def load_windows(directory: str | Path, langs: tuple[str, ...] | None = None, rows: int | None = None,
                 role: str | None = None, refused: tuple[str, ...] = ()) -> Windows:
    """The windows of one role, the first ``rows`` dates (default all). A directory of another role than ``role``
    (when given) or of a role in ``refused``, also inside a list of roles, is refused before any array is hashed or
    read. Then the files must match SHA256SUMS."""
    directory = Path(directory)
    meta = json.loads((directory / WINDOWS_FILE).read_text())
    if role is not None and meta["role"] != role:
        raise ValueError(f"{directory}: role {meta['role']!r}, expected {role!r}")
    bad = refused_roles(meta["role"], refused)
    if bad:
        raise ValueError(f"{directory}: role {meta['role']!r} may not be opened by this run ({bad})")
    verify_checksums(directory, (WINDOWS_FILE, *{f"{lg}.npy" for lg in (*meta["languages"], *(langs or ()))}))
    tokens = {}
    for lg in langs or meta["languages"]:
        arr = np.load(directory / f"{lg}.npy", allow_pickle=False)
        if arr.dtype != np.int32 or arr.shape != (len(meta["dates"]), WINDOW_TOKENS):
            raise ValueError(f"{directory}/{lg}.npy: {arr.dtype} {arr.shape}, expected int32 "
                             f"{(len(meta['dates']), WINDOW_TOKENS)}")
        tokens[lg] = torch.from_numpy(arr.astype("int64")[:rows])
    sums = dict(line.split("  ", 1)[::-1] for line in (directory / "SHA256SUMS").read_text().splitlines())
    return Windows(meta["role"], meta["dates"][:rows], tokens, meta.get("source", {}), sums)


def load_donor(donor_dir: str | Path) -> dict:
    """Fit record of the donor repair: the inputs of its one-time cost."""
    donor_dir = Path(donor_dir)
    verify_checksums(donor_dir, DONOR_FILES)
    return json.loads((donor_dir / "fit_record.json").read_text())
