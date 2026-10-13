"""Exported readouts, calibration records and protocol of the selective-editing study.
Data roles: papers/selective-editing/index.qmd#sec-access.

A cell is one target and draw of one split (dev or confirm). Its cell.json holds the scalars of every edit
(status, scale, k, the English guards, the target-test CE rise and the English KL, on confirm also the protected CE
rise) and the date of every protected document. ``protected.npz`` holds the protected per-document CE of the base model
and of each matched edit (float64, unrounded, document order). posthoc.npz holds, for the same documents (the target
and protected sets share their dates), the target per-document CE of the base model and of each matched edit and the
protected per-document KL of each matched edit. Only the post-hoc analyses read it. The calibration record of a dev
cell holds, for every scanned method and effect point, the evaluated (dose, effect) points, the recorded decision
and the edit norm of a matched decision, and for each method the atom or learned update its edits scale.

``load_study`` reads a complete folder (every expected cell of both splits). ``load_dev`` reads the dev split of a
folder written before the confirm readout, with the cells it holds. Records of a reduced run (code-path checks only,
the protocol says ``reduced``) load only with ``allow_reduced``. The writers give equal bytes for equal records.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..checksums import sha256_file, verify_checksums

SPLITS = ("dev", "confirm")
CELL_FILES = ("cell.json", "protected.npz", "posthoc.npz")


@dataclass(frozen=True)
class Cell:
    info: dict
    base_ce: torch.Tensor
    ce_doc: dict[str, torch.Tensor]
    posthoc: dict[str, np.ndarray]  # target_base_ce (n,), target_ce_doc and protected_kl_doc (len(measured), n)


@dataclass(frozen=True)
class Study:
    protocol: dict
    cells: dict[str, dict[tuple[str, int], Cell]]  # split -> (target, draw) -> cell
    calibration: dict[tuple[str, int], dict]


def expected_cells(protocol: dict) -> list[tuple[str, int]]:
    return [(t, int(d)) for t, d in (c.split(":") for c in protocol["expected_cells"])]


def _cell_dir(t: str, d: int) -> str:
    return f"{t}_draw{d}"


def _arrays(path: Path, n: int, m: int, base: str, docs: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in (base, *docs)}
    shapes = {base: (n,)} | {k: (m, n) for k in docs}
    if any(v.dtype != np.float64 or v.shape != shapes[k] for k, v in out.items()):
        raise ValueError(f"{path}: {[(k, v.dtype, v.shape) for k, v in out.items()]}. Expected float64 {shapes}")
    return out


def _load_cell(directory: Path) -> Cell:
    info = json.loads((directory / "cell.json").read_text())
    n, names = len(info["groups"]), info["measured"]
    z = _arrays(directory / "protected.npz", n, len(names), "base_ce", ("ce_doc",))
    base, ce = z["base_ce"], z["ce_doc"]
    posthoc = _arrays(directory / "posthoc.npz", n, len(names), "target_base_ce",
                      ("target_ce_doc", "protected_kl_doc"))
    return Cell(info, torch.from_numpy(base.copy()), {m: torch.from_numpy(ce[i].copy()) for i, m in enumerate(names)},
                posthoc)


def _protocol(root: Path, allow_reduced: bool) -> dict:
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol.get("reduced") and not allow_reduced:
        raise ValueError(f"{root}: records of a reduced run (code-path checks only). They load with allow_reduced")
    return protocol


def load_study(root: str | Path, allow_reduced: bool = False) -> Study:
    root = Path(root)
    protocol = _protocol(root, allow_reduced)
    cells = expected_cells(protocol)
    files = ["protocol.json"] + [f"calibration/{_cell_dir(t, d)}.json" for t, d in cells]
    files += [f"{s}/{_cell_dir(t, d)}/{f}" for s in SPLITS for t, d in cells for f in CELL_FILES]
    verify_checksums(root, tuple(files))
    return Study(
        protocol=protocol,
        cells={s: {(t, d): _load_cell(root / s / _cell_dir(t, d)) for t, d in cells} for s in SPLITS},
        calibration={(t, d): json.loads((root / f"calibration/{_cell_dir(t, d)}.json").read_text()) for t, d in cells},
    )


def load_dev(root: str | Path, allow_reduced: bool = False) -> Study:
    """The dev split of a folder written before the confirm readout: the expected cells that have a dev record (the
    analysis reads the others as "missing readout") and the calibration records present. SHA256SUMS must list
    exactly these files."""
    root = Path(root)
    protocol = _protocol(root, allow_reduced)
    cells = [c for c in expected_cells(protocol) if (root / "dev" / _cell_dir(*c)).is_dir()]
    calibrated = [c for c in expected_cells(protocol) if (root / f"calibration/{_cell_dir(*c)}.json").is_file()]
    files = ["protocol.json"] + [f"calibration/{_cell_dir(*c)}.json" for c in calibrated]
    files += [f"dev/{_cell_dir(*c)}/{f}" for c in cells for f in CELL_FILES]
    verify_checksums(root, tuple(files))
    return Study(
        protocol=protocol,
        cells={"dev": {c: _load_cell(root / "dev" / _cell_dir(*c)) for c in cells}},
        calibration={c: json.loads((root / f"calibration/{_cell_dir(*c)}.json").read_text()) for c in calibrated},
    )


# --------------------------------------------------------------------------- writers
def write_json(path: str | Path, obj) -> None:
    """``obj`` as JSON at ``path``, written to a temporary sibling and renamed into place: the file is complete or
    absent, so a stage that resumes on existing records never takes an interrupted write for a finished one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def write_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """NPZ with a fixed member order and timestamp, so equal arrays give equal bytes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for key in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())


def write_cell(directory: str | Path, info: dict, protected: dict[str, np.ndarray],
               posthoc: dict[str, np.ndarray]) -> None:
    directory = Path(directory)
    write_json(directory / "cell.json", info)
    write_npz(directory / "protected.npz", protected)
    write_npz(directory / "posthoc.npz", posthoc)


def write_sums(root: str | Path) -> list[str]:
    """SHA256SUMS over every file under ``root``. The listed paths."""
    root = Path(root)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(root / f)}  {f}\n" for f in files))
    return files
