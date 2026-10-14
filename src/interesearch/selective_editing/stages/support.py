"""Shared stage context, run records, item execution, and cached readout sets."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Sequence
import torch
import torch.nn as nn
from interesearch.assets import ASSETS, MODULES, check, load_target
from interesearch.checksums import sha256_file
from interesearch.edits import capture_base
from interesearch.training import disable_tf32
from interesearch.selective_editing.stages.access import (
    FREEZE,
    NOT_OK,
    READOUT_ROLES,
    SELECTION_ROLES,
    STATUS_FILE,
    Access,
    AccessError,
    Windows,
    package_identity,
    utc,
    verify_freeze,
    window_identity,
)
from interesearch.selective_editing.data import write_json
from interesearch.selective_editing.method.dictionary import Dictionary, load_dictionary, randomized_dictionary
from interesearch.selective_editing.method.dose import PAIRS
from interesearch.selective_editing.stages.readout import Sets, base_ce, rebuild
from interesearch.selective_editing.method.references import CI_BATCH
from interesearch.selective_editing.method.selector import (
    BATCH,
    DRAW_ROWS,
    MAX_WALK,
    NULL_SEED,
    default_device,
    draw_rows,
)

TARGETS = tuple(PAIRS)
DRAWS = (0, 1, 2)
REDUCED_DRAWS = (0, 1)
RUN_FILE = "RUN.json"


@dataclass(frozen=True)
class Sizes:
    """Rows each step reads (None: all) and the shortlist. ``REDUCED`` checks the code path only."""

    q_rows: int | None  # retain and q-check rows of q
    draw_rows: int
    n_short: int
    retain_rows: int | None  # English retain rows of the learned baselines
    val_rows: int | None  # target and English validation rows of the rate choice and the calibration
    ci_rows: int | None  # target and English rows of ci_difference
    neuron_rows: int | None  # English rows of the neuron ranking
    n_rates: int | None  # learning rates of the grid
    atlas_top: int

FULL = Sizes(None, DRAW_ROWS, MAX_WALK, None, None, None, None, None, 20)
REDUCED = Sizes(8, 8, 4, 8, 4, 4, 8, 1, 2)


@dataclass
class Context:
    """What a stage runs on. ``ci(target_npy, english_npy, english_cache, out)`` runs ci_difference in the reference
    env (None: not run). ``identity`` names the model and dictionary files for the run record."""

    out: Path
    windows: Path
    model: nn.Module
    vpd: Dictionary
    device: torch.device
    gate: Path | None = None
    bs: int = BATCH
    assets: Path | None = None
    ci: Callable[[Path, Path, Path, Path], None] | None = None
    identity: dict = field(default_factory=dict)
    invocation: Path | None = None
    recipe: dict | None = None


def rel(out: Path, p: Path) -> str:
    return p.relative_to(out).as_posix()


def select_path(out: Path, t: str, d: int, dic: str) -> Path:
    return out / "select" / f"{t}_draw{d}_{dic}.json"


def run_dir(out: Path, t: str, d: int, condition: str, lr: float) -> Path:
    return out / "finetune" / f"{t}_draw{d}_{condition}_lr{lr:g}"


def choice_path(out: Path, t: str, condition: str) -> Path:
    return out / "finetune" / f"{t}_{condition}_lr_choice.json"


def ref_path(out: Path, method: str, t: str, d: int) -> Path:
    return out / "references" / f"{method}_{t}_draw{d}.json"


def cell_path(out: Path, kind: str, t: str, d: int) -> Path:
    """``neurons``, ``calibration``, ``readout``, ``atlas`` or ``confirm/readout``."""
    return out / kind / f"{t}_draw{d}.json"


def null_name(d: int, recipe: dict | None = None) -> str:
    seeds = NULL_SEED if recipe is None else recipe["selection"]["null_seeds"]
    return f"null{seeds[d]}"


def parse_cells(text: str | None, draws: Sequence[int], recipe: dict | None = None) -> list[tuple[str, int]]:
    targets = TARGETS if recipe is None else tuple(recipe["targets"])
    allowed_draws = DRAWS if recipe is None else recipe["draws"]
    if not text:
        return [(t, d) for t in targets for d in draws]
    cells = [(t, int(d)) for t, d in (c.split(":") for c in text.split(","))]
    bad = [c for c in cells if c[0] not in targets or c[1] not in allowed_draws]
    if bad:
        raise ValueError(f"unknown cells {bad}")
    return sorted(set(cells), key=lambda c: (targets.index(c[0]), c[1]))


def bind(out: Path, reduced: bool | None, settings: dict | None = None, code: dict | None = None) -> tuple[dict, bool]:
    """Bind the run's scientific inputs and runtime settings. Source changes are recorded, never a resume gate."""
    p, code = out / RUN_FILE, package_identity() if code is None else code
    if not p.exists():
        if reduced is None:
            raise AccessError(f"no {p}: the run folder starts with the pre-freeze stage")
        write_json(p, {"created_utc": utc(), "code": code, "reduced": reduced, "settings": settings, "windows": {}})
        return code, reduced
    rec = json.loads(p.read_text())
    if reduced is not None and reduced != rec["reduced"]:
        raise AccessError(f"{out} is a {'reduced' if rec['reduced'] else 'full'} run folder")
    if settings is not None and settings != rec["settings"]:
        raise AccessError(f"{out} was run with {rec['settings']}, not {settings}")
    return code, rec["reduced"]


def start_invocation(out: Path, stage: str, argv=None) -> Path:
    """Record current sources before a stage starts; one record per invocation, including resumes."""
    path = out / "runs" / f"{stage}-{time.time_ns()}.json"
    write_json(path, {"stage": stage, "argv": list(argv or sys.argv), "started_utc": utc(),
                      "status": "started", "code": package_identity()})
    return path


def finish_invocation(path: Path, **fields) -> None:
    write_json(path, json.loads(path.read_text()) | fields | {"finished_utc": utc()})


def bind_windows(out: Path, identities: dict[str, str]) -> None:
    """Every role of ``identities`` (role -> ``window_identity``) must have the identity ``RUN.json`` recorded for
    it. A role read for the first time is recorded."""
    p = out / RUN_FILE
    rec = json.loads(p.read_text())
    changed = sorted(k for k, v in identities.items() if rec["windows"].get(k, v) != v)
    if changed:
        raise AccessError(f"{out}: the windows of {changed} differ from those the run folder was started with. "
                          "use a fresh output folder")
    new = {k: v for k, v in identities.items() if k not in rec["windows"]}
    if new:
        write_json(p, rec | {"windows": rec["windows"] | new})


class Runner:
    def __init__(self, ctx: Context, stage: str, reduced: bool | None = None, standin: bool = False) -> None:
        self.ctx, self.out, self.stage = ctx, ctx.out, stage
        self.recipe = ctx.recipe
        self.invocation = ctx.invocation or start_invocation(ctx.out, stage)
        invocation = json.loads(self.invocation.read_text())
        self.code, self.reduced = bind(ctx.out, reduced, {"device": str(ctx.device), "batch_size": ctx.bs},
                                       invocation["code"])
        if self.recipe is not None:
            record = json.loads((self.out / RUN_FILE).read_text())
            if record.get("recipe", self.recipe) != self.recipe:
                raise AccessError("scientific recipe changed; use a fresh run folder")
            if "recipe" not in record:
                write_json(self.out / RUN_FILE, record | {"format": 2, "recipe": self.recipe})
        if stage in ("readout", "confirm"):
            if standin != self.reduced:
                raise AccessError("a reduced run reads stand-in sets only (--standin), and only a reduced run does")
            verify_freeze(ctx.out)  # before any readout-role manifest is read
        roles = SELECTION_ROLES + (READOUT_ROLES if stage == "readout" and not standin else ())
        self.inputs = {role: window_identity(ctx.windows / role, role) for role in roles}
        bind_windows(ctx.out, self.inputs)
        self.sizes = REDUCED if self.reduced else FULL
        if self.recipe is not None:
            if self.reduced:
                self.sizes = Sizes(**{k: self.recipe["reduced"][k] for k in asdict(REDUCED)})
            else:
                self.sizes = Sizes(**(asdict(FULL) | {"draw_rows": self.recipe["selection"]["draw_rows"],
                    "n_short": self.recipe["selection"]["max_walk"], "atlas_top": self.recipe["readout"]["atlas_top"]}))
        self.access = Access(ctx.windows, ctx.out, standin)
        self.base = capture_base(ctx.model, MODULES)
        self.started, self.rows, self.state = utc(), [], {}
        self._windows, self._sets, self._null = {}, {}, None

    def _row(self, name: str, status: str, reason: str, seconds: float,
             output: Path | None = None, reused: bool = False) -> str:
        row = {"utc": utc(), "stage": self.stage, "item": name, "status": status, "reason": reason,
               "seconds": round(seconds, 3), "invocation": rel(self.out, self.invocation)}
        if output is not None:
            row.update(output=rel(self.out, output), reused=reused)
        with open(self.out / STATUS_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)
        self.state[name] = status
        print(f"{row['utc']} {name}: {status}{' (' + reason + ')' if reason else ''}", flush=True)
        return status

    def blocked(self, name: str, deps: Sequence[str] = (), needs: Sequence[Path] = ()) -> bool:
        """True, with a ``blocked`` row, when a dependency failed or was blocked here or an input file is absent."""
        bad = [d for d in deps if self.state[d] in NOT_OK]
        if bad:
            self._row(name, "blocked", f"{bad[0]} {self.state[bad[0]]}", 0.0)
            return True
        absent = [p for p in needs if not p.exists()]
        if absent:
            self._row(name, "blocked", f"missing {rel(self.out, absent[0])}", 0.0)
            return True
        return False

    def declare(self, name: str, reason: str) -> None:
        """An item that does not run by a declared rule of the protocol (not a technical gap)."""
        self._row(name, "declared", reason, 0.0)

    def item(self, name: str, output: Path | None, fn: Callable[[], None], deps: Sequence[str] = (),
             needs: Sequence[Path] = ()) -> str:
        if output is not None and output.exists():
            return self._row(name, "ok", "output exists", 0.0, output, reused=True)
        if self.blocked(name, deps, needs):
            return "blocked"
        t0 = time.time()
        try:
            fn()
        except (FloatingPointError, subprocess.CalledProcessError) as e:
            return self._row(name, "failed", f"{type(e).__name__}: {e}", time.time() - t0)
        return self._row(name, "ok", "", time.time() - t0, output)

    def finish(self, argv: Sequence[str] | None = None) -> int:
        """Write the run record. 3 when an item of this invocation failed or was blocked."""
        not_ok = [r["item"] for r in self.rows if r["status"] in NOT_OK]
        finish_invocation(self.invocation, argv=list(argv or sys.argv),
                          status="items not ok" if not_ok else "ok", not_ok=not_ok, items=self.rows,
                          reduced=self.reduced, windows_opened=self.access.opened, identity=self.ctx.identity,
                          device=str(self.ctx.device), batch_size=self.ctx.bs)
        print(f"{self.stage}: {'items not ok: ' + ', '.join(not_ok) if not_ok else 'ok'}", flush=True)
        return 3 if not_ok else 0

    # ----------------------------------------------------------------------- inputs
    def _w(self, role: str, lang: str) -> Windows:
        if (role, lang) not in self._windows:
            self._windows[role, lang] = self.access.windows(role, [lang])
        return self._windows[role, lang]

    def windows(self, role: str, lang: str) -> torch.Tensor:
        return self._w(role, lang).tokens[lang]

    def draw(self, t: str, d: int) -> torch.Tensor:
        w = self._w("train", t)
        return w.tokens[t][draw_rows(w.groups, d, self.sizes.draw_rows)]

    def dictionary(self, name: str) -> Dictionary:
        """The VPD dictionary, or a randomized one (one held at a time)."""
        if name == "vpd":
            return self.ctx.vpd
        if self._null is None or self._null.name != name:
            self._null = None  # released before the next one is built
            self._null = randomized_dictionary(self.ctx.vpd, int(name.removeprefix("null")))
        return self._null

    def make_of(self, method: str, rec: dict):
        return rebuild(rec, self.out, self.ctx.model, self.dictionary, self.ctx.device)


def readout_record(r: Runner, t: str, d: int, sets: Sets, ce0: dict, edits: dict, t0: float) -> dict:
    return {"target": t, "protected": PAIRS[t] if r.recipe is None else r.recipe["targets"][t], "draw": d, "standin": r.access.standin, "target_key": sets.target_key,
            "protected_key": sets.protected_key, "english_keys": sets.english, "guards": sets.guards,
            "sources": sets.sources, "sets": {k: int(v.shape[0]) for k, v in sets.tokens.items()},
            "groups": sets.groups, "base_ce": {k: v.tolist() for k, v in ce0.items()}, "edits": edits,
            "freeze_sha256": sha256_file(r.out / FREEZE), "seconds": time.time() - t0,
            "invocation": rel(r.out, r.invocation)}


def sets_for(r: Runner, key: tuple, make: Callable[[], Sets]) -> tuple[Sets, dict]:
    """The sets of a target and split with their original-model CE (read and measured once per process)."""
    if key not in r._sets:
        s = make()
        r._sets[key] = (s, base_ce(r.ctx.model, r.base, s, r.ctx.device, r.ctx.bs))
    return r._sets[key]


def ci_subprocess(python: str, pythonpath: Sequence[Path], assets: Path, device, bs: int, layout: Path):
    """ci_difference in the reference env: ``python -m interesearch.selective_editing.method.references ci``."""
    env = os.environ | {"PYTHONPATH": os.pathsep.join(str(Path(p).resolve()) for p in pythonpath)}

    def run(target: Path, english: Path, cache: Path, out: Path) -> None:
        subprocess.run([python, "-m", "interesearch.selective_editing.method.references", "ci", str(assets), "--target",
                        str(target), "--english", str(english), "--english-cache", str(cache), "--out", str(out),
                        "--layout", str(layout), "--device", str(device), "--bs", str(min(bs, CI_BATCH))],
                       check=True, env=env)
    return run


def prepare(a: argparse.Namespace) -> None:
    """Resolve and save the applied scientific recipe before loading model assets."""
    from interesearch.selective_editing.configuration import resolve, saved_recipe
    from interesearch.selective_editing.method.finetune import config
    recipe = resolve(a.recipe)
    # FineTuneConfig owns checkpoint normalization; persist its effective schedules.
    effective = config("matrix", 0, recipe["training"]["learning_rates"]["matrix"][0], recipe=recipe)
    recipe["training"]["checkpoints"] = list(effective.checkpoints)
    reduced_config = config("matrix", 0, recipe["training"]["learning_rates"]["matrix"][0], True, recipe=recipe)
    recipe["reduced"]["training"]["checkpoints"] = list(reduced_config.checkpoints)
    recipe = saved_recipe(a.out, recipe, a.recipe is not None)
    a.recipe = recipe
    a.bs = BATCH if a.bs is None else a.bs
    code = json.loads(a.invocation.read_text())["code"]
    _, reduced = bind(a.out, a.reduced if a.stage == "prefreeze" else None,
                      None if a.stage == "freeze" else {"device": str(torch.device(a.device) if a.device else default_device()),
                                                         "batch_size": a.bs}, code)
    p = a.out / RUN_FILE
    record = json.loads(p.read_text())
    if record.get("recipe", recipe) != recipe:
        raise AccessError(f"{p}: scientific recipe changed; use a fresh run folder")
    write_json(p, record | {"format": 2, "recipe": recipe})
    invocation = json.loads(a.invocation.read_text())
    write_json(a.invocation, invocation | {"recipe": recipe, "reduced": reduced})


def context(a: argparse.Namespace) -> Context:
    """The model, the dictionary and the device of a model stage. The target files must match their pins."""
    bad = [p for x in ASSETS if x.path.startswith("target/model") and (p := check(a.assets_dir, x)) is not None]
    if bad:
        raise ValueError(f"asset files differ from their pins: {bad}")
    disable_tf32()
    device = torch.device(a.device) if a.device else default_device()
    vpd = load_dictionary(a.assets_dir)
    ci_fn = None
    if a.stage == "prefreeze" and not a.no_ci:
        src = Path(__file__).resolve().parents[3]
        ci_fn = ci_subprocess(a.ci_python, [src, a.param_decomp], a.assets_dir, device, a.bs,
                              a.out / "cache" / "ci_layout")
    return Context(a.out, a.windows, load_target(a.assets_dir, device), vpd, device, a.gate, a.bs, a.assets_dir, ci_fn,
                   {"assets": {x.path: x.sha256 for x in ASSETS if x.path.startswith("target/model")},
                    "dictionary": vpd.meta["tensor_digest"]}, invocation=a.invocation, recipe=a.recipe)
