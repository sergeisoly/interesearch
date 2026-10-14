"""Input contracts and run records shared by the donor and target stages."""

from __future__ import annotations
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Mapping
import torch
from ... import assets
from ...assets import MODULES
from ...checksums import sha256_file
from ...edits import capture_base
from ...training import disable_tf32
from ..data import READOUT_ROLES, load_windows, refused_roles, window_role
from ..method.finetune import TRAIN_ROLE, VAL_ROLE, target_model_identity, load_update
from ..method.fitting import RETAIN, finite
from ..method.selection import point

LANGS = ("de", "nl", "en", "fr", "es", "it", "pt")

FIT_LANGS = ("de", "nl", *RETAIN)

TARGETS = ("primary_target", "replication_target", "standin")

FAMILIES = ("strengths", "donor_init", "donor_init_output", "random_init", "fresh", "fresh_seed2",
            "random_init_seed2")

CAPPED = FAMILIES[:5]


RC_OK, RC_REFUSED, RC_FAILED_FAMILY, RC_TECHNICAL, RC_PRE_READOUT_STOP = 0, 3, 8, 9, 10

STATUS = {RC_OK: "ok", RC_FAILED_FAMILY: "failed_family", RC_TECHNICAL: "technical",
          RC_PRE_READOUT_STOP: "pre_readout_stop"}

POLISH_KEYS = ("finite", "converged", "stop", "iters", "evals", "L_start", "L_end", "stationarity_start",
               "stationarity_end", "stationarity_ref", "path_parity_abs", "norm_over_cap", "seconds")

DIRECTION_PAIRS = (("donor_init", "fresh"), ("donor_init_output", "fresh"), ("donor_init", "donor_init_output"),
                   ("random_init", "fresh"), ("donor_init", "random_init"), ("fresh_seed2", "fresh"),
                   ("donor_init", "strengths_best"))

DONOR_REPAIR_SHA256 = "603b62c776856654c925fecd37b7bb43d8dc98994334e5582bb5838c5a9371d7"


FT_BINDING = {"primary_target": {"seed": 14, "rows": [0, 64]}, "replication_target": {"seed": 15, "rows": [64, 128]}}

FT_SOURCE_ROLE = "donor_readout"  # the fine-tune text: rows of the donor stage's readout windows (128 dates)


PREVIEWED_READOUT = ("1998-01-16", "2006-05-17", "1998-04-02", "1996-12-10", "2006-12-18", "2005-09-07", "2002-03-14",
                     "2003-09-03", "2006-12-12", "1999-09-17", "2010-01-18", "2001-07-05", "1998-10-08", "2002-04-25",
                     "2000-02-14", "2001-12-11")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def logger():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)
    return log, t0


def setup(a):
    """The target model on the device with W_0 captured. Threads and precision as recorded."""
    torch.set_num_threads(a.threads)
    precision = disable_tf32()
    model = assets.load_target(a.assets, a.device)
    return model, capture_base(model, MODULES), precision


def finish(out: Path, info: dict, rc: int, log, reasons: list[str], export=None) -> int:
    """Complete run.json, write it and log the end. ``export(record_sha256)`` writes the files of a successful run
    first, naming the hash of the record's exact bytes: a record that says ok exists only after its files."""
    info.update(status=STATUS[rc], rc=rc, reasons=reasons, total_s=time.time() - info.pop("_t0"), utc_end=utc())
    if info["config"]["device"] == "cuda":
        info["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    text = json.dumps(info, indent=1, default=str)
    if export is not None:
        export(hashlib.sha256(text.encode()).hexdigest())
    (out / "run.json").write_text(text)
    log(f"{info['command']} done: {info['status']} (exit {rc}){': ' + '; '.join(reasons) if reasons else ''}")
    return rc


def role_errors(dirs: Mapping[str, tuple[str, str | None]], reduced: bool) -> list[str]:
    """Each window directory must declare its expected role (None: a stand-in of any role). A reduced run opens no
    readout role. Reads only windows.json."""
    err = []
    for arg, (d, want) in dirs.items():
        role = window_role(d)
        if reduced and refused_roles(role, READOUT_ROLES):
            err.append(f"--{arg} {d}: role {role!r} may not be opened by a reduced run")
        elif want is not None and role != want:
            err.append(f"--{arg} {d}: role {role!r}, expected {want!r}")
    return err


def window_dates(d: str | Path) -> list[str]:
    return json.loads((Path(d) / "windows.json").read_text())["dates"]


def finetune_config_errors(ft: dict, recipe: dict, seed: int, lrs: list[float]) -> list[str]:
    """Bind the consumed update to its complete scientific fine-tuning profile."""
    expected = recipe["finetune"]
    want = {**expected["config"], "seed": seed}
    lr = ft["config"].get("lr")
    if lr in lrs:
        want["lr"] = lr
    got = {key: ft["config"].get(key) for key in want}
    errors = [] if got == want else [f"fine-tune config {got} is not the required recipe {want}"]
    snapshot = ft.get("resolved_recipe")
    if snapshot is None:
        # Older public records predate the resolved-recipe field. Their complete FTConfig is checked above;
        # ignored historical metadata remains ignored. New records explicitly identify their format.
        if ft.get("format") is not None:
            errors.append("new fine-tune record has no resolved recipe")
        return errors
    saved = snapshot.get("finetune", {})
    for key in expected:
        if key == "threads":
            continue
        value = want if key == "config" else expected[key]
        if saved.get(key) != value:
            errors.append(f"fine-tune resolved recipe differs at finetune.{key}")
    return errors


def binding_errors(target: str, ft: dict, train_dir: str | None, val_dir: str | None, recipe: dict) -> list[str]:
    """A target's fine-tune against its recipe and seed, and against the window directories it was prepared from
    (--finetune-train, --finetune-val): each of its role and matching its SHA256SUMS, the hashes the fine-tune
    recorded equal to the directory's, and its text drawn from this target's rows of the donor readout windows."""
    err = finetune_config_errors(ft, recipe, FT_BINDING[target]["seed"], [recipe["finetune"]["config"]["lr"]])
    draw = {"role": FT_SOURCE_ROLE, "rows": FT_BINDING[target]["rows"]}
    for part, d, role in (("train", train_dir, TRAIN_ROLE), ("val", val_dir, VAL_ROLE)):
        if d is None:
            err.append(f"--finetune-{part} is required for {target}")
            continue
        try:
            w = load_windows(d, role=role)
        except ValueError as e:
            err.append(str(e))
            continue
        rec = ft.get("data", {}).get(part) or {}
        if rec.get("role") != w.role or rec.get("sha256") != w.sha256:
            err.append(f"fine-tune {part} data: the recorded role and hashes are not those of {d}")
        if part == "train" and ({k: w.source.get(k) for k in draw} != draw or rec.get("source") != w.source):
            err.append(f"{d}: fine-tune text from {w.source} (recorded: {rec.get('source')}) is not the draw of "
                       f"{target} {draw}")
    return err


def finetune_errors(ft_dir: str | Path) -> tuple[tuple | None, list[str]]:
    """Validate the exact update bytes that this stage will use, once."""
    try:
        return load_update(ft_dir), []
    except ValueError as error:
        return None, [str(error)]


def donor_repair_errors(path: str | Path, model_id: dict) -> tuple[dict, list[str], dict | None]:
    """The donor repair's identity: the pinned file, or a file of a donor run (its run.json beside it) that was not
    reduced, ended ok, passed its gate, loaded the same target model, and recorded this file's sha256, its chosen
    candidate and its fine-tune."""
    path = Path(path)
    payload = path.read_bytes()
    s = hashlib.sha256(payload).hexdigest()
    if s == DONOR_REPAIR_SHA256:
        obj = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        return {"sha256": s, "source": "pinned"}, [], obj
    obj = None
    rec_path = path.parent / "run.json"
    if not rec_path.exists():
        return {"sha256": s}, [f"{path}: sha256 {s} is not the pinned donor repair's {DONOR_REPAIR_SHA256}, and no "
                               f"donor run record is beside it"], obj
    rec = json.loads(rec_path.read_text())
    ident = {"sha256": s, "source": "donor run", "run_record_sha256": sha256_file(rec_path)}
    got = {k: rec.get(k) for k in ("command", "status", "reduced", "donor_repair_sha256")}
    want = {"command": "donor", "status": "ok", "reduced": False, "donor_repair_sha256": s}
    err = [] if got == want else [f"{rec_path}: {got} is not a full, successful donor run that wrote {path.name}"]
    if rec.get("gate", {}).get("pass") is not True:
        err.append(f"{rec_path}: the donor repair did not pass its gate")
    if rec.get("inputs", {}).get("target_model") != model_id:
        err.append(f"{rec_path}: the donor run loaded another target model")
    if not err:
        obj = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if obj.get("name") != rec["selection"]["selected"]:
            err.append(f"{path}: candidate {obj.get('name')!r} is not the run's choice "
                       f"{rec['selection']['selected']!r}")
        if obj.get("donor_update_sha256") != rec["inputs"]["finetune"]["update_sha256"]:
            err.append(f"{path}: fine-tune {obj.get('donor_update_sha256')} is not the run's fine-tune")
    return ident, err, obj


def ft_record(ft: dict) -> dict:
    return {"update_sha256": ft["saved"]["update_sha256"], "selection": ft["selection"], "gates": ft["gates"],
            "config": ft["config"],
            "lr": ft["config"]["lr"], "seed": ft["config"]["seed"], "saved_step": ft["saved"]["saved_step"]}


def block_inputs(a) -> tuple[dict, list[str]]:
    model_id, err = target_model_identity(a.assets)
    donor, e, a.donor_payload = donor_repair_errors(a.donor_repair, model_id)
    err += e
    a.finetune_update, e = finetune_errors(a.finetune)
    err += e
    if e:
        return {}, err
    _, ft = a.finetune_update
    if a.target in FT_BINDING:
        err += binding_errors(a.target, ft, a.finetune_train, a.finetune_val, a.resolved_recipe)
    err += role_errors({"fit": (a.fit, "fit"), "selection": (a.selection, "selection"),
                        "readout": (a.readout, None if a.reduced else "readout")}, a.reduced)
    blocks = a.resolved_recipe["block"]["profiles"]
    budgets = a.resolved_recipe["fitting"]["budgets"]
    n_readout_min = a.resolved_recipe["block"]["readout_min"]
    need = max(budgets[B][g][0] for B in blocks[a.block]["budgets"] for g in ("de", "nl", "ret"))
    if not err and len(window_dates(a.fit)) < need:  # fitting.fit_sets: the budgets' prefixes, never fewer windows
        err.append(f"--fit {a.fit}: fewer than the {need} windows of budgets {blocks[a.block]['budgets']}")
    if not a.reduced and not err and len(window_dates(a.readout)) < n_readout_min:
        err.append(f"--readout {a.readout}: fewer than {n_readout_min} dates")
    return {"target_model": model_id, "donor_repair": donor, "finetune": ft_record(ft)}, err


def donor_inputs(a) -> tuple[dict, list[str]]:
    model_id, err = target_model_identity(a.assets)
    a.finetune_update, e = finetune_errors(a.finetune)
    err += e
    if e:
        return {}, err
    _, ft = a.finetune_update
    profile_errors = finetune_config_errors(
        ft, a.resolved_recipe, a.resolved_recipe["donor"]["finetune_seed"],
        a.resolved_recipe["donor"]["finetune_lrs"])
    if profile_errors:
        a.reduced = True
        a.deviations["finetune_input"] = profile_errors
    if ft.get("target_model") != model_id:
        err.append(f"{a.finetune}: the fine-tune was trained on another target model")
    err += role_errors({"fit": (a.fit, "donor_fit"), "selection": (a.selection, VAL_ROLE),
                        "readout": (a.readout, None if a.reduced else "donor_readout")}, a.reduced)
    if not a.reduced and not err:
        n_donor_readout = a.resolved_recipe["donor"]["readout_dates"]
        dates = window_dates(a.readout)
        if len(dates) != n_donor_readout or tuple(dates[:len(PREVIEWED_READOUT)]) != PREVIEWED_READOUT:
            err.append(f"--readout {a.readout}: not the {n_donor_readout} donor readout dates with the previewed "
                       f"dates first")
    return {"target_model": model_id, "finetune": ft_record(ft)}, err


def windows(a, info: dict, arg: str, langs: tuple[str, ...], want: str | None, rows: int | None = None,
            prefixes: dict | None = None):
    """Load a window directory (role contract first, then SHA256SUMS) and record what was read in
    ``info['inputs']['windows'][arg]``: role, checked hashes, dates, source, rows and the fit prefixes (windows,
    tokens per window) of each budget, or None for whole windows."""
    w = load_windows(getattr(a, arg), langs, rows, role=want, refused=READOUT_ROLES if a.reduced else ())
    info["inputs"].setdefault("windows", {})[arg] = {
        "role": w.role, "sha256": w.sha256, "dates": w.dates, "source": w.source, "languages": list(w.tokens),
        "rows": len(w.dates), "prefixes": prefixes}
    return w


def nonfinite_runs(runs: dict) -> dict:
    """The runs that stopped on a non-finite loss or gradient, with where they stopped and what they saved."""
    return {n: {k: r["status"].get(k) for k in ("nonfinite_step", "updates_applied", "batches_presented",
                                                 "checkpoints_saved")}
            for n, r in runs.items() if r["status"].get("finite") is False}


def failed_family_reason(fam: str, runs: dict, every: int) -> str:
    """Why a planned family has no candidate: the status of each of its runs (run names start with 'family:')."""
    parts = []
    for n, r in runs.items():
        if n.split(":")[0] != fam:
            continue
        st = r["status"]
        parts.append(f"{n.split(':', 1)[1]}: " + (f"non-finite at step {st.get('nonfinite_step')} after "
                                                    f"{st.get('updates_applied')} updates" if st.get("finite") is False
                                                    else "no checkpoint saved"))
    return (f"no checkpoint saved by any run (first checkpoint at step {every}): "
            + ("; ".join(parts) if parts else "no run"))


def nonfinite_log(lg: list) -> list[str]:
    return [f"step {r['step']}: {k}" for r in lg for k, v in r.items() if k != "step" and not finite(v)][:5]


def nonfinite_selection(vper: dict, names, sidx, selection: dict) -> dict:
    """Candidates with a non-finite per-window value or point estimate on the selection windows (the selection rule
    would drop them silently)."""
    bad = {}
    for n in names:
        why = [f"{lg}/{k}" for lg, vals in vper[n].items() for k, t in vals.items()
               if not bool(torch.isfinite(t[sidx]).all())]
        p = point(vper, n, sidx, selection)
        why += [k for k in ("r_nl", "r_kl", "rho_de", "phi_de") if not finite(p.get(k))]
        why += [f"retain[{g}]" for g, v in (p.get("retain") or {}).items() if not finite(v)]
        if why:
            bad[n] = why[:5]
    return bad
